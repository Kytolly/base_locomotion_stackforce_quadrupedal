#!/usr/bin/env python3
"""Train the StackForce locomotion policy from an OmegaConf YAML contract."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import time

from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from base_locomotion_stackforce_quadrupedal.training import (
    apply_config,
    launcher_kwargs,
    load_config,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Resolve and validate YAML without starting Isaac Sim.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, for example agent.max_iterations=10 launcher.viz=none.",
    )
    return parser.parse_args()


def _configure_wandb(config, resolved: dict) -> None:
    if not bool(config.wandb.enabled):
        return
    os.environ["WANDB_MODE"] = str(config.wandb.mode)
    if config.wandb.get("entity"):
        os.environ["WANDB_ENTITY"] = str(config.wandb.entity)

    import wandb

    original_init = wandb.init
    original_settings = wandb.Settings

    def compatible_settings(*args, **kwargs):
        kwargs.pop("start_method", None)
        return original_settings(*args, **kwargs)

    def configured_init(*args, **kwargs):
        kwargs.setdefault("entity", config.wandb.get("entity"))
        kwargs.setdefault("group", config.wandb.get("group"))
        kwargs.setdefault("tags", list(config.wandb.get("tags", [])))
        kwargs.setdefault("name", str(config.wandb.run_name))
        run_config = dict(kwargs.get("config") or {})
        run_config["resolved_training_config"] = resolved
        kwargs["config"] = run_config
        return original_init(*args, **kwargs)

    wandb.Settings = compatible_settings
    wandb.init = configured_init


def _make_log_dir(config) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = str(config.agent.get("run_name", "")).strip()
    directory = timestamp if not run_name else f"{timestamp}_{run_name}"
    return (
        PROJECT_ROOT
        / str(config.runtime.log_root)
        / str(config.agent.experiment_name)
        / directory
    ).resolve()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config, args.overrides)
    validate_training_config(config)
    if args.validate_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    resolved = OmegaConf.to_container(config, resolve=True)
    app_launcher = AppLauncher(launcher_kwargs(config))
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import (
        get_checkpoint_path,
        load_cfg_from_registry,
        parse_env_cfg,
    )

    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric import (
        TrainingMetricsWrapper,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    task = str(config.task)
    env_cfg = parse_env_cfg(
        task,
        device=str(config.launcher.device),
        num_envs=int(config.env.num_envs),
        use_fabric=bool(config.launcher.get("use_fabric", True)),
    )
    env_cfg.seed = int(config.env.seed)
    env_cfg.sim.dt = float(config.env.sim_dt)
    env_cfg.decimation = int(config.env.decimation)
    env_cfg.sim.render_interval = int(config.env.render_interval)
    env_cfg.episode_length_s = float(config.env.episode_length_s)
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None:
        terrain_generator.seed = int(config.env.seed)

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    apply_config(agent_cfg, config.agent, "agent")
    agent_cfg.seed = int(config.env.seed)
    agent_cfg.device = str(config.launcher.device)
    agent_cfg.logger = "wandb" if bool(config.wandb.enabled) else "tensorboard"
    agent_cfg.wandb_project = str(config.wandb.project)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    log_dir = _make_log_dir(config)
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)
    OmegaConf.save(config, log_dir / "resolved_training_config.yaml", resolve=True)
    shutil.copy2(args.config.resolve(), log_dir / "source_training_config.yaml")
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent.yaml"), agent_cfg)
    _configure_wandb(config, resolved)

    render_mode = "rgb_array" if bool(config.video.enabled) else None
    env = None
    try:
        env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
        if bool(config.video.enabled):
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos/train"),
                step_trigger=lambda step: step % int(config.video.interval) == 0,
                video_length=int(config.video.length),
                disable_logger=True,
            )
        env = TrainingMetricsWrapper(env)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=float(agent_cfg.clip_actions))
        runner = OnPolicyRunner(
            vec_env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume:
            checkpoint = get_checkpoint_path(
                str(log_dir.parent), agent_cfg.load_run, agent_cfg.load_checkpoint
            )
            print(f"[TRAIN] resuming checkpoint={checkpoint}")
            runner.load(checkpoint)

        print(f"[TRAIN] config={args.config.resolve()}")
        print(f"[TRAIN] log_dir={log_dir}")
        print(f"[TRAIN] viz={config.launcher.viz} wandb={agent_cfg.logger}")
        started = time.time()
        runner.learn(
            num_learning_iterations=int(agent_cfg.max_iterations),
            init_at_random_ep_len=bool(config.runtime.init_at_random_episode_length),
        )
        print(f"[TRAIN] completed_seconds={time.time() - started:.2f}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
