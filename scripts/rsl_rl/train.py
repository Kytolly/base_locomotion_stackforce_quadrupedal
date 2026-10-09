#!/usr/bin/env python3
"""Train the StackForce locomotion policy from an OmegaConf YAML contract."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

# Keep source checkouts usable with IsaacLab's launcher even when the extension
# packages have not been installed into the selected Python environment yet.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/home/kytolly/Library/IsaacLab"))
_VENV_ROOT = Path(os.environ.get("VIRTUAL_ENV", sys.prefix))
_CUDA_LIB_DIRS = [
    path
    for path in (_VENV_ROOT / "lib/python3.12/site-packages/nvidia").glob("*/lib")
    if path.is_dir()
]
for _library_name in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
    for _library_dir in _CUDA_LIB_DIRS:
        _library_path = _library_dir / _library_name
        if _library_path.is_file():
            try:
                ctypes.CDLL(str(_library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
if _CUDA_LIB_DIRS:
    _existing_library_path = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    _library_paths = [str(path) for path in _CUDA_LIB_DIRS]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        dict.fromkeys(_library_paths + [path for path in _existing_library_path if path])
    )
for _source in (_PROJECT_ROOT / "source/base_locomotion_stackforce_quadrupedal", *_ISAACLAB_ROOT.glob("source/*")):
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the Isaac Lab runtime
    if exc.name == "omegaconf":
        raise RuntimeError(
            "OmegaConf is missing from the active Isaac Lab Python environment. "
            "Install this project with `pip install -e .` (or `pip install omegaconf>=2.3.0`) "
            "and rerun through IsaacLab/isaaclab.sh -p."
        ) from exc
    raise

from isaaclab.app import AppLauncher


PROJECT_ROOT = _PROJECT_ROOT
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml"
ISAACLAB_PYTHON = Path("/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python")


def _load_training_config_module():
    config_module_path = PROJECT_ROOT / (
        "source/base_locomotion_stackforce_quadrupedal/"
        "base_locomotion_stackforce_quadrupedal/training/config.py"
    )
    config_spec = importlib.util.spec_from_file_location("_training_config", config_module_path)
    if config_spec is None or config_spec.loader is None:
        raise RuntimeError(f"Unable to load training config helpers from {config_module_path}.")
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    return config_module


def _load_yaml_config(path: Path, overrides: list[str]):
    """Load the launcher contract before importing task modules into Kit."""
    return _load_training_config_module().load_config(path, overrides)


def _launcher_kwargs(config) -> dict[str, object]:
    viz = str(config.launcher.viz)
    visualizers = None if viz == "none" else viz.split(",")
    return {
        "device": str(config.launcher.device),
        "visualizer": visualizers,
        "visualizer_explicit": True,
        "visualizer_disable_all": viz == "none",
        "headless": viz == "none",
        "enable_cameras": bool(config.launcher.get("enable_cameras", False)),
        "max_visible_envs": int(config.launcher.get("max_visible_envs", 16)),
    }


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


def _require_isaac_sim_runtime() -> None:
    """Fail before AppLauncher when the active Python cannot bootstrap Isaac Sim."""
    exp_path = os.environ.get("EXP_PATH")
    if exp_path and Path(exp_path).is_dir():
        return
    raise RuntimeError(
        "Isaac Sim runtime is not initialized for "
        f"{sys.executable}. Run this script with {ISAACLAB_PYTHON}, or activate "
        "the env_isaaclab Conda environment before invoking isaaclab.sh."
    )


def main() -> None:
    args = _parse_args()
    config = _load_yaml_config(args.config, args.overrides)
    if args.validate_config:
        # The YAML contract can be checked without importing the task package;
        # importing it would initialize every Isaac Lab task and require Kit.
        validate_training_config = _load_training_config_module().validate_training_config

        validate_training_config(config)
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    resolved = OmegaConf.to_container(config, resolve=True)
    _require_isaac_sim_runtime()
    app_launcher = AppLauncher(_launcher_kwargs(config))
    simulation_app = app_launcher.app

    # Importing the project package before AppLauncher starts Kit can leave
    # SensorBaseCfg classes split across pre-Kit and Kit module instances.
    from base_locomotion_stackforce_quadrupedal.training import (
        apply_config,
        validate_training_config,
    )

    validate_training_config(config)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import (
        load_cfg_from_registry,
        parse_env_cfg,
    )

    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric import (
        TrainingMetricsWrapper,
    )
    from base_locomotion_stackforce_quadrupedal.training.checkpoint import (
        build_training_contract,
        checkpoint_runner_config,
        require_compatible_training_contract,
        save_training_contract,
        resume_training_runner,
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
    if config.env.get("terrain_profile"):
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import (
            get_complex_terrain_cfg,
        )

        env_cfg.terrain_profile = str(config.env.terrain_profile)
        env_cfg.scene.terrain.terrain_generator = get_complex_terrain_cfg(
            seed=int(config.env.seed), profile=env_cfg.terrain_profile
        )
    if config.get("curriculum"):
        apply_config(env_cfg.curriculum, config.curriculum, "curriculum")
        if terrain_generator := getattr(env_cfg.scene.terrain, "terrain_generator", None):
            terrain_generator.curriculum = env_cfg.curriculum.terrain_levels is not None
    physics_values = config.env.get("physics", {})
    if physics_values:
        physics_cfg = getattr(env_cfg.sim, "physics", None)
        if physics_cfg is None:
            raise ValueError("env.physics overrides require a PhysX simulation backend.")
        for field_name, field_value in physics_values.items():
            if not hasattr(physics_cfg, field_name):
                raise KeyError(f"Unknown environment physics field: env.physics.{field_name}")
            setattr(physics_cfg, field_name, int(field_value))
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
    training_contract = build_training_contract(task, env_cfg, agent_cfg)
    save_training_contract(log_dir, training_contract)
    _configure_wandb(config, resolved)

    render_mode = "rgb_array" if bool(config.video.enabled) else None
    env = None
    exit_code = 0
    try:
        print(f"[TRAIN] creating_env task={task}", flush=True)
        env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
        print("[TRAIN] environment_created", flush=True)
        if bool(config.video.enabled):
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos/train"),
                step_trigger=lambda step: step % int(config.video.interval) == 0,
                video_length=int(config.video.length),
                disable_logger=True,
            )
        env = TrainingMetricsWrapper(env, config.get("logging", {}))
        vec_env = RslRlVecEnvWrapper(env, clip_actions=float(agent_cfg.clip_actions))
        print(f"[TRAIN] observation_space={vec_env.observation_space} action_space={vec_env.action_space}", flush=True)
        runner = OnPolicyRunner(
            vec_env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume:
            checkpoint = str(log_dir.parent / agent_cfg.load_run / agent_cfg.load_checkpoint)
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            print(f"[TRAIN] resuming checkpoint={checkpoint}")
            require_compatible_training_contract(
                Path(checkpoint),
                training_contract,
                allow_terrain_mix_transition=bool(
                    config.runtime.get("allow_terrain_stage_resume", False)
                ),
            )
            saved_cfg = checkpoint_runner_config(Path(checkpoint), agent_cfg.to_dict())
            if saved_cfg["actor"]["distribution_cfg"] != agent_cfg.to_dict()["actor"]["distribution_cfg"]:
                raise ValueError("Resume cannot change the action distribution. Start a new run with agent.resume=false.")
            resume_training_runner(runner, checkpoint)
            print(f"[TRAIN] next_iteration={runner.current_learning_iteration} learning_rate={runner.alg.learning_rate}")

        print(f"[TRAIN] config={args.config.resolve()}")
        print(f"[TRAIN] log_dir={log_dir}")
        print(f"[TRAIN] viz={config.launcher.viz} wandb={agent_cfg.logger}")
        started = time.time()
        runner.learn(
            num_learning_iterations=int(agent_cfg.max_iterations),
            init_at_random_ep_len=bool(config.runtime.init_at_random_episode_length),
        )
        print(f"[TRAIN] completed_seconds={time.time() - started:.2f}")
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
