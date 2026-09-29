#!/usr/bin/env python3
"""Run a frozen policy on a YAML-configured plateau or washboard benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from base_locomotion_stackforce_quadrupedal.training import (
    launcher_kwargs,
    load_config,
    validate_benchmark_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/benchmark/plateau.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = _parse_args()
    config = load_config(args.config, args.overrides)
    validate_benchmark_config(config)
    if args.validate_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    app_launcher = AppLauncher(launcher_kwargs(config))
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401
    from base_locomotion_stackforce_quadrupedal.benchmark import track_length
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import (
        LEG_JOINTS,
        WHEEL_JOINTS,
    )
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (
        POLICY_ACTION_DIM,
    )

    task = str(config.task)
    env_cfg = parse_env_cfg(
        task,
        device=str(config.launcher.device),
        num_envs=1,
        use_fabric=bool(config.launcher.get("use_fabric", True)),
    )
    env_cfg.seed = int(config.benchmark.seed)
    env_cfg.events.spawn_track.params = {
        "kind": str(config.benchmark.kind),
        "seed": int(config.benchmark.seed),
        "randomized": bool(config.benchmark.randomized),
    }

    base_env = gym.make(task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(base_env, clip_actions=1.0)
    checkpoint = config.policy.get("checkpoint")
    checkpoint_path = (
        Path(str(checkpoint)).expanduser().resolve() if checkpoint else None
    )
    policy = None
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = str(config.launcher.device)
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(
            vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=agent_cfg.device)

    try:
        vec_env.reset()
        env = base_env.unwrapped
        robot = env.scene["robot"]
        leg_ids, _ = robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        wheel_ids, _ = robot.find_joints(list(WHEEL_JOINTS), preserve_order=True)
        rewards: list[float] = []
        progress = 0.0
        leg_energy = 0.0
        wheel_energy = 0.0
        tracking_error_sq = 0.0
        action_saturation = 0.0
        all_finite = True
        terminated_seen = False
        truncated_seen = False

        for _ in range(int(config.benchmark.max_steps)):
            observations = vec_env.get_observations()
            with torch.inference_mode():
                actions = (
                    torch.zeros((1, POLICY_ACTION_DIM), device=env.device)
                    if policy is None
                    else policy(observations)
                )
            y_before = robot.data.root_pos_w[:, 1].clone()
            _, reward, dones, _ = vec_env.step(actions)
            y_after = robot.data.root_pos_w[:, 1]
            terminated = bool(env.reset_terminated[0].item())
            truncated = bool(env.reset_time_outs[0].item())
            if not bool(dones[0].item()):
                progress += float((y_after - y_before)[0].item())

            joint_velocity = robot.data.joint_vel
            joint_torque = robot.data.applied_torque
            leg_energy += float(
                torch.sum(
                    torch.abs(joint_velocity[:, leg_ids] * joint_torque[:, leg_ids])
                ).item()
                * env.step_dt
            )
            wheel_energy += float(
                torch.sum(
                    torch.abs(joint_velocity[:, wheel_ids] * joint_torque[:, wheel_ids])
                ).item()
                * env.step_dt
            )
            command = env.command_manager.get_command("locomotion")[:, 0]
            tracking_error_sq += float(
                torch.square(command - robot.data.root_lin_vel_b[:, 1]).mean().item()
            )
            action_saturation += float(
                (torch.abs(actions) >= 0.95).float().mean().item()
            )
            all_finite &= all(
                bool(torch.isfinite(value).all())
                for value in (
                    actions,
                    reward,
                    robot.data.root_pos_w,
                    robot.data.root_lin_vel_b,
                )
            )
            rewards.append(float(reward[0].item()))
            if bool(dones[0].item()):
                terminated_seen = terminated
                truncated_seen = truncated
                break

        parameters = env.benchmark_track_parameters
        required_progress = float(config.benchmark.success_fraction) * track_length(
            parameters
        )
        steps = len(rewards)
        report = {
            "schema_version": 1,
            "task": task,
            "track": str(config.benchmark.kind),
            "seed": int(config.benchmark.seed),
            "randomized": bool(config.benchmark.randomized),
            "policy": "zero_policy_baseline" if policy is None else "checkpoint",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
            "git_commit": _git_commit(),
            "ppo_updates": 0,
            "steps": steps,
            "episode_complete": terminated_seen or truncated_seen,
            "forward_progress_m": progress,
            "required_progress_m": required_progress,
            "success": progress >= required_progress and not terminated_seen,
            "fall_termination": terminated_seen,
            "timeout_termination": truncated_seen,
            "mean_step_reward": sum(rewards) / max(steps, 1),
            "command_tracking_rmse_mps": (tracking_error_sq / max(steps, 1)) ** 0.5,
            "action_saturation_rate": action_saturation / max(steps, 1),
            "leg_mechanical_energy_j": leg_energy,
            "wheel_mechanical_energy_j": wheel_energy,
            "runtime_all_finite": all_finite,
            "track_length_m": track_length(parameters),
            "track_parameters": parameters,
        }
        output = (PROJECT_ROOT / str(config.output)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"[BENCHMARK] report={output}")
    finally:
        vec_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
