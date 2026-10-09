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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/benchmark/plateau.yaml"


def _load_yaml_config(path: Path, overrides: list[str]):
    """Load the launcher contract before importing task modules into Kit."""
    config = OmegaConf.load(path.expanduser().resolve())
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


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
    config = _load_yaml_config(args.config, args.overrides)
    if args.validate_config:
        from base_locomotion_stackforce_quadrupedal.training import validate_benchmark_config

        validate_benchmark_config(config)
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    app_launcher = AppLauncher(_launcher_kwargs(config))
    simulation_app = app_launcher.app

    # Importing the project package before AppLauncher starts Kit can leave
    # SensorBaseCfg classes split across pre-Kit and Kit module instances.
    from base_locomotion_stackforce_quadrupedal.training import validate_benchmark_config

    validate_benchmark_config(config)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401
    from base_locomotion_stackforce_quadrupedal.benchmark import TrackTraversal, track_length
    from base_locomotion_stackforce_quadrupedal.benchmark.acceptance import passes_thresholds
    from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric import LocomotionEpisodeMetrics
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
    env_cfg.episode_length_s = float(config.benchmark.get("episode_length_s", 35.0))
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
            vec_env, checkpoint_runner_config(checkpoint_path, agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=agent_cfg.device)

    try:
        vec_env.reset()
        env = base_env.unwrapped
        robot = env.scene["robot"]
        metrics = LocomotionEpisodeMetrics(env)
        snapshot = {}

        def capture_step():
            metrics.observe_state(env.action_manager.action)
            metrics.observe_reward(env.reward_buf)
            for name in ("root_pos_w", "root_lin_vel_b", "joint_vel", "applied_torque"):
                snapshot[name] = getattr(robot.data, name).clone()
            snapshot["command"] = env.command_manager.get_command("locomotion").clone()

        env.post_physics_callbacks = [capture_step]
        expected_command = torch.tensor(
            [
                float(env_cfg.events.fixed_command.params["forward_velocity"]),
                0.0,
                0.0,
                float(env_cfg.events.fixed_command.params["body_height"]),
            ],
            device=env.device,
        )
        command_term = env.command_manager.get_term("locomotion")
        command_term.set_command(expected_command.unsqueeze(0))
        parameters = env.benchmark_track_parameters
        traversal = TrackTraversal(parameters)
        traversal.update(
            float(robot.data.root_pos_w[0, 0].item()),
            float(robot.data.root_pos_w[0, 1].item()),
        )
        leg_ids, _ = robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        wheel_ids, _ = robot.find_joints(list(WHEEL_JOINTS), preserve_order=True)
        rewards: list[float] = []
        initial_command = env.command_manager.get_command("locomotion")[0].clone()
        command_min = initial_command.clone()
        command_max = initial_command.clone()
        fixed_command_pass = bool(
            torch.allclose(initial_command, expected_command, atol=1.0e-6, rtol=0.0)
        )
        progress = 0.0
        leg_energy = 0.0
        wheel_energy = 0.0
        tracking_error_sq = 0.0
        action_saturation = 0.0
        all_finite = True
        terminated_seen = False
        truncated_seen = False

        for _ in range(int(config.benchmark.max_steps)):
            # Keep the command fixed even if a future command implementation
            # resamples during reset or at a manager update boundary.
            command_term.set_command(expected_command.unsqueeze(0))
            observations = vec_env.get_observations()
            with torch.inference_mode():
                actions = (
                    torch.zeros((1, POLICY_ACTION_DIM), device=env.device)
                    if policy is None
                    else policy(observations)
                )
            y_before = robot.data.root_pos_w[:, 1].clone()
            _, reward, dones, _ = vec_env.step(actions)
            y_after = snapshot["root_pos_w"][:, 1]
            terminated = bool(env.reset_terminated[0].item())
            truncated = bool(env.reset_time_outs[0].item())
            progress += float((y_after - y_before)[0].item())
            traversal.update(
                float(snapshot["root_pos_w"][0, 0].item()),
                float(snapshot["root_pos_w"][0, 1].item()),
            )

            joint_velocity = snapshot["joint_vel"]
            joint_torque = snapshot["applied_torque"]
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
            command = snapshot["command"][:, 0]
            full_command = snapshot["command"][0]
            command_min = torch.minimum(command_min, full_command)
            command_max = torch.maximum(command_max, full_command)
            fixed_command_pass &= bool(
                torch.allclose(full_command, expected_command, atol=1.0e-6, rtol=0.0)
            )
            tracking_error_sq += float(
                torch.square(command - snapshot["root_lin_vel_b"][:, 1]).mean().item()
            )
            action_saturation += float(
                (torch.abs(actions) >= 0.95).float().mean().item()
            )
            all_finite &= all(
                bool(torch.isfinite(value).all())
                for value in (
                    actions,
                    reward,
                    snapshot["root_pos_w"],
                    snapshot["root_lin_vel_b"],
                )
            )
            rewards.append(float(reward[0].item()))
            if bool(dones[0].item()):
                terminated_seen = terminated
                truncated_seen = truncated
                break

        required_progress = float(config.benchmark.success_fraction) * track_length(
            parameters
        )
        steps = len(rewards)
        manager = env.termination_manager
        termination_reasons = []
        if terminated_seen or truncated_seen:
            termination_reasons = [
                name for index, name in enumerate(manager._term_names)
                if bool(manager._last_episode_dones[0, index])
            ]
        episode_metrics = metrics.episode_metrics(
            torch.tensor([0], device=env.device),
            torch.tensor([terminated_seen], device=env.device),
            torch.tensor([truncated_seen], device=env.device),
            torch.tensor(["base_height" in termination_reasons], device=env.device),
        )
        measured = {key: float(value[0].item()) for key, value in episode_metrics.items()}
        measured["safety/base_tilt_max_p95_rad"] = measured["safety/base_tilt_max_rad"]
        performance_pass = passes_thresholds(measured)
        report = {
            "schema_version": 2,
            "metrics": measured,
            "performance_pass": performance_pass,
            "traversal_complete": traversal.complete,
            "termination_reasons": termination_reasons,
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
            "episode_complete": truncated_seen and not terminated_seen,
            "forward_progress_m": progress,
            "required_progress_m": required_progress,
            "success": (
                traversal.complete
                and truncated_seen
                and not terminated_seen
                and all_finite
                and fixed_command_pass
                and performance_pass
                and policy is not None
            ),
            "track_checkpoints_passed": traversal.gate_index,
            "track_checkpoint_count": len(traversal.gates),
            "track_corridor_violation": traversal.corridor_violation,
            "track_trajectory_discontinuity": traversal.trajectory_discontinuity,
            "fall_termination": terminated_seen,
            "timeout_termination": truncated_seen,
            "mean_step_reward": sum(rewards) / max(steps, 1),
            "command_tracking_rmse_mps": (tracking_error_sq / max(steps, 1)) ** 0.5,
            "action_saturation_rate": action_saturation / max(steps, 1),
            "leg_mechanical_energy_j": leg_energy,
            "wheel_mechanical_energy_j": wheel_energy,
            "runtime_all_finite": all_finite,
            "fixed_command_pass": fixed_command_pass,
            "locomotion_command": {
                "order": [
                    "forward_velocity_mps",
                    "lateral_velocity_mps",
                    "yaw_rate_rps",
                    "body_height_m",
                ],
                "expected": expected_command.tolist(),
                "observed_initial": initial_command.tolist(),
                "observed_min": command_min.tolist(),
                "observed_max": command_max.tolist(),
            },
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
