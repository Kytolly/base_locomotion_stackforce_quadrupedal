#!/usr/bin/env python3
"""Deterministic vs exploration-noise rollouts of one checkpoint (no training).

Separates what the policy commands from what PPO's Gaussian exploration adds:
both modes run the same env (training YAML + overrides) for one full episode
length from a common reset, and report the training-metric actuation terms
(averaged over every episode, the unfinished ones closed at the horizon), the
mean completed episode length, and the distribution of per-step leg
target-angle changes against what a DS041MG servo can travel in one control
step (8.73 rad/s x 0.02 s = 0.175 rad).

    python scripts/eval_action_noise.py --headless --checkpoint <model.pt> --num-envs 1024 \\
        env.terrain_profile=probe_low_steps ...
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import traceback

_CUDA_LIB = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib"
if _CUDA_LIB.is_dir():
    for _name in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
        _path = _CUDA_LIB / _name
        if _path.is_file():
            ctypes.CDLL(str(_path), mode=ctypes.RTLD_GLOBAL)
    os.environ["LD_LIBRARY_PATH"] = f"{_CUDA_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", type=Path, default=ROOT / "configs/train/main/b2_bisec_full.yaml")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=None, help="default: one episode length")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides for the training YAML.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: E402,F401
from base_locomotion_stackforce_quadrupedal.training import apply_config, load_config  # noqa: E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (  # noqa: E402,E501
    apply_e0_env_options,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import (  # noqa: E402,E501
    get_complex_terrain_cfg,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric import (  # noqa: E402,E501
    TrainingMetricsWrapper,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric.locomotion import (  # noqa: E402,E501
    aggregate_metric_batch,
)

SERVO_STEP_RAD = 8.73 * 0.02
METRICS = (
    "train/actuation/leg_saturation_rate",
    "train/actuation/leg_torque_saturation_rate",
    "train/actuation/leg_torque_over_rated_rate",
    "train/actuation/joint_velocity_limit_rate",
    "train/actuation/leg_tracking_error_rad",
    "train/actuation/action_rate_rms",
    "train/actuation/leg_action_rms",
    "train/runtime/closure_residual_max_mm",
    "train/safety/unsafe_termination",
    "train/safety/base_collision_rate",
)


def build_env(config):
    """Same env construction as ``scripts/profile_env_step.py`` (and training)."""
    env_cfg = parse_env_cfg(str(config.task), device=args.device, num_envs=args.num_envs,
                            use_fabric=bool(config.launcher.get("use_fabric", True)))
    env_cfg.seed = int(config.env.seed)
    env_cfg.sim.dt = float(config.env.sim_dt)
    env_cfg.decimation = int(config.env.decimation)
    env_cfg.sim.render_interval = int(config.env.render_interval)
    env_cfg.episode_length_s = float(config.env.episode_length_s)
    if config.env.get("terrain_profile"):
        env_cfg.terrain_profile = str(config.env.terrain_profile)
        env_cfg.scene.terrain.terrain_generator = get_complex_terrain_cfg(
            seed=int(config.env.seed), profile=env_cfg.terrain_profile
        )
    apply_e0_env_options(env_cfg, OmegaConf.to_container(config.env, resolve=True))
    if config.get("curriculum"):
        apply_config(env_cfg.curriculum, config.curriculum, "curriculum")
    for name, value in (config.env.get("physics") or {}).items():
        setattr(env_cfg.sim.physics, name, int(value))
    return TrainingMetricsWrapper(gym.make(str(config.task), cfg=env_cfg), config.get("logging", {}))


def rollout(env, vec, policy, stochastic: bool, steps: int) -> dict:
    base = env.unwrapped
    term = base.action_manager.get_term("leg_position")
    metrics = env.metrics
    vec.reset()
    metrics.reset()
    obs = vec.get_observations()
    sums = {key: 0.0 for key in METRICS}
    weight = 0.0
    lengths = []
    deltas = []
    previous = None

    def absorb(batch: dict, count: int) -> None:
        nonlocal weight
        for key in METRICS:
            if key in batch:
                sums[key] += float(batch[key]) * count
        weight += count

    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(obs, stochastic_output=stochastic)
        length_before = base.episode_length_buf.clone()
        obs, _, dones, extras = vec.step(actions)
        target = term.processed_actions.clone()
        if previous is not None:
            alive = length_before > 0
            deltas.append((target - previous)[alive].abs().flatten())
        previous = target
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids):
            lengths.append(length_before[done_ids].float() + 1.0)
            log = extras.get("log") or extras.get("episode") or {}
            absorb({k: (v.item() if hasattr(v, "item") else v) for k, v in log.items()}, len(done_ids))
    running = (base.episode_length_buf > 0).nonzero(as_tuple=False).flatten()
    if len(running):
        zeros = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        truncated = zeros.clone()
        truncated[running] = True
        batch = aggregate_metric_batch(metrics.episode_metrics(running, zeros, truncated, zeros), "train")
        absorb({k: (v.item() if hasattr(v, "item") else v) for k, v in batch.items()}, len(running))
    delta = torch.cat(deltas)
    completed = torch.cat(lengths) if lengths else torch.zeros(0)
    quantiles = torch.quantile(delta[torch.randperm(len(delta), device=delta.device)[:2_000_000]],
                               torch.tensor([0.5, 0.9, 0.99], device=delta.device))
    return {
        "metrics": {key: (sums[key] / weight if weight else None) for key in METRICS},
        "episodes_completed": int(len(completed)),
        "mean_completed_episode_length": float(completed.mean()) if len(completed) else None,
        "survived_to_horizon": int(len(running)),
        "target_step_change_rad": {
            "mean": float(delta.mean()),
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p99": float(quantiles[2]),
            "fraction_over_servo_step": float((delta > SERVO_STEP_RAD).float().mean()),
            "servo_step_rad": SERVO_STEP_RAD,
        },
    }


def main() -> None:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    import importlib.metadata as metadata
    from isaaclab_tasks.utils import load_cfg_from_registry
    from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config

    config = load_config(args.config, args.overrides)
    env = build_env(config)
    try:
        vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
        agent = handle_deprecated_rsl_rl_cfg(
            load_cfg_from_registry(str(config.task), "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent.to_dict()), log_dir=None,
                                device=args.device)
        runner.load(str(args.checkpoint))
        runner.alg.eval_mode()
        actor = runner.alg.actor
        steps = args.steps or int(round(env.unwrapped.max_episode_length))
        report = {"checkpoint": str(args.checkpoint), "overrides": args.overrides, "num_envs": args.num_envs,
                  "steps": steps, "asset_root": os.environ.get("SF_QUAD_ASSET_ROOT")}
        for label, stochastic in (("deterministic", False), ("noisy", True)):
            report[label] = rollout(env, vec, actor, stochastic, steps)
            print(f"[EVAL] {label}: {json.dumps(report[label])}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[EVAL] wrote {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
