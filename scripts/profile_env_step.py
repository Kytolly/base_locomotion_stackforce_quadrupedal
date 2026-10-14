#!/usr/bin/env python3
"""G0: where does the sampling time of one control step go?

Builds the training environment from a training YAML (default
``configs/train/main/b2_bisec_full.yaml``) with the same env construction as
``scripts/rsl_rl/train.py``, wraps it in ``TrainingMetricsWrapper`` like
training does, and times each segment of ``env.step`` with
``torch.cuda.synchronize()`` before and after every segment:

physics (``sim.step`` + ``scene.update`` over the decimation substeps), the
action / observation / reward / termination / curriculum / command / event
managers, the recorder (``LocomotionStepRecorder`` + training metrics) and the
reset path.  Zero actions by default; ``--checkpoint`` runs a policy.

Not part of the training path.  Output: ``output/fix_plan/G0/step_profile.json``.
Pass: the segment sum is within 10 % of the measured step time.

    python scripts/profile_env_step.py --headless
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import functools
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Isaac Sim can load its CUDA 12 NVRTC first; resolve the active environment's
# CUDA 13 builtins before Kit starts (as in the train and validation launchers).
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
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--output", type=Path, default=Path("output/fix_plan/G0/step_profile.json"))
parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides for the training YAML.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_plan_common import write_json  # noqa: E402

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


TIMES: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)
_DEPTH = [0]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(owner, attribute: str, key: str) -> None:
    """Replace ``owner.attribute`` with a synchronised timer (outermost call only)."""
    original = getattr(owner, attribute)

    @functools.wraps(original)
    def wrapper(*a, **kw):
        if _DEPTH[0]:
            return original(*a, **kw)
        _DEPTH[0] += 1
        _sync()
        start = time.perf_counter()
        try:
            return original(*a, **kw)
        finally:
            _sync()
            TIMES[key] += time.perf_counter() - start
            CALLS[key] += 1
            _DEPTH[0] -= 1

    setattr(owner, attribute, wrapper)


def build_env(config):
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
    env = gym.make(str(config.task), cfg=env_cfg)
    return TrainingMetricsWrapper(env, config.get("logging", {}))


def instrument(env) -> None:
    base = env.unwrapped
    timed(base.sim, "step", "physics/sim_step")
    timed(base.scene, "update", "physics/scene_update")
    timed(base.scene, "write_data_to_sim", "physics/write_data_to_sim")
    timed(base.action_manager, "process_action", "manager/action_process")
    timed(base.action_manager, "apply_action", "manager/action_apply")
    timed(base.observation_manager, "compute", "manager/observation")
    timed(base.reward_manager, "compute", "manager/reward")
    timed(base.termination_manager, "compute", "manager/termination")
    timed(base.command_manager, "compute", "manager/command")
    if getattr(base, "curriculum_manager", None) is not None:
        timed(base.curriculum_manager, "compute", "manager/curriculum")
    timed(base.event_manager, "apply", "manager/event")
    timed(base.recorder_manager, "record_post_step", "recorder/post_step_and_training_metrics")
    timed(base, "_reset_idx", "reset/reset_idx")


def main() -> None:
    def progress(message: str) -> None:
        print(f"[G0 profile] {message}", file=sys.stderr, flush=True)

    config = load_config(args.config, args.overrides)
    progress("building environment")
    env = build_env(config)
    progress("environment built")
    policy = None
    try:
        if args.checkpoint is not None:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
            from rsl_rl.runners import OnPolicyRunner
            import importlib.metadata as metadata
            from isaaclab_tasks.utils import load_cfg_from_registry
            from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config

            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            agent = handle_deprecated_rsl_rl_cfg(
                load_cfg_from_registry(str(config.task), "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
            )
            runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent.to_dict()), log_dir=None,
                                    device=args.device)
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=args.device)
            step = lambda: vec.step(policy(vec.get_observations()))  # noqa: E731
        else:
            progress("resetting environment")
            env.reset()
            progress("reset complete")
            action = torch.zeros((args.num_envs, env.unwrapped.action_manager.total_action_dim), device=args.device)
            step = lambda: env.step(action)  # noqa: E731
        for index in range(args.warmup_steps):
            step()
            progress(f"warmup {index + 1}/{args.warmup_steps}")
        progress("instrumenting")
        instrument(env)
        TIMES.clear()
        CALLS.clear()
        _sync()
        started = time.perf_counter()
        for index in range(args.steps):
            with torch.inference_mode():
                step()
            if (index + 1) % 10 == 0 or index + 1 == args.steps:
                progress(f"measured {index + 1}/{args.steps}")
        _sync()
        wall = time.perf_counter() - started
        progress("measurement complete")
    finally:
        progress("closing environment")
        env.close()
        progress("environment closed")

    total = wall  # includes the metrics wrapper and Python glue between segments
    segments = {
        key: {"ms_per_step": 1000.0 * value / args.steps, "share": value / total, "calls_per_step": CALLS[key] / args.steps}
        for key, value in sorted(TIMES.items(), key=lambda item: -item[1])
    }
    covered = sum(TIMES.values())
    deviation = abs(covered - total) / total
    write_json(args.output, {
        "plan_section": "G0",
        "config": str(args.config),
        "overrides": list(args.overrides),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "policy": "checkpoint" if args.checkpoint else "zero_action",
        "measured_step_ms": 1000.0 * total / args.steps,
        "segment_sum_ms": 1000.0 * covered / args.steps,
        "segment_sum_deviation": deviation,
        "estimated_collection_s_per_iteration": total / args.steps * int(config.agent.num_steps_per_env),
        "segments": segments,
        "unattributed_ms_per_step": 1000.0 * (total - covered) / args.steps,
        "status": "PASS" if deviation <= 0.10 else "FAIL",
        "on_fail": "re-measure with torch.profiler; does not block G1-G3",
    })
    print(json.dumps({k: round(v["ms_per_step"], 2) for k, v in segments.items()}, indent=2))


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(exit_code=exit_code)
