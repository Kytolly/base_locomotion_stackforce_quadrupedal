#!/usr/bin/env python3
"""Does the policy respond to its command?  Fixed commands, deterministic policy, no training.

For each fixed command (forward, lateral, yaw, height) every env runs one full
episode with the command overwritten on the command term each step (target and
shaped value, so observation and rewards see it).  Reports mean body-frame
forward/lateral velocity and yaw rate after a 2 s settle, and prints the
observation's command slice next to the commanded value for a few steps.

    python scripts/diagnose_command_response.py --headless --checkpoint <model.pt> [overrides]
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
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("overrides", nargs="*")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
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


def build_env(config):
    cfg = parse_env_cfg(str(config.task), device=args.device, num_envs=args.num_envs,
                        use_fabric=bool(config.launcher.get("use_fabric", True)))
    cfg.seed = int(config.env.seed)
    cfg.sim.dt = float(config.env.sim_dt)
    cfg.decimation = int(config.env.decimation)
    cfg.episode_length_s = 1.0e6  # no time-outs inside a case
    if config.env.get("terrain_profile"):
        cfg.terrain_profile = str(config.env.terrain_profile)
        cfg.scene.terrain.terrain_generator = get_complex_terrain_cfg(seed=int(config.env.seed), profile=cfg.terrain_profile)
    apply_e0_env_options(cfg, OmegaConf.to_container(config.env, resolve=True))
    if config.get("curriculum"):
        apply_config(cfg.curriculum, config.curriculum, "curriculum")
    for name, value in (config.env.get("physics") or {}).items():
        setattr(cfg.sim.physics, name, int(value))
    return gym.make(str(config.task), cfg=cfg)


CASES = {
    "forward +0.30": (0.30, 0.0, 0.0),
    "forward +0.45": (0.45, 0.0, 0.0),
    "stop": (0.0, 0.0, 0.0),
    "backward -0.30": (-0.30, 0.0, 0.0),
    "yaw +0.25": (0.0, 0.0, 0.25),
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
        agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(str(config.task), "rsl_rl_cfg_entry_point"),
                                             metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent.to_dict()), log_dir=None,
                                device=args.device)
        runner.load(str(args.checkpoint))
        runner.alg.eval_mode()
        actor = runner.alg.actor
        base = env.unwrapped
        term = base.command_manager.get_term("locomotion")
        robot = base.scene["robot"]
        steps = round(args.seconds / base.step_dt)
        settle = round(2.0 / base.step_dt)
        report = {"checkpoint": str(args.checkpoint), "overrides": args.overrides, "cases": {}}
        for name, (fwd, lat, yaw) in CASES.items():
            vec.reset()
            height = float(term.command[0, 3])
            value = torch.tensor([fwd, lat, yaw, height], device=base.device)
            sums = torch.zeros(3, device=base.device)
            count = 0
            resets = 0
            samples = []
            for step in range(steps):
                term._target_command[:] = value
                term._command[:] = value
                obs = vec.get_observations()
                if step in (0, 1, settle):
                    samples.append({"step": step, "obs_command_slice": obs["policy"][0, :4].tolist(),
                                    "commanded": value.tolist()})
                with torch.inference_mode():
                    actions = actor(obs, stochastic_output=False)
                _, _, dones, _ = vec.step(actions)
                resets += int(dones.sum())
                if step >= settle:
                    lin, ang = robot.data.root_lin_vel_b, robot.data.root_ang_vel_b
                    lin = lin.torch if hasattr(lin, "torch") else lin
                    ang = ang.torch if hasattr(ang, "torch") else ang
                    sums += torch.stack((lin[:, 1].mean(), lin[:, 0].mean(), ang[:, 2].mean()))
                    count += 1
            mean = (sums / count).tolist()
            report["cases"][name] = {"command": value.tolist(), "forward_mps": mean[0], "lateral_mps": mean[1],
                                     "yaw_radps": mean[2], "terminations": resets, "obs_samples": samples}
            print(f"[CMD] {name:16s} forward={mean[0]:+.3f} m/s lateral={mean[1]:+.3f} yaw={mean[2]:+.3f} "
                  f"terminations={resets}  obs[0,:4]@settle={[round(v, 3) for v in samples[-1]['obs_command_slice']]}",
                  flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        simulation_app.close(exit_code=code)
