#!/usr/bin/env python3
"""Verify that the G5-T6 contact period preserves contact and air-time state."""

from __future__ import annotations

import argparse
import ctypes
import json
import traceback
from pathlib import Path


def _preload_cuda_nvrtc() -> None:
    """Load CUDA 13 NVRTC dependencies before Kit initializes."""
    root = Path("/usr/local/cuda-13.0/targets/x86_64-linux/lib")
    for name in ("libnvJitLink.so.13", "libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
        path = root / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_cuda_nvrtc()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--period", type=float, default=0.02)
parser.add_argument("--air-seconds", type=float, default=0.30)
parser.add_argument("--contact-seconds", type=float, default=0.50)
parser.add_argument("--output", type=Path, default=Path("output/fix_plan/G5/T6_contact_period.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from fix_plan_common import LEGS, Rig, make_env, write_json


def _sensor_values(env, field: str) -> torch.Tensor:
    return torch.stack(
        [getattr(env.scene[f"contact_{leg.lower()}"].data, field).reshape(env.num_envs, -1).amax(dim=1) for leg in LEGS],
        dim=1,
    )


def _step_with_root(rig: Rig, root_state: torch.Tensor) -> None:
    rig.robot.write_root_pose_to_sim(root_state[:, :7])
    rig.robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    rig.env.scene.write_data_to_sim()
    rig.env.sim.step(render=False)
    rig.env.scene.update(dt=rig.physics_dt)


def main() -> int:
    env = make_env(
        args.num_envs,
        args.device,
        suspended=False,
        foot_contact_update_period=args.period,
    )
    try:
        rig = Rig(env, suspended=False)
        root = rig.root_state.clone()
        grounded = root.clone()
        grounded[:, 2] -= 0.08
        airborne = root.clone()
        airborne[:, 2] += 0.25
        air_trace = []
        for _ in range(max(1, round(args.air_seconds / rig.physics_dt))):
            _step_with_root(rig, airborne)
            air_trace.append(_sensor_values(env, "current_air_time").clone())

        for _ in range(max(1, round(args.contact_seconds / rig.physics_dt))):
            _step_with_root(rig, grounded)
        contact_time = _sensor_values(env, "current_contact_time")
        contact_force = torch.stack(
            [env.scene[f"contact_{leg.lower()}"].data.net_forces_w.norm(dim=-1).amax(dim=-1) for leg in LEGS],
            dim=1,
        )

        air = torch.stack(air_trace)
        sampled = air[1:] - air[:-1]
        positive_jumps = sampled[sampled > 1.0e-6]
        checks = {
            "finite": bool(torch.isfinite(air).all() and torch.isfinite(contact_time).all()),
            "air_time_nonzero": bool(air[-1].min() >= args.air_seconds - args.period - rig.physics_dt),
            "air_time_jump_bounded": bool(
                positive_jumps.numel() > 0 and positive_jumps.max() <= args.period + rig.physics_dt + 1.0e-6
            ),
            "contact_time_nonzero": bool(contact_time.max(dim=1).values.min() > 0.0),
            "contact_force_nonzero": bool(contact_force.max(dim=1).values.min() > 0.0),
        }
        payload = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "period_s": args.period,
            "physics_dt_s": rig.physics_dt,
            "air_time_final_min_s": float(air[-1].min()),
            "air_time_final_max_s": float(air[-1].max()),
            "air_time_positive_jump_max_s": float(positive_jumps.max()) if positive_jumps.numel() else 0.0,
            "supported_contact_time_final_min_s": float(contact_time.max(dim=1).values.min()),
            "contact_force_final_max_n": float(contact_force.max()),
            "checks": checks,
        }
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0 if payload["status"] == "PASS" else 1
    finally:
        env.close()


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(code)
