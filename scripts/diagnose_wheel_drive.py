#!/usr/bin/env python3
"""Open-loop wheel-drive check (no policy): does commanded wheel speed move the base?

Free base on a plane, legs held at the E0-v2 stance, all wheel joints given a
velocity target directly (bypassing the action term).  For each case it
reports mean base velocity over the last second in the world and body frames
and compares body +y (the task's "forward") with r * omega.  Cases: equal
speeds at several values, then left/right opposite speeds for the yaw sign.
Also prints the wheel joint drive parameters and the policy-action path
(raw -> target) for one wheel.

    python scripts/diagnose_wheel_drive.py --headless
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[1] / "output/diagnostics/wheel_drive.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from fix_plan_common import LEGS, Rig, make_env  # noqa: E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.support import (  # noqa: E402,E501
    WHEEL_RADIUS_M,
)

LEFT = {"FL", "RL"}


def main() -> None:
    env = make_env(1, args.device, suspended=False)
    rig = Rig(env, suspended=False)
    robot = rig.robot
    wheel_ids, wheel_names = robot.find_joints([f"{leg}_Wheel_Joint" for leg in LEGS], preserve_order=True)
    report = {
        "wheel_radius_m": WHEEL_RADIUS_M,
        "wheel_joints": wheel_names,
        "wheel_stiffness": robot.data.joint_stiffness.torch[0, wheel_ids].tolist()
        if hasattr(robot.data.joint_stiffness, "torch") else None,
        "wheel_damping": robot.data.joint_damping.torch[0, wheel_ids].tolist()
        if hasattr(robot.data.joint_damping, "torch") else None,
        "cases": [],
    }
    term = env.action_manager.get_term("wheel_velocity")
    report["wheel_action_scale"] = float(term._scale) if not torch.is_tensor(term._scale) else term._scale.tolist()
    stance = rig.default_leg_pos
    rig.run(stance, 0.5, start=stance)
    rig.hold(1.0)
    cases = [(f"all {w:+g}", [w] * 4) for w in (2.0, 5.0, 10.0, -5.0)]
    cases.append(("left +5 / right -5", [5.0 if leg in LEFT else -5.0 for leg in LEGS]))
    try:
        for name, speeds in cases:
            target = torch.tensor([speeds], device=env.device)
            steps = round(args.seconds / rig.physics_dt)
            tail = round(1.0 / rig.physics_dt)
            acc = {"v_w": torch.zeros(3, device=env.device), "v_b": torch.zeros(3, device=env.device),
                   "w_b": torch.zeros(3, device=env.device), "wheel": torch.zeros(4, device=env.device)}
            start_pos = None
            for i in range(steps):
                robot.set_joint_position_target(stance, joint_ids=rig.leg_ids)
                robot.set_joint_velocity_target(target, joint_ids=wheel_ids)
                rig.step_physics()
                data = robot.data
                if i == steps - tail:
                    start_pos = data.root_pos_w[0].clone()
                if i >= steps - tail:
                    acc["v_w"] += data.root_lin_vel_w[0]
                    acc["v_b"] += data.root_lin_vel_b[0]
                    acc["w_b"] += data.root_ang_vel_b[0]
                    acc["wheel"] += data.joint_vel[0, wheel_ids]
            displacement = (robot.data.root_pos_w[0] - start_pos).tolist()
            mean = {k: (v / tail).tolist() for k, v in acc.items()}
            row = {
                "case": name, "wheel_target_rad_s": speeds,
                "base_vel_world_mps": mean["v_w"], "base_vel_body_mps": mean["v_b"],
                "base_angvel_body_radps": mean["w_b"], "wheel_vel_rad_s": mean["wheel"],
                "displacement_last_1s_world_m": displacement,
                "r_omega_mps": WHEEL_RADIUS_M * sum(speeds) / 4,
            }
            report["cases"].append(row)
            print(f"[WHEEL] {name:22s} body v=({mean['v_b'][0]:+.3f}, {mean['v_b'][1]:+.3f}, {mean['v_b'][2]:+.3f}) m/s "
                  f"yaw={mean['w_b'][2]:+.3f} rad/s  wheel={[round(v, 2) for v in mean['wheel']]}  "
                  f"r*w={row['r_omega_mps']:+.3f}  disp={[round(d, 3) for d in displacement]}", flush=True)
            # Settle between cases.
            for _ in range(round(1.0 / rig.physics_dt)):
                robot.set_joint_position_target(stance, joint_ids=rig.leg_ids)
                robot.set_joint_velocity_target(torch.zeros_like(target), joint_ids=wheel_ids)
                rig.step_physics()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[WHEEL] wrote {args.output}", flush=True)
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
