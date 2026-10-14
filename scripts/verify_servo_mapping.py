#!/usr/bin/env python3
"""G3.0: verify the section 3A servo mapping in simulation (suspended, held root).

Checks, per leg (docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md G3 step 1):

1. straight legs (alpha = beta = 90 deg): wheel centre 158 mm below the hip axes;
2. new default stance (128.4 / 66.6 deg): wheel centre about 137 mm below;
   both with closure residual <= 1 mm and wheel position error <= 3 mm against
   the URDF five-bar model;
3. +0.1 rad on each hip joint from the stance turns its crank in the direction
   of the section 3A ``s_i`` table.

Output: ``output/fix_plan/G3/mapping_check.json`` (``status`` PASS/FAIL).  On
FAIL stop and correct section 3A before any further G3 step.

    python scripts/verify_servo_mapping.py --headless
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import select
import time
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
parser.add_argument("--output", type=Path, default=Path("output/fix_plan/G3/mapping_check.json"))
parser.add_argument("--settle-s", type=float, default=1.5)
parser.add_argument("--residual-limit-mm", type=float, default=1.0)
parser.add_argument("--position-limit-mm", type=float, default=3.0)
parser.add_argument("--diagnostic-outer-knee-limit-rad", type=float, default=None,
                    help="Temporary symmetric passive limit; diagnostic only, never gate evidence")
parser.add_argument("--interactive", action="store_true",
                    help="Keep the GUI open and pause after each checked pose")
parser.add_argument("--gui-inspect", action="store_true",
                    help="Free-base stance/extend/bend cycle for visual inspection, without gate evidence")
AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3.0 needs explicit Kit visualizer intent to open a viewport.
if ("--gui-inspect" in sys.argv or "--interactive" in sys.argv) and "--headless" not in sys.argv and not any(
    option == flag or option.startswith(flag + "=")
    for option in sys.argv for flag in ("--viz", "--visualizer")
):
    sys.argv.extend(["--viz", "kit"])
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from fix_plan_common import LEGS, Rig, make_env, write_json  # noqa: E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots.kinematics import (  # noqa: E402,E501
    LegKinematics,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E402,E501
    load_servo_contract,
)

EXPECTED_SIGNS = {"FR": (1, -1), "FL": (1, 1), "RL": (-1, -1), "RR": (-1, 1)}  # section 3A s_i (alpha, beta)
EXPECTED_DEPTH_M = {"straight": 0.158, "stance": 0.137}


def inspect_gui() -> None:
    if args.headless:
        raise ValueError("--gui-inspect requires a GUI; omit --headless")
    env = make_env(1, args.device, suspended=False)
    rig = Rig(env, suspended=False)
    kin = LegKinematics()
    knee_ids, _ = rig.robot.find_joints([f"{leg}_Outer_Knee_Joint" for leg in LEGS], preserve_order=True)
    original_knee_limits = rig.robot.data.joint_pos_limits[:, knee_ids].clone()
    if args.diagnostic_outer_knee_limit_rad is not None:
        limit = args.diagnostic_outer_knee_limit_rad
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("diagnostic knee limit must be positive and finite")
        limits = original_knee_limits.clone()
        limits[..., 0], limits[..., 1] = -limit, limit
        rig.robot.write_joint_position_limit_to_sim(limits, joint_ids=knee_ids)
        print(f"[GUI] DIAGNOSTIC ONLY: outer-knee limits changed from ±{float(original_knee_limits[0, 0, 1]):.3f} "
              f"to ±{limit:.3f} rad for this process; the asset is unchanged.", flush=True)
    env.sim.set_camera_view(eye=(0.65, 0.65, 0.4), target=(0.0, 0.0, 0.12))
    poses = [("STANCE 128.4/66.6", None), ("EXTEND 90/90", (90.0, 90.0)),
             ("STANCE 128.4/66.6", None), ("BEND 150/45", (150.0, 45.0))]
    mode = "press Enter to advance" if args.interactive else "automatic 3-second holds"
    print(f"[GUI] Original knee limits retained; {mode}. Close the window to stop.", flush=True)
    start = rig.default_leg_pos
    try:
        while simulation_app.is_running():
            for name, angles in poses:
                if not simulation_app.is_running():
                    return
                target = rig.default_leg_pos if angles is None else torch.tensor(
                    [[kin.hips_for_crank_angles(leg, *angles)[0] for leg in LEGS]
                     + [kin.hips_for_crank_angles(leg, *angles)[1] for leg in LEGS]], device=env.device)
                before_q = rig.robot.data.joint_pos[0, rig.leg_ids].clone()
                before_z = float(rig.robot.data.root_pos_w[0, 2])
                commanded_change = float((target - start).abs().max())
                print(f"\n[GUI] MOVING TO {name}", flush=True)
                for index in range(round(4.0 / rig.physics_dt)):
                    if not simulation_app.is_running():
                        return
                    alpha = min(1.0, (index + 1) * rig.physics_dt / 4.0)
                    rig.robot.set_joint_position_target(start + alpha * (target - start), joint_ids=rig.leg_ids)
                    rig.step_physics()
                    env.sim.render()
                    time.sleep(rig.physics_dt)
                start = target
                actual_q = rig.robot.data.joint_pos[0, rig.leg_ids]
                print(f"[GUI] HOLDING {name}: commanded change={commanded_change:.3f} rad, "
                      f"observed change={float((actual_q - before_q).abs().max()):.3f} rad, "
                      f"base dz={float(rig.robot.data.root_pos_w[0, 2]) - before_z:+.4f} m", flush=True)
                last_log = 0.0
                hold_start = time.monotonic()
                while simulation_app.is_running():
                    rig.step_physics()
                    env.sim.render()
                    now = time.monotonic()
                    if now - last_log >= 1.0:
                        print(f"  base z={rig.robot.data.root_pos_w[0, 2].item():.4f} m, "
                              f"hip error max={rig.tracking_error().max().item():.4f} rad, "
                              f"closure max={rig.closure_residual_mm().max().item():.4f} mm, "
                              f"outer-knee |q| max={rig.robot.data.joint_pos[0, knee_ids].abs().max().item():.4f} rad",
                              flush=True)
                        last_log = now
                    if args.interactive and select.select([sys.stdin], [], [], 0)[0]:
                        if not sys.stdin.readline():
                            return
                        break
                    if not args.interactive and now - hold_start >= 3.0:
                        break
                    time.sleep(rig.physics_dt)
    finally:
        env.close()


def _crank_deg(vector, front: bool) -> float:
    """Section 3A crank angle from a base-frame (y, z) crank vector."""
    sign = 1.0 if front else -1.0
    return math.degrees(math.atan2(-float(vector[2]), sign * float(vector[1])))


def main() -> None:
    kin = LegKinematics()
    contract = load_servo_contract()
    env = make_env(1, args.device, suspended=True)
    rig = Rig(env, suspended=True)
    report: dict = {"plan_section": "G3.0", "contract_status": contract["status"], "poses": {}, "signs": {}}
    if args.diagnostic_outer_knee_limit_rad is not None:
        if args.output == Path("output/fix_plan/G3/mapping_check.json"):
            raise ValueError("diagnostic override requires a separate --output path")
        limit = args.diagnostic_outer_knee_limit_rad
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("diagnostic knee limit must be positive and finite")
        ids, _ = rig.robot.find_joints([f"{leg}_Outer_Knee_Joint" for leg in LEGS], preserve_order=True)
        original = rig.robot.data.joint_pos_limits[:, ids].clone()
        limits = original.clone()
        limits[..., 0], limits[..., 1] = -limit, limit
        rig.robot.write_joint_position_limit_to_sim(limits, joint_ids=ids)
        report["diagnostic_override"] = {
            "outer_knee_limit_rad": limit, "original_limits_rad": original.cpu().tolist(),
        }
    passed = True
    try:
        # Hip-axis height in the base frame: mean of the outer/inner hip joint origins (URDF).
        hip_z = {
            leg: 0.5 * (kin.joints[f"{leg}_Outer_Hip_Joint"].xyz[2] + kin.joints[f"{leg}_Inner_Hip_Joint"].xyz[2])
            for leg in LEGS
        }
        poses = {
            "straight": {leg: kin.hips_for_crank_angles(leg, 90.0, 90.0) for leg in LEGS},
            "stance": {
                leg: (contract["joints"][f"{leg}_Outer_Hip_Joint"]["default"],
                      contract["joints"][f"{leg}_Inner_Hip_Joint"]["default"])
                for leg in LEGS
            },
        }
        start = rig.default_leg_pos
        for name, hips in poses.items():
            target = torch.tensor(
                [[hips[leg][0] for leg in LEGS] + [hips[leg][1] for leg in LEGS]], device=env.device
            )
            rig.run(target, 1.0, start=start)
            rig.hold(args.settle_s)
            start = target
            feet = rig.feet_base()[0]
            residual = rig.closure_residual_mm()[0]
            rows = {}
            for index, leg in enumerate(LEGS):
                q_outer, q_inner = hips[leg]
                k_outer, k_inner, _ = kin.solve_knees(leg, q_outer, q_inner)
                _, wheel_model = kin.outer_chain(leg, q_outer, k_outer)
                depth = hip_z[leg] - float(feet[index, 2])
                error_mm = float(((feet[index, 1:] - torch.tensor(wheel_model[1:], device=env.device)).norm()) * 1000.0)
                ok = bool(
                    residual[index].item() <= args.residual_limit_mm
                    and error_mm <= args.position_limit_mm
                    and abs(depth - EXPECTED_DEPTH_M[name]) <= 0.003
                )
                passed &= ok
                rows[leg] = {
                    "hip_targets_rad": [q_outer, q_inner],
                    "wheel_depth_below_hips_m": depth,
                    "expected_depth_m": EXPECTED_DEPTH_M[name],
                    "wheel_vs_urdf_model_error_mm": error_mm,
                    "closure_residual_mm": float(residual[index]),
                    "tracking_error_max_rad": float(rig.tracking_error()[0, [index, index + 4]].max()),
                    "pass": ok,
                }
            report["poses"][name] = rows
            if args.interactive:
                print(f"[G3.0] GUI pose ready: {name}. Inspect joint stops, leg closure, and body height; press Enter to continue.", flush=True)
                while simulation_app.is_running():
                    env.sim.render()
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        break
                    time.sleep(0.02)

        # Sign check: +0.1 rad on one hip from the stance.
        stance = start.clone()
        for index, leg in enumerate(LEGS):
            front = leg in ("FR", "FL")
            outer_vec, inner_vec = rig.crank_vectors_base()
            alpha0, beta0 = _crank_deg(outer_vec[0, index], front), _crank_deg(inner_vec[0, index], front)
            measured = []
            for joint_offset in (0, 4):  # outer, inner
                target = stance.clone()
                target[0, index + joint_offset] += 0.1
                rig.run(target, 0.5, start=stance)
                rig.hold(0.5)
                outer_vec, inner_vec = rig.crank_vectors_base()
                alpha1, beta1 = _crank_deg(outer_vec[0, index], front), _crank_deg(inner_vec[0, index], front)
                delta = (alpha1 - alpha0) if joint_offset == 0 else (beta1 - beta0)
                measured.append(1 if delta > 0 else -1)
                rig.run(stance, 0.5, start=target)
                rig.hold(0.3)
            ok = tuple(measured) == EXPECTED_SIGNS[leg]
            passed &= ok
            report["signs"][leg] = {"measured": measured, "expected": list(EXPECTED_SIGNS[leg]), "pass": ok}
        report["status"] = "DIAGNOSTIC" if args.diagnostic_outer_knee_limit_rad is not None else ("PASS" if passed else "FAIL")
        report["checks_passed"] = bool(passed)
        report["stance_crank_angles_deg"] = contract["stance_crank_angles_deg"]
        write_json(args.output, report)
        print(f"[G3.0] mapping check {report['status']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        inspect_gui() if args.gui_inspect else main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
