#!/usr/bin/env python3
"""G3 steps 2-5: closed-loop feasibility scan, final ranges, stance and lift.

``--mode scan`` (suspended, root held in place, all grid points in parallel):
    For each leg, a 2-D grid over (outer, inner) inside the current servo
    contract ranges (section 3A while provisional), step 0.05 rad.  One leg moves
    per pass; the other three stay at the stance.  Every point is ramped to in
    0.5 s and held for 1 s; the last 0.2 s is measured.  A point is feasible when
    closure residual <= 1 mm, no self-contact (base or moving-wheel contact
    sensor > 1 N while the plane is out of reach), tracking error <= 0.05 rad on
    both hips, and all values are finite.
    Writes range_scan.json, final_ranges.json (largest per-leg box around the
    stance, lift side grown first; ``projection_recommended`` when the box keeps
    less than ``--box-keep-min`` of the feasible lift-side points),
    feasible_polygon.json and lift.json (max body-relative wheel lift inside the
    final box with fore-aft wheel shift <= 25 mm, as in section 3A).

``--mode stance`` (free base, plane, zero hip motion for 5 s):
    Writes stance.json: standing body height above the wheel support plane, wheel
    contacts and steady hip torques against stall (0.392 N m) and rated
    (0.127 N m) torque.

Leg-to-leg contacts are not sensed (only base and wheel sensors exist); the
report says so.  Run both modes, then ``scripts/fix_plan_gate.py g3``.

    python scripts/probe_action_range.py --mode scan --headless
    python scripts/probe_action_range.py --mode stance --headless
"""

from __future__ import annotations

import argparse
import ctypes
import math
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
parser.add_argument("--mode", choices=("scan", "stance"), required=True)
parser.add_argument("--output-dir", type=Path, default=Path("output/fix_plan/G3"))
parser.add_argument("--step-rad", type=float, default=0.05)
parser.add_argument("--residual-limit-mm", type=float, default=1.0)
parser.add_argument("--tracking-limit-rad", type=float, default=0.05)
parser.add_argument("--contact-limit-n", type=float, default=1.0)
parser.add_argument("--max-fore-aft-shift-m", type=float, default=0.025)
parser.add_argument("--box-keep-min", type=float, default=0.6)
parser.add_argument("--stance-seconds", type=float, default=5.0)
parser.add_argument("--max-envs", type=int, default=4096)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from fix_plan_common import LEGS, Rig, make_env, write_json  # noqa: E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E402,E501
    load_servo_contract,
)

STALL_NM = 0.392
RATED_NM = 0.127


def _grid(lower: float, upper: float) -> list[float]:
    count = int(math.ceil((upper - lower) / args.step_rad)) + 1
    return [lower + (upper - lower) * index / (count - 1) for index in range(count)]


def _lift_signs(leg: str) -> tuple[int, int]:
    """dq signs that raise the wheel: alpha up and beta down (section 3A s_i table)."""
    s_alpha, s_beta = {"FR": (1, -1), "FL": (1, 1), "RL": (-1, -1), "RR": (-1, 1)}[leg]
    return s_alpha, -s_beta


def _largest_box(feasible, outer, inner, i0, j0, lift):
    """Grow an index box around the stance cell while every cell stays feasible; lift sides first."""
    lo_i = hi_i = i0
    lo_j = hi_j = j0
    sides = []
    sides.append("hi_i" if lift[0] > 0 else "lo_i")
    sides.append("hi_j" if lift[1] > 0 else "lo_j")
    sides += [side for side in ("lo_i", "hi_i", "lo_j", "hi_j") if side not in sides]
    grown = True
    while grown:
        grown = False
        for side in sides:
            if side == "lo_i" and lo_i > 0 and all(feasible[lo_i - 1][j] for j in range(lo_j, hi_j + 1)):
                lo_i -= 1
                grown = True
            elif side == "hi_i" and hi_i < len(outer) - 1 and all(feasible[hi_i + 1][j] for j in range(lo_j, hi_j + 1)):
                hi_i += 1
                grown = True
            elif side == "lo_j" and lo_j > 0 and all(feasible[i][lo_j - 1] for i in range(lo_i, hi_i + 1)):
                lo_j -= 1
                grown = True
            elif side == "hi_j" and hi_j < len(inner) - 1 and all(feasible[i][hi_j + 1] for i in range(lo_i, hi_i + 1)):
                hi_j += 1
                grown = True
    return lo_i, hi_i, lo_j, hi_j


def scan() -> None:
    contract = load_servo_contract()
    joints = contract["joints"]
    grids = {}
    for leg in LEGS:
        outer = joints[f"{leg}_Outer_Hip_Joint"]
        inner = joints[f"{leg}_Inner_Hip_Joint"]
        grids[leg] = (_grid(outer["lower"], outer["upper"]), _grid(inner["lower"], inner["upper"]))
    points = max(len(o) * len(i) for o, i in grids.values())
    if args.max_envs <= 0:
        raise ValueError("--max-envs must be positive")
    batch_size = min(points, args.max_envs)
    env = make_env(batch_size, args.device, suspended=True)
    rig = Rig(env, suspended=True)
    stance = rig.default_leg_pos
    results = {}
    try:
        for index, leg in enumerate(LEGS):
            outer_grid, inner_grid = grids[leg]
            pairs = [(qo, qi) for qo in outer_grid for qi in inner_grid]
            collected = {name: [] for name in ("residual", "contact", "track", "feet")}
            for offset in range(0, len(pairs), batch_size):
                chunk = pairs[offset : offset + batch_size]
                count = len(chunk)
                target = stance.clone()
                target[:count, index] = torch.tensor([p[0] for p in chunk], device=env.device)
                target[:count, index + 4] = torch.tensor([p[1] for p in chunk], device=env.device)
                rig.run(target, 0.5, start=stance)
                rig.hold(0.8)
                samples = []
                for _ in range(4):  # 0.2 s at 20 Hz sampling
                    rig.hold(0.05)
                    base_force, foot_force = rig.contact_forces()
                    samples.append({
                        "residual": rig.closure_residual_mm()[:, index],
                        "base": base_force,
                        "foot": foot_force[:, index],
                        "track": rig.tracking_error()[:, [index, index + 4]].amax(dim=-1),
                        "feet": rig.feet_base()[:, index],
                    })
                collected["residual"].append(torch.stack([s["residual"] for s in samples]).amax(0)[:count].cpu())
                collected["contact"].append(torch.maximum(
                    torch.stack([s["base"] for s in samples]).amax(0),
                    torch.stack([s["foot"] for s in samples]).amax(0),
                )[:count].cpu())
                collected["track"].append(torch.stack([s["track"] for s in samples]).amax(0)[:count].cpu())
                collected["feet"].append(samples[-1]["feet"][:count].cpu())
                rig.run(stance, 0.5, start=target)
                rig.hold(0.5)
                print(f"[G3] {leg}: scanned {offset + count}/{len(pairs)}", flush=True)
            residual = torch.cat(collected["residual"])
            contact = torch.cat(collected["contact"])
            track = torch.cat(collected["track"])
            feet = torch.cat(collected["feet"])
            finite = torch.isfinite(residual) & torch.isfinite(contact) & torch.isfinite(track) & torch.isfinite(feet).all(-1)
            feasible = (
                finite
                & (residual <= args.residual_limit_mm)
                & (contact <= args.contact_limit_n)
                & (track <= args.tracking_limit_rad)
            )
            results[leg] = {
                "outer_grid": outer_grid,
                "inner_grid": inner_grid,
                "feasible": feasible.view(len(outer_grid), len(inner_grid)).tolist(),
                "closure_residual_mm": torch.nan_to_num(residual, nan=1e6).view(len(outer_grid), -1).tolist(),
                "contact_n": torch.nan_to_num(contact, nan=1e6).view(len(outer_grid), -1).tolist(),
                "tracking_error_rad": torch.nan_to_num(track, nan=1e6).view(len(outer_grid), -1).tolist(),
                "wheel_base_yz_m": torch.nan_to_num(feet[:, 1:], nan=0.0).view(len(outer_grid), len(inner_grid), 2).tolist(),
            }
            print(f"[G3] {leg}: {int(feasible.sum())}/{len(pairs)} feasible", flush=True)
    finally:
        env.close()

    write_json(args.output_dir / "range_scan.json", {
        "plan_section": "G3 step 2",
        "contract_status": contract["status"],
        "criteria": {
            "closure_residual_mm_max": args.residual_limit_mm,
            "self_contact_n_max": args.contact_limit_n,
            "tracking_error_rad_max": args.tracking_limit_rad,
            "finite": True,
        },
        "batch_size": batch_size,
        "limitations": ["leg-to-leg contacts are not sensed; only base and moving-wheel contact sensors"],
        "legs": results,
    })

    final, polygon, projection, lift = {}, {}, {}, {}
    box_ok = True
    for leg in LEGS:
        data = results[leg]
        outer, inner = data["outer_grid"], data["inner_grid"]
        q_outer = joints[f"{leg}_Outer_Hip_Joint"]["default"]
        q_inner = joints[f"{leg}_Inner_Hip_Joint"]["default"]
        i0 = min(range(len(outer)), key=lambda i: abs(outer[i] - q_outer))
        j0 = min(range(len(inner)), key=lambda j: abs(inner[j] - q_inner))
        feasible = np.asarray(data["feasible"], dtype=bool)
        if not feasible[i0, j0]:
            raise RuntimeError(f"{leg}: the default stance itself is infeasible; stop and revisit section 3A")
        labels, _ = ndimage.label(feasible, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        component = labels == labels[i0, j0]
        component_list = component.tolist()
        signs = _lift_signs(leg)
        box_lo_i, box_hi_i, box_lo_j, box_hi_j = _largest_box(
            component_list, outer, inner, i0, j0, signs
        )
        lift_side = [
            (i, j) for i in range(len(outer)) for j in range(len(inner))
            if component[i, j] and (i - i0) * signs[0] >= 0 and (j - j0) * signs[1] >= 0
        ]
        in_box = [
            (i, j) for i, j in lift_side
            if box_lo_i <= i <= box_hi_i and box_lo_j <= j <= box_hi_j
        ]
        keep = len(in_box) / max(1, len(lift_side))
        box_ok &= keep >= args.box_keep_min
        use_projection = keep < args.box_keep_min
        if use_projection:
            indices = np.argwhere(component)
            lo_i, lo_j = indices.min(axis=0)
            hi_i, hi_j = indices.max(axis=0)
            region = component[lo_i : hi_i + 1, lo_j : hi_j + 1]
            nearest = ndimage.distance_transform_edt(~region, return_distances=False, return_indices=True)
            width = region.shape[1]
            projection[leg] = {
                "outer_joint": f"{leg}_Outer_Hip_Joint",
                "inner_joint": f"{leg}_Inner_Hip_Joint",
                "outer_grid": [outer[lo_i], outer[hi_i], int(region.shape[0])],
                "inner_grid": [inner[lo_j], inner[hi_j], int(region.shape[1])],
                "feasible": region.reshape(-1).tolist(),
                "nearest_feasible_flat": (nearest[0] * width + nearest[1]).reshape(-1).tolist(),
            }
        else:
            lo_i, hi_i, lo_j, hi_j = box_lo_i, box_hi_i, box_lo_j, box_hi_j
        final[f"{leg}_Outer_Hip_Joint"] = {"lower": outer[lo_i], "upper": outer[hi_i]}
        final[f"{leg}_Inner_Hip_Joint"] = {"lower": inner[lo_j], "upper": inner[hi_j]}
        polygon[leg] = {
            "outer_grid": outer,
            "inner_grid": inner,
            "feasible": component_list,
            "largest_box_index": [box_lo_i, box_hi_i, box_lo_j, box_hi_j],
            "selected_bounds_index": [int(lo_i), int(hi_i), int(lo_j), int(hi_j)],
            "lift_side_points_kept_by_box": keep,
            "projection_enabled": use_projection,
        }
        wheel = data["wheel_base_yz_m"]
        y0, z0 = wheel[i0][j0]
        best = max(
            (
                (wheel[i][j][1] - z0, i, j)
                for i in range(lo_i, hi_i + 1)
                for j in range(lo_j, hi_j + 1)
                if component[i, j] and abs(wheel[i][j][0] - y0) <= args.max_fore_aft_shift_m
            ),
            default=(0.0, i0, j0),
        )
        lift[leg] = {"max_lift_m": best[0], "outer_rad": outer[best[1]], "inner_rad": inner[best[2]]}

    write_json(args.output_dir / "final_ranges.json", {
        "plan_section": "G3 step 3",
        "status": "PASS",
        "rule": "section 3A 0.9-travel interval intersected with the feasible set; per-joint box around the stance, no second shrink",
        "projection_recommended": not box_ok,
        "joints": final,
        "projection": projection,
    })
    write_json(args.output_dir / "feasible_polygon.json", {"plan_section": "G3 step 3", "legs": polygon})
    min_lift = min(item["max_lift_m"] for item in lift.values())
    write_json(args.output_dir / "lift.json", {
        "plan_section": "G3 step 5",
        "fore_aft_shift_limit_m": args.max_fore_aft_shift_m,
        "legs": lift,
        "min_lift_m": min_lift,
        "status": "PASS" if min_lift >= 0.030 else ("CONDITIONAL" if min_lift >= 0.025 else "FAIL"),
    })


def stance() -> None:
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.support import (
        WHEEL_RADIUS_M,
    )

    contract = load_servo_contract()
    env = make_env(1, args.device, suspended=False)
    rig = Rig(env, suspended=False)
    try:
        stance_q = rig.default_leg_pos
        rig.run(stance_q, 0.2, start=stance_q)
        heights, torques, contacts = [], [], []
        steps_per_sample = max(1, round(0.02 / rig.physics_dt))
        samples = round(args.stance_seconds / 0.02)
        for _ in range(samples):
            for _ in range(steps_per_sample):
                rig.step_physics()
            data = rig.robot.data
            wheel_z = data.body_pos_w[0, rig.foot_ids, 2]
            support = float(wheel_z.median()) - WHEEL_RADIUS_M
            heights.append(float(data.root_pos_w[0, 2]) - support)
            torques.append(rig.leg_torque()[0].abs())
            _, feet = rig.contact_forces()
            contacts.append(feet[0] > 1.0)
        tail = max(1, round(1.0 / 0.02))
        steady = torch.stack(torques[-tail:]).mean(0)
        height = sum(heights[-tail:]) / tail
        drift = max(heights[-tail:]) - min(heights[-tail:])
        all_contact = bool(torch.stack(contacts[-tail:]).all())
        tilt = math.acos(max(-1.0, min(1.0, -float(rig.robot.data.projected_gravity_b[0, 2]))))
        finite = all(math.isfinite(h) for h in heights)
        stable = finite and all_contact and tilt < 0.2 and drift < 0.005
        below_stall = bool((steady < STALL_NM).all())
        joints = {name: float(joint["default"]) for name, joint in contract["joints"].items()}
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import (
            LEG_JOINTS,
        )

        write_json(args.output_dir / "stance.json", {
            "plan_section": "G3 step 4",
            "status": "PASS" if stable and below_stall else "FAIL",
            "joints": joints,
            "stance_crank_angles_deg": contract["stance_crank_angles_deg"],
            "standing_body_height_m": round(height, 4),
            "geometric_standing_m": contract["body_height"]["geometric_standing_m"],
            "height_drift_last_1s_m": drift,
            "tilt_rad": tilt,
            "all_four_wheels_in_contact": all_contact,
            "steady_hip_torque_nm": dict(zip(LEG_JOINTS, steady.tolist())),
            "fraction_of_stall": dict(zip(LEG_JOINTS, (steady / STALL_NM).tolist())),
            "fraction_of_rated": dict(zip(LEG_JOINTS, (steady / RATED_NM).tolist())),
            "below_rated": bool((steady < RATED_NM).all()),
            "advice_if_fail": "move the stance 10 mm towards straight legs (STANCE_CRANKS_DEG in scripts/build_servo_contract.py) and rerun",
        })
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        scan() if args.mode == "scan" else stance()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
