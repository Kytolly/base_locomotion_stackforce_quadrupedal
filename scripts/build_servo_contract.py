#!/usr/bin/env python3
"""Write the E0-v2 servo action contract (mdp/action/e0_v2_servo_contract.json).

Without arguments it reproduces section 3A of
docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md (status ``provisional_section_3A``).
After G3, pass the measured files to replace the provisional values::

    python scripts/build_servo_contract.py \
        --final-ranges output/fix_plan/G3/final_ranges.json \
        --stance output/fix_plan/G3/stance.json \
        --lift output/fix_plan/G3/lift.json

Passive knee initial angles are always recomputed from the URDF five-bar
kinematics so the default stance is closure-consistent.  Pure Python + NumPy;
it does not start Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/tasks/manager_based/base_locomotion_stackforce_quadrupedal"
OUTPUT = TASK / "mdp/action/e0_v2_servo_contract.json"
WHEEL_RADIUS_M = 0.033  # mdp/support.py

# Section 3A (v3.4): channel, joint and firmware trim.  The 300-degree servo
# matches the firmware's 0..300 degree command scale, so the 0.9 travel window
# is command 15..285 degrees.  Joint coordinates are derived from the URDF
# instead of being maintained as a second hand-written table.
CHANNELS = (
    ("ch1", "FR_Outer_Hip_Joint", -4),
    ("ch2", "FR_Inner_Hip_Joint", -5),
    ("ch3", "FL_Inner_Hip_Joint", 5),
    ("ch4", "FL_Outer_Hip_Joint", -7),
    ("ch5", "RL_Outer_Hip_Joint", -3),
    ("ch6", "RL_Inner_Hip_Joint", -5),
    ("ch7", "RR_Inner_Hip_Joint", -8),
    ("ch8", "RR_Outer_Hip_Joint", 8),
)
SERVO_TRAVEL_DEG = 300.0
TRAVEL_FRACTION = 0.9
COMMAND_LOWER_DEG = 0.5 * SERVO_TRAVEL_DEG * (1.0 - TRAVEL_FRACTION)
COMMAND_UPPER_DEG = SERVO_TRAVEL_DEG - COMMAND_LOWER_DEG
FIRMWARE_ZERO_DEG = 180.0
STANCE_CRANKS_DEG = (128.4, 66.6)
LEGACY_NOMINAL_HEIGHT_M = 0.105
LEGACY_WARNING_M = 0.08
LEGACY_MINIMUM_M = 0.06
HEIGHT_COMMAND_HALF_RANGE_M = 0.03
HEIGHT_REACH_MARGIN_M = 0.010
INIT_DROP_M = 0.005  # E0-v1 spawned 5 mm above its nominal height as well
PHYSICAL_NOISE_RAD = 0.15  # E0-v1: latent 0.3 x scale 0.5
STEP_START_M = 0.02
STEP_CAP_FRACTION = 0.8
LEGACY_STEP_RANGES_M = {"pyramid_stairs": (0.04, 0.14), "boxes": (0.03, 0.12), "continuous_steps": (0.03, 0.12)}


def _kinematics():
    path = TASK / "env/robots/kinematics.py"
    spec = importlib.util.spec_from_file_location("_e0_leg_kinematics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def build(final_ranges: dict | None, stance: dict | None, lift: dict | None) -> dict:
    kin_module = _kinematics()
    kin = kin_module.LegKinematics()
    joints = {}
    for channel, name, trim in CHANNELS:
        leg, branch, _ = name.split("_", 2)
        zero_command = FIRMWARE_ZERO_DEG + trim
        mounting_sign = 1.0 if leg in ("FL", "RR") else -1.0

        def joint_angle(command_deg: float) -> float:
            crank = 90.0 + mounting_sign * (command_deg - zero_command)
            outer, inner = kin.hips_for_crank_angles(
                leg,
                crank if branch == "Outer" else 90.0,
                crank if branch == "Inner" else 90.0,
            )
            return outer if branch == "Outer" else inner

        default_outer, default_inner = kin.hips_for_crank_angles(leg, *STANCE_CRANKS_DEG)
        default = default_outer if branch == "Outer" else default_inner
        endpoints = (joint_angle(COMMAND_LOWER_DEG), joint_angle(COMMAND_UPPER_DEG))
        joints[name] = {
            "channel": channel,
            "default": round(default, 3),
            "lower": round(min(endpoints), 3),
            "upper": round(max(endpoints), 3),
        }
    sources = ["docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md section 3A (v3.4)", "wiki E-CAL-002"]
    if final_ranges is not None:
        if final_ranges.get("status") != "PASS":
            raise ValueError("final_ranges.json is not a PASS G3 result")
        for name, values in final_ranges["joints"].items():
            joints[name]["lower"], joints[name]["upper"] = float(values["lower"]), float(values["upper"])
        sources.append("output/fix_plan/G3/final_ranges.json")
    if stance is not None:
        for name, value in stance["joints"].items():
            joints[name]["default"] = float(value)
        sources.append("output/fix_plan/G3/stance.json")

    knees: dict[str, float] = {}
    residuals_mm: dict[str, float] = {}
    wheel_z = []
    straight_z = []
    for leg in kin_module.LEGS:
        q_outer = joints[f"{leg}_Outer_Hip_Joint"]["default"]
        q_inner = joints[f"{leg}_Inner_Hip_Joint"]["default"]
        k_outer, k_inner, residual = kin.solve_knees(leg, q_outer, q_inner)
        knees[f"{leg}_Outer_Knee_Joint"] = round(k_outer, 5)
        knees[f"{leg}_Inner_Knee_Joint"] = round(k_inner, 5)
        residuals_mm[leg] = residual * 1000.0
        wheel_z.append(kin.wheel_centre_z_m(leg, q_outer, q_inner))
        straight_z.append(kin.wheel_centre_z_m(leg, *kin.hips_for_crank_angles(leg, 90.0, 90.0)))
    geometric_height = -sum(wheel_z) / len(wheel_z) + WHEEL_RADIUS_M
    straight_height = -max(straight_z) + WHEEL_RADIUS_M

    if stance is not None and stance.get("standing_body_height_m") is not None:
        nominal = float(stance["standing_body_height_m"])
        height_source = "G3 measured standing height"
    else:
        # E0-v1 used 0.105 m against a 0.106 m geometric height: keep the same 1 mm sag.
        nominal = round(geometric_height - 0.001, 3)
        height_source = "URDF five-bar geometry (provisional until G3 step 4)"
    # Round down so serialization cannot shrink the required straight-leg margin.
    upper_bound = min(nominal + HEIGHT_COMMAND_HALF_RANGE_M, straight_height - HEIGHT_REACH_MARGIN_M)
    upper = math.floor(upper_bound * 1000.0) / 1000.0
    scale = nominal / LEGACY_NOMINAL_HEIGHT_M

    min_lift = None if lift is None else float(lift["min_lift_m"])
    step_caps = {
        name: [
            STEP_START_M,
            math.floor(min(high, STEP_CAP_FRACTION * min_lift) * 1000.0) / 1000.0 if min_lift else high,
        ]
        for name, (_, high) in LEGACY_STEP_RANGES_M.items()
    }

    return {
        "schema_version": 1,
        "protocol_version": "E0-v2-46d",
        "status": "g3_measured" if final_ranges is not None and stance is not None else "provisional_section_3A",
        "sources": sources,
        "servo": {
            "model": "GDW DS041MG (300 deg configured)",
            "travel_fraction": TRAVEL_FRACTION,
            "pulse_to_angle_deg_per_us": 0.15,
            "physical_endpoints_measured": False,
            "stall_torque_nm": 0.392,
            "rated_torque_nm": 0.127,
        },
        "stance_crank_angles_deg": list(STANCE_CRANKS_DEG),
        "joints": joints,
        "projection": {} if final_ranges is None else final_ranges.get("projection", {}),
        "passive_knees": knees,
        "closure_residual_mm": {leg: round(value, 6) for leg, value in residuals_mm.items()},
        "body_height": {
            "geometric_standing_m": round(geometric_height, 4),
            "straight_leg_m": round(straight_height, 4),
            "nominal_m": nominal,
            "command_range_m": [round(nominal - HEIGHT_COMMAND_HALF_RANGE_M, 3), upper],
            "init_root_m": round(nominal + INIT_DROP_M, 3),
            "warning_m": round(LEGACY_WARNING_M * scale, 3),
            "minimum_m": round(LEGACY_MINIMUM_M * scale, 3),
            "source": height_source,
        },
        "exploration": {"physical_noise_rad": PHYSICAL_NOISE_RAD, "wheel_init_std": 0.3},
        "obstacles": {
            "min_measured_lift_m": min_lift,
            "cap_fraction_of_lift": STEP_CAP_FRACTION,
            "height_ranges_m": step_caps,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-ranges", type=Path)
    parser.add_argument("--stance", type=Path)
    parser.add_argument("--lift", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the file on disk differs from the rebuild.")
    args = parser.parse_args()
    if args.check and args.final_ranges is None and args.output.is_file():
        current = json.loads(args.output.read_text(encoding="utf-8"))
        if current.get("status") == "g3_measured":
            args.final_ranges = ROOT / "output/fix_plan/G3/final_ranges.json"
            args.stance = ROOT / "output/fix_plan/G3/stance.json"
            args.lift = ROOT / "output/fix_plan/G3/lift.json"
    contract = build(_read(args.final_ranges), _read(args.stance), _read(args.lift))
    text = json.dumps(contract, indent=2) + "\n"
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != text:
            print(f"{args.output} is out of date; rerun scripts/build_servo_contract.py", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date")
        return 0
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({contract['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
