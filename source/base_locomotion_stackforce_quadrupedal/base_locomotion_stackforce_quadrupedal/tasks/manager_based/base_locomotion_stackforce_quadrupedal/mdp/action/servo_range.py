"""E0-v2 leg action contract derived from the DS041MG servo travel.

The numbers live in ``e0_v2_servo_contract.json`` next to this module and are
produced by ``scripts/build_servo_contract.py``:

* before G3 they are the section 3A table of
  docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md (``status: provisional_section_3A``);
* after G3 the builder rewrites them from the measured
  ``output/fix_plan/G3/final_ranges.json`` and ``stance.json``
  (``status: g3_measured``).

This module is pure Python so the contract can be tested without Isaac Sim.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(__file__).with_name("e0_v2_servo_contract.json")
LEGACY_ACTION_CONTRACT = "legacy_symmetric"
SERVO_ACTION_CONTRACT = "servo_range_v2"
ACTION_CONTRACTS = (LEGACY_ACTION_CONTRACT, SERVO_ACTION_CONTRACT)

# E0-v1 values, kept for the P0 control group and for evaluating old checkpoints.
LEGACY_BODY_HEIGHT_RANGE_M = (0.095, 0.115)
LEGACY_NOMINAL_BODY_HEIGHT_M = 0.105
LEGACY_INIT_ROOT_HEIGHT_M = 0.11
LEGACY_LOW_BODY_WARNING_M = 0.08
LEGACY_MINIMUM_BODY_HEIGHT_M = 0.06
LEGACY_STEP_HEIGHT_RANGE_M = (0.04, 0.14)
LEGACY_BOX_HEIGHT_RANGE_M = (0.03, 0.12)
LEGACY_CONTINUOUS_STEP_RANGE_M = (0.03, 0.12)


@lru_cache(maxsize=4)
def load_servo_contract(path: str | None = None) -> dict[str, Any]:
    """Load and validate the E0-v2 servo contract."""
    contract = json.loads(Path(path or CONTRACT_PATH).read_text(encoding="utf-8"))
    for name, joint in contract["joints"].items():
        lower, default, upper = float(joint["lower"]), float(joint["default"]), float(joint["upper"])
        if not lower < default < upper:
            raise ValueError(f"{name}: default {default} is outside its range [{lower}, {upper}].")
    heights = contract["body_height"]
    low, high = heights["command_range_m"]
    if not heights["minimum_m"] < heights["warning_m"] < low <= heights["nominal_m"] <= high:
        raise ValueError("body_height values must satisfy minimum < warning < low <= nominal <= high.")
    return contract


def leg_joint_ranges(joint_names: tuple[str, ...] | list[str], contract: dict[str, Any] | None = None):
    """Return ``(lower, default, upper)`` lists in ``joint_names`` order."""
    contract = contract or load_servo_contract()
    joints = contract["joints"]
    missing = [name for name in joint_names if name not in joints]
    if missing:
        raise KeyError(f"Servo contract has no range for {missing}.")
    lower = [float(joints[name]["lower"]) for name in joint_names]
    default = [float(joints[name]["default"]) for name in joint_names]
    upper = [float(joints[name]["upper"]) for name in joint_names]
    return lower, default, upper


def leg_init_std(
    joint_names: tuple[str, ...] | list[str],
    physical_noise_rad: float | None = None,
    contract: dict[str, Any] | None = None,
) -> list[float]:
    """Latent std per leg dimension bounding the initial joint noise to ``physical_noise_rad``.

    The raw-to-target map has a different gain on each side of the stance, so the
    larger half-range is used: neither side exceeds the bound.  (Scaling by the
    smaller half-range left 0.22-0.42 rad on the larger side, more than the
    0.175 rad a DS041MG travels in one 20 ms control step.)
    """
    contract = contract or load_servo_contract()
    noise = float(physical_noise_rad or contract["exploration"]["physical_noise_rad"])
    lower, default, upper = leg_joint_ranges(joint_names, contract)
    return [round(noise / max(d - lo, hi - d), 4) for lo, d, hi in zip(lower, default, upper)]


def raw_to_joint_target(raw, lower, default, upper):
    """Piecewise-linear map: raw 0 -> default, +1 -> upper, -1 -> lower.

    Works on floats or tensors (anything supporting ``where``-free arithmetic via
    ``clamp``); the action term uses the tensor form.
    """
    positive = raw.clamp(min=0.0) if hasattr(raw, "clamp") else max(raw, 0.0)
    negative = raw.clamp(max=0.0) if hasattr(raw, "clamp") else min(raw, 0.0)
    return default + positive * (upper - default) + negative * (default - lower)


SERVO_MODEL_SPEED_RAD_S = 8.73  # DS041MG no-load speed (0.12 s / 60 deg)


def servo_model_step(estimate, target, max_step_rad):
    """One control step of the deployable servo-position estimate.

    The DS041MG has no position feedback, so the policy observes this model on
    the robot and in simulation alike: the estimate moves toward the (possibly
    delayed) commanded target by at most ``max_step_rad`` (speed x control
    period).  Velocity is the step difference divided by the control period.
    docs/REAL_ROBOT_IO_SPEC.md gives the firmware form.
    """
    return estimate + (target - estimate).clamp(min=-max_step_rad, max=max_step_rad)


def joint_target_to_raw(target, lower, default, upper):
    """Inverse of ``raw_to_joint_target`` for targets inside ``[lower, upper]``."""
    offset = target - default
    return (offset.clamp(min=0.0) / (upper - default)) + (offset.clamp(max=0.0) / (default - lower))


def project_leg_target_grid(target, spec):
    """Project infeasible outer/inner pairs to a precomputed feasible grid point."""
    outer_lo, outer_hi, outer_count = spec["outer_grid"]
    inner_lo, inner_hi, inner_count = spec["inner_grid"]
    outer_index = ((target[..., 0] - outer_lo) / (outer_hi - outer_lo) * (outer_count - 1)).round()
    inner_index = ((target[..., 1] - inner_lo) / (inner_hi - inner_lo) * (inner_count - 1)).round()
    outer_index = outer_index.clamp(0, outer_count - 1).long()
    inner_index = inner_index.clamp(0, inner_count - 1).long()
    flat = outer_index * inner_count + inner_index
    feasible = spec["feasible_tensor"][flat]
    nearest = spec["nearest_tensor"][flat]
    projected_outer = outer_lo + (nearest // inner_count) * (outer_hi - outer_lo) / (outer_count - 1)
    projected_inner = inner_lo + (nearest % inner_count) * (inner_hi - inner_lo) / (inner_count - 1)
    projected = target.new_empty(target.shape)
    projected[..., 0] = projected_outer
    projected[..., 1] = projected_inner
    return target.where(feasible.unsqueeze(-1), projected)


__all__ = [
    "ACTION_CONTRACTS",
    "CONTRACT_PATH",
    "LEGACY_ACTION_CONTRACT",
    "LEGACY_BODY_HEIGHT_RANGE_M",
    "LEGACY_BOX_HEIGHT_RANGE_M",
    "LEGACY_CONTINUOUS_STEP_RANGE_M",
    "LEGACY_INIT_ROOT_HEIGHT_M",
    "LEGACY_LOW_BODY_WARNING_M",
    "LEGACY_MINIMUM_BODY_HEIGHT_M",
    "LEGACY_NOMINAL_BODY_HEIGHT_M",
    "LEGACY_STEP_HEIGHT_RANGE_M",
    "SERVO_ACTION_CONTRACT",
    "SERVO_MODEL_SPEED_RAD_S",
    "joint_target_to_raw",
    "servo_model_step",
    "leg_init_std",
    "leg_joint_ranges",
    "load_servo_contract",
    "raw_to_joint_target",
    "project_leg_target_grid",
]
