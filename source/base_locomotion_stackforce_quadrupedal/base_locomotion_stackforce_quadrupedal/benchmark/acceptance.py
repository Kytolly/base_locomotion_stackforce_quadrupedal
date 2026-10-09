"""Shared fixed performance gates for validation and benchmark reports."""

import math


MAXIMUMS = {
    "locomotion/forward_velocity_rmse_mps": 0.15,
    "locomotion/lateral_velocity_rmse_mps": 0.12,
    "locomotion/yaw_rate_rmse_radps": 0.10,
    "locomotion/body_height_rmse_m": 0.025,
    "safety/unsafe_termination": 0.05,
    "safety/base_collision_rate": 0.01,
    "support/invalid_rate": 0.01,
    "actuation/action_saturation_rate": 0.05,
    "safety/base_tilt_max_p95_rad": 1.05,
}
MINIMUMS = {"support/wheel_contact_fraction": 0.50, "runtime/all_finite": 1.0}


def passes_thresholds(metrics):
    timeout = metrics.get("safety/timeout", metrics.get("safety/timeout_survival_rate", 0.0))
    return timeout >= 0.95 and all(math.isfinite(metrics.get(k, float("nan"))) and metrics[k] <= v for k, v in MAXIMUMS.items()) and all(
        math.isfinite(metrics.get(k, float("nan"))) and metrics[k] >= v for k, v in MINIMUMS.items()
    )
