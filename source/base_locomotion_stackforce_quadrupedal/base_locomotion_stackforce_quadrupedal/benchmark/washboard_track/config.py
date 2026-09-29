"""Versioned washboard-track parameters."""

from __future__ import annotations

import random


def washboard_track_parameters(
    seed: int = 8201, randomized: bool = False
) -> dict[str, float | int | bool | str]:
    """Return deterministic or seeded-randomized washboard geometry."""
    rng = random.Random(seed)

    def value(nominal: float, spread: float) -> float:
        return (
            round(rng.uniform(nominal - spread, nominal + spread), 4)
            if randomized
            else nominal
        )

    radius = value(0.06, 0.012)
    return {
        "kind": "washboard",
        "version": 1,
        "seed": seed,
        "randomized": randomized,
        "width_m": value(1.4, 0.15),
        "approach_m": value(0.7, 0.1),
        "ramp_length_m": value(0.55, 0.08),
        "ramp_height_m": 2.0 * radius,
        "ridge_radius_m": radius,
        "ridge_spacing_m": value(0.16, 0.02),
        "ridge_count": 14,
        "release_m": value(0.8, 0.1),
    }
