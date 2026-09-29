"""Versioned plateau-track parameters."""

from __future__ import annotations

import math
import random


def plateau_track_parameters(
    seed: int = 8101, randomized: bool = False
) -> dict[str, float | int | bool | str]:
    """Return deterministic or seeded-randomized plateau geometry."""
    rng = random.Random(seed)

    def value(nominal: float, spread: float) -> float:
        return (
            round(rng.uniform(nominal - spread, nominal + spread), 4)
            if randomized
            else nominal
        )

    slope_deg = value(12.0, 3.0)
    ramp_length = value(1.5, 0.25)
    return {
        "kind": "plateau",
        "version": 1,
        "seed": seed,
        "randomized": randomized,
        "width_m": value(2.0, 0.25),
        "approach_m": value(0.8, 0.1),
        "ramp_length_m": ramp_length,
        "ramp_height_m": round(math.tan(math.radians(slope_deg)) * ramp_length, 5),
        "plateau_length_m": value(1.8, 0.3),
        "release_m": value(0.9, 0.1),
        "slope_deg": slope_deg,
    }
