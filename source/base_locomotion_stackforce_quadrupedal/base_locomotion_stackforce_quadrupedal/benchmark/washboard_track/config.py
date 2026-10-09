"""Versioned washboard-track parameters."""

from __future__ import annotations

import math
import random
from typing import Any


def washboard_track_parameters(
    seed: int = 8201, randomized: bool = False
) -> dict[str, Any]:
    """Port the demo_mode_switching bottle-bed sampling contract without importing it."""
    rng = random.Random(seed)

    def value(nominal: float, spread: float, digits: int = 4) -> float:
        return (
            round(rng.uniform(nominal - spread, nominal + spread), digits)
            if randomized
            else nominal
        )

    radius = value(0.075, 0.015, 5)
    length = value(1.0, 0.04, 5)
    gap = round(rng.uniform(0.0, 0.02), 5) if randomized else 0.0
    spacing = round(2.0 * radius + gap, 5)
    count = rng.randint(6, 8) if randomized else 7
    approach = value(0.8, 0.1)
    slope = value(6.0, 2.0)
    ramp = round(2.0 * radius / math.tan(math.radians(slope)), 4)
    position_jitter = round(rng.uniform(0.0, 0.025), 5) if randomized else 0.0
    height_jitter = round(rng.uniform(0.0, 0.015), 5) if randomized else 0.0
    orientation_jitter = round(rng.uniform(0.0, 2.0), 4) if randomized else 0.0
    static_friction = value(0.72, 0.16)
    dynamic_friction = min(static_friction, value(0.62, 0.14))
    leading_row = rng.choice((0, 1)) if randomized else 0
    staggers = [radius, radius]
    staggers[leading_row] = 0.0
    caps = [["full_cylinder", "full_cylinder"] for _ in range(2)]
    caps[leading_row][0] = "box"
    caps[1 - leading_row][1] = "box"
    xy_offsets, z_offsets, rotations = [], [], []
    for _ in range(2):
        row_xy, row_z, row_rot = [], [], []
        for _ in range(count):
            if randomized:
                row_xy.append([round(rng.uniform(-position_jitter, position_jitter), 5) for _ in range(2)])
                row_z.append(round(rng.uniform(-height_jitter, height_jitter), 5))
                row_rot.append(round(rng.uniform(-orientation_jitter, orientation_jitter), 5))
            else:
                row_xy.append([0.0, 0.0])
                row_z.append(0.0)
                row_rot.append(0.0)
        xy_offsets.append(row_xy)
        z_offsets.append(row_z)
        rotations.append(row_rot)
    rotations[leading_row][0] = 0.0
    rotations[1 - leading_row][-1] = 0.0
    return {
        "kind": "washboard",
        "version": 2,
        "seed": seed,
        "randomized": randomized,
        "width_m": 2.0,
        "base_surface_height_m": 0.005,
        "approach_m": approach,
        "slope_deg": slope,
        "ramp_length_m": ramp,
        "ramp_height_m": round(2.0 * radius, 5),
        "cylinder_radius_m": radius,
        "cylinder_length_m": length,
        "cylinder_spacing_m": spacing,
        "cylinder_gap_m": gap,
        "cylinder_row_count": 2,
        "cylinders_per_row": count,
        "lateral_row_centers_m": [-0.5, 0.5],
        "row_stagger_offsets_m": staggers,
        "boundary_cap_types": caps,
        "position_jitter_m": position_jitter,
        "height_jitter_m": height_jitter,
        "orientation_jitter_deg": orientation_jitter,
        "cylinder_position_offsets_xy_m": xy_offsets,
        "cylinder_height_offsets_m": z_offsets,
        "cylinder_orientation_offsets_deg": rotations,
        "arrangement": "left_row_leads" if leading_row == 0 else "right_row_leads",
        "cylinder_axis": "+X",
        "static_cylinders": True,
        "friction": {"static_friction": static_friction, "dynamic_friction": dynamic_friction},
        "washboard_length_m": round((count - 1) * spacing + max(staggers), 4),
        "release_m": value(0.8, 0.1),
    }
