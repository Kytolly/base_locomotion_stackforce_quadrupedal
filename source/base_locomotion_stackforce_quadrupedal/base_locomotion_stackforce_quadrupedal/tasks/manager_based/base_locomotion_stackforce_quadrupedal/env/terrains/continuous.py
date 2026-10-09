"""Continuous no-reset terrain route used by composition-stage training."""

from __future__ import annotations

import numpy as np

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass


@configclass
class HfContinuousMixedTerrainCfg(HfTerrainBaseCfg):
    """One of five forward routes with flat recovery segments."""

    function = (
        "base_locomotion_stackforce_quadrupedal.tasks.manager_based."
        "base_locomotion_stackforce_quadrupedal.env.terrains.continuous:continuous_mixed_terrain"
    )
    rough_height_range: tuple[float, float] = (0.01, 0.05)
    wave_amplitude_range: tuple[float, float] = (0.015, 0.06)
    slope_height_range: tuple[float, float] = (0.03, 0.12)
    route_variant_count: int = 5


@height_field_to_mesh
def continuous_mixed_terrain(
    difficulty: float, cfg: HfContinuousMixedTerrainCfg
) -> np.ndarray:
    """Generate ordered segments along positive Y while keeping the center spawn flat."""
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    heights_m = np.zeros((width, length), dtype=np.float64)
    center = length // 2
    route_length = max(length - center - 1, 1)
    route_y = np.linspace(0.0, 1.0, route_length, endpoint=True)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    rough_height = np.interp(difficulty, (0.0, 1.0), cfg.rough_height_range)
    wave_amplitude = np.interp(difficulty, (0.0, 1.0), cfg.wave_amplitude_range)
    slope_height = np.interp(difficulty, (0.0, 1.0), cfg.slope_height_range)

    x_phase = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False)[:, None]
    y_phase = np.linspace(0.0, 4.0 * np.pi, route_length, endpoint=False)[None, :]
    deterministic_rough = 0.5 * rough_height * (
        np.sin(2.3 * x_phase + y_phase) + np.sin(0.7 * x_phase - 1.7 * y_phase)
    )
    route = np.zeros((width, route_length), dtype=np.float64)

    def section(start: float, end: float) -> np.ndarray:
        return (route_y >= start) & (route_y < end)

    def ramp(mask: np.ndarray, start: float, end: float) -> None:
        route[:, mask] = np.linspace(start, end, int(mask.sum()), endpoint=False)

    variant = int(np.random.randint(max(int(cfg.route_variant_count), 1)))
    if variant == 0:  # flat -> rough -> flat
        rough = section(0.20, 0.78)
        route[:, rough] = deterministic_rough[:, rough]
    elif variant == 1:  # flat -> slope -> inverted slope -> flat
        ramp(section(0.20, 0.42), 0.0, slope_height)
        ramp(section(0.42, 0.58), slope_height, 0.0)
        ramp(section(0.58, 0.76), 0.0, -slope_height)
        ramp(section(0.76, 0.88), -slope_height, 0.0)
    elif variant == 2:  # wave -> boxes -> flat
        wave = section(0.12, 0.48)
        boxes = section(0.48, 0.82)
        route[:, wave] = wave_amplitude * np.sin(y_phase[:, wave].reshape(1, -1))
        box_pattern = (
            (np.arange(width)[:, None] // 4 + np.arange(route_length)[None, :] // 4) % 3
        ) == 0
        route[:, boxes] = box_pattern[:, boxes] * slope_height
    elif variant == 3:  # stairs up -> platform -> stairs down
        up = section(0.15, 0.42)
        platform = section(0.42, 0.62)
        down = section(0.62, 0.88)
        step_count = max(2, int(3 + 4 * difficulty))
        up_values = np.floor(np.linspace(0, step_count, int(up.sum()), endpoint=False)) / step_count
        down_values = np.ceil(np.linspace(step_count, 0, int(down.sum()), endpoint=False)) / step_count
        route[:, up] = slope_height * up_values
        route[:, platform] = slope_height
        route[:, down] = slope_height * down_values
    else:  # rough -> inverted slope -> pit/stepping stones -> recovery flat
        rough = section(0.12, 0.38)
        route[:, rough] = deterministic_rough[:, rough]
        ramp(section(0.38, 0.52), 0.0, -slope_height)
        ramp(section(0.52, 0.64), -slope_height, 0.0)
        obstacle = section(0.64, 0.84)
        route[:, obstacle] = -slope_height
        stones = ((np.arange(width)[:, None] // 3 + np.arange(route_length)[None, :] // 3) % 2) == 0
        route[:, obstacle] = np.where(stones[:, obstacle], 0.0, route[:, obstacle])
    heights_m[:, center + 1 :] = route

    return np.rint(heights_m / cfg.vertical_scale).astype(np.int16)


__all__ = ["HfContinuousMixedTerrainCfg", "continuous_mixed_terrain"]
