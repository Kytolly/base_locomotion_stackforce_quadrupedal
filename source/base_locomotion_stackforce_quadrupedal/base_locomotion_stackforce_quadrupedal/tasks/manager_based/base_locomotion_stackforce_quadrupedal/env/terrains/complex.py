"""Eight-family procedural terrain distribution for complex locomotion."""

from __future__ import annotations

import isaaclab.terrains as terrain
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


TERRAIN_FAMILIES = (
    "random_rough",
    "boxes",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "wave",
    "stepping_stones",
    "pit",
)


def get_complex_terrain_cfg(*, seed: int = 42, num_rows: int = 8, num_cols: int = 8) -> TerrainGeneratorCfg:
    """Return an equal-proportion generator for the eight reference terrain families."""
    sub_terrains = {
        "random_rough": terrain.HfRandomUniformTerrainCfg(
            proportion=0.125, noise_range=(0.02, 0.08), noise_step=0.01, border_width=0.5
        ),
        "boxes": terrain.MeshRandomGridTerrainCfg(
            proportion=0.125,
            grid_width=0.45,
            grid_height_range=(0.03, 0.12),
            platform_width=2.0,
            holes=False,
        ),
        "pyramid_stairs": terrain.MeshPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.04, 0.14),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.04, 0.14),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain.HfPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.05, 0.30),
            platform_width=2.0,
            border_width=0.5,
            inverted=False,
        ),
        "wave": terrain.HfWaveTerrainCfg(
            proportion=0.125, amplitude_range=(0.03, 0.10), num_waves=4, border_width=0.5
        ),
        "stepping_stones": terrain.HfSteppingStonesTerrainCfg(
            proportion=0.125,
            stone_height_max=0.08,
            stone_width_range=(0.4, 0.8),
            stone_distance_range=(0.05, 0.15),
            holes_depth=-0.1,
            platform_width=2.0,
            border_width=0.5,
        ),
        "pit": terrain.MeshPitTerrainCfg(
            proportion=0.125,
            pit_depth_range=(0.05, 0.12),
            platform_width=2.0,
            double_pit=False,
        ),
    }
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=10.0,
        border_height=0.5,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        seed=seed,
        curriculum=True,
        difficulty_range=(0.0, 0.6),
        sub_terrains=sub_terrains,
    )


__all__ = ["TERRAIN_FAMILIES", "get_complex_terrain_cfg"]
