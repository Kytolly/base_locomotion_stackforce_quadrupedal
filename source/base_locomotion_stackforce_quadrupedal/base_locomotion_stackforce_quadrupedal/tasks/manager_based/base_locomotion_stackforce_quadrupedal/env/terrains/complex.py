"""Versioned terrain-union distributions for complex locomotion."""

from __future__ import annotations

import isaaclab.terrains as terrain
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .continuous import HfContinuousMixedTerrainCfg


DEEP_ROBOTICS_TERRAIN_SOURCE = (
    "https://github.com/DeepRoboticsLab/rl_training.git",
    "3d05d3b97f7a2390f003547f23a02b3b42eea195",
)

TERRAIN_FAMILIES = (
    "random_rough",
    "boxes",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "wave",
    "stepping_stones",
    "pit",
)

COMPOSITION_FAMILY = "continuous_mixed"
TRAINING_TERRAIN_FAMILIES = TERRAIN_FAMILIES + ("flat", COMPOSITION_FAMILY)
TERRAIN_PROFILES = {
    "legacy_equal": {
        **{name: 0.125 for name in TERRAIN_FAMILIES if name != "hf_pyramid_slope_inv"},
        "hf_pyramid_slope_inv": 0.0,
        "flat": 0.0,
        COMPOSITION_FAMILY: 0.0,
    },
    "union_foundation": {
        "flat": 0.60,
        "random_rough": 0.12,
        "hf_pyramid_slope": 0.10,
        "wave": 0.10,
        "hf_pyramid_slope_inv": 0.08,
    },
    "union_expansion": {
        "flat": 0.20,
        **{name: 0.09 for name in TERRAIN_FAMILIES if name != "hf_pyramid_slope_inv"},
        "hf_pyramid_slope_inv": 0.08,
    },
    "union_composition": {
        "flat": 0.12,
        **{name: 0.06 for name in TERRAIN_FAMILIES if name != "hf_pyramid_slope_inv"},
        "hf_pyramid_slope_inv": 0.08,
        COMPOSITION_FAMILY: 0.32,
    },
    "union_consolidation": {
        "flat": 0.15,
        **{name: 0.06 for name in TERRAIN_FAMILIES if name != "hf_pyramid_slope_inv"},
        "hf_pyramid_slope_inv": 0.07,
        COMPOSITION_FAMILY: 0.30,
    },
    "source_alignment": {
        "pyramid_stairs": 0.20,
        "pyramid_stairs_inv": 0.20,
        "boxes": 0.20,
        "random_rough": 0.20,
        "hf_pyramid_slope": 0.10,
        "hf_pyramid_slope_inv": 0.10,
    },
}


def get_complex_terrain_cfg(
    *,
    seed: int = 42,
    num_rows: int = 8,
    num_cols: int = 12,
    profile: str = "union_expansion",
) -> TerrainGeneratorCfg:
    """Return the requested versioned terrain-union generator."""
    if profile not in TERRAIN_PROFILES:
        raise ValueError(f"Unknown terrain profile {profile!r}; expected one of {sorted(TERRAIN_PROFILES)}.")
    proportions = TERRAIN_PROFILES[profile]
    sub_terrains = {
        "random_rough": terrain.HfRandomUniformTerrainCfg(
            proportion=proportions.get("random_rough", 0.0),
            noise_range=(0.02, 0.08), noise_step=0.01, border_width=0.5
        ),
        "boxes": terrain.MeshRandomGridTerrainCfg(
            proportion=proportions.get("boxes", 0.0),
            grid_width=0.45,
            grid_height_range=(0.03, 0.12),
            platform_width=2.0,
            holes=False,
        ),
        "pyramid_stairs": terrain.MeshPyramidStairsTerrainCfg(
            proportion=proportions.get("pyramid_stairs", 0.0),
            step_height_range=(0.04, 0.14),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain.MeshInvertedPyramidStairsTerrainCfg(
            proportion=proportions.get("pyramid_stairs_inv", 0.0),
            step_height_range=(0.04, 0.14),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain.HfPyramidSlopedTerrainCfg(
            proportion=proportions.get("hf_pyramid_slope", 0.0),
            slope_range=(0.05, 0.30),
            platform_width=2.0,
            border_width=0.5,
            inverted=False,
        ),
        "hf_pyramid_slope_inv": terrain.HfInvertedPyramidSlopedTerrainCfg(
            proportion=proportions.get("hf_pyramid_slope_inv", 0.0),
            slope_range=(0.0, 0.30),
            platform_width=2.0,
            border_width=0.5,
        ),
        "wave": terrain.HfWaveTerrainCfg(
            proportion=proportions.get("wave", 0.0),
            amplitude_range=(0.03, 0.10), num_waves=4, border_width=0.5
        ),
        "stepping_stones": terrain.HfSteppingStonesTerrainCfg(
            proportion=proportions.get("stepping_stones", 0.0),
            stone_height_max=0.08,
            stone_width_range=(0.4, 0.8),
            stone_distance_range=(0.05, 0.15),
            holes_depth=-0.1,
            platform_width=2.0,
            border_width=0.5,
        ),
        "pit": terrain.MeshPitTerrainCfg(
            proportion=proportions.get("pit", 0.0),
            pit_depth_range=(0.05, 0.12),
            platform_width=2.0,
            double_pit=False,
        ),
        "flat": terrain.MeshPlaneTerrainCfg(proportion=proportions.get("flat", 0.0)),
        COMPOSITION_FAMILY: HfContinuousMixedTerrainCfg(
            proportion=proportions.get(COMPOSITION_FAMILY, 0.0),
            border_width=0.5,
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


__all__ = [
    "COMPOSITION_FAMILY",
    "DEEP_ROBOTICS_TERRAIN_SOURCE",
    "TERRAIN_FAMILIES",
    "TERRAIN_PROFILES",
    "TRAINING_TERRAIN_FAMILIES",
    "get_complex_terrain_cfg",
]
