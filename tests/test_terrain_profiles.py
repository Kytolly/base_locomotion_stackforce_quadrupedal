from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/base_locomotion_stackforce_quadrupedal"))

from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import (  # noqa: E402,E501
    HfContinuousMixedTerrainCfg,
    TERRAIN_PROFILES,
    TRAINING_TERRAIN_FAMILIES,
    get_complex_terrain_cfg,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains.continuous import (  # noqa: E402,E501
    continuous_mixed_terrain,
)


@pytest.mark.parametrize("profile", sorted(TERRAIN_PROFILES))
def test_terrain_profiles_are_normalized_and_keep_stable_family_order(profile):
    cfg = get_complex_terrain_cfg(profile=profile)
    assert tuple(cfg.sub_terrains) == TRAINING_TERRAIN_FAMILIES
    proportions = [subterrain.proportion for subterrain in cfg.sub_terrains.values()]
    assert sum(proportions) == pytest.approx(1.0)


def test_expansion_profile_matches_documented_union_mix():
    cfg = get_complex_terrain_cfg(profile="union_expansion")
    proportions = {name: value.proportion for name, value in cfg.sub_terrains.items()}
    assert proportions["flat"] == pytest.approx(0.20)
    assert proportions["hf_pyramid_slope_inv"] == pytest.approx(0.08)
    original_eight = set(TRAINING_TERRAIN_FAMILIES[:9]) - {"hf_pyramid_slope_inv"}
    assert sum(proportions[name] for name in original_eight) == pytest.approx(0.72)


def test_continuous_mixed_height_field_matches_isaac_border_contract():
    cfg = HfContinuousMixedTerrainCfg(
        proportion=1.0,
        size=(8.0, 8.0),
        border_width=0.5,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
    )
    meshes, origin = cfg.function(0.5, cfg)
    assert len(meshes) == 1
    assert tuple(origin.shape) == (3,)
    assert meshes[0].vertices.shape[0] > 0


@pytest.mark.parametrize("variant", range(5))
def test_all_documented_continuous_route_variants_are_nonflat(monkeypatch, variant):
    cfg = HfContinuousMixedTerrainCfg(
        proportion=1.0,
        size=(7.0, 7.0),
        border_width=0.5,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
    )
    monkeypatch.setattr("numpy.random.randint", lambda *_args, **_kwargs: variant)
    height_field = continuous_mixed_terrain.__wrapped__(0.5, cfg)
    assert height_field.shape == (70, 70)
    assert height_field.max() != height_field.min()
