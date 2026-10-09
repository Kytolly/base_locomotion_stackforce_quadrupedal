"""Aggregation checks for the fixed locomotion validation suite."""

import torch

from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.validation.suite import (
    DEFAULT_VALIDATION_SCENARIOS,
    build_validation_report,
    validation_terrain_layout,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import (
    TRAINING_TERRAIN_FAMILIES,
)


def _scenario_metrics() -> dict[str, torch.Tensor]:
    terrain_type = torch.arange(11, dtype=torch.float32)
    size = len(terrain_type)
    return {
        "terrain/type_id": terrain_type,
        "locomotion/forward_velocity_rmse_mps": torch.arange(1, size + 1, dtype=torch.float32),
        "locomotion/lateral_velocity_rmse_mps": torch.full((size,), 0.05),
        "locomotion/yaw_rate_rmse_radps": torch.full((size,), 0.2),
        "locomotion/body_height_rmse_m": torch.full((size,), 0.01),
        "safety/unsafe_termination": torch.zeros(size),
        "safety/timeout": torch.ones(size),
        "safety/base_collision_rate": torch.zeros(size),
        "support/invalid_rate": torch.zeros(size),
        "support/wheel_contact_fraction": torch.full((size,), 0.75),
        "safety/base_tilt_max_rad": torch.linspace(0.1, 0.9, size),
        "actuation/action_saturation_rate": torch.zeros(size),
        "actuation/leg_mechanical_energy_j": torch.ones(size),
        "actuation/wheel_mechanical_energy_j": torch.full((size,), 2.0),
        "runtime/episode_complete": torch.ones(size),
        "runtime/all_finite": torch.ones(size),
        "terrain/continuous_route_complete": torch.ones(size),
        "terrain/continuous_route_ordered_checkpoints": torch.full((size,), 3.0),
        "terrain/continuous_route_corridor_valid": torch.ones(size),
    }


def test_complete_suite_is_eligible_and_reports_macro_worst() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }

    report = build_validation_report(metrics, {"ppo_updates": 0})

    assert report["selection_evidence"]["eligible_for_checkpoint_comparison"]
    assert not report["selection_evidence"]["performance_pass"]
    forward = report["scenarios"]["forward"]["terrain_aggregate"]
    assert forward["macro/locomotion/forward_velocity_rmse_mps"] == 6.0
    assert forward["worst/locomotion/forward_velocity_rmse_mps"] == 11.0
    assert forward["worst/safety/timeout_survival_rate"] == 1.0


def test_missing_terrain_family_blocks_checkpoint_comparison() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }
    for values in metrics.values():
        mask = values["terrain/type_id"] != 10
        for name in values:
            values[name] = values[name][mask]

    report = build_validation_report(metrics, {"ppo_updates": 0})

    assert not report["selection_evidence"]["eligible_for_checkpoint_comparison"]
    assert report["scenarios"]["forward"]["terrain_coverage"]["missing"] == ["continuous_mixed"]


def test_source_alignment_requires_its_six_real_families() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }
    for values in metrics.values():
        mask = values["terrain/type_id"] < 6
        for name in values:
            values[name] = values[name][mask]
    report = build_validation_report(
        metrics, {"ppo_updates": 0, "terrain_profile": "source_alignment"}
    )
    assert report["selection_evidence"]["eligible_for_checkpoint_comparison"]
    assert report["scenarios"]["forward"]["terrain_coverage"]["missing"] == []


def test_validation_profiles_generate_distinct_truthful_layouts() -> None:
    source_cfg, source_ids = validation_terrain_layout("source_alignment", 1001)
    union_cfg, union_ids = validation_terrain_layout("union_consolidation", 1001)

    assert source_cfg.num_cols == union_cfg.num_cols == 15
    assert source_ids != union_ids
    assert {TRAINING_TERRAIN_FAMILIES[index] for index in source_ids} == {
        "random_rough", "boxes", "pyramid_stairs", "pyramid_stairs_inv",
        "hf_pyramid_slope", "hf_pyramid_slope_inv",
    }
    assert {TRAINING_TERRAIN_FAMILIES[index] for index in union_ids} == set(
        TRAINING_TERRAIN_FAMILIES
    )
    assert source_cfg.sub_terrains["flat"].proportion == 0.0
    assert union_cfg.sub_terrains["flat"].proportion == 0.15
    assert union_cfg.sub_terrains["continuous_mixed"].proportion == 0.30


def test_incomplete_episode_blocks_checkpoint_comparison() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }
    metrics["forward"]["runtime/episode_complete"][0] = 0.0

    report = build_validation_report(metrics, {"ppo_updates": 0})

    evidence = report["selection_evidence"]
    assert not evidence["complete_episodes"]
    assert not evidence["eligible_for_checkpoint_comparison"]


def test_fixed_thresholds_can_only_pass_a_qualified_run() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }
    for values in metrics.values():
        values["locomotion/forward_velocity_rmse_mps"] = torch.full((11,), 0.05)
        values["locomotion/yaw_rate_rmse_radps"] = torch.full((11,), 0.05)

    report = build_validation_report(metrics, {
        "ppo_updates": 0, "policy": "checkpoint", "curriculum_frozen": True,
        "terrain_level": -1, "num_envs": 64, "episode_steps": 1500, "policy_dt_s": 0.02,
    })

    assert report["selection_evidence"]["performance_pass"]


def test_good_overall_average_cannot_hide_a_failing_family() -> None:
    metrics = {scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS}
    for values in metrics.values():
        values["locomotion/forward_velocity_rmse_mps"] = torch.full((11,), 0.05)
        values["locomotion/yaw_rate_rmse_radps"] = torch.full((11,), 0.05)
    metrics["forward"]["actuation/action_saturation_rate"][0] = 0.30
    report = build_validation_report(metrics, {
        "policy": "checkpoint", "curriculum_frozen": True, "terrain_level": -1,
        "num_envs": 64, "episode_steps": 1500, "policy_dt_s": 0.02,
    })
    assert report["scenarios"]["forward"]["overall"]["actuation/action_saturation_rate"] < 0.05
    assert not report["selection_evidence"]["performance_pass"]


def test_continuous_route_is_a_required_performance_gate() -> None:
    metrics = {scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS}
    for values in metrics.values():
        values["locomotion/forward_velocity_rmse_mps"] = torch.full((11,), 0.05)
        values["locomotion/yaw_rate_rmse_radps"] = torch.full((11,), 0.05)
    metrics["forward"]["terrain/continuous_route_complete"][-1] = 0.0
    report = build_validation_report(metrics, {
        "policy": "checkpoint", "curriculum_frozen": True, "terrain_level": -1,
        "num_envs": 64, "episode_steps": 1500, "policy_dt_s": 0.02,
    })
    assert not report["selection_evidence"]["continuous_route_complete"]
    assert not report["selection_evidence"]["performance_pass"]
