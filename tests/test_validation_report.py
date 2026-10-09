"""Aggregation checks for the fixed locomotion validation suite."""

import torch

from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.validation import (
    DEFAULT_VALIDATION_SCENARIOS,
    build_validation_report,
)


def _scenario_metrics() -> dict[str, torch.Tensor]:
    terrain_type = torch.arange(9, dtype=torch.float32)
    return {
        "terrain/type_id": terrain_type,
        "locomotion/forward_velocity_rmse_mps": torch.arange(1, 10, dtype=torch.float32),
        "locomotion/lateral_velocity_rmse_mps": torch.full((9,), 0.05),
        "locomotion/yaw_rate_rmse_radps": torch.full((9,), 0.2),
        "locomotion/body_height_rmse_m": torch.full((9,), 0.01),
        "safety/unsafe_termination": torch.zeros(9),
        "safety/timeout": torch.ones(9),
        "safety/base_collision_rate": torch.zeros(9),
        "support/invalid_rate": torch.zeros(9),
        "support/wheel_contact_fraction": torch.full((9,), 0.75),
        "safety/base_tilt_max_rad": torch.linspace(0.1, 0.9, 9),
        "actuation/action_saturation_rate": torch.zeros(9),
        "actuation/leg_mechanical_energy_j": torch.ones(9),
        "actuation/wheel_mechanical_energy_j": torch.full((9,), 2.0),
        "runtime/episode_complete": torch.ones(9),
        "runtime/all_finite": torch.ones(9),
    }


def test_complete_suite_is_eligible_and_reports_macro_worst() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }

    report = build_validation_report(metrics, {"ppo_updates": 0})

    assert report["selection_evidence"]["eligible_for_checkpoint_comparison"]
    assert not report["selection_evidence"]["performance_pass"]
    forward = report["scenarios"]["forward"]["terrain_aggregate"]
    assert forward["macro/locomotion/forward_velocity_rmse_mps"] == 5.0
    assert forward["worst/locomotion/forward_velocity_rmse_mps"] == 9.0
    assert forward["worst/safety/timeout_survival_rate"] == 1.0


def test_missing_terrain_family_blocks_checkpoint_comparison() -> None:
    metrics = {
        scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS
    }
    for values in metrics.values():
        mask = values["terrain/type_id"] != 8
        for name in values:
            values[name] = values[name][mask]

    report = build_validation_report(metrics, {"ppo_updates": 0})

    assert not report["selection_evidence"]["eligible_for_checkpoint_comparison"]
    assert report["scenarios"]["forward"]["terrain_coverage"]["missing"] == ["pit"]


def test_source_alignment_requires_only_its_six_declared_families() -> None:
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
        values["locomotion/forward_velocity_rmse_mps"] = torch.full((9,), 0.05)
        values["locomotion/yaw_rate_rmse_radps"] = torch.full((9,), 0.05)

    report = build_validation_report(metrics, {
        "ppo_updates": 0, "policy": "checkpoint", "curriculum_frozen": True,
        "terrain_level": -1, "num_envs": 64, "episode_steps": 1500, "policy_dt_s": 0.02,
    })

    assert report["selection_evidence"]["performance_pass"]


def test_good_overall_average_cannot_hide_a_failing_family() -> None:
    metrics = {scenario.name: _scenario_metrics() for scenario in DEFAULT_VALIDATION_SCENARIOS}
    for values in metrics.values():
        values["locomotion/forward_velocity_rmse_mps"] = torch.full((9,), 0.05)
        values["locomotion/yaw_rate_rmse_radps"] = torch.full((9,), 0.05)
    metrics["forward"]["actuation/action_saturation_rate"][0] = 0.30
    report = build_validation_report(metrics, {
        "policy": "checkpoint", "curriculum_frozen": True, "terrain_level": -1,
        "num_envs": 64, "episode_steps": 1500, "policy_dt_s": 0.02,
    })
    assert report["scenarios"]["forward"]["overall"]["actuation/action_saturation_rate"] < 0.05
    assert not report["selection_evidence"]["performance_pass"]
