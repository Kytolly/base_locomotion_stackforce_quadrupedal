"""Deterministic command suite and macro/worst-family report construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...env.terrains import TERRAIN_FAMILIES


@dataclass(frozen=True)
class ValidationScenario:
    """One fixed locomotion command evaluated from the same initial-state distribution."""

    name: str
    command: tuple[float, float, float, float]


DEFAULT_VALIDATION_SCENARIOS = (
    ValidationScenario("stop", (0.0, 0.0, 0.0, 0.105)),
    ValidationScenario("forward", (0.35, 0.0, 0.0, 0.105)),
    ValidationScenario("backward", (-0.35, 0.0, 0.0, 0.105)),
    ValidationScenario("left_turn", (0.0, 0.0, 0.25, 0.105)),
    ValidationScenario("right_turn", (0.0, 0.0, -0.25, 0.105)),
)


CORE_FAMILY_METRICS = (
    "locomotion/forward_velocity_rmse_mps",
    "locomotion/lateral_velocity_rmse_mps",
    "locomotion/yaw_rate_rmse_radps",
    "locomotion/body_height_rmse_m",
    "safety/unsafe_termination",
    "safety/timeout",
    "safety/base_collision_rate",
    "support/invalid_rate",
    "support/wheel_contact_fraction",
    "actuation/action_saturation_rate",
    "actuation/leg_mechanical_energy_j",
    "actuation/wheel_mechanical_energy_j",
    "runtime/all_finite",
)


def _mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item())


def _scenario_report(metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    overall = {
        name: _mean(values)
        for name, values in metrics.items()
        if name != "terrain/type_id"
    }
    tilt = metrics["safety/base_tilt_max_rad"].float()
    overall["safety/base_tilt_max_p50_rad"] = float(torch.quantile(tilt, 0.50).item())
    overall["safety/base_tilt_max_p95_rad"] = float(torch.quantile(tilt, 0.95).item())

    terrain_type = metrics["terrain/type_id"].long()
    families: dict[str, dict[str, float | int]] = {}
    observed_families: list[str] = []
    for family_id, family in enumerate(TERRAIN_FAMILIES):
        mask = terrain_type == family_id
        if not bool(mask.any()):
            families[family] = {"episodes": 0}
            continue
        observed_families.append(family)
        family_result: dict[str, float | int] = {"episodes": int(mask.sum().item())}
        for name in CORE_FAMILY_METRICS:
            output_name = (
                "safety/timeout_survival_rate" if name == "safety/timeout" else name
            )
            family_result[output_name] = _mean(metrics[name][mask])
        families[family] = family_result

    aggregate: dict[str, float] = {}
    for name in CORE_FAMILY_METRICS:
        output_name = (
            "safety/timeout_survival_rate" if name == "safety/timeout" else name
        )
        values = [float(families[family][output_name]) for family in observed_families]
        if not values:
            continue
        aggregate[f"macro/{output_name}"] = sum(values) / len(values)
        if output_name in (
            "safety/timeout_survival_rate",
            "runtime/all_finite",
            "support/wheel_contact_fraction",
        ):
            aggregate[f"worst/{output_name}"] = min(values)
        else:
            aggregate[f"worst/{output_name}"] = max(values)

    return {
        "episodes": int(terrain_type.numel()),
        "overall": overall,
        "terrain_families": families,
        "terrain_aggregate": aggregate,
        "terrain_coverage": {
            "observed": observed_families,
            "missing": [
                family for family in TERRAIN_FAMILIES if family not in observed_families
            ],
            "complete": len(observed_families) == len(TERRAIN_FAMILIES),
        },
    }


def build_validation_report(
    scenario_metrics: dict[str, dict[str, torch.Tensor]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable report with checkpoint-selection evidence."""
    scenarios = {
        name: _scenario_report(metrics) for name, metrics in scenario_metrics.items()
    }
    expected_scenarios = {scenario.name for scenario in DEFAULT_VALIDATION_SCENARIOS}
    complete_commands = expected_scenarios.issubset(scenarios)
    complete_terrains = all(
        result["terrain_coverage"]["complete"] for result in scenarios.values()
    )
    complete_episodes = all(
        result["overall"].get("runtime/episode_complete", 0.0) == 1.0
        for result in scenarios.values()
    )
    all_finite = all(
        result["overall"]["runtime/all_finite"] == 1.0 for result in scenarios.values()
    )
    thresholds = {
        "forward_velocity_rmse_mps_max": 0.15,
        "lateral_velocity_rmse_mps_max": 0.12,
        "yaw_rate_rmse_radps_max": 0.10,
        "body_height_rmse_m_max": 0.025,
        "unsafe_termination_rate_max": 0.05,
        "base_collision_rate_max": 0.01,
        "support_invalid_rate_max": 0.01,
        "wheel_contact_fraction_min": 0.50,
        "action_saturation_rate_max": 0.05,
        "base_tilt_max_p95_rad_max": 1.05,
    }
    performance_pass = complete_commands and complete_terrains and complete_episodes and all_finite
    for result in scenarios.values():
        overall = result["overall"]
        performance_pass &= (
            overall["locomotion/forward_velocity_rmse_mps"] <= thresholds["forward_velocity_rmse_mps_max"]
            and overall["locomotion/lateral_velocity_rmse_mps"] <= thresholds["lateral_velocity_rmse_mps_max"]
            and overall["locomotion/yaw_rate_rmse_radps"] <= thresholds["yaw_rate_rmse_radps_max"]
            and overall["locomotion/body_height_rmse_m"] <= thresholds["body_height_rmse_m_max"]
            and overall["safety/unsafe_termination"] <= thresholds["unsafe_termination_rate_max"]
            and overall["safety/base_collision_rate"] <= thresholds["base_collision_rate_max"]
            and overall["support/invalid_rate"] <= thresholds["support_invalid_rate_max"]
            and overall["support/wheel_contact_fraction"] >= thresholds["wheel_contact_fraction_min"]
            and overall["actuation/action_saturation_rate"] <= thresholds["action_saturation_rate_max"]
            and overall["safety/base_tilt_max_p95_rad"] <= thresholds["base_tilt_max_p95_rad_max"]
        )
    return {
        "schema_version": 1,
        "metadata": metadata,
        "scenarios": scenarios,
        "selection_evidence": {
            "eligible_for_checkpoint_comparison": (
                complete_commands
                and complete_terrains
                and complete_episodes
                and all_finite
            ),
            "complete_command_suite": complete_commands,
            "complete_terrain_coverage": complete_terrains,
            "complete_episodes": complete_episodes,
            "all_finite": all_finite,
            "performance_pass": bool(performance_pass),
            "thresholds": thresholds,
            "rule": (
                "Performance PASS requires all recorded command and terrain coverage, complete finite episodes, "
                "and every fixed threshold; episode reward alone is not a selection criterion."
            ),
        },
    }
