"""Deterministic command suite and macro/worst-family report construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from base_locomotion_stackforce_quadrupedal.benchmark.acceptance import MAXIMUMS, MINIMUMS, passes_thresholds

from ...env.terrains import TRAINING_TERRAIN_FAMILIES, get_complex_terrain_cfg


VALIDATION_FAMILIES_BY_PROFILE = {
    "legacy_equal": tuple(
        family
        for family in TRAINING_TERRAIN_FAMILIES
        if family not in {"hf_pyramid_slope_inv", "flat", "continuous_mixed"}
    ),
    "source_alignment": (
        "random_rough",
        "boxes",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
    ),
    "union_consolidation": TRAINING_TERRAIN_FAMILIES,
}
VALIDATION_TERRAIN_COLUMNS = 15


def validation_terrain_layout(
    profile: str, seed: int
) -> tuple[Any, tuple[int, ...]]:
    """Build the real profile and map Isaac terrain columns to family IDs."""
    if profile not in VALIDATION_FAMILIES_BY_PROFILE:
        raise ValueError(
            f"Unsupported validation terrain profile {profile!r}; expected one of "
            f"{sorted(VALIDATION_FAMILIES_BY_PROFILE)}."
        )
    config = get_complex_terrain_cfg(
        seed=seed, num_cols=VALIDATION_TERRAIN_COLUMNS, profile=profile
    )
    proportions = [float(cfg.proportion) for cfg in config.sub_terrains.values()]
    total = sum(proportions)
    cumulative: list[float] = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)
    family_ids: list[int] = []
    names = tuple(config.sub_terrains)
    for column in range(config.num_cols):
        sample = column / config.num_cols + 0.001
        subterrain_index = next(
            index for index, boundary in enumerate(cumulative) if sample < boundary
        )
        family_ids.append(TRAINING_TERRAIN_FAMILIES.index(names[subterrain_index]))
    observed = {TRAINING_TERRAIN_FAMILIES[index] for index in family_ids}
    expected = set(VALIDATION_FAMILIES_BY_PROFILE[profile])
    if observed != expected:
        raise RuntimeError(
            f"Validation terrain layout for {profile!r} produced {sorted(observed)}, "
            f"expected {sorted(expected)}."
        )
    return config, tuple(family_ids)


@dataclass(frozen=True)
class ValidationScenario:
    """One fixed locomotion command evaluated from the same initial-state distribution."""

    name: str
    command: tuple[float, float, float, float]
    requires_continuous_route: bool = False


DEFAULT_VALIDATION_SCENARIOS = (
    ValidationScenario("stop", (0.0, 0.0, 0.0, 0.105)),
    ValidationScenario("forward", (0.35, 0.0, 0.0, 0.105), True),
    ValidationScenario("backward", (-0.35, 0.0, 0.0, 0.105)),
    ValidationScenario("lateral_left", (0.0, 0.10, 0.0, 0.105)),
    ValidationScenario("lateral_right", (0.0, -0.10, 0.0, 0.105)),
    ValidationScenario("left_turn", (0.0, 0.0, 0.25, 0.105)),
    ValidationScenario("right_turn", (0.0, 0.0, -0.25, 0.105)),
    ValidationScenario("forward_left_turn", (0.35, 0.0, 0.20, 0.105)),
    ValidationScenario("backward_right_turn", (-0.35, 0.0, -0.20, 0.105)),
    ValidationScenario("low_body", (0.0, 0.0, 0.0, 0.095)),
    ValidationScenario("high_body", (0.0, 0.0, 0.0, 0.115)),
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


def _scenario_report(
    metrics: dict[str, torch.Tensor], terrain_families: tuple[str, ...]
) -> dict[str, Any]:
    overall = {
        name: _mean(values)
        for name, values in metrics.items()
        if name not in {"terrain/type_id", "terrain/route_role_id"}
    }
    tilt = metrics["safety/base_tilt_max_rad"].float()
    overall["safety/base_tilt_max_p50_rad"] = float(torch.quantile(tilt, 0.50).item())
    overall["safety/base_tilt_max_p95_rad"] = float(torch.quantile(tilt, 0.95).item())

    terrain_type = metrics["terrain/type_id"].long()
    families: dict[str, dict[str, float | int]] = {}
    observed_families: list[str] = []
    for family in terrain_families:
        family_id = TRAINING_TERRAIN_FAMILIES.index(family)
        mask = terrain_type == family_id
        if not bool(mask.any()):
            families[family] = {"episodes": 0}
            continue
        observed_families.append(family)
        family_result: dict[str, float | int] = {"episodes": int(mask.sum().item())}
        family_result["safety/base_tilt_max_p95_rad"] = float(torch.quantile(tilt[mask], 0.95).item())
        for name in CORE_FAMILY_METRICS:
            output_name = (
                "safety/timeout_survival_rate" if name == "safety/timeout" else name
            )
            family_result[output_name] = _mean(metrics[name][mask])
        if family == "continuous_mixed":
            for name in (
                "terrain/continuous_route_complete",
                "terrain/continuous_route_ordered_checkpoints",
                "terrain/continuous_route_corridor_valid",
            ):
                family_result[name] = _mean(metrics[name][mask])
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
                family for family in terrain_families if family not in observed_families
            ],
            "complete": len(observed_families) == len(terrain_families),
        },
    }


def build_validation_report(
    scenario_metrics: dict[str, dict[str, torch.Tensor]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable report with checkpoint-selection evidence."""
    terrain_families = VALIDATION_FAMILIES_BY_PROFILE.get(
        str(metadata.get("terrain_profile", "union_consolidation")),
        TRAINING_TERRAIN_FAMILIES,
    )
    scenarios = {
        name: _scenario_report(metrics, terrain_families)
        for name, metrics in scenario_metrics.items()
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
    thresholds = {"maximum": MAXIMUMS, "minimum": {**MINIMUMS, "safety/timeout": 0.95}}
    protocol_complete = (
        metadata.get("policy") == "checkpoint"
        and metadata.get("curriculum_frozen") is True
        and metadata.get("terrain_level") == -1
        and metadata.get("num_envs", 0) >= 64
        and metadata.get("num_envs", 0) % 64 == 0
        and metadata.get("episode_steps", 0) * metadata.get("policy_dt_s", 0) >= 30.0
    )
    route_scenarios = {
        scenario.name for scenario in DEFAULT_VALIDATION_SCENARIOS
        if scenario.requires_continuous_route
    }
    continuous_route_complete = "continuous_mixed" not in terrain_families or (
        complete_commands
        and all(
            scenarios[name]["terrain_families"]["continuous_mixed"].get(
                "terrain/continuous_route_complete", 0.0
            ) == 1.0
            and scenarios[name]["terrain_families"]["continuous_mixed"].get(
                "terrain/continuous_route_ordered_checkpoints", 0.0
            ) == 3.0
            and scenarios[name]["terrain_families"]["continuous_mixed"].get(
                "terrain/continuous_route_corridor_valid", 0.0
            ) == 1.0
            for name in route_scenarios
        )
    )
    performance_pass = (
        complete_commands and complete_terrains and complete_episodes and all_finite
        and protocol_complete and continuous_route_complete
    )
    for result in scenarios.values():
        overall = result["overall"]
        performance_pass &= all(
            family.get("episodes", 0) > 0 and passes_thresholds(family)
            for family in result["terrain_families"].values()
        )
        performance_pass &= passes_thresholds(overall)
    return {
        "schema_version": 3,
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
            "continuous_route_complete": continuous_route_complete,
            "performance_pass": bool(performance_pass),
            "protocol_complete": protocol_complete,
            "thresholds": thresholds,
            "rule": (
                "Performance PASS requires all recorded command and terrain coverage, complete finite episodes, "
                "the ordered continuous-mixed route when that family is in the profile, and every fixed threshold; "
                "episode reward alone is not a selection criterion."
            ),
        },
    }
