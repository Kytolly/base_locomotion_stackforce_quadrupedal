"""Versioned metric collection and dashboard visibility switches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


METRIC_GROUPS = (
    "core",
    "safety",
    "runtime",
    "command",
    "support",
    "actuation",
    "terrain",
    "motion",
    "reward",
)

# Core tracking, safety and finite-value health are the recommended training view.
DEFAULT_METRIC_GROUPS = {
    "core": True,
    "safety": True,
    "runtime": True,
    "command": False,
    "support": False,
    "actuation": False,
    "terrain": False,
    "motion": False,
    "reward": False,
}
DEFAULT_WANDB_PANELS = dict(DEFAULT_METRIC_GROUPS)


def _resolve_groups(values: Mapping[str, Any] | None, defaults: Mapping[str, bool]) -> dict[str, bool]:
    values = values or {}
    return {name: bool(values.get(name, enabled)) for name, enabled in defaults.items()}


def resolve_metric_logging_config(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize YAML logging settings and keep collection separate from panels."""
    values = values or {}
    metrics = values.get("metrics", {})
    panels = values.get("wandb_panels", {})
    metric_enabled = bool(metrics.get("enabled", True))
    panel_enabled = bool(panels.get("enabled", True))
    metric_groups = _resolve_groups(metrics.get("groups"), DEFAULT_METRIC_GROUPS)
    panel_groups = _resolve_groups(panels.get("groups"), DEFAULT_WANDB_PANELS)
    if not metric_enabled:
        metric_groups = {name: False for name in METRIC_GROUPS}
    if not panel_enabled:
        panel_groups = {name: False for name in METRIC_GROUPS}
    return {
        "metrics_enabled": metric_enabled,
        "metric_groups": metric_groups,
        "panels_enabled": panel_enabled,
        "panel_groups": panel_groups,
    }


def metric_group_for_name(name: str) -> str:
    """Map a metric namespace to its configurable group."""
    parts = name.split("/")
    namespace = parts[0]
    if namespace in {"train", "val", "diagnostic"} and len(parts) > 1:
        namespace = parts[1]
    if namespace in {"locomotion", "optimization"}:
        return "core"
    if namespace in METRIC_GROUPS:
        return namespace
    return "core"


def filter_metric_log(metric_log: Mapping[str, Any], panel_groups: Mapping[str, bool]) -> dict[str, Any]:
    """Keep only groups selected for the dashboard sink."""
    return {
        name: value
        for name, value in metric_log.items()
        if panel_groups.get(metric_group_for_name(name), True)
    }


__all__ = [
    "DEFAULT_METRIC_GROUPS",
    "DEFAULT_WANDB_PANELS",
    "METRIC_GROUPS",
    "filter_metric_log",
    "metric_group_for_name",
    "resolve_metric_logging_config",
]
