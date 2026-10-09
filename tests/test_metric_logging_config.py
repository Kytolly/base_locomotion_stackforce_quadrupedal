"""Keep the training log and W&B dashboard configuration explicit."""

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/evaluation/metric/logging.py"
)
SPEC = importlib.util.spec_from_file_location("metric_logging", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
metric_logging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metric_logging)


def test_default_logging_enables_only_recommended_groups() -> None:
    config = metric_logging.resolve_metric_logging_config()

    assert config["metric_groups"] == {
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
    assert config["panel_groups"] == config["metric_groups"]


def test_metric_collection_and_dashboard_visibility_are_independent() -> None:
    config = metric_logging.resolve_metric_logging_config(
        {
            "metrics": {"groups": {"actuation": True, "reward": False}},
            "wandb_panels": {"groups": {"actuation": False, "reward": True}},
        }
    )
    values = {
        "train/locomotion/forward_velocity_rmse_mps": 0.2,
        "train/actuation/action_rate_rms": 0.3,
        "train/reward/weighted/tracking": 1.0,
    }

    assert config["metric_groups"]["actuation"]
    assert not config["metric_groups"]["reward"]
    assert metric_logging.filter_metric_log(values, config["panel_groups"]) == {
        "train/locomotion/forward_velocity_rmse_mps": 0.2,
        "train/reward/weighted/tracking": 1.0,
    }


def test_disabling_collection_skips_all_custom_metric_groups() -> None:
    config = metric_logging.resolve_metric_logging_config({"metrics": {"enabled": False}})

    assert not config["metrics_enabled"]
    assert not any(config["metric_groups"].values())
