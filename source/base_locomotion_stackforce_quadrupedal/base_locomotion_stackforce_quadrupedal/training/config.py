"""Strict OmegaConf loading and configclass override helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Load one YAML file and merge OmegaConf dotlist overrides."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(
            f"Expected a mapping at {config_path}, received {type(config).__name__}."
        )
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def _require(config: DictConfig, paths: Sequence[str]) -> None:
    missing = [
        path for path in paths if OmegaConf.select(config, path, default=None) is None
    ]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")


def _validate_launcher(config: DictConfig) -> None:
    _require(config, ("launcher.device", "launcher.viz"))
    valid_visualizers = {"kit", "newton", "rerun", "viser", "none"}
    visualizers = str(config.launcher.viz).split(",")
    invalid = [name for name in visualizers if name not in valid_visualizers]
    if invalid or ("none" in visualizers and len(visualizers) != 1):
        raise ValueError(f"Invalid launcher.viz value: {config.launcher.viz!r}.")
    if int(config.launcher.get("max_visible_envs", 0)) < 0:
        raise ValueError("launcher.max_visible_envs must be non-negative.")


def validate_training_config(config: DictConfig) -> None:
    """Validate the fields required by the YAML training entrypoint."""
    _require(
        config,
        (
            "schema_version",
            "task",
            "env.num_envs",
            "env.seed",
            "env.sim_dt",
            "env.decimation",
            "agent.max_iterations",
            "agent.num_steps_per_env",
            "agent.experiment_name",
            "wandb.enabled",
            "wandb.project",
            "runtime.log_root",
        ),
    )
    _validate_launcher(config)
    positive = {
        "env.num_envs": config.env.num_envs,
        "env.sim_dt": config.env.sim_dt,
        "env.decimation": config.env.decimation,
        "agent.max_iterations": config.agent.max_iterations,
        "agent.num_steps_per_env": config.agent.num_steps_per_env,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"Configuration values must be positive: {', '.join(invalid)}")
    if bool(config.wandb.enabled) and not str(config.wandb.project).strip():
        raise ValueError("wandb.project must be non-empty when W&B is enabled.")


def validate_benchmark_config(config: DictConfig) -> None:
    """Validate one fixed-track benchmark configuration."""
    _require(
        config,
        (
            "schema_version",
            "task",
            "benchmark.kind",
            "benchmark.seed",
            "benchmark.max_steps",
            "policy.zero_policy",
            "output",
        ),
    )
    _validate_launcher(config)
    if config.benchmark.kind not in {"plateau", "washboard"}:
        raise ValueError("benchmark.kind must be 'plateau' or 'washboard'.")
    checkpoint = config.policy.get("checkpoint")
    if not bool(config.policy.zero_policy) and not checkpoint:
        raise ValueError(
            "policy.checkpoint is required when policy.zero_policy is false."
        )
    if int(config.benchmark.max_steps) <= 0:
        raise ValueError("benchmark.max_steps must be positive.")


def apply_config(
    target: Any, values: Mapping[str, Any] | DictConfig, prefix: str = ""
) -> None:
    """Apply a strict nested mapping to an Isaac Lab configclass instance."""
    plain = (
        OmegaConf.to_container(values, resolve=True)
        if isinstance(values, DictConfig)
        else values
    )
    if not isinstance(plain, Mapping):
        raise TypeError(f"Expected mapping at {prefix or '<root>'}.")
    for key, value in plain.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if not hasattr(target, key):
            raise KeyError(f"Unknown configuration field: {path}")
        current = getattr(target, key)
        if (
            isinstance(value, Mapping)
            and current is not None
            and not isinstance(current, Mapping)
        ):
            apply_config(current, value, path)
        else:
            setattr(target, key, value)


def launcher_kwargs(config: DictConfig) -> dict[str, Any]:
    """Translate the concise YAML launcher block to AppLauncher arguments."""
    viz = str(config.launcher.viz)
    visualizers = None if viz == "none" else viz.split(",")
    return {
        "device": str(config.launcher.device),
        "visualizer": visualizers,
        "visualizer_explicit": True,
        "visualizer_disable_all": viz == "none",
        "headless": viz == "none",
        "enable_cameras": bool(config.launcher.get("enable_cameras", False)),
        "max_visible_envs": int(config.launcher.get("max_visible_envs", 16)),
    }
