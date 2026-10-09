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
    protocol_version = str(config.get("protocol_version", ""))
    if protocol_version.endswith("46d") and protocol_version != "E0-v1-46d":
        raise ValueError(f"Unsupported 46D protocol version: {protocol_version!r}.")
    if protocol_version == "E0-v1-46d" and int(config.get("actor_observation_dim", 0)) != 46:
        raise ValueError("E0-v1-46d requires actor_observation_dim=46.")
    if int(config.get("actor_observation_dim", 0)) <= 0:
        raise ValueError("actor_observation_dim must be positive.")
    if config.get("curriculum"):
        required_curricula = ("terrain_levels", "motion_family", "robustness_bin")
        missing_curricula = [name for name in required_curricula if name not in config.curriculum]
        if missing_curricula:
            raise ValueError(
                "46D E0 training requires independent curriculum sections: "
                + ", ".join(missing_curricula)
            )
    logging = config.get("logging", {})
    for section in ("metrics", "wandb_panels"):
        section_config = logging.get(section, {})
        if not isinstance(section_config.get("enabled", True), bool):
            raise ValueError(f"logging.{section}.enabled must be a boolean.")
        groups = section_config.get("groups", {})
        allowed_groups = {
            "core", "safety", "runtime", "command", "support",
            "actuation", "terrain", "motion", "reward",
        }
        unknown = sorted(set(groups) - allowed_groups)
        if unknown:
            raise ValueError(f"Unknown logging.{section}.groups entries: {', '.join(unknown)}")
        non_boolean = [name for name, value in groups.items() if not isinstance(value, bool)]
        if non_boolean:
            raise ValueError(
                f"logging.{section}.groups values must be booleans: {', '.join(non_boolean)}"
            )
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
    physics_values = config.env.get("physics", {})
    invalid_physics = [
        f"env.physics.{name}"
        for name, value in physics_values.items()
        if float(value) <= 0
    ]
    if invalid_physics:
        raise ValueError(
            f"Configuration values must be positive: {', '.join(invalid_physics)}"
        )
    terrain_profile = config.env.get("terrain_profile")
    if terrain_profile is not None:
        allowed_terrain_profiles = {
            "legacy_equal",
            "union_foundation",
            "union_expansion",
            "union_composition",
            "union_consolidation",
            "source_alignment",
        }
        if str(terrain_profile) not in allowed_terrain_profiles:
            raise ValueError(
                f"Unknown env.terrain_profile {terrain_profile!r}; "
                f"expected one of {sorted(allowed_terrain_profiles)}."
            )
    if bool(config.wandb.enabled) and not str(config.wandb.project).strip():
        raise ValueError("wandb.project must be non-empty when W&B is enabled.")
    if bool(config.agent.get("resume", False)):
        load_run = config.agent.get("load_run")
        load_checkpoint = config.agent.get("load_checkpoint")
        if not load_run or not load_checkpoint:
            raise ValueError(
                "agent.resume=true requires exact agent.load_run and "
                "agent.load_checkpoint values."
            )
        pattern_tokens = ("*", "?", "[", "]", "(", ")", "|", "+", "^", "$", "\\")
        if any(token in str(load_run) for token in pattern_tokens) or any(
            token in str(load_checkpoint) for token in pattern_tokens
        ):
            raise ValueError(
                "Resume selectors must be exact names; regular expressions and "
                "wildcards are not allowed."
            )
        if not str(load_checkpoint).endswith(".pt"):
            raise ValueError("agent.load_checkpoint must name one .pt checkpoint file.")
    distribution = config.agent.get("actor", {}).get("distribution_cfg", {})
    if str(distribution.get("class_name", "")).endswith(":SquashedGaussianDistribution"):
        if distribution.get("std_type") != "log" or not (
            0 < float(distribution.get("min_std", 0))
            <= float(distribution.get("init_std", 0))
            <= float(distribution.get("max_std", 0))
            <= 0.5
        ):
            raise ValueError("Bounded policy requires log std and 0 < min_std <= init_std <= max_std <= 0.5.")
    for model_name in ("actor", "critic"):
        model = config.agent.get(model_name, {})
        architecture = str(model.get("architecture", "mlp"))
        if architecture not in {"mlp", "sparse_moe"}:
            raise ValueError(f"Unknown agent.{model_name}.architecture: {architecture!r}.")
        if architecture == "sparse_moe":
            num_experts = int(model.get("num_experts", 6))
            top_k = int(model.get("top_k", 2))
            if not 1 <= top_k <= num_experts:
                raise ValueError(f"agent.{model_name}.top_k must be in [1, num_experts].")
    for coefficient in (
        "gate_entropy_coef",
        "expert_orthogonality_coef",
        "temporal_consistency_coef",
    ):
        if float(config.agent.get("algorithm", {}).get(coefficient, 0.0)) < 0.0:
            raise ValueError(f"agent.algorithm.{coefficient} must be non-negative.")


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
    if float(config.benchmark.get("episode_length_s", 35.0)) <= 0:
        raise ValueError("benchmark.episode_length_s must be positive.")


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
