"""Pure-Python checks for OmegaConf training and benchmark contracts."""

from pathlib import Path

import pytest

from base_locomotion_stackforce_quadrupedal.training import (
    launcher_kwargs,
    load_config,
    validate_benchmark_config,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_yaml_defaults_to_headless_and_wandb() -> None:
    config = load_config(PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml")

    validate_training_config(config)

    assert config.launcher.viz == "none"
    assert config.wandb.enabled
    assert launcher_kwargs(config)["visualizer"] is None
    assert launcher_kwargs(config)["headless"] is True


def test_training_dotlist_override_is_resolved() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml",
        ("agent.max_iterations=3", "launcher.viz=none", "wandb.mode=offline"),
    )

    validate_training_config(config)

    assert config.agent.max_iterations == 3
    assert launcher_kwargs(config)["headless"]
    assert config.wandb.mode == "offline"


def test_training_yaml_inheritance_is_resolved() -> None:
    config = load_config(PROJECT_ROOT / "configs/train/main/b2_bisec_full.yaml")

    validate_training_config(config)

    assert config.env.terrain_profile == "union_foundation"
    assert config.agent.actor.architecture == "sparse_moe"
    assert config.agent.actor.reflex_enabled is True
    assert config.agent.algorithm.gate_entropy_coef == 0.03
    assert "extends" not in config


def test_resume_requires_exact_run_and_checkpoint() -> None:
    config_path = PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml"
    missing = load_config(config_path, ("agent.resume=true",))
    with pytest.raises(ValueError, match="requires exact"):
        validate_training_config(missing)

    wildcard = load_config(
        config_path,
        (
            "agent.resume=true",
            "agent.load_run=.*",
            "agent.load_checkpoint=model_.*.pt",
        ),
    )
    with pytest.raises(ValueError, match="exact names"):
        validate_training_config(wildcard)

    exact = load_config(
        config_path,
        (
            "agent.resume=true",
            "agent.load_run=2026-09-30_13-29-44_directional_curriculum_v6_gate1k",
            "agent.load_checkpoint=model_999.pt",
        ),
    )
    validate_training_config(exact)


def test_benchmark_yaml_contracts() -> None:
    for name, kind in (("plateau", "plateau"), ("washboard", "washboard")):
        config = load_config(PROJECT_ROOT / f"configs/benchmark/{name}.yaml")
        validate_benchmark_config(config)
        assert config.benchmark.kind == kind
        assert config.launcher.viz == "kit"
