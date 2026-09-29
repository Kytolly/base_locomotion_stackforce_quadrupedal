"""Pure-Python checks for OmegaConf training and benchmark contracts."""

from pathlib import Path

from base_locomotion_stackforce_quadrupedal.training import (
    launcher_kwargs,
    load_config,
    validate_benchmark_config,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_yaml_defaults_to_kit_and_wandb() -> None:
    config = load_config(PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml")

    validate_training_config(config)

    assert config.launcher.viz == "kit"
    assert config.wandb.enabled
    assert launcher_kwargs(config)["visualizer"] == ["kit"]


def test_training_dotlist_override_is_resolved() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/train/base_locomotion_complex.yaml",
        ("agent.max_iterations=3", "launcher.viz=none", "wandb.mode=offline"),
    )

    validate_training_config(config)

    assert config.agent.max_iterations == 3
    assert launcher_kwargs(config)["headless"]
    assert config.wandb.mode == "offline"


def test_benchmark_yaml_contracts() -> None:
    for name, kind in (("plateau", "plateau"), ("washboard", "washboard")):
        config = load_config(PROJECT_ROOT / f"configs/benchmark/{name}.yaml")
        validate_benchmark_config(config)
        assert config.benchmark.kind == kind
        assert config.launcher.viz == "kit"
