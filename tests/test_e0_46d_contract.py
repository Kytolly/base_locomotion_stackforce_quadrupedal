"""Contracts for the E0 task that must remain sensor-neutral at the Actor."""

from pathlib import Path

from omegaconf import OmegaConf

from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy import (
    ACTOR_INPUT_DIM,
)
from base_locomotion_stackforce_quadrupedal.training import load_config, validate_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_e0_training_contract_keeps_the_46d_actor() -> None:
    config = load_config(PROJECT_ROOT / "configs/train/base_locomotion_e0_46d.yaml")
    validate_training_config(config)
    assert ACTOR_INPUT_DIM == 46
    assert config.actor_observation_dim == 46
    assert config.protocol_version == "E0-v1-46d"
    assert set(config.curriculum) == {"terrain_levels", "motion_family", "robustness_bin"}


def test_curricula_are_split_into_independent_files() -> None:
    curriculum_dir = PROJECT_ROOT / "configs/curriculum"
    assert {path.name for path in curriculum_dir.glob("*.yaml")} == {
        "terrain.yaml",
        "motion.yaml",
        "robustness.yaml",
    }


def test_experiment_configs_are_split_and_enable_all_logging() -> None:
    main_dir = PROJECT_ROOT / "configs/train/main"
    auxiliary_dir = PROJECT_ROOT / "configs/train/auxiliary"
    assert {path.stem for path in main_dir.glob("*.yaml")} == {
        "b0_t1_ppo",
        "b1_moe_core",
        "b2_moe_reflex",
        "b2_bisec_full",
    }
    assert len(list(auxiliary_dir.glob("*.yaml"))) == 15

    for config_path in (*main_dir.glob("*.yaml"), *auxiliary_dir.glob("*.yaml")):
        config = load_config(config_path)
        validate_training_config(config)
        assert config.wandb.enabled is True
        assert config.wandb.mode == "online"
        assert config.logging.metrics.enabled is True
        assert config.logging.wandb_panels.enabled is True
        assert all(config.logging.metrics.groups.values())
        assert all(config.logging.wandb_panels.groups.values())


def test_experiment_matrix_points_to_named_training_configs() -> None:
    matrix = OmegaConf.load(PROJECT_ROOT / "configs/experiments/e0_experiments.yaml")

    for name, experiment in matrix.experiments.items():
        config_path = PROJECT_ROOT / experiment.config
        assert config_path.is_file()
        assert config_path.stem == name
