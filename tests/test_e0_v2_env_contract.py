"""E0-v2-46d contract wiring: YAML validation, terrain caps, env config and resume rejection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from base_locomotion_stackforce_quadrupedal.training import load_config, validate_training_config
from base_locomotion_stackforce_quadrupedal.training.checkpoint import (
    build_training_contract,
    checkpoint_action_contract,
    require_compatible_training_contract,
    save_training_contract,
)
from base_locomotion_stackforce_quadrupedal.training.distribution import SquashedGaussianDistribution
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import (
    LEGACY_OBSTACLE_HEIGHTS,
    get_complex_terrain_cfg,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (
    load_servo_contract,
)


ROOT = Path(__file__).resolve().parents[1]
E0 = ROOT / "configs/train/base_locomotion_e0_46d.yaml"


def _env_cfg():
    try:
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
            BaseLocomotionComplexEnvCfg,
        )

        return BaseLocomotionComplexEnvCfg()
    except Exception as error:  # pragma: no cover - depends on the Isaac Lab install
        pytest.skip(f"env config needs the Isaac Lab runtime here: {error}")


def test_e0_yaml_is_v2_with_explicit_contract_and_switches():
    config = load_config(E0)
    validate_training_config(config)
    assert config.protocol_version == "E0-v2-46d"
    assert config.env.action_contract == "servo_range_v2"
    assert config.env.terrain_init_level is None
    assert config.agent.experiment_name == "base_locomotion_e0_v2_46d"
    assert set(config.env.throughput) == {
        "enabled_self_collisions", "solver_position_iteration_count", "foot_track_pose",
        "foot_contact_update_period", "replicate_physics",
    }


def test_solver_iterations_hold_the_closures_and_match_the_env_default():
    # 8 iterations leave the closures converged only at position level (docs/OUTER_HIP_STATIC_FORCE_DIAGNOSIS.md).
    cfg = _env_cfg()
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots.stackforce import (
        SOLVER_POSITION_ITERATIONS,
    )

    assert SOLVER_POSITION_ITERATIONS == 32
    assert load_config(E0).env.throughput.solver_position_iteration_count == SOLVER_POSITION_ITERATIONS
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == SOLVER_POSITION_ITERATIONS


def test_pre_v2_experiment_lines_keep_legacy_legs():
    for name in ("base_locomotion_complex.yaml", "base_locomotion_hybrid.yaml"):
        config = load_config(ROOT / "configs/train" / name)
        validate_training_config(config)
        assert config.env.action_contract == "legacy_symmetric"


def test_validator_rejects_incomplete_v2_configs():
    with pytest.raises(ValueError, match="action_contract"):
        validate_training_config(load_config(E0, ("env.action_contract=null",)))
    with pytest.raises(ValueError, match="Unknown env.action_contract"):
        validate_training_config(load_config(E0, ("env.action_contract=wide",)))
    with pytest.raises(ValueError, match="throughput"):
        validate_training_config(load_config(E0, ("env.throughput.num_threads=4",)))
    with pytest.raises(ValueError, match="terrain_init_level"):
        validate_training_config(load_config(E0, ("env.terrain_init_level=-1",)))
    with pytest.raises(ValueError, match="init_std"):
        validate_training_config(load_config(E0, ("agent.actor.distribution_cfg.init_std=[0.6]",)))


def test_per_dimension_std_initialises_legs_and_wheels_separately():
    distribution = load_config(E0).agent.actor.distribution_cfg
    init_std = list(distribution.init_std)
    bounds = {"min_std": distribution.min_std, "max_std": distribution.max_std}
    dist = SquashedGaussianDistribution(12, init_std=init_std, **bounds)
    dist.update(torch.zeros(4, 12))
    assert torch.allclose(dist.std[0], torch.tensor(init_std), atol=1e-6)
    with pytest.raises(ValueError):
        SquashedGaussianDistribution(12, init_std=init_std[:8], **bounds)
    with pytest.raises(ValueError):
        SquashedGaussianDistribution(12, init_std=[0.01] * 12, **bounds)


def test_obstacle_heights_follow_the_contract_and_legacy_is_unchanged():
    legacy = get_complex_terrain_cfg(profile="union_expansion")
    assert legacy.sub_terrains["pyramid_stairs"].step_height_range == (0.04, 0.14)
    assert legacy.sub_terrains["boxes"].grid_height_range == (0.03, 0.12)
    assert legacy.sub_terrains["continuous_mixed"].step_height_range == (0.03, 0.12)
    assert LEGACY_OBSTACLE_HEIGHTS["pyramid_stairs"] == (0.04, 0.14)
    heights = {k: tuple(v) for k, v in load_servo_contract()["obstacles"]["height_ranges_m"].items()}
    v2 = get_complex_terrain_cfg(profile="union_expansion", obstacle_heights=heights)
    assert v2.sub_terrains["pyramid_stairs"].step_height_range[0] == pytest.approx(0.02)
    assert v2.sub_terrains["pyramid_stairs_inv"].step_height_range == v2.sub_terrains["pyramid_stairs"].step_height_range
    assert v2.sub_terrains["boxes"].grid_height_range[0] == pytest.approx(0.02)
    assert v2.sub_terrains["continuous_mixed"].step_height_range[0] == pytest.approx(0.02)
    with pytest.raises(KeyError):
        get_complex_terrain_cfg(obstacle_heights={"stairs": (0.02, 0.1)})


def test_env_config_applies_servo_contract_and_restores_legacy():
    cfg = _env_cfg()
    contract = load_servo_contract()
    assert cfg.action_contract == "servo_range_v2" and cfg.protocol_version == "E0-v2-46d"
    leg = cfg.actions.leg_position
    assert leg.lower["FR_Outer_Hip_Joint"] == contract["joints"]["FR_Outer_Hip_Joint"]["lower"]
    assert leg.upper["RR_Outer_Hip_Joint"] == contract["joints"]["RR_Outer_Hip_Joint"]["upper"]
    joint_pos = cfg.scene.robot.init_state.joint_pos
    assert joint_pos["FR_Inner_Hip_Joint"] == contract["joints"]["FR_Inner_Hip_Joint"]["default"]
    assert joint_pos["FR_Outer_Knee_Joint"] == contract["passive_knees"]["FR_Outer_Knee_Joint"]
    heights = contract["body_height"]
    assert cfg.scene.robot.init_state.pos[2] == heights["init_root_m"]
    assert tuple(cfg.commands.locomotion.ranges.body_height_m) == tuple(heights["command_range_m"])
    assert cfg.rewards.low_body_height_margin.params["warning_height"] == heights["warning_m"]
    assert cfg.terminations.base_height.params["minimum_height"] == heights["minimum_m"]
    assert cfg.events.servo_joint_limits is not None
    assert cfg.events.servo_joint_limits.mode == "startup"
    assert cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range[0] == 0.02

    cfg.action_contract = "legacy_symmetric"
    cfg.apply_action_contract()
    assert cfg.protocol_version == "E0-v1-46d"
    assert cfg.actions.leg_position.scale == 0.5
    assert cfg.scene.robot.init_state.joint_pos == {".*": 0.0}
    assert cfg.scene.robot.init_state.pos[2] == pytest.approx(0.11)
    assert tuple(cfg.commands.locomotion.ranges.body_height_m) == (0.095, 0.115)
    assert cfg.terminations.base_height.params["minimum_height"] == pytest.approx(0.06)
    assert cfg.events.servo_joint_limits is None
    assert cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range == (0.04, 0.14)


def test_env_options_set_throughput_switches_and_initial_level():
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
        apply_e0_env_options,
    )

    cfg = _env_cfg()
    apply_e0_env_options(cfg, {
        "action_contract": "servo_range_v2",
        "terrain_init_level": 4,
        "throughput": {
            "enabled_self_collisions": False,
            "solver_position_iteration_count": 4,
            "foot_track_pose": False,
            "foot_contact_update_period": 0.02,
            "replicate_physics": True,
        },
    })
    assert cfg.terrain_init_level == 4
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 4
    assert cfg.scene.contact_fr.track_pose is False and cfg.scene.contact_rr.update_period == 0.02
    assert cfg.scene.base_contact.update_period == pytest.approx(cfg.sim.dt)
    assert cfg.scene.replicate_physics is True


def test_checkpoint_contract_detection(tmp_path):
    old = tmp_path / "old_run"
    old.mkdir()
    (old / "resolved_training_config.yaml").write_text("protocol_version: E0-v1-46d\nenv: {seed: 1}\n")
    new = tmp_path / "new_run"
    new.mkdir()
    (new / "resolved_training_config.yaml").write_text(
        "protocol_version: E0-v2-46d\nenv: {action_contract: servo_range_v2}\n"
    )
    assert checkpoint_action_contract(old / "model_19999.pt") == "legacy_symmetric"
    assert checkpoint_action_contract(new / "model_999.pt") == "servo_range_v2"
    assert checkpoint_action_contract(None) == "servo_range_v2"


def test_v1_checkpoints_cannot_resume_under_v2(tmp_path):
    from omegaconf import DictConfig

    cfg = _env_cfg()
    agent = OmegaConf.to_container(load_config(E0).agent, resolve=True)
    v2 = build_training_contract("task", cfg, DictConfig(agent))
    cfg.action_contract = "legacy_symmetric"
    cfg.apply_action_contract()
    v1 = build_training_contract("task", cfg, DictConfig(agent))
    assert v1["sha256"] != v2["sha256"]
    save_training_contract(tmp_path, v1)
    with pytest.raises(ValueError, match="differs"):
        require_compatible_training_contract(tmp_path / "model_19999.pt", v2, allow_terrain_mix_transition=True)
