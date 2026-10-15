"""Formal E0-v2 training settings chosen by the group B/E/G short runs (docs/REWARD_CURRICULUM_FIX_PLAN.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

from base_locomotion_stackforce_quadrupedal.training import apply_config, load_config, validate_training_config

ROOT = Path(__file__).resolve().parents[1]
E0 = ROOT / "configs/train/base_locomotion_e0_46d.yaml"


def test_formal_config_values():
    config = load_config(E0)
    validate_training_config(config)
    assert config.agent.actor.obs_normalization is True
    assert config.agent.critic.obs_normalization is True
    assert list(config.agent.actor.distribution_cfg.init_std)[8:] == [0.1] * 4
    assert config.rewards.action_rate.weight == pytest.approx(-0.05)
    assert config.rewards.joint_velocity_limit.weight == pytest.approx(-0.2)
    assert config.rewards.track_forward_velocity.params.std == pytest.approx(0.15)
    # Open decision: no slew limit in the default long run; actor sees what the robot can compute.
    assert config.env.leg_target_max_step_rad is None
    assert config.env.policy_joint_observation == "servo_model"


def test_reward_overrides_apply_to_the_env_config():
    try:
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
            BaseLocomotionComplexEnvCfg,
        )

        cfg = BaseLocomotionComplexEnvCfg()
    except Exception as error:  # pragma: no cover - depends on the Isaac Lab install
        pytest.skip(f"env config needs the Isaac Lab runtime here: {error}")
    apply_config(cfg.rewards, load_config(E0).rewards, "rewards")  # the train.py path, strict on names
    assert cfg.rewards.action_rate.weight == pytest.approx(-0.05)
    assert cfg.rewards.joint_velocity_limit.weight == pytest.approx(-0.2)
    assert cfg.rewards.track_forward_velocity.params["std"] == pytest.approx(0.15)
