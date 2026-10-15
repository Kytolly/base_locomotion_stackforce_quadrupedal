"""Leg target slew limit: an optional switch, off by default and in the formal E0-v2 config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from base_locomotion_stackforce_quadrupedal.training import load_config, validate_training_config
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (
    servo_range as servo,
)

ROOT = Path(__file__).resolve().parents[1]
E0 = ROOT / "configs/train/base_locomotion_e0_46d.yaml"
SERVO_STEP_RAD = 8.73 * 0.02  # DS041MG no-load speed x control period


def test_target_to_raw_inverts_the_piecewise_map():
    lower, default, upper = (torch.tensor(v) for v in ([-2.0, -1.0], [0.5, 0.0], [1.0, 3.0]))
    raw = torch.tensor([[-1.0, -1.0], [-0.3, 0.4], [0.0, 0.0], [0.7, 1.0]])
    target = servo.raw_to_joint_target(raw, lower, default, upper)
    assert servo.joint_target_to_raw(target, lower, default, upper) == pytest.approx(raw, abs=1e-6)


def test_formal_config_leaves_the_limit_off_until_rollouts_justify_it():
    config = load_config(E0)
    validate_training_config(config)
    assert config.env.leg_target_max_step_rad is None
    enabled = load_config(E0, [f"env.leg_target_max_step_rad={SERVO_STEP_RAD}"])
    validate_training_config(enabled)
    assert enabled.env.leg_target_max_step_rad == pytest.approx(SERVO_STEP_RAD)


def test_validator_rejects_a_non_positive_limit():
    config = load_config(E0, ["env.leg_target_max_step_rad=0"])
    with pytest.raises(ValueError, match="leg_target_max_step_rad"):
        validate_training_config(config)


def _env_cfg():
    try:
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
            BaseLocomotionComplexEnvCfg,
        )

        return BaseLocomotionComplexEnvCfg()
    except Exception as error:  # pragma: no cover - depends on the Isaac Lab install
        pytest.skip(f"env config needs the Isaac Lab runtime here: {error}")


def test_limit_is_off_by_default_and_set_from_env_options():
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
        apply_e0_env_options,
    )

    cfg = _env_cfg()
    assert cfg.actions.leg_position.max_target_step_rad is None
    apply_e0_env_options(cfg, {"action_contract": "servo_range_v2", "leg_target_max_step_rad": 0.175})
    assert cfg.actions.leg_position.max_target_step_rad == pytest.approx(0.175)
    apply_e0_env_options(cfg, {"action_contract": "servo_range_v2", "leg_target_max_step_rad": None})
    assert cfg.actions.leg_position.max_target_step_rad is None
