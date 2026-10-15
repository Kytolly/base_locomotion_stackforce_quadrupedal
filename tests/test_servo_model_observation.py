"""Deployable actor joint channels: servo-model hip estimate and commanded wheel velocity.

The robot computes the same estimate in firmware (docs/REAL_ROBOT_IO_SPEC.md);
``_firmware_estimate`` below is that scalar form, and the simulator's batched
``servo_model_step`` must agree with it step for step.
"""

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
CONTROL_DT = 0.02


def _firmware_estimate(estimate: float, target: float, speed: float, dt: float) -> float:
    step = speed * dt
    delta = target - estimate
    if delta > step:
        delta = step
    elif delta < -step:
        delta = -step
    return estimate + delta


def test_simulator_and_firmware_estimates_agree():
    generator = torch.Generator().manual_seed(0)
    targets = torch.rand(50, 8, generator=generator) * 3.0 - 1.5
    estimate = torch.zeros(8)
    scalar = [0.0] * 8
    max_step = servo.SERVO_MODEL_SPEED_RAD_S * CONTROL_DT
    for target in targets:
        previous = estimate
        estimate = servo.servo_model_step(estimate, target, max_step)
        scalar = [_firmware_estimate(e, float(t), servo.SERVO_MODEL_SPEED_RAD_S, CONTROL_DT) for e, t in zip(scalar, target)]
        assert estimate.tolist() == pytest.approx(scalar, abs=1e-6)
        assert ((estimate - previous).abs() <= max_step + 1e-6).all()


def test_estimate_reaches_a_held_target_at_servo_speed():
    max_step = servo.SERVO_MODEL_SPEED_RAD_S * CONTROL_DT
    estimate = torch.zeros(1)
    steps = 0
    while not torch.isclose(estimate, torch.ones(1)):
        estimate = servo.servo_model_step(estimate, torch.ones(1), max_step)
        steps += 1
    assert steps == 6  # 1 rad at 0.1746 rad per 20 ms step


def test_formal_config_uses_the_servo_model():
    config = load_config(E0)
    validate_training_config(config)
    assert config.env.policy_joint_observation == "servo_model"
    assert config.env.servo_model_delay_steps == 0


@pytest.mark.parametrize("override, message", [
    ("env.policy_joint_observation=encoder", "policy_joint_observation"),
    ("env.servo_model_delay_steps=-1", "servo_model_delay_steps"),
    ("env.action_contract=legacy_symmetric", "servo_range_v2"),
])
def test_validator_rejects_inconsistent_observation_settings(override, message):
    with pytest.raises(ValueError, match=message):
        validate_training_config(load_config(E0, [override]))


def _env_cfg():
    try:
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
            BaseLocomotionComplexEnvCfg,
        )

        return BaseLocomotionComplexEnvCfg()
    except Exception as error:  # pragma: no cover - depends on the Isaac Lab install
        pytest.skip(f"env config needs the Isaac Lab runtime here: {error}")


def test_env_options_switch_actor_channels_and_critic_truth():
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.complex_env_cfg import (
        apply_e0_env_options,
    )
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.observation.proprioception import (
        terms as proprioception,
    )

    cfg = _env_cfg()
    assert cfg.observations.policy.joint_pos.func is proprioception.active_joint_position
    assert cfg.observations.privileged.true_leg_joint_pos is None
    apply_e0_env_options(cfg, {"action_contract": "servo_range_v2", "policy_joint_observation": "servo_model",
                               "servo_model_delay_steps": 2})
    assert cfg.observations.policy.joint_pos.func is proprioception.servo_model_joint_position
    assert cfg.observations.policy.joint_vel.func is proprioception.servo_model_joint_velocity
    assert cfg.observations.privileged.true_leg_joint_pos is not None
    assert cfg.actions.leg_position.servo_model_delay_steps == 2
    apply_e0_env_options(cfg, {"action_contract": "servo_range_v2", "policy_joint_observation": "measured"})
    assert cfg.observations.policy.joint_pos.func is proprioception.active_joint_position
    assert cfg.observations.privileged.true_leg_joint_pos is None
