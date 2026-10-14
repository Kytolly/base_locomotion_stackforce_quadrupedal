"""Twelve-dimensional policy action contract and Isaac Lab decoder config."""

from __future__ import annotations

import torch

from isaaclab.envs import mdp
from isaaclab.utils.configclass import configclass

from ...env.robots import LEG_JOINTS, WHEEL_JOINTS


LEG_ACTION_DIM = 8
WHEEL_ACTION_DIM = 4
POLICY_ACTION_DIM = LEG_ACTION_DIM + WHEEL_ACTION_DIM
LEG_ACTION_SLICE = slice(0, LEG_ACTION_DIM)
WHEEL_ACTION_SLICE = slice(LEG_ACTION_DIM, POLICY_ACTION_DIM)
POLICY_ACTION_JOINTS = LEG_JOINTS + WHEEL_JOINTS

RAW_ACTION_CLIP = (-1.0, 1.0)
LEG_POSITION_SCALE = 0.5
WHEEL_VELOCITY_SCALE = 20.0
LEG_POSITION_TARGET_CLIP = (-LEG_POSITION_SCALE, LEG_POSITION_SCALE)
WHEEL_VELOCITY_TARGET_CLIP = (-WHEEL_VELOCITY_SCALE, WHEEL_VELOCITY_SCALE)


def split_policy_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a policy action into the ordered leg and wheel segments.

    Raises:
        ValueError: If the trailing action dimension is not exactly 12.
    """
    if action.ndim == 0 or action.shape[-1] != POLICY_ACTION_DIM:
        shape = tuple(action.shape)
        raise ValueError(f"Expected policy action shape (..., {POLICY_ACTION_DIM}), received {shape}.")
    return action[..., LEG_ACTION_SLICE], action[..., WHEEL_ACTION_SLICE]


def legacy_leg_position_cfg() -> mdp.JointPositionActionCfg:
    """E0-v1 legs: symmetric +-0.5 rad around a zero stance (P0 control, old checkpoints)."""
    return mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEG_JOINTS),
        scale=LEG_POSITION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": LEG_POSITION_TARGET_CLIP},
    )


def servo_leg_position_cfg(contract: dict | None = None):
    """E0-v2 legs: raw 0 -> stance, raw +-1 -> per-joint DS041MG 0.9-travel limits."""
    from .asymmetric import AsymmetricJointPositionActionCfg
    from .servo_range import leg_joint_ranges, load_servo_contract

    contract = contract or load_servo_contract()
    lower, _, upper = leg_joint_ranges(LEG_JOINTS, contract)
    return AsymmetricJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEG_JOINTS),
        use_default_offset=True,
        preserve_order=True,
        lower=dict(zip(LEG_JOINTS, lower)),
        upper=dict(zip(LEG_JOINTS, upper)),
        projection=dict(contract.get("projection", {})),
    )


@configclass
class RobotActionCfg:
    """Define the 12-D action space from which RSL-RL derives the Actor output width."""

    # The task config replaces this with ``servo_leg_position_cfg()`` for E0-v2
    # (``BaseLocomotionComplexEnvCfg.apply_action_contract``).
    leg_position = legacy_leg_position_cfg()
    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(WHEEL_JOINTS),
        scale=WHEEL_VELOCITY_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": WHEEL_VELOCITY_TARGET_CLIP},
    )
