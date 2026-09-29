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


@configclass
class RobotActionCfg:
    """Define the 12-D action space from which RSL-RL derives the Actor output width."""

    leg_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEG_JOINTS),
        scale=LEG_POSITION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": LEG_POSITION_TARGET_CLIP},
    )
    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(WHEEL_JOINTS),
        scale=WHEEL_VELOCITY_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": WHEEL_VELOCITY_TARGET_CLIP},
    )
