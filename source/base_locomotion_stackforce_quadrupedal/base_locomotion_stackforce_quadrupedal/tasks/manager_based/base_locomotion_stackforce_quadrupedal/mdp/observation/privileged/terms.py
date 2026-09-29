"""Simulator-only current-state observations for an asymmetric critic."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import base_lin_vel, joint_effort
from isaaclab.managers import SceneEntityCfg


BASE_LINEAR_VELOCITY_DIM = 3
BASE_HEIGHT_DIM = 1
APPLIED_JOINT_TORQUE_DIM = 12
PRIVILEGED_DIM = BASE_LINEAR_VELOCITY_DIM + BASE_HEIGHT_DIM + APPLIED_JOINT_TORQUE_DIM

base_linear_velocity = base_lin_vel
applied_joint_torque = joint_effort


def base_height_above_env_origin(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return true root height relative to each terrain environment origin."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height.unsqueeze(-1)
