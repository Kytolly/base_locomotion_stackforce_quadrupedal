"""Simulator-only current-state observations for an asymmetric critic."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import base_lin_vel, joint_effort, joint_pos_rel, joint_vel_rel
from isaaclab.managers import SceneEntityCfg

from ...support import body_height_above_support


BASE_LINEAR_VELOCITY_DIM = 3
BASE_HEIGHT_DIM = 1
APPLIED_JOINT_TORQUE_DIM = 12
LEG_JOINT_STATE_DIM = 16  # true hip positions and velocities (the actor sees the servo model)
PRIVILEGED_DIM = BASE_LINEAR_VELOCITY_DIM + BASE_HEIGHT_DIM + APPLIED_JOINT_TORQUE_DIM
SERVO_MODEL_PRIVILEGED_DIM = PRIVILEGED_DIM + LEG_JOINT_STATE_DIM

base_linear_velocity = base_lin_vel
applied_joint_torque = joint_effort
true_joint_position = joint_pos_rel
true_joint_velocity = joint_vel_rel


base_height_above_support = body_height_above_support
