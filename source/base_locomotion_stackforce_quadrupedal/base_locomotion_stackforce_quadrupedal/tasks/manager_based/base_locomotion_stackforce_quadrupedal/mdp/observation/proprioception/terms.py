"""Deployable 30-dimensional proprioceptive observation terms."""

from isaaclab.envs.mdp import base_ang_vel, joint_pos_rel, joint_vel_rel, projected_gravity


BASE_ANGULAR_VELOCITY_DIM = 3
PROJECTED_GRAVITY_DIM = 3
ACTIVE_JOINT_POSITION_DIM = 12
ACTIVE_JOINT_VELOCITY_DIM = 12
PROPRIOCEPTION_DIM = (
    BASE_ANGULAR_VELOCITY_DIM
    + PROJECTED_GRAVITY_DIM
    + ACTIVE_JOINT_POSITION_DIM
    + ACTIVE_JOINT_VELOCITY_DIM
)

base_angular_velocity = base_ang_vel
active_joint_position = joint_pos_rel
active_joint_velocity = joint_vel_rel
