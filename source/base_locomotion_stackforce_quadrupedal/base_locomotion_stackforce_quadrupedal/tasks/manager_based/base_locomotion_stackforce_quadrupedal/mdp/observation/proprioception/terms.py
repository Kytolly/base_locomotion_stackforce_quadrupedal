"""Deployable 30-dimensional proprioceptive observation terms."""

from __future__ import annotations

import torch

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


# Deployable joint channels (``env.policy_joint_observation: servo_model``).  The
# robot's PWM hip servos report no position and the wheel feedback is incomplete,
# so the actor sees what the robot can compute itself (docs/REAL_ROBOT_IO_SPEC.md):
# the eight hips as the servo-model estimate kept by the leg action term, the
# wheels as their commanded velocity and no wheel angle.  Order and size match
# ``active_joint_position``/``active_joint_velocity`` (8 legs, then 4 wheels).


def _servo_terms(env, leg_term: str, wheel_term: str):
    manager = env.action_manager
    return manager.get_term(leg_term), manager.get_term(wheel_term)


def servo_model_joint_position(env, leg_term: str = "leg_position", wheel_term: str = "wheel_velocity") -> torch.Tensor:
    """Hip estimate relative to the stance; wheel angles are not observed (zeros)."""
    legs, wheels = _servo_terms(env, leg_term, wheel_term)
    return torch.cat((legs.servo_estimate - legs.stance, torch.zeros_like(wheels.processed_actions)), dim=-1)


def servo_model_joint_velocity(env, leg_term: str = "leg_position", wheel_term: str = "wheel_velocity") -> torch.Tensor:
    """Hip estimate velocity, then the commanded wheel velocities (rad/s)."""
    legs, wheels = _servo_terms(env, leg_term, wheel_term)
    return torch.cat((legs.servo_estimate_vel, wheels.processed_actions), dim=-1)
