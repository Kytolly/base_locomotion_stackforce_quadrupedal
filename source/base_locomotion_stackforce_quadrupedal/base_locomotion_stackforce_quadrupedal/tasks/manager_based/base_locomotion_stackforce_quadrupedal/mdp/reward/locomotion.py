"""Decomposed rewards for command-conditioned StackForce locomotion."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from ..policy.commands import (
    BODY_HEIGHT,
    FORWARD,
    LATERAL,
    LOCOMOTION_COMMAND_NAME,
    YAW_RATE,
)
from ..support import local_support_height


def _command(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _exp_tracking(error: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError(
            f"Tracking standard deviation must be positive, received {std}."
        )
    return torch.exp(-torch.square(error) / (std * std))


def track_forward_velocity_exp(
    env,
    std: float,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = _command(env, command_name)[:, FORWARD] - asset.data.root_lin_vel_b[:, 1]
    return _exp_tracking(error, std)


def track_lateral_velocity_exp(
    env,
    std: float,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = _command(env, command_name)[:, LATERAL] - asset.data.root_lin_vel_b[:, 0]
    return _exp_tracking(error, std)


def track_yaw_rate_exp(
    env,
    std: float,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = _command(env, command_name)[:, YAW_RATE] - asset.data.root_ang_vel_b[:, 2]
    return _exp_tracking(error, std)


def track_body_height_exp(
    env,
    std: float,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    support_height, _ = local_support_height(env, asset_cfg=asset_cfg)
    height = asset.data.root_pos_w[:, 2] - support_height
    error = _command(env, command_name)[:, BODY_HEIGHT] - height
    return _exp_tracking(error, std)


def directed_planar_progress(
    env,
    command_deadband: float = 0.05,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward motion in the commanded planar direction without rewarding overspeed."""
    command = _command(env, command_name)[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / speed.clamp_min(command_deadband).unsqueeze(1)
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, [1, 0]]
    progress_ratio = torch.sum(velocity * direction, dim=1) / speed.clamp_min(command_deadband)
    return torch.where(speed >= command_deadband, progress_ratio.clamp(0.0, 1.0), 0.0)


def planar_velocity_error_l2(
    env,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar velocity error even when exponential tracking has saturated."""
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, [1, 0]]
    return torch.sum(torch.square(velocity - _command(env, command_name)[:, :2]), dim=1)


def wrong_way_velocity_ratio_l2(
    env,
    command_deadband: float = 0.05,
    command_name: str = LOCOMOTION_COMMAND_NAME,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize velocity opposite to a non-zero planar command."""
    command = _command(env, command_name)[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / speed.clamp_min(command_deadband).unsqueeze(1)
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, [1, 0]]
    reverse_ratio = torch.relu(-torch.sum(velocity * direction, dim=1) / speed.clamp_min(command_deadband))
    penalty = torch.square(reverse_ratio.clamp(max=2.0))
    return torch.where(speed >= command_deadband, penalty, 0.0)


def low_body_height_margin_l2(
    env,
    warning_height: float = 0.08,
    margin: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a continuous warning before the hard low-body termination."""
    if warning_height <= 0.0 or margin <= 0.0:
        raise ValueError("warning_height and margin must be positive.")
    asset = env.scene[asset_cfg.name]
    support_height, support_valid = local_support_height(env, asset_cfg=asset_cfg)
    height = asset.data.root_pos_w[:, 2] - support_height
    penalty = torch.square(torch.relu((warning_height - height) / margin))
    return torch.where(support_valid, penalty, torch.ones_like(penalty))


def orientation_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize roll and pitch using horizontal projected-gravity components."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def mechanical_power(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return instantaneous active-joint mechanical power in watts."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def action_saturation(env, threshold: float = 0.95) -> torch.Tensor:
    """Penalize normalized actions approaching the policy clipping boundary."""
    if not 0.0 <= threshold < 1.0:
        raise ValueError(
            f"Action saturation threshold must be in [0, 1), received {threshold}."
        )
    excess = torch.relu(torch.abs(env.action_manager.action) - threshold)
    return torch.mean(torch.square(excess / (1.0 - threshold)), dim=1)
