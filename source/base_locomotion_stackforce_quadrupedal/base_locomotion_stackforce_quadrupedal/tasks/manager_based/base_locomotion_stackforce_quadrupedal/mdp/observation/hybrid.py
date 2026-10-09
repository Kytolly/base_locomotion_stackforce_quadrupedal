"""Deployable contact, end-effector, and forward-terrain observations."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from ...env.robots import FOOT_LINKS

CONTACT_DIM = 4
FORCE_DIM = 4
ENDPOINT_POSITION_DIM = 12
ENDPOINT_VELOCITY_DIM = 12
TERRAIN_HEIGHT_DIM = 15
TERRAIN_VALID_DIM = 15
HYBRID_OBSERVATION_DIM = 62


def hybrid_terrain_pattern(cfg: PatternBaseCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the fixed 3-column x 5-row forward terrain contract."""
    x = torch.tensor((-0.30, 0.0, 0.30), device=device)
    y = torch.tensor((0.15, 0.35, 0.55, 0.75, 0.95), device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    starts = torch.zeros((grid_x.numel(), 3), device=device)
    starts[:, 0] = grid_x.flatten()
    starts[:, 1] = grid_y.flatten()
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


@configclass
class HybridTerrainPatternCfg(PatternBaseCfg):
    """Configuration marker for the fixed 15-ray terrain pattern."""

    func = hybrid_terrain_pattern


def _foot_ids(env, asset_cfg: SceneEntityCfg) -> tuple[object, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(list(FOOT_LINKS), preserve_order=True)
    return asset, body_ids


def contact_flags(env) -> torch.Tensor:
    """Return filtered contact flags in FR, FL, RL, RR order."""
    flags = []
    for name in ("contact_fr", "contact_fl", "contact_rl", "contact_rr"):
        sensor = env.scene[name]
        history = sensor.data.net_forces_w_history.torch[..., 2].abs().squeeze(-1)
        flags.append((history > 1.0).float().mean(dim=1) > 0.5)
    return torch.stack(flags, dim=-1).to(env.device)


def normal_forces(env) -> torch.Tensor:
    """Return clipped, normalized normal-force estimates in FR, FL, RL, RR order."""
    values = []
    for name in ("contact_fr", "contact_fl", "contact_rl", "contact_rr"):
        sensor = env.scene[name]
        force = sensor.data.net_forces_w.torch[:, 0, 2].abs()
        values.append(torch.clamp(force / 20.0, 0.0, 1.0))
    return torch.stack(values, dim=-1)


def _yaw_inverse(quat: torch.Tensor) -> torch.Tensor:
    """Keep only base yaw before rotating world-frame vectors."""
    yaw = torch.atan2(2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                      1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
    half = -0.5 * yaw
    result = torch.zeros_like(quat)
    result[:, 0] = torch.cos(half)
    result[:, 3] = torch.sin(half)
    return result


def _expand_yaw_inverse(quat: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat one base-frame yaw quaternion for each endpoint row."""
    return _yaw_inverse(quat).unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)


def endpoint_positions(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset, body_ids = _foot_ids(env, asset_cfg)
    relative = asset.data.body_pos_w[:, body_ids] - asset.data.root_pos_w[:, None, :]
    aligned = quat_apply_inverse(
        _expand_yaw_inverse(asset.data.root_quat_w, relative.shape[1]), relative.reshape(-1, 3)
    )
    return aligned.reshape(relative.shape[0], -1)


def endpoint_velocities(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset, body_ids = _foot_ids(env, asset_cfg)
    relative = asset.data.body_pos_w[:, body_ids] - asset.data.root_pos_w[:, None, :]
    endpoint_linear_velocity = asset.data.body_lin_vel_w[:, body_ids]
    base_motion = asset.data.root_lin_vel_w[:, None, :] + torch.cross(
        asset.data.root_ang_vel_w[:, None, :], relative, dim=-1
    )
    relative_velocity = endpoint_linear_velocity - base_motion
    aligned = quat_apply_inverse(
        _expand_yaw_inverse(asset.data.root_quat_w, relative_velocity.shape[1]),
        relative_velocity.reshape(-1, 3),
    )
    return aligned.reshape(relative_velocity.shape[0], -1)


def forward_terrain_height(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("forward_terrain")) -> torch.Tensor:
    hits = env.scene[sensor_cfg.name].data.ray_hits_w.torch
    root_z = env.scene["robot"].data.root_pos_w[:, 2:3]
    finite = torch.isfinite(hits).all(dim=-1)
    values = torch.where(finite, torch.clamp((hits[..., 2] - root_z) / 0.30, -1.0, 1.0), torch.zeros_like(hits[..., 2]))
    return values


def forward_terrain_valid(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("forward_terrain")) -> torch.Tensor:
    hits = env.scene[sensor_cfg.name].data.ray_hits_w.torch
    return torch.isfinite(hits).all(dim=-1).to(dtype=torch.float32)
