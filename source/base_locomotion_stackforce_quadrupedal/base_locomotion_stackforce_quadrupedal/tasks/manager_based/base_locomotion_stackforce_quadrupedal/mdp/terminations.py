"""Termination terms for the StackForce locomotion environment."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .support import local_support_height


def base_height_failure(env, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    support_height, support_valid = local_support_height(env, asset_cfg=asset_cfg)
    height = asset.data.root_pos_w[:, 2] - support_height
    return (height < minimum_height) | ~support_valid


def excessive_tilt(
    env, maximum_tilt_rad: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    if not 0.0 < maximum_tilt_rad < torch.pi / 2:
        raise ValueError("maximum_tilt_rad must be between 0 and pi/2.")
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
    return tilt > maximum_tilt_rad


def base_collision(
    env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    peak_force = torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1).amax(dim=-1)
    return peak_force > threshold
