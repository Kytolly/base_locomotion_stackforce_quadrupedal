"""Shared local support and wheel-contact queries for training and evaluation."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from ..env.robots import FOOT_LINKS

WHEEL_RADIUS_M = 0.033


def local_support_height(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate support height from finite scanner hits near the robot base."""
    asset = env.scene[asset_cfg.name]
    root = asset.data.root_pos_w
    foot_ids, _ = asset.find_bodies(list(FOOT_LINKS), preserve_order=True)
    contact = wheel_contact_state(env)
    wheel_support = asset.data.body_pos_w[:, foot_ids, 2] - WHEEL_RADIUS_M
    wheel_support = torch.where(contact, wheel_support, torch.nan)
    wheel_height = torch.nanmedian(wheel_support, dim=1).values
    wheel_valid = contact.sum(dim=1) >= 2

    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    if hasattr(env, "benchmark_track_parameters"):
        from base_locomotion_stackforce_quadrupedal.benchmark.surface import track_surface_height

        # The scanner targets the ground plane; replace its Z with the actual
        # static track surface, including Cylinder/Cube primitives.
        hits = hits.clone()
        hits[..., 2] = track_surface_height(env.benchmark_track_parameters, hits[..., :2])
    ray_valid = torch.isfinite(hits).all(dim=-1) & (hits[..., 2] < root[:, 2:3] + 0.15)
    ray_values = torch.where(ray_valid, hits[..., 2], torch.nan)
    ray_height = torch.nanmedian(ray_values, dim=1).values
    ray_valid = torch.isfinite(ray_height)

    valid_support = wheel_valid | ray_valid
    height = torch.where(wheel_valid, wheel_height, ray_height)
    fallback = env.scene.env_origins[:, 2]
    height = torch.where(valid_support, height, fallback)
    return height, valid_support


def body_height_above_support(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    support, _ = local_support_height(env, sensor_cfg, asset_cfg)
    root = env.scene[asset_cfg.name].data.root_pos_w
    return (root[:, 2] - support).unsqueeze(-1)


def wheel_contact_state(
    env,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return contact flags in FR, FL, RL, RR body order."""
    sensor_names = ("contact_fr", "contact_fl", "contact_rl", "contact_rr")
    flags = []
    for name, body_name in zip(sensor_names, FOOT_LINKS):
        sensor = env.scene[name]
        if sensor.body_names != [body_name]:
            raise RuntimeError(f"{name} must bind only {body_name}, received {sensor.body_names}.")
        force = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1)
        flags.append(force[:, 0] > threshold)
    return torch.stack(flags, dim=1)
