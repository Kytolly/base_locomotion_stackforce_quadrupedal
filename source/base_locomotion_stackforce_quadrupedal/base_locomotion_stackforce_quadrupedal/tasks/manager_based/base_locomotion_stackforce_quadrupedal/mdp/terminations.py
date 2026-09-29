"""Termination terms for the StackForce locomotion environment."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def base_height_failure(env, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height < minimum_height
