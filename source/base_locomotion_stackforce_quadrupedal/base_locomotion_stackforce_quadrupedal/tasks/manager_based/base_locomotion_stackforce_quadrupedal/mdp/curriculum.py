"""Performance-driven terrain-level updates for complex locomotion training."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg


def terrain_levels_by_episode_performance(
    env, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Promote completed traversals and demote unsafe or very short episodes."""
    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("locomotion")[env_ids, :2]
    distance = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    completed = env.episode_length_buf[env_ids] > 0
    timeout = getattr(env, "reset_time_outs", torch.zeros_like(env.episode_length_buf))[env_ids]
    terminated = getattr(env, "reset_terminated", torch.zeros_like(env.episode_length_buf))[env_ids]
    required_distance = terrain.cfg.terrain_generator.size[0] / 2.0
    move_up = completed & timeout & (distance > required_distance)
    required_command_distance = (
        torch.linalg.vector_norm(command, dim=1) * env.max_episode_length_s * 0.5
    )
    move_down = completed & (terminated | (distance < required_command_distance))
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
