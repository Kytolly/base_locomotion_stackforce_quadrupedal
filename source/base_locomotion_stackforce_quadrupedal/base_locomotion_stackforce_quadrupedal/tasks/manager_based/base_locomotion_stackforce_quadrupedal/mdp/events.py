"""Reset events for the StackForce locomotion environment."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .policy import LOCOMOTION_COMMAND_NAME


def reset_root_to_default(
    env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def set_fixed_benchmark_command(
    env,
    env_ids: torch.Tensor,
    forward_velocity: float = 0.28,
    body_height: float = 0.105,
) -> None:
    """Set a fixed straight-line command for a benchmark episode."""
    command = torch.zeros((len(env_ids), 4), device=env.device)
    command[:, 0] = forward_velocity
    command[:, 3] = body_height
    env.command_manager.get_term(LOCOMOTION_COMMAND_NAME).set_command(command, env_ids)
