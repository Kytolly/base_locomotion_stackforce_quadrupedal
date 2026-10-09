"""Motion-family coverage, promotion, and rehearsal curriculum."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg


MOTION_FAMILY_COUNT = 7


def motion_family_rehearsal(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    success_tracking_rmse: float = 0.30,
    success_yaw_rmse: float = 0.35,
    sampling_floor: float = 0.05,
) -> torch.Tensor:
    """Maintain per-family success statistics and non-zero sampling floors.

    The sampler remains a single 46D command stream. This term only changes
    family probabilities and records coverage; it never adds an observation.
    """
    history = getattr(env, "motion_history", None)
    command_term = env.command_manager.get_term("locomotion")
    family = getattr(command_term, "motion_family", None)
    if history is None or family is None:
        return torch.zeros((), device=env.device)

    if not hasattr(env, "motion_family_stats"):
        env.motion_family_stats = torch.zeros((MOTION_FAMILY_COUNT, 3), device=env.device)
    family = family.to(dtype=torch.long)
    completed = history[env_ids, 4] > 0
    tracking = torch.sqrt(
        history[env_ids, 2] / history[env_ids, 4].clamp_min(1.0)
    )
    yaw = torch.sqrt(history[env_ids, 3] / history[env_ids, 4].clamp_min(1.0))
    terminated = getattr(env, "reset_terminated", torch.zeros_like(env.episode_length_buf))[env_ids]
    success = completed & ~terminated & (tracking <= success_tracking_rmse) & (yaw <= success_yaw_rmse)
    selected = family[env_ids].clamp(0, MOTION_FAMILY_COUNT - 1)
    for index in range(MOTION_FAMILY_COUNT):
        mask = selected == index
        env.motion_family_stats[index, 0] += mask.float().sum()
        env.motion_family_stats[index, 1] += (mask & success).float().sum()
        env.motion_family_stats[index, 2] = torch.where(
            env.motion_family_stats[index, 0] > 0,
            env.motion_family_stats[index, 1] / env.motion_family_stats[index, 0],
            env.motion_family_stats[index, 2],
        )

    probabilities = getattr(command_term, "sampling_probabilities", None)
    if probabilities is not None:
        success_rates = env.motion_family_stats[:, 2].clamp(0.0, 1.0)
        adaptive = command_term._base_sampling_probabilities * (1.0 + 0.5 * (1.0 - success_rates))
        command_term.set_sampling_probabilities(adaptive, floor=sampling_floor)
    return success.float().mean()


__all__ = ["MOTION_FAMILY_COUNT", "motion_family_rehearsal"]
