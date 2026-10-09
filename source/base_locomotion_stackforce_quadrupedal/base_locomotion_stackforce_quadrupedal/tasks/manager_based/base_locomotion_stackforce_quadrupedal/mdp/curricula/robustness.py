"""Robustness-bin bookkeeping kept separate from terrain progression."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def robustness_bin_schedule(
    env,
    env_ids: Sequence[int],
    nominal_level: int = 0,
    moderate_level: int = 2,
    stress_level: int = 5,
) -> torch.Tensor:
    """Record nominal/moderate/mixed robustness bins without leaking them to Actor."""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels[env_ids]
    bins = torch.full_like(levels, 1, dtype=torch.long)
    bins[levels <= nominal_level] = 0
    bins[levels >= stress_level] = 3
    if not hasattr(env, "robustness_bins"):
        env.robustness_bins = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.robustness_bins[env_ids] = bins
    return bins.float().mean()


__all__ = ["robustness_bin_schedule"]
