"""Terrain-difficulty curriculum for the complex locomotion task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg


def terrain_levels_by_episode_performance(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tracking_rmse_range: tuple[float, float] = (0.30, 0.18),
    yaw_rmse_range: tuple[float, float] = (0.35, 0.20),
    progress_ratio_range: tuple[float, float] = (0.60, 0.80),
    demotion_progress_ratio: float = 0.25,
) -> torch.Tensor:
    """Promote viable episodes and rehearse old terrain levels.

    Promotion is based on command-aligned progress and tracking, not distance
    from the reset origin. Missing recorder data leaves the terrain unchanged.
    """
    terrain = env.scene.terrain
    if not hasattr(env, "motion_history"):
        return torch.mean(terrain.terrain_levels.float())
    history = env.motion_history[env_ids]
    timeout = getattr(env, "reset_time_outs", torch.zeros_like(env.episode_length_buf))[env_ids]
    terminated = getattr(env, "reset_terminated", torch.zeros_like(env.episode_length_buf))[env_ids]
    move_up, move_down = curriculum_decisions(
        history,
        timeout,
        terminated,
        terrain_levels=terrain.terrain_levels[env_ids],
        max_terrain_level=terrain.max_terrain_level,
        tracking_rmse_range=tracking_rmse_range,
        yaw_rmse_range=yaw_rmse_range,
        progress_ratio_range=progress_ratio_range,
        demotion_progress_ratio=demotion_progress_ratio,
    )
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def curriculum_decisions(
    history,
    timeout,
    terminated,
    *,
    terrain_levels=None,
    max_terrain_level: int = 8,
    tracking_rmse_range: tuple[float, float] = (0.30, 0.18),
    yaw_rmse_range: tuple[float, float] = (0.35, 0.20),
    progress_ratio_range: tuple[float, float] = (0.60, 0.80),
    demotion_progress_ratio: float = 0.25,
):
    """Return terrain promotion and demotion masks for one episode batch."""
    requested, directed, error_sq, yaw_error_sq, steps = history.unbind(-1)
    complete = steps > 0
    if terrain_levels is None:
        terrain_levels = torch.zeros_like(steps)
    level_fraction = terrain_levels.float() / max(max_terrain_level - 1, 1)

    def interpolate(bounds: tuple[float, float]) -> torch.Tensor:
        return bounds[0] + level_fraction * (bounds[1] - bounds[0])

    tracking = torch.sqrt(error_sq / steps.clamp_min(1)) <= interpolate(tracking_rmse_range)
    turning = torch.sqrt(yaw_error_sq / steps.clamp_min(1)) <= interpolate(yaw_rmse_range)
    moving = requested >= 1.0e-6
    progress_ratio = directed / requested.clamp_min(1.0e-6)
    move_up = (
        complete
        & moving
        & timeout
        & ~terminated
        & tracking
        & turning
        & (progress_ratio >= interpolate(progress_ratio_range))
    )
    move_down = complete & (
        terminated | (moving & (progress_ratio < demotion_progress_ratio))
    ) & ~move_up
    return move_up, move_down


__all__ = ["curriculum_decisions", "terrain_levels_by_episode_performance"]
