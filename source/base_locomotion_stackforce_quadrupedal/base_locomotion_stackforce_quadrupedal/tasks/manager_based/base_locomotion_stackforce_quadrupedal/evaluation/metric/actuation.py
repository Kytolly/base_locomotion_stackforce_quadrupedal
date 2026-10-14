"""Per-step actuation diagnostics for G1 (docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md).

Each function maps per-environment tensors of one control step to one value per
environment; ``LocomotionEpisodeMetrics`` averages or maximises them over the
episode.  Pure torch, so the definitions are unit-tested without Isaac Sim.
"""

from __future__ import annotations

import torch


SATURATION_THRESHOLD = 0.95
TORQUE_SATURATION_FRACTION = 0.95
DS041MG_RATED_TORQUE_NM = 0.127


def saturation_fraction(raw_action: torch.Tensor, threshold: float = SATURATION_THRESHOLD) -> torch.Tensor:
    """Fraction of dimensions with ``|raw| > threshold``."""
    return (raw_action.abs() > threshold).float().mean(dim=-1)


def torque_saturation_fraction(
    torque: torch.Tensor, effort_limit: torch.Tensor, fraction: float = TORQUE_SATURATION_FRACTION
) -> torch.Tensor:
    """Fraction of joints with ``|applied torque| >= fraction * effort limit``."""
    return (torque.abs() >= fraction * effort_limit.abs()).float().mean(dim=-1)


def over_rated_fraction(torque: torch.Tensor, rated_nm: float = DS041MG_RATED_TORQUE_NM) -> torch.Tensor:
    """Fraction of joints above the servo's continuous (rated) torque."""
    return (torque.abs() > rated_nm).float().mean(dim=-1)


def tracking_error_sq(target: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Mean squared target-minus-position error over joints (RMS after episode averaging)."""
    return torch.square(target - actual).mean(dim=-1)


def closure_residual_mm(
    closure_point: torch.Tensor, wheel_point: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    """Largest loop-closure gap over legs, perpendicular to the closure axis, in mm.

    ``closure_point`` / ``wheel_point`` are ``(num_envs, legs, 3)`` world positions
    of the inner-chain closure frame and the outer-chain wheel frame; ``axis``
    is the unit closure axis in world coordinates.  The two frames are offset
    along the axis by design (about 45 mm), so only the radial gap is a defect.
    """
    delta = closure_point - wheel_point
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    radial = delta - (delta * axis).sum(dim=-1, keepdim=True) * axis
    return radial.norm(dim=-1).amax(dim=-1) * 1000.0


class RollingMean:
    """Per-environment moving average over a fixed number of control steps."""

    def __init__(self, num_envs: int, width: int, window: int, device) -> None:
        self.buffer = torch.zeros((num_envs, window, width), device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.window = window

    def reset(self, env_ids) -> None:
        self.buffer[env_ids] = 0.0
        self.count[env_ids] = 0

    def push(self, values: torch.Tensor) -> torch.Tensor:
        """Insert one step and return the current window mean (over the filled part)."""
        slot = self.count % self.window
        self.buffer[torch.arange(len(slot), device=slot.device), slot] = values
        self.count += 1
        filled = self.count.clamp(max=self.window).unsqueeze(-1).float()
        return self.buffer.sum(dim=1) / filled


__all__ = [
    "DS041MG_RATED_TORQUE_NM",
    "RollingMean",
    "SATURATION_THRESHOLD",
    "TORQUE_SATURATION_FRACTION",
    "closure_residual_mm",
    "over_rated_fraction",
    "saturation_fraction",
    "torque_saturation_fraction",
    "tracking_error_sq",
]
