"""Flat Actor-input layout assembled by the policy observation group."""

from __future__ import annotations

import torch

from ..observation.history import HISTORY_DIM
from ..observation.proprioception import PROPRIOCEPTION_DIM
from .commands import LOCOMOTION_COMMAND_DIM


ACTOR_INPUT_DIM = LOCOMOTION_COMMAND_DIM + PROPRIOCEPTION_DIM + HISTORY_DIM
COMMAND_SLICE = slice(0, LOCOMOTION_COMMAND_DIM)
PROPRIOCEPTION_SLICE = slice(
    LOCOMOTION_COMMAND_DIM, LOCOMOTION_COMMAND_DIM + PROPRIOCEPTION_DIM
)
HISTORY_SLICE = slice(LOCOMOTION_COMMAND_DIM + PROPRIOCEPTION_DIM, ACTOR_INPUT_DIM)


def split_actor_input(
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flat Actor observation into command, proprioception, and history."""
    if observation.ndim == 0 or observation.shape[-1] != ACTOR_INPUT_DIM:
        raise ValueError(
            f"Expected Actor input shape (..., {ACTOR_INPUT_DIM}), received {tuple(observation.shape)}."
        )
    return (
        observation[..., COMMAND_SLICE],
        observation[..., PROPRIOCEPTION_SLICE],
        observation[..., HISTORY_SLICE],
    )
