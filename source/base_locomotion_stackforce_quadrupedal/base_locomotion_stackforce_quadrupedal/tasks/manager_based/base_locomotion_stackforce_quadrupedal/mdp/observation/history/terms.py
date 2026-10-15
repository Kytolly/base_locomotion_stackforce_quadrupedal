"""Deployable single-step memory terms for the first policy contract."""

from __future__ import annotations

import torch

from ...action import POLICY_ACTION_DIM

HISTORY_STEPS = 1
PREVIOUS_ACTION_DIM = POLICY_ACTION_DIM
HISTORY_DIM = HISTORY_STEPS * PREVIOUS_ACTION_DIM



def previous_action(env) -> torch.Tensor:
    """Last applied action: every term's ``raw_actions`` in action-manager order.

    Equals ``isaaclab.envs.mdp.last_action`` unless a term rewrites its raw
    actions (the leg slew limit stores the limited command).
    """
    manager = env.action_manager
    return torch.cat([manager.get_term(name).raw_actions for name in manager.active_terms], dim=-1)
