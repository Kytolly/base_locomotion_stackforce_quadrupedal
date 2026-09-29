"""Deployable single-step memory terms for the first policy contract."""

from isaaclab.envs.mdp import last_action

from ...action import POLICY_ACTION_DIM

HISTORY_STEPS = 1
PREVIOUS_ACTION_DIM = POLICY_ACTION_DIM
HISTORY_DIM = HISTORY_STEPS * PREVIOUS_ACTION_DIM

previous_action = last_action
