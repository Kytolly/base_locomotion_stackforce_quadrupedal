"""Pure tensor checks for the Actor input and Robot action interfaces."""

import pytest
import torch

from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (
    LEG_ACTION_DIM,
    POLICY_ACTION_DIM,
    WHEEL_ACTION_DIM,
    split_policy_action,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.observation.history import (
    HISTORY_DIM,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.observation.proprioception import (
    PROPRIOCEPTION_DIM,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy import (
    ACTOR_INPUT_DIM,
    LOCOMOTION_COMMAND_DIM,
    split_actor_input,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy.commands import compose_velocity_command
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy.commands import LocomotionCommandCfg


def test_actor_input_segments_cover_the_flat_observation() -> None:
    observation = torch.arange(ACTOR_INPUT_DIM).reshape(1, -1)

    command, proprioception, history = split_actor_input(observation)

    assert command.shape == (1, LOCOMOTION_COMMAND_DIM)
    assert proprioception.shape == (1, PROPRIOCEPTION_DIM)
    assert history.shape == (1, HISTORY_DIM)
    assert torch.equal(
        torch.cat((command, proprioception, history), dim=-1), observation
    )


def test_robot_action_segments_cover_the_policy_output() -> None:
    action = torch.arange(POLICY_ACTION_DIM).reshape(1, -1)

    leg_action, wheel_action = split_policy_action(action)

    assert leg_action.shape == (1, LEG_ACTION_DIM)
    assert wheel_action.shape == (1, WHEEL_ACTION_DIM)
    assert torch.equal(torch.cat((leg_action, wheel_action), dim=-1), action)


@pytest.mark.parametrize(
    ("splitter", "width"),
    ((split_actor_input, ACTOR_INPUT_DIM), (split_policy_action, POLICY_ACTION_DIM)),
)
def test_contracts_reject_wrong_trailing_dimension(splitter, width: int) -> None:
    with pytest.raises(ValueError):
        splitter(torch.zeros(2, width - 1))


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (0, (0.0, 0.0, 0.0)),
        (1, (0.3, 0.0, 0.0)),
        (2, (-0.3, 0.0, 0.0)),
        (3, (0.0, 0.1, 0.0)),
        (4, (0.0, 0.0, -0.2)),
        (5, (0.3, 0.0, -0.2)),
        (6, (-0.3, 0.0, -0.2)),
    ),
)
def test_command_modes_preserve_body_frame_intent(mode: int, expected: tuple[float, ...]) -> None:
    assert compose_velocity_command(mode, 0.3, 0.1, -0.2) == expected


def test_training_command_mix_is_forward_dominant_without_dropping_modes() -> None:
    probabilities = LocomotionCommandCfg().mode_probabilities

    assert len(probabilities) == 7
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in probabilities)
    assert probabilities[1] + probabilities[5] == pytest.approx(0.5)
    assert probabilities[2] + probabilities[6] == pytest.approx(0.2)
