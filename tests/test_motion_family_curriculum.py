"""Segment-level attribution checks for the motion-family curriculum."""

from types import SimpleNamespace

import torch

from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.curricula.motion import (
    motion_family_rehearsal,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy.commands import (
    LocomotionCommand,
)


def _command_term() -> LocomotionCommand:
    term = LocomotionCommand.__new__(LocomotionCommand)
    term._env = SimpleNamespace(step_dt=0.5, num_envs=2, device="cpu")
    term.cfg = SimpleNamespace(resampling_time_range=(2.0, 2.0))
    term._motion_family = torch.tensor([0, 1], dtype=torch.long)
    term._segment_tracking_error_sq = torch.zeros(2)
    term._segment_yaw_error_sq = torch.zeros(2)
    term._segment_steps = torch.zeros(2, dtype=torch.long)
    term._completed_motion_segments = []
    term._base_sampling_probabilities = torch.tensor(
        [0.10, 0.30, 0.10, 0.10, 0.10, 0.20, 0.10]
    )
    term._sampling_probabilities = term._base_sampling_probabilities.clone()
    return term


def test_multiple_segments_are_attributed_before_episode_reset() -> None:
    term = _command_term()
    for _ in range(4):
        term.record_motion_step(torch.tensor([0.01, 0.01]), torch.tensor([0.01, 0.01]))

    term._finalize_motion_segments(
        torch.tensor([0]),
        terminated=torch.tensor([False]),
        command_completed=torch.tensor([True]),
    )
    term._motion_family[0] = 2
    for _ in range(2):
        term.record_motion_step(torch.tensor([0.01, 0.01]), torch.tensor([0.01, 0.01]))

    env = SimpleNamespace(
        device="cpu",
        episode_length_buf=torch.zeros(2),
        reset_terminated=torch.tensor([False, True]),
        command_manager=SimpleNamespace(get_term=lambda _: term),
    )
    result = motion_family_rehearsal(env, torch.tensor([0, 1]))

    assert torch.isclose(result, torch.tensor(1.0 / 3.0))
    assert env.motion_family_stats[:, 0].tolist()[:3] == [1.0, 1.0, 1.0]
    assert env.motion_family_stats[:, 1].tolist()[:3] == [1.0, 0.0, 0.0]
    assert term._completed_motion_segments == []
    assert term._segment_steps.tolist() == [0, 0]
    assert term.sampling_probabilities[2] > term._base_sampling_probabilities[2]


def test_reset_drains_only_selected_environment_records() -> None:
    term = _command_term()
    term.record_motion_step(torch.tensor([0.01, 0.04]), torch.tensor([0.01, 0.04]))
    term._finalize_motion_segments(
        torch.tensor([0, 1]),
        terminated=torch.tensor([False, False]),
        command_completed=torch.tensor([True, True]),
    )

    first = term.finish_episode_motion_segments(torch.tensor([0]), torch.tensor([False]))
    second = term.finish_episode_motion_segments(torch.tensor([1]), torch.tensor([False]))

    assert first["env_id"].tolist() == [0]
    assert second["env_id"].tolist() == [1]
    assert term._completed_motion_segments == []
