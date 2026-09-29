"""Determinism and geometry-contract tests for both fixed tracks."""

from base_locomotion_stackforce_quadrupedal.benchmark import (
    TrackTraversal,
    plateau_track_parameters,
    track_length,
    washboard_track_parameters,
)


def test_plateau_parameters_are_seeded_and_complete() -> None:
    nominal = plateau_track_parameters()
    randomized_a = plateau_track_parameters(9001, True)
    randomized_b = plateau_track_parameters(9001, True)

    assert nominal["slope_deg"] == 12.0
    assert nominal["ramp_height_m"] > 0.0
    assert track_length(nominal) > 5.0
    assert randomized_a == randomized_b


def test_washboard_parameters_are_seeded_and_complete() -> None:
    nominal = washboard_track_parameters()
    randomized_a = washboard_track_parameters(9002, True)
    randomized_b = washboard_track_parameters(9002, True)

    assert nominal["ridge_count"] == 14
    assert nominal["ramp_height_m"] == 2.0 * nominal["ridge_radius_m"]
    assert track_length(nominal) > 4.0
    assert randomized_a == randomized_b


def test_track_success_requires_ordered_traversal_and_staying_in_corridor() -> None:
    parameters = plateau_track_parameters()
    traversal = TrackTraversal(parameters)
    traversal.update(0.0, track_length(parameters) + 0.1)
    assert not traversal.complete

    traversal = TrackTraversal(parameters)
    for step in range(int(track_length(parameters) / 0.1) + 2):
        y = min(step * 0.1, traversal.gates[-1])
        traversal.update(0.0, y)
    traversal.update(0.0, traversal.gates[-1])
    assert traversal.complete

    traversal = TrackTraversal(parameters)
    for step in range(int(track_length(parameters) / 0.1) + 2):
        y = min(step * 0.1, traversal.gates[-1])
        x = traversal.half_width_m + 0.01 if 1.0 <= y <= 1.2 else 0.0
        traversal.update(x, y)
    traversal.update(0.0, traversal.gates[-1])
    assert not traversal.complete
