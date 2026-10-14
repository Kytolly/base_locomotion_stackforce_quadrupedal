"""G1 actuation diagnostics on constructed tensors (no Isaac Sim)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / (
    "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/tasks/"
    "manager_based/base_locomotion_stackforce_quadrupedal/evaluation/metric/actuation.py"
)
spec = importlib.util.spec_from_file_location("_actuation_under_test", MODULE)
actuation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = actuation
spec.loader.exec_module(actuation)


def test_leg_and_wheel_saturation_fractions():
    raw = torch.tensor([[0.96, -0.99, 0.5, 0.0, 0.95, -0.951, 1.0, 0.2]])
    # |raw| > 0.95: 0.96, 0.99, 0.951, 1.0 -> 4 of 8 (0.95 itself does not count).
    assert actuation.saturation_fraction(raw).item() == pytest.approx(0.5)
    wheel = torch.tensor([[1.0, 0.0, 0.0, -0.97], [0.1, 0.1, 0.1, 0.1]])
    assert actuation.saturation_fraction(wheel).tolist() == pytest.approx([0.5, 0.0])


def test_torque_saturation_and_rated_fractions():
    limit = torch.full((1, 8), 0.392)
    torque = torch.tensor([[0.392, -0.38, 0.3724, 0.2, 0.0, 0.13, -0.12, 0.5]])
    # >= 0.95 * 0.392 = 0.3724: 0.392, 0.38, 0.3724, 0.5 -> 4 / 8.
    assert actuation.torque_saturation_fraction(torque, limit).item() == pytest.approx(0.5)
    # > 0.127 N m: 0.392, 0.38, 0.3724, 0.2, 0.13, 0.5 -> 6 / 8.
    assert actuation.over_rated_fraction(torque).item() == pytest.approx(0.75)


def test_tracking_error_is_mean_square_over_joints():
    target = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
    actual = torch.tensor([[0.0, 0.0, 0.0, 0.2]])
    assert actuation.tracking_error_sq(target, actual).item() == pytest.approx((0.01 + 0.04 + 0.04) / 4)


def test_closure_residual_ignores_the_axial_offset():
    wheel = torch.zeros((2, 4, 3))
    closure = torch.zeros((2, 4, 3))
    closure[..., 0] = 0.045  # designed axial offset along the closure axis
    closure[1, 2, 1] = 0.0007  # 0.7 mm radial gap on one leg of env 1
    axis = torch.zeros((2, 4, 3))
    axis[..., 0] = 1.0
    residual = actuation.closure_residual_mm(closure, wheel, axis)
    assert residual.tolist() == pytest.approx([0.0, 0.7], abs=1e-6)


def test_rolling_mean_window():
    window = actuation.RollingMean(num_envs=2, width=1, window=3, device="cpu")
    means = [window.push(torch.tensor([[v], [2 * v]])) for v in (1.0, 2.0, 3.0, 4.0)]
    assert means[0].flatten().tolist() == pytest.approx([1.0, 2.0])
    assert means[2].flatten().tolist() == pytest.approx([2.0, 4.0])
    assert means[3].flatten().tolist() == pytest.approx([3.0, 6.0])  # (2+3+4)/3
    window.reset(torch.tensor([0]))
    # env 0 restarts; env 1 keeps its last three values (4, 6, 8) -> (10, 6, 8).
    assert window.push(torch.tensor([[10.0], [10.0]])).flatten().tolist() == pytest.approx([10.0, 8.0])
