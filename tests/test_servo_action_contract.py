"""E0-v2 servo action contract (plan section 3A and G4) without Isaac Sim."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/tasks/manager_based/base_locomotion_stackforce_quadrupedal"
LEG_JOINTS = tuple(f"{leg}_Outer_Hip_Joint" for leg in ("FR", "FL", "RL", "RR")) + tuple(
    f"{leg}_Inner_Hip_Joint" for leg in ("FR", "FL", "RL", "RR")
)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, TASK / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


servo = _load("_servo_range_under_test", "mdp/action/servo_range.py")
kinematics = _load("_kinematics_under_test", "env/robots/kinematics.py")
CONTRACT = servo.load_servo_contract()
KIN = kinematics.LegKinematics()

SECTION_3A = {  # channel table: default, lower, upper (rad)
    "FR_Outer_Hip_Joint": (-0.447, -3.019, 1.693),
    "FR_Inner_Hip_Joint": (-1.049, -4.250, 0.463),
    "FL_Inner_Hip_Joint": (1.049, -1.510, 3.203),
    "FL_Outer_Hip_Joint": (-0.447, -3.875, 0.838),
    "RL_Outer_Hip_Joint": (0.447, -1.710, 3.002),
    "RL_Inner_Hip_Joint": (-1.049, -4.250, 0.463),
    "RR_Inner_Hip_Joint": (1.049, -1.283, 3.429),
    "RR_Outer_Hip_Joint": (0.447, -0.576, 4.136),
}


def test_contract_file_matches_its_builder():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_servo_contract.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_provisional_contract_reproduces_section_3a():
    if CONTRACT["status"] != "provisional_section_3A":
        pytest.skip("contract already replaced by G3 measurements")
    for name, (default, lower, upper) in SECTION_3A.items():
        joint = CONTRACT["joints"][name]
        assert (joint["default"], joint["lower"], joint["upper"]) == (default, lower, upper)
    assert CONTRACT["servo"]["model"] == "GDW DS041MG (300 deg configured)"
    assert CONTRACT["servo"]["pulse_to_angle_deg_per_us"] == 0.15
    assert CONTRACT["servo"]["physical_endpoints_measured"] is False


def test_urdf_mapping_matches_section_3a_signs_and_depths():
    # G3.0 kinematic pre-check: s_i table, sim zero = 154/6.5 deg, 79 / 137 / 158 mm.
    expected_signs = {"FR": (1, -1), "FL": (1, 1), "RL": (-1, -1), "RR": (-1, 1)}
    for leg in kinematics.LEGS:
        assert KIN.hip_signs(leg) == expected_signs[leg]
        alpha, beta = KIN.crank_angles_deg(leg, 0.0, 0.0)
        assert alpha == pytest.approx(154.0, abs=0.2) and beta == pytest.approx(6.5, abs=0.2)
        assert KIN.foot_depth_below_hips_m(leg, 0.0, 0.0) == pytest.approx(0.079, abs=0.001)
        straight = KIN.hips_for_crank_angles(leg, 90.0, 90.0)
        assert KIN.foot_depth_below_hips_m(leg, *straight) == pytest.approx(0.158, abs=0.001)
        stance = (CONTRACT["joints"][f"{leg}_Outer_Hip_Joint"]["default"],
                  CONTRACT["joints"][f"{leg}_Inner_Hip_Joint"]["default"])
        alpha, beta = KIN.crank_angles_deg(leg, *stance)
        assert (alpha, beta) == pytest.approx(tuple(CONTRACT["stance_crank_angles_deg"]), abs=0.1)
        assert KIN.foot_depth_below_hips_m(leg, *stance) == pytest.approx(0.137, abs=0.002)


def test_default_stance_knees_close_the_loop_within_1mm():
    for leg in kinematics.LEGS:
        error = KIN.closure_error(
            leg,
            CONTRACT["joints"][f"{leg}_Outer_Hip_Joint"]["default"],
            CONTRACT["joints"][f"{leg}_Inner_Hip_Joint"]["default"],
            CONTRACT["passive_knees"][f"{leg}_Outer_Knee_Joint"],
            CONTRACT["passive_knees"][f"{leg}_Inner_Knee_Joint"],
        )
        assert float(abs(error).max()) * 1000.0 <= 1.0


def test_raw_action_maps_to_default_and_range_endpoints():
    lower, default, upper = (torch.tensor(v) for v in servo.leg_joint_ranges(LEG_JOINTS))
    for raw, expected in ((0.0, default), (1.0, upper), (-1.0, lower)):
        target = servo.raw_to_joint_target(torch.full((8,), raw), lower, default, upper)
        assert torch.allclose(target, expected, atol=1e-6)
    half = servo.raw_to_joint_target(torch.full((8,), 0.5), lower, default, upper)
    assert torch.allclose(half, default + 0.5 * (upper - default), atol=1e-6)


def test_grid_projection_preserves_feasible_pairs_and_moves_infeasible_pairs():
    spec = {
        "outer_grid": [0.0, 2.0, 3],
        "inner_grid": [0.0, 2.0, 3],
        "feasible_tensor": torch.tensor([True, True, False, True, True, False, False, False, False]),
        "nearest_tensor": torch.tensor([0, 1, 1, 3, 4, 4, 3, 4, 4]),
    }
    targets = torch.tensor([[0.1, 0.1], [1.9, 1.9]])
    projected = servo.project_leg_target_grid(targets, spec)
    assert torch.allclose(projected[0], targets[0])
    assert torch.allclose(projected[1], torch.tensor([1.0, 1.0]))


def test_body_heights_follow_the_new_stance():
    heights = CONTRACT["body_height"]
    low, high = heights["command_range_m"]
    assert heights["nominal_m"] == pytest.approx(heights["geometric_standing_m"], abs=0.002)
    assert high <= heights["straight_leg_m"] - 0.0099  # reachable with straight legs
    assert low == pytest.approx(heights["nominal_m"] - 0.03, abs=1e-6)
    scale = heights["nominal_m"] / 0.105
    assert heights["warning_m"] == pytest.approx(0.08 * scale, abs=0.001)
    assert heights["minimum_m"] == pytest.approx(0.06 * scale, abs=0.001)


def test_yaml_leg_std_keeps_the_old_physical_noise():
    config = OmegaConf.load(ROOT / "configs/train/base_locomotion_e0_46d.yaml")
    init_std = list(config.agent.actor.distribution_cfg.init_std)
    assert init_std[:8] == servo.leg_init_std(LEG_JOINTS)
    assert init_std[8:] == [CONTRACT["exploration"]["wheel_init_std"]] * 4
    lower, default, upper = servo.leg_joint_ranges(LEG_JOINTS)
    for std, lo, d, hi in zip(init_std[:8], lower, default, upper):
        assert std * min(d - lo, hi - d) == pytest.approx(0.15, abs=0.001)
        assert 0.05 <= std <= 0.5


def test_obstacles_start_at_two_centimetres():
    for low, high in CONTRACT["obstacles"]["height_ranges_m"].values():
        assert low == 0.02 and low < high
    lift = CONTRACT["obstacles"]["min_measured_lift_m"]
    if lift is not None:
        for _, high in CONTRACT["obstacles"]["height_ranges_m"].values():
            assert high <= 0.8 * lift + 1e-9
    assert math.isclose(CONTRACT["obstacles"]["cap_fraction_of_lift"], 0.8)
