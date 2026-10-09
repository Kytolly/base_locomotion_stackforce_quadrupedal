"""Prevent terrain progress and fixture motion from masquerading as traversal."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location(
    "stair_probe", Path(__file__).resolve().parents[1] / "scripts/probe_stair_climb.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def probe_with_rows(*, rear_y=0.6, terminated=False, contact=4):
    probe = module.Probe.__new__(module.Probe)
    probe.args = SimpleNamespace(approach=0.45, step_count=1, tread=0.35, step_height=0.04, speed=0.1)
    probe.radius = 0.033
    probe.env = SimpleNamespace(step_dt=0.02)
    probe.done = terminated
    probe.trial = "test"
    probe.samples = [dict(
        phase="move", root_xyz=[0, 0.6, 0.145], tilt_rad=0,
        feet_xyz=[[0.1, 0.7, 0.073], [-0.1, 0.7, 0.073], [-0.1, rear_y, 0.073], [0.1, rear_y, 0.073]],
        feet_body_xyz=[[0, 0, -0.072]] * 4,
        contact_force_n=[4] * contact + [0] * (4 - contact),
        terminations=["base_height"] if terminated else [],
        velocity_body=[0, 0.1, 0], yaw_rate=0, time_s=i * 0.02,
    ) for i in range(20)]
    return probe


def test_root_over_edge_is_not_four_foot_traversal():
    assert not probe_with_rows(rear_y=0.42).summary()["complete"]
    assert probe_with_rows().summary()["complete"]


def test_terminal_or_unsupported_pose_never_passes():
    assert not probe_with_rows(terminated=True).summary()["complete"]
    assert not probe_with_rows(contact=2).summary()["complete"]


def test_contact_crossing_is_not_active_lift():
    probe = probe_with_rows()
    probe.samples[0]["feet_xyz"][0][1] = 0.40
    result = probe.summary()
    assert result["crossings"]
    assert not result["crossings"][0]["active_lift_evidence"]
