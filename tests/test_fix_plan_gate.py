from pathlib import Path
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fix_plan_gate as gates


def test_g3_records_mapping_failure_before_missing_later_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G3"
    folder.mkdir()
    mapping = folder / "mapping_check.json"
    mapping.write_text(json.dumps({"status": "FAIL"}))
    assert gates.g3(SimpleNamespace()) == 1
    result = json.loads((folder / "gate.json").read_text())
    assert result["status"] == "FAIL"
    assert "G3.0" in result["details"]["failures"][0]
    assert str(mapping) in result["evidence"]


def test_g3_records_missing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    assert gates.g3(SimpleNamespace()) == 1
    result = json.loads((tmp_path / "G3/gate.json").read_text())
    assert "missing evidence" in result["details"]["failures"][0]


def test_g3_rejects_diagnostic_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G3"
    folder.mkdir()
    (folder / "mapping_check.json").write_text(json.dumps({"status": "DIAGNOSTIC", "checks_passed": True}))
    assert gates.g3(SimpleNamespace()) == 1


def _g5_args(*manual, reject=()):
    return SimpleNamespace(manual=list(manual), reject=list(reject))


def test_g5_rejects_incomplete_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G5"
    folder.mkdir()
    (folder / "timing_matrix.json").write_text(
        json.dumps({"items": {"baseline": {"collection_time_s": 8.0}}})
    )
    assert gates.g5(_g5_args()) == 1
    result = json.loads((folder / "gate.json").read_text())
    assert "timing matrix is incomplete" in result["details"]["failures"][0]


def test_g5_accepts_documented_rejected_candidate_but_requires_fast_item_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G5"
    folder.mkdir()
    items = {"baseline": {"collection_time_s": 8.0}}
    for name in ("T2", "T3b", "T4", "T5", "T6"):
        items[name] = {"collection_time_s": 7.8, "gain_vs_baseline": 0.025, "manual_checks": {}}
    items["T3a"] = {
        "collection_time_s": 5.0,
        "gain_vs_baseline": 0.375,
        "manual_checks": {"closure": None},
        "overrides": ["solver=4"],
    }
    items["T7"] = {"status": "rejected", "rejected_reason": "incompatible", "collection_time_s": None}
    (folder / "timing_matrix.json").write_text(json.dumps({"items": items}))

    assert gates.g5(_g5_args()) == 1
    result = json.loads((folder / "gate.json").read_text())
    assert any("T3a physical checks are incomplete" in failure for failure in result["details"]["failures"])


def test_g5_can_record_a_failed_combined_candidate_as_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G5"
    folder.mkdir()
    items = {"baseline": {"collection_time_s": 8.0}}
    for name in ("T2", "T3a", "T3b", "T4", "T5"):
        items[name] = {"collection_time_s": 7.9, "gain_vs_baseline": 0.0125, "manual_checks": {}}
    items["T6"] = {
        "collection_time_s": 6.5,
        "gain_vs_baseline": 0.1875,
        "manual_checks": {"contact": "pass"},
        "overrides": ["period=0.02"],
    }
    items["T7"] = {"status": "rejected", "rejected_reason": "incompatible", "collection_time_s": None}
    items["combined"] = {"collection_time_s": 8.2}
    (folder / "timing_matrix.json").write_text(json.dumps({"items": items}))

    assert gates.g5(_g5_args(reject=("T6=combined run slower than baseline",))) == 0
    matrix = json.loads((folder / "timing_matrix.json").read_text())
    assert matrix["items"]["T6"]["status"] == "rejected"
    assert matrix["items"]["T6"]["kept"] is False


def _g6_summary(**metrics):
    return {"metrics": {name: {"mean": value} for name, value in metrics.items()}}


def test_g6_qualifies_contract_without_claiming_short_policy_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G6"
    folder.mkdir()
    p0 = _g6_summary(**{
        "train/safety/base_collision_rate": 0.002,
        "train/terrain/pyramid_stairs/safety/unsafe_termination": 0.20,
        "train/terrain/level": 0.4,
        "train/actuation/joint_velocity_limit_rate": 0.04,
        "train/actuation/leg_torque_saturation_rate": 0.2,
    })
    p1 = _g6_summary(**{
        "train/actuation/action_saturation_rate": 0.04,
        "train/actuation/leg_saturation_rate": 0.03,
        "train/safety/base_collision_rate": 0.001,
        "train/runtime/finite_rate": 1.0,
        "train/terrain/pyramid_stairs/safety/unsafe_termination": 0.8,
        "train/terrain/level": 0.0,
        "train/runtime/closure_residual_max_mm": 12.0,
        "train/actuation/joint_velocity_limit_rate": 0.3,
        "train/actuation/leg_torque_saturation_rate": 0.6,
    })
    (folder / "p0_summary.json").write_text(json.dumps(p0))
    (folder / "p1_summary.json").write_text(json.dumps(p1))

    assert gates.g6(SimpleNamespace()) == 0
    result = json.loads((folder / "gate.json").read_text())
    assert result["status"] == "PASS"
    assert result["details"]["policy_quality_accepted"] is False
    assert result["details"]["risks"]


def test_g6_rejects_saturated_or_nonfinite_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    folder = tmp_path / "G6"
    folder.mkdir()
    common = {
        "train/safety/base_collision_rate": 0.001,
        "train/terrain/pyramid_stairs/safety/unsafe_termination": 0.2,
        "train/terrain/level": 0.1,
        "train/actuation/joint_velocity_limit_rate": 0.1,
        "train/actuation/leg_torque_saturation_rate": 0.2,
    }
    (folder / "p0_summary.json").write_text(json.dumps(_g6_summary(**common)))
    (folder / "p1_summary.json").write_text(json.dumps(_g6_summary(**{
        **common,
        "train/actuation/action_saturation_rate": 0.2,
        "train/actuation/leg_saturation_rate": 0.2,
        "train/runtime/finite_rate": 0.99,
        "train/runtime/closure_residual_max_mm": 0.5,
    })))

    assert gates.g6(SimpleNamespace()) == 1
    result = json.loads((folder / "gate.json").read_text())
    assert result["status"] == "FAIL"
    assert len(result["details"]["failures"]) == 3


def test_g7_qualifies_free_base_clearance_without_claiming_scripted_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    report = tmp_path / "supported_clearance.json"
    best = {
        "stable_free_base_support": True,
        "terminated": False,
        "sustained_clearance_m": 0.07,
        "body_relative_lift_m": 0.07,
    }
    report.write_text(json.dumps({
        "action_contract": "servo_range_v2",
        "free_base": True,
        "args": {"step_height": 0.03},
        "clearance_scan": {leg: {"best": best} for leg in ("FR", "FL", "RL", "RR")},
        "manual": {"complete": False, "termination_reasons": ["base_collision"], "crossings": []},
    }))

    assert gates.g7(SimpleNamespace(report=[report])) == 0
    result = json.loads((tmp_path / "G7/gate.json").read_text())
    assert result["status"] == "PASS"
    assert result["details"]["scripted_traversal_accepted"] is False


def test_g7_rejects_clearance_without_free_base_support(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "OUT", tmp_path)
    report = tmp_path / "unsupported_clearance.json"
    report.write_text(json.dumps({
        "action_contract": "servo_range_v2",
        "free_base": True,
        "args": {"step_height": 0.03},
        "clearance_scan": {},
        "manual": {},
    }))
    assert gates.g7(SimpleNamespace(report=[report])) == 1
