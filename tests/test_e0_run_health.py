"""Stage gates, abort rules and T1 level inheritance on synthetic scalar series."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e0_run_health as health  # noqa: E402


def series(values, start=0):
    return [(start + index, float(value)) for index, value in enumerate(values)]


def test_inherited_level_uses_tail_median_and_offset():
    scalars = {"train/terrain/level": series([1.0] * 500 + [4.4] * 60 + [5.2] * 40)}
    level, info = health.inherited_terrain_level(scalars, health_level_tags(), window=100)
    assert level == 4 and info["median"] == pytest.approx(4.4)
    level, _ = health.inherited_terrain_level(scalars, health_level_tags(), window=100, offset=1)
    assert level == 3


def test_inherited_level_never_negative_and_requires_a_tag():
    assert health.inherited_terrain_level({"Curriculum/terrain_levels": series([0.2] * 10)},
                                          health_level_tags(), offset=1)[0] == 0
    with pytest.raises(KeyError):
        health.inherited_terrain_level({"other": series([1.0])}, health_level_tags())


def health_level_tags():
    return ("train/terrain/level", "Curriculum/terrain_levels")


def test_suffix_match_finds_prefixed_tags():
    scalars = {"Episode/train/actuation/leg_saturation_rate": series([0.1])}
    assert health.resolve_tag(scalars, ["train/actuation/leg_saturation_rate"]) == (
        "Episode/train/actuation/leg_saturation_rate"
    )


def test_stage_checks_pass_fail_and_missing():
    spec = health.load_gate_spec()
    scalars = {
        "train/runtime/all_finite": series([1.0] * 300),
        "train/actuation/leg_saturation_rate": series([0.12] * 300),
        "train/terrain/flat/timeout_survival_rate": series([0.95] * 300),
    }
    assert health.evaluate_checks(scalars, spec["stages"]["foundation"])["status"] == "PASS"
    scalars["train/actuation/leg_saturation_rate"] = series([0.2] * 300)
    report = health.evaluate_checks(scalars, spec["stages"]["foundation"])
    assert report["status"] == "FAIL"
    assert [c["name"] for c in report["checks"] if c["status"] == "FAIL"] == ["leg_saturation_rate"]
    del scalars["train/terrain/flat/timeout_survival_rate"]
    report = health.evaluate_checks(scalars, spec["stages"]["foundation"])
    assert any(c["status"] == "MISSING" for c in report["checks"])


def test_expansion_fallback_check_uses_inherited_level():
    spec = health.load_gate_spec()
    check = next(c for c in spec["stages"]["expansion"] if c["name"] == "no_level_fallback_after_resume")
    scalars = {"train/terrain/level": series([4.1] * 100 + [5.0] * 200)}
    assert health.evaluate_checks(scalars, [check], {"inherited_level": 5})["status"] == "PASS"
    scalars = {"train/terrain/level": series([1.2] * 100 + [5.0] * 200)}
    assert health.evaluate_checks(scalars, [check], {"inherited_level": 5})["status"] == "FAIL"
    assert health.evaluate_checks(scalars, [check])["status"] == "FAIL"  # no context -> MISSING


def test_abort_on_non_finite_or_rising_unsafe_rate():
    spec = health.load_gate_spec()
    healthy = {"train/runtime/all_finite": series([1.0] * 600),
               "train/safety/unsafe_termination": series([0.2] * 600)}
    assert health.abort_reasons(healthy, spec) == []
    broken = dict(healthy, **{"train/runtime/all_finite": series([1.0] * 10 + [0.0])})
    assert health.abort_reasons(broken, spec)
    rising = dict(healthy, **{"train/safety/unsafe_termination": series([0.31 + i * 1e-4 for i in range(500)])})
    assert health.abort_reasons(rising, spec)
    flat_high = dict(healthy, **{"train/safety/unsafe_termination": series([0.35 - i * 1e-5 for i in range(500)])})
    assert health.abort_reasons(flat_high, spec) == []  # above 30 % but falling


def test_physx_buffer_error_pattern():
    assert health.PHYSX_BUFFER_ERROR.search(
        "[Error] PhysX error: GPU found_lost_aggregate_pairs buffer overflow, increase capacity"
    )
    assert not health.PHYSX_BUFFER_ERROR.search("[INFO] PhysX initialized")
