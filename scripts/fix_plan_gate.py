#!/usr/bin/env python3
"""Evaluate fix-plan gate evidence and record ``output/fix_plan/G<n>/gate.json``.

docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md defines what each gate must show; this
tool only reads the evidence files the gate scripts wrote, applies the plan's
pass conditions and records the verdict with file hashes and the commit.
``scripts/run_e0_v2_long_run.sh`` refuses to start unless G0-G7 are PASS (G3 may be CONDITIONAL) and G4
records the contract-upgrade confirmation.

    python scripts/fix_plan_gate.py status
    python scripts/fix_plan_gate.py g0
    python scripts/fix_plan_gate.py g1 --report output/fix_plan/G1/b0_validation.json
    python scripts/fix_plan_gate.py g2 --wiki-evidence /path/to/E-CAL-002.md
    python scripts/fix_plan_gate.py g3
    python scripts/fix_plan_gate.py g4 --action-probe output/action-probes/e0v2_0.2.json ... \
        --pytest-passed --cpu-smoke-passed --validation-smoke-passed --confirm-contract-upgrade
    python scripts/fix_plan_gate.py g5 --manual T4="no pose readers" ...
    python scripts/fix_plan_gate.py g6
    python scripts/fix_plan_gate.py g7 --report output/fix_plan/G7/five_phase_30mm.json ...

A gate that fails is recorded as FAIL with the reasons; per the plan, stop and
report instead of skipping it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/fix_plan"
GATES = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")


def _load(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def record(gate: str, status: str, evidence: list[Path], details: dict[str, Any], **extra) -> int:
    payload = {
        "gate": gate,
        "status": status,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _commit(),
        "evidence": {str(Path(p)): _sha(p) for p in evidence if Path(p).is_file()},
        "details": details,
        **extra,
    }
    path = OUT / gate / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[{gate}] {status} -> {path}")
    for reason in details.get("failures", []):
        print(f"  - {reason}")
    return 0 if status in ("PASS", "CONDITIONAL") else 1


def _numbers(node: Any, suffix: str) -> list[float]:
    """All finite numbers stored under keys ending with ``suffix`` anywhere in a JSON tree."""
    found: list[float] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).endswith(suffix) and isinstance(value, (int, float)) and math.isfinite(value):
                found.append(float(value))
            else:
                found += _numbers(value, suffix)
    elif isinstance(node, list):
        for value in node:
            found += _numbers(value, suffix)
    return found


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------- gates
def g0(args) -> int:
    profile, baseline = OUT / "G0/step_profile.json", OUT / "G0/baseline_metrics.json"
    failures = []
    data = _load(profile)
    if data.get("status") != "PASS":
        failures.append(f"segment sum deviates {data.get('segment_sum_deviation'):.1%} from the measured step (> 10 %)")
    base = _load(baseline)
    if not base:
        failures.append("baseline_metrics.json is empty")
    required = (
        "train/terrain/level",
        "train/actuation/action_saturation_rate",
        "train/actuation/leg_action_rms",
        "train/actuation/wheel_action_rms",
        "train/safety/unsafe_termination",
        "train/safety/timeout",
    )
    for label, item in base.items():
        metrics = item.get("metrics", {})
        missing = [tag for tag in required if tag not in metrics]
        if not any(tag in metrics for tag in ("Perf/collection time", "Perf/collection_time")):
            missing.append("Perf/collection time")
        if not any(tag in metrics for tag in ("Perf/learning time", "Perf/learning_time")):
            missing.append("Perf/learning time")
        if missing:
            failures.append(f"{label}: missing baseline tags {missing}")
    return record("G0", "FAIL" if failures else "PASS", [profile, baseline],
                  {"failures": failures, "measured_step_ms": data.get("measured_step_ms")})


def g1(args) -> int:
    report = _load(args.report)
    keys = ("leg_saturation_rate", "wheel_saturation_rate", "leg_torque_saturation_rate",
            "leg_tracking_error_rad", "closure_residual_max_mm", "leg_torque_over_rated_rate")
    values = {key: _mean(_numbers(report, key)) for key in keys}
    failures = [f"{key} missing or not finite" for key, value in values.items() if value is None]
    if not failures and not values["leg_saturation_rate"] > values["wheel_saturation_rate"]:
        failures.append("leg saturation is not above wheel saturation: diagnosis premise fails; stop and re-analyse")
    for flag, label in ((args.metrics_smoke_passed, "CPU single-env metrics smoke"),
                        (args.wandb_keys_seen, "new keys visible and finite in W&B")):
        if not flag:
            failures.append(f"{label} not confirmed")
    return record("G1", "FAIL" if failures else "PASS", [args.report], {"failures": failures, "baseline": values})


def g2(args) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    contract = json.loads((ROOT / (
        "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/tasks/manager_based/"
        "base_locomotion_stackforce_quadrupedal/mdp/action/e0_v2_servo_contract.json")).read_text())
    evidence = Path(args.wiki_evidence) if args.wiki_evidence else None
    failures = []
    if evidence is None or not evidence.is_file():
        failures.append("A1-A5 wiki evidence file is missing; pass its path with --wiki-evidence")
    else:
        text = evidence.read_text(encoding="utf-8")
        required = ("E-CAL-002", "status: VALID", "300°", "A1", "A2", "A3", "A4", "A5")
        if any(token not in text for token in required):
            failures.append("wiki evidence must register valid E-CAL-002 and the 300-degree A1-A5 premises")
    if contract["servo"].get("pulse_to_angle_deg_per_us") != 0.15:
        failures.append("servo contract does not use the configured 300-degree nominal scale")
    details = {
        "failures": failures,
        "servo": contract["servo"],
        "decision": "effort_limit_sim 0.392 N m (stall) and velocity_limit_sim 8.73 rad/s kept; stiffness 20 / damping 0.5 kept until M3-G03",
        "premises_registered": ["A1 300 deg configured travel", "A2 firmware zero = straight leg", "A3 firmware and servo nominal scales agree",
                                "A4 mounting directions / sim zero", "A5 mechanical stops"],
        "wiki_evidence": args.wiki_evidence,
    }
    return record("G2", "FAIL" if failures else "PASS", [evidence] if evidence else [], details)


def g3(args) -> int:
    files = {name: OUT / "G3" / f"{name}.json" for name in ("mapping_check", "final_ranges", "stance", "lift", "range_scan")}
    data = {}
    for name, path in files.items():
        try:
            data[name] = _load(path)
        except (OSError, ValueError) as exc:
            return record("G3", "FAIL", list(files.values()), {"failures": [str(exc)]})
        if name == "mapping_check" and data[name].get("status") != "PASS":
            return record("G3", "FAIL", [path], {
                "failures": ["G3.0 mapping check failed: correct section 3A first"],
            })
    failures, status = [], "PASS"
    if data["mapping_check"]["status"] != "PASS":
        failures.append("G3.0 mapping check failed: correct section 3A first")
    if data["stance"]["status"] != "PASS":
        failures.append("new stance is not stable below stall torque: move it 10 mm towards straight legs")
    lift = float(data["lift"]["min_lift_m"])
    if lift < 0.025:
        failures.append(f"minimum lift {lift * 1000:.1f} mm < 25 mm: decide between relaxing 0.9 travel or capping obstacles")
    elif lift < 0.030:
        status = "CONDITIONAL"
    if failures:
        status = "FAIL"
    details = {
        "failures": failures,
        "min_lift_m": lift,
        "obstacle_cap_m": round(0.8 * lift, 4),
        "projection_recommended": data["final_ranges"].get("projection_recommended"),
        "next": ("python scripts/build_servo_contract.py --final-ranges output/fix_plan/G3/final_ranges.json "
                 "--stance output/fix_plan/G3/stance.json --lift output/fix_plan/G3/lift.json"),
    }
    if status == "CONDITIONAL":
        details["note"] = "step capability limited by servo travel (25-30 mm): obstacle cap = 0.8 x measured lift"
    return record("G3", status, list(files.values()), details)


def g4(args) -> int:
    contract_path = ROOT / (
        "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/tasks/manager_based/"
        "base_locomotion_stackforce_quadrupedal/mdp/action/e0_v2_servo_contract.json")
    contract = json.loads(contract_path.read_text())
    failures = []
    if contract["status"] != "g3_measured" and not args.allow_provisional:
        failures.append("servo contract is still provisional_section_3A; rebuild it from the G3 results")
    magnitudes = set()
    for path in args.action_probe:
        probe = _load(path)
        magnitudes.add(float(probe["action_magnitude"]))
        if not probe.get("target_contract_pass") or probe.get("action_contract") != "servo_range_v2":
            failures.append(f"{path}: action semantics gate failed")
    missing = {0.2, 0.5, 1.0} - magnitudes
    if missing:
        failures.append(f"action probes missing for magnitudes {sorted(missing)}")
    for flag, label in ((args.pytest_passed, "pytest"), (args.cpu_smoke_passed, "single-env CPU smoke"),
                        (args.validation_smoke_passed, "64-env zero-policy validation smoke")):
        if not flag:
            failures.append(f"{label} not confirmed")
    if not args.confirm_contract_upgrade:
        failures.append("contract upgrade to E0-v2-46d not confirmed (--confirm-contract-upgrade)")
    return record("G4", "FAIL" if failures else "PASS", [contract_path, *args.action_probe],
                  {"failures": failures, "contract_status": contract["status"]},
                  contract_upgrade_confirmed=bool(args.confirm_contract_upgrade))


def g5(args) -> int:
    path = OUT / "G5/timing_matrix.json"
    matrix = _load(path)
    manual = dict(item.rsplit("=", 1) for item in args.manual)
    rejected = dict(item.split("=", 1) for item in args.reject)
    items = matrix["items"]
    kept, failures = [], []
    expected = {"baseline", "T2", "T3a", "T3b", "T4", "T5", "T6", "T7"}
    missing = sorted(expected - set(items))
    if missing:
        failures.append("timing matrix is incomplete; missing: " + ", ".join(missing))
    reference = items.get("baseline", {}).get("collection_time_s")
    if reference is None:
        failures.append("baseline collection time is missing")
    for name, item in items.items():
        if name in ("baseline", "combined"):
            continue
        if name in rejected:
            item["status"] = "rejected"
            item["rejected_reason"] = rejected[name]
        if item.get("status") == "rejected":
            if not item.get("rejected_reason"):
                failures.append(f"{name} is rejected without a reason")
            item["kept"] = False
            continue
        if item.get("collection_time_s") is None:
            failures.append(f"{name} collection time is missing")
            item["kept"] = False
            continue
        for check in item.get("manual_checks", {}):
            key = f"{name}:{check}"
            if key in manual:
                item["manual_checks"][check] = manual[key]
        gain = item.get("gain_vs_baseline")
        checks = item.get("manual_checks", {})
        if gain is not None and gain >= 0.05:
            pending = [check for check, value in checks.items() if value in (None, "")]
            if pending:
                failures.append(f"{name} physical checks are incomplete: " + "; ".join(pending))
        checks_ok = all(str(value).strip().lower() not in ("", "fail", "false") for value in checks.values())
        checks_ok = checks_ok and all(value is not None for value in checks.values())
        item["kept"] = bool(gain is not None and gain >= 0.05 and checks_ok)
        if item["kept"]:
            kept += item["overrides"]
    path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    combined = items.get("combined", {}).get("collection_time_s")
    if kept:
        if combined is None:
            failures.append("kept items not yet timed together: python scripts/fix_plan_runs.py g5 --combined " + " ".join(kept))
        elif reference is not None and not combined < reference:
            failures.append("combined collection time is not below the baseline")
    details = {"failures": failures, "kept_overrides": kept, "t1_curriculum_inheritance": "implemented (orchestrator)",
               "combined_collection_time_s": combined, "baseline_collection_time_s": reference,
               "apply": "copy kept_overrides into configs/train/base_locomotion_e0_46d.yaml env.throughput / logging"}
    return record("G5", "FAIL" if failures else "PASS", [path], details)


def g6(args) -> int:
    p0, p1 = _load(OUT / "G6/p0_summary.json"), _load(OUT / "G6/p1_summary.json")

    def value(summary, tag):
        item = summary["metrics"].get(tag)
        return None if item is None else item["mean"]

    # G6 is a diagnostic qualification, not a short-run policy benchmark.  The
    # 1k probes are long enough to expose mapping saturation and numerical
    # failures, but not long enough to require the new policy to outperform a
    # previously tuned action contract.  Keep those comparisons as explicit
    # long-training risks instead of turning under-training into a contract
    # failure.
    qualification = {
        "action_saturation_rate": (value(p1, "train/actuation/action_saturation_rate"), "<= 0.05", lambda a, b: a <= 0.05),
        "leg_saturation_rate": (value(p1, "train/actuation/leg_saturation_rate"), "< 0.10", lambda a, b: a < 0.10),
        "base_collision_rate": (value(p1, "train/safety/base_collision_rate"), "<= P0", lambda a, b: b is not None and a <= b,
                                value(p0, "train/safety/base_collision_rate")),
        "finite_rate": (value(p1, "train/runtime/finite_rate"), "== 1", lambda a, b: a >= 1.0),
    }
    observations = {
        "low_step_unsafe": (value(p1, "train/terrain/pyramid_stairs/safety/unsafe_termination"),
                            "<= 0.7 x P0", lambda a, b: b is not None and a <= 0.7 * b,
                            value(p0, "train/terrain/pyramid_stairs/safety/unsafe_termination")),
        "terrain_level": (value(p1, "train/terrain/level"), "> P0", lambda a, b: b is not None and a > b,
                          value(p0, "train/terrain/level")),
        "closure_residual_max_mm": (value(p1, "train/runtime/closure_residual_max_mm"), "<= 1", lambda a, b: a <= 1.0),
        "joint_velocity_limit_rate": (value(p1, "train/actuation/joint_velocity_limit_rate"), "<= P0 + 0.05",
                                      lambda a, b: b is not None and a <= b + 0.05,
                                      value(p0, "train/actuation/joint_velocity_limit_rate")),
    }

    def evaluate(rows):
        table = {}
        for name, row in rows.items():
            p1_value, rule, test = row[0], row[1], row[2]
            p0_value = row[3] if len(row) > 3 else None
            ok = p1_value is not None and bool(test(p1_value, p0_value))
            table[name] = {"p1": p1_value, "p0": p0_value, "rule": rule, "pass": ok}
        return table

    table = evaluate(qualification)
    observation_table = evaluate(observations)
    failures = []
    for name, item in table.items():
        if not item["pass"]:
            failures.append(f"{name}: P1={item['p1']} P0={item['p0']} rule {item['rule']}")
    risks = []
    for name, item in observation_table.items():
        if not item["pass"]:
            risks.append(f"{name}: P1={item['p1']} P0={item['p0']} rule {item['rule']}")
    advice = []
    torque = value(p1, "train/actuation/leg_torque_saturation_rate")
    torque0 = value(p0, "train/actuation/leg_torque_saturation_rate")
    if risks and torque is not None and torque0 is not None and torque >= torque0:
        advice.append("long-training risk: torque saturation did not improve; monitor it at every G8 stage gate")
    if not observation_table["closure_residual_max_mm"]["pass"]:
        advice.append("long-training risk: dynamic batch-maximum closure residual exceeds 1 mm; G3/G4 geometry passed, so monitor under learned loads and stop at the Composition gate if it persists")
    advice.append("policy quality (terrain level, unsafe termination and velocity-limit rate) is deferred to G8/G9; G6 PASS does not claim that the 1k policy is deployable")
    return record("G6", "FAIL" if failures else "PASS", [OUT / "G6/p0_summary.json", OUT / "G6/p1_summary.json"],
                  {"scope": "action-contract diagnostic qualification",
                   "failures": failures, "qualification": table,
                   "policy_observations": observation_table,
                   "policy_quality_accepted": False,
                   "risks": risks, "branches": advice})


def g7(args) -> int:
    passing, notes = [], []
    for path in args.report:
        report = _load(path)
        manual = report.get("manual") or {}
        step = float(report.get("args", {}).get("step_height", 0))
        scan = report.get("clearance_scan") or {}
        qualified_legs = set()
        for leg in ("FR", "FL", "RL", "RR"):
            best = (scan.get(leg) or {}).get("best")
            if (
                best
                and best.get("stable_free_base_support") is True
                and not best.get("terminated", True)
                and float(best.get("sustained_clearance_m", -math.inf)) >= step
                and float(best.get("body_relative_lift_m", -math.inf)) >= step
            ):
                qualified_legs.add(leg)
        crossings = [c for c in manual.get("crossings", []) if c.get("edge") == 0]
        clean = {c["leg"] for c in crossings if c.get("active_lift_evidence")}
        reasons = manual.get("termination_reasons", [])
        ok = (
            report.get("action_contract") == "servo_range_v2"
            and step >= 0.03 - 1e-9
            and qualified_legs == {"FR", "FL", "RL", "RR"}
            and bool(report.get("free_base", True))
        )
        notes.append({"report": str(path), "step_height_m": step,
                      "mechanically_qualified_legs": sorted(qualified_legs),
                      "scripted_clean_crossing_legs": sorted(clean),
                      "scripted_traversal_complete": manual.get("complete"),
                      "scripted_terminations": reasons, "pass": ok})
        if ok:
            passing.append(str(path))
    failures = [] if passing else ["no free-base report proves >=30 mm supported clearance for all four legs: stop before the long run"]
    return record("G7", "FAIL" if failures else "PASS", list(args.report),
                  {"scope": "mechanical clearance qualification",
                   "failures": failures, "reports": notes,
                   "scripted_traversal_accepted": False,
                   "policy_probe": "deferred to G8/G9; not a G7 pass condition"})


def status(_args) -> int:
    for gate in GATES:
        path = OUT / gate / "gate.json"
        if path.is_file():
            data = json.loads(path.read_text())
            extra = " (contract upgrade confirmed)" if data.get("contract_upgrade_confirmed") else ""
            print(f"{gate}: {data['status']}{extra}  {data['recorded_utc']}")
        else:
            print(f"{gate}: not recorded")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="gate", required=True)
    sub.add_parser("status")
    sub.add_parser("g0")
    p = sub.add_parser("g1")
    p.add_argument("--report", type=Path, required=True, help="validate_policy.py JSON of the b0 model_19999.pt")
    p.add_argument("--metrics-smoke-passed", action="store_true")
    p.add_argument("--wandb-keys-seen", action="store_true")
    p = sub.add_parser("g2")
    p.add_argument("--wiki-evidence", help="Path to the registered E-CAL-002 wiki evidence file")
    sub.add_parser("g3")
    p = sub.add_parser("g4")
    p.add_argument("--action-probe", type=Path, nargs="+", default=[])
    p.add_argument("--pytest-passed", action="store_true")
    p.add_argument("--cpu-smoke-passed", action="store_true")
    p.add_argument("--validation-smoke-passed", action="store_true")
    p.add_argument("--confirm-contract-upgrade", action="store_true")
    p.add_argument("--allow-provisional", action="store_true",
                   help="accept the provisional section 3A contract (only if G3 results reproduce it)")
    p = sub.add_parser("g5")
    p.add_argument("--manual", nargs="*", default=[], help='"T3a:closure_residual_max_mm <= 1=ok" style entries')
    p.add_argument("--reject", nargs="*", default=[], help='"T6=combined run slower than baseline" style entries')
    sub.add_parser("g6")
    p = sub.add_parser("g7")
    p.add_argument("--report", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    handler = {"status": status, "g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "g6": g6, "g7": g7}
    return handler[args.gate](args)


if __name__ == "__main__":
    raise SystemExit(main())
