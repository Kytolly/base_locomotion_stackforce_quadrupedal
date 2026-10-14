#!/usr/bin/env python3
"""Read E0 TensorBoard scalars for stage gates, abort checks and level inheritance.

This module is imported by ``e0_training_orchestrator.py``.  It never changes a
run; it only reads ``events.out.tfevents.*`` written by the RSL-RL logger (the
W&B logger keeps writing local TensorBoard events as well).

Run it directly to inspect one finished stage::

    python scripts/e0_run_health.py --run logs/rsl_rl/<exp>/<run> --stage expansion
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES = ROOT / "configs/experiments/e0_v2_stage_gates.yaml"

Series = list[tuple[int, float]]
Scalars = dict[str, Series]

# PhysX writes these when a GPU buffer overflows; the run is no longer trustworthy.
PHYSX_BUFFER_ERROR = re.compile(
    r"(PhysX error.*(buffer|capacity|overflow))|"
    r"(GPU.*(found_lost|aggregate).*(overflow|exceed|insufficient))|"
    r"(increase.*gpu_(found_lost|total_aggregate|max_rigid|heap)\w*)",
    re.IGNORECASE,
)


def load_scalars(run_dir: Path) -> Scalars:
    """Return every scalar series in ``run_dir`` as ``{tag: [(step, value), ...]}``."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover - env_isaaclab ships tensorboard
        raise RuntimeError("Reading training scalars requires tensorboard in env_isaaclab") from exc
    files = sorted(Path(run_dir).rglob("events.out.tfevents.*"))
    merged: dict[str, dict[int, float]] = {}
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            target = merged.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                target[int(event.step)] = float(event.value)
    return {tag: sorted(values.items()) for tag, values in merged.items()}


def resolve_tag(scalars: Mapping[str, Series], candidates: Sequence[str]) -> str | None:
    """Pick the first candidate present exactly, then by ``/``-suffix match."""
    for candidate in candidates:
        if candidate in scalars and scalars[candidate]:
            return candidate
    for candidate in candidates:
        suffix = "/" + candidate.strip("/")
        matches = sorted(tag for tag in scalars if tag.endswith(suffix) and scalars[tag])
        if matches:
            return matches[0]
    return None


def _values(series: Series) -> list[float]:
    return [value for _, value in series if math.isfinite(value)]


def window_stat(series: Series, stat: str, window: int) -> float | None:
    """Apply ``stat`` to the last (``tail_*``) or first (``head_*``) ``window`` points."""
    if not series:
        return None
    position, _, reducer = stat.partition("_")
    if position not in {"tail", "head"} or reducer not in {"mean", "median", "min", "max", "last"}:
        raise ValueError(f"Unknown statistic: {stat!r}")
    chunk = series[-window:] if position == "tail" else series[:window]
    values = _values(chunk)
    if not values:
        return None
    if reducer == "mean":
        return statistics.fmean(values)
    if reducer == "median":
        return statistics.median(values)
    if reducer == "min":
        return min(values)
    if reducer == "max":
        return max(values)
    return values[-1]


def inherited_terrain_level(
    scalars: Mapping[str, Series], candidates: Sequence[str], *, window: int = 100, offset: int = 0
) -> tuple[int, dict[str, Any]]:
    """T1: median level of the last ``window`` iterations, rounded, minus ``offset``."""
    tag = resolve_tag(scalars, candidates)
    if tag is None:
        raise KeyError(f"No terrain-level scalar found; looked for {list(candidates)}")
    median = window_stat(scalars[tag], "tail_median", window)
    if median is None:
        raise ValueError(f"Terrain-level scalar {tag} has no finite values")
    level = max(0, int(round(median)) - int(offset))
    return level, {"tag": tag, "window": window, "median": median, "offset": offset, "level": level}


_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}


def _threshold(value: Any, context: Mapping[str, Any]) -> float:
    """Thresholds may reference context, e.g. ``inherited_level-1``."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"\s*([a-z_]+)\s*(?:([+-])\s*([0-9.]+))?\s*", str(value))
    if not match or match.group(1) not in context:
        raise ValueError(f"Unresolvable gate threshold {value!r}")
    base = float(context[match.group(1)])
    if match.group(2):
        delta = float(match.group(3))
        base = base + delta if match.group(2) == "+" else base - delta
    return base


def evaluate_checks(
    scalars: Mapping[str, Series], checks: Iterable[Mapping[str, Any]], context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate gate checks; a missing metric is MISSING, which blocks like FAIL."""
    context = dict(context or {})
    results = []
    for check in checks:
        name = str(check["name"])
        tag = resolve_tag(scalars, list(check["tags"]))
        entry: dict[str, Any] = {"name": name, "tag": tag, "op": check["op"]}
        try:
            threshold = _threshold(check["value"], context)
        except ValueError as exc:
            entry.update(status="MISSING", reason=str(exc))
            results.append(entry)
            continue
        entry["threshold"] = threshold
        if tag is None:
            entry.update(status="MISSING", reason=f"none of {list(check['tags'])} were logged")
            results.append(entry)
            continue
        value = window_stat(scalars[tag], str(check.get("stat", "tail_mean")), int(check.get("window", 300)))
        entry["value"] = value
        if value is None:
            entry.update(status="MISSING", reason="no finite values")
        else:
            entry["status"] = "PASS" if _OPS[str(check["op"])](value, threshold) else "FAIL"
        results.append(entry)
    status = "PASS" if results and all(item["status"] == "PASS" for item in results) else "FAIL"
    if not results:
        status = "PASS"
    return {"status": status, "checks": results, "context": context}


def load_gate_spec(path: Path = DEFAULT_GATES) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or "stages" not in spec:
        raise ValueError(f"Gate spec {path} must contain a 'stages' mapping")
    return spec


def _slope(series: Series) -> float:
    if len(series) < 2:
        return 0.0
    xs = [float(step) for step, _ in series]
    ys = [value for _, value in series]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def abort_reasons(scalars: Mapping[str, Series], spec: Mapping[str, Any]) -> list[str]:
    """G8 abort conditions that may fire at any time during a stage."""
    abort = spec.get("abort", {})
    reasons = []
    finite_tag = resolve_tag(scalars, abort.get("finite_tags", []))
    if finite_tag is not None:
        bad = [step for step, value in scalars[finite_tag] if not math.isfinite(value) or value < 0.5]
        if bad:
            reasons.append(f"{finite_tag} reported 0 at iteration {bad[0]}")
    unsafe_tag = resolve_tag(scalars, abort.get("unsafe_tags", []))
    if unsafe_tag is not None:
        window = int(abort.get("unsafe_window", 500))
        limit = float(abort.get("unsafe_limit", 0.30))
        series = scalars[unsafe_tag][-window:]
        if len(series) >= window and all(value > limit for _, value in series) and _slope(series) > 0:
            reasons.append(
                f"{unsafe_tag} stayed above {limit:.2f} for {window} iterations and is still rising"
            )
    return reasons


def stage_gate(run_dir: Path, stage: str, spec: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scalars = load_scalars(run_dir)
    checks = spec["stages"].get(stage, [])
    report = evaluate_checks(scalars, checks, context)
    report.update(stage=stage, run=str(run_dir), abort=abort_reasons(scalars, spec))
    if report["abort"]:
        report["status"] = "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--inherited-level", type=float)
    parser.add_argument("--list-tags", action="store_true")
    args = parser.parse_args()
    if args.list_tags:
        for tag in sorted(load_scalars(args.run)):
            print(tag)
        return 0
    context = {} if args.inherited_level is None else {"inherited_level": args.inherited_level}
    report = stage_gate(args.run, args.stage, load_gate_spec(args.gates), context)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
