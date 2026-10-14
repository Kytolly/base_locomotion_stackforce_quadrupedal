#!/usr/bin/env python3
"""Export tail statistics of TensorBoard scalars from training runs to JSON.

Used by the fix plan for

* G0: the b0/b2 consolidation baseline (``output/fix_plan/G0/baseline_metrics.json``);
* G6: the P0/P1 short-probe summaries (``output/fix_plan/G6/p*_summary.json``).

Each ``--run`` is a run directory (or a glob).  The default tag set covers the
plan's comparison table; ``--tags`` adds more.  Tags are matched exactly, then
by ``/`` suffix, so ``Episode/`` prefixes from older RSL-RL loggers are found.

    python scripts/export_tf_metrics.py --label b2_consolidation \
        --run "logs/rsl_rl/base_locomotion_e0_46d/*_b2_bisec_full_consolidation_seed42" \
        --output output/fix_plan/G0/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_run_health as health  # noqa: E402


DEFAULT_TAGS = (
    "train/terrain/level",
    "train/actuation/action_saturation_rate",
    "train/actuation/leg_saturation_rate",
    "train/actuation/wheel_saturation_rate",
    "train/actuation/leg_torque_saturation_rate",
    "train/actuation/leg_torque_over_rated_rate",
    "train/actuation/leg_tracking_error_rad",
    "train/actuation/leg_action_rms",
    "train/actuation/wheel_action_rms",
    "train/actuation/joint_velocity_limit_rate",
    "train/safety/unsafe_termination",
    "train/safety/timeout",
    "train/safety/base_collision_rate",
    "train/runtime/finite_rate",
    "train/runtime/all_finite",
    "train/runtime/closure_residual_max_mm",
    "Perf/collection time",
    "Perf/collection_time",
    "Perf/learning_time",
    "Perf/learning time",
    "Perf/total_fps",
)
FAMILY_TAG_SUFFIXES = ("safety/unsafe_termination", "timeout_survival_rate")


def summarise(run: Path, tags: list[str], window: int) -> dict:
    scalars = health.load_scalars(run)
    result: dict = {"run": str(run), "window": window, "metrics": {}, "missing": []}
    wanted = list(tags)
    wanted += sorted(
        tag for tag in scalars
        if tag.startswith("train/terrain/") and tag.endswith(FAMILY_TAG_SUFFIXES)
    )
    for tag in dict.fromkeys(wanted):
        resolved = health.resolve_tag(scalars, [tag])
        if resolved is None:
            result["missing"].append(tag)
            continue
        series = scalars[resolved][-window:]
        values = [value for _, value in series]
        result["metrics"][tag] = {
            "tag": resolved,
            "mean": statistics.fmean(values),
            "last_iteration": series[-1][0],
            "points": len(values),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--label", action="append", help="One label per --run (default: run directory name).")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for pattern in args.run:
        matches = sorted(glob.glob(pattern)) or [pattern]
        runs.append(Path(matches[-1]))
    labels = args.label or [run.name for run in runs]
    if len(labels) != len(runs):
        parser.error("give one --label per --run")
    payload = {
        label: summarise(run, list(DEFAULT_TAGS) + list(args.tags), args.window)
        for label, run in zip(labels, runs)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
