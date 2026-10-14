#!/usr/bin/env python3
"""Training runs for fix-plan gates G5 (throughput) and G6 (short probes).

Every run goes through ``scripts/rsl_rl/train.py`` with dotlist overrides on
``configs/train/main/b2_bisec_full.yaml`` (seed 42, 4096 env), into its own
experiment directory so it never mixes with formal runs.

G5 (one change at a time, 200 iterations, mean ``Perf/collection time`` of the
last 100)::

    python scripts/fix_plan_runs.py g5                 # baseline + T2..T7
    python scripts/fix_plan_runs.py g5 --only T3a T3b  # re-run some items
    python scripts/fix_plan_runs.py g5 --combined env.throughput.foot_track_pose=false ...

G6 (1500 iterations on ``probe_low_steps`` with 0.02-0.04 m steps)::

    python scripts/fix_plan_runs.py g6                 # P0 (legacy legs) and P1 (servo ranges)
    python scripts/fix_plan_runs.py g6 --only P1 --extra agent.actor.distribution_cfg.init_std=[...]

Outputs: ``output/fix_plan/G5/timing_matrix.json`` and
``output/fix_plan/G6/{p0,p1}_summary.json``; ``scripts/fix_plan_gate.py``
turns them into gate results.  Item physics checks that need a person or
another probe are listed as ``manual_checks`` and must be recorded there.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_run_health as health  # noqa: E402
from export_tf_metrics import DEFAULT_TAGS, summarise  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("E0_PYTHON", "/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python")
CONFIG = ROOT / "configs/train/main/b2_bisec_full.yaml"
COLLECTION_TAGS = ("Perf/collection time", "Perf/collection_time")
BISEC_LABEL = "BiSECv2"

COMMON = [
    "env.seed=42",
    "env.num_envs=4096",
    "env.terrain_profile=union_foundation",
    "wandb.mode=offline",
    "agent.save_interval=100000",
]

G5_ITEMS = {
    "baseline": ([], [], "E0-v2 configuration as committed"),
    "T2": (["env.throughput.enabled_self_collisions=false"],
           ["G3 range scan with self-collisions off shows no leg interpenetration",
            "base_collision rate and standing-height error unchanged"],
           "self-collision off"),
    "T3a": (["env.throughput.solver_position_iteration_count=4"],
            ["closure_residual_max_mm <= 1", "body-height RMSE change < 2 mm"],
            "solver position iterations 8 -> 4"),
    "T3b": (["env.throughput.solver_position_iteration_count=6"],
            ["closure_residual_max_mm <= 1", "body-height RMSE change < 2 mm"],
            "solver position iterations 8 -> 6 (if 4 fails)"),
    "T4": (["env.throughput.foot_track_pose=false"],
           ["no reward or metric reads contact-sensor pose (verified by code search: none)"],
           "foot contact sensors stop tracking pose"),
    "T5": ([f"logging.metrics.groups.{g}=false" for g in ("command", "support", "motion", "reward")]
           + ["logging.wandb_panels.enabled=false"],
           ["curriculum keys (terrain level, tracking RMSE, progress) still logged"],
           "training metrics: core/safety/runtime/actuation/terrain only"),
    "T6": (["env.throughput.foot_contact_update_period=0.02"],
           ["wheel contact fraction and air-time-based metrics stay plausible (no zeros or jumps)"],
           "foot contact sensors update at the 50 Hz control rate"),
    "T7": (["env.throughput.replicate_physics=true"],
           ["per-env mass/material randomisation still differs between envs", "closure residual normal"],
           "replicate_physics true"),
}

# The robot's nested base/wheel rigid bodies require PhysxContactReportAPI to be
# authored per environment before simulation starts.  Isaac Lab intentionally
# rejects pre-startup USD edits when physics replication is enabled because the
# replicated instances share those properties.  Keep T7 in the matrix as an
# evaluated candidate, but do not repeatedly launch a configuration that cannot
# create the environment.
G5_REJECTED = {
    "T7": (
        "replicate_physics=true is incompatible with the required pre-startup "
        "per-environment contact-report activation; Isaac Lab rejects this combination"
    ),
}

G6_GROUPS = {
    # E0-v1 legs and the E0-v1 scalar exploration std (written per dimension).
    "P0": ["env.action_contract=legacy_symmetric", "agent.actor.distribution_cfg.init_std=[" + ",".join(["0.3"] * 12) + "]"],
    "P1": [],
}
G6_COMMON = [
    "env.terrain_profile=probe_low_steps",
    "env.obstacle_heights.pyramid_stairs=[0.02,0.04]",
    "env.obstacle_heights.boxes=[0.02,0.04]",
]


def _train(run_name: str, experiment: str, iterations: int, overrides: list[str], log: Path) -> Path:
    command = [
        PYTHON, str(ROOT / "scripts/rsl_rl/train.py"), "--config", str(CONFIG), *COMMON,
        f"agent.max_iterations={iterations}", f"agent.experiment_name={experiment}",
        f"agent.run_name={run_name}", f"wandb.run_name={run_name}", "agent.resume=false", *overrides,
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError(f"{run_name} failed with exit code {code}; see {log}")
    runs = [p for p in (ROOT / "logs/rsl_rl" / experiment).glob(f"*_{run_name}") if p.stat().st_mtime >= started]
    if not runs:
        raise FileNotFoundError(f"no run directory for {run_name}")
    return max(runs, key=lambda p: p.stat().st_mtime_ns)


def _collection_time(run: Path, window: int = 100) -> float | None:
    scalars = health.load_scalars(run)
    tag = health.resolve_tag(scalars, COLLECTION_TAGS)
    return None if tag is None else health.window_stat(scalars[tag], "tail_mean", window)


def g5(args) -> None:
    out = ROOT / "output/fix_plan/G5"
    path = out / "timing_matrix.json"
    matrix = json.loads(path.read_text()) if path.is_file() else {"items": {}}
    items = dict(G5_ITEMS)
    if args.combined is not None:
        items = {"combined": (list(args.combined), ["all kept items' checks re-verified"], "kept items together")}
    for name, (overrides, checks, description) in items.items():
        if args.only and name not in args.only:
            continue
        if name in G5_REJECTED:
            matrix["items"][name] = {
                "description": description,
                "overrides": overrides,
                "status": "rejected",
                "rejected_reason": G5_REJECTED[name],
                "log": str(out / f"{name}.log"),
                "collection_time_s": None,
                "metrics": {},
                "manual_checks": {},
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
            print(f"recorded {name} as rejected: {G5_REJECTED[name]}")
            continue
        run_name = f"{datetime.now():%Y%m%d-%H%M}-{BISEC_LABEL}-g5-{name.lower()}-seed42"
        run = _train(run_name, "fix_plan_g5_timing", 200, overrides, out / f"{name}.log")
        summary = summarise(run, ["train/locomotion/body_height_rmse_m", "train/support/wheel_contact_fraction"], 100)
        matrix["items"][name] = {
            "description": description,
            "overrides": overrides,
            "status": "complete",
            "run": str(run),
            "collection_time_s": _collection_time(run),
            "metrics": summary["metrics"],
            "manual_checks": {check: None for check in checks},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    base = matrix["items"].get("baseline", {}).get("collection_time_s")
    for name, item in matrix["items"].items():
        value = item.get("collection_time_s")
        item["gain_vs_baseline"] = None if not base or value is None else (base - value) / base
    path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def g6(args) -> None:
    out = ROOT / "output/fix_plan/G6"
    existing = dict(item.split("=", 1) for item in args.existing_run)
    for group, overrides in G6_GROUPS.items():
        if args.only and group not in args.only:
            continue
        if group in existing:
            run = Path(existing[group]).resolve()
            if not run.is_dir():
                raise FileNotFoundError(f"existing {group} run does not exist: {run}")
        else:
            run_name = f"{datetime.now():%Y%m%d-%H%M}-{BISEC_LABEL}-g6-{group.lower()}-seed42"
            run = _train(run_name, "fix_plan_g6_probe", args.iterations,
                         G6_COMMON + overrides + list(args.extra), out / f"{group.lower()}.log")
        summary = summarise(
            run,
            list(DEFAULT_TAGS)
            + [
                "train/terrain/pyramid_stairs/safety/unsafe_termination",
                "train/terrain/boxes/safety/unsafe_termination",
            ],
            300,
        )
        summary.update(group=group, overrides=G6_COMMON + overrides + list(args.extra))
        target = out / f"{group.lower()}_summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="gate", required=True)
    p5 = sub.add_parser("g5")
    p5.add_argument("--only", nargs="*")
    p5.add_argument("--combined", nargs="*", help="Overrides of all kept items, timed together.")
    p6 = sub.add_parser("g6")
    p6.add_argument("--only", nargs="*")
    p6.add_argument("--iterations", type=int, default=1500)
    p6.add_argument("--extra", nargs="*", default=[], help="Extra overrides, e.g. the one allowed leg init_std change.")
    p6.add_argument(
        "--existing-run",
        nargs="*",
        default=[],
        help="Summarize an interrupted run without relaunching it, e.g. P0=/path/to/run.",
    )
    args = parser.parse_args()
    g5(args) if args.gate == "g5" else g6(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
