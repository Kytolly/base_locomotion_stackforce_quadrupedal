#!/usr/bin/env python3
"""Run the E0 terrain stages as one resumable, auditable training job.

E0-v2 additions (docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md):

* T1 terrain-level inheritance: each resumed stage starts its environments in
  the level band ``[m-1, m]``, where ``m`` is the rounded median terrain level of
  the previous stage's last 100 logged iterations.
* G8 stage-end gates: after every stage the scalars are checked against
  ``configs/experiments/e0_v2_stage_gates.yaml``.  A failed or missing check
  pauses the lineage (exit code 3) so the run can be reviewed and resumed with
  ``--completed-stage``.
* G8 abort conditions: a PhysX buffer error in the training output, a zero
  ``all_finite`` scalar, or ``unsafe_termination`` above 30 % for 500 rising
  iterations terminates the stage (exit code 4).

All three are on by default for the formal matrix and off for custom matrices.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/experiments/e0_experiments.yaml"
STAGES = ("foundation", "expansion", "composition", "consolidation")
FORMAL_ITERATIONS = (1000, 5000, 7000, 7000)
GATE_PAUSE_EXIT = 3
ABORT_EXIT = 4
LEVEL_TAGS = ("train/terrain/level", "Curriculum/terrain_levels")
RUN_LABELS = {
    "b0_t1_ppo": "simplePPOv2",
    "b2_bisec_full": "BiSECv2",
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_run_health as health  # noqa: E402


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--start-stage", choices=STAGES, default="foundation")
    p.add_argument("--completed-stage", choices=STAGES)
    p.add_argument("--python", dest="python_executable", default=sys.executable)
    p.add_argument("--train-script", type=Path, default=ROOT / "scripts/rsl_rl/train.py")
    p.add_argument("--level-inheritance", choices=("auto", "on", "off"), default="auto",
                   help="T1: start resumed stages near the previous stage's terrain level.")
    p.add_argument("--level-offset", type=int, default=0,
                   help="Subtract this from the inherited level (use 1 if resumes show difficulty jumps).")
    p.add_argument("--level-window", type=int, default=100)
    p.add_argument("--gates", choices=("auto", "on", "off"), default="auto",
                   help="G8: pause the lineage when a stage-end check fails.")
    p.add_argument("--gate-spec", type=Path, default=health.DEFAULT_GATES)
    p.add_argument("--abort-monitor", choices=("auto", "on", "off"), default="auto",
                   help="G8: terminate a stage on PhysX buffer errors, non-finite values or runaway unsafe rate.")
    p.add_argument("--health-poll-s", type=float, default=300.0)
    p.add_argument("--require-protocol", help="Refuse to start unless the config has this protocol_version.")
    p.add_argument("--dry-run", action="store_true", help="Print the first stage command and exit.")
    return p.parse_args()


def _enabled(value: str, formal: bool) -> bool:
    return formal if value == "auto" else value == "on"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - active IsaacLab env supplies torch
        raise RuntimeError("checkpoint validation requires torch in env_isaaclab") from exc
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    required = {"iter", "actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Checkpoint {path} is missing runner state: {', '.join(missing)}")
    return state


def _config(args: argparse.Namespace):
    matrix = OmegaConf.load(args.matrix.resolve())
    if args.experiment not in matrix.experiments:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    stage_map = matrix.terrain_stages
    if tuple(stage_map.keys()) != STAGES:
        raise ValueError(f"E0 stage order must be {STAGES}, got {tuple(stage_map.keys())}")
    iterations = tuple(int(stage_map[name].iterations) for name in STAGES)
    if args.matrix.resolve() == DEFAULT_MATRIX.resolve() and iterations != FORMAL_ITERATIONS:
        raise ValueError(
            f"Formal E0 stage iterations must be {FORMAL_ITERATIONS}, got {iterations}"
        )
    config_path = (ROOT / str(matrix.experiments[args.experiment].config)).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return matrix, config_path


def _load_training_config(path: Path):
    module_path = ROOT / (
        "source/base_locomotion_stackforce_quadrupedal/"
        "base_locomotion_stackforce_quadrupedal/training/config.py"
    )
    spec = importlib.util.spec_from_file_location("_e0_training_config", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(path)


def _expected(stage_map, stage_index: int) -> tuple[int, int]:
    cumulative = sum(int(stage_map[STAGES[i]].iterations) for i in range(stage_index + 1))
    return cumulative - 1, cumulative


def _run_name(experiment: str, stage: str, seed: int) -> str:
    """Return the stable user-facing lineage name; train.py prepends wall-clock time."""
    label = RUN_LABELS.get(experiment)
    return f"{label}-{stage}-seed{seed:02d}" if label else f"{experiment}_{stage}_seed{seed}"


def _valid_run(run: Path, checkpoint_name: str, expected_iter: int) -> bool:
    checkpoint = run / checkpoint_name
    if not run.is_dir() or not checkpoint.is_file():
        return False
    if not (run / "training_contract.json").is_file() or not (run / "agent.yaml").is_file():
        return False
    try:
        return int(_load_checkpoint(checkpoint)["iter"]) == expected_iter
    except (OSError, ValueError, RuntimeError, TypeError):
        return False


def _find_run(log_parent: Path, run_name: str, checkpoint_name: str, expected_iter: int, *, after: float = 0.0) -> Path:
    candidates = [
        path for path in log_parent.glob(f"*_{run_name}")
        if path.stat().st_mtime >= after and _valid_run(path, checkpoint_name, expected_iter)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No valid {run_name} checkpoint {checkpoint_name} (iter={expected_iter}) under {log_parent}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _AbortMonitor:
    """Poll the live run directory and stop the trainer when an abort condition fires."""

    def __init__(self, process: subprocess.Popen, log_parent: Path, run_name: str, launched_at: float,
                 spec: dict[str, Any], poll_s: float):
        self.process, self.log_parent, self.run_name = process, log_parent, run_name
        self.launched_at, self.spec, self.poll_s = launched_at, spec, poll_s
        self.reasons: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def trip(self, reason: str) -> None:
        if not self.reasons:
            self.reasons.append(reason)
            if self.process.poll() is None:
                self.process.terminate()

    def _live_run(self) -> Path | None:
        runs = [p for p in self.log_parent.glob(f"*_{self.run_name}") if p.stat().st_mtime >= self.launched_at]
        return max(runs, key=lambda p: p.stat().st_mtime_ns) if runs else None

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            run = self._live_run()
            if run is None:
                continue
            try:
                reasons = health.abort_reasons(health.load_scalars(run), self.spec)
            except Exception as exc:  # noqa: BLE001 - monitoring must never kill a healthy run
                print(f"[E0] health poll skipped: {exc}", flush=True)
                continue
            if reasons:
                self.trip("; ".join(reasons))
                return


def _run_and_tee(command: list[str], audit, monitor_factory=None) -> tuple[int, list[str]]:
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        message = f"[E0] unable to start training: {exc}\n"
        sys.stderr.write(message)
        audit.write(message)
        audit.flush()
        return 127, []
    monitor = monitor_factory(process) if monitor_factory else None
    if monitor:
        monitor.start()
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        audit.write(line)
        audit.flush()
        if monitor and health.PHYSX_BUFFER_ERROR.search(line):
            monitor.trip(f"PhysX buffer error: {line.strip()[:200]}")
    code = process.wait()
    reasons: list[str] = []
    if monitor:
        monitor.stop()
        reasons = monitor.reasons
    return code, reasons


def _inherit_level(run: Path, args: argparse.Namespace) -> dict[str, Any]:
    scalars = health.load_scalars(run)
    _, info = health.inherited_terrain_level(
        scalars, LEVEL_TAGS, window=args.level_window, offset=args.level_offset
    )
    return info


def main() -> int:
    args = _args()
    if args.completed_stage and args.start_stage != "foundation":
        raise ValueError("Use either --start-stage or --completed-stage, not both")
    matrix, config_path = _config(args)
    formal = args.matrix.resolve() == DEFAULT_MATRIX.resolve()
    use_levels = _enabled(args.level_inheritance, formal)
    use_gates = _enabled(args.gates, formal)
    use_abort = _enabled(args.abort_monitor, formal)
    gate_spec = health.load_gate_spec(args.gate_spec) if (use_gates or use_abort) else {}
    if args.completed_stage:
        completed_index = STAGES.index(args.completed_stage)
        args.start_stage = STAGES[completed_index + 1] if completed_index + 1 < len(STAGES) else None
    start_index = len(STAGES) if args.start_stage is None else STAGES.index(args.start_stage)
    base = _load_training_config(config_path)
    if args.require_protocol and str(base.get("protocol_version", "")) != args.require_protocol:
        raise ValueError(
            f"{config_path} has protocol_version={base.get('protocol_version')!r}, "
            f"expected {args.require_protocol!r}"
        )
    experiment_name = str(base.get("agent", {}).get("experiment_name", ""))
    log_parent = (ROOT / str(base.get("runtime", {}).get("log_root", "logs/rsl_rl")) / experiment_name).resolve()
    orchestration_log = (ROOT / "logs/e0_orchestration" / f"{args.experiment}_seed{args.seed}.log").resolve()
    orchestration_log.parent.mkdir(parents=True, exist_ok=True)
    manifest = orchestration_log.with_suffix(".json")
    previous_manifest = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    records: list[dict[str, Any]] = [
        record for record in previous_manifest.get("stages", [])
        if record.get("stage") in STAGES[:start_index]
    ]
    manifest_payload: dict[str, Any] = {
        "experiment": args.experiment,
        "seed": args.seed,
        "protocol_version": str(base.get("protocol_version", "")),
        "config": str(config_path),
        "level_inheritance": use_levels,
        "stage_gates": use_gates,
        "abort_monitor": use_abort,
        "stages": records,
    }

    previous_run: Path | None = None
    previous_checkpoint: str | None = None
    if start_index:
        prev_index = start_index - 1
        prev_final, _ = _expected(matrix.terrain_stages, prev_index)
        prev_stage = STAGES[prev_index]
        prev_name = _run_name(args.experiment, prev_stage, args.seed)
        previous_run = _find_run(log_parent, prev_name, f"model_{prev_final}.pt", prev_final)
        previous_checkpoint = f"model_{prev_final}.pt"
        print(f"[E0] resume source: {previous_run.name}/{previous_checkpoint}")
    if start_index == len(STAGES):
        print("[E0] consolidation checkpoint verified; all E0 stages are complete.")
        return 0

    with orchestration_log.open("a", encoding="utf-8") as audit:
        for index in range(start_index, len(STAGES)):
            stage = STAGES[index]
            stage_cfg = matrix.terrain_stages[stage]
            final_iteration, _ = _expected(matrix.terrain_stages, index)
            run_name = _run_name(args.experiment, stage, args.seed)
            checkpoint_name = f"model_{final_iteration}.pt"
            overrides = [
                f"env.seed={args.seed}",
                f"env.terrain_profile={stage_cfg.profile}",
                f"agent.max_iterations={int(stage_cfg.iterations)}",
                f"agent.run_name={run_name}",
                f"wandb.run_name={run_name}",
            ]
            level_info: dict[str, Any] | None = None
            if previous_run is None:
                overrides.append("agent.resume=false")
            else:
                overrides.extend([
                    "agent.resume=true",
                    f"agent.load_run={previous_run.name}",
                    f"agent.load_checkpoint={previous_checkpoint}",
                    "runtime.allow_terrain_stage_resume=true",
                ])
                if use_levels:
                    level_info = _inherit_level(previous_run, args)
                    overrides.append(f"env.terrain_init_level={level_info['level']}")
                    print(f"[E0] T1 inherited terrain level: {json.dumps(level_info)}", flush=True)
            command = [args.python_executable, str(args.train_script), "--config", str(config_path), *overrides]
            print("[E0] " + " ".join(command), flush=True)
            audit.write("[E0] " + " ".join(command) + "\n")
            audit.flush()
            if args.dry_run:
                return 0
            launched_at = time.time()
            factory = None
            if use_abort:
                def factory(process, _name=run_name, _at=launched_at):
                    return _AbortMonitor(process, log_parent, _name, _at, gate_spec, args.health_poll_s)
            return_code, abort = _run_and_tee(command, audit, factory)
            if abort:
                message = f"[E0] {stage} aborted: {'; '.join(abort)}; logs retained at {orchestration_log}"
                print(message, file=sys.stderr)
                audit.write(message + "\n")
                manifest_payload["aborted"] = {"stage": stage, "reasons": abort}
                _write_manifest(manifest, manifest_payload)
                return ABORT_EXIT
            if return_code != 0:
                print(f"[E0] {stage} failed with exit code {return_code}; logs retained at {orchestration_log}", file=sys.stderr)
                return return_code
            run = _find_run(log_parent, run_name, checkpoint_name, final_iteration, after=launched_at)
            state = _load_checkpoint(run / checkpoint_name)
            record: dict[str, Any] = {
                "stage": stage,
                "profile": str(stage_cfg.profile),
                "iterations": int(stage_cfg.iterations),
                "run": str(run),
                "checkpoint": str(run / checkpoint_name),
                "iteration": int(state["iter"]),
            }
            if level_info is not None:
                record["inherited_terrain_level"] = level_info
            if use_gates:
                context = {} if level_info is None else {"inherited_level": level_info["level"]}
                report = health.stage_gate(run, stage, gate_spec, context)
                gate_path = orchestration_log.with_name(
                    f"{args.experiment}_seed{args.seed}_{stage}_gate.json"
                )
                gate_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                record["gate"] = {"status": report["status"], "report": str(gate_path)}
            records.append(record)
            _write_manifest(manifest, manifest_payload)
            print(f"[E0] {stage} complete: {run / checkpoint_name}")
            if use_gates and record["gate"]["status"] != "PASS":
                failed = [c["name"] for c in report["checks"] if c["status"] != "PASS"] + report["abort"]
                print(
                    f"[E0] {stage} stage gate did not pass ({', '.join(failed)}); see {record['gate']['report']}.\n"
                    f"[E0] Review, then continue with --completed-stage {stage}.",
                    file=sys.stderr,
                )
                return GATE_PAUSE_EXIT
            previous_run, previous_checkpoint = run, checkpoint_name
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
