#!/usr/bin/env python3
"""Run the E0 terrain stages as one resumable, auditable training job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/experiments/e0_experiments.yaml"
STAGES = ("foundation", "expansion", "composition", "consolidation")
FORMAL_ITERATIONS = (1000, 5000, 7000, 7000)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--start-stage", choices=STAGES, default="foundation")
    p.add_argument("--completed-stage", choices=STAGES)
    p.add_argument("--python", dest="python_executable", default=sys.executable)
    p.add_argument("--train-script", type=Path, default=ROOT / "scripts/rsl_rl/train.py")
    return p.parse_args()


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


def _write_manifest(path: Path, experiment: str, seed: int, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": experiment, "seed": seed, "stages": records}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_and_tee(command: list[str], audit) -> int:
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
        return 127
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        audit.write(line)
        audit.flush()
    return process.wait()


def main() -> int:
    args = _args()
    if args.completed_stage and args.start_stage != "foundation":
        raise ValueError("Use either --start-stage or --completed-stage, not both")
    matrix, config_path = _config(args)
    if args.completed_stage:
        completed_index = STAGES.index(args.completed_stage)
        args.start_stage = STAGES[completed_index + 1] if completed_index + 1 < len(STAGES) else None
    start_index = len(STAGES) if args.start_stage is None else STAGES.index(args.start_stage)
    base = _load_training_config(config_path)
    experiment_name = str(base.get("agent", {}).get("experiment_name", ""))
    log_parent = (ROOT / str(base.get("runtime", {}).get("log_root", "logs/rsl_rl")) / experiment_name).resolve()
    orchestration_log = (ROOT / "logs/e0_orchestration" / f"{args.experiment}_seed{args.seed}.log").resolve()
    orchestration_log.parent.mkdir(parents=True, exist_ok=True)
    manifest = orchestration_log.with_suffix(".json")
    records: list[dict[str, Any]] = []

    previous_run: Path | None = None
    previous_checkpoint: str | None = None
    if start_index:
        prev_index = start_index - 1
        prev_final, _ = _expected(matrix.terrain_stages, prev_index)
        prev_stage = STAGES[prev_index]
        prev_name = f"{args.experiment}_{prev_stage}_seed{args.seed}"
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
            run_name = f"{args.experiment}_{stage}_seed{args.seed}"
            checkpoint_name = f"model_{final_iteration}.pt"
            overrides = [
                f"env.seed={args.seed}",
                f"env.terrain_profile={stage_cfg.profile}",
                f"agent.max_iterations={int(stage_cfg.iterations)}",
                f"agent.run_name={run_name}",
                f"wandb.run_name={run_name}",
            ]
            if previous_run is None:
                overrides.append("agent.resume=false")
            else:
                overrides.extend([
                    "agent.resume=true",
                    f"agent.load_run={previous_run.name}",
                    f"agent.load_checkpoint={previous_checkpoint}",
                    "runtime.allow_terrain_stage_resume=true",
                ])
            command = [args.python_executable, str(args.train_script), "--config", str(config_path), *overrides]
            print("[E0] " + " ".join(command), flush=True)
            audit.write("[E0] " + " ".join(command) + "\n")
            audit.flush()
            launched_at = time.time()
            return_code = _run_and_tee(command, audit)
            if return_code != 0:
                print(f"[E0] {stage} failed with exit code {return_code}; logs retained at {orchestration_log}", file=sys.stderr)
                return return_code
            run = _find_run(log_parent, run_name, checkpoint_name, final_iteration, after=launched_at)
            state = _load_checkpoint(run / checkpoint_name)
            record = {
                "stage": stage,
                "profile": str(stage_cfg.profile),
                "iterations": int(stage_cfg.iterations),
                "run": str(run),
                "checkpoint": str(run / checkpoint_name),
                "iteration": int(state["iter"]),
            }
            records.append(record)
            _write_manifest(manifest, args.experiment, args.seed, records)
            print(f"[E0] {stage} complete: {run / checkpoint_name}")
            previous_run, previous_checkpoint = run, checkpoint_name
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
