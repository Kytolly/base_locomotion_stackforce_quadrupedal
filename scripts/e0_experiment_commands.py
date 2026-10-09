#!/usr/bin/env python3
"""Print reproducible long-run commands for every E0 experiment and terrain stage."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/experiments/e0_experiments.yaml"
PYTHON = Path("/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--stage", choices=("foundation", "expansion", "composition", "consolidation"))
    parser.add_argument("--load-run", help="Exact previous-stage run directory for resume.")
    parser.add_argument("--load-checkpoint", help="Exact previous-stage checkpoint filename for resume.")
    parser.add_argument("--include-sweep", action="store_true")
    return parser.parse_args()


def _resolved_overrides(experiments, name: str) -> dict:
    experiment = experiments[name]
    result = {}
    if experiment.get("parent"):
        result.update(_resolved_overrides(experiments, str(experiment.parent)))
    result.update(OmegaConf.to_container(experiment.get("overrides", {}), resolve=True))
    return result


def _format_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(map(str, value)) + "]"
    return str(value)


def _command(matrix, name: str, seed: int, stage_name: str, overrides: dict, args) -> str:
    stage = matrix.terrain_stages[stage_name]
    run_name = f"{name}_{stage_name}_seed{seed}"
    values = {
        "env.seed": seed,
        "env.terrain_profile": stage.profile,
        "agent.max_iterations": stage.iterations,
        "agent.run_name": run_name,
        "wandb.run_name": run_name,
        "wandb.group": f"e0-{name}",
        **overrides,
    }
    if stage_name != "foundation":
        if not args.load_run or not args.load_checkpoint:
            values["agent.resume"] = True
            values["agent.load_run"] = "PREVIOUS_STAGE_RUN"
            values["agent.load_checkpoint"] = "PREVIOUS_STAGE_CHECKPOINT.pt"
        else:
            values["agent.resume"] = True
            values["agent.load_run"] = args.load_run
            values["agent.load_checkpoint"] = args.load_checkpoint
        values["runtime.allow_terrain_stage_resume"] = True
    pieces = [str(PYTHON), "scripts/rsl_rl/train.py", "--config", str(matrix.base_config)]
    pieces.extend(f"{key}={_format_value(value)}" for key, value in values.items())
    return " ".join(shlex.quote(piece) for piece in pieces)


def main() -> None:
    args = _parse_args()
    matrix = OmegaConf.load(args.matrix.resolve())
    experiments = matrix.experiments
    names = args.experiment or list(experiments)
    unknown = sorted(set(names) - set(experiments))
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")
    seeds = args.seed or list(matrix.seeds)
    stages = [args.stage] if args.stage else list(matrix.terrain_stages)
    for name in names:
        overrides = _resolved_overrides(experiments, name)
        for seed in seeds:
            for stage in stages:
                print(_command(matrix, name, seed, stage, overrides, args))
    if args.include_sweep:
        sweep = matrix.regularizer_sweep
        base = _resolved_overrides(experiments, str(sweep.parent))
        for entropy in sweep.gate_entropy_coef:
            for orthogonality in sweep.expert_orthogonality_coef:
                for temporal in sweep.temporal_consistency_coef:
                    name = f"b2_sweep_e{entropy}_o{orthogonality}_t{temporal}"
                    overrides = {
                        **base,
                        "agent.algorithm.gate_entropy_coef": entropy,
                        "agent.algorithm.expert_orthogonality_coef": orthogonality,
                        "agent.algorithm.temporal_consistency_coef": temporal,
                    }
                    for seed in seeds:
                        for stage in stages:
                            print(_command(matrix, name, seed, stage, overrides, args))


if __name__ == "__main__":
    main()
