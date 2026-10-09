"""Run a frozen multi-seed validation/track matrix without policy updates."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def evaluation_jobs(config, checkpoint, output, wandb_args=None):
    wandb_args = list(wandb_args or [])
    jobs = []
    for terrain_profile in config.validation.get("terrain_profiles", ["union_consolidation"]):
        for seed in config.validation.seeds:
            name = f"validation_{terrain_profile}_{seed}"
            args = ["scripts/validate_policy.py", "--checkpoint", str(checkpoint), "--seed", str(seed),
                    "--terrain-profile", str(terrain_profile),
                    "--num-envs", str(config.validation.num_envs), "--episode-steps", str(config.validation.episode_steps),
                    "--device", str(config.launcher.device), "--output", str(output / f"{name}.json"), *wandb_args]
            if config.validation.get("task"):
                args.extend(["--task", str(config.validation.task)])
            if config.launcher.viz == "none":
                args.append("--headless")
            jobs.append((name, args))
    for kind in ("plateau", "washboard"):
        for randomized in (False, True):
            for seed in config.benchmark[f"{kind}_seeds"]:
                name = f"{kind}_{'randomized' if randomized else 'nominal'}_{seed}"
                args = ["scripts/run_benchmark.py", "--config", f"configs/benchmark/{kind}.yaml",
                        f"launcher.viz={config.launcher.viz}", f"launcher.device={config.launcher.device}",
                        "policy.zero_policy=false", f"policy.checkpoint={checkpoint}",
                        f"benchmark.seed={seed}", f"benchmark.randomized={str(randomized).lower()}",
                        f"output={output / (name + '.json')}"]
                jobs.append((name, args))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/generalization.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Upload each checkpoint validation result to W&B.")
    parser.add_argument("--wandb-project", default="stackforce-quadrupedal-locomotion")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="checkpoint-validation")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if config.launcher.viz not in ("kit", "none"):
        parser.error("Suite launcher.viz must be kit or none.")
    for seeds in (config.validation.seeds, config.benchmark.plateau_seeds, config.benchmark.washboard_seeds):
        if len(seeds) < 5 or len(set(seeds)) != len(seeds):
            parser.error("Each evaluation seed list must contain at least five unique seeds.")
    training_config = checkpoint.parent / "resolved_training_config.yaml"
    if training_config.is_file():
        training_seed = OmegaConf.load(training_config).env.seed
        if training_seed in config.validation.seeds:
            parser.error("Validation seeds must be held out from the training terrain seed.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (ROOT / config.output_root / f"{checkpoint.stem}_{stamp}").resolve()
    wandb_args = []
    if args.wandb:
        wandb_args = [
            "--wandb",
            "--wandb-project", args.wandb_project,
            "--wandb-group", args.wandb_group,
            "--wandb-mode", args.wandb_mode,
        ]
        if args.wandb_entity:
            wandb_args.extend(["--wandb-entity", args.wandb_entity])
    jobs = evaluation_jobs(config, checkpoint, output, wandb_args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / "logs/evaluation" / output.name
    logs.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "config.yaml")
    results = []
    for name, command in jobs:
        print(f"[SUITE] {name}", flush=True)
        with (logs / f"{name}.log").open("w") as stream:
            process = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        path = output / f"{name}.json"
        report = json.loads(path.read_text()) if process.returncode == 0 and path.is_file() else {}
        passed = report.get("selection_evidence", {}).get("performance_pass", False) if name.startswith("validation") else report.get("success", False)
        results.append({"name": name, "exit_code": process.returncode, "pass": bool(passed), "report": str(path)})
        summary = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "expected_runs": len(jobs), "completed_runs": len(results), "runs": results,
                   "pass": len(results) == len(jobs) and all(item["pass"] for item in results)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[SUITE] report={output / 'summary.json'} pass={summary['pass']}")


if __name__ == "__main__":
    main()
