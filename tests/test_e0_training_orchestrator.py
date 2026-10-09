import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "scripts/e0_training_orchestrator.py"


def test_formal_matrix_has_20k_cumulative_endpoints():
    matrix = OmegaConf.load(ROOT / "configs/experiments/e0_experiments.yaml")
    iterations = [int(matrix.terrain_stages[name].iterations) for name in matrix.terrain_stages]
    cumulative = []
    total = 0
    for count in iterations:
        total += count
        cumulative.append(total - 1)
    assert iterations == [1000, 5000, 7000, 7000]
    assert cumulative == [999, 5999, 12999, 19999]


def _fixture(tmp_path: Path, *, fail_stage: str | None = None):
    log_root = tmp_path / "training"
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump({"agent": {"experiment_name": "test_e0"}, "runtime": {"log_root": str(log_root)}}),
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump({
            "terrain_stages": {
                "foundation": {"profile": "union_foundation", "iterations": 1},
                "expansion": {"profile": "union_expansion", "iterations": 2},
                "composition": {"profile": "union_composition", "iterations": 3},
                "consolidation": {"profile": "union_consolidation", "iterations": 4},
            },
            "experiments": {"smoke": {"config": str(config)}},
        }, sort_keys=False), encoding="utf-8",
    )
    trainer = tmp_path / "fake_train.py"
    trainer.write_text(
        """import json, os, pathlib, sys, time, torch
values = dict(item.split('=', 1) for item in sys.argv[sys.argv.index('--config') + 2:])
stage = values['agent.run_name'].split('_')[-2]
root = pathlib.Path(os.environ['FAKE_LOG_ROOT']) / 'test_e0'
run = root / (str(time.time_ns()) + '_' + values['agent.run_name'])
run.mkdir(parents=True)
print('LIVE_OUTPUT:' + stage, flush=True)
(run / 'invocation.json').write_text(json.dumps(values))
(run / 'training_contract.json').write_text('{}')
(run / 'agent.yaml').write_text('{}')
if stage == os.environ.get('FAIL_STAGE'):
    (run / 'failure-retained.txt').write_text('retained')
    raise SystemExit(9)
if values.get('agent.resume') == 'true':
    previous = root / values['agent.load_run'] / values['agent.load_checkpoint']
    state = torch.load(previous, map_location='cpu', weights_only=False)
    start = state['iter'] + 1
else:
    start = 0
final = start + int(values['agent.max_iterations']) - 1
state = {'iter': final, 'actor_state_dict': {'w': torch.tensor([final])},
         'critic_state_dict': {'w': torch.tensor([final])},
         'optimizer_state_dict': {'state': {0: {'step': torch.tensor(final + 1)}}}}
torch.save(state, run / f'model_{final}.pt')
""", encoding="utf-8")
    env = {**os.environ, "FAKE_LOG_ROOT": str(log_root)}
    if fail_stage:
        env["FAIL_STAGE"] = fail_stage
    return matrix, trainer, log_root, env


def _run(matrix, trainer, env, *extra):
    return subprocess.run(
        [sys.executable, str(ORCHESTRATOR), "--experiment", "smoke", "--seed", "7",
         "--matrix", str(matrix), "--python", sys.executable, "--train-script", str(trainer), *extra],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )


def test_four_stages_use_exact_previous_checkpoint_and_profiles(tmp_path):
    matrix, trainer, log_root, env = _fixture(tmp_path)
    result = _run(matrix, trainer, env)
    assert result.returncode == 0, result.stderr
    assert [line for line in result.stdout.splitlines() if line.startswith("LIVE_OUTPUT:")] == [
        "LIVE_OUTPUT:foundation",
        "LIVE_OUTPUT:expansion",
        "LIVE_OUTPUT:composition",
        "LIVE_OUTPUT:consolidation",
    ]
    runs = sorted((log_root / "test_e0").iterdir(), key=lambda p: p.stat().st_mtime_ns)
    assert len(runs) == 4
    expected = [
        ("foundation", "union_foundation", 0),
        ("expansion", "union_expansion", 2),
        ("composition", "union_composition", 5),
        ("consolidation", "union_consolidation", 9),
    ]
    for index, (stage, profile, final) in enumerate(expected):
        invocation = json.loads((runs[index] / "invocation.json").read_text())
        assert invocation["env.terrain_profile"] == profile
        checkpoint = runs[index] / f"model_{final}.pt"
        assert torch.load(checkpoint, weights_only=False)["iter"] == final
        if index:
            assert invocation["agent.load_run"] == runs[index - 1].name
            assert invocation["agent.load_checkpoint"] == f"model_{expected[index - 1][2]}.pt"


def test_resume_after_completed_stage_and_failure_retains_artifacts(tmp_path):
    matrix, trainer, log_root, env = _fixture(tmp_path, fail_stage="expansion")
    failed = _run(matrix, trainer, env)
    assert failed.returncode == 9
    assert any((run / "failure-retained.txt").is_file() for run in (log_root / "test_e0").iterdir())

    env.pop("FAIL_STAGE")
    resumed = _run(matrix, trainer, env, "--completed-stage", "foundation")
    assert resumed.returncode == 0, resumed.stderr
    consolidation = list((log_root / "test_e0").glob("*_smoke_consolidation_seed7"))
    assert len(consolidation) == 1
    assert (consolidation[0] / "model_9.pt").is_file()


def test_resume_rejects_incomplete_checkpoint(tmp_path):
    matrix, trainer, log_root, env = _fixture(tmp_path)
    root = log_root / "test_e0" / "old_smoke_foundation_seed7"
    root.mkdir(parents=True)
    (root / "training_contract.json").write_text("{}")
    (root / "agent.yaml").write_text("{}")
    torch.save({"iter": 0}, root / "model_0.pt")
    result = _run(matrix, trainer, env, "--completed-stage", "foundation")
    assert result.returncode != 0
    assert "No valid" in result.stderr
