from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fix_plan_runs as runs


def test_g6_exports_every_gate_metric(tmp_path, monkeypatch):
    captured = {}
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "output/fix_plan/G6").mkdir(parents=True)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.setattr(runs, "G6_GROUPS", {"P0": []})
    monkeypatch.setattr(runs, "G6_COMMON", [])
    monkeypatch.setattr(runs, "_train", lambda *args, **kwargs: run)

    def summarise(_run, tags, window):
        captured.update(tags=tags, window=window)
        return {"run": str(_run), "window": window, "metrics": {}, "missing": []}

    monkeypatch.setattr(runs, "summarise", summarise)
    runs.g6(SimpleNamespace(only=None, iterations=1500, extra=[], existing_run=[]))

    assert set(runs.DEFAULT_TAGS).issubset(captured["tags"])
    assert captured["window"] == 300


def test_g6_can_summarize_an_existing_interrupted_run(tmp_path, monkeypatch):
    run = tmp_path / "interrupted"
    run.mkdir()
    (tmp_path / "output/fix_plan/G6").mkdir(parents=True)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.setattr(runs, "G6_GROUPS", {"P0": ["legacy=true"]})
    monkeypatch.setattr(runs, "G6_COMMON", ["terrain=low"])
    monkeypatch.setattr(runs, "_train", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("relaunched")))
    monkeypatch.setattr(
        runs,
        "summarise",
        lambda path, tags, window: {"run": str(path), "window": window, "metrics": {}, "missing": []},
    )

    runs.g6(
        SimpleNamespace(
            only=["P0"],
            iterations=1006,
            extra=[],
            existing_run=[f"P0={run}"],
        )
    )

    assert (tmp_path / "output/fix_plan/G6/p0_summary.json").is_file()
