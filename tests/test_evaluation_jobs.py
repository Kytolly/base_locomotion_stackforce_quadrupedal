"""Check the frozen evaluation matrix without launching simulation processes."""

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def test_generalization_matrix_is_complete_and_has_unique_outputs():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("evaluation_jobs", root / "scripts/evaluate_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = OmegaConf.load(root / "configs/evaluation/generalization.yaml")
    jobs = module.evaluation_jobs(config, Path("/models/model.pt"), Path("/reports"))
    assert len(jobs) == 30
    assert len({name for name, _ in jobs}) == 30
    assert sum(name.startswith("validation_") for name, _ in jobs) == 10
    for kind in ("plateau", "washboard"):
        for mode in ("nominal", "randomized"):
            assert sum(name.startswith(f"{kind}_{mode}_") for name, _ in jobs) == 5
