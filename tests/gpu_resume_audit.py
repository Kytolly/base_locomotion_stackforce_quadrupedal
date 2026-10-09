"""Instrument the real YAML entrypoint to verify loaded checkpoint state.

Run manually in env_isaaclab with the same arguments as scripts/rsl_rl/train.py.
Only diagnostic configurations with at most two iterations are accepted.
"""

import json
from pathlib import Path
import runpy
import sys

import torch
from rsl_rl.runners import OnPolicyRunner


ROOT = Path(__file__).resolve().parents[1]
original_load = OnPolicyRunner.load
original_learn = OnPolicyRunner.learn


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equal(a, b)
    else:
        assert actual == expected


def audited_load(self, path, *args, **kwargs):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    result = original_load(self, path, *args, **kwargs)
    assert_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"])
    assert_equal(self.alg.actor.state_dict(), saved["actor_state_dict"])
    assert_equal(self.alg.critic.state_dict(), saved["critic_state_dict"])
    assert self.current_learning_iteration == saved["iter"]
    self._audit_saved_iteration = saved["iter"]
    print("[AUDIT] actor, critic, Adam moments/steps/groups exactly restored", flush=True)
    return result


def audited_learn(self, num_learning_iterations, **kwargs):
    assert 0 < num_learning_iterations <= 2
    start = self.current_learning_iteration
    if hasattr(self, "_audit_saved_iteration"):
        assert start == self._audit_saved_iteration + 1
        assert self.alg.learning_rate == self.alg.optimizer.param_groups[0]["lr"]
    original_learn(self, num_learning_iterations, **kwargs)
    final = start + num_learning_iterations - 1
    checkpoint = Path(self.logger.log_dir) / f"model_{final}.pt"
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert saved["iter"] == self.current_learning_iteration == final
    assert all(torch.isfinite(p).all() for p in self.alg.actor.parameters())
    audit = {"first_iteration": start, "last_iteration": final,
             "updates": num_learning_iterations, "checkpoint": str(checkpoint),
             "optimizer_exact_restore": hasattr(self, "_audit_saved_iteration")}
    (checkpoint.parent / "resume_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[AUDIT] {audit}", flush=True)


if __name__ == "__main__":
    OnPolicyRunner.load = audited_load
    OnPolicyRunner.learn = audited_learn
    sys.argv[0] = str(ROOT / "scripts/rsl_rl/train.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
