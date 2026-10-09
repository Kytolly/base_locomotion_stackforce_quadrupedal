from types import SimpleNamespace

import torch

from base_locomotion_stackforce_quadrupedal.training.checkpoint import resume_training_runner


def test_resume_restores_adam_and_starts_after_completed_update(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=0.000123)
    parameter.square().sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "model_999.pt"
    torch.save({"iter": 999, "optimizer": optimizer.state_dict()}, checkpoint)
    restored = torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))], lr=0.001)
    runner = SimpleNamespace(alg=SimpleNamespace(optimizer=restored, learning_rate=0.001))

    def load(path):
        state = torch.load(path, weights_only=False)
        runner.alg.optimizer.load_state_dict(state["optimizer"])
        runner.current_learning_iteration = state["iter"]

    runner.load = load
    resume_training_runner(runner, str(checkpoint))
    assert runner.current_learning_iteration == 1000
    assert runner.alg.learning_rate == 0.000123
    for key, value in optimizer.state_dict()["state"][0].items():
        assert torch.equal(restored.state_dict()["state"][0][key], value)
