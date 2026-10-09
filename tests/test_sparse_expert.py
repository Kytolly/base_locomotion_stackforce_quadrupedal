from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/base_locomotion_stackforce_quadrupedal"))

from base_locomotion_stackforce_quadrupedal.training.distribution import (  # noqa: E402
    SquashedGaussianDistribution,
)
from base_locomotion_stackforce_quadrupedal.training.sparse_expert import (  # noqa: E402
    SharedTopKGate,
    SparseExpertModel,
)


def _observations(batch_size: int = 8) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 46),
            "privileged": torch.randn(batch_size, 16),
            "training_auxiliary": torch.cat(
                (torch.ones(batch_size, 1), torch.randn(batch_size, 12)), dim=-1
            ),
        },
        batch_size=[batch_size],
    )


def _distribution_cfg() -> dict:
    return {
        "class_name": SquashedGaussianDistribution,
        "init_std": 0.3,
        "std_type": "log",
        "min_std": 0.05,
        "max_std": 0.5,
    }


def test_shared_top2_gate_is_sparse_and_normalized():
    gate = SharedTopKGate(46, num_experts=6, top_k=2)
    weights = gate(torch.randn(16, 46))
    assert weights.shape == (16, 6)
    assert torch.all((weights > 0).sum(dim=-1) == 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16))


def test_actor_and_critic_share_gate_without_privileged_gate_input():
    obs = _observations()
    groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    gate = SharedTopKGate(46, num_experts=6, top_k=2)
    actor = SparseExpertModel(
        obs,
        groups,
        "actor",
        12,
        architecture="sparse_moe",
        shared_gate=gate,
        distribution_cfg=_distribution_cfg(),
    )
    critic = SparseExpertModel(
        obs,
        groups,
        "critic",
        1,
        architecture="sparse_moe",
        shared_gate=gate,
    )
    assert actor.mlp.shared_gate is critic.mlp.shared_gate
    action = actor(obs, stochastic_output=True)
    value = critic(obs)
    assert action.shape == (8, 12)
    assert value.shape == (8, 1)
    assert torch.isfinite(action).all() and torch.isfinite(value).all()
    assert (action.abs() < 1.0).all()


def test_auxiliary_losses_are_finite_and_transition_mask_disables_temporal_term():
    obs = _observations()
    obs["training_auxiliary"][:, 0].zero_()
    model = SparseExpertModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        12,
        architecture="sparse_moe",
        reflex_enabled=True,
        distribution_cfg=_distribution_cfg(),
    )
    model(obs, stochastic_output=True)
    losses = model.auxiliary_losses(obs)
    assert set(losses) == {"gate_entropy", "expert_orthogonality", "temporal_consistency"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["gate_entropy"] > 0
    assert losses["temporal_consistency"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("batch_size", [1, 17])
def test_dispatch_matches_dense_outputs_gradients_and_work(batch_size):
    from copy import deepcopy

    obs = _observations(batch_size)
    model = SparseExpertModel(obs, {"actor": ["policy"]}, "actor", 12,
                              architecture="sparse_moe", distribution_cfg=_distribution_cfg())
    dense = deepcopy(model)
    dense.mlp.expert_dispatch = "dense"
    rows = []
    handles = [expert.register_forward_pre_hook(lambda _, inputs: rows.append(inputs[0].shape[0]))
               for expert in model.mlp.experts]
    output = model(obs)
    reference = dense(obs)
    assert sum(rows) == batch_size * 2
    assert torch.allclose(output, reference, atol=1e-6)
    loss = output.square().mean() + model.auxiliary_losses(obs)["expert_orthogonality"]
    ref_loss = reference.square().mean() + dense.auxiliary_losses(obs)["expert_orthogonality"]
    loss.backward()
    ref_loss.backward()
    for param, ref_param in zip(model.parameters(), dense.parameters()):
        if param.grad is None:
            assert ref_param.grad is None or not ref_param.grad.any()
        else:
            assert torch.allclose(param.grad, ref_param.grad, atol=1e-6)
    for handle in handles:
        handle.remove()


def test_active_orthogonality_ignores_unselected_experts():
    obs = _observations(4)
    model = SparseExpertModel(obs, {"actor": ["policy"]}, "actor", 12,
                              architecture="sparse_moe", distribution_cfg=_distribution_cfg())
    with torch.no_grad():
        for param in model.mlp.shared_gate.parameters():
            param.zero_()
    model(obs)
    weights = model.mlp.last_gate_weights
    inactive = (weights.sum(0) == 0).nonzero().flatten().tolist()
    assert len(inactive) == 4
    loss = model.auxiliary_losses(obs)["expert_orthogonality"]
    assert model.mlp.last_expert_outputs.shape == (4, 2, 12)
    loss.backward()
    for index in inactive:
        assert all(p.grad is None for p in model.mlp.experts[index].parameters())
        with torch.no_grad():
            for param in model.mlp.experts[index].parameters():
                param.add_(100)
    model(obs)
    assert torch.allclose(loss, model.auxiliary_losses(obs)["expert_orthogonality"])
    model.mlp.orthogonality_scope = "all"
    model(obs)
    assert model.mlp.last_expert_outputs.shape == (4, 6, 12)


def test_router_keeps_logit_selection_when_a_selected_weight_underflows():
    gate = SharedTopKGate(6)
    gate.network = torch.nn.Identity()
    weights, indices = gate.route(torch.tensor([[1000., 100., 0., -1., -2., -3.]]))
    assert indices.tolist() == [[0, 1]]
    assert weights[0, 1] == 0


def test_temporal_loss_uses_adjacent_raw_means_and_masks_with_two_sided_gradients():
    obs = _observations(4)
    obs["temporal_pair_valid"] = torch.ones(4, 1)
    model = SparseExpertModel(obs, {"actor": ["policy"]}, "actor", 12,
                              architecture="sparse_moe", distribution_cfg=_distribution_cfg())
    model(obs)
    current = model.mlp.last_final_output
    previous = (current.detach() + 0.2).requires_grad_()
    current.retain_grad()
    loss = model.auxiliary_losses(obs, previous)["temporal_consistency"]
    assert loss.item() == pytest.approx(12 * 0.2**2)
    loss.backward()
    assert previous.grad.abs().sum() > 0
    assert current.grad.abs().sum() > 0
    obs["training_auxiliary"][:, 1:].fill_(1000)
    assert model.auxiliary_losses(obs, previous)["temporal_consistency"].item() == pytest.approx(loss.item())
    obs["temporal_pair_valid"].zero_()
    assert model.auxiliary_losses(obs, previous)["temporal_consistency"].item() == 0
    obs["temporal_pair_valid"].fill_(1)
    obs["training_auxiliary"][:, 0].zero_()
    assert model.auxiliary_losses(obs, previous)["temporal_consistency"].item() == 0


def test_temporal_pairs_keep_environment_identity_and_reset_validity(monkeypatch):
    from rsl_rl.algorithms import PPO
    from base_locomotion_stackforce_quadrupedal.training.sparse_expert import SparseExpertPPO

    monkeypatch.setattr(PPO, "act", lambda self, obs: obs)
    monkeypatch.setattr(PPO, "process_env_step", lambda *args: None)
    algorithm = object.__new__(SparseExpertPPO)
    algorithm.temporal_consistency_coef = 0.01
    algorithm._previous_policy_observation = None
    algorithm._temporal_pair_valid = None
    obs = _observations(4)
    first = algorithm.act(obs)
    assert not first["temporal_pair_valid"].any()
    algorithm.process_env_step(obs, None, torch.tensor([False, True, False, True]), {})
    next_obs = _observations(4)
    second = algorithm.act(next_obs)
    assert torch.equal(second["previous_policy_observation"], obs["policy"])
    assert second["temporal_pair_valid"].flatten().tolist() == [1, 0, 1, 0]
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.equal(second[permutation]["previous_policy_observation"], obs["policy"][permutation])
