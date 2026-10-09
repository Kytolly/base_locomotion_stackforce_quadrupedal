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
