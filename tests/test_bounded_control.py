"""Probability, support, and curriculum regressions independent of Isaac Sim."""

import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from base_locomotion_stackforce_quadrupedal.training.distribution import SquashedGaussianDistribution
from base_locomotion_stackforce_quadrupedal.benchmark import plateau_track_parameters, washboard_track_parameters
from base_locomotion_stackforce_quadrupedal.benchmark.surface import track_surface_height
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.curriculum import curriculum_decisions
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.reward.locomotion import (
    directed_planar_progress,
    planar_velocity_error_l2,
    wrong_way_velocity_ratio_l2,
)
from base_locomotion_stackforce_quadrupedal.training.checkpoint import (
    build_training_contract,
    checkpoint_runner_config,
    require_compatible_training_contract,
    save_training_contract,
)


def test_squashed_density_matches_torch_and_backpropagates():
    dist = SquashedGaussianDistribution(12)
    output = torch.randn(64, 12, requires_grad=True)
    dist.update(output)
    action = dist.sample()
    assert (action.abs() <= 1).all()
    mean, std = dist.params
    expected = TransformedDistribution(Normal(mean, std), [TanhTransform()])
    assert torch.allclose(dist.log_prob(action), expected.log_prob(action).sum(-1), atol=1e-5)
    loss = -(dist.log_prob(action).mean() + 0.001 * dist.entropy.mean())
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert torch.isfinite(dist.log_std_param.grad).all()
    assert torch.allclose(dist.kl_divergence(dist.params, dist.params), torch.zeros(64))
    assert torch.equal(dist.deterministic_output(output), dist.as_deterministic_output_module()(output))


def test_exploration_is_bounded_even_after_parameter_drift():
    dist = SquashedGaussianDistribution(12)
    with torch.no_grad():
        dist.log_std_param.fill_(10)
    dist.update(torch.full((32, 12), 1000.0))
    assert dist.std.max() <= 0.5
    assert torch.isfinite(dist.log_prob(dist.sample())).all()
    with pytest.raises(ValueError):
        SquashedGaussianDistribution(12, init_std=2.66)


def test_benchmark_support_uses_slope_and_bottle_crowns():
    p = plateau_track_parameters()
    xy = torch.tensor([[0, p['approach_m'] + p['ramp_length_m'] / 2], [10, 2]])
    assert track_surface_height(p, xy).tolist() == pytest.approx([p['ramp_height_m'] / 2, 0])
    p = washboard_track_parameters()
    start = p['approach_m'] + p['ramp_length_m']
    xy = torch.tensor([[-0.5, start + 0.15], [0.5, start + 0.075], [-0.5, start + 0.03]])
    assert track_surface_height(p, xy).tolist() == pytest.approx([0.155] * 3)


def test_curriculum_does_not_cancel_reverse_travel_or_promote_standing():
    history = torch.tensor([
        [6., 5.5, 1., 1., 1500.],
        [0., 0., 0., 0., 1500.],
        [6., 1., 100., 1., 1500.],
        [6., 6., 1., 1., 1500.],
    ])
    up, down = curriculum_decisions(history, torch.ones(4, dtype=torch.bool), torch.tensor([False, False, False, True]))
    assert up.tolist() == [True, False, False, False]
    assert down.tolist() == [False, False, True, True]


def test_curriculum_tightens_promotion_thresholds_by_level():
    history = torch.tensor([
        [6.0, 4.2, 120.0, 120.0, 1500.0],
        [6.0, 4.2, 120.0, 120.0, 1500.0],
    ])
    up, down = curriculum_decisions(
        history,
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        terrain_levels=torch.tensor([0, 7]),
    )
    assert up.tolist() == [True, False]
    assert down.tolist() == [False, False]


class _RewardData:
    def __init__(self, velocity):
        self.root_lin_vel_b = torch.tensor(velocity, dtype=torch.float32)


class _RewardAsset:
    def __init__(self, velocity):
        self.data = _RewardData(velocity)


class _RewardCommandManager:
    def __init__(self, command):
        self.command = torch.tensor(command, dtype=torch.float32)

    def get_command(self, _):
        return self.command


class _RewardEnv:
    def __init__(self, command, velocity):
        self.command_manager = _RewardCommandManager(command)
        self.scene = {"robot": _RewardAsset(velocity)}


def test_directional_rewards_distinguish_progress_from_reverse_motion():
    command = [[0.3, 0.0, 0.0, 0.105], [0.3, 0.0, 0.0, 0.105]]
    env = _RewardEnv(command, [[0.0, 0.3, 0.0], [0.0, -0.15, 0.0]])
    assert directed_planar_progress(env).tolist() == pytest.approx([1.0, 0.0])
    assert wrong_way_velocity_ratio_l2(env).tolist() == pytest.approx([0.0, 0.25])
    assert planar_velocity_error_l2(env).tolist() == pytest.approx([0.0, 0.2025])


def test_stop_command_has_nonzero_planar_penalty_for_forward_drift():
    env = _RewardEnv(
        [[0.0, 0.0, 0.0, 0.105]],
        [[0.0, 0.15, 0.0]],
    )
    assert planar_velocity_error_l2(env).item() == pytest.approx(0.0225)


def test_checkpoint_architecture_is_required_and_not_reinterpreted(tmp_path):
    from omegaconf import OmegaConf

    checkpoint = tmp_path / "model_1.pt"
    with pytest.raises(FileNotFoundError):
        checkpoint_runner_config(checkpoint, {})
    OmegaConf.save(OmegaConf.create({"actor": {"distribution_cfg": {"class_name": "GaussianDistribution"}}}), tmp_path / "agent.yaml")
    result = checkpoint_runner_config(checkpoint, {"actor": {"distribution_cfg": {"class_name": "new"}}})
    assert result["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution"


def test_checkpoint_resume_requires_matching_training_contract(tmp_path):
    checkpoint = tmp_path / "model_1.pt"
    env = {
        "sim": {"dt": 0.005},
        "decimation": 4,
        "episode_length_s": 30.0,
        "scene": {"num_envs": 64, "terrain": {"kind": "rough"}},
        "observations": {"policy": {"dim": 51}},
        "actions": {"dim": 12},
        "events": {},
        "commands": {"forward": [-0.45, 0.45]},
        "rewards": {"tracking": {"weight": 2.5}},
        "terminations": {"tilt": 1.0},
        "curriculum": {"terrain": True},
    }
    agent = {
        "num_steps_per_env": 24,
        "clip_actions": 1.0,
        "actor": {"hidden_dims": [128, 128]},
        "critic": {"hidden_dims": [128, 128]},
        "algorithm": {"gamma": 0.99},
    }
    contract = build_training_contract("task-v0", env, agent)
    save_training_contract(tmp_path, contract)
    require_compatible_training_contract(checkpoint, contract)

    changed_env = dict(env)
    changed_env["rewards"] = {"tracking": {"weight": 3.0}}
    changed = build_training_contract("task-v0", changed_env, agent)
    with pytest.raises(ValueError, match="training contract differs"):
        require_compatible_training_contract(checkpoint, changed)

    staged_env = dict(env)
    staged_env["terrain_profile"] = "union_foundation"
    staged_env["scene"] = {
        "num_envs": 64,
        "terrain": {"terrain_generator": {"sub_terrains": {"flat": {"proportion": 0.6}}}},
    }
    staged_contract = build_training_contract("task-v0", staged_env, agent)
    save_training_contract(tmp_path, staged_contract)
    next_stage_env = dict(staged_env)
    next_stage_env["terrain_profile"] = "union_expansion"
    next_stage_env["scene"] = {
        "num_envs": 64,
        "terrain": {"terrain_generator": {"sub_terrains": {"flat": {"proportion": 0.2}}}},
    }
    next_stage_contract = build_training_contract("task-v0", next_stage_env, agent)
    require_compatible_training_contract(
        checkpoint, next_stage_contract, allow_terrain_mix_transition=True
    )
    with pytest.raises(ValueError, match="training contract differs"):
        require_compatible_training_contract(checkpoint, next_stage_contract)

    next_stage_env["commands"] = {"proportion": 0.5}
    with pytest.raises(ValueError, match="training contract differs"):
        require_compatible_training_contract(
            checkpoint, build_training_contract("task-v0", next_stage_env, agent),
            allow_terrain_mix_transition=True,
        )

    (tmp_path / "training_contract.json").unlink()
    with pytest.raises(FileNotFoundError, match="formal training must start"):
        require_compatible_training_contract(checkpoint, contract)
