# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedDistributionCfg:
    class_name: str = "base_locomotion_stackforce_quadrupedal.training.distribution:SquashedGaussianDistribution"
    init_std: float = 0.3
    std_type: str = "log"
    min_std: float = 0.05
    max_std: float = 0.5


@configclass
class SparseExpertModelCfg(RslRlMLPModelCfg):
    class_name: str = "base_locomotion_stackforce_quadrupedal.training.sparse_expert:SparseExpertModel"
    architecture: str = "mlp"
    num_experts: int = 6
    top_k: int = 2
    expert_hidden_dims: list[int] = [128, 128]
    gate_hidden_dims: list[int] = [64, 64]
    reflex_enabled: bool = False
    reflex_hidden_dims: list[int] = [64]
    reflex_scale: float = 0.25


@configclass
class SparseExpertPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "base_locomotion_stackforce_quadrupedal.training.sparse_expert:SparseExpertPPO"
    gate_entropy_coef: float = 0.0
    expert_orthogonality_coef: float = 0.0
    temporal_consistency_coef: float = 0.0


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    actor = SparseExpertModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = SparseExpertModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ComplexPPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 250
    experiment_name = "base_locomotion_complex"
    logger = "wandb"
    wandb_project = "stackforce-quadrupedal-locomotion"
    clip_actions = 1.0
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }
    actor = SparseExpertModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=BoundedDistributionCfg(),
    )
    critic = SparseExpertModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = SparseExpertPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
