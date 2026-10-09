"""Sparse-expert and reflex policy components for the E0 experiment batches."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer, unpad_trajectories


class SharedTopKGate(nn.Module):
    """Shared Top-K router used by the Actor and Critic expert banks."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 6,
        top_k: int = 2,
        hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.network = MLP(input_dim, num_experts, hidden_dims, activation)

    def forward(self, policy_observation: torch.Tensor) -> torch.Tensor:
        logits = self.network(policy_observation)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(logits)
        sparse_weights.scatter_(-1, top_indices, torch.softmax(top_values, dim=-1))
        return sparse_weights


class SparseExpertNetwork(nn.Module):
    """Expert mixture with an optional bounded residual reflex head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_gate: SharedTopKGate,
        register_shared_gate: bool,
        policy_input_dim: int,
        expert_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        reflex_enabled: bool,
        reflex_hidden_dims: tuple[int, ...] | list[int],
        reflex_scale: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= reflex_scale <= 1.0:
            raise ValueError("reflex_scale must be in [0, 1].")
        if register_shared_gate:
            self.shared_gate = shared_gate
        else:
            # The Actor owns the shared parameters. The Critic keeps a plain
            # reference so PPO sees each parameter exactly once.
            object.__setattr__(self, "shared_gate", shared_gate)
        self.policy_input_dim = policy_input_dim
        self.reflex_enabled = reflex_enabled
        self.reflex_scale = reflex_scale
        self.experts = nn.ModuleList(
            MLP(input_dim, output_dim, expert_hidden_dims, activation)
            for _ in range(shared_gate.num_experts)
        )
        self.reflex = (
            MLP(input_dim + output_dim, output_dim, reflex_hidden_dims, activation)
            if reflex_enabled
            else None
        )
        self.last_gate_weights: torch.Tensor | None = None
        self.last_expert_outputs: torch.Tensor | None = None
        self.last_pre_reflex: torch.Tensor | None = None
        self.last_reflex_delta: torch.Tensor | None = None
        self.last_final_output: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        weights = self.shared_gate(latent[..., : self.policy_input_dim])
        expert_outputs = torch.stack([expert(latent) for expert in self.experts], dim=-2)
        mixture = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=-2)
        pre_reflex = mixture
        reflex_delta = torch.zeros_like(mixture)
        if self.reflex is not None:
            reflex_delta = self.reflex_scale * torch.tanh(
                self.reflex(torch.cat((latent, mixture), dim=-1))
            )
            mixture = mixture + reflex_delta
        self.last_gate_weights = weights
        self.last_expert_outputs = expert_outputs
        self.last_pre_reflex = pre_reflex
        self.last_reflex_delta = reflex_delta
        self.last_final_output = mixture
        return mixture


class SparseExpertModel(MLPModel):
    """RSL-RL model supporting either the PPO baseline or sparse experts."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        architecture: str = "mlp",
        num_experts: int = 6,
        top_k: int = 2,
        expert_hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        gate_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        reflex_enabled: bool = False,
        reflex_hidden_dims: tuple[int, ...] | list[int] = (64,),
        reflex_scale: float = 0.25,
        shared_gate: SharedTopKGate | None = None,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        if architecture not in {"mlp", "sparse_moe"}:
            raise ValueError(f"Unknown model architecture: {architecture!r}.")
        self.architecture = architecture
        self._last_policy_mean: torch.Tensor | None = None
        self._last_temporal_mask: torch.Tensor | None = None
        if architecture == "sparse_moe":
            policy_input_dim = int(obs["policy"].shape[-1])
            shared_gate = shared_gate or SharedTopKGate(
                policy_input_dim, num_experts, top_k, gate_hidden_dims, activation
            )
            mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
            self.mlp = SparseExpertNetwork(
                self.obs_dim,
                mlp_output_dim,
                shared_gate,
                obs_set == "actor",
                policy_input_dim,
                expert_hidden_dims,
                activation,
                reflex_enabled,
                reflex_hidden_dims,
                reflex_scale,
            )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            self._last_policy_mean = self.distribution.deterministic_output(mlp_output)
        if "training_auxiliary" in obs.keys():
            self._last_temporal_mask = obs["training_auxiliary"][..., :1]
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self._last_policy_mean
        return mlp_output

    def auxiliary_losses(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        """Return the three BiSEC regularizers for the current Actor batch."""
        zero = next(self.parameters()).new_zeros(())
        if self.architecture != "sparse_moe":
            return {"gate_entropy": zero, "expert_orthogonality": zero, "temporal_consistency": zero}
        network: SparseExpertNetwork = self.mlp  # type: ignore[assignment]
        weights = network.last_gate_weights
        expert_outputs = network.last_expert_outputs
        if weights is None or expert_outputs is None or self._last_policy_mean is None:
            raise RuntimeError("Auxiliary losses require a preceding Actor forward pass.")
        gate_entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        normalized = torch.nn.functional.normalize(expert_outputs, dim=-1)
        gram = normalized @ normalized.transpose(-1, -2)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        expert_orthogonality = ((gram - eye) ** 2).mean()
        auxiliary_observation = obs.get("training_auxiliary")
        previous_action = (
            auxiliary_observation[..., 1 : 1 + self._last_policy_mean.shape[-1]]
            if auxiliary_observation is not None
            else obs["policy"][..., -self._last_policy_mean.shape[-1] :]
        )
        temporal_error = (self._last_policy_mean - previous_action).pow(2).mean(dim=-1, keepdim=True)
        mask = self._last_temporal_mask
        temporal_consistency = (
            (temporal_error * mask).sum() / mask.sum().clamp_min(1.0)
            if mask is not None
            else temporal_error.mean()
        )
        return {
            "gate_entropy": gate_entropy,
            "expert_orthogonality": expert_orthogonality,
            "temporal_consistency": temporal_consistency,
        }


def _unique_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in chain.from_iterable(module.parameters() for module in modules):
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result


class SparseExpertPPO(PPO):
    """PPO with shared routing and optional BiSEC auxiliary losses."""

    def __init__(
        self,
        *args,
        gate_entropy_coef: float = 0.0,
        expert_orthogonality_coef: float = 0.0,
        temporal_consistency_coef: float = 0.0,
        optimizer: str = "adam",
        **kwargs,
    ) -> None:
        super().__init__(*args, optimizer=optimizer, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("SparseExpertPPO does not combine BiSEC losses with RND or symmetry.")
        self.gate_entropy_coef = gate_entropy_coef
        self.expert_orthogonality_coef = expert_orthogonality_coef
        self.temporal_consistency_coef = temporal_consistency_coef
        self.optimizer = resolve_optimizer(optimizer)(
            _unique_parameters(self.actor, self.critic), lr=self.learning_rate
        )

    def update(self) -> dict[str, float]:
        totals = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "gate_entropy": 0.0,
            "expert_orthogonality": 0.0,
            "temporal_consistency": 0.0,
        }
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )
            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    ).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            surrogate = -batch.advantages.squeeze(-1) * ratio
            clipped = -batch.advantages.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (values - batch.returns).pow(2),
                    (value_clipped - batch.returns).pow(2),
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            auxiliary = self.actor.auxiliary_losses(batch.observations)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.gate_entropy_coef * auxiliary["gate_entropy"]
                + self.expert_orthogonality_coef * auxiliary["expert_orthogonality"]
                + self.temporal_consistency_coef * auxiliary["temporal_consistency"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(_unique_parameters(self.actor, self.critic), self.max_grad_norm)
            self.optimizer.step()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            for name, value in auxiliary.items():
                totals[name] += value.item()
            if self.actor.architecture == "sparse_moe":
                network: SparseExpertNetwork = self.actor.mlp
                usage = network.last_gate_weights.mean(dim=0)
                for expert_index, expert_usage in enumerate(usage):
                    totals.setdefault(f"expert_usage_{expert_index}", 0.0)
                    totals[f"expert_usage_{expert_index}"] += expert_usage.item()
                totals.setdefault("reflex_delta_rms", 0.0)
                totals["reflex_delta_rms"] += network.last_reflex_delta.square().mean().sqrt().item()

        count = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {name: value / count for name, value in totals.items()}

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "SparseExpertPPO":
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"].pop("share_cnn_encoders", None)

        shared_gate = None
        if cfg["actor"].get("architecture") == "sparse_moe":
            shared_gate = SharedTopKGate(
                int(obs["policy"].shape[-1]),
                int(cfg["actor"]["num_experts"]),
                int(cfg["actor"]["top_k"]),
                cfg["actor"]["gate_hidden_dims"],
                cfg["actor"]["activation"],
            ).to(device)
        actor = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, shared_gate=shared_gate, **cfg["actor"]
        ).to(device)
        critic = critic_class(
            obs, cfg["obs_groups"], "critic", 1, shared_gate=shared_gate, **cfg["critic"]
        ).to(device)
        print(f"Actor Model: {actor}")
        print(f"Critic Model: {critic}")
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )


__all__ = ["SharedTopKGate", "SparseExpertModel", "SparseExpertPPO"]
