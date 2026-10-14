"""Bounded policy outputs with PPO-consistent transformed densities."""

import math

import torch
from torch import nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from rsl_rl.modules.distribution import GaussianDistribution


class SquashedGaussianDistribution(GaussianDistribution):
    """Apply tanh to a Normal with bounded latent mean and exploration scale.

    KL is invariant under the shared invertible transform. ``std`` reports the
    latent Normal scale, not the standard deviation of the bounded action.
    """

    def __init__(self, output_dim, init_std=0.3, std_type="log", min_std=0.05, max_std=0.5):
        per_dim = [float(v) for v in init_std] if isinstance(init_std, (list, tuple)) else None
        values = per_dim if per_dim is not None else [float(init_std)]
        if std_type != "log" or not all(0 < min_std <= v <= max_std for v in values):
            raise ValueError("Use log std with 0 < min_std <= init_std <= max_std.")
        if per_dim is not None and len(per_dim) != output_dim:
            raise ValueError(f"Per-dimension init_std has {len(per_dim)} entries for {output_dim} outputs.")
        super().__init__(output_dim, values[0], std_type)
        if per_dim is not None:
            # E0-v2: legs and wheels start with different latent noise so the
            # initial physical joint noise matches the old contract (plan G4).
            with torch.no_grad():
                self.log_std_param.copy_(torch.log(torch.tensor(per_dim, dtype=self.log_std_param.dtype)))
        self.min_std, self.max_std = min_std, max_std

    def update(self, mlp_output):
        mean = self._latent_mean(mlp_output)
        std = self.log_std_param.clamp(math.log(self.min_std), math.log(self.max_std)).exp()
        self._distribution = Normal(mean, std.expand_as(mean))
        self._bounded = TransformedDistribution(self._distribution, [TanhTransform(cache_size=1)])

    @staticmethod
    def _latent_mean(output):
        # Avoid floating-point tanh saturation while retaining nearly the full action range.
        return 3.0 * torch.tanh(output / 3.0)

    def sample(self):
        return self._bounded.sample()

    def log_prob(self, outputs):
        eps = torch.finfo(outputs.dtype).eps
        return self._bounded.log_prob(outputs.clamp(-1 + eps, 1 - eps)).sum(-1)

    @property
    def entropy(self):
        # Reparameterized Monte Carlo entropy includes the transform Jacobian.
        sample = self._bounded.rsample()
        return -self.log_prob(sample)

    def deterministic_output(self, mlp_output):
        return torch.tanh(self._latent_mean(mlp_output))

    def as_deterministic_output_module(self):
        return _SquashedOutput()


class _SquashedOutput(nn.Module):
    def forward(self, value):
        return torch.tanh(3.0 * torch.tanh(value / 3.0))
