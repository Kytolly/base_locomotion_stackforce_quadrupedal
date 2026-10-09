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
        if std_type != "log" or not 0 < min_std <= init_std <= max_std:
            raise ValueError("Use log std with 0 < min_std <= init_std <= max_std.")
        super().__init__(output_dim, init_std, std_type)
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
