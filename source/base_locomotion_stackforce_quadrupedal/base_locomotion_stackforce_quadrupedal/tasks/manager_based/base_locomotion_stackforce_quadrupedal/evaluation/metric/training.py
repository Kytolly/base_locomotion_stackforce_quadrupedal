"""Gym wrapper that exports episode metrics through RSL-RL logging extras."""

from __future__ import annotations

import gymnasium as gym
import torch

from .locomotion import (
    LocomotionEpisodeMetrics,
    aggregate_metric_batch,
    terrain_family_metrics,
)


class TrainingMetricsWrapper(gym.Wrapper):
    """Attach measured training metrics to ``extras['log']`` when episodes finish."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.metrics = LocomotionEpisodeMetrics(self.unwrapped)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.metrics.reset()
        return result

    def step(self, action: torch.Tensor):
        self.metrics.observe_state(action)
        observations, reward, terminated, truncated, extras = self.env.step(action)
        self.metrics.observe_reward(reward)
        done = terminated | truncated
        if bool(done.any()):
            env_ids = done.nonzero(as_tuple=False).flatten()
            base_height_failure = self._termination_flags("base_height")
            episode_metrics = self.metrics.episode_metrics(
                env_ids, terminated, truncated, base_height_failure
            )
            metric_log = aggregate_metric_batch(episode_metrics, "train")
            metric_log.update(terrain_family_metrics(episode_metrics, "train"))
            metric_log["train/terrain/seed"] = float(self.unwrapped.cfg.seed or 0)
            extras = dict(extras)
            # RSL-RL prioritizes ``episode`` over ``log`` when both are present.
            # Merge into both channels so the metrics survive either wrapper path.
            episode = dict(extras.get("episode", {}))
            episode.update(metric_log)
            log = dict(extras.get("log", {}))
            log.update(metric_log)
            extras["episode"] = episode
            extras["log"] = log
            self.metrics.reset(env_ids)
        else:
            # ManagerBasedRLEnv may expose an empty episode/log mapping every step.
            # RSL-RL uses the keys from the first mapping in a rollout, so only
            # forward episode data when at least one environment actually ended.
            extras = dict(extras)
            extras.pop("episode", None)
            extras.pop("log", None)
        return observations, reward, terminated, truncated, extras

    def _termination_flags(self, term_name: str) -> torch.Tensor:
        manager = self.unwrapped.termination_manager
        flags = torch.zeros(
            self.unwrapped.num_envs, dtype=torch.bool, device=self.unwrapped.device
        )
        if term_name in manager._term_names:
            index = manager._term_names.index(term_name)
            flags = manager._last_episode_dones[:, index]
        return flags
