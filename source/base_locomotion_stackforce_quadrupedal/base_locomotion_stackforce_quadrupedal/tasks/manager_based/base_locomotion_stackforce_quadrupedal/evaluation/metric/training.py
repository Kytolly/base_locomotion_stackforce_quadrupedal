"""Gym wrapper that exports episode metrics through RSL-RL logging extras."""

from __future__ import annotations

import gymnasium as gym
import torch

from .locomotion import (
    LocomotionEpisodeMetrics,
    aggregate_metric_batch,
    motion_family_metrics,
    route_role_metrics,
    terrain_family_metrics,
)
from base_locomotion_stackforce_quadrupedal.evaluation.metric.logging import (
    filter_metric_log,
    resolve_metric_logging_config,
)


class TrainingMetricsWrapper(gym.Wrapper):
    """Attach measured training metrics to ``extras['log']`` when episodes finish."""

    def __init__(self, env, logging_config=None) -> None:
        super().__init__(env)
        self.logging_config = resolve_metric_logging_config(logging_config)
        self.metrics = None
        if self.logging_config["metrics_enabled"]:
            self.metrics = LocomotionEpisodeMetrics(
                self.unwrapped,
                metric_groups=self.logging_config["metric_groups"],
            )
            callbacks = list(getattr(self.unwrapped, "post_physics_callbacks", ()))
            callbacks.append(self._observe_step)
            self.unwrapped.post_physics_callbacks = callbacks

    def _observe_step(self):
        if self.metrics is None:
            return
        self.metrics.observe_state(self.unwrapped.action_manager.action)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if self.metrics is not None:
            self.metrics.reset()
        return result

    def step(self, action: torch.Tensor):
        observations, reward, terminated, truncated, extras = self.env.step(action)
        if self.metrics is not None:
            self.metrics.observe_reward(reward)
        done = terminated | truncated
        if self.metrics is not None and bool(done.any()):
            env_ids = done.nonzero(as_tuple=False).flatten()
            base_height_failure = self._termination_flags("base_height")
            episode_metrics = self.metrics.episode_metrics(
                env_ids, terminated, truncated, base_height_failure
            )
            metric_log = aggregate_metric_batch(episode_metrics, "train")
            if self.metrics.metric_groups["terrain"]:
                metric_log.update(
                    terrain_family_metrics(
                        episode_metrics,
                        "train",
                        tuple(self.unwrapped.cfg.terrain_families),
                    )
                )
                metric_log.update(route_role_metrics(episode_metrics, "train"))
            if self.metrics.metric_groups["motion"]:
                metric_log.update(motion_family_metrics(episode_metrics, "train"))
            metric_log["train/terrain/seed"] = float(self.unwrapped.cfg.seed or 0)
            metric_log = filter_metric_log(
                metric_log, self.logging_config["panel_groups"]
            )
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
        elif self.metrics is not None:
            # Preserve RSL-RL's stable extras schema when no episode completed.
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
