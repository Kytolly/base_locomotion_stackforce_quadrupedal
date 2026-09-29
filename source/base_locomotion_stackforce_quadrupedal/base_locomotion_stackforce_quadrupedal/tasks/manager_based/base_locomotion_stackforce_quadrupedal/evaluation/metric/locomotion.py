"""Vectorized episode metrics shared by training and validation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...env.robots import LEG_JOINTS, WHEEL_JOINTS
from ...env.terrains import TERRAIN_FAMILIES
from ...mdp.action import LEG_ACTION_SLICE, WHEEL_ACTION_SLICE
from ...mdp.policy.commands import (
    BODY_HEIGHT,
    FORWARD,
    LATERAL,
    LOCOMOTION_COMMAND_NAME,
    YAW_RATE,
)
from ...mdp.support import local_support_height, wheel_contact_state


def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class LocomotionEpisodeMetrics:
    """Accumulate physical and optimization metrics independently for every environment."""

    def __init__(self, env, action_saturation_threshold: float = 0.95) -> None:
        if not 0.0 <= action_saturation_threshold < 1.0:
            raise ValueError("action_saturation_threshold must be in [0, 1).")
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.dt = float(env.step_dt)
        self.action_saturation_threshold = action_saturation_threshold
        robot = env.scene["robot"]
        self.leg_joint_ids, _ = robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        self.wheel_joint_ids, _ = robot.find_joints(
            list(WHEEL_JOINTS), preserve_order=True
        )
        self.active_joint_ids = list(self.leg_joint_ids) + list(self.wheel_joint_ids)
        self.reward_names = list(env.reward_manager._term_names)
        self.reward_weights = {
            name: float(env.reward_manager._term_cfgs[index].weight)
            for index, name in enumerate(self.reward_names)
        }
        self._buffers: dict[str, torch.Tensor] = {}
        self._reward_weighted = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self.reward_names
        }
        self._last_action = torch.zeros(
            (self.num_envs, env.action_manager.total_action_dim), device=self.device
        )
        self._episode_terrain_type = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_terrain_level = torch.zeros_like(self._episode_terrain_type)
        self._create_buffers()
        self.reset()

    def _create_buffers(self) -> None:
        names = (
            "steps",
            "reward_total",
            "forward_error_sq",
            "lateral_error_sq",
            "yaw_error_sq",
            "height_error_sq",
            "forward_distance",
            "tilt_sum",
            "tilt_sq",
            "tilt_max",
            "leg_action_sq",
            "wheel_action_sq",
            "action_rate_sq",
            "action_saturation_count",
            "joint_position_limit_count",
            "joint_velocity_limit_count",
            "leg_mechanical_energy",
            "wheel_mechanical_energy",
            "finite_steps",
            "raw_forward_sum",
            "raw_lateral_sum",
            "raw_yaw_sum",
            "raw_height_sum",
            "shaped_forward_sum",
            "shaped_lateral_sum",
            "shaped_yaw_sum",
            "shaped_height_sum",
            "actual_forward_sum",
            "actual_lateral_sum",
            "actual_yaw_sum",
            "actual_height_sum",
            "support_invalid_count",
            "supported_wheel_count",
            "base_collision_count",
        )
        self._buffers = {
            name: torch.zeros(self.num_envs, device=self.device) for name in names
        }

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        for value in self._buffers.values():
            value[env_ids] = 0.0
        for value in self._reward_weighted.values():
            value[env_ids] = 0.0
        self._last_action[env_ids] = 0.0
        terrain = self.env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_types is not None:
            self._episode_terrain_type[env_ids] = terrain_types[env_ids]
        if terrain_levels is not None:
            self._episode_terrain_level[env_ids] = terrain_levels[env_ids]

    def observe_state(self, action: torch.Tensor) -> None:
        """Accumulate the pre-transition state and clipped raw policy action."""
        robot = self.env.scene["robot"]
        command_term = self.env.command_manager.get_term(LOCOMOTION_COMMAND_NAME)
        command = command_term.command
        raw_command = command_term.raw_command
        root_lin_vel_b = robot.data.root_lin_vel_b
        root_ang_vel_b = robot.data.root_ang_vel_b
        projected_gravity_b = robot.data.projected_gravity_b
        joint_pos = robot.data.joint_pos[:, self.active_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.active_joint_ids]
        joint_torque = robot.data.applied_torque[:, self.active_joint_ids]
        state_tensors = (
            action,
            command,
            raw_command,
            root_lin_vel_b,
            root_ang_vel_b,
            projected_gravity_b,
            joint_pos,
            joint_vel,
            joint_torque,
        )
        finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for value in state_tensors:
            finite &= torch.isfinite(value).flatten(start_dim=1).all(dim=1)

        action = _finite_or_zero(action)
        command = _finite_or_zero(command)
        raw_command = _finite_or_zero(raw_command)
        root_lin_vel_b = _finite_or_zero(root_lin_vel_b)
        root_ang_vel_b = _finite_or_zero(root_ang_vel_b)
        projected_gravity_b = _finite_or_zero(projected_gravity_b)
        joint_pos = _finite_or_zero(joint_pos)
        joint_vel = _finite_or_zero(joint_vel)
        joint_torque = _finite_or_zero(joint_torque)

        support_height, support_valid = local_support_height(self.env)
        height = _finite_or_zero(robot.data.root_pos_w[:, 2] - support_height)
        contact = wheel_contact_state(self.env)
        base_force = self.env.scene["base_contact"].data.net_forces_w_history
        base_collision = torch.linalg.vector_norm(base_force, dim=-1).amax(dim=-1).amax(dim=-1) > 8.0
        forward_error = command[:, FORWARD] - root_lin_vel_b[:, 1]
        lateral_error = command[:, LATERAL] - root_lin_vel_b[:, 0]
        yaw_error = command[:, YAW_RATE] - root_ang_vel_b[:, 2]
        height_error = command[:, BODY_HEIGHT] - height
        tilt = torch.acos(torch.clamp(-projected_gravity_b[:, 2], -1.0, 1.0))

        self._buffers["steps"] += 1.0
        self._buffers["forward_error_sq"] += torch.square(forward_error)
        self._buffers["lateral_error_sq"] += torch.square(lateral_error)
        self._buffers["yaw_error_sq"] += torch.square(yaw_error)
        self._buffers["height_error_sq"] += torch.square(height_error)
        self._buffers["forward_distance"] += root_lin_vel_b[:, 1] * self.dt
        self._buffers["tilt_sum"] += tilt
        self._buffers["tilt_sq"] += torch.square(tilt)
        self._buffers["tilt_max"] = torch.maximum(self._buffers["tilt_max"], tilt)
        self._buffers["leg_action_sq"] += torch.mean(
            torch.square(action[:, LEG_ACTION_SLICE]), dim=1
        )
        self._buffers["wheel_action_sq"] += torch.mean(
            torch.square(action[:, WHEEL_ACTION_SLICE]), dim=1
        )
        self._buffers["action_rate_sq"] += torch.mean(
            torch.square(action - self._last_action), dim=1
        )
        self._buffers["action_saturation_count"] += (
            (torch.abs(action) >= self.action_saturation_threshold).float().mean(dim=1)
        )

        soft_position_limits = robot.data.soft_joint_pos_limits[
            :, self.active_joint_ids
        ]
        soft_velocity_limits = robot.data.soft_joint_vel_limits[
            :, self.active_joint_ids
        ]
        position_limited = (joint_pos < soft_position_limits[:, :, 0]) | (
            joint_pos > soft_position_limits[:, :, 1]
        )
        velocity_limited = torch.abs(joint_vel) > 0.9 * soft_velocity_limits
        self._buffers["joint_position_limit_count"] += position_limited.any(
            dim=1
        ).float()
        self._buffers["joint_velocity_limit_count"] += velocity_limited.any(
            dim=1
        ).float()

        leg_count = len(self.leg_joint_ids)
        power = torch.abs(joint_torque * joint_vel)
        self._buffers["leg_mechanical_energy"] += (
            torch.sum(power[:, :leg_count], dim=1) * self.dt
        )
        self._buffers["wheel_mechanical_energy"] += (
            torch.sum(power[:, leg_count:], dim=1) * self.dt
        )
        self._buffers["finite_steps"] += finite.float()
        self._buffers["support_invalid_count"] += (~support_valid).float()
        self._buffers["supported_wheel_count"] += contact.sum(dim=1).float()
        self._buffers["base_collision_count"] += base_collision.float()

        for index, name in enumerate(("forward", "lateral", "yaw", "height")):
            self._buffers[f"raw_{name}_sum"] += raw_command[:, index]
            self._buffers[f"shaped_{name}_sum"] += command[:, index]
        self._buffers["actual_forward_sum"] += root_lin_vel_b[:, 1]
        self._buffers["actual_lateral_sum"] += root_lin_vel_b[:, 0]
        self._buffers["actual_yaw_sum"] += root_ang_vel_b[:, 2]
        self._buffers["actual_height_sum"] += height
        self._last_action.copy_(action)

    def observe_reward(self, reward: torch.Tensor) -> None:
        reward = _finite_or_zero(reward)
        self._buffers["reward_total"] += reward
        step_reward = _finite_or_zero(self.env.reward_manager._step_reward)
        for index, name in enumerate(self.reward_names):
            self._reward_weighted[name] += step_reward[:, index] * self.dt

    def episode_metrics(
        self,
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        base_height_failure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-environment episode metrics without reducing the batch."""
        steps = self._buffers["steps"][env_ids].clamp_min(1.0)
        metrics = {
            "optimization/episode_reward": self._buffers["reward_total"][env_ids],
            "optimization/episode_length_s": self._buffers["steps"][env_ids] * self.dt,
            "locomotion/forward_velocity_rmse_mps": torch.sqrt(
                self._buffers["forward_error_sq"][env_ids] / steps
            ),
            "locomotion/lateral_velocity_rmse_mps": torch.sqrt(
                self._buffers["lateral_error_sq"][env_ids] / steps
            ),
            "locomotion/yaw_rate_rmse_radps": torch.sqrt(
                self._buffers["yaw_error_sq"][env_ids] / steps
            ),
            "locomotion/body_height_rmse_m": torch.sqrt(
                self._buffers["height_error_sq"][env_ids] / steps
            ),
            "locomotion/forward_distance_m": self._buffers["forward_distance"][env_ids],
            "safety/unsafe_termination": terminated[env_ids].float(),
            "safety/timeout": truncated[env_ids].float(),
            "safety/base_height_failure": base_height_failure[env_ids].float(),
            "safety/base_collision_rate": self._buffers["base_collision_count"][env_ids] / steps,
            "support/invalid_rate": self._buffers["support_invalid_count"][env_ids] / steps,
            "support/wheel_contact_fraction": self._buffers["supported_wheel_count"][env_ids] / (4.0 * steps),
            "safety/base_tilt_mean_rad": self._buffers["tilt_sum"][env_ids] / steps,
            "safety/base_tilt_rms_rad": torch.sqrt(
                self._buffers["tilt_sq"][env_ids] / steps
            ),
            "safety/base_tilt_max_rad": self._buffers["tilt_max"][env_ids],
            "actuation/leg_action_rms": torch.sqrt(
                self._buffers["leg_action_sq"][env_ids] / steps
            ),
            "actuation/wheel_action_rms": torch.sqrt(
                self._buffers["wheel_action_sq"][env_ids] / steps
            ),
            "actuation/action_rate_rms": torch.sqrt(
                self._buffers["action_rate_sq"][env_ids] / steps
            ),
            "actuation/action_saturation_rate": self._buffers[
                "action_saturation_count"
            ][env_ids]
            / steps,
            "actuation/joint_position_limit_rate": self._buffers[
                "joint_position_limit_count"
            ][env_ids]
            / steps,
            "actuation/joint_velocity_limit_rate": self._buffers[
                "joint_velocity_limit_count"
            ][env_ids]
            / steps,
            "actuation/leg_mechanical_energy_j": self._buffers["leg_mechanical_energy"][
                env_ids
            ],
            "actuation/wheel_mechanical_energy_j": self._buffers[
                "wheel_mechanical_energy"
            ][env_ids],
            "runtime/finite_rate": self._buffers["finite_steps"][env_ids] / steps,
            "runtime/all_finite": (
                self._buffers["finite_steps"][env_ids] == steps
            ).float(),
            "terrain/type_id": self._episode_terrain_type[env_ids].float(),
            "terrain/level": self._episode_terrain_level[env_ids].float(),
        }
        terrain_cfg = self.env.cfg.scene.terrain.terrain_generator
        if terrain_cfg is None:
            metrics["terrain/difficulty_midpoint"] = torch.zeros_like(steps)
        else:
            num_rows = max(int(terrain_cfg.num_rows), 1)
            lower, upper = terrain_cfg.difficulty_range
            metrics["terrain/difficulty_midpoint"] = lower + (upper - lower) * (
                (self._episode_terrain_level[env_ids].float() + 0.5) / num_rows
            )
        for name in ("forward", "lateral", "yaw", "height"):
            metrics[f"command/raw_{name}_mean"] = (
                self._buffers[f"raw_{name}_sum"][env_ids] / steps
            )
            metrics[f"command/shaped_{name}_mean"] = (
                self._buffers[f"shaped_{name}_sum"][env_ids] / steps
            )
            metrics[f"locomotion/actual_{name}_mean"] = (
                self._buffers[f"actual_{name}_sum"][env_ids] / steps
            )
        for name, weighted in self._reward_weighted.items():
            values = weighted[env_ids]
            metrics[f"reward/weighted/{name}"] = values
            weight = self.reward_weights[name]
            metrics[f"reward/raw/{name}"] = (
                values / weight if abs(weight) > 1.0e-12 else torch.zeros_like(values)
            )
        return metrics


def aggregate_metric_batch(
    metrics: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Reduce per-environment episode metrics into logger-ready scalars."""
    if not metrics:
        return {}
    result = {
        f"{prefix}/{name}": values.float().mean() for name, values in metrics.items()
    }
    tilt = metrics["safety/base_tilt_max_rad"].float()
    result[f"{prefix}/safety/base_tilt_max_p50_rad"] = torch.quantile(tilt, 0.50)
    result[f"{prefix}/safety/base_tilt_max_p95_rad"] = torch.quantile(tilt, 0.95)
    return result


def terrain_family_metrics(
    metrics: dict[str, torch.Tensor],
    prefix: str,
    terrain_families: tuple[str, ...] = TERRAIN_FAMILIES,
) -> dict[str, torch.Tensor]:
    """Aggregate core metrics separately for each terrain family represented in the batch."""
    result: dict[str, torch.Tensor] = {}
    terrain_type = metrics["terrain/type_id"].long()
    selected_names = (
        "locomotion/forward_velocity_rmse_mps",
        "locomotion/yaw_rate_rmse_radps",
        "safety/unsafe_termination",
        "safety/timeout",
        "actuation/action_saturation_rate",
        "actuation/leg_mechanical_energy_j",
        "actuation/wheel_mechanical_energy_j",
        "runtime/all_finite",
    )
    for family_id, family in enumerate(terrain_families):
        mask = terrain_type == family_id
        if not bool(mask.any()):
            continue
        for name in selected_names:
            output_name = "timeout_survival_rate" if name == "safety/timeout" else name
            result[f"{prefix}/terrain/{family}/{output_name}"] = (
                metrics[name][mask].float().mean()
            )
    return result
