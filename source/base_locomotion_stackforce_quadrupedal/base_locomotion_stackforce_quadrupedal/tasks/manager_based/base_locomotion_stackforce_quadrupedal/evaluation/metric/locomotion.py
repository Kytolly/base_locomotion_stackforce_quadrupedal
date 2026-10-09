"""Vectorized episode metrics shared by training and validation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...env.robots import LEG_JOINTS, WHEEL_JOINTS
from ...env.terrains import COMPOSITION_FAMILY, TERRAIN_FAMILIES, TRAINING_TERRAIN_FAMILIES
from ...mdp.action import LEG_ACTION_SLICE, WHEEL_ACTION_SLICE
from ...mdp.policy.commands import (
    BODY_HEIGHT,
    FORWARD,
    LATERAL,
    LOCOMOTION_COMMAND_NAME,
    MOTION_FAMILY_NAMES,
    YAW_RATE,
)
from ...mdp.support import local_support_height, wheel_contact_state
from base_locomotion_stackforce_quadrupedal.evaluation.metric.logging import (
    DEFAULT_METRIC_GROUPS,
)


ROUTE_ROLE_NAMES = ("single", "entry", "interior", "exit_recovery")
CONTINUOUS_ROUTE_CHECKPOINTS_M = (1.0, 2.5, 3.5)
CONTINUOUS_ROUTE_HALF_WIDTH_M = 1.5


def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class LocomotionEpisodeMetrics:
    """Accumulate physical and optimization metrics independently for every environment."""

    def __init__(
        self,
        env,
        action_saturation_threshold: float = 0.95,
        metric_groups: dict[str, bool] | None = None,
    ) -> None:
        if not 0.0 <= action_saturation_threshold < 1.0:
            raise ValueError("action_saturation_threshold must be in [0, 1).")
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.dt = float(env.step_dt)
        self.action_saturation_threshold = action_saturation_threshold
        self.metric_groups = dict(DEFAULT_METRIC_GROUPS)
        if metric_groups is not None:
            self.metric_groups.update({name: bool(value) for name, value in metric_groups.items()})
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
        } if self.metric_groups["reward"] else {}
        self._last_action = torch.zeros(
            (self.num_envs, env.action_manager.total_action_dim), device=self.device
        )
        self._episode_terrain_type = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_terrain_level = torch.zeros_like(self._episode_terrain_type)
        self._episode_motion_family = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
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
            "command_feasible_count",
            "continuous_route_ordered_checkpoints",
            "continuous_route_corridor_valid",
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
        self._buffers["continuous_route_corridor_valid"][env_ids] = 1.0
        terrain = self.env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_types is not None:
            family_by_column = getattr(terrain, "validation_family_by_column", None)
            if family_by_column is None:
                self._episode_terrain_type[env_ids] = terrain_types[env_ids]
            else:
                self._episode_terrain_type[env_ids] = family_by_column[
                    terrain_types[env_ids]
                ]
        if terrain_levels is not None:
            self._episode_terrain_level[env_ids] = terrain_levels[env_ids]
        command_term = self.env.command_manager.get_term(LOCOMOTION_COMMAND_NAME)
        family = getattr(command_term, "motion_family", None)
        if family is not None:
            self._episode_motion_family[env_ids] = family[env_ids]

    def observe_state(self, action: torch.Tensor) -> None:
        """Accumulate the pre-transition state and clipped raw policy action."""
        robot = self.env.scene["robot"]
        command_term = self.env.command_manager.get_term(LOCOMOTION_COMMAND_NAME)
        command = command_term.command
        self._buffers["steps"] += 1.0

        if self.metric_groups["core"] or self.metric_groups["runtime"]:
            root_lin_vel_b = robot.data.root_lin_vel_b
            root_ang_vel_b = robot.data.root_ang_vel_b
            projected_gravity_b = robot.data.projected_gravity_b
            if self.metric_groups["runtime"]:
                finite = torch.isfinite(command).all(dim=1)
                finite &= torch.isfinite(action).all(dim=1)
                finite &= torch.isfinite(root_lin_vel_b).all(dim=1)
                finite &= torch.isfinite(root_ang_vel_b).all(dim=1)
                self._buffers["finite_steps"] += finite.float()
            if self.metric_groups["core"]:
                command = _finite_or_zero(command)
                root_lin_vel_b = _finite_or_zero(root_lin_vel_b)
                root_ang_vel_b = _finite_or_zero(root_ang_vel_b)
                support_height, _ = local_support_height(self.env)
                height = _finite_or_zero(robot.data.root_pos_w[:, 2] - support_height)
                errors = {
                    "forward_error_sq": command[:, FORWARD] - root_lin_vel_b[:, 1],
                    "lateral_error_sq": command[:, LATERAL] - root_lin_vel_b[:, 0],
                    "yaw_error_sq": command[:, YAW_RATE] - root_ang_vel_b[:, 2],
                    "height_error_sq": command[:, BODY_HEIGHT] - height,
                }
                for name, error in errors.items():
                    self._buffers[name] += torch.square(error)
                self._buffers["forward_distance"] += root_lin_vel_b[:, 1] * self.dt

        if self.metric_groups["safety"]:
            projected_gravity_b = _finite_or_zero(robot.data.projected_gravity_b)
            tilt = torch.acos(torch.clamp(-projected_gravity_b[:, 2], -1.0, 1.0))
            self._buffers["tilt_sum"] += tilt
            self._buffers["tilt_sq"] += torch.square(tilt)
            self._buffers["tilt_max"] = torch.maximum(self._buffers["tilt_max"], tilt)
            base_force = self.env.scene["base_contact"].data.net_forces_w_history
            base_collision = torch.linalg.vector_norm(base_force, dim=-1).amax(dim=-1).amax(dim=-1) > 8.0
            self._buffers["base_collision_count"] += base_collision.float()

        if self.metric_groups["actuation"]:
            action = _finite_or_zero(action)
            joint_pos = _finite_or_zero(robot.data.joint_pos[:, self.active_joint_ids])
            joint_vel = _finite_or_zero(robot.data.joint_vel[:, self.active_joint_ids])
            joint_torque = _finite_or_zero(robot.data.applied_torque[:, self.active_joint_ids])
            self._buffers["leg_action_sq"] += torch.mean(torch.square(action[:, LEG_ACTION_SLICE]), dim=1)
            self._buffers["wheel_action_sq"] += torch.mean(torch.square(action[:, WHEEL_ACTION_SLICE]), dim=1)
            self._buffers["action_rate_sq"] += torch.mean(torch.square(action - self._last_action), dim=1)
            self._buffers["action_saturation_count"] += (
                (torch.abs(action) >= self.action_saturation_threshold).float().mean(dim=1)
            )
            soft_position_limits = robot.data.soft_joint_pos_limits[:, self.active_joint_ids]
            soft_velocity_limits = robot.data.soft_joint_vel_limits[:, self.active_joint_ids]
            position_limited = (joint_pos < soft_position_limits[:, :, 0]) | (
                joint_pos > soft_position_limits[:, :, 1]
            )
            velocity_limited = torch.abs(joint_vel) > 0.9 * soft_velocity_limits
            self._buffers["joint_position_limit_count"] += position_limited.any(dim=1).float()
            self._buffers["joint_velocity_limit_count"] += velocity_limited.any(dim=1).float()
            leg_count = len(self.leg_joint_ids)
            power = torch.abs(joint_torque * joint_vel)
            self._buffers["leg_mechanical_energy"] += torch.sum(power[:, :leg_count], dim=1) * self.dt
            self._buffers["wheel_mechanical_energy"] += torch.sum(power[:, leg_count:], dim=1) * self.dt
            self._last_action.copy_(action)

        if self.metric_groups["support"]:
            _, support_valid = local_support_height(self.env)
            contact = wheel_contact_state(self.env)
            self._buffers["support_invalid_count"] += (~support_valid).float()
            self._buffers["supported_wheel_count"] += contact.sum(dim=1).float()

        if self.metric_groups["command"]:
            raw_command = _finite_or_zero(command_term.raw_command)
            feasible_mask = getattr(command_term, "feasible_mask", torch.ones_like(command))
            self._buffers["command_feasible_count"] += feasible_mask.all(dim=1).float()
            for index, name in enumerate(("forward", "lateral", "yaw", "height")):
                self._buffers[f"raw_{name}_sum"] += raw_command[:, index]
                self._buffers[f"shaped_{name}_sum"] += command[:, index]

        if self.metric_groups["terrain"] or self.metric_groups["motion"]:
            family = getattr(command_term, "motion_family", None)
            if family is not None:
                unset = self._episode_motion_family < 0
                self._episode_motion_family[unset] = family[unset]
        if self.metric_groups["terrain"]:
            continuous_id = TRAINING_TERRAIN_FAMILIES.index(COMPOSITION_FAMILY)
            continuous = self._episode_terrain_type == continuous_id
            local_position = robot.data.root_pos_w[:, :2] - self.env.scene.terrain.env_origins[:, :2]
            in_corridor = torch.abs(local_position[:, 0]) <= CONTINUOUS_ROUTE_HALF_WIDTH_M
            self._buffers["continuous_route_corridor_valid"][continuous & ~in_corridor] = 0.0
            progress = self._buffers["continuous_route_ordered_checkpoints"]
            checkpoint_index = progress.long().clamp_max(len(CONTINUOUS_ROUTE_CHECKPOINTS_M) - 1)
            checkpoint_targets = torch.tensor(
                CONTINUOUS_ROUTE_CHECKPOINTS_M, device=self.device
            )[checkpoint_index]
            reached = (
                continuous
                & in_corridor
                & (progress < len(CONTINUOUS_ROUTE_CHECKPOINTS_M))
                & (local_position[:, 1] >= checkpoint_targets)
            )
            progress[reached] += 1.0

    def observe_reward(self, reward: torch.Tensor) -> None:
        if self.metric_groups["core"]:
            self._buffers["reward_total"] += _finite_or_zero(reward)
        if self.metric_groups["reward"]:
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
        metrics: dict[str, torch.Tensor] = {"optimization/episode_length_s": steps * self.dt}
        if self.metric_groups["core"]:
            metrics.update({
                "optimization/episode_reward": self._buffers["reward_total"][env_ids],
                "locomotion/forward_velocity_rmse_mps": torch.sqrt(self._buffers["forward_error_sq"][env_ids] / steps),
                "locomotion/lateral_velocity_rmse_mps": torch.sqrt(self._buffers["lateral_error_sq"][env_ids] / steps),
                "locomotion/yaw_rate_rmse_radps": torch.sqrt(self._buffers["yaw_error_sq"][env_ids] / steps),
                "locomotion/body_height_rmse_m": torch.sqrt(self._buffers["height_error_sq"][env_ids] / steps),
                "locomotion/forward_distance_m": self._buffers["forward_distance"][env_ids],
            })
        if self.metric_groups["safety"]:
            metrics.update({
                "safety/unsafe_termination": terminated[env_ids].float(),
                "safety/timeout": truncated[env_ids].float(),
                "safety/base_height_failure": base_height_failure[env_ids].float(),
                "safety/base_collision_rate": self._buffers["base_collision_count"][env_ids] / steps,
                "safety/base_tilt_mean_rad": self._buffers["tilt_sum"][env_ids] / steps,
                "safety/base_tilt_rms_rad": torch.sqrt(self._buffers["tilt_sq"][env_ids] / steps),
                "safety/base_tilt_max_rad": self._buffers["tilt_max"][env_ids],
            })
        if self.metric_groups["support"]:
            metrics.update({
                "support/invalid_rate": self._buffers["support_invalid_count"][env_ids] / steps,
                "support/wheel_contact_fraction": self._buffers["supported_wheel_count"][env_ids] / (4.0 * steps),
            })
        if self.metric_groups["actuation"]:
            metrics.update({
                "actuation/leg_action_rms": torch.sqrt(self._buffers["leg_action_sq"][env_ids] / steps),
                "actuation/wheel_action_rms": torch.sqrt(self._buffers["wheel_action_sq"][env_ids] / steps),
                "actuation/action_rate_rms": torch.sqrt(self._buffers["action_rate_sq"][env_ids] / steps),
                "actuation/action_saturation_rate": self._buffers["action_saturation_count"][env_ids] / steps,
                "actuation/joint_position_limit_rate": self._buffers["joint_position_limit_count"][env_ids] / steps,
                "actuation/joint_velocity_limit_rate": self._buffers["joint_velocity_limit_count"][env_ids] / steps,
                "actuation/leg_mechanical_energy_j": self._buffers["leg_mechanical_energy"][env_ids],
                "actuation/wheel_mechanical_energy_j": self._buffers["wheel_mechanical_energy"][env_ids],
            })
        if self.metric_groups["runtime"]:
            metrics.update({
                "runtime/finite_rate": self._buffers["finite_steps"][env_ids] / steps,
                "runtime/all_finite": (self._buffers["finite_steps"][env_ids] == steps).float(),
            })
        if self.metric_groups["terrain"]:
            route_role = torch.zeros_like(steps)
            continuous_id = TRAINING_TERRAIN_FAMILIES.index(COMPOSITION_FAMILY)
            continuous = self._episode_terrain_type[env_ids] == continuous_id
            local_forward = (
                self.env.scene["robot"].data.root_pos_w[env_ids, 1]
                - self.env.scene.terrain.env_origins[env_ids, 1]
            )
            route_role[continuous & (local_forward <= 1.0)] = 1.0
            route_role[continuous & (local_forward > 1.0) & (local_forward <= 3.0)] = 2.0
            route_role[continuous & (local_forward > 3.0)] = 3.0
            metrics.update({
                "terrain/type_id": self._episode_terrain_type[env_ids].float(),
                "terrain/level": self._episode_terrain_level[env_ids].float(),
                "terrain/route_role_id": route_role,
                "terrain/continuous_route_ordered_checkpoints": self._buffers[
                    "continuous_route_ordered_checkpoints"
                ][env_ids],
                "terrain/continuous_route_complete": (
                    self._buffers["continuous_route_ordered_checkpoints"][env_ids]
                    == len(CONTINUOUS_ROUTE_CHECKPOINTS_M)
                ).float(),
                "terrain/continuous_route_corridor_valid": self._buffers[
                    "continuous_route_corridor_valid"
                ][env_ids],
            })
            terrain_cfg = self.env.cfg.scene.terrain.terrain_generator
            if terrain_cfg is None:
                metrics["terrain/difficulty_midpoint"] = torch.zeros_like(steps)
            else:
                num_rows = max(int(terrain_cfg.num_rows), 1)
                lower, upper = terrain_cfg.difficulty_range
                metrics["terrain/difficulty_midpoint"] = lower + (upper - lower) * (
                    (self._episode_terrain_level[env_ids].float() + 0.5) / num_rows
                )
        if self.metric_groups["motion"]:
            metrics["motion/family_id"] = self._episode_motion_family[env_ids].float()
        if self.metric_groups["command"]:
            metrics["command/feasible_rate"] = self._buffers["command_feasible_count"][env_ids] / steps
            for name in ("forward", "lateral", "yaw", "height"):
                metrics[f"command/raw_{name}_mean"] = self._buffers[f"raw_{name}_sum"][env_ids] / steps
                metrics[f"command/shaped_{name}_mean"] = self._buffers[f"shaped_{name}_sum"][env_ids] / steps
        if self.metric_groups["reward"]:
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
    if "safety/base_tilt_max_rad" in metrics:
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
            if name not in metrics:
                continue
            output_name = "timeout_survival_rate" if name == "safety/timeout" else name
            result[f"{prefix}/terrain/{family}/{output_name}"] = (
                metrics[name][mask].float().mean()
            )
    return result


def route_role_metrics(
    metrics: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Aggregate route outcomes by single/entry/interior/recovery role."""
    result: dict[str, torch.Tensor] = {}
    role_ids = metrics["terrain/route_role_id"].long()
    for role_id, role_name in enumerate(ROUTE_ROLE_NAMES):
        mask = role_ids == role_id
        if not bool(mask.any()):
            continue
        for name in (
            "locomotion/forward_velocity_rmse_mps",
            "safety/unsafe_termination",
            "actuation/action_saturation_rate",
            "runtime/all_finite",
        ):
            if name in metrics:
                result[f"{prefix}/route/{role_name}/{name}"] = metrics[name][mask].float().mean()
    return result


def motion_family_metrics(
    metrics: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Aggregate command-aligned outcomes without treating family as an observation."""
    result: dict[str, torch.Tensor] = {}
    family_ids = metrics["motion/family_id"].long()
    for family_id, family in enumerate(MOTION_FAMILY_NAMES):
        mask = family_ids == family_id
        if not bool(mask.any()):
            continue
        for name in (
            "locomotion/forward_velocity_rmse_mps",
            "locomotion/lateral_velocity_rmse_mps",
            "locomotion/yaw_rate_rmse_radps",
            "safety/unsafe_termination",
            "command/feasible_rate",
        ):
            if name not in metrics:
                continue
            result[f"{prefix}/motion/{family}/{name}"] = metrics[name][mask].float().mean()
    return result
