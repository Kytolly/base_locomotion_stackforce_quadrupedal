"""Per-joint asymmetric joint-position action for the E0-v2 servo contract.

raw 0 maps to the default stance, raw +1 to the joint's upper servo limit and
raw -1 to its lower limit; each side is linear, so the two sides may have
different gains.  Targets are finally clamped to ``[lower, upper]``.

``max_target_step_rad`` optionally limits how far a target may move per control
step (slew limit).  The limited target is mapped back to raw space and stored in
``raw_actions``, which ``mdp.observation.history.previous_action`` reads, so the
policy observes what was actually commanded.  A deployment must apply the same
limit before the servos.

The term also keeps the deployable servo-position estimate
(``servo_model_step``): the policy observes it instead of simulated joint
state, because the robot's PWM servos report no position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.configclass import configclass

from .servo_range import joint_target_to_raw, project_leg_target_grid, raw_to_joint_target, servo_model_step

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AsymmetricJointPositionAction(JointPositionAction):
    """Map raw actions piecewise-linearly onto per-joint servo ranges."""

    cfg: "AsymmetricJointPositionActionCfg"

    def __init__(self, cfg: "AsymmetricJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        missing = [name for name in self._joint_names if name not in cfg.lower or name not in cfg.upper]
        if missing:
            raise ValueError(f"Asymmetric action has no range for joints {missing}.")
        shape = (self.num_envs, self.action_dim)
        self._lower = torch.tensor([cfg.lower[name] for name in self._joint_names], device=self.device).expand(shape)
        self._upper = torch.tensor([cfg.upper[name] for name in self._joint_names], device=self.device).expand(shape)
        offset = self._offset if isinstance(self._offset, torch.Tensor) else torch.full(shape, float(self._offset), device=self.device)
        self._offset = offset.clone()
        if bool((self._offset <= self._lower).any()) or bool((self._offset >= self._upper).any()):
            raise ValueError("Default joint positions must lie strictly inside the asymmetric action range.")
        name_to_index = {name: index for index, name in enumerate(self._joint_names)}
        self._projection = []
        for leg, source in cfg.projection.items():
            spec = dict(source)
            spec["feasible_tensor"] = torch.tensor(spec.pop("feasible"), dtype=torch.bool, device=self.device)
            spec["nearest_tensor"] = torch.tensor(
                spec.pop("nearest_feasible_flat"), dtype=torch.long, device=self.device
            )
            self._projection.append((name_to_index[spec.pop("outer_joint")], name_to_index[spec.pop("inner_joint")], spec))
        if cfg.max_target_step_rad is not None and not cfg.max_target_step_rad > 0:
            raise ValueError("max_target_step_rad must be positive or None.")
        # The slew limit steps from the previous target; start every episode at the stance.
        self._processed_actions[:] = self._offset
        if cfg.servo_model_delay_steps < 0:
            raise ValueError("servo_model_delay_steps must be non-negative.")
        self._control_dt = float(env.step_dt)
        self._servo_max_step = float(cfg.servo_model_speed_rad_s) * self._control_dt
        self._target_history = self._offset.unsqueeze(0).repeat(cfg.servo_model_delay_steps + 1, 1, 1)
        self._servo_estimate = self._offset.clone()
        self._servo_estimate_vel = torch.zeros_like(self._offset)

    @property
    def joint_lower(self) -> torch.Tensor:
        return self._lower

    @property
    def joint_upper(self) -> torch.Tensor:
        return self._upper

    @property
    def servo_estimate(self) -> torch.Tensor:
        """Deployable servo-position estimate (rad), end of the current control step."""
        return self._servo_estimate

    @property
    def servo_estimate_vel(self) -> torch.Tensor:
        """Finite-difference velocity of ``servo_estimate`` over the last control step (rad/s)."""
        return self._servo_estimate_vel

    @property
    def stance(self) -> torch.Tensor:
        return self._offset

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        target = raw_to_joint_target(self._raw_actions, self._lower, self._offset, self._upper)
        target = torch.clamp(target, min=self._lower, max=self._upper)
        step = self.cfg.max_target_step_rad
        if step is not None:
            previous = self._processed_actions
            target = previous + torch.clamp(target - previous, min=-step, max=step)
        for outer_index, inner_index, spec in self._projection:
            pair = torch.stack((target[:, outer_index], target[:, inner_index]), dim=-1)
            projected = project_leg_target_grid(pair, spec)
            target[:, outer_index], target[:, inner_index] = projected[:, 0], projected[:, 1]
        self._processed_actions = target
        if step is not None:
            self._raw_actions[:] = joint_target_to_raw(target, self._lower, self._offset, self._upper)
        self._target_history = torch.roll(self._target_history, shifts=1, dims=0)
        self._target_history[0] = target
        previous = self._servo_estimate
        self._servo_estimate = servo_model_step(previous, self._target_history[-1], self._servo_max_step)
        self._servo_estimate_vel = (self._servo_estimate - previous) / self._control_dt

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # The robot starts every run holding the commanded stance; the estimate does too.
        self._servo_estimate[ids] = self._offset[ids]
        self._servo_estimate_vel[ids] = 0.0
        self._target_history[:, ids] = self._offset[ids]
        if self.cfg.max_target_step_rad is not None:
            ids = slice(None) if env_ids is None else env_ids
            # Joint state is written by the reset events before the action manager resets.
            self._processed_actions[ids] = self._asset.data.joint_pos.torch[ids][:, self._joint_ids]


@configclass
class AsymmetricJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action whose raw [-1, 1] spans each joint's own [lower, upper]."""

    class_type: type = AsymmetricJointPositionAction
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    projection: dict[str, dict] = {}
    max_target_step_rad: float | None = None
    """Per-control-step target change limit (rad); None disables the slew limit."""
    servo_model_speed_rad_s: float = 8.73
    """Speed of the deployable servo-position estimate (DS041MG no-load)."""
    servo_model_delay_steps: int = 0
    """Control steps between a command and the estimate starting to follow it."""
    scale: float = 1.0
    clip: dict | None = None


__all__ = ["AsymmetricJointPositionAction", "AsymmetricJointPositionActionCfg"]
