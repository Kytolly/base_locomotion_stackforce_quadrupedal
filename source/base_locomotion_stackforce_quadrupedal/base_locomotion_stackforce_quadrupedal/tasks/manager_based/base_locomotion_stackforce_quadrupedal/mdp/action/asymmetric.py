"""Per-joint asymmetric joint-position action for the E0-v2 servo contract.

raw 0 maps to the default stance, raw +1 to the joint's upper servo limit and
raw -1 to its lower limit; each side is linear, so the two sides may have
different gains.  Targets are finally clamped to ``[lower, upper]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.configclass import configclass

from .servo_range import project_leg_target_grid, raw_to_joint_target

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

    @property
    def joint_lower(self) -> torch.Tensor:
        return self._lower

    @property
    def joint_upper(self) -> torch.Tensor:
        return self._upper

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        target = raw_to_joint_target(self._raw_actions, self._lower, self._offset, self._upper)
        target = torch.clamp(target, min=self._lower, max=self._upper)
        for outer_index, inner_index, spec in self._projection:
            pair = torch.stack((target[:, outer_index], target[:, inner_index]), dim=-1)
            projected = project_leg_target_grid(pair, spec)
            target[:, outer_index], target[:, inner_index] = projected[:, 0], projected[:, 1]
        self._processed_actions = target


@configclass
class AsymmetricJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action whose raw [-1, 1] spans each joint's own [lower, upper]."""

    class_type: type = AsymmetricJointPositionAction
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    projection: dict[str, dict] = {}
    scale: float = 1.0
    clip: dict | None = None


__all__ = ["AsymmetricJointPositionAction", "AsymmetricJointPositionActionCfg"]
