"""Shared helpers for the ACTION_RANGE_THROUGHPUT_FIX_PLAN gate scripts (G3/G7).

Import this only after ``AppLauncher`` has started Kit.  The helpers drive the
articulation directly (no policy, no action manager) so a probe measures the
mechanism, not a controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = "Base-Locomotion-Stackforce-Quadrupedal-Complex-v0"
LEGS = ("FR", "FL", "RL", "RR")
SUSPENDED_ROOT_HEIGHT_M = 0.35  # wheels stay > 0.15 m above the plane at full extension


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"git_commit": git_commit(), **payload}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[FIX-PLAN] wrote {path}", flush=True)
    return path


def make_env(
    num_envs: int,
    device: str,
    *,
    suspended: bool,
    seed: int = 1001,
    foot_contact_update_period: float | None = None,
):
    """Create the E0-v2 task on a plane with randomisation off."""
    import gymnasium as gym
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401

    cfg = parse_env_cfg(TASK, device=device, num_envs=num_envs)
    cfg.seed = seed
    cfg.action_contract = "servo_range_v2"
    cfg.apply_action_contract(rebuild_terrain=False)
    if foot_contact_update_period is not None:
        for leg in LEGS:
            getattr(cfg.scene, f"contact_{leg.lower()}").update_period = foot_contact_update_period
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1, env_spacing=1.0, debug_vis=False
    )
    cfg.scene.env_spacing = 1.0
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.motion_family = None
    cfg.curriculum.robustness_bin = None
    for name in ("randomize_contact_material", "randomize_base_mass", "randomize_actuator_gains"):
        setattr(cfg.events, name, None)
    cfg.events.reset_root.params["slope_approach_fraction"] = 0.0
    cfg.commands.locomotion.resampling_time_range = (1.0e9, 1.0e9)
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = 1.0e6
    if suspended:
        cfg.scene.robot.init_state.pos = (0.0, 0.0, SUSPENDED_ROOT_HEIGHT_M)
        cfg.terminations.base_height = None
        cfg.terminations.excessive_tilt = None
        cfg.terminations.base_collision = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset(seed=seed)
    return env


@dataclass
class Rig:
    """Direct joint-target driver with an optional held (fixed) root."""

    env: Any
    suspended: bool

    def __post_init__(self) -> None:
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import (
            FOOT_LINKS,
            LEG_JOINTS,
        )

        self.robot = self.env.scene["robot"]
        self.leg_ids, _ = self.robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        self.foot_ids, _ = self.robot.find_bodies(list(FOOT_LINKS), preserve_order=True)
        self.outer_thigh_ids, _ = self.robot.find_bodies([f"{l}_Outer_Thigh_Link" for l in LEGS], preserve_order=True)
        self.outer_calf_ids, _ = self.robot.find_bodies([f"{l}_Outer_Calf_Link" for l in LEGS], preserve_order=True)
        self.inner_thigh_ids, _ = self.robot.find_bodies([f"{l}_Inner_Thigh_Link" for l in LEGS], preserve_order=True)
        self.inner_calf_ids, _ = self.robot.find_bodies([f"{l}_Inner_Calf_Link" for l in LEGS], preserve_order=True)
        self.root_state = self.robot.data.default_root_state.clone()
        self.root_state[:, :3] += self.env.scene.env_origins
        self.physics_dt = float(self.env.physics_dt)
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots.kinematics import (
            LegKinematics,
        )

        kin = LegKinematics()
        device = self.env.device
        self.closure_offset = torch.tensor(
            [kin.joints[f"{l}_Closure_Frame_Joint"].xyz for l in LEGS], dtype=torch.float32, device=device
        )
        self.closure_axis = torch.tensor(
            [kin.joints[f"{l}_Inner_Knee_Joint"].axis for l in LEGS], dtype=torch.float32, device=device
        )

    @property
    def default_leg_pos(self) -> torch.Tensor:
        return self.robot.data.default_joint_pos[:, self.leg_ids].clone()

    def _hold_root(self) -> None:
        if self.suspended:
            self.robot.write_root_pose_to_sim(self.root_state[:, :7])
            self.robot.write_root_velocity_to_sim(torch.zeros_like(self.root_state[:, 7:]))

    def run(self, leg_targets: torch.Tensor, seconds: float, start: torch.Tensor | None = None) -> None:
        """Ramp (smoothstep) from ``start`` to ``leg_targets`` over ``seconds``; wheels held at zero speed."""
        steps = max(1, round(seconds / self.physics_dt))
        start = self.robot.data.joint_pos_target[:, self.leg_ids].clone() if start is None else start
        for index in range(steps):
            alpha = (index + 1) / steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            self.robot.set_joint_position_target(start + alpha * (leg_targets - start), joint_ids=self.leg_ids)
            self.step_physics()

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.physics_dt))):
            self.step_physics()

    def step_physics(self) -> None:
        self._hold_root()
        self.env.scene.write_data_to_sim()
        self.env.sim.step(render=False)
        self.env.scene.update(dt=self.physics_dt)

    # ----------------------------------------------------------- measurements
    def base_frame(self, points_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        data = self.robot.data
        rel = points_w - data.root_pos_w[:, None, :]
        quat = data.root_quat_w[:, None, :].expand(-1, rel.shape[1], -1)
        return quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape)

    def feet_base(self) -> torch.Tensor:
        return self.base_frame(self.robot.data.body_pos_w[:, self.foot_ids])

    def crank_vectors_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Outer and inner crank vectors (knee origin minus hip origin) in the base frame."""
        pos = self.robot.data.body_pos_w
        outer = self.base_frame(pos[:, self.outer_calf_ids]) - self.base_frame(pos[:, self.outer_thigh_ids])
        inner = self.base_frame(pos[:, self.inner_calf_ids]) - self.base_frame(pos[:, self.inner_thigh_ids])
        return outer, inner

    def closure_residual_mm(self) -> torch.Tensor:
        """Per-leg radial loop-closure gap in mm, shape (num_envs, 4)."""
        from isaaclab.utils.math import quat_apply

        data = self.robot.data
        calf_pos = data.body_pos_w[:, self.inner_calf_ids]
        calf_quat = data.body_quat_w[:, self.inner_calf_ids]
        shape = calf_quat.shape[:2]
        closure = calf_pos + quat_apply(
            calf_quat.reshape(-1, 4), self.closure_offset.expand(*shape, 3).reshape(-1, 3)
        ).reshape(*shape, 3)
        axis = quat_apply(calf_quat.reshape(-1, 4), self.closure_axis.expand(*shape, 3).reshape(-1, 3)).reshape(*shape, 3)
        wheel = data.body_pos_w[:, self.foot_ids]
        delta = closure - wheel
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        radial = delta - (delta * axis).sum(-1, keepdim=True) * axis
        return radial.norm(dim=-1) * 1000.0

    def contact_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(base force norm (N,), per-foot force norms (N, 4)) from the scene contact sensors."""
        base = self.env.scene["base_contact"].data.net_forces_w.norm(dim=-1).amax(dim=-1)
        feet = torch.stack(
            [self.env.scene[f"contact_{l.lower()}"].data.net_forces_w.norm(dim=-1).amax(dim=-1) for l in LEGS], dim=-1
        )
        return base, feet

    def tracking_error(self) -> torch.Tensor:
        data = self.robot.data
        return (data.joint_pos_target[:, self.leg_ids] - data.joint_pos[:, self.leg_ids]).abs()

    def leg_torque(self) -> torch.Tensor:
        return self.robot.data.applied_torque[:, self.leg_ids]


__all__ = ["LEGS", "ROOT", "Rig", "SUSPENDED_ROOT_HEIGHT_M", "TASK", "git_commit", "make_env", "sha256", "write_json"]
