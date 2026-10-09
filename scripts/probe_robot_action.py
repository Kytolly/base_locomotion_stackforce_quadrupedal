#!/usr/bin/env python3
"""Probe the policy action target contract and coordinated wheel motion semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Base-Locomotion-Stackforce-Quadrupedal-Complex-v0")
parser.add_argument("--action-magnitude", type=float, default=0.2)
parser.add_argument("--steps-per-probe", type=int, default=5)
parser.add_argument("--settle-steps", type=int, default=50)
parser.add_argument("--motion-steps", type=int, default=200)
parser.add_argument("--output", type=Path, default=Path("output/action-probes/latest.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.terrains import TerrainImporterCfg

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import (  # noqa: E501,E402
    ACTIVE_JOINTS,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E501,E402
    LEG_POSITION_SCALE,
    POLICY_ACTION_DIM,
    WHEEL_VELOCITY_SCALE,
    WHEEL_ACTION_SLICE,
)


def _tensor(value: object) -> torch.Tensor:
    return getattr(value, "torch", value)


def _configure_flat_probe(cfg: object) -> None:
    cfg.scene.num_envs = 1
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=4.0,
        debug_vis=False,
    )
    cfg.curriculum.terrain_levels = None
    cfg.events.randomize_contact_material = None
    cfg.events.randomize_base_mass = None
    cfg.events.randomize_actuator_gains = None


def _coordinated_wheel_probe(
    env: object,
    pattern_name: str,
    wheel_pattern: tuple[float, float, float, float],
) -> dict[str, object]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action = torch.zeros((1, POLICY_ACTION_DIM), device=unwrapped.device)
    action[0, WHEEL_ACTION_SLICE] = args_cli.action_magnitude * torch.tensor(
        wheel_pattern, device=unwrapped.device
    )
    zero_action = torch.zeros_like(action)

    env.reset()
    settle_terminated = False
    for _ in range(args_cli.settle_steps):
        _, _, terminated, truncated, _ = env.step(zero_action)
        if bool((terminated | truncated)[0].item()):
            settle_terminated = True
            env.reset()

    root_start = _tensor(robot.data.root_pos_w)[0].clone()
    forward_velocity = []
    lateral_velocity = []
    yaw_velocity = []
    executed_steps = 0
    termination_reason = None

    for _ in range(args_cli.motion_steps):
        root_before = _tensor(robot.data.root_pos_w)[0].clone()
        velocity_before = _tensor(robot.data.root_lin_vel_b)[0].clone()
        yaw_velocity_before = _tensor(robot.data.root_ang_vel_b)[0, 2].clone()
        _, _, terminated, truncated, _ = env.step(action)
        executed_steps += 1
        forward_velocity.append(float(velocity_before[1].item()))
        lateral_velocity.append(float(velocity_before[0].item()))
        yaw_velocity.append(float(yaw_velocity_before.item()))
        if bool((terminated | truncated)[0].item()):
            root_end = root_before
            for name in unwrapped.termination_manager.active_terms:
                value = unwrapped.termination_manager.get_term(name)
                if value is not None and bool(value[0].item()):
                    termination_reason = name
                    break
            termination_reason = termination_reason or "unknown"
            break
    else:
        root_end = _tensor(robot.data.root_pos_w)[0].clone()

    displacement_w = root_end - root_start
    step_dt = float(unwrapped.step_dt)
    wheel_term = unwrapped.action_manager.get_term("wheel_velocity")
    wheel_joint_ids, _ = robot.find_joints(list(ACTIVE_JOINTS[8:]), preserve_order=True)
    wheel_velocity = _tensor(robot.data.joint_vel)[0, wheel_joint_ids]
    return {
        "pattern": pattern_name,
        "wheel_action": action[0, WHEEL_ACTION_SLICE].tolist(),
        "wheel_target_radps": wheel_term.processed_actions[0].tolist(),
        "wheel_measured_radps": wheel_velocity.tolist(),
        "settle_terminated": settle_terminated,
        "executed_steps": executed_steps,
        "termination_reason": termination_reason,
        "world_displacement_xyz_m": displacement_w.tolist(),
        "integrated_body_forward_displacement_m": float(sum(forward_velocity) * step_dt),
        "integrated_body_lateral_displacement_m": float(sum(lateral_velocity) * step_dt),
        "mean_body_forward_velocity_mps": float(torch.tensor(forward_velocity).mean().item()),
        "mean_body_lateral_velocity_mps": float(torch.tensor(lateral_velocity).mean().item()),
        "mean_body_yaw_velocity_radps": float(torch.tensor(yaw_velocity).mean().item()),
    }


def main() -> None:
    if not 0.0 < args_cli.action_magnitude <= 1.0:
        raise ValueError("--action-magnitude must be in (0, 1].")
    if args_cli.steps_per_probe <= 0:
        raise ValueError("--steps-per-probe must be positive.")
    if args_cli.settle_steps < 0 or args_cli.motion_steps <= 0:
        raise ValueError("--settle-steps must be non-negative and --motion-steps must be positive.")
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    _configure_flat_probe(cfg)
    env = gym.make(args_cli.task, cfg=cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        joint_ids, _ = robot.find_joints(list(ACTIVE_JOINTS), preserve_order=True)
        initial_pos = _tensor(robot.data.joint_pos)[0, joint_ids].clone()
        initial_vel = _tensor(robot.data.joint_vel)[0, joint_ids].clone()
        traces = []
        contract_pass = True
        for action_index in range(POLICY_ACTION_DIM):
            env.reset()
            robot = unwrapped.scene["robot"]
            root_before = _tensor(robot.data.root_pos_w)[0].clone()
            action = torch.zeros((1, POLICY_ACTION_DIM), device=unwrapped.device)
            action[0, action_index] = args_cli.action_magnitude
            for _ in range(args_cli.steps_per_probe):
                env.step(action)
            position_targets = _tensor(robot.data.joint_pos_target)[0, joint_ids]
            velocity_targets = _tensor(robot.data.joint_vel_target)[0, joint_ids]
            position = _tensor(robot.data.joint_pos)[0, joint_ids]
            velocity = _tensor(robot.data.joint_vel)[0, joint_ids]
            torque = _tensor(robot.data.applied_torque)[0, joint_ids]
            root = _tensor(robot.data.root_pos_w)[0]
            expected_position = torch.zeros_like(position_targets)
            expected_velocity = torch.zeros_like(velocity_targets)
            if action_index < 8:
                expected_position[action_index] = args_cli.action_magnitude * LEG_POSITION_SCALE
            else:
                expected_velocity[action_index] = args_cli.action_magnitude * WHEEL_VELOCITY_SCALE
            channel_pass = bool(
                torch.allclose(position_targets, expected_position, atol=1.0e-5, rtol=0.0)
                and torch.allclose(velocity_targets, expected_velocity, atol=1.0e-5, rtol=0.0)
            )
            contract_pass &= channel_pass
            traces.append(
                {
                    "action_index": action_index,
                    "target_contract_pass": channel_pass,
                    "joint_targets_position_rad": position_targets.tolist(),
                    "joint_targets_velocity_radps": velocity_targets.tolist(),
                    "joint_position_delta_rad": (position - initial_pos).tolist(),
                    "joint_velocity_radps": velocity.tolist(),
                    "applied_torque_nm": torque.tolist(),
                    "root_displacement_m": (root - root_before).tolist(),
                }
            )
        coordinated_patterns = {
            "all_positive": (1.0, 1.0, 1.0, 1.0),
            "all_negative": (-1.0, -1.0, -1.0, -1.0),
            "right_positive_left_negative": (1.0, -1.0, -1.0, 1.0),
            "front_positive_rear_negative": (1.0, 1.0, -1.0, -1.0),
        }
        coordinated_motion = [
            _coordinated_wheel_probe(env, name, pattern)
            for name, pattern in coordinated_patterns.items()
        ]
        report = {
            "schema_version": 1,
            "task": args_cli.task,
            "action_dim": POLICY_ACTION_DIM,
            "joint_order": list(ACTIVE_JOINTS),
            "action_magnitude": args_cli.action_magnitude,
            "steps_per_probe": args_cli.steps_per_probe,
            "settle_steps": args_cli.settle_steps,
            "motion_steps": args_cli.motion_steps,
            "probe_terrain": "plane",
            "target_contract_pass": contract_pass,
            "expected_leg_target_delta_rad": args_cli.action_magnitude * LEG_POSITION_SCALE,
            "expected_wheel_target_radps": args_cli.action_magnitude * WHEEL_VELOCITY_SCALE,
            "initial_joint_position_rad": initial_pos.tolist(),
            "initial_joint_velocity_radps": initial_vel.tolist(),
            "traces": traces,
            "coordinated_motion": coordinated_motion,
        }
        output = args_cli.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ACTION_PROBE] target_contract_pass={contract_pass} report={output}")
        if not contract_pass:
            raise RuntimeError(f"Action target contract failed; inspect {output}.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
