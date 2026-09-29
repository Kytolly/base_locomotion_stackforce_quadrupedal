#!/usr/bin/env python3
"""Probe each policy action channel against its simulated joint target and motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Base-Locomotion-Stackforce-Quadrupedal-Complex-v0")
parser.add_argument("--action-magnitude", type=float, default=0.2)
parser.add_argument("--steps-per-probe", type=int, default=5)
parser.add_argument("--output", type=Path, default=Path("output/action-probes/latest.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import (  # noqa: E501,E402
    ACTIVE_JOINTS,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E501,E402
    LEG_POSITION_SCALE,
    POLICY_ACTION_DIM,
    WHEEL_VELOCITY_SCALE,
)


def _tensor(value: object) -> torch.Tensor:
    return getattr(value, "torch", value)


def main() -> None:
    if not 0.0 < args_cli.action_magnitude <= 1.0:
        raise ValueError("--action-magnitude must be in (0, 1].")
    if args_cli.steps_per_probe <= 0:
        raise ValueError("--steps-per-probe must be positive.")
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        joint_ids, _ = robot.find_joints(list(ACTIVE_JOINTS), preserve_order=True)
        initial_pos = _tensor(robot.data.joint_pos)[0, joint_ids].clone()
        initial_vel = _tensor(robot.data.joint_vel)[0, joint_ids].clone()
        initial_root = _tensor(robot.data.root_pos_w)[0].clone()
        traces = []
        contract_pass = True
        for action_index in range(POLICY_ACTION_DIM):
            env.reset()
            robot = unwrapped.scene["robot"]
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
                    "root_displacement_m": (root - initial_root).tolist(),
                }
            )
        report = {
            "schema_version": 1,
            "task": args_cli.task,
            "action_dim": POLICY_ACTION_DIM,
            "joint_order": list(ACTIVE_JOINTS),
            "action_magnitude": args_cli.action_magnitude,
            "steps_per_probe": args_cli.steps_per_probe,
            "target_contract_pass": contract_pass,
            "expected_leg_target_delta_rad": args_cli.action_magnitude * LEG_POSITION_SCALE,
            "expected_wheel_target_radps": args_cli.action_magnitude * WHEEL_VELOCITY_SCALE,
            "initial_joint_position_rad": initial_pos.tolist(),
            "initial_joint_velocity_radps": initial_vel.tolist(),
            "traces": traces,
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
