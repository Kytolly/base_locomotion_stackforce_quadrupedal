#!/usr/bin/env python3
"""Probe the policy action target contract and coordinated wheel motion semantics."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import traceback

_CUDA_LIB = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib"
if _CUDA_LIB.is_dir():
    for _name in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
        _path = _CUDA_LIB / _name
        if _path.is_file():
            ctypes.CDLL(str(_path), mode=ctypes.RTLD_GLOBAL)
    os.environ["LD_LIBRARY_PATH"] = f"{_CUDA_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Base-Locomotion-Stackforce-Quadrupedal-Complex-v0")
parser.add_argument("--action-magnitude", type=float, default=0.2)
parser.add_argument("--steps-per-probe", type=int, default=5)
parser.add_argument("--settle-steps", type=int, default=50)
parser.add_argument("--motion-steps", type=int, default=200)
parser.add_argument("--output", type=Path, default=Path("output/action-probes/latest.json"))
parser.add_argument("--action-contract", choices=("servo_range_v2", "legacy_symmetric"), default="servo_range_v2",
                    help="E0-v2 servo ranges (default) or the E0-v1 symmetric +-0.5 rad legs.")
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
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action.servo_range import (  # noqa: E501,E402
    load_servo_contract,
    project_leg_target_grid,
    raw_to_joint_target,
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
    cfg.curriculum.motion_family = None
    cfg.curriculum.robustness_bin = None
    cfg.events.randomize_contact_material = None
    cfg.events.randomize_base_mass = None
    cfg.events.randomize_actuator_gains = None


def _expected_servo_target(raw: torch.Tensor, lower: torch.Tensor, default: torch.Tensor, upper: torch.Tensor):
    target = raw_to_joint_target(raw, lower, default, upper)
    contract = load_servo_contract()
    index = {name: i for i, name in enumerate(ACTIVE_JOINTS[:8])}
    for source in contract.get("projection", {}).values():
        spec = dict(source)
        spec["feasible_tensor"] = torch.tensor(spec.pop("feasible"), dtype=torch.bool, device=raw.device)
        spec["nearest_tensor"] = torch.tensor(spec.pop("nearest_feasible_flat"), dtype=torch.long, device=raw.device)
        outer, inner = index[spec.pop("outer_joint")], index[spec.pop("inner_joint")]
        pair = torch.stack((target[outer], target[inner])).unsqueeze(0)
        projected = project_leg_target_grid(pair, spec)[0]
        target[outer], target[inner] = projected[0], projected[1]
    return target


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
    cfg.action_contract = args_cli.action_contract
    cfg.apply_action_contract(rebuild_terrain=False)
    _configure_flat_probe(cfg)
    env = gym.make(args_cli.task, cfg=cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        joint_ids, _ = robot.find_joints(list(ACTIVE_JOINTS), preserve_order=True)
        initial_pos = _tensor(robot.data.joint_pos)[0, joint_ids].clone()
        initial_vel = _tensor(robot.data.joint_vel)[0, joint_ids].clone()
        default_pos = _tensor(robot.data.default_joint_pos)[0, joint_ids].clone()
        leg_term = unwrapped.action_manager.get_term("leg_position")
        servo = hasattr(leg_term, "joint_lower")
        if servo:
            leg_lower = leg_term.joint_lower[0].clone()
            leg_upper = leg_term.joint_upper[0].clone()
            if not torch.allclose(leg_term._offset[0], default_pos[:8], atol=1.0e-6):
                raise RuntimeError("Servo action offset differs from the articulation default stance.")
        traces = []
        contract_pass = True
        probes = [(index, sign) for index in range(POLICY_ACTION_DIM) for sign in (1.0, -1.0)]
        for action_index, sign in probes:
            env.reset()
            robot = unwrapped.scene["robot"]
            root_before = _tensor(robot.data.root_pos_w)[0].clone()
            action = torch.zeros((1, POLICY_ACTION_DIM), device=unwrapped.device)
            action[0, action_index] = sign * args_cli.action_magnitude
            for _ in range(args_cli.steps_per_probe):
                env.step(action)
            position_targets = _tensor(robot.data.joint_pos_target)[0, joint_ids]
            velocity_targets = _tensor(robot.data.joint_vel_target)[0, joint_ids]
            position = _tensor(robot.data.joint_pos)[0, joint_ids]
            velocity = _tensor(robot.data.joint_vel)[0, joint_ids]
            torque = _tensor(robot.data.applied_torque)[0, joint_ids]
            root = _tensor(robot.data.root_pos_w)[0]
            # Non-actuated joints keep their default targets (zero for E0-v1, the stance for E0-v2).
            expected_position = default_pos.clone()
            expected_position[8:] = 0.0
            expected_velocity = torch.zeros_like(velocity_targets)
            magnitude = sign * args_cli.action_magnitude
            if action_index < 8:
                if servo:
                    expected_position[:8] = _expected_servo_target(
                        action[0, :8], leg_lower, default_pos[:8], leg_upper
                    )
                else:
                    expected_position[action_index] = default_pos[action_index] + magnitude * LEG_POSITION_SCALE
            else:
                expected_velocity[action_index] = magnitude * WHEEL_VELOCITY_SCALE
            finite = bool(
                torch.isfinite(position).all() and torch.isfinite(velocity).all() and torch.isfinite(torque).all()
            )
            channel_pass = bool(
                torch.allclose(position_targets, expected_position, atol=1.0e-5, rtol=0.0)
                and torch.allclose(velocity_targets, expected_velocity, atol=1.0e-5, rtol=0.0)
                and finite
            )
            contract_pass &= channel_pass
            traces.append(
                {
                    "action_index": action_index,
                    "action_sign": sign,
                    "target_contract_pass": channel_pass,
                    "finite": finite,
                    "expected_joint_targets_position_rad": expected_position.tolist(),
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
            "action_contract": args_cli.action_contract,
            "default_joint_position_rad": default_pos.tolist(),
            "leg_lower_rad": leg_lower.tolist() if servo else None,
            "leg_upper_rad": leg_upper.tolist() if servo else None,
            "expected_leg_target_delta_rad": (
                "per-joint servo span" if servo else args_cli.action_magnitude * LEG_POSITION_SCALE
            ),
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
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
