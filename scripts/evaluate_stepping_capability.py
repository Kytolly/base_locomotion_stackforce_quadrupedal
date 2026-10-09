#!/usr/bin/env python3
"""Evaluate wheel-only, stepping-only, and hybrid locomotion on flat ground."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

TASK = "Base-Locomotion-Stackforce-Quadrupedal-Stepping-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", choices=("all", "wheel_only", "stepping_only", "hybrid"), default="all")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--command", type=float, default=0.20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import WHEEL_JOINTS

LEGS = ("FR", "FL", "RL", "RR")


def run_mode(mode: str, task_env, vec, policy, steps: int, command_speed: float) -> dict:
    env = task_env.unwrapped
    robot = env.scene["robot"]
    wheel_ids, _ = robot.find_joints(list(WHEEL_JOINTS), preserve_order=True)
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import FOOT_LINKS
    feet, _ = robot.find_bodies(list(FOOT_LINKS), preserve_order=True)
    env.reset(seed=args.seed)
    policy.reset(torch.ones(1, dtype=torch.long, device=env.device))
    command = torch.tensor([[command_speed, 0.0, 0.0, 0.105]], device=env.device)
    env.command_manager.get_term("locomotion").set_command(command)
    contacts = [[] for _ in LEGS]
    rel_y = [[] for _ in LEGS]
    base_y, base_v = [], []
    wheel_vel, wheel_torque = [], []
    joint_pos, joint_vel = [], []
    actions = []

    for _ in range(steps):
        env.command_manager.get_term("locomotion").set_command(command)
        with torch.inference_mode():
            inferred = policy(vec.get_observations()).clamp(-1, 1)
        action = torch.empty_like(inferred, memory_format=torch.preserve_format)
        action.copy_(inferred)
        if mode == "wheel_only":
            action[:, :8] = 0.0
        elif mode == "stepping_only":
            action[:, 8:] = 0.0
        actions.append(action[0].detach().cpu().tolist())
        vec.step(action)
        data = robot.data
        forces = [float(env.scene[f"contact_{leg.lower()}"].data.net_forces_w[0].norm().item()) for leg in LEGS]
        for i, force in enumerate(forces):
            contacts[i].append(force > 1.0)
            rel_y[i].append(float(data.body_pos_w[0, feet[i], 1] - data.root_pos_w[0, 1]))
        base_y.append(float(data.root_pos_w[0, 1]))
        base_v.append(float(data.root_lin_vel_b[0, 1]))
        wheel_vel.append(data.joint_vel[0, wheel_ids].detach().cpu().tolist())
        wheel_torque.append(data.applied_torque[0, wheel_ids].detach().cpu().tolist())
        joint_pos.append(data.joint_pos[0, :8].detach().cpu().tolist())
        joint_vel.append(data.joint_vel[0, :8].detach().cpu().tolist())
        if bool(env.reset_terminated[0] or env.reset_time_outs[0]):
            break

    cycles = {}
    for i, leg in enumerate(LEGS):
        state = contacts[i]
        events = []
        for j in range(1, len(state)):
            if state[j - 1] and not state[j]:
                start = rel_y[i][j - 1]
                touchdown = next((k for k in range(j + 1, len(state)) if not state[k]), None)
                if touchdown is not None:
                    peak = max(rel_y[i][j:touchdown + 1])
                    end = rel_y[i][touchdown]
                    events.append({"lift_off": j, "touchdown": touchdown,
                                   "peak_forward_swing_m": peak - start,
                                   "contact_relocation_m": end - start,
                                   "valid": peak - start > 0.01 and end - start > 0.005})
        cycles[leg] = {"count": len(events), "valid_count": sum(e["valid"] for e in events), "events": events}
    valid_cycles = sum(v["valid_count"] for v in cycles.values())
    report = {
        "mode": mode, "steps": len(base_y), "command_forward_mps": command_speed,
        "active_wheel_action_max": max((max(abs(a) for a in x[8:]) for x in actions), default=0.0),
        "base_forward_displacement_m": base_y[-1] - base_y[0] if base_y else 0.0,
        "base_forward_velocity_mean_mps": sum(base_v) / max(1, len(base_v)),
        "wheel_velocity_abs_mean_radps": sum(sum(abs(x) for x in row) for row in wheel_vel) / max(1, 4 * len(wheel_vel)),
        "wheel_torque_abs_mean_nm": sum(sum(abs(x) for x in row) for row in wheel_torque) / max(1, 4 * len(wheel_torque)),
        "cycles": cycles, "valid_stepping_cycles": valid_cycles,
        "stepping_capability": "PASS" if mode == "stepping_only" and valid_cycles >= 2 and (base_y[-1] - base_y[0]) > 0.10 else "FAIL",
        "trace": {"contact": contacts, "foot_body_y": rel_y, "base_y": base_y, "base_v": base_v,
                  "wheel_velocity": wheel_vel, "wheel_torque": wheel_torque,
                  "leg_joint_position": joint_pos, "leg_joint_velocity": joint_vel, "actions": actions},
    }
    return report


def main() -> None:
    print("[STEPPING] creating flat environment", flush=True)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = args.steps * float(cfg.sim.dt) * int(cfg.decimation)
    cfg.commands.locomotion.resampling_time_range = (1e9, 1e9)
    cfg.curriculum.terrain_levels = None
    base = gym.make(TASK, cfg=cfg)
    vec = RslRlVecEnvWrapper(base, clip_actions=1.0)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent_cfg.to_dict()), log_dir=None, device=args.device)
    print("[STEPPING] loading checkpoint", flush=True)
    runner.load(str(args.checkpoint))
    print("[STEPPING] checkpoint loaded", flush=True)
    policy = runner.get_inference_policy(device=args.device)
    modes = ("wheel_only", "stepping_only", "hybrid") if args.mode == "all" else (args.mode,)
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "task": TASK, "flat_terrain": True, "active_wheel_control_disabled_in_stepping_only": True,
              "modes": {mode: run_mode(mode, base, vec, policy, args.steps, args.command) for mode in modes}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {m: v[m] for m in ("base_forward_displacement_m", "valid_stepping_cycles", "stepping_capability", "active_wheel_action_max")}
                      for k, v in report["modes"].items()}, indent=2))
    vec.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
