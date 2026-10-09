#!/usr/bin/env python3
"""Record the flat-ground stepping-only capability experiment."""

from __future__ import annotations

import argparse
import json
import importlib.metadata as metadata
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--no-video", action="store_true", help="Run the replay and write telemetry without rendering frames.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaacsim.core.rendering_manager import ViewportManager
from rsl_rl.runners import OnPolicyRunner

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import FOOT_LINKS

TASK = "Base-Locomotion-Stackforce-Quadrupedal-Stepping-v0"
LEGS = ("FR", "FL", "RL", "RR")


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = args.steps * float(cfg.sim.dt) * int(cfg.decimation)
    env = gym.make(TASK, cfg=cfg, **({"render_mode": "rgb_array"} if not args.no_video else {}))
    print("[STEPPING-REPLAY] environment created", flush=True)
    base = env.unwrapped
    vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent.to_dict()), log_dir=None, device=args.device)
    print("[STEPPING-REPLAY] loading checkpoint", flush=True)
    runner.load(str(args.checkpoint))
    print("[STEPPING-REPLAY] checkpoint loaded", flush=True)
    policy = runner.get_inference_policy(device=args.device)
    robot = base.scene["robot"]
    feet, _ = robot.find_bodies(list(FOOT_LINKS), preserve_order=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None if args.no_video else imageio.get_writer(
        args.output, fps=25, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    trace = args.output.with_suffix(".jsonl").open("w", encoding="utf-8")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    frame_count = 0
    step = 0
    initial_y = 0.0

    def capture():
        nonlocal frame_count
        if step % 2:
            return
        pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        forces = [float(base.scene[f"contact_{leg.lower()}"].data.net_forces_w[0].norm().item()) for leg in LEGS]
        feet_y = [float(robot.data.body_pos_w[0, i, 1] - pos[1]) for i in feet]
        row = {"step": step, "time_s": step * base.step_dt, "base_y": float(pos[1]),
               "base_forward_velocity": float(robot.data.root_lin_vel_b[0, 1]),
               "contact_force_n": forces, "contact": [x > 1.0 for x in forces],
               "feet_body_y": feet_y, "wheel_action": [0.0] * 4}
        trace.write(json.dumps(row) + "\n")
        if writer is None:
            return
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=(0.95, -0.9, 0.58), target=(0.0, 0.35, 0.05))
        image = Image.fromarray(base.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 82), fill=(20, 24, 27))
        draw.text((18, 8), "Stepping-only capability | wheel action = 0", font=font, fill="white")
        contact_text = " ".join(f"{n}:{'C' if c else 'A'}" for n, c in zip(LEGS, row["contact"]))
        draw.text((18, 38), f"t={row['time_s']:.2f}s  dy={pos[1]-initial_y:+.3f}m  v={row['base_forward_velocity']:+.3f}m/s  {contact_text}", font=font, fill="white")
        writer.append_data(np.asarray(image))
        frame_count += 1

    try:
        print("[STEPPING-REPLAY] resetting", flush=True)
        base.reset(seed=args.seed)
        policy.reset(torch.ones(1, dtype=torch.long, device=base.device))
        print("[STEPPING-REPLAY] rollout started", flush=True)
        initial_y = float(robot.data.root_pos_w[0, 1])
        base.post_physics_callbacks = [capture]
        for step in range(args.steps):
            command = torch.tensor([[0.20, 0.0, 0.0, 0.105]], device=base.device)
            base.command_manager.get_term("locomotion").set_command(command)
            with torch.inference_mode():
                inferred = policy(vec.get_observations()).clamp(-1, 1)
            action = torch.empty_like(inferred)
            action.copy_(inferred)
            action[:, 8:] = 0.0
            _, _, dones, _ = vec.step(action)
            if step == 0:
                print("[STEPPING-REPLAY] first step complete", flush=True)
            policy.reset(dones)
            if bool(dones[0]):
                break
        print(f"[STEPPING-REPLAY] rollout complete steps={step + 1}", flush=True)
    finally:
        trace.close()
        if writer is not None:
            writer.close()
        base.post_physics_callbacks = []
        vec.close()
    args.output.with_suffix(".json").write_text(json.dumps({
        "task": TASK, "mode": "stepping_only", "checkpoint": str(args.checkpoint.resolve()),
        "steps": step + 1, "frames": frame_count, "fps": 25,
        "trace": str(args.output.with_suffix('.jsonl')), "wheel_action_forced_zero": True,
    }, indent=2) + "\n")
    print(f"[VIDEO] output={args.output} frames={frame_count}", flush=True)


try:
    main()
finally:
    simulation_app.close()
