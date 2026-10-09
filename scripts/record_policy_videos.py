#!/usr/bin/env python3
"""Record a frozen checkpoint on training, Plateau, or Washboard terrain."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--kind", choices=("training", "plateau", "washboard"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1750)
parser.add_argument("--seed", type=int, default=8101)
parser.add_argument("--camera-eye", nargs=3, type=float, default=(0.95, -0.8, 0.60))
parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.12, 0.05))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not args.checkpoint.is_file():
    parser.error(f"checkpoint not found: {args.checkpoint}")
if args.steps <= 0:
    parser.error("--steps must be positive")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config


TASKS = {
    "training": "Base-Locomotion-Stackforce-Quadrupedal-Complex-v0",
    "plateau": "Base-Locomotion-Stackforce-Quadrupedal-Plateau-v0",
    "washboard": "Base-Locomotion-Stackforce-Quadrupedal-Washboard-v0",
}


def main() -> None:
    task = TASKS[args.kind]
    cfg = parse_env_cfg(task, device=args.device, num_envs=8 if args.kind == "training" else 1)
    cfg.seed = args.seed
    cfg.viewer.eye = tuple(args.camera_eye)
    cfg.viewer.lookat = tuple(args.camera_lookat)
    if args.kind != "training":
        cfg.events.spawn_track.params["seed"] = args.seed
        cfg.events.spawn_track.params["kind"] = args.kind
        cfg.events.spawn_track.params["randomized"] = False
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    video_dir = args.output.resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-lib"))
    agent.device = args.device
    runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent.to_dict()), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=args.device)
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from isaacsim.core.rendering_manager import ViewportManager
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import TERRAIN_FAMILIES

    base = env.unwrapped
    robot = base.scene["robot"]
    fps = round(1 / (2 * base.step_dt))
    path = video_dir / f"{args.kind}_20k.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    frames = 0
    selected = 0
    step = 0

    def capture():
        nonlocal frames
        if step % 2:
            return
        pos = robot.data.root_pos_w[selected].cpu().numpy()
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=(pos + np.array(args.camera_eye)).tolist(),
                                        target=(pos + np.array(args.camera_lookat)).tolist())
        frame = base.render()
        if frame is None:
            raise RuntimeError("RGB renderer returned no frame")
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        command = base.command_manager.get_command("locomotion")[selected].tolist()
        label = args.kind
        if args.kind == "training":
            label += " / " + TERRAIN_FAMILIES[int(base.scene.terrain.terrain_types[selected])]
        draw.rectangle((0, 0, image.width, 68), fill=(20, 24, 27))
        draw.text((18, 8), f"20k policy | {label} | t={step * base.step_dt:.2f}s", font=font, fill="white")
        draw.text((18, 36), f"command: forward {command[0]:+.2f} m/s   lateral {command[1]:+.2f} m/s   yaw {command[2]:+.2f} rad/s", font=font, fill="white")
        writer.append_data(np.asarray(image))
        frames += 1

    try:
        vec.reset()
        for _ in range(12):
            base.render()
        base.post_physics_callbacks = [capture]
        for step in range(args.steps):
            selected = min(7, step * 8 // args.steps) if args.kind == "training" else 0
            if args.kind != "training":
                base.command_manager.get_term("locomotion").set_command(torch.tensor([[0.28, 0, 0, 0.105]], device=base.device))
            with torch.inference_mode():
                action = policy(vec.get_observations())
            _, _, dones, _ = vec.step(action)
            policy.reset(dones)
            if step % 250 == 0:
                print(f"[VIDEO] {args.kind} step={step}/{args.steps} frames={frames}", flush=True)
            if args.kind != "training" and bool(dones[0]):
                break
    finally:
        writer.close()
        base.post_physics_callbacks = []
        vec.close()
    (video_dir / f"{args.kind}_20k.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "kind": args.kind, "seed": args.seed, "steps": step + 1, "frames": frames, "fps": fps,
        "replay": "new frozen-checkpoint rollout, not historical training capture",
        "resolution": [1280, 720], "simulation_seconds": (step + 1) * float(cfg.sim.dt) * int(cfg.decimation),
        "camera": "world-oriented follow view", "training_environments": 8 if args.kind == "training" else 1,
    }, indent=2) + "\n")
    print(f"[VIDEO] kind={args.kind} output={video_dir}")


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
