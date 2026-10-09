#!/usr/bin/env python3
"""Record the real firmware's open-loop dog gait on flat ground.

The phase schedule follows SF_serveo_control/src/main.cpp::trot(): one-second
period, 50% swing duty, and same-side pairs alternating (FL+RL, then FR+RR).
Wheel actions remain zero so the replay is a stepping-only mechanical reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--terrain-usd", type=Path, default=None, help="Optional USD obstacle terrain.")
parser.add_argument("--swing-amplitude", type=float, default=0.85, help="Normalized leg lift target.")
parser.add_argument("--fore-aft-amplitude", type=float, default=0.75, help="Normalized swing fore-aft target.")
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
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.terrains import TerrainImporterCfg
from isaacsim.core.rendering_manager import ViewportManager

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import FOOT_LINKS

TASK = "Base-Locomotion-Stackforce-Quadrupedal-Stepping-v0"
LEGS = ("FR", "FL", "RL", "RR")
DT = 0.02
PERIOD = 0.8
SWING_DUTY = 0.5


def firmware_action(phase: float, device: str) -> torch.Tensor:
    """Approximate the firmware Cartesian trot with the simulation's 8 joint targets.

    The real controller lifts a same-side pair with a half-cosine vertical
    trajectory and uses a cycloidal fore-aft trajectory.  Outer/inner targets
    here are bounded joint-space surrogates for that Cartesian path.
    """
    action = torch.zeros((1, 12), device=device)
    swing = (0, 2) if phase < SWING_DUTY else (1, 3)  # FR+RL, then FL+RR (diagonal trot)
    local = (phase % SWING_DUTY) / SWING_DUTY
    lift = 0.5 * (1.0 - np.cos(2.0 * np.pi * local))
    # Move the swing pair forward during lift and return it during descent.
    # The simulation's positive body-forward axis is opposite the firmware
    # Cartesian x sign for this asset, so negate the firmware fore-aft phase.
    fore_aft = -args.fore_aft_amplitude * (1.0 - 2.0 * local)
    for leg in swing:
        action[0, leg] = fore_aft
        action[0, leg + 4] = args.swing_amplitude * lift
    # The stance pair stays extended, matching the firmware's fixed height.
    for leg in range(4):
        if leg not in swing:
            action[0, leg + 4] = -0.32 if leg < 2 else 0.32
    return action


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration + 2.0
    if args.terrain_usd is not None:
        if not args.terrain_usd.is_file():
            raise FileNotFoundError(args.terrain_usd)
        cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="usd", usd_path=str(args.terrain_usd.resolve()),
            collision_group=-1, env_spacing=4.0, debug_vis=False)
        cfg.curriculum.terrain_levels = None
        cfg.events.reset_root.params["slope_approach_fraction"] = 0.0
        cfg.commands.locomotion.resampling_time_range = (1.0e9, 1.0e9)
        cfg.actions.leg_position.clip = {".*": (-1.0, 1.0)}
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array").unwrapped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None if args.no_video else imageio.get_writer(
        args.output, fps=25, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    trace_path = args.output.with_suffix(".jsonl")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    feet, _ = env.scene["robot"].find_bodies(list(FOOT_LINKS), preserve_order=True)
    robot = env.scene["robot"]
    initial_y = 0.0
    frames = 0
    step = 0
    final_base_y = None

    def capture() -> None:
        nonlocal frames
        if step % 2:
            return
        pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        forces = [float(env.scene[f"contact_{leg.lower()}"].data.net_forces_w[0].norm().item()) for leg in LEGS]
        row = {
            "step": step,
            "time_s": step * env.step_dt,
            "phase": (step * env.step_dt / PERIOD) % 1.0,
            "base_y": float(pos[1]),
            "base_forward_velocity": float(robot.data.root_lin_vel_b[0, 1]),
            "contact_force_n": forces,
            "contact": [x > 1.0 for x in forces],
            "wheel_action": [0.0] * 4,
            "gait": "FL+RL_then_FR+RR",
        }
        trace.write(json.dumps(row) + "\n")
        if writer is None:
            return
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=(0.95, -0.9, 0.58), target=(0.0, 0.35, 0.05))
        image = Image.fromarray(env.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 82), fill=(20, 24, 27))
        phase = row["phase"]
        swing = "FL+RL" if phase < 0.5 else "FR+RR"
        contact_text = " ".join(f"{n}:{'C' if c else 'A'}" for n, c in zip(LEGS, row["contact"]))
        draw.text((18, 8), "Firmware reference gait | wheel action = 0", font=font, fill="white")
        draw.text((18, 38), f"t={row['time_s']:.2f}s phase={phase:.2f} swing={swing} dy={pos[1]-initial_y:+.3f}m {contact_text}", font=font, fill="white")
        writer.append_data(np.asarray(image))
        frames += 1

    try:
        with trace_path.open("w", encoding="utf-8") as trace:
            env.reset(seed=args.seed)
            initial_y = float(robot.data.root_pos_w[0, 1])
            env.post_physics_callbacks = [capture]
            steps = round(args.duration / env.step_dt)
            for step in range(steps):
                phase = (step * env.step_dt / PERIOD) % 1.0
                env.step(firmware_action(phase, env.device))
            final_base_y = float(robot.data.root_pos_w[0, 1])
    finally:
        if writer is not None:
            writer.close()
        env.close()
    args.output.with_suffix(".json").write_text(json.dumps({
        "task": TASK, "mode": "firmware_reference_stepping_only", "duration_s": args.duration,
        "terrain_usd": str(args.terrain_usd.resolve()) if args.terrain_usd else None,
        "swing_amplitude": args.swing_amplitude, "fore_aft_amplitude": args.fore_aft_amplitude,
        "period_s": PERIOD, "swing_duty": SWING_DUTY, "phase_groups": "FR+RL_then_FL+RR",
        "wheel_action_forced_zero": True, "frames": frames, "trace": str(trace_path),
        "acceptance": "PASS_FORWARD_DISPLACEMENT" if final_base_y is not None and final_base_y - initial_y > 0.10 else "FAIL_FORWARD_DISPLACEMENT",
    }, indent=2) + "\n")
    print(f"[VIDEO] output={args.output} frames={frames}", flush=True)


try:
    main()
finally:
    simulation_app.close()
