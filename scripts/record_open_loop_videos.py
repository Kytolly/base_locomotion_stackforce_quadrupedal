#!/usr/bin/env python3
"""Record deterministic open-loop leg-motion replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("stretch_bend_forward", "stepping", "stair_tripod_recovery"), required=True)
parser.add_argument("--gait", choices=("diagonal", "lateral"), default="diagonal",
                    help="Two-support stepping gait: FR+RL/FL+RR or FL+RL/FR+RR.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--scan-report", type=Path, required=True)
parser.add_argument("--step-height", type=float, default=0.04)
parser.add_argument("--duration", type=float, default=16.0)
parser.add_argument("--speed", type=float, default=0.12)
parser.add_argument("--leg-limit", type=float, default=1.4)
parser.add_argument("--camera-eye", nargs=3, type=float, default=(0.95, -0.8, 0.60))
parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.12, 0.05))
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.rendering_manager import ViewportManager

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_stair_climb import Probe, TASK, write_terrain  # noqa: E402

LEGS = ("FR", "FL", "RL", "RR")


def make_action(torch_module, device, leg_limit):
    return torch_module.zeros((1, 12), device=device)


def main() -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    terrain = output.with_suffix(".terrain.usda")
    probe_args = argparse.Namespace(
        step_height=args.step_height, step_count=1, terrain="stairs", tread=0.35, approach=0.45,
        speed=args.speed, advance_time=8.0 if args.mode == "stair_tripod_recovery" else 1.5,
        recovery_time=4.0 if args.mode == "stair_tripod_recovery" else 2.0,
        duration=args.duration, leg_limit=args.leg_limit,
        support_search=False, lift_report=None, support_pattern="symmetric", fixed_base=False,
        seed=1001, device=args.device, scan_report=None, skip_scan=True, grid=3, checkpoint=None,
    )
    write_terrain(terrain, probe_args)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = 1001
    cfg.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd",
                                           usd_path=str(terrain), collision_group=-1, env_spacing=4.0, debug_vis=False)
    cfg.curriculum.terrain_levels = None
    cfg.episode_length_s = args.duration + 5
    for name in ("randomize_contact_material", "randomize_base_mass", "randomize_actuator_gains"):
        setattr(cfg.events, name, None)
    cfg.events.reset_root.params["slope_approach_fraction"] = 0.0
    cfg.commands.locomotion.resampling_time_range = (1e9, 1e9)
    cfg.observations.policy.enable_corruption = False
    cfg.actions.leg_position.clip = {".*": (-args.leg_limit, args.leg_limit)}
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array").unwrapped
    writer = imageio.get_writer(output, fps=25, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    trace_path = output.with_suffix(".jsonl")
    frames = 0
    probe = None
    step = 0
    initial_y = 0.0

    def capture():
        nonlocal frames
        if probe is None or step % 2:
            return
        pos = probe.robot.data.root_pos_w[0].detach().cpu().numpy()
        if args.mode in ("stepping", "stair_tripod_recovery"):
            eye = np.asarray((0.95, -0.95, 0.62)).tolist()
            target = np.asarray((0.0, 0.42, 0.05)).tolist()
        else:
            eye = (pos + np.asarray(args.camera_eye)).tolist()
            target = (pos + np.asarray(args.camera_lookat)).tolist()
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=target)
        frame = env.render()
        if frame is None:
            raise RuntimeError("RGB renderer returned no frame")
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 72), fill=(20, 24, 27))
        if args.mode == "stretch_bend_forward":
            label = "open-loop: four-leg stretch/bend + forward"
        elif args.mode == "stepping":
            label = f"open-loop: {args.gait} two-support stepping"
        else:
            label = "open-loop: tripod stair recovery"
        draw.text((18, 8), label, font=font, fill="white")
        draw.text((18, 38), f"t={probe.clock * env.step_dt:.2f}s  dy={pos[1] - initial_y:+.3f}m  tilt={probe.state['tilt_rad']:.3f}rad", font=font, fill="white")
        writer.append_data(np.asarray(image))
        frames += 1

    try:
        with trace_path.open("w", encoding="utf-8") as trace:
            probe = Probe(env, probe_args, trace)
            probe.reset(args.mode)
            initial_y = float(probe.state["root_xyz"][1])
            env.post_physics_callbacks = [probe.capture, capture]
            action = make_action(torch, env.device, args.leg_limit)
            command = torch.tensor([[args.speed, 0, 0, 0.105]], device=env.device)
            if args.mode == "stretch_bend_forward":
                # Hold a global crouch, roll forward, then extend all four legs.
                bend = torch.tensor([[-2.8, -2.8, 2.8, 2.8, 2.8, -2.8, -2.8, 2.8, 0, 0, 0, 0]], device=env.device)
                extend = torch.zeros_like(bend)
                for phase, target, seconds, wheels in (("bend_all", bend, 1.0, False),
                                                        ("roll_bent", bend, 4.0, True),
                                                        ("extend_all", extend, 1.2, False),
                                                        ("roll_extended", extend, 5.0, True),
                                                        ("bend_again", bend, 1.0, False),
                                                        ("roll_again", bend, 2.0, True)):
                    target = target.clone()
                    if wheels:
                        target[0, 8:] = args.speed / (0.033 * 20.0)
                    count = round(seconds / env.step_dt)
                    start = action.clone()
                    for index in range(count):
                        alpha = (index + 1) / count
                        action = start + alpha * (target - start)
                        probe.step(action, phase)
                        step += 1
                        if probe.done:
                            break
                    if probe.done:
                        break
            elif args.mode == "stair_tripod_recovery":
                selected = json.loads(args.scan_report.read_text())["clearance_scan"]
                probe.five_phase_obstacle(selected)
            else:
                selected = json.loads(args.scan_report.read_text())["clearance_scan"]
                # Wheel commands stay zero throughout this replay. Two legs
                # support while the other two bend and swing forward.
                action.zero_()
                groups = ((0, 2), (1, 3)) if args.gait == "diagonal" else ((1, 2), (0, 3))
                for cycle, swing_legs in enumerate(groups * 2):
                    support_legs = tuple(i for i in range(4) if i not in swing_legs)
                    target = action.clone()
                    for support in support_legs:
                        target[0, support + 4] = -args.leg_limit / 0.5 if support < 2 else args.leg_limit / 0.5
                    for leg in swing_legs:
                        target[0, leg], target[0, leg + 4] = selected[LEGS[leg]]["best"]["action_pair"]
                    for index in range(round(0.8 / env.step_dt)):
                        action = action + (target - action) / max(1, round(0.8 / env.step_dt) - index)
                        probe.step(action, f"gait_{args.gait}_lift_{''.join(LEGS[i] for i in swing_legs)}")
                        step += 1
                    swing = target.clone()
                    for leg in swing_legs:
                        swing[0, leg] = 0.0
                        swing[0, leg + 4] = args.leg_limit / 0.5
                    for index in range(round(0.8 / env.step_dt)):
                        action = action + (swing - action) / max(1, round(0.8 / env.step_dt) - index)
                        probe.step(action, f"gait_{args.gait}_swing_{''.join(LEGS[i] for i in swing_legs)}")
                        step += 1
                    for index in range(round(0.8 / env.step_dt)):
                        action = action + (target - action) / max(1, round(0.8 / env.step_dt) - index)
                        probe.step(action, f"gait_{args.gait}_place_{''.join(LEGS[i] for i in swing_legs)}")
                        step += 1
            while not probe.done and step < round(args.duration / env.step_dt):
                step += 1
                probe.step(action, "hold")
    finally:
        writer.close()
        env.close()
    output.with_suffix(".json").write_text(json.dumps({"mode": args.mode, "output": str(output),
        "frames": frames, "fps": 25, "trace": str(trace_path), "terrain": str(terrain),
        "checkpoint": None, "open_loop": True}, indent=2) + "\n")
    print(f"[VIDEO] mode={args.mode} output={output} frames={frames}", flush=True)


try:
    main()
finally:
    simulation_app.close()
