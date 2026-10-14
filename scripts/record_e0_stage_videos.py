#!/usr/bin/env python3
"""Record high-fidelity visual replay videos for the four E0 curriculum stages.

Features:
- Long continuous per-terrain showcase (12s per terrain segment, 3 segments = 36s total).
- Anti-reflective, matte slate terrain visual material with balanced dome & directional sunlight.
- Comprehensive telemetry HUD displaying commanded vs. actual execution state, velocity tracking error,
  individual wheel contacts, and execution status.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

# Configure NVRTC and IsaacLab library paths before loading omniverse
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/home/kytolly/Library/IsaacLab"))
VENV_ROOT = Path(sys.prefix)
CUDA_LIB_DIRS = [
    path
    for path in (VENV_ROOT / "lib/python3.12/site-packages/nvidia").glob("*/lib")
    if path.is_dir()
]
for library_name in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
    for library_dir in CUDA_LIB_DIRS:
        library_path = library_dir / library_name
        if library_path.is_file():
            try:
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
if CUDA_LIB_DIRS:
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        dict.fromkeys([str(path) for path in CUDA_LIB_DIRS] + [path for path in existing_library_path if path])
    )
for source in (PROJECT_ROOT / "source/base_locomotion_stackforce_quadrupedal", *ISAACLAB_ROOT.glob("source/*")):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--stage",
    choices=("stage1_foundation", "stage2_expansion", "stage3_composition", "stage4_consolidation", "all"),
    default="all",
)
parser.add_argument(
    "--experiment",
    choices=("b0_t1_ppo", "b2_bisec_full"),
    default="b0_t1_ppo",
)
parser.add_argument("--output-base", type=Path)
parser.add_argument(
    "--steps",
    type=int,
    default=1800,
    help="Simulation steps per video (1800 steps = 36s at 50Hz, rendering 900 frames at 25fps)",
)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--camera-eye", nargs=3, type=float, default=(-1.15, -0.85, 0.70))
parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.10, 0.0, 0.0))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.output_base is None:
    args.output_base = PROJECT_ROOT / "output/videos" / args.experiment

# Recreating multiple Isaac Sim scenes in one process can leave stale render-delegate
# geometry state. Run each stage in a fresh process so every video gets a clean scene.
if args.stage == "all":
    stages = (
        "stage1_foundation",
        "stage2_expansion",
        "stage3_composition",
        "stage4_consolidation",
    )
    base_args = list(sys.argv[1:])
    if "--stage" in base_args:
        stage_index = base_args.index("--stage") + 1
        base_args[stage_index] = "{stage}"
    else:
        stage_index = None
        base_args.append("--stage")
        base_args.append("{stage}")
    for stage in stages:
        child_args = [stage if value == "{stage}" else value for value in base_args]
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *child_args], check=True)
    raise SystemExit(0)

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import trimesh
import isaaclab.terrains.utils as t_utils
import isaaclab.terrains.terrain_generator as tg_module
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.lights import DistantLightCfg, DomeLightCfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import math
from isaaclab.utils.math import euler_xyz_from_quat
from isaacsim.core.rendering_manager import ViewportManager
from rsl_rl.runners import OnPolicyRunner

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import (
    get_complex_terrain_cfg,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.support import (
    local_support_height,
    wheel_contact_state,
)
from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config


def tiered_height_color(mesh: trimesh.Trimesh, **kwargs) -> trimesh.Trimesh:
    """Color terrain mesh vertices by distinct height tiers for maximum structural clarity."""
    z = mesh.vertices[:, 2]
    colors = np.zeros((len(z), 4), dtype=np.uint8)

    # Base ground level: Deep Slate Charcoal (dark matte, zero specular glare)
    colors[:] = [42, 48, 56, 255]

    # Trenches / depressions / pits (z < -0.02m): Deep Midnight Navy
    colors[z < -0.02] = [28, 30, 42, 255]

    # Tier 1 (0.025m ~ 0.08m): Steel Cyan (low bumps, step 1, low stones)
    colors[(z >= 0.025) & (z < 0.08)] = [38, 145, 175, 255]

    # Tier 2 (0.08m ~ 0.16m): Emerald Green (step 2, medium boxes)
    colors[(z >= 0.08) & (z < 0.16)] = [46, 178, 120, 255]

    # Tier 3 (0.16m ~ 0.24m): Amber Gold (step 3, slope mid)
    colors[(z >= 0.16) & (z < 0.24)] = [230, 185, 45, 255]

    # Tier 4 (0.24m ~ 0.33m): Coral Flame Orange (step 4, high slope)
    colors[(z >= 0.24) & (z < 0.33)] = [238, 108, 52, 255]

    # Tier 5 (0.33m ~ 0.43m): Crimson Rose (step 5, high boxes)
    colors[(z >= 0.33) & (z < 0.43)] = [215, 52, 98, 255]

    # Tier 6 (>= 0.43m): Vivid Violet Orchid (summit, extreme obstacles)
    colors[z >= 0.43] = [165, 65, 215, 255]

    mesh.visual.vertex_colors = colors
    return mesh


def _custom_color_meshes_by_height(meshes, **kwargs):
    if isinstance(meshes, list):
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = meshes
    return tiered_height_color(mesh, **kwargs)


t_utils.color_meshes_by_height = _custom_color_meshes_by_height
tg_module.color_meshes_by_height = _custom_color_meshes_by_height

EXPERIMENT_CONFIGS = {
    "b0_t1_ppo": {
        "label": "B0 T1 PPO",
        "checkpoints": {
            "stage1_foundation": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-09_23-24-00_b0_t1_ppo_foundation_seed42/model_999.pt",
            "stage2_expansion": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-10_01-12-56_b0_t1_ppo_expansion_seed42/model_5999.pt",
            "stage3_composition": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-10_11-01-49_b0_t1_ppo_composition_seed42/model_12999.pt",
            "stage4_consolidation": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-11_01-49-36_b0_t1_ppo_consolidation_seed42/model_19999.pt",
        },
    },
    "b2_bisec_full": {
        "label": "B2 BISEC",
        "checkpoints": {
            "stage1_foundation": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-11_19-46-12_b2_bisec_full_foundation_seed42/model_999.pt",
            "stage2_expansion": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-11_21-41-51_b2_bisec_full_expansion_seed42/model_5999.pt",
            "stage3_composition": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-12_08-10-41_b2_bisec_full_composition_seed42/model_12999.pt",
            "stage4_consolidation": PROJECT_ROOT / "logs/rsl_rl/base_locomotion_e0_46d/2026-10-12_22-56-35_b2_bisec_full_consolidation_seed42/model_19999.pt",
        },
    },
}
EXPERIMENT_CONFIG = EXPERIMENT_CONFIGS[args.experiment]

STAGE_CONFIGS = {
    "stage1_foundation": {
        "title": "Stage 1: Foundation",
        "profile": "union_foundation",
        "checkpoint": EXPERIMENT_CONFIG["checkpoints"]["stage1_foundation"],
        "desc": "Flat and basic rough terrain, initial wheel-leg support and forward gait coordination",
        "segments": [
            {"terrain": "flat", "title": "Flat Terrain Forward Locomotion", "duration_ratio": 0.40},
            {"terrain": "random_rough", "title": "Random Rough Floor Traversal", "duration_ratio": 0.30},
            {"terrain": "wave", "title": "Gentle Wave Terrain Adaptation", "duration_ratio": 0.30},
        ],
    },
    "stage2_expansion": {
        "title": "Stage 2: Expansion",
        "profile": "union_expansion",
        "checkpoint": EXPERIMENT_CONFIG["checkpoints"]["stage2_expansion"],
        "desc": "Expanded terrain coverage: stairs, slopes, stepping stones, and rough terrain",
        "segments": [
            {"terrain": "pyramid_stairs", "title": "Pyramid Stairs Climbing", "duration_ratio": 0.34},
            {"terrain": "hf_pyramid_slope", "title": "Sloped Surface Traversal", "duration_ratio": 0.33},
            {"terrain": "stepping_stones", "title": "Stepping Stones Foothold Negotiation", "duration_ratio": 0.33},
        ],
    },
    "stage3_composition": {
        "title": "Stage 3: Composition",
        "profile": "union_composition",
        "checkpoint": EXPERIMENT_CONFIG["checkpoints"]["stage3_composition"],
        "desc": "Continuous mixed terrain challenge, dynamic speed tracking and posture recovery",
        "segments": [
            {"terrain": "continuous_mixed", "title": "Continuous Mixed Obstacle Traversal", "duration_ratio": 0.40},
            {"terrain": "pyramid_stairs", "title": "Stair Ascent & Descent Coordination", "duration_ratio": 0.30},
            {"terrain": "boxes", "title": "Discontinuous Box Obstacle Traversal", "duration_ratio": 0.30},
        ],
    },
    "stage4_consolidation": {
        "title": "Stage 4: Consolidation",
        "profile": "union_consolidation",
        "checkpoint": EXPERIMENT_CONFIG["checkpoints"]["stage4_consolidation"],
        "desc": "Final consolidated policy: full terrain suite mastery with high wheel contact & tracking",
        "segments": [
            {"terrain": "continuous_mixed", "title": "High-Agility Continuous Mixed Traversal", "duration_ratio": 0.40},
            {"terrain": "pyramid_stairs", "title": "Pyramid Stairs Robust Climbing", "duration_ratio": 0.30},
            {"terrain": "hf_pyramid_slope", "title": "Extreme Slopes Navigation", "duration_ratio": 0.30},
        ],
    },
}


def draw_hud(
    image: Image.Image,
    stage_info: dict,
    ckpt_name: str,
    seg_idx: int,
    total_segs: int,
    seg_title: str,
    terrain_name: str,
    env_id: int,
    sim_time: float,
    total_time: float,
    frame_idx: int,
    cmd_fwd: float,
    act_fwd: float,
    cmd_yaw: float,
    act_yaw: float,
    cmd_height: float,
    act_height: float,
    contacts: list[bool],
    fonts: dict,
) -> None:
    draw = ImageDraw.Draw(image)
    font_title = fonts["title"]
    font_sub = fonts["sub"]
    font_lbl = fonts["lbl"]
    font_val = fonts["val"]
    font_badge = fonts["badge"]

    # Top Status Bar
    draw.rectangle((0, 0, 1280, 96), fill=(12, 16, 22, 235))
    draw.line((0, 96, 1280, 96), fill=(0, 229, 255, 160), width=2)

    draw.text(
        (20, 7),
        f"{EXPERIMENT_CONFIG['label']} | {stage_info['title'].upper()} | Locomotion Replay",
        font=font_title,
        fill="#00E5FF",
    )
    draw.text(
        (20, 31),
        f"Model: {ckpt_name}  |  Terrain: {terrain_name} (Env #{env_id})  |  Sim: {sim_time:4.1f}s / {total_time:4.1f}s (Frame {frame_idx})",
        font=font_sub,
        fill="#CFD8DC",
    )

    # Telemetry Columns
    # Col 1: Forward Velocity
    draw.text((20, 56), "FWD VELOCITY:", font=font_lbl, fill="#B0BEC5")
    draw.text((20, 72), f"Cmd {cmd_fwd:+.2f} | Act {act_fwd:+.2f} m/s", font=font_val, fill="#76FF03")

    # Col 2: Yaw Rate
    draw.text((245, 56), "YAW RATE:", font=font_lbl, fill="#B0BEC5")
    draw.text((245, 72), f"Cmd {cmd_yaw:+.2f} | Act {act_yaw:+.2f} r/s", font=font_val, fill="#76FF03")

    # Col 3: Body Height
    draw.text((470, 56), "BODY HEIGHT:", font=font_lbl, fill="#B0BEC5")
    draw.text((470, 72), f"Cmd {cmd_height:.2f} | Act {act_height:.2f} m", font=font_val, fill="#76FF03")

    # Col 4: Wheel Contacts
    contact_dots = "".join(["●" if c else "○" for c in contacts])
    num_grounded = sum(contacts)
    draw.text((670, 56), "WHEEL CONTACT:", font=font_lbl, fill="#B0BEC5")
    draw.text((670, 72), f"FR FL RL RR [{contact_dots}] ({num_grounded}/4)", font=font_val, fill="#00E5FF")

    # Col 5: Forward Tracking Error
    fwd_err = abs(cmd_fwd - act_fwd)
    err_color = "#76FF03" if fwd_err < 0.15 else ("#FFD600" if fwd_err < 0.30 else "#FF5252")
    draw.text((915, 56), "FWD ERROR:", font=font_lbl, fill="#B0BEC5")
    draw.text((915, 72), f"{fwd_err:.2f} m/s", font=font_val, fill=err_color)

    # Col 6: Execution Badge
    badge_color = (20, 65, 30, 230) if fwd_err < 0.20 else (70, 55, 10, 230)
    badge_outline = (118, 255, 3, 220) if fwd_err < 0.20 else (255, 214, 0, 220)
    badge_text = "EXEC: TRACKING OK" if fwd_err < 0.20 else "EXEC: ADAPTING"
    badge_font_fill = "#76FF03" if fwd_err < 0.20 else "#FFD600"
    draw.rectangle((1045, 58, 1260, 88), fill=badge_color, outline=badge_outline)
    draw.text((1060, 65), badge_text, font=font_badge, fill=badge_font_fill)

    # Bottom Segment Title Banner
    draw.rectangle((20, 665, 560, 705), fill=(12, 16, 22, 215), outline=(0, 229, 255, 120))
    draw.text((32, 678), f"SEGMENT [{seg_idx + 1}/{total_segs}]: {seg_title}", font=font_badge, fill="#00E5FF")


def record_stage(stage_key: str, stage_info: dict) -> None:
    task = "Base-Locomotion-Stackforce-Quadrupedal-Complex-v0"
    out_dir = (args.output_base / stage_key).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{stage_key}_replay.mp4"
    json_path = out_dir / f"{stage_key}_replay.json"

    ckpt_path = stage_info["checkpoint"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for {stage_key}: {ckpt_path}")

    print(f"\n========================================================")
    print(f"[REPLAY VIDEO] Starting recording for {stage_info['title']}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Profile: {stage_info['profile']}")
    print(f"  Target Video: {video_path}")
    print(f"  Total Steps: {args.steps} (~{args.steps * 0.02:.1f}s simulation)")
    print(f"========================================================", flush=True)

    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.scene.terrain.terrain_generator = get_complex_terrain_cfg(
        seed=args.seed,
        num_rows=4,
        num_cols=16,
        profile=stage_info["profile"],
    )
    # Enable height tiered coloring and clear visual_material to show vertex displayColor
    cfg.scene.terrain.terrain_generator.color_scheme = "height"
    cfg.scene.terrain.visual_material = None

    # Configure studio lighting: balanced soft fill + distinct directional sun light to eliminate glare
    cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/ReplayDomeLight",
        spawn=DomeLightCfg(intensity=800.0, color=(0.90, 0.92, 0.96)),
    )
    cfg.scene.sun_light = AssetBaseCfg(
        prim_path="/World/ReplaySunLight",
        spawn=DistantLightCfg(intensity=700.0, color=(1.0, 0.98, 0.95), angle=0.53),
    )

    # Freeze level progression to evaluate standard terrain capabilities
    cfg.curriculum.terrain_levels = None
    cfg.events.reset_root.params["slope_approach_fraction"] = 0.0

    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-lib"))
    agent.device = args.device

    runner = OnPolicyRunner(
        vec,
        checkpoint_runner_config(ckpt_path, agent.to_dict()),
        log_dir=None,
        device=args.device,
    )
    runner.load(str(ckpt_path))
    policy = runner.get_inference_policy(device=args.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    t = base.scene.terrain
    # Set all environments to level 2 for pronounced stairs, slopes, and stepping obstacles
    t.terrain_levels[:] = 2
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]

    fps = round(1 / (2 * base.step_dt))  # 25 fps
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    fonts = {
        "title": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18),
        "sub": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13),
        "lbl": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12),
        "val": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13),
        "badge": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12),
    }

    # Map terrain names to column indices in the 16-column grid
    tg = cfg.scene.terrain.terrain_generator
    proportions = np.array([sub_cfg.proportion for sub_cfg in tg.sub_terrains.values()])
    proportions /= np.sum(proportions)
    cum_prop = np.cumsum(proportions)
    sub_keys = list(tg.sub_terrains.keys())
    col_terrain_map = [
        sub_keys[min(int(np.searchsorted(cum_prop, (c + 0.5) / 16.0)), len(sub_keys) - 1)]
        for c in range(16)
    ]
    # Build terrain to env index map using actual terrain_types of environments
    terrain_to_env: dict[str, int] = {}
    for env_i in range(args.num_envs):
        c = t.terrain_types[env_i].item()
        t_name = col_terrain_map[c]
        if t_name not in terrain_to_env:
            terrain_to_env[t_name] = env_i

    segments = stage_info["segments"]
    segment_plan = []
    accum_steps = 0
    for idx, seg in enumerate(segments):
        target_name = seg["terrain"]
        env_idx = terrain_to_env.get(target_name, 0)
        seg_steps = int(round(args.steps * seg["duration_ratio"]))
        if idx == len(segments) - 1:
            seg_steps = args.steps - accum_steps
        segment_plan.append({
            "seg_idx": idx,
            "title": seg["title"],
            "terrain_name": target_name,
            "env_id": env_idx,
            "start_step": accum_steps,
            "end_step": accum_steps + seg_steps,
            "duration_steps": seg_steps,
        })
        accum_steps += seg_steps

    print("  Curriculum Segment Plan:")
    for plan in segment_plan:
        print(
            f"    [{plan['seg_idx'] + 1}/{len(segments)}] {plan['title']} -> env #{plan['env_id']} "
            f"({plan['terrain_name']}) for steps {plan['start_step']}..{plan['end_step']} "
            f"({plan['duration_steps'] * base.step_dt:.1f}s)"
        )

    state = {
        "step": 0,
        "frames": 0,
        "current_segment": segment_plan[0],
    }

    def capture():
        if state["step"] % 2 != 0:
            return

        current_seg = state["current_segment"]
        selected_env = current_seg["env_id"]

        pos = robot.data.root_pos_w[selected_env].cpu().numpy()
        quat = robot.data.root_quat_w[selected_env].cpu() # [w, x, y, z]
        w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Body-frame chase offset: rear-left elevated perspective
        # dx_b < 0 (behind), dy_b < 0 (to the right of robot), dz > 0 (elevated)
        dx_b, dy_b, dz = args.camera_eye
        tx_b, ty_b, tz = args.camera_lookat

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Rotate horizontal offset from body heading to world frame
        eye_x = pos[0] + (dx_b * cos_y - dy_b * sin_y)
        eye_y = pos[1] + (dx_b * sin_y + dy_b * cos_y)
        eye_z = pos[2] + dz

        lookat_x = pos[0] + (tx_b * cos_y - ty_b * sin_y)
        lookat_y = pos[1] + (tx_b * sin_y + ty_b * cos_y)
        lookat_z = pos[2] + tz

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp", eye=[eye_x, eye_y, eye_z], target=[lookat_x, lookat_y, lookat_z]
        )
        if state["step"] == 0:
            print(f"DEBUG CAPTURE: selected_env={selected_env}, pos={pos}, eye={[eye_x, eye_y, eye_z]}, target={[lookat_x, lookat_y, lookat_z]}")

        frame = base.render()
        if frame is None:
            return
        image = Image.fromarray(frame)

        # Telemetry readouts
        cmd = base.command_manager.get_command("locomotion")[selected_env].tolist()
        root_lin_vel = robot.data.root_lin_vel_b[selected_env].tolist()
        root_ang_vel = robot.data.root_ang_vel_b[selected_env].tolist()

        supp_h, _ = local_support_height(base)
        act_height = (robot.data.root_pos_w[selected_env, 2] - supp_h[selected_env]).item()

        contacts = wheel_contact_state(base)[selected_env].tolist()

        sim_time = state["step"] * base.step_dt
        total_time = args.steps * base.step_dt

        draw_hud(
            image=image,
            stage_info=stage_info,
            ckpt_name=ckpt_path.name,
            seg_idx=current_seg["seg_idx"],
            total_segs=len(segments),
            seg_title=current_seg["title"],
            terrain_name=current_seg["terrain_name"],
            env_id=selected_env,
            sim_time=sim_time,
            total_time=total_time,
            frame_idx=state["frames"] + 1,
            cmd_fwd=cmd[0],
            act_fwd=root_lin_vel[0],
            cmd_yaw=cmd[2],
            act_yaw=root_ang_vel[2],
            cmd_height=cmd[3],
            act_height=act_height,
            contacts=contacts,
            fonts=fonts,
        )

        writer.append_data(np.asarray(image))
        state["frames"] += 1

    try:
        # Pre-align camera to initial target environment before reset
        first_env = segment_plan[0]["env_id"]
        init_pos = t.env_origins[first_env].cpu().numpy()
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[
                init_pos[0] + args.camera_eye[0],
                init_pos[1] + args.camera_eye[1],
                init_pos[2] + args.camera_eye[2],
            ],
            target=[
                init_pos[0] + args.camera_lookat[0],
                init_pos[1] + args.camera_lookat[1],
                init_pos[2] + args.camera_lookat[2],
            ],
        )
        vec.reset()
        for _ in range(12):
            base.render()
        base.post_physics_callbacks = [capture]

        # Execute continuous simulation steps
        for step_idx in range(args.steps):
            state["step"] = step_idx
            # Update active segment according to schedule
            for plan in segment_plan:
                if plan["start_step"] <= step_idx < plan["end_step"]:
                    state["current_segment"] = plan
                    break

            with torch.inference_mode():
                action = policy(vec.get_observations())
            _, _, dones, _ = vec.step(action)
            policy.reset(dones)

            if step_idx % 300 == 0 or step_idx == args.steps - 1:
                print(
                    f"  [{stage_key}] step {step_idx}/{args.steps} "
                    f"({step_idx * base.step_dt:.1f}s) | "
                    f"active segment: {state['current_segment']['terrain_name']} | "
                    f"frames: {state['frames']}",
                    flush=True,
                )

    finally:
        writer.close()
        base.post_physics_callbacks = []
        vec.close()

    json_metadata = {
        "experiment": args.experiment,
        "experiment_label": EXPERIMENT_CONFIG["label"],
        "stage": stage_key,
        "title": stage_info["title"],
        "description": stage_info["desc"],
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "terrain_profile": stage_info["profile"],
        "steps": args.steps,
        "frames": state["frames"],
        "fps": fps,
        "resolution": [1280, 720],
        "simulation_seconds": args.steps * float(cfg.sim.dt) * int(cfg.decimation),
        "video_file": str(video_path.name),
        "segments": segment_plan,
    }
    json_path.write_text(json.dumps(json_metadata, indent=2) + "\n")
    print(
        f"[REPLAY VIDEO] Successfully generated video for {stage_info['title']} -> {video_path} "
        f"({state['frames']} frames, {args.steps * base.step_dt:.1f}s)\n",
        flush=True,
    )


def main():
    stages_to_run = (
        list(STAGE_CONFIGS.keys())
        if args.stage == "all"
        else [args.stage]
    )
    for stage_key in stages_to_run:
        record_stage(stage_key, STAGE_CONFIGS[stage_key])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
