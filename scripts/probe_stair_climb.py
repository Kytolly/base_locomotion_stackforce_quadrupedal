#!/usr/bin/env python3
"""Measure free-base leg clearance and compare scripted/rolling/frozen-policy stair trials.

A failed search is inconclusive about mechanical impossibility. All positive
claims refer to the loaded simulation asset and its configured actuators.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import itertools
import json
import math
from pathlib import Path


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/probes/stair_climb/report.json"))
    parser.add_argument("--step-height", type=float, default=0.04)
    parser.add_argument("--step-count", type=int, default=1)
    parser.add_argument("--terrain", choices=("stairs", "pyramid"), default="pyramid")
    parser.add_argument("--tread", type=float, default=0.35)
    parser.add_argument("--approach", type=float, default=0.45)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--advance-time", type=float, default=2.5,
                        help="Maximum seconds spent advancing each lifted leg before placing it.")
    parser.add_argument("--recovery-time", type=float, default=2.0,
                        help="Seconds of wheel-driven posture recovery after the front pair reaches the step.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--leg-limit", type=float, default=0.5,
                        help="Scripted target limit in radians; policy always retains +/-0.5 rad.")
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--fixed-base", action="store_true", help="Suspended kinematic diagnostic, never traversal evidence.")
    parser.add_argument("--support-search", action="store_true", help="Search symmetric support poses before lifting each leg.")
    parser.add_argument("--lift-report", type=Path, help="Search support poses for lift pairs measured by a fixed-base scan.")
    parser.add_argument("--support-pattern", choices=("symmetric", "translate"), default="symmetric")
    parser.add_argument("--skip-scan", action="store_true", help="Run rolling/policy trials only.")
    parser.add_argument("--scan-report", type=Path, help="Reuse a scan from this probe, with matching action limit.")
    parser.add_argument("--five-phase", action="store_true", help="Run the staged five-phase obstacle maneuver.")
    parser.add_argument("--seed", type=int, default=1001)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for name in ("step_height", "tread", "approach", "speed", "advance_time", "recovery_time", "duration", "leg_limit"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.step_count < 1 or args.grid < 3 or args.grid % 2 != 1:
        parser.error("--step-count must be positive; --grid must be odd and at least 3")
    if args.leg_limit > 1.4:
        parser.error("--leg-limit must be <= 1.4 rad (inside the asset's +/-1.57 rad limits)")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.fixed_base and args.skip_scan:
        parser.error("--fixed-base requires a scan")
    return args


TASK = "Base-Locomotion-Stackforce-Quadrupedal-Complex-v0"
LEGS = ("FR", "FL", "RL", "RR")


def write_terrain(path, args):
    """Use one collision mesh so the existing support scanner sees the stairs."""
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    pieces = [trimesh.creation.box((12, 12, 0.1), transform=trimesh.transformations.translation_matrix((0, 3, -0.05)))]
    if args.terrain == "pyramid":
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.terrains import get_complex_terrain_cfg

        reference = get_complex_terrain_cfg().sub_terrains["pyramid_stairs"].copy()
        # The generator adds a central block after the rings. A tiny size
        # offset makes floor division produce exactly step_count - 1 rings.
        size = 2 * reference.border_width + reference.platform_width + 2 * (args.step_count - 1) * args.tread - 1e-6
        reference.size = (size, size)
        reference.step_width = args.tread
        reference.step_height_range = (args.step_height, args.step_height)
        generated, _ = reference.function(0.0, reference)
        for mesh in generated:
            mesh.apply_translation((-size / 2, args.approach - reference.border_width, 0))
        pieces.extend(generated)
    for index in range(args.step_count if args.terrain == "stairs" else 0):
        start = args.approach + index * args.tread
        end = args.approach + (args.step_count - 1) * args.tread + 2.5
        height = (index + 1) * args.step_height
        pieces.append(trimesh.creation.box(
            (2.5 + 2 * (args.step_count - 1 - index) * args.tread, end - start, args.step_height),
            transform=trimesh.transformations.translation_matrix((0, (start + end) / 2, height - args.step_height / 2)),
        ))
    combined = trimesh.util.concatenate(pieces)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Terrain")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Terrain/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in combined.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(combined.faces))
    mesh.CreateFaceVertexIndicesAttr(combined.faces.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, "/Terrain/Material")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.0)
    physics.CreateDynamicFrictionAttr(0.9)
    physics.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


class Probe:
    def __init__(self, env, args, trace):
        import torch
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots import ACTIVE_JOINTS, FOOT_LINKS
        from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.support import WHEEL_RADIUS_M

        self.torch, self.env, self.args, self.trace = torch, env, args, trace
        self.robot = env.scene["robot"]
        self.joints, _ = self.robot.find_joints(list(ACTIVE_JOINTS), preserve_order=True)
        self.feet, _ = self.robot.find_bodies(list(FOOT_LINKS), preserve_order=True)
        self.radius = WHEEL_RADIUS_M
        self.action = torch.zeros((1, 12), device=env.device)
        self.state = None
        self.samples = []
        self.trial = ""
        self.phase = ""
        self.done = False
        self.clock = 0
        env.post_physics_callbacks = [self.capture]

    def capture(self):
        from isaaclab.utils.math import quat_apply_inverse

        data = self.robot.data
        feet = data.body_pos_w[:, self.feet]
        relative = feet - data.root_pos_w[:, None, :]
        relative = quat_apply_inverse(data.root_quat_w[:, None, :].expand(-1, 4, -1), relative)
        contacts = [float(self.env.scene[f"contact_{leg.lower()}"].data.net_forces_w[0].norm().item()) for leg in LEGS]
        gravity = data.projected_gravity_b[0]
        tilt = math.acos(max(-1, min(1, -float(gravity[2]))))
        terminations = [name for name in self.env.termination_manager.active_terms
                        if bool(self.env.termination_manager.get_term(name)[0])]
        row = {
            "trial": self.trial, "phase": self.phase, "time_s": self.clock * self.env.step_dt,
            "root_xyz": data.root_pos_w[0].tolist(), "tilt_rad": tilt,
            "feet_xyz": feet[0].tolist(), "feet_body_xyz": relative[0].tolist(),
            "contact_force_n": contacts, "joint_position_rad": data.joint_pos[0, self.joints].tolist(),
            "joint_target_rad": data.joint_pos_target[0, self.joints[:8]].tolist(),
            "torque_nm": data.applied_torque[0, self.joints].tolist(),
            "velocity_body": data.root_lin_vel_b[0].tolist(),
            "yaw_rate": float(data.root_ang_vel_b[0, 2]), "action": self.action[0].tolist(),
            "terminations": terminations,
        }
        self.state = row
        self.samples.append(row)
        self.trace.write(json.dumps(row, allow_nan=False) + "\n")
        self.clock += 1

    def step(self, action, phase):
        if self.done:
            return False
        self.phase = phase
        self.action = action.clone()
        command = self.torch.tensor([[self.args.speed, 0, 0, 0.105]], device=self.env.device)
        self.env.command_manager.get_term("locomotion").set_command(command)
        _, _, terminated, truncated, _ = self.env.step(action)
        self.done = bool((terminated | truncated)[0])
        return not self.done

    def reset(self, trial):
        self.trial, self.samples, self.clock, self.done = trial, [], 0, False
        self.env.reset(seed=self.args.seed)
        self.action.zero_()
        self.state = None
        for _ in range(round(0.5 / self.env.step_dt)):
            if not self.step(self.action, "settle"):
                break
        return not self.done

    def ramp(self, target, phase, seconds=0.5):
        start = self.action.clone()
        count = max(1, round(seconds / self.env.step_dt))
        for index in range(count):
            alpha = (index + 1) / count
            alpha = alpha * alpha * (3 - 2 * alpha)
            if not self.step(start + alpha * (target - start), phase):
                return False
        return True

    def scan(self):
        candidates = {}
        limit = self.args.leg_limit / 0.5
        values = self.torch.linspace(-limit, limit, self.args.grid).tolist()
        lifts = json.loads(self.args.lift_report.read_text())["clearance_scan"] if self.args.lift_report else None
        for leg, name in enumerate(LEGS):
            trials = []
            supports = [(0.0, 0.0)]
            if self.args.support_search:
                supports += list(itertools.product((-0.5, 0, 0.5), repeat=2))
                supports = list(dict.fromkeys(supports))
            pairs = [lifts[name]["best"]["action_pair"]] if lifts else list(itertools.product(values, repeat=2))
            for (outer, inner), support in itertools.product(pairs, supports):
                if outer == inner == 0:
                    continue
                self.reset(f"scan_{name}_{outer:.3f}_{inner:.3f}_{support}")
                baseline = self.state
                action = self.torch.zeros_like(self.action)
                signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
                if self.args.support_pattern == "translate":
                    signs = ((1, 1), (1, -1), (1, -1), (1, 1))
                for other in range(4):
                    if other != leg:
                        action[0, other] = support[0] * signs[other][0]
                        action[0, other + 4] = support[1] * signs[other][1]
                if support != (0.0, 0.0):
                    self.ramp(action, "shift_support")
                action[0, leg], action[0, leg + 4] = outer, inner
                self.ramp(action, "lift")
                for _ in range(math.ceil(0.25 / self.env.step_dt)):
                    if not self.step(action, "hold"):
                        break
                held = [s for s in self.samples if s["phase"] == "hold"]
                valid = bool(held) and not self.done and all(
                    s["tilt_rad"] < 0.35 and s["contact_force_n"][leg] < 1.0
                    and all(s["contact_force_n"][other] > 1.0 for other in range(4) if other != leg)
                    for s in held
                )
                clearance = min((s["feet_xyz"][leg][2] - self.radius for s in held), default=0)
                relative_lift = min((s["feet_body_xyz"][leg][2] - baseline["feet_body_xyz"][leg][2] for s in held), default=0)
                result = {"action_pair": [outer, inner], "stable_three_supports": valid,
                          "support_pair": list(support), "full_action": action[0].tolist(),
                          "sustained_clearance_m": clearance, "body_relative_lift_m": relative_lift,
                          "terminated": self.done, "trial": self.trial}
                result["sustained_clearance_gain_m"] = min(
                    (s["feet_xyz"][leg][2] - baseline["feet_xyz"][leg][2] for s in held), default=0)
                trials.append(result)
            qualified = [t for t in trials if (t["stable_three_supports"] or self.args.fixed_base and not t["terminated"])
                         and t["body_relative_lift_m"] > 0.005]
            best = max(qualified, key=lambda t: t["body_relative_lift_m"] if self.args.fixed_base else t["sustained_clearance_m"], default=None)
            fallback = max((t for t in trials if not t["terminated"]), key=lambda t: t["body_relative_lift_m"], default=None)
            candidates[name] = {"best": best, "fallback": fallback, "trials": trials}
            print(f"[PROBE] {name}: {best}", flush=True)
        return candidates

    def summary(self):
        rows = self.samples
        motion = [s for s in rows if s["phase"] != "settle"]
        last = rows[-1]
        top_edge = self.args.approach + (self.args.step_count - 1) * self.args.tread
        on_top = lambda s: all(f[1] - self.radius > top_edge and abs(f[0]) < 0.55
                              and abs(f[2] - self.radius - self.args.step_count * self.args.step_height) < 0.015
                              for f in s["feet_xyz"])
        tail = motion[-max(1, math.ceil(0.25 / self.env.step_dt)):]
        complete = bool(tail) and len(tail) >= math.ceil(0.25 / self.env.step_dt) and not self.done and all(
            on_top(s) and s["tilt_rad"] < 0.35 and sum(f > 1.0 for f in s["contact_force_n"]) >= 3 for s in tail)
        crossings = []
        for edge_index in range(self.args.step_count):
            edge = self.args.approach + edge_index * self.args.tread
            height = (edge_index + 1) * self.args.step_height
            for leg, name in enumerate(LEGS):
                for before, after in zip(rows, rows[1:]):
                    # The leading wheel rim must pass above the riser, while unloaded.
                    if before["feet_xyz"][leg][1] + self.radius < edge <= after["feet_xyz"][leg][1] + self.radius:
                        clearance = min(s["feet_xyz"][leg][2] - self.radius - height for s in (before, after))
                        unloaded = all(s["contact_force_n"][leg] < 1.0 for s in (before, after))
                        relative_lift = after["feet_body_xyz"][leg][2] - rows[0]["feet_body_xyz"][leg][2]
                        crossings.append({"edge": edge_index, "leg": name, "time_s": after["time_s"],
                                          "clearance_m": clearance, "unloaded": unloaded,
                                          "active_lift_evidence": clearance > 0.002 and unloaded and relative_lift > 0.005})
        leg_motion = {}
        baseline = next((s for s in reversed(rows) if s["phase"] == "settle"), rows[0])
        for leg, name in enumerate(LEGS):
            airborne = [s for s in motion if s["contact_force_n"][leg] < 1.0]
            leg_motion[name] = {
                "peak_body_relative_lift_m": max((s["feet_body_xyz"][leg][2] - baseline["feet_body_xyz"][leg][2] for s in motion), default=0),
                "unloaded_fraction": len(airborne) / max(1, len(motion)),
            }
        return {"trial": self.trial, "complete": complete, "terminated": self.done,
                "leg_motion": leg_motion,
                "termination_reasons": last["terminations"], "final_root_xyz": last["root_xyz"],
                "forward_displacement_m": last["root_xyz"][1] - rows[0]["root_xyz"][1],
                "max_tilt_rad": max(s["tilt_rad"] for s in rows), "crossings": crossings,
                "forward_tracking_rmse_mps": math.sqrt(sum((s["velocity_body"][1] - self.args.speed) ** 2 for s in motion) / max(1, len(motion))),
                "lateral_velocity_rmse_mps": math.sqrt(sum(s["velocity_body"][0] ** 2 for s in motion) / max(1, len(motion))),
                "yaw_rate_rmse_radps": math.sqrt(sum(s["yaw_rate"] ** 2 for s in motion) / max(1, len(motion)))}

    def rolling_or_policy(self, policy=None, vec=None):
        self.reset("policy" if policy is not None else "rolling")
        if policy is not None:
            policy.reset(self.torch.ones(1, device=self.env.device, dtype=self.torch.long))
        for _ in range(round(self.args.duration / self.env.step_dt)):
            action = self.torch.zeros_like(self.action)
            if policy is None:
                action[0, 8:] = self.args.speed / (self.radius * 20)
            else:
                with self.torch.inference_mode():
                    action = policy(vec.get_observations()).clamp(-1, 1)
            if not self.step(action, "policy" if policy is not None else "rolling"):
                break
        return self.summary()

    def manual(self, candidates):
        selected = {}
        for name in LEGS:
            leg = candidates[name]
            selected[name] = leg["best"] or max((t for t in leg["trials"] if not t["terminated"]),
                                               key=lambda t: t["body_relative_lift_m"], default=None)
        if any(value is None for value in selected.values()):
            return {"complete": False, "status": "not_attempted_all_candidates_terminated"}
        self.reset("manual")
        action = self.torch.zeros_like(self.action)
        wheel = self.args.speed / (self.radius * 20)
        budget = round(self.args.duration / self.env.step_dt)
        for edge_index in range(self.args.step_count):
            edge = self.args.approach + edge_index * self.args.tread
            for leg in (0, 1, 3, 2):
                while not self.done and self.clock < budget and self.state["feet_xyz"][leg][1] + self.radius < edge - 0.02:
                    action[0, 8:] = wheel
                    self.step(action, f"approach_{edge_index}_{LEGS[leg]}")
                action[0, 8:] = 0
                pair = selected[LEGS[leg]]["action_pair"]
                action[0, :8] = self.torch.tensor(selected[LEGS[leg]]["full_action"][:8], device=self.env.device)
                action[0, leg], action[0, leg + 4] = pair
                self.ramp(action, f"lift_{edge_index}_{LEGS[leg]}")
                advance_start = self.clock
                target_y = edge - 0.02 if leg in (0, 1) else edge + 0.22
                while (not self.done and self.clock < budget
                       and self.clock - advance_start < round(self.args.advance_time / self.env.step_dt)
                       and ((self.state["feet_xyz"][leg][1] - self.radius < target_y)
                            if leg in (0, 1) else self.state["root_xyz"][1] < target_y)):
                    action[0, 8:] = wheel
                    action[0, 8 + leg] = 0
                    self.step(action, f"advance_{edge_index}_{LEGS[leg]}")
                action[0, 8:] = 0
                # The lift pair unloads the wheel; this placement pair swings
                # it forward before lowering it: bend -> advance body -> extend.
                place_inner = (self.args.leg_limit / 0.5) * (1 if leg in (0, 1) else -1)
                action[0, leg] = 0.0
                action[0, leg + 4] = place_inner
                self.ramp(action, f"place_{edge_index}_{LEGS[leg]}")
                if leg == 1:
                    # The front pair is on the upper surface.  A short
                    # wheel-driven recovery lets the body pitch settle and
                    # brings the rear axle toward the riser before unloading
                    # either rear leg.
                    action[0, 8:] = wheel
                    for _ in range(round(self.args.recovery_time / self.env.step_dt)):
                        if not self.step(action, f"front_recovery_{edge_index}"):
                            break
                    action[0, 8:] = 0
                if self.done or self.clock >= budget:
                    return self.summary()
        for _ in range(round(0.5 / self.env.step_dt)):
            if not self.step(action, "hold_top"):
                break
        return self.summary()

    def five_phase_obstacle(self, candidates):
        """Execute the staged unload-front / recover-rear obstacle maneuver."""
        selected = {name: candidates[name]["best"] for name in LEGS}
        if any(value is None for value in selected.values()):
            return {"complete": False, "status": "missing_lift_candidate"}
        self.reset("five_phase")
        action = self.torch.zeros_like(self.action)
        wheel = self.args.speed / (self.radius * 20)
        slow = wheel * 0.4
        budget = round(self.args.duration / self.env.step_dt)
        edge = self.args.approach

        def run(target, seconds, phase, wheel_command=0.0):
            nonlocal action
            target = target.clone()
            target[0, 8:] = wheel_command
            count = max(1, round(seconds / self.env.step_dt))
            start = action.clone()
            for index in range(count):
                if self.done or self.clock >= budget:
                    return False
                alpha = (index + 1) / count
                action = start + alpha * (target - start)
                self.step(action, phase)
            return not self.done

        # 1. Approach slowly with a small symmetric leg preload.
        approach = action.clone()
        approach[0, 0:4] = self.args.leg_limit / 0.5 * -0.25
        approach[0, 4:8] = self.args.leg_limit / 0.5 * 0.25
        run(approach, 1.0, "phase1_approach_raise", slow)
        while not self.done and self.clock < budget and self.state["root_xyz"][1] < edge - 0.10:
            self.step(approach, "phase1_approach_slow")

        # 2. Shift load rearward and unload the front pair with a mild nose-up pose.
        unload_front = approach.clone()
        unload_front[0, 0], unload_front[0, 4] = selected["FR"]["action_pair"]
        unload_front[0, 1], unload_front[0, 5] = selected["FL"]["action_pair"]
        run(unload_front, 0.9, "phase2_shift_rear_nose_up")

        # 3. Crawl the front wheels over the riser and then extend the front pair.
        cross = unload_front.clone()
        while not self.done and self.clock < budget and self.state["root_xyz"][1] < edge + 0.08:
            self.step(cross, "phase3_front_cross", slow)
        place_front = cross.clone()
        place_front[0, 0:2] = 0
        place_front[0, 4:6] = 0
        run(place_front, 0.8, "phase3_front_place")

        # 4. Recover pitch toward level, retract rear legs, and drive rear axle.
        recover = place_front.clone()
        recover[0, 2], recover[0, 6] = selected["RL"]["action_pair"]
        recover[0, 3], recover[0, 7] = selected["RR"]["action_pair"]
        run(recover, 0.8, "phase4_shift_front_retract_rear")
        while not self.done and self.clock < budget and self.state["root_xyz"][1] < edge + 0.30:
            self.step(recover, "phase4_rear_cross", slow)

        # 5. Return to neutral posture and normal speed, then hold for evidence.
        normal = self.torch.zeros_like(action)
        run(normal, 1.0, "phase5_restore_posture", wheel)
        for _ in range(round(0.5 / self.env.step_dt)):
            if not self.step(normal, "phase5_hold_top"):
                break
        return self.summary()


def main(args):
    import gymnasium as gym
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    terrain = output.with_suffix(".terrain.usda")
    write_terrain(terrain, args)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd", usd_path=str(terrain),
                                           collision_group=-1, env_spacing=4.0, debug_vis=False)
    cfg.curriculum.terrain_levels = None
    cfg.episode_length_s = args.duration + 5
    for name in ("randomize_contact_material", "randomize_base_mass", "randomize_actuator_gains"):
        setattr(cfg.events, name, None)
    cfg.events.reset_root.params["slope_approach_fraction"] = 0.0
    cfg.commands.locomotion.resampling_time_range = (1e9, 1e9)
    cfg.observations.policy.enable_corruption = False
    cfg.actions.leg_position.clip = {".*": (-args.leg_limit, args.leg_limit)}
    if args.fixed_base:
        cfg.scene.robot.spawn.articulation_props.fix_root_link = None
        cfg.scene.robot.init_state.pos = (0, 0, 0.3)
        cfg.terminations.base_height = None
        original_spawn = cfg.scene.robot.spawn.func

        def spawn_fixed(path, spawn_cfg, *pos, **kwargs):
            from isaaclab.sim import get_current_stage
            from pxr import Gf, UsdPhysics

            result = original_spawn(path, spawn_cfg, *pos, **kwargs)
            stage = get_current_stage()
            base = stage.GetPrimAtPath("/World/envs/env_0/Robot/Geometry/base_link")
            joint = UsdPhysics.FixedJoint.Define(stage, "/World/ProbeFixture")
            joint.CreateBody1Rel().SetTargets([base.GetPath()])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.3))
            return result

        cfg.scene.robot.spawn.func = spawn_fixed
    env = gym.make(TASK, cfg=cfg).unwrapped
    from datetime import datetime, timezone
    from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots.stackforce import CLOSED_USD_PATH

    report = {"schema_version": 1, "task": TASK, "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "asset_sha256": hashlib.sha256(CLOSED_USD_PATH.read_bytes()).hexdigest(),
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "mechanical_impossibility": "not_established_by_finite_search", "free_base": not args.fixed_base,
              "root_teleport_during_trial": False, "actuator_limits_modified": False,
              "policy_action_limit_rad": 0.5, "scripted_action_limit_rad": args.leg_limit,
              "trace": str(output.with_suffix(".jsonl")), "terrain": str(terrain)}
    try:
        with output.with_suffix(".jsonl").open("w", encoding="utf-8") as trace:
            probe = Probe(env, args, trace)
            if args.scan_report:
                saved = json.loads(args.scan_report.read_text())
                if saved["scripted_action_limit_rad"] != args.leg_limit:
                    raise ValueError("scan report action limit differs from --leg-limit")
                candidates = saved["clearance_scan"]
                report["scan_source"] = str(args.scan_report.resolve())
            else:
                candidates = None if args.skip_scan else probe.scan()
            report["clearance_scan"] = candidates
            report["manual"] = (probe.five_phase_obstacle(candidates) if args.five_phase else probe.manual(candidates)) if candidates and not args.fixed_base else {"status": "not_requested"}
            if candidates and not args.fixed_base:
                report["manual"]["all_lifts_prevalidated_on_three_supports"] = all(
                    candidates[name]["best"] is not None and candidates[name]["best"]["stable_three_supports"]
                    for name in LEGS
                )
            report["rolling"] = probe.rolling_or_policy() if not args.fixed_base else {"status": "not_applicable_fixed_base"}
            if args.checkpoint and not args.fixed_base:
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
                from rsl_rl.runners import OnPolicyRunner
                from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_runner_config

                agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
                agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-lib"))
                vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
                runner = OnPolicyRunner(vec, checkpoint_runner_config(args.checkpoint, agent.to_dict()), log_dir=None, device=env.device)
                runner.load(str(args.checkpoint))
                policy = runner.get_inference_policy(device=env.device)
                with args.checkpoint.open("rb") as stream:
                    report["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                report["policy"] = probe.rolling_or_policy(policy, vec)
            else:
                report["policy"] = {"status": "not_requested"}
    except Exception as error:
        report["runtime_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        env.close()
    print(json.dumps({k: report[k] for k in ("manual", "rolling", "policy")}, indent=2))
    print(f"[PROBE] report={output}")


if __name__ == "__main__":
    args = parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    failed = False
    try:
        main(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        failed = True
    finally:
        # Kit can terminate the process with status zero during close(); print
        # the exception first and use an explicit status for failed runs.
        if failed:
            import os
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        launcher.app.close()
