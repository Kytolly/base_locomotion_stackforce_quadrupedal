#!/usr/bin/env python3
"""Audit composed closure frames and run a CPU, suspended-base gravity smoke test.

No input asset is saved. This is not a contact, actuator, or RL qualification.
"""
import argparse
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=2400)
parser.add_argument("--disable-closures", action="store_true", help="Negative control; expected to fail drift checks")
parser.add_argument("--output", type=Path, default=ROOT / "validation/simulation_report.json")
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "fast_shutdown": False})
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import carb
import omni.usd
from isaacsim.core.api import SimulationContext


def world(prim):
    return UsdGeom.XformCache().GetLocalToWorldTransform(prim)


def unique(stage, name, body=False):
    items = [p for p in stage.Traverse() if p.GetName() == name
             and (not body or p.HasAPI(UsdPhysics.RigidBodyAPI))]
    if len(items) != 1:
        raise RuntimeError(f"{name}: expected one prim, got {[str(p.GetPath()) for p in items]}")
    return items[0]


def errors(stage, joint):
    poses = []
    for i in (0, 1):
        body = stage.GetPrimAtPath(joint.GetPrim().GetRelationship(f"physics:body{i}").GetTargets()[0])
        pos = joint.GetPrim().GetAttribute(f"physics:localPos{i}").Get()
        quat = joint.GetPrim().GetAttribute(f"physics:localRot{i}").Get()
        local = Gf.Matrix4d(1)
        local.SetRotate(Gf.Quatd(quat))
        local.SetTranslateOnly(Gf.Vec3d(pos))
        poses.append(local * world(body))
    token = joint.GetAxisAttr().Get()
    axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[token]
    axes = [p.TransformDir(axis).GetNormalized() for p in poses]
    return ((poses[0].ExtractTranslation() - poses[1].ExtractTranslation()).GetLength(),
            math.acos(min(1., abs(Gf.Dot(*axes)))))


def main(report):
    path = ROOT / "usd/stackforce_quadrupedal_wheeled_robot_closed.usda"
    omni.usd.get_context().open_stage(str(path))
    stage = omni.usd.get_context().get_stage()
    if stage.GetCompositionErrors():
        raise RuntimeError(str(stage.GetCompositionErrors()))
    config = json.loads((ROOT / "config/closure_frames.json").read_text())
    base = unique(stage, "base_link", body=True)
    report["rigid_bodies"] = sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    tree_joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
                   if p.IsA(UsdPhysics.RevoluteJoint) and "closure_joint" not in str(p.GetPath())]
    if len(tree_joints) != 20 or report["rigid_bodies"] != 21:
        raise RuntimeError("Expected 21 rigid bodies and 20 tree revolute joints")
    report["tree_joints"] = {}
    for joint in tree_joints:
        pos, angle = errors(stage, joint)
        report["tree_joints"][joint.GetPrim().GetName()] = {"anchor_error_m": pos, "axis_error_rad": angle}
        if pos > 2e-6 or angle > 1e-4:
            raise RuntimeError(f"{joint.GetPath()}: tree joint frame mismatch")
    report["legs"] = {}
    closures = []
    for leg, entry in config["legs"].items():
        a = world(unique(stage, entry["w1_frame"])).ExtractTranslation()
        b = world(unique(stage, entry["w2_frame"])).ExtractTranslation()
        axis = world(base).TransformDir(Gf.Vec3d(*entry["axis_in_base"])).GetNormalized()
        delta = b - a
        radial = (delta - axis * Gf.Dot(axis, delta)).GetLength()
        joint = UsdPhysics.RevoluteJoint(unique(stage, f"{leg}_Closure_Joint"))
        pos, angle = errors(stage, joint)
        report["legs"][leg] = {"imported_radial_m": radial, "axial_offset_m": Gf.Dot(axis, delta),
                               "initial_anchor_error_m": pos, "initial_axis_error_rad": angle,
                               "max_anchor_error_m": pos, "max_axis_error_rad": angle}
        if radial > 2e-6 or pos > 2e-6 or angle > 1e-4:
            raise RuntimeError(f"{leg}: imported frame validation failed")
        if not joint.GetExcludeFromArticulationAttr().Get():
            raise RuntimeError(f"{leg}: closure must be excluded from tree articulation")
        closures.append((leg, joint))
        if args.disable_closures:
            joint.CreateJointEnabledAttr(False)
    report["static_status"] = "PASS"
    # Anchor at the existing base pose; authored only in this unsaved test stage.
    fixed = UsdPhysics.FixedJoint.Define(stage, "/ValidationBaseAnchor")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(world(base).ExtractTranslation()))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(world(base).ExtractRotationQuat()))
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    carb.settings.get_settings().set_bool("/physics/updateVelocitiesToUsd", True)
    sim = SimulationContext(physics_dt=1 / 240, rendering_dt=1 / 60, backend="numpy", device="cpu")
    sim.get_physics_context().set_gravity(-9.81)
    before = world(unique(stage, "FR_Inner_Calf_Link", body=True)).ExtractTranslation()
    sim.initialize_physics()
    sim.play()
    for step in range(args.steps):
        sim.step(render=False)
        for leg, joint in closures:
            pos, angle = errors(stage, joint)
            if not math.isfinite(pos + angle):
                raise RuntimeError(f"{leg}: nonfinite closure pose at step {step}")
            row = report["legs"][leg]
            row["max_anchor_error_m"] = max(row["max_anchor_error_m"], pos)
            row["max_axis_error_rad"] = max(row["max_axis_error_rad"], angle)
    after = world(unique(stage, "FR_Inner_Calf_Link", body=True)).ExtractTranslation()
    report["body_motion_m"] = (after - before).GetLength()
    report["steps"] = args.steps
    report["dt_s"] = 1 / 240
    report["simulation_time_s"] = sim.current_time
    sim.stop()
    if report["body_motion_m"] < 1e-6:
        raise RuntimeError("No measured link motion; simulation result is inconclusive")
    if any(r["max_anchor_error_m"] > 1e-3 or r["max_axis_error_rad"] > .01
           for r in report["legs"].values()):
        raise RuntimeError("Closure drift exceeds smoke-test limits (1 mm / 0.01 rad)")
    report["simulation_status"] = "PASS"


report = {"static_status": "NOT_RUN", "simulation_status": "NOT_RUN", "rl_ready": False,
          "negative_control": args.disable_closures,
          "test": "CPU suspended-base gravity; no ground or commanded actuation"}
try:
    main(report)
except Exception:
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    app.close()
if "error" in report:
    raise SystemExit(1)
