#!/usr/bin/env python3
"""Add four PhysX revolute closure constraints to an imported loop-cut USD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("input_usd", type=Path)
PARSER.add_argument("--output", type=Path)
ARGS = PARSER.parse_args()

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True, "fast_shutdown": False})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


ASSET_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ASSET_ROOT / "config/closure_frames.json"


def find_unique(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    bodies = [prim for prim in matches if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies:
        matches = bodies
    xforms = [prim for prim in matches if prim.IsA(UsdGeom.Xform)]
    if xforms:
        matches = xforms
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Expected one prim named {name!r}, found {len(matches)}: {paths}")
    return matches[0]


def world_matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def joint_world_matrix(position: Gf.Vec3d, axis: Gf.Vec3d) -> Gf.Matrix4d:
    x_axis = axis.GetNormalized()
    helper = Gf.Vec3d(0.0, 0.0, 1.0)
    if abs(Gf.Dot(x_axis, helper)) > 0.9:
        helper = Gf.Vec3d(0.0, 1.0, 0.0)
    y_axis = Gf.Cross(helper, x_axis).GetNormalized()
    z_axis = Gf.Cross(x_axis, y_axis).GetNormalized()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(x_axis[0], x_axis[1], x_axis[2], 0.0))
    matrix.SetRow(1, Gf.Vec4d(y_axis[0], y_axis[1], y_axis[2], 0.0))
    matrix.SetRow(2, Gf.Vec4d(z_axis[0], z_axis[1], z_axis[2], 0.0))
    matrix.SetRow(3, Gf.Vec4d(position[0], position[1], position[2], 1.0))
    return matrix


def local_pose(joint_world: Gf.Matrix4d, body: Usd.Prim) -> tuple[Gf.Vec3f, Gf.Quatf]:
    local = joint_world * world_matrix(body).GetInverse()
    position = local.ExtractTranslation()
    rotation = local.ExtractRotationQuat()
    return Gf.Vec3f(position), Gf.Quatf(rotation)


def add_closures(input_path: Path, output_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {input_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError("Imported asset has no default prim")
    scope_path = default_prim.GetPath().AppendChild(config["closure_joint_scope"])
    if stage.GetPrimAtPath(scope_path).IsValid():
        stage.RemovePrim(scope_path)
    UsdGeom.Scope.Define(stage, scope_path)

    base = find_unique(stage, "base_link")
    base_world = world_matrix(base)
    for leg, entry in config["legs"].items():
        inner_lower = find_unique(stage, entry["inner_lower_link"])
        foot = find_unique(stage, entry["foot_link"])
        w2_frame = find_unique(stage, entry["w2_frame"])
        w1_frame = find_unique(stage, entry["w1_frame"])
        if not all(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in (inner_lower, foot)):
            raise RuntimeError(f"{leg}: closure endpoints must be rigid bodies")

        w2_world = world_matrix(w2_frame).ExtractTranslation()
        axis_base = Gf.Vec3d(*entry["axis_in_base"])
        axis_world = base_world.TransformDir(axis_base).GetNormalized()
        delta = w2_world - world_matrix(w1_frame).ExtractTranslation()
        radial = (delta - axis_world * Gf.Dot(delta, axis_world)).GetLength()
        if radial > 2e-6:
            raise RuntimeError(f"{leg}: imported Wheel/Closure radial residual {radial * 1000:.6f} mm")
        common_world = joint_world_matrix(w2_world, axis_world)
        pos0, rot0 = local_pose(common_world, inner_lower)
        pos1, rot1 = local_pose(common_world, foot)

        joint = UsdPhysics.RevoluteJoint.Define(stage, scope_path.AppendChild(f"{leg}_Closure_Joint"))
        joint.CreateBody0Rel().SetTargets([inner_lower.GetPath()])
        joint.CreateBody1Rel().SetTargets([foot.GetPath()])
        joint.CreateLocalPos0Attr().Set(pos0)
        joint.CreateLocalRot0Attr().Set(rot0)
        joint.CreateLocalPos1Attr().Set(pos1)
        joint.CreateLocalRot1Attr().Set(rot1)
        joint.CreateAxisAttr(config["joint_axis_token"])
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten resolves composition arcs before moving the asset to another directory.
    stage.Flatten().Export(str(output_path))
    reopened = Usd.Stage.Open(str(output_path))
    for entry in config["legs"].values():
        find_unique(reopened, entry["inner_lower_link"])
        find_unique(reopened, entry["w2_frame"])
    print(f"Closed-loop USD: {output_path}")


def main() -> None:
    output = ARGS.output or ARGS.input_usd.with_name(ARGS.input_usd.stem + "_closed.usda")
    try:
        add_closures(ARGS.input_usd.resolve(), output.resolve())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        APP.close()


if __name__ == "__main__":
    main()
