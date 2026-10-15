#!/usr/bin/env python3
"""Write a diagnostic copy of the closed USD with replaced wheel colliders.

The packaged wheel colliders are ``convexHull`` approximations of four
separately exported wheel meshes; PhysX cooks each to a different hull, so the
wheels rest at different centre heights (docs/OUTER_HIP_STATIC_FORCE_DIAGNOSIS.md).
This script disables those colliders and adds one identical collider per wheel:

* ``prism``  : a uniform 32-gon prism with its flats at the 33 mm wheel radius
  and the mesh's 4.5-54.4 mm axial span (proposed fix);
* ``sphere`` : a 33 mm sphere at the wheel centre (diagnostic only; wrong width).

The packaged asset is never modified.  Use the copy with
``SF_QUAD_ASSET_ROOT=<output-root>``::

    python scripts/make_wheel_collider_variant.py --shape prism --output-root /tmp/asset_prism
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/assets/robots/stackforce_quadrupedal_wheeled_robot"
USD_NAME = "usd/stackforce_quadrupedal_wheeled_robot_closed.usda"
WHEEL_RADIUS_M = 0.033
AXIAL_SPAN_M = (0.0045, 0.0544)  # measured from the packaged wheel meshes, along the foot-link y axis
SIDES = 32  # 64 hull vertices: the PhysX GPU convex limit


def _disable_mesh_colliders(stage: Usd.Stage) -> None:
    for top in stage.GetPseudoRoot().GetAllChildren():
        for prim in Usd.PrimRange(top, Usd.PrimAllPrimsPredicate):
            if prim.GetName().endswith("_foot") and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)


def _add_prism(stage: Usd.Stage, link: Usd.Prim, side: float) -> None:
    circum = WHEEL_RADIUS_M / math.cos(math.pi / SIDES)
    points = [
        Gf.Vec3f(circum * math.cos(2 * math.pi * (k + 0.5) / SIDES), side * y, circum * math.sin(2 * math.pi * (k + 0.5) / SIDES))
        for y in AXIAL_SPAN_M
        for k in range(SIDES)
    ]
    indices, counts = [], []
    for k in range(SIDES):
        indices += [k, (k + 1) % SIDES, SIDES + (k + 1) % SIDES, SIDES + k]
        counts.append(4)
    indices += list(range(SIDES)) + list(range(2 * SIDES - 1, SIDES - 1, -1))
    counts += [SIDES, SIDES]
    mesh = UsdGeom.Mesh.Define(stage, link.GetPath().AppendChild("wheel_collider"))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


def _add_sphere(stage: Usd.Stage, link: Usd.Prim, side: float) -> None:
    sphere = UsdGeom.Sphere.Define(stage, link.GetPath().AppendChild("wheel_collider"))
    sphere.GetRadiusAttr().Set(WHEEL_RADIUS_M)
    sphere.AddTranslateOp().Set(Gf.Vec3d(0.0, side * sum(AXIAL_SPAN_M) / 2, 0.0))
    sphere.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=("prism", "sphere"), default="prism")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_root / USD_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ASSET / USD_NAME, target)
    stage = Usd.Stage.Open(str(target))
    _disable_mesh_colliders(stage)
    for prim in list(stage.Traverse()):
        if prim.GetName().endswith("_Foot_Link"):
            # Right wheels extend along +y of the foot link, left wheels along -y.
            side = 1.0 if prim.GetName()[:2] in ("FR", "RR") else -1.0
            (_add_prism if args.shape == "prism" else _add_sphere)(stage, prim, side)
    stage.GetRootLayer().Save()
    print(f"wrote {target} ({args.shape} wheel colliders)")


if __name__ == "__main__":
    main()
