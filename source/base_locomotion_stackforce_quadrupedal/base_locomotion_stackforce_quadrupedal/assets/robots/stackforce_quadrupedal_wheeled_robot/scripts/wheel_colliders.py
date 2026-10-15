"""Identical analytic wheel colliders for the four feet (pure ``pxr``, no Isaac Sim app).

The four wheel meshes are separate exports; PhysX cooks each into a different
vertex-limited convex hull, so the wheels rest at centre heights 31.4-33.0 mm
and the stance carries a diagonal (twist) preload.  This step disables those
mesh colliders and gives every ``*_Foot_Link`` the same uniform prism:
``SIDES`` flats whose distance to the axle equals the measured wheel radius,
spanning the measured axial extent.  The wheel meshes stay as visuals.
See ``docs/OUTER_HIP_STATIC_FORCE_DIAGNOSIS.md`` in the locomotion project.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Pcp, Usd, UsdGeom, UsdPhysics, Vt


SIDES = 32  # 64 hull vertices: the PhysX GPU convex-hull limit
COLLIDER_NAME = "wheel_collider"
FOOT_SUFFIX = "_Foot_Link"
MESH_SUFFIX = "_foot"


def _source_prim(stage: Usd.Stage, prim: Usd.Prim) -> Usd.Prim:
    """Editable prim behind an instance proxy (follows the instance's internal reference)."""
    if not prim.IsInstanceProxy():
        return prim
    instance = prim
    while not instance.IsInstance():
        instance = instance.GetParent()
    arcs = [
        arc for arc in Usd.PrimCompositionQuery(instance).GetCompositionArcs()
        if arc.GetArcType() == Pcp.ArcTypeReference
    ]
    if len(arcs) != 1:
        raise RuntimeError(f"{instance.GetPath()}: expected one reference arc, found {len(arcs)}")
    target = arcs[0].GetTargetPrimPath()
    source = stage.GetPrimAtPath(prim.GetPath().ReplacePrefix(instance.GetPath(), target))
    if not source.IsValid():
        raise RuntimeError(f"{prim.GetPath()}: reference source {target} not found")
    return source


def _foot_links(stage: Usd.Stage) -> list[Usd.Prim]:
    links = [
        prim for prim in stage.Traverse()
        if prim.GetName().endswith(FOOT_SUFFIX) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(links) != 4:
        raise RuntimeError(f"expected 4 foot links, found {[str(p.GetPath()) for p in links]}")
    return sorted(links, key=lambda prim: prim.GetName())


def _mesh_colliders(link: Usd.Prim) -> list[Usd.Prim]:
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    return [
        prim for prim in Usd.PrimRange(link, predicate)
        if prim.IsA(UsdGeom.Mesh) and prim.GetName().endswith(MESH_SUFFIX) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _wheel_extent(link: Usd.Prim, meshes: list[Usd.Prim]) -> tuple[float, float, float]:
    """(radius, axial min, axial max) of the wheel meshes in the foot-link frame (axle = local y)."""
    cache = UsdGeom.XformCache()
    to_link = np.array(cache.GetLocalToWorldTransform(link).GetInverse())
    points = []
    for mesh in meshes:
        local = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=float)
        matrix = np.array(cache.GetLocalToWorldTransform(mesh)) @ to_link  # row-vector convention
        points.append(np.c_[local, np.ones(len(local))] @ matrix)
    xyz = np.concatenate(points)[:, :3]
    return float(np.hypot(xyz[:, 0], xyz[:, 2]).max()), float(xyz[:, 1].min()), float(xyz[:, 1].max())


def prism_points(radius: float, axial_min: float, axial_max: float, sides: int = SIDES) -> list[Gf.Vec3f]:
    """Uniform prism around the local y axle with its flats at ``radius``."""
    circum = radius / math.cos(math.pi / sides)
    return [
        Gf.Vec3f(circum * math.cos(2 * math.pi * (k + 0.5) / sides), y, circum * math.sin(2 * math.pi * (k + 0.5) / sides))
        for y in (axial_min, axial_max)
        for k in range(sides)
    ]


def _prism_topology(sides: int) -> tuple[list[int], list[int]]:
    indices, counts = [], []
    for k in range(sides):
        indices += [k, (k + 1) % sides, sides + (k + 1) % sides, sides + k]
        counts.append(4)
    indices += list(range(sides)) + list(range(2 * sides - 1, sides - 1, -1))
    counts += [sides, sides]
    return indices, counts


def replace_wheel_colliders(stage: Usd.Stage, sides: int = SIDES) -> dict[str, dict[str, float]]:
    """Disable the mesh wheel colliders and add one identical prism per foot link."""
    extents = {}
    for link in _foot_links(stage):
        meshes = _mesh_colliders(link)
        if not meshes:
            raise RuntimeError(f"{link.GetPath()}: no *_foot mesh collider found")
        extents[link.GetName()] = (link, meshes, _wheel_extent(link, meshes))
    radii = [extent[0] for _, _, extent in extents.values()]
    spans = [sorted((abs(lo), abs(hi))) for _, _, (_, lo, hi) in extents.values()]  # left wheels span -y
    if max(radii) - min(radii) > 1e-5 or np.ptp(np.array(spans), axis=0).max() > 1e-5:
        raise RuntimeError(f"wheel meshes are not mirror images: radii {radii}, axial spans {spans}")
    radius = float(np.mean(radii))
    indices, counts = _prism_topology(sides)
    report = {}
    for name, (link, meshes, (_, axial_min, axial_max)) in extents.items():
        for mesh in meshes:
            UsdPhysics.CollisionAPI(_source_prim(link.GetStage(), mesh)).CreateCollisionEnabledAttr(False)
        collider = UsdGeom.Mesh.Define(stage, link.GetPath().AppendChild(COLLIDER_NAME))
        collider.CreatePointsAttr(Vt.Vec3fArray(prism_points(radius, axial_min, axial_max, sides)))
        collider.CreateFaceVertexIndicesAttr(indices)
        collider.CreateFaceVertexCountsAttr(counts)
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        report[name] = {"radius_m": radius, "axial_min_m": axial_min, "axial_max_m": axial_max, "sides": sides}
    return report


__all__ = ["COLLIDER_NAME", "SIDES", "prism_points", "replace_wheel_colliders"]
