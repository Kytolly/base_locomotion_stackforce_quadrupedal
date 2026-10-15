"""The closed USD gives all four wheels the same analytic collider.

Separately cooked wheel-mesh hulls left the wheels resting at 31.4-33.0 mm and
preloaded the stance diagonally (docs/OUTER_HIP_STATIC_FORCE_DIAGNOSIS.md).
The settled-stance check (centre spread <= 0.3 mm, twist <= 10 %) needs PhysX:
``scripts/diagnose_outer_hip_force.py --phase stand --check-wheel-settle``.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ASSET = (
    Path(__file__).resolve().parents[1]
    / "source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/assets/robots/stackforce_quadrupedal_wheeled_robot"
)
# Same override as env/robots/stackforce.py, so a candidate rebuild can be checked before it is installed.
CLOSED_USD = Path(os.getenv("SF_QUAD_ASSET_ROOT", ASSET)) / "usd/stackforce_quadrupedal_wheeled_robot_closed.usda"
WHEEL_RADIUS_M = 0.033  # mdp/support.py


def _wheel_colliders_module():
    spec = importlib.util.spec_from_file_location("_wheel_colliders", ASSET / "scripts/wheel_colliders.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prism_flats_sit_at_the_wheel_radius():
    module = _wheel_colliders_module()
    points = np.array(module.prism_points(WHEEL_RADIUS_M, 0.0045, 0.0544), dtype=float)
    assert len(points) == 2 * module.SIDES == 64
    rim = points[: module.SIDES][:, [0, 2]]
    edges = np.roll(rim, -1, axis=0) - rim
    normals = np.c_[edges[:, 1], -edges[:, 0]] / np.linalg.norm(edges, axis=1, keepdims=True)
    apothems = np.abs((rim * normals).sum(axis=1))
    assert apothems == pytest.approx(np.full(module.SIDES, WHEEL_RADIUS_M), abs=1e-7)
    # A flat faces straight down, so a wheel at rest sits exactly one radius above the ground.
    assert -points[:, 2].min() == pytest.approx(WHEEL_RADIUS_M, abs=1e-7)


@pytest.fixture(scope="module")
def stage():
    return Usd.Stage.Open(str(CLOSED_USD))


def _enabled_colliders(link):
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    found = []
    for prim in Usd.PrimRange(link, predicate):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is None or enabled:
                found.append(prim)
    return found


def test_each_foot_has_only_the_shared_prism_collider(stage):
    module = _wheel_colliders_module()
    links = [p for p in stage.Traverse() if p.GetName().endswith("_Foot_Link")]
    assert sorted(p.GetName()[:2] for p in links) == ["FL", "FR", "RL", "RR"]
    rims = {}
    for link in links:
        colliders = _enabled_colliders(link)
        assert [p.GetName() for p in colliders] == [module.COLLIDER_NAME], link.GetPath()
        mesh = UsdGeom.Mesh(colliders[0])
        assert UsdPhysics.MeshCollisionAPI(colliders[0]).GetApproximationAttr().Get() == "convexHull"
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        assert len(points) == 64
        side = 1.0 if link.GetName()[:2] in ("FR", "RR") else -1.0
        mirrored = points * np.array([1.0, side, 1.0])
        rims[link.GetName()[:2]] = mirrored[np.lexsort(np.round(mirrored, 6).T)]
    reference = rims["FR"]
    for leg, rim in rims.items():
        assert rim == pytest.approx(reference, abs=1e-7), leg
    radial = np.hypot(reference[:, 0], reference[:, 2])
    assert radial * math.cos(math.pi / module.SIDES) == pytest.approx(np.full(64, WHEEL_RADIUS_M), abs=1e-7)
