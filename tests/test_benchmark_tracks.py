"""Determinism and geometry-contract tests for both fixed tracks."""

from base_locomotion_stackforce_quadrupedal.benchmark import (
    TrackTraversal,
    plateau_track_parameters,
    track_length,
    washboard_track_parameters,
)


def test_plateau_parameters_are_seeded_and_complete() -> None:
    nominal = plateau_track_parameters()
    randomized_a = plateau_track_parameters(9001, True)
    randomized_b = plateau_track_parameters(9001, True)

    assert nominal["slope_deg"] == 12.0
    assert nominal["ramp_height_m"] > 0.0
    assert track_length(nominal) > 5.0
    assert randomized_a == randomized_b


def test_washboard_parameters_are_seeded_and_complete() -> None:
    nominal = washboard_track_parameters()
    randomized_a = washboard_track_parameters(9002, True)
    randomized_b = washboard_track_parameters(9002, True)

    assert nominal["version"] == 2
    assert nominal["cylinders_per_row"] == 7
    assert nominal["ramp_height_m"] == 2.0 * nominal["cylinder_radius_m"]
    assert nominal["row_stagger_offsets_m"] == [0.0, 0.075]
    assert nominal["boundary_cap_types"] == [["box", "full_cylinder"], ["full_cylinder", "box"]]
    assert abs(track_length(nominal) - 5.4294) < 1e-6
    assert randomized_a == randomized_b


def test_washboard_stagger_and_traversal_contract() -> None:
    for seed in range(100):
        parameters = washboard_track_parameters(seed, True)
        staggers = parameters["row_stagger_offsets_m"]
        assert sorted(staggers) == [0.0, parameters["cylinder_radius_m"]]
        leading = staggers.index(0.0)
        assert parameters["boundary_cap_types"][leading] == ["box", "full_cylinder"]
        assert parameters["boundary_cap_types"][1 - leading] == ["full_cylinder", "box"]
        assert track_length(parameters) < 35.0 * 0.28
        traversal = TrackTraversal(parameters)
        assert len(traversal.gates) == 4
        assert abs(traversal.gates[1] - traversal.gates[0] - parameters["washboard_length_m"]) < 1e-9
        traversal.update(0.0, 0.0)
        for step in range(int(track_length(parameters) / 0.1) + 1):
            traversal.update(0.0, step * 0.1)
        traversal.update(0.0, track_length(parameters))
        assert traversal.complete


def test_washboard_usd_geometry(monkeypatch) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    import pytest

    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from base_locomotion_stackforce_quadrupedal.benchmark.geometry import spawn_benchmark_track

    stage = Usd.Stage.CreateInMemory()
    sim = ModuleType("isaaclab.sim")
    sim.get_current_stage = lambda: stage
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    env = SimpleNamespace()
    spawn_benchmark_track(env, None, "washboard", 8201, False)
    root = "/World/BenchmarkTrack"
    cylinders = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Cylinder)]
    boxes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Cube)]
    assert len(cylinders) == 12
    assert len(boxes) == 2
    for prim in cylinders + boxes:
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    for prim in cylinders:
        cylinder = UsdGeom.Cylinder(prim)
        assert cylinder.GetAxisAttr().Get() == "X"
        assert cylinder.GetRadiusAttr().Get() == pytest.approx(0.075)
        assert cylinder.GetHeightAttr().Get() == pytest.approx(1.0)
    parameters = env.benchmark_track_parameters
    bed_start = parameters["approach_m"] + parameters["ramp_length_m"]
    bed_end = bed_start + parameters["washboard_length_m"]
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    entry = cache.ComputeWorldBound(stage.GetPrimAtPath(root + "/BoundaryBox_Row_0_00")).ComputeAlignedRange()
    exit_box = cache.ComputeWorldBound(stage.GetPrimAtPath(root + "/BoundaryBox_Row_1_06")).ComputeAlignedRange()
    assert entry.GetMin()[1] == pytest.approx(bed_start)
    assert exit_box.GetMax()[1] == pytest.approx(bed_end)
    assert entry.GetMax()[2] == pytest.approx(0.155)
    assert exit_box.GetMax()[2] == pytest.approx(0.155)
    for name in ("Approach", "EntryRamp", "ExitRamp", "Release"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(root + "/" + name))
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        a, b, c = [points[index] for index in indices[:3]]
        assert Gf.Cross(b - a, c - a)[2] > 0.0
    entry_points = UsdGeom.Mesh(stage.GetPrimAtPath(root + "/EntryRamp")).GetPointsAttr().Get()
    exit_points = UsdGeom.Mesh(stage.GetPrimAtPath(root + "/ExitRamp")).GetPointsAttr().Get()
    assert entry_points[2][2] == pytest.approx(entry.GetMax()[2])
    assert exit_points[0][2] == pytest.approx(exit_box.GetMax()[2])

    spawn_benchmark_track(env, None, "plateau", 8101, False)
    assert env.benchmark_track_parameters["version"] == 2
    for name in ("Approach", "Uphill", "Plateau", "Downhill", "Release"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(root + "/" + name))
        points = mesh.GetPointsAttr().Get()
        a, b, c = [points[index] for index in mesh.GetFaceVertexIndicesAttr().Get()[:3]]
        assert Gf.Cross(b - a, c - a)[2] > 0.0


def test_track_success_requires_ordered_traversal_and_staying_in_corridor() -> None:
    parameters = plateau_track_parameters()
    traversal = TrackTraversal(parameters)
    traversal.update(0.0, track_length(parameters) + 0.1)
    assert not traversal.complete

    traversal = TrackTraversal(parameters)
    for step in range(int(track_length(parameters) / 0.1) + 2):
        y = min(step * 0.1, traversal.gates[-1])
        traversal.update(0.0, y)
    traversal.update(0.0, traversal.gates[-1])
    assert traversal.complete

    traversal = TrackTraversal(parameters)
    for step in range(int(track_length(parameters) / 0.1) + 2):
        y = min(step * 0.1, traversal.gates[-1])
        x = traversal.half_width_m + 0.01 if 1.0 <= y <= 1.2 else 0.0
        traversal.update(x, y)
    traversal.update(0.0, traversal.gates[-1])
    assert not traversal.complete
