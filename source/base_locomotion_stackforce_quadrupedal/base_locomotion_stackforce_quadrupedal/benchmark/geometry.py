"""USD/PhysX geometry construction shared by fixed benchmarks."""

from __future__ import annotations

from typing import Any

from .plateau_track import plateau_track_parameters
from .washboard_track import washboard_track_parameters


def _parameters(kind: str, seed: int, randomized: bool) -> dict[str, Any]:
    if kind == "plateau":
        return plateau_track_parameters(seed, randomized)
    if kind == "washboard":
        return washboard_track_parameters(seed, randomized)
    raise ValueError("kind must be 'plateau' or 'washboard'.")


def track_length(parameters: dict[str, Any]) -> float:
    """Return the commanded travel distance for one track."""
    if parameters["kind"] == "plateau":
        return float(
            parameters["approach_m"]
            + 2.0 * parameters["ramp_length_m"]
            + parameters["plateau_length_m"]
            + parameters["release_m"]
        )
    return float(
        parameters["approach_m"]
        + 2.0 * parameters["ramp_length_m"]
        + (parameters["ridge_count"] - 1) * parameters["ridge_spacing_m"]
        + parameters["release_m"]
    )


class TrackTraversal:
    """Track-local traversal proof using ordered gates and lateral bounds."""

    def __init__(
        self, parameters: dict[str, Any], margin_m: float = 0.15, max_step_m: float = 0.5
    ) -> None:
        if max_step_m <= 0.0:
            raise ValueError("max_step_m must be positive.")
        self.parameters = parameters
        self.track_length_m = track_length(parameters)
        self.half_width_m = float(parameters["width_m"]) / 2.0 - margin_m
        if self.half_width_m <= 0.0:
            raise ValueError("Track width must exceed twice the corridor margin.")
        approach = float(parameters["approach_m"])
        ramp = float(parameters["ramp_length_m"])
        if parameters["kind"] == "plateau":
            plateau_end = approach + ramp + float(parameters["plateau_length_m"])
            gates = (approach + ramp, plateau_end, plateau_end + ramp)
        else:
            bed_end = approach + ramp + (int(parameters["ridge_count"]) - 1) * float(
                parameters["ridge_spacing_m"]
            )
            gates = (approach + ramp, bed_end + ramp)
        self.gates = tuple(gates) + (self.track_length_m,)
        self.gate_index = 0
        self.corridor_violation = False
        self.trajectory_discontinuity = False
        self.max_step_m = max_step_m
        self.max_y_m = float("-inf")
        self._last_y_m: float | None = None

    def update(self, x_m: float, y_m: float) -> None:
        if self._last_y_m is None:
            self.trajectory_discontinuity = y_m > self.gates[0]
        elif abs(y_m - self._last_y_m) > self.max_step_m:
            self.trajectory_discontinuity = True
        self._last_y_m = y_m
        self.max_y_m = max(self.max_y_m, y_m)
        if -0.3 <= y_m <= self.track_length_m and abs(x_m) > self.half_width_m:
            self.corridor_violation = True
        while self.gate_index < len(self.gates) and y_m >= self.gates[self.gate_index]:
            self.gate_index += 1

    @property
    def complete(self) -> bool:
        return (
            self.gate_index == len(self.gates)
            and not self.corridor_violation
            and not self.trajectory_discontinuity
        )


def spawn_benchmark_track(
    env,
    env_ids,
    kind: str = "plateau",
    seed: int = 8101,
    randomized: bool = False,
) -> None:
    """Create one fixed collision track along the robot's positive Y direction."""
    del env_ids
    from isaaclab.sim import get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    parameters = _parameters(kind, int(seed), bool(randomized))
    stage = get_current_stage()
    root_path = "/World/BenchmarkTrack"
    previous = stage.GetPrimAtPath(root_path)
    if previous.IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    material = UsdShade.Material.Define(stage, f"{root_path}/Material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.0)
    physics_material.CreateDynamicFrictionAttr(0.9)
    physics_material.CreateRestitutionAttr(0.0)

    def bind(prim) -> None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def ramp(name: str, y0: float, y1: float, z0: float, z1: float) -> None:
        width = float(parameters["width_m"])
        thickness = 0.08
        points = (
            (-width / 2, y0, z0),
            (width / 2, y0, z0),
            (-width / 2, y1, z1),
            (width / 2, y1, z1),
            (-width / 2, y0, z0 - thickness),
            (width / 2, y0, z0 - thickness),
            (-width / 2, y1, z1 - thickness),
            (width / 2, y1, z1 - thickness),
        )
        mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(
            [0, 2, 3, 1, 4, 5, 7, 6, 0, 4, 6, 2, 1, 3, 7, 5, 0, 1, 5, 4, 2, 6, 7, 3]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        bind(mesh.GetPrim())

    approach = float(parameters["approach_m"])
    ramp_length = float(parameters["ramp_length_m"])
    ramp_height = float(parameters["ramp_height_m"])
    ramp("Approach", -0.3, approach, 0.0, 0.0)
    if kind == "plateau":
        plateau_length = float(parameters["plateau_length_m"])
        release = float(parameters["release_m"])
        ramp("Uphill", approach, approach + ramp_length, 0.0, ramp_height)
        ramp(
            "Plateau",
            approach + ramp_length,
            approach + ramp_length + plateau_length,
            ramp_height,
            ramp_height,
        )
        ramp(
            "Downhill",
            approach + ramp_length + plateau_length,
            approach + 2.0 * ramp_length + plateau_length,
            ramp_height,
            0.0,
        )
        ramp(
            "Release",
            approach + 2.0 * ramp_length + plateau_length,
            approach + 2.0 * ramp_length + plateau_length + release,
            0.0,
            0.0,
        )
    else:
        radius = float(parameters["ridge_radius_m"])
        spacing = float(parameters["ridge_spacing_m"])
        count = int(parameters["ridge_count"])
        width = float(parameters["width_m"])
        ramp("EntryRamp", approach, approach + ramp_length, 0.0, 2.0 * radius)
        bed_start = approach + ramp_length
        for index in range(count):
            ridge = UsdGeom.Cylinder.Define(stage, f"{root_path}/Ridge_{index:02d}")
            ridge.CreateAxisAttr(UsdGeom.Tokens.x)
            ridge.CreateRadiusAttr(radius)
            ridge.CreateHeightAttr(width)
            ridge.AddTranslateOp().Set(
                Gf.Vec3d(0.0, bed_start + index * spacing, radius)
            )
            UsdPhysics.CollisionAPI.Apply(ridge.GetPrim())
            bind(ridge.GetPrim())
        bed_end = bed_start + (count - 1) * spacing
        ramp("ExitRamp", bed_end, bed_end + ramp_length, 2.0 * radius, 0.0)
        ramp(
            "Release",
            bed_end + ramp_length,
            bed_end + ramp_length + float(parameters["release_m"]),
            0.0,
            0.0,
        )

    env.benchmark_track_parameters = parameters
