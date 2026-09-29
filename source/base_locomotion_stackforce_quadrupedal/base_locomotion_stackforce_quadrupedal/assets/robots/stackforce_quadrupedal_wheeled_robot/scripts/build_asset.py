#!/usr/bin/env python3
"""Build the loop-cut URDF, local STL meshes, and imported Isaac Sim USD asset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom  # noqa: E402
from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig  # noqa: E402


ASSET_ROOT = Path(__file__).resolve().parents[1]
ROBOTS_ROOT = ASSET_ROOT.parent
SOURCE_ROOT = ROBOTS_ROOT / "closed_link_robot"
SOURCE_URDF = SOURCE_ROOT / "urdf/sf_robot.urdf"
SOURCE_STAGE = SOURCE_ROOT / "urdf/sf_robot/sf_robot.usda"
VALIDATION = SOURCE_ROOT / "ref_b_real_fr/four_inner_chains_validation.json"
URDF_PATH = ASSET_ROOT / "urdf/stackforce_quadrupedal_wheeled_robot.urdf"
USD_DIR = ASSET_ROOT / "usd"
CONFIG_PATH = ASSET_ROOT / "config/closure_frames.json"
MESH_DIR = ASSET_ROOT / "meshes"

LEGS = ("FR", "FL", "RL", "RR")
CHAIN_ROOT = "/RefB_C629_FourInnerChains"
ROBOT_BASE = "/tn__111_/Geometry/base_link"


def vec3(value) -> np.ndarray:
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def world_matrix(stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Required prim is absent: {path}")
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def transform_point(matrix: Gf.Matrix4d, value) -> np.ndarray:
    return vec3(matrix.Transform(Gf.Vec3d(*map(float, value))))


def transform_direction(matrix: Gf.Matrix4d, value) -> np.ndarray:
    result = vec3(matrix.TransformDir(Gf.Vec3d(*map(float, value))))
    return result / np.linalg.norm(result)


def read_attr(prim: Usd.Prim, name: str) -> np.ndarray:
    attr = prim.GetAttribute(name)
    if not attr.IsValid() or not attr.HasAuthoredValueOpinion():
        raise RuntimeError(f"Required attribute {name!r} is absent on {prim.GetPath()}")
    return vec3(attr.Get())


def mesh_world_triangles(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    if not points or not counts or not indices:
        raise RuntimeError(f"Mesh data is absent: {path}")
    matrix = world_matrix(stage, path)
    world_points = np.asarray([transform_point(matrix, point) for point in points])
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        for offset in range(1, count - 1):
            triangles.append(world_points[[face[0], face[offset], face[offset + 1]]])
    return np.asarray(triangles, dtype=np.float32)


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"Stackforce validated mesh".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = float(np.linalg.norm(normal))
            normal = normal / length if length > 1.0e-12 else np.zeros(3)
            stream.write(struct.pack("<12fH", *normal, *triangle.reshape(-1), 0))


def box_inertial(triangles: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    points = triangles.reshape(-1, 3).astype(np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = maximum - minimum
    center = 0.5 * (minimum + maximum)
    inertia = mass / 12.0 * np.asarray(
        [size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2]
    )
    return center, inertia


def append_inertial(link: ET.Element, triangles: np.ndarray, mass: float) -> None:
    center, inertia = box_inertial(triangles, mass)
    node = ET.SubElement(link, "inertial")
    ET.SubElement(node, "origin", {"xyz": fmt(center), "rpy": "0 0 0"})
    ET.SubElement(node, "mass", {"value": f"{mass:.8g}"})
    ET.SubElement(
        node,
        "inertia",
        {
            "ixx": f"{inertia[0]:.12g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{inertia[1]:.12g}",
            "iyz": "0",
            "izz": f"{inertia[2]:.12g}",
        },
    )


def append_mesh(link: ET.Element, filename: str, material: str) -> None:
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"../meshes/{filename}"})
        if tag == "visual":
            ET.SubElement(node, "material", {"name": material})


def append_joint(
    robot: ET.Element,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin,
    axis=None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "origin", {"xyz": fmt(origin), "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    if axis is not None:
        ET.SubElement(joint, "axis", {"xyz": fmt(axis)})
        ET.SubElement(joint, "limit", {"lower": "-3.14159265359", "upper": "3.14159265359", "effort": "0.392", "velocity": "8.73"})


def normalized_outer_meshes(robot: ET.Element) -> None:
    names = {"base_link": "base_link.stl"}
    for leg in LEGS:
        names.update(
            {
                f"{leg}_thigh_Link": f"{leg.lower()}_outer_upper.stl",
                f"{leg}_calf_Link": f"{leg.lower()}_outer_lower.stl",
                f"{leg}_foot_Link": f"{leg.lower()}_foot.stl",
            }
        )
    for link in robot.findall("link"):
        link_name = link.get("name")
        if link_name not in names:
            continue
        for mesh in link.findall(".//mesh"):
            mesh.set("filename", f"../meshes/{names[link_name]}")
        source_name = "base_link.STL" if link_name == "base_link" else f"{link_name}.STL"
        shutil.copy2(SOURCE_ROOT / "meshes" / source_name, MESH_DIR / names[link_name])


def build() -> Path:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError(f"Four-leg validation is not PASS: {VALIDATION}")

    stage = Usd.Stage.Open(str(SOURCE_STAGE))
    if stage is None:
        raise RuntimeError(f"Could not open source stage: {SOURCE_STAGE}")
    base_world = world_matrix(stage, ROBOT_BASE)
    world_to_base = base_world.GetInverse()

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    URDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    USD_DIR.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(SOURCE_URDF)
    robot = tree.getroot()
    robot.set("name", "stackforce_quadrupedal_wheeled_robot")
    normalized_outer_meshes(robot)
    ET.SubElement(robot, "material", {"name": "inner_upper_yellow"}).append(
        ET.Element("color", {"rgba": "1.0 0.55 0.0 1.0"})
    )
    ET.SubElement(robot, "material", {"name": "inner_lower_cyan"}).append(
        ET.Element("color", {"rgba": "0.0 0.85 1.0 1.0"})
    )

    closure_config = {
        "schema_version": 1,
        "robot_name": robot.get("name"),
        "source_validation": str(VALIDATION),
        "source_validation_status": "PASS",
        "joint_axis_token": "X",
        "closure_joint_scope": "closure_joints",
        "legs": {},
    }

    for leg in LEGS:
        root_path = f"{CHAIN_ROOT}/{leg}"
        leg_prim = stage.GetPrimAtPath(root_path)
        if not leg_prim.IsValid():
            raise RuntimeError(f"Validated leg is absent: {root_path}")

        inner_pivot = transform_point(world_to_base, read_attr(leg_prim, "refb:M2"))
        p2_upper = transform_point(world_to_base, read_attr(leg_prim, "refb:P2Upper"))
        p2_lower = transform_point(world_to_base, read_attr(leg_prim, "refb:P2Lower"))
        w1 = transform_point(world_to_base, read_attr(leg_prim, "refb:W1"))
        w2 = transform_point(world_to_base, read_attr(leg_prim, "refb:W2"))
        axis = transform_direction(world_to_base, read_attr(leg_prim, "refb:closureAxis"))

        upper_world = mesh_world_triangles(stage, f"{root_path}/InnerUpper/Mesh")
        lower_world = mesh_world_triangles(stage, f"{root_path}/FrontLower/Mesh")
        upper_base = np.asarray([[transform_point(world_to_base, point) for point in tri] for tri in upper_world])
        lower_base = np.asarray([[transform_point(world_to_base, point) for point in tri] for tri in lower_world])
        upper_local = upper_base - inner_pivot
        lower_local = lower_base - p2_upper

        prefix = leg.lower()
        upper_mesh = f"{prefix}_inner_upper.stl"
        lower_mesh = f"{prefix}_inner_lower.stl"
        write_binary_stl(MESH_DIR / upper_mesh, upper_local)
        write_binary_stl(MESH_DIR / lower_mesh, lower_local)

        upper_link_name = f"{leg}_Inner_Thigh_Link"
        lower_link_name = f"{leg}_Inner_Calf_Link"
        w1_link_name = f"{leg}_Wheel_Frame"
        w2_link_name = f"{leg}_Closure_Frame"

        upper_link = ET.SubElement(robot, "link", {"name": upper_link_name})
        append_inertial(upper_link, upper_local, 0.016)
        append_mesh(upper_link, upper_mesh, "inner_upper_yellow")
        append_joint(robot, f"{leg}_Inner_Hip_Joint", "revolute", "base_link", upper_link_name, inner_pivot, axis)

        lower_link = ET.SubElement(robot, "link", {"name": lower_link_name})
        append_inertial(lower_link, lower_local, 0.0176)
        append_mesh(lower_link, lower_mesh, "inner_lower_cyan")
        append_joint(robot, f"{leg}_Inner_Knee_Joint", "revolute", upper_link_name, lower_link_name, p2_upper - inner_pivot, axis)

        ET.SubElement(robot, "link", {"name": w1_link_name})
        foot_path = f"{ROBOT_BASE}/{leg}_thigh_Link/{leg}_calf_Link/{leg}_foot_Link"
        foot_to_base = world_matrix(stage, foot_path) * world_to_base
        w1_in_foot = transform_point(foot_to_base.GetInverse(), w1)
        append_joint(robot, f"{leg}_Wheel_Frame_Joint", "fixed", f"{leg}_Foot_Link", w1_link_name, w1_in_foot)

        ET.SubElement(robot, "link", {"name": w2_link_name})
        append_joint(robot, f"{leg}_Closure_Frame_Joint", "fixed", lower_link_name, w2_link_name, w2 - p2_upper)

        delta = w2 - w1
        radial = delta - np.dot(delta, axis) * axis
        closure_config["legs"][leg] = {
            "inner_lower_link": lower_link_name,
            "foot_link": f"{leg}_Foot_Link",
            "w1_frame": w1_link_name,
            "w2_frame": w2_link_name,
            "axis_in_base": axis.tolist(),
            "w1_in_base": w1.tolist(),
            "w2_in_base": w2.tolist(),
            "axial_offset_m": float(np.dot(delta, axis)),
            "radial_residual_m": float(np.linalg.norm(radial)),
            "p2_axis_offset_m": float(np.dot(p2_lower - p2_upper, axis)),
            "p2_surface_gap_m": float(validation["legs"][leg]["P2_surface_gap_mm"] / 1000.0),
        }

    ET.indent(tree, space="  ")
    tree.write(URDF_PATH, encoding="utf-8", xml_declaration=True)
    CONFIG_PATH.write_text(json.dumps(closure_config, indent=2) + "\n", encoding="utf-8")
    return URDF_PATH


def import_usd(urdf_path: Path) -> Path:
    config = URDFImporterConfig()
    config.urdf_path = str(urdf_path)
    config.usd_path = str(USD_DIR)
    config.merge_fixed_joints = False
    config.merge_mesh = False
    config.collision_from_visuals = False
    config.allow_self_collision = False
    config.fix_base = False
    config.joint_target_type = "none"
    importer = URDFImporter(config)
    output = importer.import_urdf()
    return Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-usd", action="store_true", help="Generate meshes and URDF without running the Isaac importer.")
    args = parser.parse_args()
    try:
        urdf_path = build()
        print(f"URDF: {urdf_path}")
        print(f"closure config: {CONFIG_PATH}")
        if not args.skip_usd:
            print(f"USD: {import_usd(urdf_path)}")
    finally:
        APP.close()


if __name__ == "__main__":
    main()
