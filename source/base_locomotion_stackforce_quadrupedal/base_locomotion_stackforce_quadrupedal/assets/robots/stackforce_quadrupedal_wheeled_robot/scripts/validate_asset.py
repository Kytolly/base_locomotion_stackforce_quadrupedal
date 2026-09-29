#!/usr/bin/env python3
"""Validate the generated URDF tree, mesh inventory, and closure metadata."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


ASSET_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ASSET_ROOT / "urdf/stackforce_quadrupedal_wheeled_robot.urdf"
CONFIG_PATH = ASSET_ROOT / "config/closure_frames.json"
LEGS = ("FR", "FL", "RL", "RR")


def main() -> None:
    robot = ET.parse(URDF_PATH).getroot()
    links = {node.get("name") for node in robot.findall("link")}
    joints = robot.findall("joint")
    if len(links) != 29 or len(joints) != 28:
        raise RuntimeError("Expected 29 links and 28 tree joints")
    if len({j.get("name") for j in joints}) != len(joints):
        raise RuntimeError("Duplicate joint names")
    children = {}
    descendants = {name: [] for name in links}
    for joint in joints:
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        if child not in links or parent not in links:
            raise RuntimeError(f"{joint.get('name')}: missing parent/child link")
        descendants[parent].append(child)
        if child in children:
            raise RuntimeError(f"URDF is not a tree: {child} has multiple parents")
        children[child] = joint.get("name")
    roots = links - set(children)
    if roots != {"base_link"}:
        raise RuntimeError(f"Expected base_link as the only root, found {sorted(roots)}")
    visited = set()
    pending = ["base_link"]
    while pending:
        node = pending.pop()
        if node in visited:
            raise RuntimeError(f"Cycle at {node}")
        visited.add(node)
        pending.extend(descendants[node])
    if visited != links:
        raise RuntimeError(f"Disconnected links: {links - visited}")

    source = ET.parse(ASSET_ROOT.parent / "closed_link_robot/urdf/sf_robot.urdf").getroot()
    generated = {j.get("name"): j for j in joints}
    role_map = {"thigh_joint": "Outer_Hip_Joint", "calf_joint": "Outer_Knee_Joint", "foot_joint": "Wheel_Joint"}
    for original in source.findall("joint"):
        old_name = original.get("name")
        prefix, role = old_name.split("_", 1)
        canonical_name = f"{prefix}_{role_map.get(role, role)}"
        current = generated[canonical_name]
        if original.attrib != current.attrib:
            raise RuntimeError(f"Outer joint changed: {original.get('name')}")
        for tag in ("parent", "child", "origin", "axis", "limit", "dynamics", "mimic"):
            old, new = original.find(tag), current.find(tag)
            if (None if old is None else old.attrib) != (None if new is None else new.attrib):
                raise RuntimeError(f"Outer joint {original.get('name')} {tag} changed")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["source_validation_status"] != "PASS":
        raise RuntimeError("Source four-leg validation is not PASS")
    for leg in LEGS:
        entry = config["legs"][leg]
        required = {
            f"{leg}_Outer_Thigh_Link",
            f"{leg}_Outer_Calf_Link",
            f"{leg}_Foot_Link",
            f"{leg}_Inner_Thigh_Link",
            f"{leg}_Inner_Calf_Link",
            f"{leg}_Wheel_Frame",
            f"{leg}_Closure_Frame",
        }
        missing = required - links
        if missing:
            raise RuntimeError(f"{leg} is missing links: {sorted(missing)}")
        if entry["radial_residual_m"] > 2.0e-6:
            raise RuntimeError(f"{leg} Wheel/Closure radial residual is too large")

    mesh_paths = {ASSET_ROOT / mesh.get("filename").replace("../", "") for mesh in robot.findall(".//mesh")}
    for path in mesh_paths:
        if not path.is_file():
            raise RuntimeError(f"Referenced mesh is absent: {path}")
        if path.suffix.lower() == ".stl":
            with path.open("rb") as stream:
                stream.seek(80)
                count = struct.unpack("<I", stream.read(4))[0]
            if path.stat().st_size != 84 + count * 50:
                raise RuntimeError(f"Malformed binary STL: {path}")

    print(f"PASS: {len(links)} connected links, {len(joints)} joints, {len(mesh_paths)} meshes, 4 closure-frame pairs; original outer joints unchanged")


if __name__ == "__main__":
    main()
