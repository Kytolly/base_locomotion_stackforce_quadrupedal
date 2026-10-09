"""Reset events for the StackForce locomotion environment."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .policy import LOCOMOTION_COMMAND_NAME


def reset_root_to_default(
    env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, slope_approach_fraction: float = 0.0
) -> None:
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    terrain = env.scene.terrain
    if slope_approach_fraction > 0 and hasattr(terrain, "terrain_types"):
        from isaaclab.utils.warp import raycast_mesh
        from ..env.terrains import TERRAIN_FAMILIES

        selected = (terrain.terrain_types[env_ids] == TERRAIN_FAMILIES.index("hf_pyramid_slope")) & (
            torch.rand(len(env_ids), device=env.device) < slope_approach_fraction
        )
        if bool(selected.any()):
            sensor = env.scene["support_scanner"]
            _ = sensor.data
            mesh = sensor.meshes[(sensor.cfg.mesh_prim_paths[0], sensor.device)]
            starts = root_state[selected, :3].clone()
            starts[:, 1] -= 3.0
            starts[:, 2] += 2.0
            directions = torch.zeros_like(starts)
            directions[:, 2] = -1
            hits = raycast_mesh(starts, directions, mesh)[0]
            if not bool(torch.isfinite(hits).all()):
                raise RuntimeError("Slope approach reset did not find terrain support.")
            root_state[selected, :3] = hits
            root_state[selected, 2] += asset.data.default_root_state[env_ids[selected], 2]
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def activate_contact_reports(env, env_ids=None) -> None:
    """Enable contact reports on the nested base and wheel rigid bodies."""
    from isaaclab.sim import get_current_stage
    from pxr import PhysxSchema, UsdPhysics

    body_paths = (
        "Geometry/base_link",
        *(
            f"Geometry/base_link/{leg}_Outer_Thigh_Link/{leg}_Outer_Calf_Link/{leg}_Foot_Link"
            for leg in ("FR", "FL", "RL", "RR")
        ),
    )
    stage = get_current_stage()
    for env_index in range(env.num_envs):
        for body_path in body_paths:
            prim_path = f"/World/envs/env_{env_index}/Robot/{body_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Contact body is not a valid rigid body: '{prim_path}'.")
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr().Set(0.0)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)


def set_fixed_benchmark_command(
    env,
    env_ids: torch.Tensor,
    forward_velocity: float = 0.28,
    body_height: float = 0.105,
) -> None:
    """Set a fixed straight-line command for a benchmark episode."""
    command = torch.zeros((len(env_ids), 4), device=env.device)
    command[:, 0] = forward_velocity
    command[:, 3] = body_height
    env.command_manager.get_term(LOCOMOTION_COMMAND_NAME).set_command(command, env_ids)
