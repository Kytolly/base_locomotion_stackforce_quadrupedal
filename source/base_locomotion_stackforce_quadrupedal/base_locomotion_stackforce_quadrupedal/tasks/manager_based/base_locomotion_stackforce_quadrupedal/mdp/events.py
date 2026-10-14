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
    body_height: float | None = None,
) -> None:
    """Set a fixed straight-line command for a benchmark episode.

    ``body_height=None`` uses the action contract's nominal height
    (``env.cfg.nominal_body_height_m``): 0.105 m for E0-v1, the new stance for E0-v2.
    """
    if body_height is None:
        body_height = float(getattr(env.cfg, "nominal_body_height_m", 0.105))
    command = torch.zeros((len(env_ids), 4), device=env.device)
    command[:, 0] = forward_velocity
    command[:, 3] = body_height
    env.command_manager.get_term(LOCOMOTION_COMMAND_NAME).set_command(command, env_ids)


def write_servo_joint_limits(
    env, env_ids, asset_cfg: SceneEntityCfg, lower: dict[str, float], upper: dict[str, float]
) -> None:
    """E0-v2 startup: write the servo-contract hip ranges as simulated hard limits.

    The asset's passive knees use +-pi. Active hip limits come from the DS041MG
    contract and are written here without rebuilding the USD (plan G4). Soft
    limits keep ``soft_joint_pos_limit_factor``.
    """
    asset = env.scene[asset_cfg.name]
    names = list(lower)
    joint_ids, joint_names = asset.find_joints(names, preserve_order=True)
    limits = torch.tensor(
        [[lower[name], upper[name]] for name in joint_names], dtype=torch.float32, device=env.device
    ).unsqueeze(0).repeat(env.num_envs, 1, 1)
    writer = getattr(asset, "write_joint_position_limit_to_sim", None)
    if writer is None:
        raise RuntimeError("Articulation has no write_joint_position_limit_to_sim(); update the E0-v2 limit event.")
    try:
        writer(limits, joint_ids=joint_ids, warn_limit_violation=False)
    except TypeError:  # older signature without warn_limit_violation
        writer(limits, joint_ids=joint_ids)
    data = asset.data
    soft = getattr(data, "soft_joint_pos_limits", None)
    factor = float(getattr(asset.cfg, "soft_joint_pos_limit_factor", 1.0))
    if isinstance(soft, torch.Tensor):
        centre = limits.mean(dim=-1)
        half = 0.5 * (limits[..., 1] - limits[..., 0]) * factor
        soft[:, joint_ids, 0] = centre - half
        soft[:, joint_ids, 1] = centre + half
    written = getattr(data, "joint_pos_limits", None)
    if isinstance(written, torch.Tensor) and not torch.allclose(
        written[:, joint_ids], limits, atol=1e-4
    ):
        raise RuntimeError("Servo joint limits were not applied to the simulation.")


def set_initial_terrain_levels(env, env_ids) -> None:
    """T1: start every environment in the level band ``[m - 1, m]``.

    ``m`` is ``env.cfg.terrain_init_level`` (set by the stage orchestrator from
    the previous stage's median level).  ``None`` keeps the importer's default
    ``[0, max_init_terrain_level]`` assignment.  Runs once at startup; the
    performance curriculum moves levels afterwards as before.
    """
    level = getattr(env.cfg, "terrain_init_level", None)
    terrain = env.scene.terrain
    if level is None or getattr(terrain, "terrain_levels", None) is None:
        return
    if getattr(terrain, "terrain_origins", None) is None:
        return
    max_level = int(terrain.max_terrain_level) - 1
    high = max(0, min(int(level), max_level))
    low = max(0, high - 1)
    count = terrain.terrain_levels.shape[0]
    levels = torch.randint(low, high + 1, (count,), device=terrain.terrain_levels.device)
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
