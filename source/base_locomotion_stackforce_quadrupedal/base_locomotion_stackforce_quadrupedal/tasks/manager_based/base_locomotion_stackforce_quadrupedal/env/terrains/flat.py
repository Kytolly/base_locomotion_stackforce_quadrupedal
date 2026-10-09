"""Flat-ground terrain configuration for locomotion capability tests."""

from isaaclab.terrains import TerrainImporterCfg


def get_flat_terrain_cfg() -> TerrainImporterCfg:
    """Return a deterministic infinite plane with no procedural terrain."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=4.0,
        debug_vis=False,
    )


__all__ = ["get_flat_terrain_cfg"]
