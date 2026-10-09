"""Deployable history and terrain observation profiles for E0 ablations."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.configclass import configclass

from ..mdp.observation import hybrid
from .complex_env_cfg import BaseLocomotionComplexEnvCfg, ComplexSceneCfg, ObservationsCfg


@configclass
class TerrainProfileSceneCfg(ComplexSceneCfg):
    forward_terrain = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        ray_alignment="yaw",
        pattern_cfg=hybrid.HybridTerrainPatternCfg(),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class HistoryObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = 4
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerrainObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        terrain_height = ObsTerm(func=hybrid.forward_terrain_height)
        terrain_valid = ObsTerm(func=hybrid.forward_terrain_valid)

    policy: PolicyCfg = PolicyCfg()


@configclass
class HistoryTerrainObservationsCfg(TerrainObservationsCfg):
    @configclass
    class PolicyCfg(TerrainObservationsCfg.PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = 4
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseLocomotionHistoryEnvCfg(BaseLocomotionComplexEnvCfg):
    observations: HistoryObservationsCfg = HistoryObservationsCfg()
    observation_profile: str = "P1-history-4x50Hz"
    actor_observation_dim: int = 184


@configclass
class BaseLocomotionTerrainEnvCfg(BaseLocomotionComplexEnvCfg):
    scene: TerrainProfileSceneCfg = TerrainProfileSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=False
    )
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    observation_profile: str = "P2-terrain-15ray"
    actor_observation_dim: int = 76


@configclass
class BaseLocomotionHistoryTerrainEnvCfg(BaseLocomotionTerrainEnvCfg):
    observations: HistoryTerrainObservationsCfg = HistoryTerrainObservationsCfg()
    observation_profile: str = "P3-history-terrain"
    actor_observation_dim: int = 304


__all__ = [
    "BaseLocomotionHistoryEnvCfg",
    "BaseLocomotionHistoryTerrainEnvCfg",
    "BaseLocomotionTerrainEnvCfg",
]
