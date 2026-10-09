"""Hybrid locomotion task with the 108-dimensional deployable Actor contract."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.configclass import configclass

from ..mdp.observation import hybrid
from .complex_env_cfg import BaseLocomotionComplexEnvCfg, ComplexSceneCfg, ObservationsCfg


@configclass
class HybridSceneCfg(ComplexSceneCfg):
    forward_terrain = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        ray_alignment="yaw",
        pattern_cfg=hybrid.HybridTerrainPatternCfg(),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class HybridObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        contact_flags = ObsTerm(func=hybrid.contact_flags)
        normal_forces = ObsTerm(func=hybrid.normal_forces)
        endpoint_positions = ObsTerm(func=hybrid.endpoint_positions)
        endpoint_velocities = ObsTerm(func=hybrid.endpoint_velocities)
        terrain_height = ObsTerm(func=hybrid.forward_terrain_height)
        terrain_valid = ObsTerm(func=hybrid.forward_terrain_valid)

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseLocomotionHybridEnvCfg(BaseLocomotionComplexEnvCfg):
    scene: HybridSceneCfg = HybridSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=False)
    observations: HybridObservationsCfg = HybridObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The first Hybrid run reached a reward plateau while held-out forward
        # and reverse commands still saturated the actions.  Keep the
        # curriculum and command distribution biased toward learning those
        # failure modes before exposing the policy to harder terrain.
        self.commands.locomotion.mode_probabilities = (
            0.08,  # stop
            0.24,  # forward
            0.20,  # reverse
            0.08,  # lateral
            0.12,  # yaw in place
            0.16,  # forward + yaw
            0.12,  # reverse + yaw
        )

        curriculum_params = self.curriculum.terrain_levels.params
        curriculum_params["tracking_rmse_range"] = (0.22, 0.14)
        curriculum_params["yaw_rmse_range"] = (0.22, 0.14)
        curriculum_params["progress_ratio_range"] = (0.75, 0.90)
        curriculum_params["demotion_progress_ratio"] = 0.40

        # Make command tracking and avoiding clipped actions visible in the
        # PPO objective instead of allowing terrain progress to dominate it.
        self.rewards.track_forward_velocity.weight = 3.5
        self.rewards.track_yaw_rate.weight = 1.5
        self.rewards.planar_velocity_error.weight = -5.0
        self.rewards.wrong_way_velocity.weight = -10.0
        self.rewards.action_saturation.weight = -1.0
        self.rewards.unsafe_termination.weight = -300.0


__all__ = ["BaseLocomotionHybridEnvCfg", "HybridSceneCfg"]
