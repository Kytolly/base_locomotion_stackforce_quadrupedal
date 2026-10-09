"""Dedicated flat-ground environment for stepping capability evaluation."""

from isaaclab.utils.configclass import configclass

from .complex_env_cfg import BaseLocomotionComplexEnvCfg, ComplexSceneCfg
from .terrains import get_flat_terrain_cfg


@configclass
class SteppingSceneCfg(ComplexSceneCfg):
    terrain = get_flat_terrain_cfg()


@configclass
class SteppingCapabilityEnvCfg(BaseLocomotionComplexEnvCfg):
    scene: SteppingSceneCfg = SteppingSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain = get_flat_terrain_cfg()
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 1
        self.episode_length_s = 30.0
        self.commands.locomotion.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.locomotion.ranges.forward_velocity_mps = (0.20, 0.20)
        self.commands.locomotion.ranges.lateral_velocity_mps = (0.0, 0.0)
        self.commands.locomotion.ranges.yaw_rate_radps = (0.0, 0.0)
        self.commands.locomotion.ranges.body_height_m = (0.105, 0.105)
        for name in ("randomize_contact_material", "randomize_base_mass", "randomize_actuator_gains"):
            setattr(self.events, name, None)


__all__ = ["SteppingCapabilityEnvCfg", "SteppingSceneCfg"]
