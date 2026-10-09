"""Fixed plateau and washboard benchmark environments."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from base_locomotion_stackforce_quadrupedal.benchmark import spawn_benchmark_track

from ..mdp import events
from .complex_env_cfg import BaseLocomotionComplexEnvCfg, ComplexSceneCfg, EventCfg


@configclass
class BenchmarkSceneCfg(ComplexSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=4.0,
        debug_vis=False,
    )


@configclass
class BenchmarkEventsCfg(EventCfg):
    randomize_contact_material = None
    randomize_base_mass = None
    randomize_actuator_gains = None
    fixed_command = EventTerm(
        func=events.set_fixed_benchmark_command,
        mode="reset",
        params={"forward_velocity": 0.28, "body_height": 0.105},
    )
    spawn_track = EventTerm(
        func=spawn_benchmark_track,
        mode="startup",
        params={"kind": "plateau", "seed": 8101, "randomized": False},
    )


@configclass
class PlateauBenchmarkEnvCfg(BaseLocomotionComplexEnvCfg):
    scene: BenchmarkSceneCfg = BenchmarkSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=False
    )
    events: BenchmarkEventsCfg = BenchmarkEventsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 35.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.scene.num_envs = 1
        self.curriculum.terrain_levels = None
        self.scene.support_scanner.update_period = self.decimation * self.sim.dt
        for sensor_name in ("contact_fr", "contact_fl", "contact_rl", "contact_rr", "base_contact"):
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        self.commands.locomotion.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.locomotion.ranges.forward_velocity_mps = (0.28, 0.28)
        self.commands.locomotion.ranges.lateral_velocity_mps = (0.0, 0.0)
        self.commands.locomotion.ranges.yaw_rate_radps = (0.0, 0.0)
        self.commands.locomotion.ranges.body_height_m = (0.105, 0.105)
        self.viewer.eye = (4.5, -3.0, 2.8)
        self.viewer.lookat = (0.0, 2.0, 0.25)


@configclass
class WashboardBenchmarkEnvCfg(PlateauBenchmarkEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.spawn_track.params = {
            "kind": "washboard",
            "seed": 8201,
            "randomized": False,
        }
