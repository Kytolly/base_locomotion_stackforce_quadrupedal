"""Manager-based StackForce task using the packaged robot and eight terrain families."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg, SimulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_physx.physics import PhysxCfg

from ..mdp import curricula, events, support, terminations
from ..mdp.action import RobotActionCfg
from ..mdp.observation import history, privileged, proprioception
from ..mdp.policy import LocomotionCommandCfg
from ..mdp.policy import commands as command_mdp
from ..mdp.reward import locomotion as reward_mdp
from .robots import ACTIVE_JOINTS, create_robot_articulation_cfg
from .terrains import TRAINING_TERRAIN_FAMILIES, get_complex_terrain_cfg
from ..evaluation.metric.recorder import LocomotionRecordersCfg
from ..mdp.policy.input import ACTOR_INPUT_DIM


@configclass
class ComplexSceneCfg(InteractiveSceneCfg):
    ground = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=get_complex_terrain_cfg(),
        max_init_terrain_level=1,
        collision_group=-1,
        debug_vis=False,
    )
    robot = create_robot_articulation_cfg()
    support_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(0.45, 0.35)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    contact_fr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/FR_Outer_Thigh_Link/FR_Outer_Calf_Link/FR_Foot_Link",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        track_pose=True,
    )
    contact_fl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/FL_Outer_Thigh_Link/FL_Outer_Calf_Link/FL_Foot_Link",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        track_pose=True,
    )
    contact_rl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/RL_Outer_Thigh_Link/RL_Outer_Calf_Link/RL_Foot_Link",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        track_pose=True,
    )
    contact_rr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/RR_Outer_Thigh_Link/RR_Outer_Calf_Link/RR_Foot_Link",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        track_pose=True,
    )
    base_contact = ContactSensorCfg(
        # Anchor the regex so PhysX binds only the outer rigid body. The visual
        # instance is named ``base_link_visual`` and must not become a sensor body.
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link$",
        history_length=3,
        update_period=0.005,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands = ObsTerm(func=command_mdp.locomotion_command)
        base_ang_vel = ObsTerm(
            func=proprioception.base_angular_velocity,
            noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        )
        projected_gravity = ObsTerm(
            func=proprioception.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=proprioception.active_joint_position,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))
            },
            noise=UniformNoiseCfg(n_min=-0.002, n_max=0.002),
        )
        joint_vel = ObsTerm(
            func=proprioception.active_joint_velocity,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))
            },
            noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        )
        previous_action = ObsTerm(func=history.previous_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=privileged.base_linear_velocity)
        base_height = ObsTerm(func=support.body_height_above_support)
        applied_torque = ObsTerm(
            func=privileged.applied_joint_torque,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TrainingAuxiliaryCfg(ObsGroup):
        temporal_loss_valid = ObsTerm(func=command_mdp.temporal_loss_valid)
        previous_action = ObsTerm(func=history.previous_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    training_auxiliary: TrainingAuxiliaryCfg = TrainingAuxiliaryCfg()


@configclass
class CommandsCfg:
    locomotion = LocomotionCommandCfg()


@configclass
class EventCfg:
    activate_contact_reports = EventTerm(
        func=events.activate_contact_reports,
        mode="prestartup",
    )
    randomize_contact_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.65, 1.15),
            "dynamic_friction_range": (0.5, 0.95),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
        },
    )
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS)),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_root = EventTerm(
        func=events.reset_root_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "slope_approach_fraction": 0.5},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    track_forward_velocity = RewTerm(
        func=reward_mdp.track_forward_velocity_exp,
        weight=2.5,
        params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_lateral_velocity = RewTerm(
        func=reward_mdp.track_lateral_velocity_exp,
        weight=0.75,
        params={"std": 0.20, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_yaw_rate = RewTerm(
        func=reward_mdp.track_yaw_rate_exp,
        weight=0.75,
        params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_body_height = RewTerm(
        func=reward_mdp.track_body_height_exp,
        weight=0.5,
        params={"std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    directed_planar_progress = RewTerm(
        func=reward_mdp.directed_planar_progress,
        weight=2.0,
        params={"command_deadband": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    planar_velocity_error = RewTerm(
        func=reward_mdp.planar_velocity_error_l2,
        # Strong enough to suppress the forward-only solution on stop/reverse commands.
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    wrong_way_velocity = RewTerm(
        func=reward_mdp.wrong_way_velocity_ratio_l2,
        # A reverse response must be more expensive than simply standing still.
        weight=-8.0,
        params={"command_deadband": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    low_body_height_margin = RewTerm(
        func=reward_mdp.low_body_height_margin_l2,
        weight=-1.0,
        params={
            "warning_height": 0.08,
            "margin": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    orientation = RewTerm(
        func=reward_mdp.orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    mechanical_power = RewTerm(
        func=reward_mdp.mechanical_power,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_position_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))},
    )
    joint_velocity_limit = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.05,
        params={
            "soft_ratio": 0.9,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS)),
        },
    )
    action_saturation = RewTerm(
        func=reward_mdp.action_saturation,
        # Prevent the policy from relying on clipped wheel/leg commands on hard terrain.
        weight=-0.5,
        params={"threshold": 0.95},
    )
    unsafe_termination = RewTerm(func=mdp.is_terminated, weight=-250.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=terminations.base_height_failure,
        params={"minimum_height": 0.06, "asset_cfg": SceneEntityCfg("robot")},
    )
    excessive_tilt = DoneTerm(
        func=terminations.excessive_tilt,
        params={"maximum_tilt_rad": 1.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_collision = DoneTerm(
        func=terminations.base_collision,
        params={"threshold": 8.0, "sensor_cfg": SceneEntityCfg("base_contact")},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=curricula.terrain_levels_by_episode_performance,
        params={
            "tracking_rmse_range": (0.30, 0.18),
            "yaw_rmse_range": (0.35, 0.20),
            "progress_ratio_range": (0.60, 0.80),
            "demotion_progress_ratio": 0.25,
        },
    )
    motion_family = CurrTerm(
        func=curricula.motion_family_rehearsal,
        params={
            "success_tracking_rmse": 0.30,
            "success_yaw_rmse": 0.35,
            "sampling_floor": 0.05,
        },
    )
    robustness_bin = CurrTerm(
        func=curricula.robustness_bin_schedule,
        params={"nominal_level": 0, "moderate_level": 2, "stress_level": 5},
    )


@configclass
class BaseLocomotionComplexEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    scene: ComplexSceneCfg = ComplexSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: RobotActionCfg = RobotActionCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    recorders: LocomotionRecordersCfg = LocomotionRecordersCfg()
    active_joint_names: list[str] = list(ACTIVE_JOINTS)
    terrain_profile: str = "union_expansion"
    terrain_families: tuple[str, ...] = TRAINING_TERRAIN_FAMILIES
    observation_profile: str = "P0"
    protocol_version: str = "E0-v1-46d"
    actor_observation_dim: int = ACTOR_INPUT_DIM

    def __post_init__(self) -> None:
        if self.observation_profile == "P0" and self.actor_observation_dim != 46:
            raise RuntimeError(
                f"The E0 P0 task requires the frozen 46D Actor contract, got {self.actor_observation_dim}."
            )
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.scene.support_scanner.update_period = self.decimation * self.sim.dt
        for sensor_name in ("contact_fr", "contact_fl", "contact_rl", "contact_rr", "base_contact"):
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        self.scene.terrain.terrain_generator = get_complex_terrain_cfg(
            seed=42 if self.seed is None else int(self.seed),
            profile=self.terrain_profile,
        )
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
