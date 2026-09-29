"""Manager-based StackForce task using the packaged robot and eight terrain families."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from ..mdp import events, terminations
from ..mdp.action import RobotActionCfg
from ..mdp.observation import history, privileged, proprioception
from ..mdp.policy import LocomotionCommandCfg
from ..mdp.policy import commands as command_mdp
from ..mdp.reward import locomotion as reward_mdp
from .robots import ACTIVE_JOINTS, create_robot_articulation_cfg
from .terrains import TERRAIN_FAMILIES, get_complex_terrain_cfg


@configclass
class ComplexSceneCfg(InteractiveSceneCfg):
    ground = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=get_complex_terrain_cfg(),
        max_init_terrain_level=None,
        collision_group=-1,
        debug_vis=False,
    )
    robot = create_robot_articulation_cfg()
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands = ObsTerm(func=command_mdp.locomotion_command)
        base_ang_vel = ObsTerm(func=proprioception.base_angular_velocity)
        projected_gravity = ObsTerm(func=proprioception.projected_gravity)
        joint_pos = ObsTerm(
            func=proprioception.active_joint_position,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))
            },
        )
        joint_vel = ObsTerm(
            func=proprioception.active_joint_velocity,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))
            },
        )
        previous_action = ObsTerm(func=history.previous_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=privileged.base_linear_velocity)
        base_height = ObsTerm(func=privileged.base_height_above_env_origin)
        applied_torque = ObsTerm(
            func=privileged.applied_joint_torque,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class CommandsCfg:
    locomotion = LocomotionCommandCfg()


@configclass
class EventCfg:
    reset_root = EventTerm(
        func=events.reset_root_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
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
        weight=1.5,
        params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_lateral_velocity = RewTerm(
        func=reward_mdp.track_lateral_velocity_exp,
        weight=0.5,
        params={"std": 0.20, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_yaw_rate = RewTerm(
        func=reward_mdp.track_yaw_rate_exp,
        weight=0.5,
        params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_body_height = RewTerm(
        func=reward_mdp.track_body_height_exp,
        weight=0.5,
        params={"std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
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
        weight=-0.05,
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


@configclass
class BaseLocomotionComplexEnvCfg(ManagerBasedRLEnvCfg):
    scene: ComplexSceneCfg = ComplexSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: RobotActionCfg = RobotActionCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    active_joint_names: list[str] = list(ACTIVE_JOINTS)
    terrain_families: tuple[str, ...] = TERRAIN_FAMILIES

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.scene.terrain.terrain_generator = get_complex_terrain_cfg(
            seed=42 if self.seed is None else int(self.seed)
        )
