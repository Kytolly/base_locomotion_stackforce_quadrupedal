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

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..mdp import curricula, events, support, terminations
from ..mdp.action import (
    LEGACY_ACTION_CONTRACT,
    SERVO_ACTION_CONTRACT,
    RobotActionCfg,
    legacy_leg_position_cfg,
    load_servo_contract,
    servo_leg_position_cfg,
)
from ..mdp.action import servo_range
from ..mdp.observation import history, privileged, proprioception
from ..mdp.observation.privileged import terms as privileged_terms
from ..mdp.observation.proprioception import terms as proprioception_terms
from ..mdp.policy import LocomotionCommandCfg
from ..mdp.policy import commands as command_mdp
from ..mdp.reward import locomotion as reward_mdp
from .robots import ACTIVE_JOINTS, LEG_JOINTS, create_robot_articulation_cfg
from .robots.stackforce import SOLVER_POSITION_ITERATIONS
from .terrains import LEGACY_OBSTACLE_HEIGHTS, TRAINING_TERRAIN_FAMILIES, get_complex_terrain_cfg
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
        # True hip state; set only when the actor sees the servo model (policy_joint_observation).
        true_leg_joint_pos: ObsTerm | None = None
        true_leg_joint_vel: ObsTerm | None = None

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
    # E0-v2: servo-contract hip limits (set by apply_action_contract; None for E0-v1).
    servo_joint_limits: EventTerm | None = None
    # T1: start resumed stages near the previous stage's level (no-op when
    # ``terrain_init_level`` is None).
    initial_terrain_levels = EventTerm(func=events.set_initial_terrain_levels, mode="startup")
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
    # E0-v2-46d: DS041MG servo-travel leg ranges and stance (plan G4).
    # legacy_symmetric reproduces E0-v1 (P0 control group, old checkpoints).
    action_contract: str = SERVO_ACTION_CONTRACT
    protocol_version: str = "E0-v2-46d"
    actor_observation_dim: int = ACTOR_INPUT_DIM
    # Values derived from the action contract (written by apply_action_contract).
    nominal_body_height_m: float = servo_range.LEGACY_NOMINAL_BODY_HEIGHT_M
    obstacle_height_ranges: dict = dict(LEGACY_OBSTACLE_HEIGHTS)
    # T1: initial terrain level band [m - 1, m] for resumed stages (None = importer default).
    terrain_init_level: int | None = None
    # Explicit obstacle height ranges overriding the contract's (G6 probe terrain only).
    obstacle_height_override: dict | None = None
    # G5 throughput switches; defaults are the E0-v1 behaviour.
    enabled_self_collisions: bool = True  # T2
    solver_position_iteration_count: int = SOLVER_POSITION_ITERATIONS  # T3
    # Leg target slew limit (rad per control step, servo_range_v2 only); None = off.
    # DS041MG: 8.73 rad/s x 0.02 s = 0.175 rad.  See mdp/action/asymmetric.py.
    leg_target_max_step_rad: float | None = None
    # Actor joint channels: "measured" = simulated joint state (E0-v1/v2 so far);
    # "servo_model" = what the robot can compute (hip servo-model estimate, commanded
    # wheel velocity), with true hip state moved to the critic.  servo_range_v2 only.
    policy_joint_observation: str = "measured"
    servo_model_delay_steps: int = 0
    foot_track_pose: bool = True  # T4
    foot_contact_update_period: float | None = None  # T6 (None = physics dt)

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
        self.apply_action_contract(rebuild_terrain=False)
        self.scene.terrain.terrain_generator = get_complex_terrain_cfg(
            seed=42 if self.seed is None else int(self.seed),
            profile=self.terrain_profile,
            obstacle_heights=self.obstacle_height_ranges,
        )
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None

    def _apply_policy_joint_observation(self) -> None:
        """Point the actor's joint channels at simulated state or at the deployable servo model."""
        mode = str(self.policy_joint_observation)
        if mode not in ("measured", "servo_model"):
            raise ValueError(f"Unknown policy_joint_observation {mode!r}; expected 'measured' or 'servo_model'.")
        if mode == "servo_model" and self.action_contract != SERVO_ACTION_CONTRACT:
            raise ValueError("policy_joint_observation='servo_model' needs the servo_range_v2 leg action term.")
        if int(self.servo_model_delay_steps) < 0:
            raise ValueError("servo_model_delay_steps must be non-negative.")
        policy, critic = self.observations.policy, self.observations.privileged
        if policy is None or getattr(policy, "joint_pos", None) is None:
            return  # probes may strip the actor group
        if mode == "servo_model":
            policy.joint_pos.func, policy.joint_pos.params = proprioception_terms.servo_model_joint_position, {}
            policy.joint_vel.func, policy.joint_vel.params = proprioception_terms.servo_model_joint_velocity, {}
            legs = SceneEntityCfg("robot", joint_names=list(LEG_JOINTS), preserve_order=True)
            critic.true_leg_joint_pos = ObsTerm(func=privileged_terms.true_joint_position, params={"asset_cfg": legs})
            critic.true_leg_joint_vel = ObsTerm(func=privileged_terms.true_joint_velocity, params={"asset_cfg": legs})
        else:
            active = {"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_JOINTS))}
            policy.joint_pos.func, policy.joint_pos.params = proprioception.active_joint_position, dict(active)
            policy.joint_vel.func, policy.joint_vel.params = proprioception.active_joint_velocity, dict(active)
            critic.true_leg_joint_pos = None
            critic.true_leg_joint_vel = None

    def apply_action_contract(self, rebuild_terrain: bool = True) -> None:
        """Write every field that depends on ``action_contract``; safe to call repeatedly.

        servo_range_v2 (E0-v2): asymmetric per-joint leg ranges, the new stance and
        its root height, body-height command range and height thresholds scaled
        with the stance, servo hard limits at startup, and step/box heights
        capped by the contract.  legacy_symmetric restores the E0-v1 values.
        """
        if self.action_contract == SERVO_ACTION_CONTRACT:
            contract = load_servo_contract()
            heights = contract["body_height"]
            joint_pos: dict[str, float] = {
                name: float(joint["default"]) for name, joint in contract["joints"].items()
            }
            joint_pos.update({name: float(value) for name, value in contract["passive_knees"].items()})
            joint_pos[".*_Wheel_Joint"] = 0.0
            self.actions.leg_position = servo_leg_position_cfg(contract)
            self.actions.leg_position.max_target_step_rad = (
                None if self.leg_target_max_step_rad is None else float(self.leg_target_max_step_rad)
            )
            self.actions.leg_position.servo_model_delay_steps = int(self.servo_model_delay_steps)
            self.events.servo_joint_limits = EventTerm(
                func=events.write_servo_joint_limits,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "lower": dict(self.actions.leg_position.lower),
                    "upper": dict(self.actions.leg_position.upper),
                },
            )
            root_height = float(heights["init_root_m"])
            nominal = float(heights["nominal_m"])
            command_range = tuple(float(value) for value in heights["command_range_m"])
            warning, minimum = float(heights["warning_m"]), float(heights["minimum_m"])
            obstacles = {name: tuple(bounds) for name, bounds in contract["obstacles"]["height_ranges_m"].items()}
            self.protocol_version = "E0-v2-46d"
        elif self.action_contract == LEGACY_ACTION_CONTRACT:
            joint_pos = {".*": 0.0}
            self.actions.leg_position = legacy_leg_position_cfg()
            self.events.servo_joint_limits = None
            root_height = servo_range.LEGACY_INIT_ROOT_HEIGHT_M
            nominal = servo_range.LEGACY_NOMINAL_BODY_HEIGHT_M
            command_range = servo_range.LEGACY_BODY_HEIGHT_RANGE_M
            warning = servo_range.LEGACY_LOW_BODY_WARNING_M
            minimum = servo_range.LEGACY_MINIMUM_BODY_HEIGHT_M
            obstacles = dict(LEGACY_OBSTACLE_HEIGHTS)
            self.protocol_version = "E0-v1-46d"
        else:
            raise ValueError(
                f"Unknown action_contract {self.action_contract!r}; expected "
                f"{LEGACY_ACTION_CONTRACT!r} or {SERVO_ACTION_CONTRACT!r}."
            )
        self._apply_policy_joint_observation()
        robot = self.scene.robot
        robot.init_state.joint_pos = joint_pos
        robot.init_state.pos = (0.0, 0.0, root_height)
        robot.spawn.articulation_props.enabled_self_collisions = bool(self.enabled_self_collisions)
        robot.spawn.articulation_props.solver_position_iteration_count = int(self.solver_position_iteration_count)
        self.nominal_body_height_m = nominal
        self.commands.locomotion.ranges.body_height_m = command_range
        # Probes may remove these terms before re-applying the contract.
        if getattr(self.rewards, "low_body_height_margin", None) is not None:
            self.rewards.low_body_height_margin.params["warning_height"] = warning
        if getattr(self.terminations, "base_height", None) is not None:
            self.terminations.base_height.params["minimum_height"] = minimum
        self.obstacle_height_ranges = dict(obstacles)
        if self.obstacle_height_override:
            self.obstacle_height_ranges.update(
                {name: tuple(bounds) for name, bounds in self.obstacle_height_override.items()}
            )
        foot_period = self.foot_contact_update_period
        for sensor_name in ("contact_fr", "contact_fl", "contact_rl", "contact_rr"):
            sensor = getattr(self.scene, sensor_name, None)
            if sensor is None:
                continue
            sensor.track_pose = bool(self.foot_track_pose)
            sensor.update_period = self.sim.dt if foot_period is None else float(foot_period)
        generator = getattr(self.scene.terrain, "terrain_generator", None)
        if rebuild_terrain and generator is not None and "pyramid_stairs" in getattr(generator, "sub_terrains", {}):
            curriculum = generator.curriculum
            self.scene.terrain.terrain_generator = get_complex_terrain_cfg(
                seed=int(generator.seed if generator.seed is not None else 42),
                num_rows=int(generator.num_rows),
                num_cols=int(generator.num_cols),
                profile=self.terrain_profile,
                obstacle_heights=self.obstacle_height_ranges,
            )
            self.scene.terrain.terrain_generator.curriculum = curriculum


ENV_OPTION_FIELDS = {
    "action_contract": "action_contract",
    "terrain_init_level": "terrain_init_level",
    "obstacle_heights": "obstacle_height_override",
    "leg_target_max_step_rad": "leg_target_max_step_rad",
    "policy_joint_observation": "policy_joint_observation",
    "servo_model_delay_steps": "servo_model_delay_steps",
}
THROUGHPUT_OPTION_FIELDS = {
    "enabled_self_collisions": "enabled_self_collisions",
    "solver_position_iteration_count": "solver_position_iteration_count",
    "foot_track_pose": "foot_track_pose",
    "foot_contact_update_period": "foot_contact_update_period",
}


def apply_e0_env_options(env_cfg: Any, options: Mapping[str, Any] | None) -> None:
    """Apply the YAML ``env`` block's E0-v2 options to a constructed env config.

    Reads ``action_contract``, ``terrain_init_level`` and ``throughput.*``
    (``replicate_physics`` goes to the scene).  Other ``env`` keys are handled
    by the caller.  Re-applies the action contract afterwards so every derived
    field (stance, heights, limits, obstacles) stays consistent.
    """
    options = dict(options or {})
    if not hasattr(env_cfg, "apply_action_contract"):
        return
    for key, field in ENV_OPTION_FIELDS.items():
        if key in options:
            setattr(env_cfg, field, options[key])
    throughput = dict(options.get("throughput") or {})
    for key, field in THROUGHPUT_OPTION_FIELDS.items():
        if key in throughput:
            setattr(env_cfg, field, throughput[key])
    if "replicate_physics" in throughput:
        env_cfg.scene.replicate_physics = bool(throughput["replicate_physics"])
    env_cfg.apply_action_contract()


def checkpoint_env_options(checkpoint: str | Path | None) -> dict[str, Any]:
    """Env options a checkpoint was trained with, for evaluation scripts.

    Reads the run's ``resolved_training_config.yaml``; runs from before E0-v2
    have no ``env.action_contract`` and used the legacy symmetric legs.
    Without a checkpoint the current (E0-v2) defaults apply.
    """
    from base_locomotion_stackforce_quadrupedal.training.checkpoint import checkpoint_action_contract

    if checkpoint is None:
        return {}
    options: dict[str, Any] = {"action_contract": checkpoint_action_contract(Path(checkpoint))}
    config_path = Path(checkpoint).parent / "resolved_training_config.yaml"
    if config_path.is_file():
        from omegaconf import OmegaConf

        resolved = OmegaConf.load(config_path)
        throughput = OmegaConf.select(resolved, "env.throughput", default=None)
        if throughput is not None:
            options["throughput"] = OmegaConf.to_container(throughput, resolve=True)
        # How actions reach the servos and what the actor observed are part of the policy.
        options["leg_target_max_step_rad"] = OmegaConf.select(resolved, "env.leg_target_max_step_rad", default=None)
        for key, default in (("policy_joint_observation", "measured"), ("servo_model_delay_steps", 0)):
            options[key] = OmegaConf.select(resolved, f"env.{key}", default=default)
    return options
