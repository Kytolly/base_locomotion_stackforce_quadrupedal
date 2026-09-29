"""StackForce four-wheel-leg asset contract used by the locomotion task."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg


def _resolve_asset_root() -> Path:
    """Resolve the packaged closed-loop asset, with an optional external override."""
    override = os.getenv("SF_QUAD_ASSET_ROOT")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(
        Path(__file__).resolve().parents[5]
        / "assets/robots/stackforce_quadrupedal_wheeled_robot"
    )
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "sf_quad" / "source/sf_quad/sf_quad/assets/robots/stackforce_quadrupedal_wheeled_robot")
    for candidate in candidates:
        if (candidate / "usd/stackforce_quadrupedal_wheeled_robot_closed.usda").is_file():
            return candidate
    raise FileNotFoundError(
        "The canonical sf_quad closed-loop asset was not found. Set SF_QUAD_ASSET_ROOT "
        "to sf_quad/source/sf_quad/sf_quad/assets/robots/stackforce_quadrupedal_wheeled_robot."
    )


ASSET_ROOT = _resolve_asset_root()
CLOSED_USD_PATH = ASSET_ROOT / "usd/stackforce_quadrupedal_wheeled_robot_closed.usda"

LEG_ORDER = ("FR", "FL", "RL", "RR")
OUTER_HIP_JOINTS = tuple(f"{leg}_Outer_Hip_Joint" for leg in LEG_ORDER)
INNER_HIP_JOINTS = tuple(f"{leg}_Inner_Hip_Joint" for leg in LEG_ORDER)
WHEEL_JOINTS = tuple(f"{leg}_Wheel_Joint" for leg in LEG_ORDER)
PASSIVE_JOINTS = tuple(
    name for leg in LEG_ORDER for name in (f"{leg}_Outer_Knee_Joint", f"{leg}_Inner_Knee_Joint")
)
CLOSURE_JOINTS = tuple(f"{leg}_Closure_Joint" for leg in LEG_ORDER)
ACTIVE_JOINTS = OUTER_HIP_JOINTS + INNER_HIP_JOINTS + WHEEL_JOINTS
LEG_JOINTS = OUTER_HIP_JOINTS + INNER_HIP_JOINTS
FOOT_LINKS = tuple(f"{leg}_Foot_Link" for leg in LEG_ORDER)


def _actuator(joints: tuple[str, ...], effort: float, velocity: float, stiffness: float) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=list(joints),
        effort_limit_sim=effort,
        velocity_limit_sim=velocity,
        stiffness=stiffness,
        damping=0.5,
        armature=0.0,
        friction=0.0,
    )


def create_robot_articulation_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Create the 12-active-joint closed-loop articulation configuration."""
    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path=None,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CLOSED_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.11),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "OUTER_SERVO": _actuator(OUTER_HIP_JOINTS, 0.392, 8.73, 20.0),
            "INNER_SERVO": _actuator(INNER_HIP_JOINTS, 0.392, 8.73, 20.0),
            "WHEEL": _actuator(WHEEL_JOINTS, 1.0, 40.0, 0.0),
        },
        soft_joint_pos_limit_factor=0.95,
        actuator_value_resolution_debug_print=True,
    )
