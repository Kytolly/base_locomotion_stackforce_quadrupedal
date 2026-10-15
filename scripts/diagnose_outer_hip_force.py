#!/usr/bin/env python3
"""Diagnose the static FL/RR outer-hip force seen in the G3 stance (short sims only).

Zero action, E0-v2 servo contract, default stance.  Phases:

* ``air``   : root held 0.35 m above the plane, feet free, 2 s hold;
* ``air0g`` : same with gravity off (any remaining hip torque is an internal fight);
* ``stand`` : free base on the plane, 3 s; adds per-wheel vertical contact force.

Per hip joint it logs target, actual, the implicit-actuator torque estimate
(``applied_torque``) and, when the PhysX view provides it, the solver's
projected joint force.  Per leg it logs the closure residual, the passive knee
angles against the contract and the foot position against the URDF five-bar
model.  Writes ``output/diagnostics/outer_hip_force_<tag>.json``.

    python scripts/diagnose_outer_hip_force.py --headless --solver-iters 8
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import sys
import traceback

_CUDA_LIB = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib"
if _CUDA_LIB.is_dir():
    for _name in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
        _path = _CUDA_LIB / _name
        if _path.is_file():
            ctypes.CDLL(str(_path), mode=ctypes.RTLD_GLOBAL)
    os.environ["LD_LIBRARY_PATH"] = f"{_CUDA_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--phase", choices=("air", "air0g", "stand"), required=True)
parser.add_argument("--solver-iters", type=int, default=8)
parser.add_argument("--seconds", type=float, default=2.0)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/diagnostics")
parser.add_argument("--fix-root", action="store_true",
                    help="air phases: fix the root link in PhysX (the default Rig re-teleports a free-falling root, "
                         "which loads no joint with gravity)")
parser.add_argument("--wheel-init-rad", type=float, default=0.0, help="initial wheel joint angle (hull phase)")
parser.add_argument("--friction", type=float, default=None, help="override ground static/dynamic friction")
parser.add_argument("--contact-points", action="store_true", help="stand: also report wheel-ground contact points")
parser.add_argument("--sample-steps", type=int, default=1,
                    help="physics steps per logged sample; 4 (= 0.02 s) aliases a period-2 solver jitter")
parser.add_argument("--foot-force-z", type=float, default=0.0,
                    help="air phases: constant upward world force [N] on every foot link (a ground reaction stand-in)")
parser.add_argument("--velocity-iters", type=int, default=None, help="override solver_velocity_iteration_count")
parser.add_argument("--check-wheel-settle", action="store_true",
                    help="stand: exit 1 unless wheel-centre spread <= 0.3 mm and diagonal twist <= 10%%")
parser.add_argument("--trace", action="store_true", help="print root/foot/wheel state every 0.5 s")
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fix_plan_common as common  # noqa: E402
from fix_plan_common import LEGS, Rig, write_json  # noqa: E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.env.robots.kinematics import (  # noqa: E402,E501
    LegKinematics,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E402,E501
    load_servo_contract,
)

STALL_NM = 0.392
KP = 20.0  # env/robots/stackforce.py hip servos
KD = 0.5
# Settled-stance acceptance for identical wheel colliders (tests/test_wheel_colliders.py).
SETTLE_SPREAD_MM = 0.3
SETTLE_TWIST = 0.10  # |(F_FL + F_RR) - (F_FR + F_RL)| / sum(F)


def make_env():
    """fix_plan_common.make_env with the solver-iteration and gravity switches."""
    import gymnasium as gym
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401

    suspended = args.phase != "stand"
    cfg = parse_env_cfg(common.TASK, device=args.device, num_envs=1)
    cfg.seed = 1001
    cfg.action_contract = "servo_range_v2"
    cfg.solver_position_iteration_count = args.solver_iters
    cfg.apply_action_contract(rebuild_terrain=False)
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1, env_spacing=1.0, debug_vis=False
    )
    cfg.scene.env_spacing = 1.0
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.motion_family = None
    cfg.curriculum.robustness_bin = None
    for name in ("randomize_contact_material", "randomize_base_mass", "randomize_actuator_gains"):
        setattr(cfg.events, name, None)
    cfg.events.reset_root.params["slope_approach_fraction"] = 0.0
    cfg.commands.locomotion.resampling_time_range = (1.0e9, 1.0e9)
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = 1.0e6
    if args.phase == "air0g":
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    if args.velocity_iters is not None:
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iters
    if args.contact_points:
        for leg in LEGS:
            sensor = getattr(cfg.scene, f"contact_{leg.lower()}")
            sensor.track_contact_points = True
            sensor.filter_prim_paths_expr = ["/World/ground"]
            sensor.max_contact_data_count_per_prim = 16
    if args.fix_root:
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        cfg.scene.robot.articulation_root_prim_path = "/Geometry"
    cfg.scene.robot.init_state.joint_pos[".*_Wheel_Joint"] = args.wheel_init_rad
    if args.friction is not None:
        material = cfg.scene.terrain.physics_material
        material.static_friction = material.dynamic_friction = args.friction
    if suspended:
        cfg.scene.robot.init_state.pos = (0.0, 0.0, common.SUSPENDED_ROOT_HEIGHT_M)
        cfg.terminations.base_height = None
        cfg.terminations.excessive_tilt = None
        cfg.terminations.base_collision = None
    print(f"[DIAG] robot usd: {cfg.scene.robot.spawn.usd_path}", flush=True)
    env = gym.make(common.TASK, cfg=cfg).unwrapped
    env.reset(seed=1001)
    return env, suspended


def projected_forces(robot, ids):
    try:
        forces = robot.root_physx_view.get_dof_projected_joint_forces()
        if not isinstance(forces, torch.Tensor):
            import warp as wp

            forces = wp.to_torch(forces)
        return forces[0, ids].clone()
    except Exception:  # noqa: BLE001 - optional API
        return None


def main() -> bool:
    contract = load_servo_contract()
    kin = LegKinematics()
    env, suspended = make_env()
    rig = Rig(env, suspended=suspended and not args.fix_root)
    robot = rig.robot
    knee_names = [f"{leg}_{b}_Knee_Joint" for leg in LEGS for b in ("Outer", "Inner")]
    knee_ids, _ = robot.find_joints(knee_names, preserve_order=True)
    leg_names = [robot.joint_names[i] for i in rig.leg_ids]
    wheel_ids, _ = robot.find_joints([f"{leg}_Wheel_Joint" for leg in LEGS], preserve_order=True)
    try:
        stance = rig.default_leg_pos
        if args.foot_force_z:
            forces = torch.zeros(1, len(rig.foot_ids), 3, device=env.device)
            forces[..., 2] = args.foot_force_z
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces, body_ids=torch.tensor(rig.foot_ids, dtype=torch.int32, device=env.device), is_global=True
            )
        rig.run(stance, 0.2, start=stance)
        steps = max(1, args.sample_steps)
        sample_dt = steps * rig.physics_dt
        samples = round(args.seconds / sample_dt)
        log = {k: [] for k in ("q", "qd", "knee_qd", "torque", "proj", "closure", "knee", "feet", "feet_w", "fz", "fxyz", "wheel_vel", "root_z")}
        for _ in range(samples):
            for _ in range(steps):
                rig.step_physics()
            data = robot.data
            log["q"].append(data.joint_pos[0, rig.leg_ids].clone())
            log["qd"].append(data.joint_vel[0, rig.leg_ids].clone())
            log["torque"].append(rig.leg_torque()[0].clone())
            proj = projected_forces(robot, rig.leg_ids)
            if proj is not None:
                log["proj"].append(proj.to(env.device))
            log["closure"].append(rig.closure_residual_mm()[0].clone())
            log["knee"].append(data.joint_pos[0, knee_ids].clone())
            log["knee_qd"].append(data.joint_vel[0, knee_ids].clone())
            log["feet"].append(rig.feet_base()[0].clone())
            log["feet_w"].append(data.body_pos_w[0, rig.foot_ids].clone() - env.scene.env_origins[0])
            log["root_z"].append(float(data.root_pos_w[0, 2]))
            if args.trace and len(log["root_z"]) % max(1, round(0.5 / sample_dt)) == 0:
                origin = env.scene.env_origins[0]
                feet_w = (data.body_pos_w[0, rig.foot_ids] - origin) * 1000.0
                print(f"[TRACE] t={len(log['root_z']) * sample_dt:.2f}s root(mm)={((data.root_pos_w[0] - origin) * 1000).tolist()} "
                      f"rootvel={data.root_lin_vel_w[0].tolist()} foot_y(mm)={feet_w[:, 1].tolist()} "
                      f"foot_vy={data.body_lin_vel_w[0, rig.foot_ids, 1].tolist()} "
                      f"wheel_w={data.joint_vel[0, wheel_ids].tolist()}", flush=True)
            if not suspended:
                log["fxyz"].append(torch.stack([
                    env.scene[f"contact_{leg.lower()}"].data.net_forces_w[0].sum(0) for leg in LEGS
                ]))
                log["wheel_vel"].append(data.joint_vel[0, wheel_ids].clone())
                log["fz"].append(torch.stack([
                    env.scene[f"contact_{leg.lower()}"].data.net_forces_w[0].sum(0)[2] for leg in LEGS
                ]))
        tail = max(1, round(1.0 / sample_dt))
        contact_points = {}
        if args.contact_points:
            for li, leg in enumerate(LEGS):
                data = env.scene[f"contact_{leg.lower()}"].data
                pos = torch.as_tensor(data.contact_pos_w)[0].reshape(-1, 3) - env.scene.env_origins[0]
                pos = pos[torch.isfinite(pos).all(-1)]
                centre = robot.data.body_pos_w[0, rig.foot_ids[li]] - env.scene.env_origins[0]
                contact_points[leg] = {
                    "points_m": pos.tolist(),
                    "foot_centre_m": centre.tolist(),
                    "centre_minus_point_mm": ((centre - pos) * 1000.0).tolist(),
                }

        def mean(key):
            return torch.stack(log[key][-tail:]).mean(0)

        target = robot.data.joint_pos_target[0, rig.leg_ids]
        q, tau = mean("q"), mean("torque")
        qd_mean = mean("qd")
        qd_fd = (log["q"][-1] - log["q"][-tail]) / ((tail - 1) * sample_dt)
        proj = mean("proj") if log["proj"] else None
        joints = {}
        for i, name in enumerate(leg_names):
            joints[name] = {
                "target_rad": float(target[i]),
                "actual_rad": float(q[i]),
                "error_rad": float(target[i] - q[i]),
                "torque_est_nm": float(tau[i]),
                "kp_times_mean_error_nm": float(KP * (target[i] - q[i])),
                "mean_reported_joint_vel_rad_s": float(qd_mean[i]),
                "finite_difference_joint_vel_rad_s": float(qd_fd[i]),
                "pd_drive_torque_nm": float(KP * (target[i] - q[i]) - KD * qd_mean[i]),
                "fraction_of_stall": float(tau[i].abs() / STALL_NM),
                "physx_projected_force_nm": None if proj is None else float(proj[i]),
            }
        knee = mean("knee")
        knee_qd = mean("knee_qd")
        feet = mean("feet")
        closure_mean = mean("closure")
        closure_max = torch.stack(log["closure"]).amax(0)
        legs = {}
        for li, leg in enumerate(LEGS):
            qo, qi = float(q[li]), float(q[li + 4])
            ko_model, ki_model, model_residual = kin.solve_knees(leg, qo, qi)
            _, wheel_model = kin.outer_chain(leg, qo, ko_model)
            to, ti = float(tau[li]), float(tau[li + 4])
            # Static drive torques for a vertical foot force F: tau_drive = -J^T (0, 0, F).
            def wheel(a, b):
                return kin.outer_chain(leg, a, kin.solve_knees(leg, a, b)[0])[1]
            h = 1.0e-5
            jac = [(wheel(qo + h, qi) - wheel(qo, qi)) / h, (wheel(qo, qi + h) - wheel(qo, qi)) / h]
            if log["fz"]:
                foot_fz = float(mean("fz")[li])
            else:
                foot_fz = args.foot_force_z
            predicted = [-float(column[2]) * foot_fz for column in jac]
            # Loop-closure velocity implied by the reported joint velocities (0 if the loop holds at velocity level).
            state = [qo, qi, float(knee[2 * li]), float(knee[2 * li + 1])]
            rates = [float(qd_mean[li]), float(qd_mean[li + 4]), float(knee_qd[2 * li]), float(knee_qd[2 * li + 1])]
            base_error = kin.closure_error(leg, *state)
            closure_rate = sum(
                (kin.closure_error(leg, *[v + (h if k == i else 0.0) for k, v in enumerate(state)]) - base_error) / h * rate
                for i, rate in enumerate(rates)
            )
            legs[leg] = {
                "reported_knee_vel_rad_s": rates[2:],
                "closure_rate_from_reported_vel_mm_s": float(np.linalg.norm(closure_rate) * 1000.0),
                "foot_vertical_force_n": foot_fz,
                "jacobian_static_drive_torque_nm": predicted,
                "kp_err_drive_torque_nm": [joints[leg_names[li]]["kp_times_mean_error_nm"],
                                           joints[leg_names[li + 4]]["kp_times_mean_error_nm"]],
                "closure_residual_mm_mean": float(closure_mean[li]),
                "closure_residual_mm_max": float(closure_max[li]),
                "outer_x_inner_torque_sign": "opposite" if to * ti < 0 else "same",
                "outer_knee_rad": float(knee[2 * li]),
                "inner_knee_rad": float(knee[2 * li + 1]),
                "contract_outer_knee_rad": contract["passive_knees"][f"{leg}_Outer_Knee_Joint"],
                "contract_inner_knee_rad": contract["passive_knees"][f"{leg}_Inner_Knee_Joint"],
                "model_knees_at_actual_hips_rad": [ko_model, ki_model],
                "model_closure_residual_mm": model_residual * 1000.0,
                "foot_base_m": feet[li].tolist(),
                "foot_world_m": mean("feet_w")[li].tolist(),
                "foot_vs_model_mm": float((feet[li, 1:].cpu() - torch.tensor(wheel_model[1:], dtype=torch.float32)).norm() * 1000.0),
            }
            if log["fz"]:
                legs[leg]["wheel_contact_fz_n"] = float(mean("fz")[li])
                legs[leg]["wheel_contact_force_world_n"] = mean("fxyz")[li].tolist()
                legs[leg]["wheel_joint_vel_rad_s"] = float(mean("wheel_vel")[li])
                legs[leg]["wheel_joint_vel_target_rad_s"] = float(robot.data.joint_vel_target[0, wheel_ids[li]])
                legs[leg]["wheel_applied_torque_nm"] = float(robot.data.applied_torque[0, wheel_ids[li]])
                legs[leg]["wheel_contact_fz_n_at_0.5s"] = float(log["fz"][min(len(log["fz"]) - 1, round(0.5 / sample_dt))][li])
        report = {
            "phase": args.phase,
            "solver_position_iterations": args.solver_iters,
            "solver_velocity_iterations": args.velocity_iters,
            "fix_root": args.fix_root,
            "wheel_init_rad": args.wheel_init_rad,
            "ground_friction": args.friction,
            "suspended": suspended,
            "hold_seconds": args.seconds,
            "sample_steps": steps,
            "averaged_over_last_s": 1.0,
            "root_z_m": sum(log["root_z"][-tail:]) / tail,
            "joints": joints,
            "legs": legs,
            "contact_points": contact_points,
        }
        if not suspended:
            heights = [legs[leg]["foot_world_m"][2] * 1000.0 for leg in LEGS]
            fz = {leg: legs[leg]["wheel_contact_fz_n"] for leg in LEGS}
            twist = abs(fz["FL"] + fz["RR"] - fz["FR"] - fz["RL"]) / max(sum(fz.values()), 1e-9)
            spread = max(heights) - min(heights)
            report["wheel_settle"] = {
                "centre_height_spread_mm": spread,
                "twist_fraction": twist,
                "limits": {"centre_height_spread_mm": SETTLE_SPREAD_MM, "twist_fraction": SETTLE_TWIST},
                "pass": bool(spread <= SETTLE_SPREAD_MM and twist <= SETTLE_TWIST),
            }
        tag = (f"{args.phase}_it{args.solver_iters}{'_fixed' if args.fix_root else ''}"
               f"{f'_F{args.foot_force_z:g}' if args.foot_force_z else ''}"
               f"{f'_v{args.velocity_iters}' if args.velocity_iters is not None else ''}{args.tag}")
        write_json(args.output_dir / f"outer_hip_force_{tag}.json", report)
        for name, row in joints.items():
            print(f"{name:22s} tgt={row['target_rad']:+.4f} q={row['actual_rad']:+.4f} "
                  f"err={row['error_rad']:+.5f} kp*err={row['kp_times_mean_error_nm']:+.4f} "
                  f"qd={row['mean_reported_joint_vel_rad_s']:+.4f} fd={row['finite_difference_joint_vel_rad_s']:+.5f} "
                  f"pd={row['pd_drive_torque_nm']:+.4f} tau={row['torque_est_nm']:+.4f} "
                  f"proj={row['physx_projected_force_nm']:+.4f}")
        for leg, row in legs.items():
            print(leg, "closure rate from reported velocities %.2f mm/s, knee qd %s" % (
                row["closure_rate_from_reported_vel_mm_s"], [round(v, 4) for v in row["reported_knee_vel_rad_s"]]))
            print(leg, "F=%.2f N  static outer/inner %+.4f %+.4f  kp*err %+.4f %+.4f  closure %.3f mm" % (
                row["foot_vertical_force_n"], *row["jacobian_static_drive_torque_nm"], *row["kp_err_drive_torque_nm"],
                row["closure_residual_mm_max"]))
        settle = report.get("wheel_settle")
        if settle is not None:
            print(f"[WHEEL-SETTLE] {'PASS' if settle['pass'] else 'FAIL'}: centre spread "
                  f"{settle['centre_height_spread_mm']:.3f} mm (<= {SETTLE_SPREAD_MM}), twist "
                  f"{settle['twist_fraction']:.3f} (<= {SETTLE_TWIST})", flush=True)
            return settle["pass"] or not args.check_wheel_settle
        return True
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        if not main():
            exit_code = 1
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
