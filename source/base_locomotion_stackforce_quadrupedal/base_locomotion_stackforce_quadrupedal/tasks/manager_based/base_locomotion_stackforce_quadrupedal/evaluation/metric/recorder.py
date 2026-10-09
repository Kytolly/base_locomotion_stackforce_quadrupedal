"""Post-physics callbacks run before automatic reset and command resampling."""

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils.configclass import configclass


class LocomotionStepRecorder(RecorderTerm):
    def record_post_step(self):
        env = self._env
        robot = env.scene["robot"]
        command = env.command_manager.get_command("locomotion")
        if not hasattr(env, "motion_history"):
            env.motion_history = torch.zeros((env.num_envs, 5), device=env.device)
        velocity = robot.data.root_lin_vel_b[:, [1, 0]]
        speed = command[:, :2].norm(dim=-1)
        direction = command[:, :2] / speed.clamp_min(1e-6).unsqueeze(-1)
        env.motion_history[:, 0] += speed * env.step_dt
        env.motion_history[:, 1] += (velocity * direction).sum(-1) * env.step_dt
        env.motion_history[:, 2] += (velocity - command[:, :2]).square().sum(-1)
        env.motion_history[:, 3] += (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).square()
        env.motion_history[:, 4] += 1
        command_term = env.command_manager.get_term("locomotion")
        record_motion_step = getattr(command_term, "record_motion_step", None)
        if record_motion_step is not None:
            record_motion_step(
                (velocity - command[:, :2]).square().sum(-1),
                (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).square(),
            )
        for callback in getattr(env, "post_physics_callbacks", ()):
            callback()
        return None, None

    def record_post_reset(self, env_ids):
        if hasattr(self._env, "motion_history"):
            self._env.motion_history[slice(None) if env_ids is None else env_ids] = 0
        return None, None


@configclass
class LocomotionRecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    locomotion_step = RecorderTermCfg(class_type=LocomotionStepRecorder)
