"""Create the complex StackForce environment and execute a bounded zero-action rollout."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", default="Base-Locomotion-Stackforce-Quadrupedal-Complex-v0"
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument(
    "--force-timeout",
    action="store_true",
    help="Shorten the episode to the requested step count and verify training metric logging.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E501,E402
    LEG_ACTION_DIM,
    POLICY_ACTION_DIM,
    WHEEL_ACTION_DIM,
    split_policy_action,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.observation.privileged import (  # noqa: E501,E402
    PRIVILEGED_DIM,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.observation.history import (  # noqa: E501,E402
    HISTORY_DIM,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.observation.proprioception import (  # noqa: E501,E402
    PROPRIOCEPTION_DIM,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy import (  # noqa: E501,E402
    ACTOR_INPUT_DIM,
    LOCOMOTION_COMMAND_DIM,
    LOCOMOTION_COMMAND_NAME,
    split_actor_input,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric import (  # noqa: E501,E402
    TrainingMetricsWrapper,
)


COMPLEX_TASK = "Base-Locomotion-Stackforce-Quadrupedal-Complex-v0"


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    if args_cli.force_timeout:
        env_cfg.episode_length_s = (
            args_cli.steps * float(env_cfg.sim.dt) * int(env_cfg.decimation)
        )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.task == COMPLEX_TASK:
        env = TrainingMetricsWrapper(env)
    try:
        observations, _ = env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                observations, rewards, terminated, truncated, extras = env.step(actions)
        policy = observations["policy"]
        observation_finite = all(
            bool(torch.isfinite(value).all()) for value in observations.values()
        )
        if args_cli.task == COMPLEX_TASK:
            leg_action, wheel_action = split_policy_action(actions)
            command, proprioception, history = split_actor_input(policy)
            managed_command = env.unwrapped.command_manager.get_command(
                LOCOMOTION_COMMAND_NAME
            )
            if (
                policy.shape[-1] != ACTOR_INPUT_DIM
                or observations["privileged"].shape[-1] != PRIVILEGED_DIM
            ):
                raise RuntimeError(
                    "Complex observation contract mismatch: "
                    f"policy={tuple(policy.shape)}, privileged={tuple(observations['privileged'].shape)}."
                )
            if (
                command.shape[-1] != LOCOMOTION_COMMAND_DIM
                or proprioception.shape[-1] != PROPRIOCEPTION_DIM
                or history.shape[-1] != HISTORY_DIM
            ):
                raise RuntimeError(
                    "Actor input segment mismatch: "
                    f"command={tuple(command.shape)}, proprioception={tuple(proprioception.shape)}, "
                    f"history={tuple(history.shape)}."
                )
            if not torch.equal(command, managed_command):
                raise RuntimeError(
                    "Actor command segment differs from the Command Manager output."
                )
            if (
                leg_action.shape[-1] != LEG_ACTION_DIM
                or wheel_action.shape[-1] != WHEEL_ACTION_DIM
            ):
                raise RuntimeError(
                    f"Complex action contract mismatch: action={tuple(actions.shape)}."
                )
            if actions.shape[-1] != POLICY_ACTION_DIM:
                raise RuntimeError(
                    f"Expected {POLICY_ACTION_DIM} policy actions, received {actions.shape[-1]}."
                )
        print(
            "[VERIFY] PASS "
            f"envs={env.unwrapped.num_envs} action_shape={tuple(actions.shape)} "
            f"observation_shapes={{{', '.join(f'{name}: {tuple(value.shape)}' for name, value in observations.items())}}} "
            f"finite={observation_finite} "
            f"reward_mean={float(rewards.mean()):.6f} "
            f"terminated={int(terminated.sum())} truncated={int(truncated.sum())}"
        )
        if args_cli.task == COMPLEX_TASK:
            reward_terms = dict(
                env.unwrapped.reward_manager.get_active_iterable_terms(0)
            )
            if not all(
                torch.isfinite(torch.tensor(value)).all()
                for value in reward_terms.values()
            ):
                raise RuntimeError(f"Non-finite reward term detected: {reward_terms}.")
            print(
                "[VERIFY] CONTRACT "
                f"command={LOCOMOTION_COMMAND_DIM} proprioception={PROPRIOCEPTION_DIM} "
                f"history={HISTORY_DIM} actor_input={ACTOR_INPUT_DIM} action={POLICY_ACTION_DIM} "
                f"command_terms={env.unwrapped.command_manager.active_terms} "
                f"reward_terms={list(reward_terms)}"
            )
            if args_cli.force_timeout:
                training_keys = {
                    name for name in extras.get("log", {}) if name.startswith("train/")
                }
                required_prefixes = (
                    "train/locomotion/",
                    "train/safety/",
                    "train/actuation/",
                    "train/reward/",
                    "train/runtime/",
                )
                missing = [
                    prefix
                    for prefix in required_prefixes
                    if not any(name.startswith(prefix) for name in training_keys)
                ]
                if missing:
                    raise RuntimeError(
                        f"Training metric namespaces missing from extras['log']: {missing}."
                    )
                print(f"[VERIFY] TRAINING_METRICS keys={len(training_keys)}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
