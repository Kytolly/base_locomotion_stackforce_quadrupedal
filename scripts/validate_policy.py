"""Evaluate a frozen policy on the fixed locomotion command and terrain suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher


TASK = "Base-Locomotion-Stackforce-Quadrupedal-Complex-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK)
policy_group = parser.add_mutually_exclusive_group(required=True)
policy_group.add_argument("--checkpoint", type=Path)
policy_group.add_argument("--zero-policy", action="store_true")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episode-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import base_locomotion_stackforce_quadrupedal.tasks  # noqa: F401,E402
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.metric import (  # noqa: E501,E402
    LocomotionEpisodeMetrics,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.evaluation.validation import (  # noqa: E501,E402
    DEFAULT_VALIDATION_SCENARIOS,
    build_validation_report,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.action import (  # noqa: E501,E402
    POLICY_ACTION_DIM,
)
from base_locomotion_stackforce_quadrupedal.tasks.manager_based.base_locomotion_stackforce_quadrupedal.mdp.policy import (  # noqa: E501,E402
    LOCOMOTION_COMMAND_NAME,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _base_height_flags(env) -> torch.Tensor:
    manager = env.termination_manager
    flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if "base_height" in manager._term_names:
        flags = manager._last_episode_dones[:, manager._term_names.index("base_height")]
    return flags


def _run_scenario(env, vec_env, policy, scenario) -> dict[str, torch.Tensor]:
    vec_env.reset()
    if policy is not None:
        policy.reset(torch.ones(env.num_envs, dtype=torch.long, device=env.device))
    accumulator = LocomotionEpisodeMetrics(env)
    accumulator.reset()
    command = torch.tensor(scenario.command, device=env.device).repeat(env.num_envs, 1)
    command_term = env.command_manager.get_term(LOCOMOTION_COMMAND_NAME)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    chunks: dict[str, list[torch.Tensor]] = {}

    for _ in range(args_cli.episode_steps + 1):
        command_term.set_command(command)
        observations = vec_env.get_observations()
        with torch.inference_mode():
            actions = (
                torch.zeros((env.num_envs, POLICY_ACTION_DIM), device=env.device)
                if policy is None
                else policy(observations)
            )
        accumulator.observe_state(actions)
        _, reward, dones, _ = vec_env.step(actions)
        accumulator.observe_reward(reward)
        terminated = env.reset_terminated.clone()
        truncated = env.reset_time_outs.clone()
        completed = active & dones.bool()
        if bool(completed.any()):
            env_ids = completed.nonzero(as_tuple=False).flatten()
            episode = accumulator.episode_metrics(
                env_ids, terminated, truncated, _base_height_flags(env)
            )
            episode["runtime/episode_complete"] = torch.ones(
                len(env_ids), device=env.device
            )
            for name, values in episode.items():
                chunks.setdefault(name, []).append(values.detach().cpu())
            active[env_ids] = False
        if policy is not None:
            policy.reset(dones)
        if not bool(active.any()):
            break

    if bool(active.any()):
        env_ids = active.nonzero(as_tuple=False).flatten()
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        truncated = torch.zeros_like(terminated)
        episode = accumulator.episode_metrics(
            env_ids, terminated, truncated, torch.zeros_like(terminated)
        )
        episode["runtime/episode_complete"] = torch.zeros(
            len(env_ids), device=env.device
        )
        for name, values in episode.items():
            chunks.setdefault(name, []).append(values.detach().cpu())

    metrics = {name: torch.cat(values) for name, values in chunks.items()}
    return metrics


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.episode_steps <= 0:
        raise ValueError("--num-envs and --episode-steps must be positive.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = (
        args_cli.episode_steps * float(env_cfg.sim.dt) * int(env_cfg.decimation)
    )
    env_cfg.commands.locomotion.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.__post_init__()
    env_cfg.episode_length_s = (
        args_cli.episode_steps * float(env_cfg.sim.dt) * int(env_cfg.decimation)
    )

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(base_env, clip_actions=1.0)
    checkpoint = (
        args_cli.checkpoint.resolve() if args_cli.checkpoint is not None else None
    )
    policy = None
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )
        agent_cfg.device = env_cfg.sim.device
        runner = OnPolicyRunner(
            vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env_cfg.sim.device)

    try:
        scenario_metrics = {
            scenario.name: _run_scenario(base_env.unwrapped, vec_env, policy, scenario)
            for scenario in DEFAULT_VALIDATION_SCENARIOS
        }
        metadata_block = {
            "task": args_cli.task,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "episode_steps": args_cli.episode_steps,
            "sim_dt_s": float(env_cfg.sim.dt),
            "policy_dt_s": float(base_env.unwrapped.step_dt),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "checkpoint_sha256": (
                _sha256(checkpoint) if checkpoint is not None else None
            ),
            "policy": (
                "checkpoint" if checkpoint is not None else "zero_policy_baseline"
            ),
            "ppo_updates": 0,
            "git_commit": _git_commit(),
        }
        report = build_validation_report(scenario_metrics, metadata_block)
        output = args_cli.output
        if output is None:
            run_name = checkpoint.stem if checkpoint is not None else "zero_policy"
            output = Path("output/evaluation") / run_name / f"seed_{args_cli.seed}.json"
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["selection_evidence"], indent=2))
        print(f"[VALIDATION] report={output}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
