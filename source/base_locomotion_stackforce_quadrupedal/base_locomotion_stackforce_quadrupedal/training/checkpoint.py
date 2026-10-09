"""Preserve the policy and environment contract across checkpoint resumes."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


CONTRACT_FILENAME = "training_contract.json"
CONTRACT_SCHEMA_VERSION = 1


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", value.__class__.__module__)
    name = getattr(value, "__qualname__", value.__class__.__qualname__)
    return f"{module}:{name}"


def _normalise(value: Any) -> Any:
    """Convert configclass values into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type) or callable(value):
        return _qualified_name(value)
    if hasattr(value, "to_dict"):
        return _normalise(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalise(item) for item in value]
    return _qualified_name(value)


def build_training_contract(task: str, env_cfg: Any, agent_cfg: Any) -> dict[str, Any]:
    """Build the semantic contract that must remain fixed during PPO resume."""
    env = _normalise(env_cfg)
    agent = _normalise(agent_cfg)
    scene = dict(env["scene"])
    scene.pop("num_envs", None)
    environment = {
        "task": task,
        "sim": env["sim"],
        "decimation": env["decimation"],
        "episode_length_s": env["episode_length_s"],
        "scene": scene,
        "observations": env["observations"],
        "actions": env["actions"],
        "events": env["events"],
        "commands": env["commands"],
        "rewards": env["rewards"],
        "terminations": env["terminations"],
        "curriculum": env["curriculum"],
        "terrain_profile": env.get("terrain_profile"),
    }
    policy = {
        key: agent[key]
        for key in (
            "num_steps_per_env",
            "clip_actions",
            "obs_groups",
            "actor",
            "critic",
            "algorithm",
        )
        if key in agent
    }
    payload = {"environment": environment, "policy": policy}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def save_training_contract(run_dir: Path, contract: Mapping[str, Any]) -> Path:
    path = run_dir / CONTRACT_FILENAME
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def require_compatible_training_contract(
    checkpoint: Path,
    current_contract: Mapping[str, Any],
    *,
    allow_terrain_mix_transition: bool = False,
) -> None:
    """Reject legacy or semantically different checkpoints before loading."""
    path = checkpoint.parent / CONTRACT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint lacks {CONTRACT_FILENAME}: {path}. "
            "It remains valid for evaluation, but formal training must start a new run."
        )
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint contract schema: {saved.get('schema_version')!r}."
        )
    compatible = saved.get("sha256") == current_contract.get("sha256")
    if not compatible and allow_terrain_mix_transition:
        compatible = _without_terrain_mix(saved.get("payload")) == _without_terrain_mix(
            current_contract.get("payload")
        )
    if not compatible:
        raise ValueError(
            "Checkpoint training contract differs from the current environment, "
            "reward, command, curriculum, observation, action, or PPO contract. "
            "Start a new run with agent.resume=false."
        )


def _without_terrain_mix(payload: Any) -> Any:
    """Remove only the staged terrain-profile fields from a contract payload."""
    if isinstance(payload, Mapping):
        return {
            key: _without_terrain_mix(value)
            for key, value in payload.items()
            if key not in {"proportion", "terrain_profile"}
        }
    if isinstance(payload, list):
        return [_without_terrain_mix(value) for value in payload]
    return payload


def checkpoint_runner_config(checkpoint: Path, fallback: dict) -> dict:
    path = checkpoint.parent / "agent.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint requires its training architecture file: {path}")
    saved = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    result = dict(fallback)
    for key in ("actor", "critic", "obs_groups", "algorithm"):
        if key in saved:
            result[key] = saved[key]
    return result
