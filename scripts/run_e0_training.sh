#!/usr/bin/env bash
set -Eeuo pipefail

# Unified E0 scheduler. It only supplies stage/seed/checkpoint overrides; all
# model, PPO, reward, terrain and logging settings remain in the YAML contracts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${E0_PYTHON:-/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python}" \
  "${SCRIPT_DIR}/e0_training_orchestrator.py" "$@"
