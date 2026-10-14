#!/usr/bin/env bash
# G8 formal long run for the E0-v2-46d contract.
#
# Preflight (refuses to start unless every item passes):
#   1. the experiment config resolves and carries protocol_version=E0-v2-46d;
#   2. gate evidence output/fix_plan/G{0..7}/gate.json reports PASS (G3 may be
#      CONDITIONAL: 25-30 mm lift with capped steps),
#      and G4 records that the contract upgrade was confirmed;
#   3. no other compute process is using the GPU (T8);
#   4. no previous E0-v2 run of the same experiment/seed exists, so the run
#      really starts from Foundation without an old checkpoint.
#
# The run itself is the four-stage orchestrator with T1 level inheritance,
# G8 stage-end gates and G8 abort monitoring enabled.  It is started detached;
# the command prints where to follow it.
#
# Usage:
#   bash scripts/run_e0_v2_long_run.sh                      # b2_bisec_full, seed 42
#   bash scripts/run_e0_v2_long_run.sh --with-b0-control    # then b0_t1_ppo, serially
#   bash scripts/run_e0_v2_long_run.sh --foreground
#   bash scripts/run_e0_v2_long_run.sh --resume-after expansion   # after a reviewed gate pause
#   bash scripts/run_e0_v2_long_run.sh --preflight-only
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${E0_PYTHON:-/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python}"
PROTOCOL="E0-v2-46d"
EXPERIMENT="b2_bisec_full"
SEED=42
CONTROL=""
FOREGROUND=0
PREFLIGHT_ONLY=0
RESUME_AFTER=""
SKIP_EVIDENCE=0
ALLOW_BUSY_GPU=0
GATES=(G0 G1 G2 G3 G4 G5 G6 G7)

run_label() {
  case "$1" in
    b0_t1_ppo) echo "simplePPOv2" ;;
    b2_bisec_full) echo "BiSECv2" ;;
    *) echo "$1" ;;
  esac
}

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --with-b0-control) CONTROL="b0_t1_ppo"; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --resume-after) RESUME_AFTER="$2"; shift 2 ;;
    --skip-gate-evidence) SKIP_EVIDENCE=1; shift ;;
    --allow-busy-gpu) ALLOW_BUSY_GPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/e0_orchestration"
mkdir -p "${LOG_DIR}"
PREFLIGHT="${LOG_DIR}/${EXPERIMENT}_seed${SEED}_preflight_${STAMP}.json"
fail() { echo "[E0-v2] PREFLIGHT FAILED: $*" >&2; exit 1; }

[[ -x "${PYTHON}" ]] || fail "Python not found: ${PYTHON} (set E0_PYTHON)"

# 1. Config and protocol --------------------------------------------------------
CONFIG="$("${PYTHON}" - "${EXPERIMENT}" <<'PY'
import sys
from omegaconf import OmegaConf
matrix = OmegaConf.load("configs/experiments/e0_experiments.yaml")
name = sys.argv[1]
if name not in matrix.experiments:
    sys.exit(f"unknown experiment {name}")
print(matrix.experiments[name].config)
PY
)" || fail "experiment ${EXPERIMENT} is not in the matrix"
"${PYTHON}" scripts/rsl_rl/train.py --validate-config --config "${CONFIG}" > /dev/null \
  || fail "${CONFIG} does not validate"
ACTUAL_PROTOCOL="$("${PYTHON}" scripts/rsl_rl/train.py --validate-config --config "${CONFIG}" \
  | awk '/^protocol_version:/ {print $2}')"
[[ "${ACTUAL_PROTOCOL}" == "${PROTOCOL}" ]] \
  || fail "${CONFIG} has protocol_version=${ACTUAL_PROTOCOL}, expected ${PROTOCOL}"

# 2. Gate evidence --------------------------------------------------------------
EVIDENCE_STATUS="checked"
if [[ "${SKIP_EVIDENCE}" == 1 ]]; then
  EVIDENCE_STATUS="SKIPPED by --skip-gate-evidence"
  echo "[E0-v2] WARNING: gate evidence check skipped; this is recorded in ${PREFLIGHT}" >&2
else
  "${PYTHON}" - "${GATES[@]}" <<'PY' || fail "gate evidence incomplete (see messages above)"
import json, sys
from pathlib import Path
problems = []
for gate in sys.argv[1:]:
    path = Path("output/fix_plan") / gate / "gate.json"
    if not path.is_file():
        problems.append(f"{path}: missing")
        continue
    report = json.loads(path.read_text())
    allowed = {"PASS", "CONDITIONAL"} if gate == "G3" else {"PASS"}  # G3: 25-30 mm lift continues with capped steps
    if report.get("status") not in allowed:
        problems.append(f"{path}: status={report.get('status')!r}")
    if gate == "G4" and report.get("contract_upgrade_confirmed") is not True:
        problems.append(f"{path}: contract_upgrade_confirmed is not true")
for problem in problems:
    print("  " + problem, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
fi

# 3. GPU idle (T8) --------------------------------------------------------------
GPU_STATUS="not checked (nvidia-smi missing)"
if command -v nvidia-smi > /dev/null; then
  GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  # Some desktop drivers report small GNOME graphics clients through
  # --query-compute-apps. Block known compute runtimes regardless of size and
  # unknown processes using at least 512 MiB, but ignore desktop-only clients.
  BUSY="$(awk -F, '
    {
      name=tolower($2); memory=$3;
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name);
      gsub(/[^0-9]/, "", memory);
      if (name ~ /(python|isaac|kit|cuda)/ || memory + 0 >= 512) print $0;
    }
  ' <<< "${GPU_PROCESSES}")"
  if [[ -n "${BUSY}" ]]; then
    if [[ "${ALLOW_BUSY_GPU}" == 1 ]]; then
      GPU_STATUS="busy, allowed by --allow-busy-gpu"
      echo "[E0-v2] WARNING: GPU already has compute processes:" >&2; echo "${BUSY}" >&2
    else
      echo "${BUSY}" >&2
      fail "GPU has other compute processes (T8). Stop them (e.g. pause b2) or pass --allow-busy-gpu."
    fi
  else
    GPU_STATUS="idle (desktop graphics clients ignored)"
  fi
fi

# 4. Fresh start ------------------------------------------------------------------
ORCH_ARGS=(--experiment "${EXPERIMENT}" --seed "${SEED}" --require-protocol "${PROTOCOL}"
           --level-inheritance on --gates on --abort-monitor on)
if [[ -n "${RESUME_AFTER}" ]]; then
  ORCH_ARGS+=(--completed-stage "${RESUME_AFTER}")
else
  RUN_LABEL="$(run_label "${EXPERIMENT}")"
  PADDED_SEED="$(printf '%02d' "${SEED}")"
  EXISTING="$(ls -d logs/rsl_rl/*/*_"${RUN_LABEL}"-foundation-seed"${PADDED_SEED}" 2>/dev/null \
    | while read -r d; do grep -l "^protocol_version: ${PROTOCOL}$" "$d/resolved_training_config.yaml" 2>/dev/null; done || true)"
  if [[ -n "${EXISTING}" ]]; then
    echo "${EXISTING}" >&2
    fail "an ${PROTOCOL} Foundation run for ${EXPERIMENT} seed ${SEED} already exists; use --resume-after <stage> to continue it, or move it aside."
  fi
fi

cat > "${PREFLIGHT}" <<JSON
{
  "experiment": "${EXPERIMENT}",
  "seed": ${SEED},
  "config": "${CONFIG}",
  "protocol_version": "${ACTUAL_PROTOCOL}",
  "gate_evidence": "${EVIDENCE_STATUS}",
  "gpu": "${GPU_STATUS}",
  "resume_after": "${RESUME_AFTER}",
  "control_experiment": "${CONTROL}",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $( [[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo true || echo false ),
  "started": "${STAMP}"
}
JSON
echo "[E0-v2] preflight passed: ${PREFLIGHT}"
[[ "${PREFLIGHT_ONLY}" == 1 ]] && exit 0

# Launch ----------------------------------------------------------------------------
RUN_LOG="${LOG_DIR}/${EXPERIMENT}_seed${SEED}_longrun_${STAMP}.out"
PID_FILE="${LOG_DIR}/${EXPERIMENT}_seed${SEED}_longrun.pid"
CHAIN=("${PYTHON}" "${SCRIPT_DIR}/e0_training_orchestrator.py" "${ORCH_ARGS[@]}")
CONTROL_CMD=""
if [[ -n "${CONTROL}" ]]; then
  # Architecture control runs strictly after the main lineage, never in parallel.
  CONTROL_CMD="${PYTHON} ${SCRIPT_DIR}/e0_training_orchestrator.py --experiment ${CONTROL} --seed ${SEED} --require-protocol ${PROTOCOL} --level-inheritance on --gates on --abort-monitor on"
fi
RUNNER="$(printf '%q ' "${CHAIN[@]}")"
[[ -n "${CONTROL_CMD}" ]] && RUNNER="${RUNNER} && ${CONTROL_CMD}"

if [[ "${FOREGROUND}" == 1 ]]; then
  set +e
  bash -c "${RUNNER}" 2>&1 | tee "${RUN_LOG}"
  exit "${PIPESTATUS[0]}"
fi

nohup setsid bash -c "${RUNNER}" > "${RUN_LOG}" 2>&1 < /dev/null &
echo $! > "${PID_FILE}"
cat <<EOF
[E0-v2] started ${EXPERIMENT} seed ${SEED}${CONTROL:+, then ${CONTROL}} (pid $(cat "${PID_FILE}"))
  output:    ${RUN_LOG}
  audit log: ${LOG_DIR}/${EXPERIMENT}_seed${SEED}.log
  manifest:  ${LOG_DIR}/${EXPERIMENT}_seed${SEED}.json
  gates:     ${LOG_DIR}/${EXPERIMENT}_seed${SEED}_<stage>_gate.json
  follow:    tail -f ${RUN_LOG}
  stop:      kill -- -\$(cat ${PID_FILE})
Exit codes: 0 done, 3 stage gate paused (review, then --resume-after <stage>), 4 aborted.
EOF
