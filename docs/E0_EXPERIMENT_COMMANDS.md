# E0 experiment configurations

Each registered experiment has one directly executable training configuration:

- main experiments: `configs/train/main/`
- auxiliary experiments: `configs/train/auxiliary/`
- matrix: `configs/experiments/e0_experiments.yaml`
- command generator: `scripts/e0_experiment_commands.py`
- four-stage scheduler: `scripts/run_e0_training.sh`
- formal E0-v2 long run with preflight: `scripts/run_e0_v2_long_run.sh`
- shared contract: `configs/train/base_locomotion_e0_46d.yaml` (`protocol_version: E0-v2-46d`)

Since 2026-10-13 the shared contract is **E0-v2-46d**: leg actions span the
DS041MG servo travel around a new default stance (see
[ACTION_RANGE_THROUGHPUT_FIX_PLAN.md](ACTION_RANGE_THROUGHPUT_FIX_PLAN.md)).
Runs go to `logs/rsl_rl/base_locomotion_e0_v2_46d/` and W&B groups prefixed
`e0v2-`. E0-v1 checkpoints (`base_locomotion_e0_46d/`) are evaluation-only;
resuming them under E0-v2 fails the checkpoint contract.

Run commands from the repository root with the Isaac Lab Python environment.

The main directory contains only B0, B1, B2-A, and B2-B. Ablations,
capacity controls, observation controls, and fixed-curriculum controls live in
the auxiliary directory. The shared contract enables every custom metric group
and every W&B panel group.

## Direct training

Start the B2-B foundation stage without architecture overrides:

```bash
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --config configs/train/main/b2_bisec_full.yaml
```

Use another YAML path to select another experiment. Command-line overrides are
reserved for stage progression, seed selection, resume paths, and diagnostics.

Direct YAML execution runs Foundation only (1,000 updates). Checkpoint names
use the last completed zero-based iteration: Foundation ends at `model_999.pt`;
Expansion resumes at 1,000 and ends at `model_5999.pt`. `max_iterations` is
the number of additional updates, not an absolute endpoint. Adam state and its
adaptive learning rate are restored along with the policy.

MoE defaults to per-sample Top-2 dispatch. Only selected expert rows are
evaluated; this reduces expert matrix work but does not guarantee lower latency
because routing/indexing adds overhead. Active-only orthogonality uses normalized
Actor outputs of shape `[batch, 2, action_dim]` and the mean squared Gram error.
`auxiliary/a3_all_expert_ortho.yaml` preserves the normalized all-six-expert loss
as a separate control; it evaluates all Actor experts and is not compute-sparse.
See [E0_REVIEW_CC84.md](E0_REVIEW_CC84.md) for fidelity limits and smoke evidence.

## Inspect the matrix

Print the foundation command for one run:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/e0_experiment_commands.py \
  --experiment b2_bisec_full --seed 42 --stage foundation
```

Omit `--experiment`, `--seed`, and `--stage` to print all 20 registered experiments, three seeds, and four terrain stages. Add `--include-sweep` to append the 32 regularizer candidates.

## Run a staged training lineage

The common 20,000-iteration budget is split into Foundation 1,000, Expansion
5,000, Composition 7,000, and Consolidation 7,000 updates. Start one complete
lineage with:

```bash
conda activate env_isaaclab
bash scripts/run_e0_training.sh --experiment b2_bisec_full --seed 42
```

The scheduler takes stage profiles and update counts directly from the matrix.
It requires the exact preceding checkpoint and verifies its Actor, Critic,
Optimizer, and iteration fields before continuing. Expected endpoints are
`model_999.pt`, `model_5999.pt`, `model_12999.pt`, and `model_19999.pt`.
Commands and subprocess output are retained in
`logs/e0_orchestration/<experiment>_seed<seed>.log`; successful stage paths are
also recorded in the adjacent JSON manifest. A failed stage stops the lineage
without deleting its run directory or checkpoint files.

For the formal matrix the scheduler also:

- **T1 terrain-level inheritance.** Each resumed stage starts all environments
  in the level band `[m-1, m]`, where `m` is the rounded median of
  `train/terrain/level` over the previous stage's last 100 iterations
  (`env.terrain_init_level`). `--level-offset 1` starts one level lower if
  resumes show difficulty jumps.
- **Stage-end checks** from `configs/experiments/e0_v2_stage_gates.yaml`
  (plan G8). A failing or missing check stops the lineage with exit code 3
  and writes `logs/e0_orchestration/<experiment>_seed<seed>_<stage>_gate.json`.
  Review it, then continue with `--completed-stage <stage>`.
- **Abort conditions.** A PhysX buffer error in the output, a zero
  `all_finite`, or `unsafe_termination` above 30 % for 500 rising iterations
  terminates the stage (exit code 4).

`python scripts/e0_run_health.py --run <run_dir> --stage <stage>` re-checks a
finished stage; add `--list-tags` to see the logged scalar names.

After Foundation has completed, resume the remaining stages with:

```bash
bash scripts/run_e0_training.sh \
  --experiment b2_bisec_full --seed 42 --completed-stage foundation
```

`--start-stage expansion` has the same checkpoint-discovery behavior. The
scheduler selects the newest valid exact run suffix for the given experiment,
stage, and seed; it rejects missing, incorrectly numbered, or incomplete
checkpoints. Non-Foundation stages enable the controlled terrain-stage resume
flag. That flag permits only terrain profile/proportion changes. A change to
observations, actions, rewards, commands, curriculum terms, PPO, or network
structure still fails the checkpoint contract.

## E0-v2 formal long run (plan G8)

```bash
bash scripts/run_e0_v2_long_run.sh --preflight-only     # checks only
bash scripts/run_e0_v2_long_run.sh                      # b2_bisec_full, seed 42, detached
bash scripts/run_e0_v2_long_run.sh --with-b0-control    # b0_t1_ppo afterwards, never in parallel
bash scripts/run_e0_v2_long_run.sh --resume-after expansion   # after a reviewed gate pause
```

Preflight refuses to start unless the config resolves to `E0-v2-46d`,
`output/fix_plan/G0..G7/gate.json` are PASS (G3 may be CONDITIONAL) with the
contract upgrade confirmed in G4, no other compute process uses the GPU (T8),
and no E0-v2 Foundation run of the same experiment/seed exists. The preflight
record goes to `logs/e0_orchestration/*_preflight_<time>.json`; the command
prints the output, audit log, manifest, stop command and exit codes.

## Evaluation

Run the frozen suite after selecting a checkpoint:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/evaluate_suite.py \
  --checkpoint /absolute/path/to/model.pt \
  --config configs/evaluation/generalization.yaml
```

The suite evaluates both `union_consolidation` and the DeepRobotics six-family `source_alignment` profile. For P1/P2/P3 checkpoints, set `validation.task` to the checkpoint's registered Gym task through an OmegaConf override.

Evaluation scripts pick the action contract from the checkpoint's run
(`resolved_training_config.yaml`): E0-v1 runs are evaluated with the legacy
symmetric legs, E0-v2 runs with the servo contract. Legacy checkpoints are
evaluation-only. Do not resume them under the E0-v2 contract.
