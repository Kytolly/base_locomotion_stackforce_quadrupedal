# E0 experiment configurations

Each registered experiment has one directly executable training configuration:

- main experiments: `configs/train/main/`
- auxiliary experiments: `configs/train/auxiliary/`
- matrix: `configs/experiments/e0_experiments.yaml`
- command generator: `scripts/e0_experiment_commands.py`
- shared contract: `configs/train/base_locomotion_e0_46d.yaml`

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

Omit `--experiment`, `--seed`, and `--stage` to print all 19 registered experiments, three seeds, and four terrain stages. Add `--include-sweep` to append the 32 regularizer candidates.

## Run a staged training lineage

The common 20,000-iteration budget is split into foundation 1,000, expansion 5,000, composition 7,000, and consolidation 7,000 iterations. Run foundation first. For each later stage, provide the exact previous run directory and checkpoint:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/e0_experiment_commands.py \
  --experiment b2_bisec_full --seed 42 --stage expansion \
  --load-run 2026-10-09_20-00-00_b2_bisec_full_foundation_seed42 \
  --load-checkpoint model_999.pt
```

The generator enables the controlled terrain-stage resume flag for non-foundation stages. That flag permits only terrain profile/proportion changes. A change to observations, actions, rewards, commands, curriculum terms, PPO, or network structure still fails the checkpoint contract.

## Evaluation

Run the frozen suite after selecting a checkpoint:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/evaluate_suite.py \
  --checkpoint /absolute/path/to/model.pt \
  --config configs/evaluation/generalization.yaml
```

The suite evaluates both `union_consolidation` and the DeepRobotics six-family `source_alignment` profile. For P1/P2/P3 checkpoints, set `validation.task` to the checkpoint's registered Gym task through an OmegaConf override.

Legacy checkpoints are evaluation-only. Do not resume them under the E0 v1.1 contract.
