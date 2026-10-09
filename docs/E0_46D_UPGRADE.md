# E0 46D Locomotion Upgrade

`Base-Locomotion-Stackforce-Quadrupedal-Complex-v0` now uses the E0 V1 training contract from the repository-level `doc/locomotion/` plan while keeping the existing deployable Actor input:

```text
command d_ref       4
proprioception     30
previous action    12
Actor input        46
```

The additional Actor terms for forward terrain scanning, contact flags, endpoint state, and terrain validity belong to the separate Hybrid task. They are not consumed by the `Complex-v0` Actor. E0 still has support/contact sensors for reward, termination, curricula, and evaluation; their presence does not expand the 46D Actor input.

## Split curricula

Curriculum implementations are in `tasks/manager_based/base_locomotion_stackforce_quadrupedal/mdp/curricula/`:

- `terrain.py`: command-aligned terrain promotion and demotion;
- `motion.py`: motion-family coverage, success statistics, and rehearsal floor;
- `robustness.py`: terrain-level-derived bin bookkeeping (0 for nominal, 1 for intermediate, 3 for stress; no bin 2), not a staged expansion of physical randomization; `moderate_level` is currently unused.

Their versioned YAML parameter references are in `configs/curriculum/`. The training entrypoint applies the `curriculum` block from `configs/train/base_locomotion_e0_46d.yaml` to the environment before creating the task; it does not automatically load or merge the separate curriculum files. The training loop and information boundary are described in [TRAINING_DESIGN.md](TRAINING_DESIGN.md).

## Training log switches

The logging contract separates custom metric collection from dashboard visibility. The resolver's recommended defaults are `core`, `safety`, and `runtime`; the current workspace YAML has W&B enabled in online mode and all metric/panel groups set to true. The recommended compact YAML is maintained in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

The intended contract is that `metrics` controls collection and `wandb_panels` filters custom scalars sent to W&B/TensorBoard; it does not remove existing web dashboard panels. RSL-RL's built-in PPO scalars are unaffected.

Both training entrypoints currently resolve this configuration before passing it to `TrainingMetricsWrapper`, which resolves it again using the raw YAML schema. This resets custom group and `enabled` overrides to the resolver defaults. Do not assume expanded logging is active merely because its YAML switch is true. The current limitation and configuration owner are documented in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Short smoke

Configuration-only validation, which does not start Isaac Sim:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/train.py \
  --config configs/train/base_locomotion_e0_46d.yaml \
  --validate-config
```

For a short run, use:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/train.py \
  --config configs/train/base_locomotion_e0_46d.yaml \
  launcher.viz=none \
  env.num_envs=128 \
  agent.max_iterations=1 \
  wandb.enabled=false
```

This is only a pipeline/contract smoke, not evidence of locomotion performance.

## Long training

The command below starts the full 20k-iteration E0 run. It is intentionally not launched as part of repository changes. E0 defaults to `launcher.viz=none`; use `launcher.viz=kit` explicitly for GUI inspection. `max_visible_envs` affects visibility, not the simulation environment count:

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/train.py \
  --config configs/train/base_locomotion_e0_46d.yaml \
  launcher.viz=none \
  wandb.enabled=true \
  wandb.mode=online
```

Use a new seed and run name for each independent seed, for example `env.seed=1001 agent.run_name=e0_46d_seed1001`. Do not resume a V0 or incompatible action-distribution checkpoint: the environment curriculum and training contract must match for a valid restart.
