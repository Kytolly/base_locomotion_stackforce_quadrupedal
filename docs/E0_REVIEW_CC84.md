# E0 review of cc84e102

## Decision

GO for the tested single-GPU short-run pipeline. NO-GO for claiming a faithful
Liang et al. reproduction or beginning the formal comparative campaign on this
evidence alone. No long training was run. Shared terrain, rewards, curricula,
commands, observations, actions and PPO hyperparameters were not modified.

## Corrections

- `training/sparse_expert.py`: selected samples are dispatched to each expert;
  inactive expert/sample pairs are not evaluated. Actor orthogonality defaults
  to normalized active outputs, with mean squared Gram error. The all-six
  normalized loss remains in `auxiliary/a3_all_expert_ortho.yaml`.
- `training/checkpoint.py`: resume starts after the saved completed iteration
  and restores adaptive LR from Adam. The stage exception removes proportions
  only under `environment.scene.terrain.terrain_generator.sub_terrains`, not
  identically named fields elsewhere.
- `scripts/rsl_rl/train.py`: exact checkpoint paths replace regex lookup;
  rejected training exits with status 1 even when Kit closes the process.
- Configuration tests cover the original 19 inherited YAMLs plus the new
  auxiliary control. Main contains four configurations; auxiliary contains 16.
- Training/logging documentation describes Foundation-only direct execution
  and single-pass logging resolution.

## Evidence (2026-10-09, env_isaaclab)

GPU smoke used the real `train.py` entrypoint: 32 environments, 24 rollout
steps, two iterations per stage, 0.2-second diagnostic episodes, unchanged PPO
settings, seed 42, W&B offline. Diagnostic episode duration is not a production
configuration change. `tests/gpu_resume_audit.py` instruments the real runner
to compare loaded tensors and iteration semantics.

Run directories under `logs/cc84_review/base_locomotion_e0_46d/`:

| Stage | Run | Result |
| --- | --- | --- |
| Foundation | `2026-10-09_21-37-27_cc84_review_foundation` | Updates 0,1; `model_1.pt`; training loop 2.54 s |
| Expansion | `2026-10-09_21-39-06_cc84_review_expansion` | Updates 2,3; `model_3.pt`; training loop 2.79 s |
| Final-code Expansion repeat | `2026-10-09_21-47-33_cc84_review_final` | Updates 2,3; `model_3.pt`; training loop 2.81 s; exit 0 |

Expansion restored Actor, Critic, every Adam moment/step and parameter group
exactly before updating. Its restored adaptive LR was `1e-5`.
`resume_audit.json` records the start/end and checkpoint. The non-terrain negative
case (`gamma=0.98`) failed the contract before loading/updating and, after the
shutdown fix, exited 1. See `/tmp/cc84_review_reject_exit.log`.

W&B offline run `y5sfr0vr` contains two history records and 135 scalar keys.
The binary `.wandb` history was decoded, not merely the stdout log: core 5,
safety 9, runtime 2, command 9, support 2, actuation 8, terrain 21, motion 16,
reward 30 matching namespace keys. Runtime finite checks were 1. Online upload
and browser dashboard layout were not tested; panel flags select scalar sinks,
not saved web dashboard layout.

CPU regression checks cover configuration inheritance/contracts, dense/dispatch
output and gradient equivalence, exactly `2 * batch` expert rows, inactive
expert gradients, active-only orthogonality, checkpoint filtering and Adam/index
restoration. Run the focused suite:

Result: 30 passed. All 20 YAMLs also passed the real `train.py --validate-config`
entrypoint. Changed-file whitespace checks passed; the pre-existing user edit
in `mdp/curriculum.py` has trailing whitespace and was left untouched.

```bash
PYTHONPATH=source/base_locomotion_stackforce_quadrupedal python -m pytest \
  tests/test_sparse_expert.py tests/test_training_config.py \
  tests/test_e0_46d_contract.py tests/test_bounded_control.py \
  tests/test_metric_logging_config.py tests/test_resume_iteration.py -q
```

Short CUDA inference timing (10 warmups, 50 forwards, synchronized, no reflex):

| Batch | Dense ms | Top-2 dispatch ms |
| --- | --- | --- |
| 1 | 0.219 | 0.284 |
| 32 | 0.261 | 0.495 |
| 4096 | 0.252 | 0.544 |

This is compute sparsity, not demonstrated latency acceleration. Six experts
can all execute on different subsets of a batch. Dynamic indexing dominates
these small MLPs. Multi-GPU is explicitly rejected because rank-dependent
inactive gradients need a fixed collective layout. Export/compilation of dynamic
dispatch is not qualified.

## Paper fidelity boundary

After the equation (7) correction, focused CPU tests passed (20 tests across
sparse experts, configuration contracts and resume iteration). Real GPU smoke
was repeated with 32 environments and two updates per stage:
`2026-10-09_22-01-54_paper_temporal_foundation/model_1.pt` and
`2026-10-09_22-02-43_paper_temporal_expansion/model_3.pt`, under the same log root
above. Expansion exactly restored model/Adam state and ran iterations 2-3.
Temporal losses were finite (2.0330, 2.1527; then 2.1852, 2.3553).

Crossref and Elsevier metadata confirm Hansheng Liang et al., *Bio-inspired
sparse expert control for agile and adaptive wheeled-legged locomotion*,
Defence Technology (2026), DOI [10.1016/j.dt.2026.06.020](https://doi.org/10.1016/j.dt.2026.06.020),
PII `S2214914726002096`. The user-provided PDF was inspected directly, including
rendered page 4, equations (1)-(8), and pages 3-5 describing the architecture
and training. The source is `Bio-inspired sparse expert control for agile and
adaptive wheeled-legged locomotion.pdf` in the user's Downloads directory.

Page 4 explicitly requires orthogonality among activated outputs. L2-normalizing each
expert output and averaging Gram entries are explicit Stackforce conventions;
the coefficient is not numerically equivalent to an unnormalized Frobenius sum.
The old all-expert scope must not be pooled into the corrected main results.
Older checkpoints lacking the new method fields fail the training contract;
they remain separate legacy evidence.

Equation (5) prints positive sum(w log w), but its text claims minimizing entropy;
these disagree mathematically. Positive entropy follows the text and supplied
plan, not the printed sign. Equation (7) is implemented as the squared L2 distance
between adjacent raw residual outputs (before the shared action squash), summed
over action dimensions and averaged over valid transitions. Both outputs are
recomputed under current parameters and both receive gradients. Previous policy
observations are paired before PPO minibatch shuffling in training-only storage;
sampled previous actions are not temporal targets. First frames, episode resets
and command-transition grace periods are masked. Pairs persist across rollout
boundaries; resuming into a new simulator invalidates the first pair.
The algorithm contract records `temporal_target=adjacent_policy_mean_v1`, so
checkpoints from the previous objective cannot silently resume.

Undisclosed conventions remain explicit: the paper does not define output
normalization for E, post-Top-K weight renormalization, or minibatch reduction.
This implementation keeps the supplied plan's normalized active Gram and
renormalized Top-K weights. It averages over valid samples for batch-size
independence. These are not claimed as author-verified hyperparameters.
50 Hz reflex, 12D squashed action, 46D Actor and privileged Critic are also
declared Stackforce adaptations. These shared interfaces were deliberately not
changed in this review. The result is a paper-grounded Stackforce adaptation,
not a platform-identical or author-code reproduction.

Short episodes qualify logging and recovery only: they do not qualify physical
behavior, curriculum continuation state, convergence, terrain coverage or 4096-env
long-run stability. Simulator/RNG/terrain-level state is not restored by the PPO
checkpoint. Stage transitions start new environments, not bitwise continuation.
