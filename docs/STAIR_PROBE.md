# Stair Capability Probe

`scripts/probe_stair_climb.py` separates actuator-driven foot motion, free-base
support, obstacle traversal, and frozen-policy performance. It writes a JSON
report, a per-control-step JSONL trace, and the collision terrain USD beside the
report. Trace samples include the terminal state before Isaac Lab resets it.

## Run

The checkpoint path below is a historical baseline, not the current E0 policy.
For an E0 comparison, substitute a frozen compatible checkpoint and record its
configuration/hash; this probe is a diagnostic, not long-duration training or a
replacement for the full locomotion acceptance suite.

Use the same Python environment as training. On this workstation:

```bash
export LD_LIBRARY_PATH=/home/kytolly/Utils/Anaconda/envs/env_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
PY=/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python
CKPT=logs/rsl_rl/base_locomotion_complex/2026-09-30_22-43-18_directional_curriculum_seed42_v7_formal20k/model_19999.pt

$PY scripts/probe_stair_climb.py --viz none --device cuda:0 \
  --fixed-base --grid 5 --output output/probes/fixed_policy_range.json

$PY scripts/probe_stair_climb.py --viz none --device cuda:0 \
  --support-search --support-pattern translate \
  --lift-report output/probes/fixed_policy_range.json \
  --checkpoint "$CKPT" --output output/probes/single_step.json

$PY scripts/probe_stair_climb.py --viz none --device cuda:0 \
  --skip-scan --step-count 3 --checkpoint "$CKPT" \
  --output output/probes/pyramid_stairs.json
```

Use `--viz kit` to inspect the simulation. Run GPU jobs sequentially on a 16 GB
GPU: separate PhysX processes reserve substantial memory even for one robot.

The default `--terrain pyramid` calls the training task's positive pyramid
generator. Defaults match its minimum riser height (0.04 m), tread width
(0.35 m), and platform parameter (2.5 m). `--step-count` controls the number of
ascending risers. The robot starts outside the pyramid and receives a forward
command, whereas the training generator's default origin is on its central
platform. This probe therefore tests ascent explicitly; it is not the training
reset distribution. `--terrain stairs` creates a straight ascending section.

## Evidence

- The fixed-base fixture is a world-to-base fixed joint at 0.30 m. Read
  `body_relative_lift_m`, not absolute foot height, for its reach result.
  It demonstrates simulated motion only, never balance or traversal.
- Free-base scans retain gravity, contacts, self-collision, joint limits,
  servo force limits, and wheel dynamics. No root pose is written during a
  trial. Each candidate begins with a reset and a settling interval.
- A stable lift requires the lifted foot to be unloaded, the other three
  contacts to exceed 1 N, tilt below 0.35 rad, and at least 5 mm of foot motion
  relative to the body throughout the hold interval. This is a conservative
  contact criterion, not a proof of a positive center-of-mass support margin.
- Manual control approaches a riser, lifts one leg, advances on the other
  wheels, then changes the lifted leg to a forward placement pose before
  lowering it. This is the explicit bend -> advance body -> extend open-loop
  hypothesis. Each advance has a finite `--advance-time` budget so a stalled
  leg cannot hide the later phases. If no stable candidate was found, it tries
  the non-terminated candidate with greatest relative lift; such a fallback is
  not labelled stable. The controller stops at termination or its step budget,
  without stitching together automatically reset episodes.
- Traversal requires all four wheel bottoms on the top level, behind the last
  riser, within the central corridor, with at least three contacts and low tilt
  for 0.25 s. It is an ascent test, not a descent test.
- `active_lift_evidence` requires the wheel's conservative radius envelope to
  cross above the riser with 2 mm clearance, unloaded, and with body-relative
  leg shortening. It is sufficient evidence of clean clearance, not a necessary
  condition for climbing: rolling contact, wheel tilt, and mixed rolling/lifting
  can produce successful traversal without passing this strict condition.
- Frozen-policy actions remain clipped to [-1, 1], producing the trained
  +/-0.5 rad leg targets. `--leg-limit 1.2` expands only the scripted diagnostic
  range and does not change actuator force limits. Never compare expanded-range
  scripted success to policy failure as evidence of a learning defect alone.

Finite grids and one scripted gait cannot prove mechanical impossibility.
Positive simulation results also depend on the packaged geometry, closure
constraints, actuator model, and contacts; they do not validate the physical
robot. Reports are individual trials, not multi-seed acceptance results.

## Official Tracks

The dedicated terrain-performance tests remain Plateau and Washboard. Preserve
their command, geometry, duration, and acceptance thresholds:

```bash
$PY scripts/run_benchmark.py --config configs/benchmark/plateau.yaml \
  launcher.viz=none policy.zero_policy=false policy.checkpoint="$CKPT" \
  output=output/probes/plateau20k.json
$PY scripts/run_benchmark.py --config configs/benchmark/washboard.yaml \
  launcher.viz=none policy.zero_policy=false policy.checkpoint="$CKPT" \
  output=output/probes/washboard20k.json
```

`traversal_complete` and `performance_pass` are separate. See
[BENCHMARKS.md](BENCHMARKS.md) for the two tracks and the multi-seed protocol.
