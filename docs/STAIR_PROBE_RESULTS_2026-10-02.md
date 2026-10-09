# 20k Policy and Stair Probe Results

> **Historical evidence boundary:** These results apply only to the checkpoint, code, probe parameters, and generated geometry recorded below. They are not evidence that the current E0-46D run has learned stair traversal or passed the locomotion acceptance gates.

The simulated mechanism can move its feet upward, but these trials do not
establish a balanced four-leg stair-climbing gait. The 20k policy raised its
front wheels onto a 40 mm step while its rear wheels remained below it. It
completed the dedicated Plateau track, but did not pass overall acceptance.

All evidence below uses the packaged closed-loop asset and its existing
actuator force limits. No training or checkpoint modification was performed.
The checkpoint is
`2026-09-30_22-43-18_directional_curriculum_seed42_v7_formal20k/model_19999.pt`,
SHA-256 `7f4de5fe669c9bccb92076de36a81f79d97d5d8965d4d35e231343f087b01ae9`.

## Dedicated Tracks

| Nominal track | Geometric traversal | Forward velocity RMSE | Action saturation | Overall acceptance |
| --- | --- | --- | --- | --- |
| Plateau, seed 8101 | Complete; 4/4 gates, no corridor violation | 0.0848 m/s | 45.42% | Fail |
| Washboard, seed 8201 | Incomplete: corridor violation despite 4/4 longitudinal gates | 0.0881 m/s | 45.49% | Fail |

Both runs lasted 35 s, preserved `[0.28, 0, 0, 0.105]` commands, and had no
unsafe termination. Saturation exceeds the existing 5% limit. These are two
nominal runs, not the five-seed nominal/randomized acceptance matrix.

Sources: [Plateau](../output/probes/stair_climb/plateau20k_nominal.json),
[Washboard](../output/probes/stair_climb/washboard20k_nominal.json).

## Leg Motion and Steps

| Experiment | Result | Meaning |
| --- | --- | --- |
| Suspended fixed-base, +/-0.5 rad, 5x5 grid excluding neutral | Each leg sustained 18.84-18.87 mm body-relative lift | Positive simulated leg-motion evidence within policy target range |
| Suspended fixed-base, +/-1.2 rad, 3x3 grid excluding neutral | Each leg sustained 38.51-38.57 mm body-relative lift | Additional motion exists outside the trained action range |
| Free-base manual lift/advance/place, 40 mm single step | Incomplete; base stopped at Y=0.304 m, first riser at Y=0.450 m | This candidate gait did not cross the step |
| Free-base bend/advance/extend, 40 mm single step | Incomplete; terminated by base collision at Y=0.350 m | Forward placement pose exposes the proposed open-loop sequence, but does not yet establish balanced traversal |
| Rolling-only baseline, same step | Incomplete | Wheel rolling at the probe speed was insufficient |
| Frozen policy, same step, 15 s motion | Incomplete; base Y=0.537 m; front wheel bottoms near 40 mm, rear wheel bottoms near ground | Partial front-wheel ascent, not four-foot traversal |
| Frozen policy, three 40 mm risers, 350 mm treads, 20 s motion | Incomplete; base Y=0.541 m | No demonstrated continuous stair ascent |

The single-step and three-step final trials use the training positive-pyramid
mesh generator with an approach spawn outside the pyramid. The fixture scans
used a flat approach area in the earlier straight-step scene. Their fixed
base at 0.30 m is an explicit experimental fixture: absolute wheel height is
not lifting performance. Grid maxima are measured lower bounds on reachable
motion, not global kinematic maxima. Increasing action limits is a diagnostic,
not a recommended training change without further constraint/collision checks.

The manual trials reused measured fixture lift targets and retained a free base.
Those targets had not passed three-support validation. The right front leg
shortened approximately 19 mm relative to the body, but that did not produce
sufficient ground clearance and forward transfer. The controller stalled at
the first leg, so its subsequent leg phases were not completed.

The latest controller explicitly implements the proposed bend -> advance body
-> extend sequence and limits each advance to 1.5 s. It reached the first
obstacle farther than the earlier candidate, but terminated on base collision
before all four wheels were beyond the riser. This is a negative result for
this particular open-loop gait, not a proof that the mechanism cannot climb.

No final traversal trial recorded the strict clean-clearance predicate:
unloaded foot, positive body-relative lift, and a conservative wheel-radius
envelope clearing the riser. Failure of that predicate does not mean that no
leg motion occurred; it excludes interpreting rolling contact or body pitch
alone as a clean lifted-foot crossing.

Sources: [fixture scan](../output/probes/stair_climb/fixed_policy_range.json),
[extended scan](../output/probes/stair_climb/fixed_extended_range.json),
[single-step trial](../output/probes/stair_climb/manual_policy_final.json),
[three-step trial](../output/probes/stair_climb/pyramid_3x4cm.json).
Each probe has an adjacent `.jsonl` trajectory and `.terrain.usda` file.

## Command Suite

Existing validation script, seed 1001, 64 environments, eight terrain families,
1500 control steps per episode, five commands: 320 complete episodes. All
monitored values were finite; the overall performance gate failed.

| Command | Forward velocity RMSE (m/s) | Yaw-rate RMSE (rad/s) | Survival to timeout |
| --- | --- | --- | --- |
| Stop | 0.0374 | 0.0473 | 100% |
| Forward | 0.2331 | 0.1374 | 82.81% |
| Backward | 0.2245 | 0.1314 | 87.50% |
| Left turn | 0.0372 | 0.2515 | 100% |
| Right turn | 0.0367 | 0.2543 | 100% |

Survival alone is not command execution: turning error is roughly the commanded
0.25 rad/s. Positive-pyramid forward/backward survival was 50%/62.5%, but the
suite uses the training platform spawn and is not a dedicated ascent test.

Source: [command-suite report](../output/probes/stair_climb/policy20k_training_suite.json).

## Limits and Reproduction

Two broader support searches were stopped before their final JSON reports;
their partial `single_step_4cm.jsonl` and `tripod_translate.jsonl` traces are
excluded from completed-trial conclusions. They do not establish exhaustive
failure. Initial concurrent track launches exhausted GPU memory; those were
discarded and both reported track runs were rerun successfully.

This evidence rules out "the simulated legs cannot lift at all". It does not
choose conclusively between an inadequate learned gait, action-range limits,
balance/control design, and mechanical limitations on complete ascent. A
successful free-base gait at unchanged force and action limits would be the
strongest next discriminator. The present failures are not proof that the
physical mechanism cannot climb stairs.

See [probe commands and criteria](STAIR_PROBE.md). The script passed syntax
compilation and three targeted metric regressions: root-only progress cannot
pass, terminal/unsupported poses cannot pass, and contact crossing cannot be
labelled clean active lift. The evidence plot can be regenerated using
`output/probes/stair_climb/plot_evidence.py`.

![Probe evidence](../output/probes/stair_climb/evidence.png)
