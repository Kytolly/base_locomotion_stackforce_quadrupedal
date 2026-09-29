# Stackforce quadrupedal wheeled robot

This asset is the loop-cut representation of the validated four-leg assembly.
The URDF contains the 12 existing outer-chain joints plus four `Inner_Hip_Joint` and four
`Inner_Knee_Joint` passive joints. Each inner chain stops at an explicit `Closure_Frame`; it does
not make `foot` a second child, so the URDF remains a strict tree.

The STL files in `meshes/` are link-local exports of the C6.29 validated USD
stage. Mirrored legs are generated from the FR master using exact X/Y chassis
reflections, including reflected winding. `config/closure_frames.json` keeps
the measured Wheel/Closure world relationship: the axial offset is about -44.950 mm,
the radial residual is below 2e-14 m, and the P2 surface gap is 0.150 mm.

## Build and validate

Run with the Isaac Sim Python shipped with the installation:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME \
  /home/kytolly/Library/isaacsim/python.sh \
  scripts/build_asset.py
python3 scripts/validate_asset.py
```

`build_asset.py` reads the accepted source stage, writes the English-named
STL files and `urdf/stackforce_quadrupedal_wheeled_robot.urdf`, and imports the
URDF with `merge_fixed_joints=False` into `usd/`.

## Restore the four closures

The imported USD is intentionally loop-cut. After loading it in Isaac Sim,
run:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME \
  /home/kytolly/Library/isaacsim/python.sh \
  scripts/recover_closed_loops.py \
  usd/stackforce_quadrupedal_wheeled_robot/stackforce_quadrupedal_wheeled_robot.usda \
  --output usd/stackforce_quadrupedal_wheeled_robot_closed.usda
```

For each leg the script uses the Closure frame as the common world mechanical axis,
then independently computes `localPos0/localRot0` for `inner_lower` and
`localPos1/localRot1` for `foot`. The four joints are `PhysicsRevoluteJoint`
objects under `closure_joints`, with collisions disabled and
`physics:excludeFromArticulation = true`. This preserves the tree articulation
while adding the physical closure constraints.

The generated USD is also a review artifact; it can be opened directly in
Isaac Sim for visual inspection of all four inner chains and closure frames.

## Validation status

`validation/simulation_report.json` records the latest Isaac Sim CPU smoke test.
It checks all 20 tree revolute joints, all four independently reconstructed
closure poses, and 2,400 physics steps with a fixed base under gravity. The
largest measured closure anchor drift was 0.060 mm and the largest axis drift
was 5.3e-6 rad. This test has no ground contact, wheel command, actuator model,
or self-collision qualification. `negative_control_report.json` is a regression
check with closures disabled; it must fail because the closure error grows to
the centimetre scale.

Run the checks with:

```bash
python3 scripts/validate_asset.py
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME \
  /home/kytolly/Library/isaacsim/python.sh scripts/simulation_validation.py
```

## RL handoff

The Isaac Lab direct and manager environments load
`usd/stackforce_quadrupedal_wheeled_robot_closed.usda` through the package
relative contract in `source/sf_quad/sf_quad/assets/closed_loop_contract.py`.
Their policy action is the fixed 12-joint order `Outer_Hip_Joint`, `Inner_Hip_Joint`, then `Wheel_Joint`;
Outer/Inner knee joints are passive and closure joints are constraints. Explicit actuator limits and drive
parameters are present, but remain marked `ESTIMATED` until hardware evidence
is attached.

Run the qualification from the `sf_quad` project root using
`scripts/closed_loop_qualification.py` and
`scripts/vectorized_smoke_test.py`. Review the generated JSON before training:
the repository keeps `rl_ready` false while any runtime gate, self-collision
sanity check, vectorized count, or physical-parameter provenance gate is
unresolved.
