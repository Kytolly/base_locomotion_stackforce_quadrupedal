# Locomotion Acceptance Criteria

## Scope

These gates assess whether the controller continuously converts body-frame velocity and body-height intent into stable wheel/leg actuator targets over supported terrain. Passing a smoke test or observing one successful rollout does not qualify a policy.

The current training baseline is E0-v1-46d (46D Actor, 62D Critic, 12D action). These gates apply to a frozen checkpoint; the separate 108D Hybrid proposal does not redefine that baseline. No current E0 checkpoint is claimed to pass these gates. Execution commands use `env_isaaclab` as specified in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Physical qualification gates

| Gate | PASS condition | Evidence |
| --- | --- | --- |
| Action semantics | Every one of 12 policy dimensions produces the documented clipped/scaled target on exactly one intended active joint; signs, limits, zero action, passive closure response, and finite simulation state are reviewed | `output/action-probes/*.json` and Isaac Sim GUI recording/log |
| Wheel support | All four wheel contact sensors resolve to the intended `FR, FL, RL, RR` links; each wheel establishes ground contact during a static support test | asset/simulation qualification report |
| Safety | A controlled fall triggers height or tilt termination; a measured base-ground collision triggers base-contact termination; normal low obstacles do not trigger either condition | validation JSON and simulator log |
| Local support | At least 99% valid support estimates on flat, slopes, steps, rough ground, and both fixed tracks; height error uses wheel support first and local ray fallback | `support/invalid_rate` and ground-truth probe |

Until these gates pass, friction, actuator, and collision parameters remain qualification values, not hardware-calibrated constants.

## Frozen policy thresholds

The following initial acceptance thresholds are enforced by `scripts/validate_policy.py` report generation. Every one of the five command scenarios and every terrain family must pass. All episodes must finish (timeout or measured unsafe termination); timeout survival must be at least 95% in every family. Validation freezes curriculum, uses all eight levels with at least 64 environments (a multiple of 64), and runs each scenario for 30 seconds. A shorter smoke or a single-level diagnostic cannot produce performance PASS.

| Metric | PASS threshold |
| --- | ---: |
| Forward velocity RMSE | <= 0.15 m/s |
| Lateral velocity RMSE | <= 0.12 m/s |
| Yaw-rate RMSE | <= 0.10 rad/s |
| Body-height RMSE above support | <= 0.025 m |
| Unsafe termination rate | <= 5% per command/terrain family |
| Base collision rate | <= 1% of episode steps |
| Invalid local-support rate | <= 1% of episode steps |
| Wheel contact fraction | >= 50% of wheel/step samples |
| Action saturation rate | <= 5% of policy dimensions/steps |
| 95th percentile episode maximum base tilt | <= 1.05 rad |
| Runtime finite rate | 100% |

Plateau and Washboard require the same numerical gates, a complete timeout episode, every track gate crossed in sequence, no lateral corridor violation, no unsafe termination, and finite monitored values. Zero policies cannot qualify. Repeated benchmark acceptance requires five distinct seeds for both nominal and randomized track geometry; each run must pass. Nominal repeats verify reproducibility, not independent geometric generalization.

## Evidence status

These are conservative initial project gates, not a claim that an existing checkpoint has passed. A final policy PASS requires the physical qualification gates, a recorded long-run PPO experiment with its configuration and W&B run, the full five-command/eight-terrain validation, and both repeated benchmark suites. The report distinguishes structural eligibility from `performance_pass`.
