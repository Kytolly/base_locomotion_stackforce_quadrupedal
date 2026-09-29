# Evaluation Metrics

This directory owns episode metrics for command tracking, stability, terrain performance, actuation, reward decomposition, and runtime health. `TrainingMetricsWrapper` publishes these values through both RSL-RL episode channels so TensorBoard and W&B receive the `train/*` namespace whenever an episode ends.
