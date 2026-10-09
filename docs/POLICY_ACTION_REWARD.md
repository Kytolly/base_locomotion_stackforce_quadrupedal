# Policy、Action 与 Reward 合同

本文是复杂地形 locomotion 任务的网络接口与优化目标参考。观测来源的详细定义仍由根目录 [`doc/locomotion/`](../../doc/locomotion/base_locomotion_spec.md) 规范约束。

## Actor 输入

Actor 输入是 46 维平铺张量，顺序固定如下：

| 区间 | 维度 | 类别 | 内容 |
| --- | ---: | --- | --- |
| `[0:4]` | 4 | Locomotion command | 前向速度、横向速度、偏航角速度、机身高度 |
| `[4:34]` | 30 | 本体感知 | 机体角速度、投影重力、12 维主动关节位置和速度 |
| `[34:46]` | 12 | 历史记忆 | 上一控制步的 policy action |

Command Manager 的 `locomotion` term 每 10 秒采样目标，并在 episode reset 时立即生效。episode 内重采样使用每秒 `(0.5, 0.5, 0.5, 0.02)` 的逐维变化率限制。当前采样范围为：

```text
forward speed magnitude: [0.15, 0.45] m/s
lateral velocity: [-0.10, 0.10] m/s
yaw rate:         [-0.25, 0.25] rad/s
body height:      [0.095, 0.115] m
```

Actor 和 reward 都读取 Command Manager 输出的整形后 command。Critic 在完整 Actor 输入之外读取 16 维 `privileged` 组；该组包含仿真真值线速度、相对支撑高度和主动关节力矩，不得接入 Actor。项目局部坐标的 `front` 为 `+Y`。

当前 history 只是上一动作，不是多帧观测堆叠，也不是循环网络状态。

## Actor 输出

`RobotActionCfg` 定义 12 维策略输出，RSL-RL 从环境 action space 自动推导 Actor 输出宽度：

| 区间 | 维度 | 语义 | scale | processed clip |
| --- | ---: | --- | ---: | --- |
| `[0:8]` | 8 | 四腿内外髋关节位置增量 | 0.5 | `[-0.5, 0.5]` rad |
| `[8:12]` | 4 | 四轮关节速度 | 20.0 | `[-20, 20]` rad/s |

RSL-RL 的 `clip_actions=1.0` 将 raw action 约束在 `[-1, 1]`。`action_saturation` 从 raw action 计算接近裁剪边界的代价。

## Reward

当前 E0 环境 reward 分解及配置权重如下；这是优化设置，不是性能通过证据：

| Term | 权重 | 定义 |
| --- | ---: | --- |
| `track_forward_velocity` | 2.5 | 前向速度指数跟踪，`std=0.25 m/s` |
| `track_lateral_velocity` | 0.75 | 横向速度指数跟踪，`std=0.20 m/s` |
| `track_yaw_rate` | 0.75 | 偏航角速度指数跟踪，`std=0.25 rad/s` |
| `track_body_height` | 0.5 | 机身高度指数跟踪，`std=0.02 m` |
| `directed_planar_progress` | 2.0 | 沿 command 方向的平面进度 |
| `planar_velocity_error` | -4.0 | 平面速度误差 |
| `wrong_way_velocity` | -8.0 | 与非零 command 反向运动 |
| `low_body_height_margin` | -1.0 | 低于预警高度的连续惩罚 |
| `orientation` | -1.0 | 投影重力水平分量平方和 |
| `mechanical_power` | -1e-4 | 主动关节 `sum(abs(torque * velocity))`，单位 W |
| `action_rate` | -0.01 | 相邻 raw action 的平方差 |
| `joint_position_limit` | -0.2 | 超过软位置限位的距离 |
| `joint_velocity_limit` | -0.05 | 超过 90% 软速度限位的距离 |
| `action_saturation` | -0.5 | raw action 超过 0.95 后的归一化平方代价 |
| `unsafe_termination` | -250.0 | 非 timeout 终止；在 50 Hz 下单次实际扣 5 |

Isaac Lab 将各项乘以环境步长 `0.02 s` 后累加。机械功项是瞬时功率代价，不是 episode 机械能；episode 机械能必须在评估中对功率乘步长并累计。

机身相对高度由局部支撑估计提供：至少两个轮接触时优先使用轮支撑高度中位数，否则使用有限射线命中高度中位数，均无效时回退到 environment origin。`support_valid` 仍需记录；它不等于接触物理资格已通过验收。轮接触阈值为 1 N，机身碰撞阈值为 8 N。

Episode 上限为 30 s（50 Hz 控制下 1500 步）。当前不安全终止包括：机身相对支撑高度低于 `0.06 m`、倾斜超过 `1.05 rad`、或机身接触力超过 `8.0 N`。这些终止不应与 timeout 混为一类。

## 验证

纯张量合同：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python -m pytest -q tests/test_policy_action_contracts.py
```

Isaac Sim 运行时合同：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/verify_complex_env.py --device cpu --num_envs 1 --steps 5
```

运行时检查要求 Command Manager 含 `locomotion`，Actor command 段与 manager 输出相同，观测和 reward 有限，并输出完整 reward term 列表。

## Training Metrics 与 Validation

训练入口在 RSL-RL wrapper 前接入 `TrainingMetricsWrapper`。每个结束 episode 的已启用指标写入 `extras["log"]`，交给 RSL-RL 的 TensorBoard/W&B logger。正式实验启用下列全部指标组；开关合同见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)：

- `train/locomotion/*`：四轴 RMSE、实际速度/高度均值、前向位移；
- `train/safety/*`：非 timeout 终止、timeout、base-height failure、姿态 mean/RMS/P95；
- `train/actuation/*`：腿/轮 action RMS、action rate、饱和率、关节限位率、轮腿机械能；
- `train/reward/raw/*` 与 `train/reward/weighted/*`：每个 reward term 的未加权和加权 episode 累积；
- `train/terrain/*` 与 `train/runtime/*`：terrain family/level/difficulty、有限值状态。

固定 validation 使用 [validate_policy.py](../scripts/validate_policy.py)：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/validate_policy.py \
  --checkpoint output/models/<task>/<run>/model_<N>.pt \
  --num-envs 64 --episode-steps 1500 --seed 1001 \
  --output output/evaluation/<run>/seed_1001.json
```

它不执行 PPO 更新，报告 `ppo_updates=0`，依次测试 `stop`、`forward`、`backward`、`left_turn`、`right_turn`，并为八类 terrain family 计算 macro/worst 聚合。只有五种 command 全部存在、所有 episode 完成、八类地形均被覆盖且 `runtime/all_finite=1` 时，`selection_evidence.eligible_for_checkpoint_comparison` 才为 true。该字段只证明评估数据结构完整、可以横向比较，不表示策略达到性能验收线。零策略可作为流程和基线检查，但不能证明学习后的 locomotion 能力。
