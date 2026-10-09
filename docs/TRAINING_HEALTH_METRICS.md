# 训练健康度参考标准

本文档用于日常 PPO 训练监控、W&B 面板配置和 checkpoint 初筛，不替代最终策略验收。

- 健康度关注数值稳定、学习趋势和动作执行质量。
- locomotion 性能 PASS 还必须通过固定 command、八类地形和 Plateau/Washboard 跑道。
- 最终 PASS 阈值的唯一来源是 [`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md)。
- 趋势应查看 100--250 个 PPO iteration 的滑动平均，不用单个 iteration 下结论。

## 核心监控与扩展诊断

解析器推荐默认仅开启 `core`、`safety`、`runtime`。PPO 内建 reward、episode length、loss、KL 和耗时始终由 runner 记录；扩展指标不是精简面板的必需项。当前工作区 E0 YAML 全组为 true，但训练入口的二次解析会重置自定义开关，详见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。修复该限制并核对实际日志键后，才能依赖扩展组做日常诊断。

| 用途 | 自定义组 |
| --- | --- |
| 跟踪 RMSE、实际运动、有限值与安全终止 | `core`、`runtime`、`safety` |
| 原始/整形 command | `command` |
| 支撑有效性与轮接触 | `support` |
| 动作饱和、限位、功率 | `actuation` |
| 地形等级/难度 | `terrain` |
| 运动意图覆盖 | `motion` |
| reward 分项 | `reward` |

下文列出完整诊断目录，不表示这些指标默认全部采集或显示。固定冻结策略评估应使用完整验收指标，不由训练的精简显示配置替代。

## 运行时健康

| 指标 | 健康标准 | 红色警报 |
| --- | --- | --- |
| `train/runtime/all_finite` | 始终为 `1` | 出现 `0` |
| `train/runtime/finite_rate` | `100%` | 低于 `100%` |
| reward、value loss、surrogate loss | 有限且无持续突增 | NaN/Inf 或发散 |
| PPO KL | 有界并围绕目标值变化 | 持续爆炸或频繁裁剪 |
| GPU 显存 | 小于总显存约 `90%` | 接近满显存或 PhysX buffer error |
| iteration 时间 | 大致稳定 | 持续增长或长时间无日志 |
| W&B history | 持续有新记录 | 长时间无记录或 run 异常结束 |

最低要求：

```text
train/runtime/all_finite = 1.0
train/runtime/finite_rate = 1.0
没有 NaN、Inf、PhysX buffer error 或未解释的进程退出
```

## 耗时与 GUI 判读

性能诊断应分开比较 collection/simulation 与 PPO learning。近期约 5 s/iteration 的运行主要耗时在采样/仿真，不能仅凭关闭 W&B 面板预期显著提速；需以同配置、同环境数的稳态计时验证。历史规模探针见 [TRAINING_SCALE_REPORT.md](TRAINING_SCALE_REPORT.md)，启动时间和不同配置不能混算。

`launcher.max_visible_envs` 只减少可见环境，不减少并行仿真数。日志无 Error/Fatal、有限值正常只能证明运行与数值状态；不能证明 Kit 视口中的姿态、接触或运动正常。GUI 行为需实际回放观察，策略趋势需冻结 checkpoint 对比，不能把进程存在、W&B 上传成功或总 reward 上升当作 locomotion 改善。

## PPO 学习趋势

健康训练不要求每个指标单调上升，但滑动窗口内应表现为：生存能力提高、command tracking 误差下降、有效运动 reward 提高。

| 指标 | 健康趋势 | 需要调查的情况 |
| --- | --- | --- |
| `Mean reward` | 早期上升，后期稳定平台 | reward 上升但运动指标不改善 |
| `Mean episode length` | 逐步接近上限 | reward 上升但 episode length 不变 |
| `train/safety/timeout` | 逐步上升 | 长期低于 `80%` |
| `train/safety/unsafe_termination` | 逐步下降 | 长期高于 `10%`，后期高于 `5%` |
| `train/locomotion/forward_velocity_rmse_mps` | 持续下降 | 不低于零动作/静止策略基线 |
| `train/locomotion/yaw_rate_rmse_radps` | 持续下降 | 长期高于 `0.20 rad/s` |
| `train/reward/weighted/track_forward_velocity` | 随训练上升 | 总 reward 上升但该项不升 |
| `train/terrain/level` | 缓慢、稳定上升 | level 上升同时 survival 或 tracking 恶化 |
| `train/terrain/difficulty_midpoint` | 与 terrain level 同步 | 难度升高后指标突然退化 |

### 阶段性参考线

这些是训练过程参考线，不是最终验收门槛。

| 阶段 | 建议参考线 |
| --- | --- |
| 初期（0--20%） | episode length > 最大长度 `50%`；unsafe termination < `30%`；forward RMSE < `0.30 m/s` |
| 中期（20--70%） | episode length > `90%`；timeout > `90%`；unsafe termination < `10%`；forward RMSE < `0.22 m/s`；lateral RMSE < `0.12 m/s`；yaw RMSE < `0.25 rad/s` |
| 后期（70--100%） | timeout > `95%`；unsafe termination <= `5%`；forward RMSE <= `0.15 m/s`；lateral RMSE <= `0.12 m/s`；yaw RMSE <= `0.10 rad/s`；body-height RMSE <= `0.025 m` |

## 动作与执行器健康

动作饱和不能被 mean reward 掩盖。它直接反映上层 command 是否转换成可控的 12D 执行器目标。

| 指标 | 健康参考 | 解释 |
| --- | ---: | --- |
| `train/actuation/leg_action_rms` | 稳定且不长期贴近 `1` | 过高可能表示腿部目标被裁剪 |
| `train/actuation/wheel_action_rms` | 随 command 和 terrain 改变 | 始终很高但实际速度低表示无效输出 |
| `train/actuation/action_saturation_rate` | `< 10%` 健康；`10--20%` 观察；`> 20%` 不健康 | 最终标准为 `<= 5%` |
| `train/actuation/action_rate_rms` | 平滑且不过度抖动 | 突然升高表示控制不稳定 |
| `train/actuation/joint_position_limit_rate` | 接近 `0` | 持续超过 `1%` 需调查 |
| `train/actuation/joint_velocity_limit_rate` | 接近 `0` | 持续超过 `5%` 需调查 |
| mechanical power | 与有效位移同步增加 | 功率升高但位移不增表示浪费或打滑 |

经验判断：

```text
action_saturation_rate < 10%          健康
10% <= action_saturation_rate <= 20%  观察
action_saturation_rate > 20%           不健康
action_saturation_rate > 50%           基本可判定执行器控制异常
```

## 支撑、接触与稳定性

| 指标 | 最终参考标准 |
| --- | ---: |
| `train/support/wheel_contact_fraction` | `>= 50%` |
| `train/support/invalid_rate` | `<= 1%` |
| `train/safety/base_collision_rate` | `<= 1%` |
| `train/safety/base_tilt_max_p95_rad` | `<= 1.05 rad` |
| `train/locomotion/body_height_rmse_m` | `<= 0.025 m` |
| 正常 terrain 的 timeout survival | `>= 95%` |
| stepping stones、pit 等困难 terrain | 不应长期低于 `90%` |

`wheel_contact_fraction` 只能证明轮子经常接触地面，不能证明接触产生有效滚动。必须与 `forward_velocity_rmse`、`wheel_action_rms` 和 `action_saturation_rate` 联合判断。

## Command 跟踪

不要单独用下列带符号均值判断运动能力：

```text
train/command/raw_forward_mean
train/locomotion/actual_forward_mean
train/locomotion/forward_distance_m
```

正向、反向和停止 command 混合后，这些均值可能相互抵消。优先观察：

```text
train/locomotion/forward_velocity_rmse_mps
train/locomotion/lateral_velocity_rmse_mps
train/locomotion/yaw_rate_rmse_radps
```

固定 command 的条件化验证应满足：

| Command | 应观察的行为 |
| --- | --- |
| `+0.35 m/s` | 前向速度为正，RMSE 下降，累计位移为正 |
| `-0.35 m/s` | 前向速度为负，RMSE 下降 |
| `0 m/s` | 低速度、低漂移，不持续爬行 |
| `+0.25 rad/s` | yaw rate 为正 |
| `-0.25 rad/s` | yaw rate 为负 |

## Terrain curriculum

`train/terrain/level` 上升只能说明 curriculum 提高了分配难度，不能单独证明 locomotion 变好。只有当 level 上升、forward RMSE 不恶化、unsafe termination 不上升、timeout survival 不下降时，level 上升才可视为健康。

如果出现 `terrain/level` 上升、`action_saturation_rate` 上升、yaw RMSE 上升、unsafe termination 上升且 episode length 下降，应暂停 curriculum 或回退 checkpoint 调查。

## 跑道和最终验收

Plateau 与 Washboard 必须满足完整轨迹条件，不能只看累计位移：

```text
episode 正常 timeout
所有 track gates 按顺序通过
没有横向走廊越界
没有 unsafe termination
command 在 rollout 内保持固定
runtime_all_finite = 1
```

最终冻结策略的数值门槛为：

| 指标 | PASS |
| --- | ---: |
| forward velocity RMSE | `<= 0.15 m/s` |
| lateral velocity RMSE | `<= 0.12 m/s` |
| yaw-rate RMSE | `<= 0.10 rad/s` |
| body-height RMSE | `<= 0.025 m` |
| unsafe termination | `<= 5%` |
| base collision | `<= 1%` |
| invalid local support | `<= 1%` |
| wheel contact fraction | `>= 50%` |
| action saturation | `<= 5%` |
| base tilt maximum p95 | `<= 1.05 rad` |
| runtime finite rate | `100%` |

上述数值还必须在五组 command、八类 terrain family 和重复的 nominal/randomized Plateau/Washboard benchmark 中满足项目验收规则。详情见 [`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md) 和 [`BENCHMARKS.md`](BENCHMARKS.md)。

## 推荐的 W&B 精简面板

精简训练面板只关注以下核心指标和 runner 内建标量：

```text
train/runtime/all_finite
train/safety/unsafe_termination
train/safety/timeout
train/locomotion/forward_velocity_rmse_mps
train/locomotion/yaw_rate_rmse_radps
train/locomotion/body_height_rmse_m
Mean reward
Mean episode length
```

地形课程异常时再诊断 `terrain/level`、`difficulty_midpoint`；输出质量异常时诊断 `actuation/action_saturation_rate`、`wheel_action_rms` 与 `support/wheel_contact_fraction`。这些扩展组的开关须先确认未被入口二次解析覆盖。

最简判断规则：

> 健康训练要求 `all_finite` 始终为 1，episode length 接近上限，unsafe termination 下降，command tracking RMSE 下降，action saturation 不持续上升，并且 terrain level 上升时其他指标不恶化。最终好坏必须由固定 command 和 Plateau/Washboard 验收决定，不能由 Mean reward 或 terrain level 单独决定。
