# Base Locomotion 项目文档

本文档目录是 `base_locomotion_stackforce_quadrupedal` 的当前状态和执行入口。它只记录本项目已经核对过的事实、实现边界和可复现的运行约定；设计规范的唯一来源仍是仓库根目录的 [`doc/locomotion/`](../../doc/locomotion/base_locomotion_spec.md)。

## 当前结论

截至 2026-09-29，项目已完成方案阶段的环境、接口、训练指标和固定验证流程实现。当前任务为 `Base-Locomotion-Stackforce-Quadrupedal-Complex-v0`：

- 机器人使用包内 `stackforce_quadrupedal_wheeled_robot_closed.usda`；
- 12 个主动关节按 8 个腿部位置动作和 4 个轮速度动作分组；
- 8 个膝关节保持被动，4 个 closure joint 保持物理闭环约束；
- policy observation 已按 4 维 Decision、30 维本体感知和 12 维上一动作组成 46 维，command 和 reward 共用 Command Manager 输出；
- critic 额外读取 16 维当前特权真值；特权量不进入 policy observation；
- 地形为 `random_rough`、`boxes`、正/反金字塔楼梯、金字塔斜坡、波浪、踏脚石和坑；
- training metrics 已接入两个 RSL-RL 训练入口，通过 `extras["log"]` 交给 W&B；
- 固定 validation 覆盖 5 组 command 和 8 类 terrain family，并输出 macro/worst-family 指标；
- CPU 单环境 metrics smoke 与 64 环境零策略 validation smoke 已通过。它们只验证流程和数据合同，不代表策略已训练成功。

## 代码目录职责

```text
env/robots/                 机器人资产路径、关节分组和 articulation 配置
env/terrains/               程序化地形族和 terrain generator 配置
mdp/action/                 动作顺序、缩放和裁剪合同
mdp/policy/                 策略指令状态、采样和观测项
mdp/observation/proprioception/  30 维可部署本体感知
mdp/observation/history/         12 维上一动作单步记忆
mdp/observation/privileged/      critic 专用仿真真值
mdp/reward/                 locomotion 奖励项
evaluation/metric/          训练/验证共享的 episode 指标与 W&B 日志封装
evaluation/validation/      固定 command/terrain 验证与 checkpoint 比较门控
benchmark/plateau_track/    plateau 固定赛道参数与 USD/PhysX 生成
benchmark/washboard_track/  washboard 固定赛道参数与 USD/PhysX 生成
```

两个 benchmark 已具备可注册任务、固定 seed 参数、Kit 可视化和 JSON 指标输出，但尚未用学习后的 checkpoint 完成性能验收。`evaluation/` 已具备完整流程实现。

参考任务 [`Wheel-Leg-Complex-Train-v0`](../../demo_wheel_leg_switching/) 仍只提供地形参数和验证方法；新任务的 smoke 结果只证明资产、地形和 ManagerBased 接口可运行，不证明 locomotion 已训练成功。

## 文档导航

1. [仿真就绪度与缺口矩阵](SIMULATION_READINESS.md)
2. [Policy、Action 与 Reward 合同](POLICY_ACTION_REWARD.md)
3. [实验、日志、模型和 W&B 运行约定](EXPERIMENT_PROTOCOL.md)
4. [Plateau 与 Washboard Benchmark](BENCHMARKS.md)
5. [Locomotion 验收标准](ACCEPTANCE_CRITERIA.md)
6. [仓库根目录 locomotion 规范](../../doc/locomotion/base_locomotion_spec.md)
7. [参考任务执行闭环](../../demo_wheel_leg_switching/docs/RL_CLOSED_LOOP.md)

## 实施顺序

```text
资产/坐标核验
  -> 12 维动作语义探针
  -> 4 维 Decision 与 30 维本体观测
  -> 平地闭环与安全终止
  -> 地形生成与支撑面估计
  -> 奖励、指标和 W&B 记录
  -> PPO smoke
  -> 固定验证集和 checkpoint 选择
  -> 长训练与模型导出
```

在前一阶段没有留下机器可读证据时，不进入后一阶段。长训练不得使用历史任务 checkpoint 初始化。

当前阶段只维护接口合同、实现缺口、验收门和实验目录约定；不启动 locomotion 训练，不导出模型，也不把模板 smoke 运行记为项目结果。
