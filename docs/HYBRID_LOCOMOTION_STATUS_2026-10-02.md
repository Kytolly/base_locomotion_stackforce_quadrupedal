# Hybrid Locomotion 当前状态与完善报告

> **历史快照（2026-10-02）：** 本文数值、checkpoint 和结论仅适用于当时的 Hybrid/complex 基线与测试条件，不代表当前 `E0-v1-46d` run、46D Actor 配置或最终验收状态。历史数据保留原值，不用于推断当前训练趋势。

## 1. 报告范围

本报告冻结截至 2026-10-02 的证据、结论和实施边界，基线是复杂地形任务
`Base-Locomotion-Stackforce-Quadrupedal-Complex-v0` 的 20k iteration 策略。报告回答三个问题：

1. 当前策略训练了什么，已经验证了什么；
2. 当前证据能否证明机器人具备 stepping locomotion 或单台阶越障能力；
3. 在保留四维高层 command、拒绝显式 gait/profile gate 的前提下，如何把 hybrid locomotion 方案落地。

本文只登记方案和验收口径。108 维 Actor 观测、接触传感器、前视 RayCaster、课程和域随机化目前尚未全部实现，因此不能把方案字段写成当前 checkpoint 已经使用的输入。

## 2. 当前基线

### 2.1 Checkpoint 与训练设置

| 项目 | 当前基线 |
| --- | --- |
| 任务 | `Base-Locomotion-Stackforce-Quadrupedal-Complex-v0` |
| checkpoint | `logs/rsl_rl/base_locomotion_complex/2026-09-30_22-43-18_directional_curriculum_seed42_v7_formal20k/model_19999.pt` |
| SHA-256 | `7f4de5fe669c9bccb92076de36a81f79d97d5d8965d4d35e231343f087b01ae9` |
| 并行环境 | 4096 |
| seed | 42 |
| 仿真频率 | 200 Hz，`dt=0.005 s` |
| policy decimation | 4，policy 频率 50 Hz |
| 训练长度 | 20,000 iterations，24 steps/env rollout |
| actor/critic | 两层 MLP，`128/128`，ELU |
| episode | 30 s |

该 checkpoint 使用训练时的 46 维 Actor 输入和 12 维动作，并不是本报告冻结的 108 维新合同。重新训练前不能直接加载它作为 108 维策略的兼容模型。

### 2.2 当前 Actor、Critic 和 Action

当前 Actor 为 46 维：4 维 command、30 维本体感知和 12 维上一动作。

```text
[command 4] + [base angular velocity 3]
+ [projected gravity 3] + [12 active joint positions]
+ [12 active joint velocities] + [previous action 12] = 46
```

Critic 读取 Actor 输入加上当前 16 维 privileged 组，合计 62 维。privileged 组包括 base linear velocity、相对支撑面的机身高度和 12 个主动关节的 applied torque；这些真值不可部署到 Actor。

动作保持 12 维：8 个腿部关节位置目标和 4 个轮关节速度目标。当前腿部策略动作裁剪约为 `+/-0.5 rad`，轮速度缩放为 `20 rad/s`。已有固定范围探针显示，将腿动作诊断性放宽到 `+/-1.2 rad` 可以提高机身相对抬腿高度，但这不是已经通过力矩、碰撞和闭链约束的训练配置。

### 2.3 当前 command 与随机化

四维 command 顺序已经固定为：

```text
[forward_velocity, lateral_velocity, yaw_rate, body_height]
```

当前复杂任务默认范围是：

| 分量 | 当前范围 |
| --- | --- |
| forward velocity | `0.15--0.45 m/s` |
| lateral velocity | `-0.10--0.10 m/s` |
| yaw rate | `-0.25--0.25 rad/s` |
| body height | 固定 `0.105 m` |

command 每 10 s 重采样一次，最大变化率为 `(0.5, 0.5, 0.5, 0.02)`，单位分别对应 m/s、m/s、rad/s、m/s。这个接口表达机身运动意图，但没有直接告诉策略何时抬腿或如何跨越台阶。

当前已经存在的窄范围 domain randomization 包括接触材料、base 质量约 +/-10% 和执行器增益约 +/-10%，并带有小幅本体观测噪声。执行器时延、动作保持、关节零偏、轮半径、初始接触组合、地形障碍参数和完整传感器缺失随机化尚未形成可验收合同。

## 3. 已完成验证与证据

### 3.1 固定 Plateau/Washboard 评测

| 赛道 | 几何结果 | 前向速度 RMSE | 动作饱和 | 总体验收 |
| --- | --- | ---: | ---: | --- |
| Plateau，seed 8101 | 完成，4/4 gate，无走廊越界 | 0.0848 m/s | 45.42% | Fail |
| Washboard，seed 8201 | 未完成，走廊越界 | 0.0881 m/s | 45.49% | Fail |

两次回放均为 35 s，command 为 `[0.28, 0, 0, 0.105]`，未发生 unsafe termination。动作饱和率远高于现有 5% 门槛，因此“几何上走完 Plateau”不能等同于策略通过 benchmark。

机器结果见 [Plateau JSON](../output/probes/stair_climb/plateau20k_nominal.json) 和 [Washboard JSON](../output/probes/stair_climb/washboard20k_nominal.json)，完整讨论见 [阶梯探针结果](STAIR_PROBE_RESULTS_2026-10-02.md)。

### 3.2 运动范围与单台阶探针

- 固定 base 的悬空扫描在当前 `+/-0.5 rad` 范围内测得每条腿约 18.84--18.87 mm 的机身相对抬升。
- `+/-1.2 rad` 的诊断扫描测得约 38.51--38.57 mm，说明仿真机构存在更大的腿部运动余量，但不证明自由 base 下能够保持平衡。
- 自由 base 的手工抬腿、前移、落足探针未越过 40 mm 单台阶；一次候选动作在第一条腿阶段停止，另一次因 base collision 结束。
- 冻结策略在 40 mm 台阶上出现前轮接近/部分上台、后轮仍在下方的现象；三段 40 mm 台阶也未完成连续上升。
- 轮式滚动基线同样未完成该单台阶。

这些结果排除了“仿真腿完全不能抬起”，但没有区分学习 gait 不足、动作/力矩范围不足、平衡控制不足和机构几何极限。当前结果不能证明机构不具备越障能力，也不能证明 20k 策略已经激活了 stepping。

### 3.3 Command suite

已有 64 环境、8 类地形、5 组 command 的验证全部保持有限值，但整体 gate 失败：

| command | forward RMSE | yaw RMSE | survival |
| --- | ---: | ---: | ---: |
| Stop | 0.0374 m/s | 0.0473 rad/s | 100% |
| Forward | 0.2331 m/s | 0.1374 rad/s | 82.81% |
| Backward | 0.2245 m/s | 0.1314 rad/s | 87.50% |
| Left turn | 0.0372 m/s | 0.2515 rad/s | 100% |
| Right turn | 0.0367 m/s | 0.2543 rad/s | 100% |

Forward/backward 跟踪误差和正金字塔 survival 说明当前 policy 的地形适应仍然有限。survival 只能说明没有立即倒下，不能说明发生了有效 stepping 或越障。

## 4. 当前结论与边界

### 已经有充分证据支持的结论

1. 轮端主动推进是当前 20k 策略最稳定、最常用的推进方式。
2. 机构和闭环仿真中存在可观测的腿部关节运动与一定的抬升余量。
3. 当前策略能在部分复杂地形上保持有限时间存活，并能完成 Plateau 的几何路径。
4. 当前动作饱和率、速度跟踪和台阶结果不足以宣称策略已经学会 stepping locomotion 或连续楼梯。

### 目前不能宣称的结论

- 不能把腿部周期运动直接解释为 lift-off、swing、touchdown 和 contact relocation。
- 不能把前轮上台、机身 pitch 变化或滚轮接触当作四足跨步成功。
- 不能由开环参考步态成功或失败推导出学习策略的最终能力。
- 不能由一次 nominal Plateau/Washboard 回放代表五 seed、随机化和长时泛化。

## 5. 评测工具限制与修正要求

现有 `stepping_only` 评测通过把 wheel action target 置零来近似关闭轮驱动。仅清零速度目标并不能严格证明轮关节 torque 为零，必须在 actuator 输出或关节力矩层增加强制零驱动断言，并记录轮端实际 torque、角速度和功率。

现有 cycle detector 还存在事件定义风险：若从非接触状态搜索下一个非接触样本，持续离地可能被错误标成 touchdown。新版本必须检测明确的 `contact=0 -> 1` transition，并要求摆动期间有正向足端/轮心前移、重新接触和接触点位置改变。

在这些修正完成前，stepping-only 结果只能标记为 `INCONCLUSIVE`，不能标记 `STEPPING_CAPABILITY=PASS`。每次 gate 需要同时保存 JSONL 轨迹、checkpoint SHA-256、command、terrain seed、轮端实际 torque 和视频。

## 6. 已批准的 108 维 Actor 观测合同

详细定义见 [Hybrid Locomotion 解决方案](HYBRID_LOCOMOTION_SOLUTION.md)。冻结排列如下：

| 区间 | 维度 | 内容 |
| --- | ---: | --- |
| `[0:4]` | 4 | 四维 command |
| `[4:34]` | 30 | 现有本体感知 |
| `[34:46]` | 12 | 上一动作 |
| `[46:50]` | 4 | FR/FL/RL/RR 接触标志 |
| `[50:54]` | 4 | 四个腿端/轮心法向力 |
| `[54:66]` | 12 | 四个腿端/轮心相对位置 |
| `[66:78]` | 12 | 四个腿端/轮心相对速度 |
| `[78:93]` | 15 | 前视地形相对高度 |
| `[93:108]` | 15 | 射线有效位 |

当前 `Foot_Link` 是闭链 USD 中承载轮关节的终端刚体。合同使用“腿端/轮心承载点”语义，不把它虚构成独立足底。接触阈值为 `F_normal > 1.0 N`，三步多数滤波；法向力按 `clip(F_normal / 20 N, 0, 1)` 归一化。15 条射线为 3 列 x 5 行，`x=[-0.30,0,+0.30] m`、`y=[0.15,0.35,0.55,0.75,0.95] m`，base 原点高度 `0.35 m`，相对高度按 `0.30 m` 裁剪归一化。

Actor 只使用仿真和真实部署都能获得的信号；Critic 可以额外使用精确接触、地形和力矩真值。新增的前视 RayCaster 与现有 `support_scanner` 分离，后者继续服务支撑估计、reward、termination 和评估。

## 7. 完善方案

### 7.1 Command 课程

保持四维 command，不增加 gait/profile 输入。推荐按 C0--C3 提升：

| 阶段 | 前向 | 横向 | yaw | body height | 地形 |
| --- | ---: | ---: | ---: | ---: | --- |
| C0 | 0.05--0.20 | +/-0.05 | +/-0.10 | 0.105--0.115 | 平地、轻粗糙 |
| C1 | 0.05--0.35 | +/-0.10 | +/-0.20 | 0.105--0.125 | 粗糙、坡、浅波浪 |
| C2 | 0.05--0.40 | +/-0.12 | +/-0.25 | 0.105--0.135 | 低台阶、正/反金字塔、washboard |
| C3 | 0.05--0.45 | +/-0.15 | +/-0.30 | 0.105--0.140 | 连续台阶、混合障碍 |

每级都要保持足够长的 command 段，确保策略能完成多个支撑/摆动周期；升级条件同时使用跟踪误差、存活、接触有效率、饱和率和越障成功率。

### 7.2 Domain randomization

在现有摩擦、质量和增益随机化之外，加入有边界的物理、执行器、观测和地形随机化：

- 质量、质心、关节刚度/阻尼/摩擦、执行器响应延迟和动作保持；
- 关节零偏、轮半径、轮关节方向、接触恢复和求解扰动；
- IMU/编码器噪声、观测延迟、接触丢帧和地形高度无效率；
- 障碍高度、宽度、间距、台阶起始距离、washboard 相位、摩擦和粗糙度；
- 初始 base 高度、pitch/roll、横向偏移和四个腿端接触组合。

每个阶段保留固定 nominal 集合和 randomized 集合，记录随机化摘要，避免把不可行参数混入训练或只报告最容易的轨迹。

### 7.3 Reward 与记录

command tracking、body height、姿态、关节限位、动作变化率、功率和安全终止继续作为基础目标。新增接触保持、有效 lift-off/swing/touchdown/contact relocation、滑移、障碍 clearance 和轮腿推进分解指标。

接触项先作为日志和 checkpoint 选择指标，再逐步加入 reward；不能用关节位置周期性代替真实接触事件，也不强迫某一种 trot、walk 或固定轮腿比例。策略仍直接输出 12 维动作，自主决定轮腿协同。

## 8. 分阶段落地计划

| 阶段 | 交付物 | 验收证据 |
| --- | --- | --- |
| P0 | 实现 108D 观测、有限值和维度断言 | observation contract JSON、零策略 smoke |
| P1 | 平地 C0 训练与 nominal 回放 | command tracking、接触/足端轨迹、无轮驱 stepping-only 对照 |
| P2 | C1/C2 command 课程与域随机化 | nominal/randomized 多 seed 指标，动作饱和率下降 |
| P3 | 低台阶、正/反金字塔、washboard 训练 | Plateau/Washboard JSONL 与视频，前后轮/腿端越障事件 |
| P4 | 评测修正后的 stepping-only、hybrid 和连续楼梯 | Gate A--D 全量矩阵、checkpoint SHA-256、可复现实验命令 |

## 9. 验收门

- **Gate A：平地 forward locomotion**。在 command 不变时稳定前进，并记录轮腿动作和接触周期。
- **Gate B：轮驱动受限**。降低轮摩擦或轮动作上限后，策略仍保持有限正向位移并增加腿部支撑/摆动。
- **Gate C：stepping-only**。实际轮 torque 为零，至少多个连续周期满足 lift-off、swing、touchdown、contact relocation，并产生稳定正向位移。
- **Gate D：单台阶与连续障碍**。随机起始距离和障碍参数下完成上台、后轮/后腿越障、恢复；连续楼梯必须逐级通过。

每个 gate 的最终判定同时需要：command tracking、base 位姿稳定、前向位移、有效接触事件、动作饱和率、轮腿功率、base collision 和 terrain clearance。视频是可视证据，轨迹和接触数据才是判定依据。

## 10. 状态表与优先级

| 工作项 | 状态 | 说明 |
| --- | --- | --- |
| 四维 command 保持 | 已实现/已冻结 | 顺序和语义不变 |
| 12 维 wheel-leg action | 已实现/已冻结 | 8 腿部位置 + 4 轮速 |
| 当前 46D Actor、62D Critic | 已实现/已验证 | 20k checkpoint 使用 |
| Plateau/Washboard 回放 | 已完成基线评测 | Plateau 几何完成但整体 Fail；Washboard 未完成 |
| 单台阶与连续楼梯 | 未通过 | 只有部分前轮上台，没有完整 traversal |
| 机构抬腿运动余量 | 已有诊断证据 | 固定 base 扫描，不代表自由 base 平衡能力 |
| 108D Actor 合同 | 已批准/文档冻结 | 代码和训练尚未迁移 |
| 接触/端点/前视射线观测实现 | 计划 | 需仿真与真实部署同构 |
| command C0--C3 课程 | 计划 | 需实现升级条件和日志 |
| 完整 domain randomization | 计划 | 需按阶段扩展并保留 nominal 对照 |
| 严格 stepping-only gate | 尚不可判定 | 先修正轮 torque 归零和 contact transition detector |
| 新 108D 策略重训 | 未开始 | 不能复用 46D checkpoint 直接验收 |

最高优先级是实现观测合同和严格评测，而不是继续用视频解释当前 20k 策略。随后先在平地验证 stepping-only 的可重复周期，再逐级加入 command 课程、随机化和低台阶训练，最后用相同 gate 验收 hybrid locomotion 和连续楼梯。

## 11. 相关资料

- [Hybrid Locomotion 解决方案](HYBRID_LOCOMOTION_SOLUTION.md)
- [阶梯探针结果](STAIR_PROBE_RESULTS_2026-10-02.md)
- [阶梯探针定义](STAIR_PROBE.md)
- [训练规模实测](TRAINING_SCALE_REPORT.md)
- [实验与日志约定](EXPERIMENT_PROTOCOL.md)
- [当前文档索引](INDEX.md)
