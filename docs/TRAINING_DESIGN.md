# 当前仿真训练设计

本文说明 `E0-v1-46d` 主线的训练闭环与能力边界。执行命令和日志配置见 [实验协议](EXPERIMENT_PROTOCOL.md)，接口与 reward 数值见 [Policy、Action 与 Reward 合同](POLICY_ACTION_REWARD.md)。

## 目标与信息边界

最终主模型配置为 `configs/train/main/b2_bisec_full.yaml`，注册任务为 `Base-Locomotion-Stackforce-Quadrupedal-Complex-v0`。目标是在不同地形上跟踪机身运动意图，保持稳定，并输出可控的轮腿执行器目标；不是给策略指定 gait 或腿序。

```text
4D command + 30D 本体状态 + 12D 上一动作
  -> 46D Actor -> 8D 腿位置增量 + 4D 轮速度
  -> 闭链机器人仿真 -> reward / 安全终止 -> PPO 更新
                          -> episode 指标 / 冻结策略评估
```

Actor 不读取地形扫描、接触标志、足端状态或机身线速度真值。Critic 使用同一观测合同并额外读取 16D 特权量：3D 机身线速度、1D 相对支撑高度、12D 主动关节力矩，共 62D。场景内接触与支撑传感器用于 reward、termination、课程和评估，不因此进入 Actor。

机器人有 12 个主动关节、8 个被动膝关节和 4 个闭环约束关节。项目局部坐标的前向为 `+Y`。

## 并行 PPO

| 参数 | E0 配置 |
| --- | --- |
| 并行环境 | 4096 |
| 物理 / 策略频率 | 200 Hz / 50 Hz，decimation=4 |
| Episode | 30 s，最多 1500 个控制步 |
| Rollout | 每环境 24 步，每 iteration 共 98304 个 transition |
| 训练预算 | 20000 iterations，checkpoint 间隔 250 |
| Actor / Critic | 各 `[128, 128]`，ELU |
| 动作分布 | Squashed Gaussian，latent std 初值 0.3，范围 0.05--0.5 |
| PPO | 5 epochs，4 minibatches，clip=0.2 |
| 学习率 | 0.001，adaptive，desired KL=0.01 |
| 折扣 / GAE | gamma=0.99，lambda=0.95 |

运行时用随机 episode 初始长度分散 reset。`launcher.viz=none` 是 E0 YAML 的显示默认值；W&B 为 online 且 enabled=true。正式实验开启全部自定义指标和面板组。

## 地形与课程

八类地形等比例混合：`random_rough`、`boxes`、`pyramid_stairs`、`pyramid_stairs_inv`、`hf_pyramid_slope`、`wave`、`stepping_stones`、`pit`。生成器为 8×8 网格，difficulty 范围 0.0--0.6，初始 terrain level 限于 0--1。它不是固定的“平地 -> 坡道 -> 台阶”顺序课程。

课程实现分别位于 `mdp/curricula/`：

- `terrain.py`：根据 episode 沿 command 方向的进度、平面跟踪/yaw RMSE、timeout 和不安全终止升降地形等级；停止不触发升级。
- `motion.py`：对 stop、forward、backward、lateral、yaw_only、curved_forward、curved_backward 七种意图统计成功率并重平衡采样，保留 rehearsal floor。
- `robustness.py`：根据地形等级记录分组编号：低等级为 0，中间为 1，stress 为 3；当前不产生编号 2。它不逐阶段扩展物理随机化范围，`moderate_level` 未参与分箱判定。

`configs/curriculum/*.yaml` 是分课程参数参考。启动器实际应用训练 YAML 的 `curriculum` 块，不自动加载或合并这些独立文件。

## 稳定性与泛化

Command 每 10 s 重采样，并限速整形；前后、横移、原地转向、弧线和停止共同训练。奖励同时约束速度/高度跟踪、沿意图运动、反向运动、姿态、动作变化、限位、饱和和机械功。安全终止独立于总 reward，数值门槛见接口合同。

基础随机化包括摩擦、机身质量约 ±10%、执行器增益约 ±10% 及本体观测噪声。当前不包含完整的延迟、零偏、轮半径等随机化合同，也不等于完成真机执行器标定。

46D 策略可以通过本体反馈学习轮腿协同，但没有前视地形输入，不能据此声称已学会 stepping 或复杂越障。训练 reward 上升不等于验收通过；需冻结 checkpoint，完成五 command、八地形及 Plateau/Washboard 验证。108D [Hybrid 方案](HYBRID_LOCOMOTION_SOLUTION.md) 与此主线独立，不是当前 E0 的输入升级。
