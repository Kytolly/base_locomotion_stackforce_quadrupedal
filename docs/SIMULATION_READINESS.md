# 仿真就绪度与缺口矩阵

## 判定

**状态：核心闭环接口与训练验证基础已实现；物理动作资格、长期 PPO 与性能 PASS 仍未完成。**

`doc/locomotion/` 是当前实现合同。目标任务创建闭链四轮足机器人、8 类复杂地形、12 维动作和 46 维 policy observation；训练命令覆盖停止、前后、横移、原地转向和弧线运动。相对支撑高度、wheel/base contacts、安全终止、表现驱动 terrain curriculum 与基础 DR 已进入环境代码。Episode 指标进入 RSL-RL extras/W&B；仿真动作 trace 和 Plateau/Washboard 有序检查点验收可运行。长期 PPO、轮接触/滑移动力学资格、真实策略 benchmark PASS 仍未完成，因此不能声称具备参考任务等价性能。

## 已核对的实现事实

| 能力 | 当前代码 | 结论 |
| --- | --- | --- |
| Gym 注册 | `tasks/.../__init__.py` | 新任务 `Base-Locomotion-Stackforce-Quadrupedal-Complex-v0` 已注册 |
| 机器人资产 | `env/robots/stackforce.py` | 包内闭链 USDA 已加载；12 active / 8 passive / 4 closure |
| 动作空间 | `mdp/action/policy.py` 中的 `RobotActionCfg` | policy 输出 12 维；前 8 维腿位置、后 4 维轮速度 |
| 策略观测 | `mdp/observation/proprioception/` 与 `history/` | 4 维 Decision + 30 维本体感知 + 12 维上一动作，共 46 维 |
| Critic 特权观察 | `mdp/observation/privileged/` | 当前 16 维，仅供 critic；完整特权合同尚未冻结 |
| 指令与奖励 | `mdp/policy/`、`mdp/reward/` 与 `RewardsCfg` | 7 种运动意图共享一个 rate-limited command；机身高度使用轮接触/射线估计支撑面，Actor 不读取仿真真值 |
| 地形 | `env/terrains/complex.py`、`mdp/curriculum.py` | 8 类等比例，初始等级限于 0-1；episode 位移和安全结果驱动升降级，difficulty 0.0-0.6 |
| 接触与安全 | `ComplexSceneCfg`、`mdp/support.py`、`mdp/terminations.py` | 四轮接触、机身碰撞、支撑有效性指标；相对支撑高度、过度倾斜和机身接触终止已配置，接触阈值仍需物理资格验收 |
| Domain randomization | `EventCfg`、`ObservationsCfg` | 摩擦、基座质量、执行器增益采用窄范围启动随机化；本体传感项有小幅噪声；延迟未随机化，需先完成 actuator 时序标定 |
| 训练配置 | `agents/rsl_rl_ppo_cfg.py` | 实验名已切换，PPO 参数尚未最终冻结 |
| W&B | `ComplexPPORunnerCfg` 与 `scripts/rsl_rl/cli_args.py` | complex runner 默认启用，正式运行仍显式传参 |
| 运行日志 | `scripts/rsl_rl/train*.py` | 运行目录写入 `logs/rsl_rl/<experiment>/<run>` |
| 模型 | RSL-RL runner | checkpoint 与该运行目录共存，尚无 `output/` 导出流程 |

## 必须补齐的实现合同

| 阶段 | 交付物 | 通过条件 | 证据位置 |
| --- | --- | --- | --- |
| G0 资产 | USD/网格、关节顺序、质量、惯量、坐标和接触几何 | 已完成导入和最小运行；仍需形成正式证据报告 | `logs/` 原始启动日志，`output/` 检查报告 |
| G1 动作 | 12 维 raw -> processed -> actuator trace | 用 `scripts/probe_robot_action.py` 检查每个输出通道的 target、关节响应、扭矩与机身运动；人工核对符号、闭链和零动作语义 | `output/action-probes/latest.json` |
| G2 观测 | 4 维 Decision + 30 维 proprioception + last action | Actor 输入维度为 46，部署侧不读特权真值 | `output/observation_contract.json` |
| G3 平地闭环 | 速度、偏航、高度跟踪和安全终止 | 停止/前进/后退/左转/右转均改变行为 | `logs/flat_*.log`，`output/flat_*.json` |
| G4 地形 | 八类程序化地形、难度和局部支撑估计 | 轮接触/射线可用率、轮支撑比例、机身碰撞率和终止原因可按 terrain 记录；接触阈值及 slope/obstacle 物理表现需验收 | W&B + `output/evaluation/` |
| G5 奖励与指标 | 跟踪、稳定、动作率、限位、饱和、机械功和 reward 分解 | 已通过 smoke：`TrainingMetricsWrapper` 每 episode 写入 `train/*`，含原始/整形 command、地形、终止和有限值状态；正式 W&B run 待 PPO smoke | W&B + `logs/` |
| G6 PPO smoke | OmegaConf YAML、固定 seed、短 rollout、有限值检查 | 8 env × 24 steps 的 1 iteration W&B offline smoke 已通过并生成 checkpoint；恢复和长期训练仍待验证 | `logs/training_yaml_wandb_metrics_final.log`，W&B offline run |
| G7 验证与导出 | 固定五 command、八类 terrain、Plateau/Washboard、checkpoint 选择 | Plateau/Washboard 依序通过障碍检查点、保持赛道走廊、无不安全终止并完整结束 episode；性能阈值按 `docs/ACCEPTANCE_CRITERIA.md` 判定，尚无策略 PASS 证据 | `output/evaluation/`、`output/benchmark/`，W&B |

## 与参考任务的边界

参考任务提供八类地形、200 Hz/50 Hz 时序、PPO 训练形状、episode 指标和 W&B 组织方式。新项目必须重新测量资产、执行器、接触阈值、吞吐和稳定性；参考任务的 checkpoint、日志和 W&B run 不得复制为本项目结果。

## 当前最小运行检查

在 `env_isaaclab` 环境中用以下命令检查注册和有限步 rollout。它们验证资产/地形接入，不是 locomotion 训练验收：

```bash
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/base_locomotion_stackforce_quadrupedal:${PYTHONPATH:-}"
python scripts/list_envs.py --keyword Base-Locomotion-Stackforce-Quadrupedal
python scripts/verify_complex_env.py --device cpu --num_envs 1 --steps 5
```

运行时会提示 12/20 joints actuated；其中未配置 actuator 的 8 个膝关节按 `sf_quad` 合同为被动关节。单环境 smoke 已验证五个接触传感器分别绑定到四个 `Foot_Link` 和 `base_link`，并通过两步 rollout 与训练 metrics 检查；12 个动作通道的目标缩放/路由探针也通过。当前关节运动和 12D target 路由可用于方案阶段训练；执行器增益、饱和及延迟仍是仿真资格参数，尚未完成真机标定。当前 smoke 不等于四轮承载、滑移或机身碰撞阈值已验收。

`/World/envs/env_0/Robot/Geometry/base_link` 是机身刚体；其子级 `/World/envs/env_0/Robot/Geometry/base_link/base_link` 是刚体连接件的 Mesh 叶节点，不是关节、执行器或独立刚体。PhysX 的 articulation/body lookup warning 来自对该 Mesh 叶节点的刚体查询，不表示资产路径缺失。动作验收只检查 8 个髋关节和 4 个轮关节；接触与支撑传感器绑定刚体节点或四个 `Foot_Link`，不得绑定该 Mesh 叶节点。
