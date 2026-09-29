# 仿真就绪度与缺口矩阵

## 判定

**状态：机器人、复杂地形、训练指标与固定验证流程已实现；PPO 训练和完整验收未完成。**

`doc/locomotion/` 是当前实现合同。目标任务已经能创建闭链四轮足机器人、8 类复杂地形、12 维动作和 46 维 policy observation；episode 指标通过 RSL-RL extras 接入 W&B，固定 command/terrain validation 也已可运行。动作语义证据、接触相关安全项、PPO checkpoint 和训练验收仍未完成，所以不能声称具备 `Wheel-Leg-Complex-Train-v0` 等价能力。

## 已核对的实现事实

| 能力 | 当前代码 | 结论 |
| --- | --- | --- |
| Gym 注册 | `tasks/.../__init__.py` | 新任务 `Base-Locomotion-Stackforce-Quadrupedal-Complex-v0` 已注册 |
| 机器人资产 | `env/robots/stackforce.py` | 包内闭链 USDA 已加载；12 active / 8 passive / 4 closure |
| 动作空间 | `mdp/action/policy.py` 中的 `RobotActionCfg` | policy 输出 12 维；前 8 维腿位置、后 4 维轮速度 |
| 策略观测 | `mdp/observation/proprioception/` 与 `history/` | 4 维 Decision + 30 维本体感知 + 12 维上一动作，共 46 维 |
| Critic 特权观察 | `mdp/observation/privileged/` | 当前 16 维，仅供 critic；完整特权合同尚未冻结 |
| 指令与奖励 | `mdp/policy/`、`mdp/reward/` 与 `RewardsCfg` | Command Manager 已统一 Actor/reward 指令；四轴跟踪、姿态、限位、功率、饱和和终止项已拆分，训练 metrics 已接入 W&B/RSL-RL |
| 地形 | `env/terrains/complex.py` | 8 类等比例，8×8 tiles，difficulty 0.0-0.6 |
| 训练配置 | `agents/rsl_rl_ppo_cfg.py` | 实验名已切换，PPO 参数尚未最终冻结 |
| W&B | `ComplexPPORunnerCfg` 与 `scripts/rsl_rl/cli_args.py` | complex runner 默认启用，正式运行仍显式传参 |
| 运行日志 | `scripts/rsl_rl/train*.py` | 运行目录写入 `logs/rsl_rl/<experiment>/<run>` |
| 模型 | RSL-RL runner | checkpoint 与该运行目录共存，尚无 `output/` 导出流程 |

## 必须补齐的实现合同

| 阶段 | 交付物 | 通过条件 | 证据位置 |
| --- | --- | --- | --- |
| G0 资产 | USD/网格、关节顺序、质量、惯量、坐标和接触几何 | 已完成导入和最小运行；仍需形成正式证据报告 | `logs/` 原始启动日志，`output/` 检查报告 |
| G1 动作 | 12 维 raw -> processed -> actuator trace | 缩放、裁剪、限位、延迟和零动作语义可观测 | `logs/action_probe_*.log`，`output/action_probe_*.json` |
| G2 观测 | 4 维 Decision + 30 维 proprioception + last action | Actor 输入维度为 46，部署侧不读特权真值 | `output/observation_contract.json` |
| G3 平地闭环 | 速度、偏航、高度跟踪和安全终止 | 停止/前进/后退/左转/右转均改变行为 | `logs/flat_*.log`，`output/flat_*.json` |
| G4 地形 | 八类程序化地形、难度和支撑面拟合 | 接触滞回、支撑面有效性和终止原因可记录 | `output/terrain_*.json` |
| G5 奖励与指标 | 跟踪、稳定、动作率、限位、饱和、机械功和 reward 分解 | 已通过 smoke：`TrainingMetricsWrapper` 每 episode 写入 `train/*`，含原始/整形 command、地形、终止和有限值状态；正式 W&B run 待 PPO smoke | W&B + `logs/` |
| G6 PPO smoke | OmegaConf YAML、固定 seed、短 rollout、有限值检查 | 8 env × 24 steps 的 1 iteration W&B offline smoke 已通过并生成 checkpoint；恢复和长期训练仍待验证 | `logs/training_yaml_wandb_metrics_final.log`，W&B offline run |
| G7 验证与导出 | 固定五 command、八类 terrain、Plateau/Washboard、checkpoint 选择 | validation 与两个固定赛道已通过零策略流程 smoke；真实 checkpoint 性能和导出仍待训练后验证 | `output/evaluation/`、`output/benchmark/`，W&B |

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

运行时会提示 12/20 joints actuated；其中未配置 actuator 的 8 个膝关节按 `sf_quad` 合同为被动关节，该提示符合闭链机构设计。进入正式训练前仍需补齐 G1 动作证据和 G4 接触/支撑面验证；G5 已完成代码与 smoke，尚待正式 W&B PPO run 验证。

PhysX 还会报告无法在 `/World/envs/env_0/Robot/Geometry/base_link/base_link` 找到 articulation。当前 articulation 随后能够初始化并完成有限步 rollout，但该资产层级告警尚未解决，必须在 G0 正式证据报告中关闭或证明不影响接触与动力学语义。
