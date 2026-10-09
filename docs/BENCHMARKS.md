# Plateau 与 Washboard Benchmark

两个 benchmark 复用复杂地形训练任务的 46 维 Actor 输入、62 维 Critic 输入和 12 维动作合同。评估加载冻结 checkpoint，不执行 PPO 更新。

这是冻结策略评估入口，不是 E0 训练配置。当前没有本页完整验收流程通过的 E0 checkpoint 证据；历史穿越结果不能替代重复 benchmark 和数值门槛。

Plateau version 2 的所有坡面使用朝外法向。评测指标在物理步结束、自动 reset 之前采集，包含终止帧；局部支撑优先使用接触轮位置，回退查询覆盖实际坡面、圆柱和边界块，不把跑道下方的平面作为跑道高度。动作分布从 checkpoint 同目录的 `agent.yaml` 加载，旧 Gaussian 和新有界策略不可互换解释。

报告分别给出 `traversal_complete`（穿越）和 `performance_pass`（数值门槛）；两者及安全、固定命令等条件都满足才有 `success=true`。阈值由 `benchmark/acceptance.py` 实现，参见 [验收标准](ACCEPTANCE_CRITERIA.md)。五个留出 seed 的批量入口见 [控制修正与重启](CONTROL_RESTART.md)。

## 任务与场景

| 任务 | 固定场景 | 默认 seed | 赛道长度 |
| --- | --- | ---: | ---: |
| `Base-Locomotion-Stackforce-Quadrupedal-Plateau-v0` | 平地接近、上坡、平台、下坡、释放段 | 8101 | 6.50 m |
| `Base-Locomotion-Stackforce-Quadrupedal-Washboard-v0` | 平地接近、6° 入口坡、双排错位瓶阵、6° 出口坡、释放段 | 8201 | 5.4294 m |

### Washboard 几何合同（version 2）

参数采样和瓶阵建模对应 `demo_mode_switching/source/wheel_leg_demo/benchmark_tracks.py` 的 `build_washboard_benchmark` 及 `envs/mdp/events.py` 的几何构造，不依赖该项目运行时。赛道沿世界 +Y，瓶轴沿 +X 横向铺设，而不是竖直立瓶。

- 名义赛道宽 2 m，左右两排中心为 X = ±0.5 m；每排 7 个瓶位，圆柱长 1 m、半径 0.075 m，相邻中心距 0.15 m。
- 两排沿行进方向相差半个直径（0.075 m）。领先排仅入口端截断，另一排仅出口端截断；其余 12 个瓶位均为完整 `UsdGeom.Cylinder`。
- 两处截断用矩形实体块，尺寸为横向 1 m、纵向 0.075 m、高 0.15 m。块体与两端坡的瓶床边界齐平；每排仅一端截断，不额外添加横跨路径的竖直挡墙。
- 接近段和释放段各 0.8 m，基面高 0.005 m。两端坡各长 1.4272 m、升高 0.15 m，顶端与瓶冠和边界块顶面衔接。瓶床长 0.975 m；圆柱与边界块为静态碰撞体。静/动摩擦系数分别为 0.72/0.62。
- 随机模式复用参考项目的半径、瓶长、间距、数量、坡度、摩擦、领先排和单瓶位置/高度/朝向扰动。均匀分布及严格相切描述的是名义跑道；随机扰动允许出现小间隙和高度差。

通过证据依次覆盖入口坡顶、瓶床末端、出口坡底和释放段末端四个检查点。固定 locomotion command 保持不变；不引入参考项目的分段变速命令。前后倾及俯仰振荡属于预期运动现象，不替代穿越和安全判定。

比较结果时必须检查 JSON 中 `track_parameters.version` 及完整几何参数。version 1 的单排圆柱跑道结果不能直接作为 version 2 的能力结论；几何合同通过也不代表训练策略已通过跑道。

YAML 默认 `launcher.viz: kit`，因此验收命令会打开 Isaac Sim GUI。正式验收对 nominal 和 randomized geometry 各使用五个不同 seed；每个 run 都必须满足下述 traversal 判定：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/run_benchmark.py \
  --config configs/benchmark/plateau.yaml \
  policy.zero_policy=false policy.checkpoint=<checkpoint>
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/run_benchmark.py \
  --config configs/benchmark/washboard.yaml \
  policy.zero_policy=false policy.checkpoint=<checkpoint>
```

无显示 smoke 可追加 `launcher.viz=none launcher.device=cpu benchmark.max_steps=5`。输出 JSON 位于 `output/benchmark/<track>/`，记录固定 command 的期望值与 rollout 实测范围、进度、成功判定、跟踪 RMSE、动作饱和率、轮腿机械能、终止类型、有限值状态、checkpoint 哈希和完整赛道参数。两个赛道固定使用 `[0.28 m/s, 0 m/s, 0 rad/s, 0.105 m]`；episode 内任一分量发生变化都会令 `fixed_command_pass=false`，并使该 run 不通过。

`success` 要求固定 command 合同成立、episode 正常到达 timeout、全部障碍检查点按顺序通过、整个赛道行驶期间未越出横向走廊、没有 unsafe termination，并且所有被监测量有限。报告仍记录前向进度和 `success_fraction` 对应的参考距离，但单独累计位移不构成成功证据。零策略只用于场景、传感器和命令注入 smoke，不是策略性能基线；障碍穿越 PASS 必须加载训练 checkpoint。正式数值门槛见 `ACCEPTANCE_CRITERIA.md`。
