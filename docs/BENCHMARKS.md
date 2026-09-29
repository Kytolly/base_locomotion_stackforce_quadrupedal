# Plateau 与 Washboard Benchmark

两个 benchmark 复用复杂地形训练任务的 46 维 Actor 输入、62 维 Critic 输入和 12 维动作合同。评估加载冻结 checkpoint，不执行 PPO 更新。

## 任务与场景

| 任务 | 固定场景 | 默认 seed | 赛道长度 |
| --- | --- | ---: | ---: |
| `Base-Locomotion-Stackforce-Quadrupedal-Plateau-v0` | 平地接近、上坡、平台、下坡、释放段 | 8101 | 6.50 m |
| `Base-Locomotion-Stackforce-Quadrupedal-Washboard-v0` | 平地接近、入口坡、14 条横向圆柱脊、出口坡、释放段 | 8201 | 4.68 m |

YAML 默认 `launcher.viz: kit`，因此验收命令会打开 Isaac Sim GUI。正式验收对 nominal 和 randomized geometry 各使用五个不同 seed；每个 run 都必须满足下述 traversal 判定：

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark/plateau.yaml \
  policy.zero_policy=false policy.checkpoint=<checkpoint>
python scripts/run_benchmark.py \
  --config configs/benchmark/washboard.yaml \
  policy.zero_policy=false policy.checkpoint=<checkpoint>
```

无显示 smoke 可追加 `launcher.viz=none launcher.device=cpu benchmark.max_steps=5`。输出 JSON 位于 `output/benchmark/<track>/`，记录固定 command 的期望值与 rollout 实测范围、进度、成功判定、跟踪 RMSE、动作饱和率、轮腿机械能、终止类型、有限值状态、checkpoint 哈希和完整赛道参数。两个赛道固定使用 `[0.28 m/s, 0 m/s, 0 rad/s, 0.105 m]`；episode 内任一分量发生变化都会令 `fixed_command_pass=false`，并使该 run 不通过。

`success` 要求固定 command 合同成立、episode 正常到达 timeout、全部障碍检查点按顺序通过、整个赛道行驶期间未越出横向走廊、没有 unsafe termination，并且所有被监测量有限。报告仍记录前向进度和 `success_fraction` 对应的参考距离，但单独累计位移不构成成功证据。零策略只用于场景、传感器和命令注入 smoke，不是策略性能基线；障碍穿越 PASS 必须加载训练 checkpoint。正式数值门槛见 `ACCEPTANCE_CRITERIA.md`。
