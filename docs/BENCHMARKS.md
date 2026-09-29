# Plateau 与 Washboard Benchmark

两个 benchmark 复用复杂地形训练任务的 46 维 Actor 输入、62 维 Critic 输入和 12 维动作合同。评估加载冻结 checkpoint，不执行 PPO 更新。

## 任务与场景

| 任务 | 固定场景 | 默认 seed | 赛道长度 |
| --- | --- | ---: | ---: |
| `Base-Locomotion-Stackforce-Quadrupedal-Plateau-v0` | 平地接近、上坡、平台、下坡、释放段 | 8101 | 6.50 m |
| `Base-Locomotion-Stackforce-Quadrupedal-Washboard-v0` | 平地接近、入口坡、14 条横向圆柱脊、出口坡、释放段 | 8201 | 4.68 m |

YAML 默认 `launcher.viz: kit`，因此验收命令会打开 Isaac Sim GUI：

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark/plateau.yaml \
  policy.zero_policy=false policy.checkpoint=<checkpoint>
python scripts/run_benchmark.py \
  --config configs/benchmark/washboard.yaml \
  policy.zero_policy=false policy.checkpoint=<checkpoint>
```

无显示 smoke 可追加 `launcher.viz=none launcher.device=cpu benchmark.max_steps=5`。输出 JSON 位于 `output/benchmark/<track>/`，记录进度、成功判定、跟踪 RMSE、动作饱和率、轮腿机械能、终止类型、有限值状态、checkpoint 哈希和完整赛道参数。

`success` 要求前向进度至少达到 YAML 中 `success_fraction` 乘以赛道长度，并且没有 unsafe termination。零策略只用于创建流程检查，不是性能基线。
