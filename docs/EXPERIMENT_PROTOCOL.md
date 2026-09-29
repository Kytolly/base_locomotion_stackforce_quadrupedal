# 实验、日志、模型与 W&B 约定

## 目录职责

```text
docs/       当前规范、状态、实验报告；不要写原始运行输出
logs/       原始 stdout/stderr、启动失败、探针和训练日志
output/     导出的 checkpoint、TorchScript/ONNX、评估 JSON、图表和视频
source/     Isaac Lab 扩展与任务实现
```

`logs/` 和 `output/` 是生成目录，内容不应提交到版本库。每次运行必须在日志首行或旁边的 JSON 中记录任务名、代码版本、seed、设备、环境数、仿真频率、policy 频率和 W&B run id。

当前 Isaac Lab/RSL-RL 脚本把 runner 的运行目录创建在 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`，其中会同时出现参数快照和 checkpoint。这是框架当前行为；完成 locomotion 任务后，发布模型必须额外导出到 `output/models/<task>/<run>/`，评估结果写入 `output/evaluation/<task>/<run>/`，并在报告中记录来源 checkpoint 的路径。

## W&B 命名合同

复杂任务 YAML 默认使用 W&B 和项目名 `stackforce-quadrupedal-locomotion`，默认 `launcher.viz: kit`。正式训练直接从 YAML 启动，OmegaConf dotlist 可覆盖单个字段：

```bash
python scripts/rsl_rl/train.py \
  --config configs/train/base_locomotion_complex.yaml \
  agent.run_name=seed42 launcher.viz=kit
```

验收或无显示环境可显式覆盖 `launcher.viz=none`；正式 GUI 训练保持 YAML 默认 `kit`，并可通过 `launcher.max_visible_envs` 控制显示环境数量。

建议的 W&B 字段：

```text
project: stackforce-quadrupedal-locomotion
group: <protocol-or-stage>
run_name: <task>_<terrain>_seed<seed>_<short-label>
tags: [base-locomotion, <stage>, <terrain-family>]
```

每个 run 至少同步：

- `task`、`seed`、`git_commit`、`device`、`num_envs`、`sim_dt`、`decimation`；
- 4 维原始/整形 Decision、30 维本体观测和 12 维动作的维度与缩放版本；
- `train/locomotion/*`、`train/safety/*`、`train/terrain/*`、`train/actuation/*`、`train/reward/*`、`train/runtime/*`；
- 每 episode 的地形族、难度、地形 seed、终止类型/原因和有限值状态；
- checkpoint 路径、导出文件路径和固定验证集结果。

无网络或凭据不可用时可使用 `WANDB_MODE=offline`，但必须保留本地 run 目录和 run id，并在网络恢复后执行同步。禁止把没有 W&B id 的训练结果标记为正式实验。

## 分阶段命令

### 1. 模板安装/注册检查

```bash
python -m pip install -e source/base_locomotion_stackforce_quadrupedal
python scripts/list_envs.py --keyword Template-
```

### 2. Locomotion 实现后的 PPO smoke

```bash
python scripts/rsl_rl/train.py \
  --config configs/train/base_locomotion_complex.yaml \
  launcher.viz=kit env.num_envs=8 agent.max_iterations=10 \
  agent.run_name=smoke_seed0 wandb.mode=offline 2>&1 | tee logs/ppo_smoke_seed0.log
```

### 3. 固定 benchmark GUI 验收

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark/plateau.yaml \
  launcher.viz=kit policy.zero_policy=false \
  policy.checkpoint=output/models/<task>/<run>/model_<N>.pt
python scripts/run_benchmark.py \
  --config configs/benchmark/washboard.yaml \
  launcher.viz=kit policy.zero_policy=false \
  policy.checkpoint=output/models/<task>/<run>/model_<N>.pt
```

### 4. checkpoint 播放与导出

```bash
python scripts/rsl_rl/play_rsl_rl.py \
  --task <LOCOMOTION_TASK> \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_<N>.pt \
  --num_envs 1
```

播放前先将待评估 checkpoint 复制或导出到 `output/models/`，并将评估 JSON、视频和导出策略放到同一个 output run 目录。评估不得执行 PPO 更新。

## 结果登记

每次实验完成后，在 `docs/` 增加一份短报告，至少包含：目的、代码提交、配置/seed、W&B URL 或 offline run id、原始日志路径、模型路径、验证指标、失败原因和下一步。报告只引用机器输出，不手工改写指标。
