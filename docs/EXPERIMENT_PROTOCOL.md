# 实验、日志、模型与 W&B 约定

本文是 E0-46D 的执行参考；训练设计见 [TRAINING_DESIGN.md](TRAINING_DESIGN.md)，性能门槛见 [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md)。以下命令均在 `base_locomotion_stackforce_quadrupedal/` 项目根目录执行，并使用 `env_isaaclab`，不要使用 Anaconda base 的 Python。

## 目录与结果合同

```text
docs/       当前规范、状态、实验报告；不存原始运行输出
logs/       stdout/stderr、启动失败、探针与训练日志
output/     发布模型、评估 JSON、图表与视频
source/     Isaac Lab 扩展与任务实现
```

`logs/` 和 `output/` 为生成目录，不应提交。RSL-RL 在 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/` 同时保存参数快照和 checkpoint；发布时另行导出到 `output/models/<task>/<run>/`，评估结果写到 `output/evaluation/<task>/<run>/`，记录来源 checkpoint。

每个实验必须记录任务名、代码版本、配置快照、seed、设备、环境数、物理/策略频率、动作与观测合同、checkpoint 路径及固定验证结果。正式 W&B 实验还需记录 run id/URL；无网络可用 offline 模式保留本地 run，恢复后同步。没有 run id 的本地诊断不得标记为正式 W&B 实验。

## 配置与启动

共享合同为 `configs/train/base_locomotion_e0_46d.yaml`：46D Actor、62D Critic、4096 环境和 24 steps/env。可执行实验按 `configs/train/main/` 与 `configs/train/auxiliary/` 分开；主目录仅包含 B0、B1、B2-A 和 B2-B。每个实验 YAML 固定网络、损失与 W&B group，命令行只覆盖 seed、训练阶段和精确恢复路径。

配置检查不启动 Isaac Sim：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --config configs/train/main/b2_bisec_full.yaml --validate-config
```

短测只验证管线，不证明策略性能：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --config configs/train/main/b2_bisec_full.yaml \
  launcher.viz=none env.num_envs=128 agent.max_iterations=1 \
  agent.run_name=e0_46d_smoke wandb.enabled=false
```

Foundation 阶段训练命令：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --config configs/train/main/b2_bisec_full.yaml
```

GUI 可覆盖 `launcher.viz=kit launcher.max_visible_envs=16`；可见环境数量不减少 `env.num_envs`，也不减少物理 rollout。离线记录须同时设置 `wandb.enabled=true wandb.mode=offline`。各独立 seed 使用独立 run name；不要从旧 V0 或不同动作分布的 checkpoint 恢复 E0，恢复时须核对环境与分布合同。

W&B 项目为 `stackforce-quadrupedal-locomotion`，E0 group 为 `e0-46d`。建议命名 `<task>_<stage>_seed<seed>_<label>`，tags 包含任务、阶段和协议。历史吞吐见 [TRAINING_SCALE_REPORT.md](TRAINING_SCALE_REPORT.md)，不是当前 run 的耗时保证。

## 日志开关与当前限制

正式训练开启全部自定义指标组和全部 W&B 面板组：

```yaml
logging:
  metrics:
    enabled: true
    groups:
      core: true
      safety: true
      runtime: true
      command: true
      support: true
      actuation: true
      terrain: true
      motion: true
      reward: true
  wandb_panels:
    enabled: true
    groups:
      core: true
      safety: true
      runtime: true
      command: true
      support: true
      actuation: true
      terrain: true
      motion: true
      reward: true
```

设计合同是：`metrics` 控制逐步采集和 episode 聚合，`wandb_panels` 控制自定义指标写入日志后端（W&B/TensorBoard），不是直接删除 W&B 网页上的现有面板。采集开启、面板关闭可以用于只保留诊断数据；`enabled=false` 设计上关闭对应全部组。RSL-RL 内建 reward、episode length、loss、KL、timing 等标量不受这两个自定义组开关控制。

两个训练入口都把原始 `logging` 配置交给 `TrainingMetricsWrapper` 解析一次，因此 YAML 中的九个指标组和九个面板组会按配置生效。健康度指标及诊断组对应关系见 [TRAINING_HEALTH_METRICS.md](TRAINING_HEALTH_METRICS.md)。

## 注册、评估与导出

安装与注册检查：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python -m pip install -e source/base_locomotion_stackforce_quadrupedal
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/list_envs.py --keyword Base-Locomotion-Stackforce-Quadrupedal
```

冻结 checkpoint 验证默认不上传 W&B；需要时显式开启：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/validate_policy.py \
  --checkpoint logs/rsl_rl/base_locomotion_e0_46d/<run>/model_<N>.pt \
  --wandb --wandb-mode online
```

它不执行 PPO 更新。上传 checkpoint 路径/SHA256、seed、git commit、固定 command suite 的整体/terrain 指标、连续混合路线完成证据与 `performance_pass`。每个 held-out seed 都以确定性等比例布局覆盖全部 11 类地形；PPO 与 BiSEC 使用同一个脚本和协议。完整 suite 使用 `scripts/evaluate_suite.py --checkpoint <checkpoint> --wandb`，为每个 validation seed 建立独立 run。不要并发启动训练与验证的独立 PhysX 进程争用 GPU；共享 Kit 流程须在 checkpoint 写盘后暂停训练再评估。

固定 benchmark 用冻结策略，零动作只作 smoke：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/run_benchmark.py \
  --config configs/benchmark/plateau.yaml launcher.viz=kit \
  policy.zero_policy=false policy.checkpoint=output/models/<task>/<run>/model_<N>.pt
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/run_benchmark.py \
  --config configs/benchmark/washboard.yaml launcher.viz=kit \
  policy.zero_policy=false policy.checkpoint=output/models/<task>/<run>/model_<N>.pt
```

播放与策略导出入口：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/rsl_rl/play_rsl_rl.py \
  --task Base-Locomotion-Stackforce-Quadrupedal-Complex-v0 \
  --checkpoint logs/rsl_rl/base_locomotion_e0_46d/<run>/model_<N>.pt --num_envs 1
```

每次实验报告至少包含目的、代码版本、配置/seed、W&B URL 或 offline id、日志与模型路径、验证指标、失败原因和下一步。历史报告数值保留，不用当前配置重解释旧结果；只引用机器输出，不手工改写指标。
