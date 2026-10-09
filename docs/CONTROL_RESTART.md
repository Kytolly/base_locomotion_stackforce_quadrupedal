# 控制修正与重启

本文记录有界策略的启动合同和 2026-09-30 验证证据；性能门槛以 [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) 为准。

当前 E0-46D 的执行入口见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。本页验证数值保留为历史 checkpoint 证据，不代表当前 E0 run 的策略表现。

## 探索噪声与限幅

旧策略在网络均值上叠加 Gaussian 随机噪声，再将动作硬截到 `[-1, 1]`。例如 1.1、2 和 5 都会变成 1，执行器无法区分这些输出。末期日志的平均标准差约 2.66 是探索尺度，不是 PPO 的 ratio clip；确定性推理不采样该噪声，但旧模型的确定性动作也存在高饱和。

新训练通过 `SquashedGaussianDistribution` 产生有界 12D 动作：latent Normal 的初始标准差 0.3，范围 0.05--0.5，经过 `tanh` 后传入原有 8 个腿位置、4 个轮速解码器。latent mean 使用平滑有界参数化以避免浮点饱和。PPO log-prob 包含变换 Jacobian，KL 使用等价的 latent Normal KL，entropy 使用重参数采样估计。`Policy/mean_std` 仍表示 latent 标准差。

保留 wrapper 的最终数值限幅和执行器位置、速度、力矩安全限制；正常新策略不依赖硬截断。entropy 系数为 0.001。降低探索噪声不保证已有策略自动恢复：必须重新比较固定 command 跟踪、动作饱和和安全指标。

## 重启合同

- 默认从新分布开始训练，`agent.resume=false`；旧 checkpoint 的输出函数不同，不允许直接恢复为新分布继续 PPO。
- 同分布断点重续必须保持分布配置一致；推理从 checkpoint 同目录读取 `agent.yaml`，因此旧模型仍按 Gaussian 解释。
- E0 使用 4096 环境，显示默认 `launcher.viz=none`；需要 Kit 时显式覆盖。当前工作区 YAML 启用 W&B online；正式命令显式指定日志模式。命令形状、动作顺序、物理限幅遵循当前合同。
- curriculum 使用整个 episode 的命令路程、沿命令方向的实际路程、平面速度误差、yaw 误差及终止状态；停止不升级，正确往返运动不因净位移为零而被降级。
- 金字塔坡地有 50% reset 从朝向中心的坡面接近位置开始，使用地形射线确定落点高度，避免训练只从顶部平台出发。固定 validation 禁用这项随机起点。

在项目根目录、已安装项目包的 Isaac Lab 环境中先运行短试：

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --config configs/train/main/b2_bisec_full.yaml \
  launcher.viz=none env.num_envs=128 agent.max_iterations=1 \
  agent.run_name=e0_46d_control_smoke wandb.enabled=false
```

1 iteration 只检查管线。延长训练仍需分命令跟踪不退化、饱和下降、安全指标改善；不要原样加载 `model_13749.pt` 的旧噪声与优化器状态。

## 泛化评测

```bash
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/evaluate_suite.py --checkpoint <checkpoint> --dry-run
/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python scripts/evaluate_suite.py --checkpoint <checkpoint>
```

参数来自 `configs/evaluation/generalization.yaml`，可用 `launcher.viz=none` 覆盖。每次新建输出目录，包含 5 个留出 seed 的五命令/八地形/八难度评测，以及两种跑道各 5 个 nominal 和 5 个 randomized run，共 25 次。日志写入 `logs/evaluation/`，汇总写入 `output/generalization/`。脚本拒绝将训练 terrain seed 放进同分布 validation seed 列表。

同生成器换 seed 是同分布新实例测试。Plateau 是新组合及部分坡度外推，Washboard 是未训练过的圆柱接触几何。nominal 的多 seed 重复用于复现，不代表五种不同地形。测试集参数不能作为训练样本直接回灌；若按测试结果调参，最终报告还需独立冻结的测试集合。

## 验证证据

- 单元测试覆盖 transformed density/gradient、噪声上界、旧 checkpoint 配置、支撑几何、curriculum 和最差地形门控。
- 64 环境两轮 PPO smoke 已运行，使用 W&B offline，不代表长期学习已验证；日志：`logs/bounded_control_slope_smoke_20260930.log`。
- 强制 timeout 的 benchmark 保持 `fixed_command_pass=true`；新 checkpoint 已在 Kit GUI 中完成短测，动作饱和为 0，仅用于加载与管线验证。
- 主动倒置碰撞探针测得机身最大接触力约 43.8 N，并触发碰撞终止，传感器刚体路径核验通过。PhysX 仍对同名子 Mesh 打告警；此告警不等同于刚体验证失败。日志：`logs/base_contact_probe_20260930.log`。
- 旧 `model_13749.pt` 在修正法向和评测采样后的 Plateau 上越过 4/4 纵向检查点，前向 RMSE 约 0.124 m/s，动作饱和约 81.7%，但出现横向走廊越界，`traversal_complete=false`。可以确认不再在入口停滞，不能认定完整跑道成功穿越。报告：`output/benchmark/plateau/v2_model_13749_final_20260930.json`。
- 同一旧模型在 Washboard v2 上通过入口坡、进入瓶阵，到约 2.63 m 因 `base_collision` 终止，1/4 检查点，动作饱和约 80.5%。目前不足是瓶阵持续支撑及输出质量，不再是“尚未爬上入口坡”。报告：`output/benchmark/washboard/v2_model_13749_final_20260930.json`。

上述策略结果是单 seed 诊断，不是五 seed 泛化 PASS。长训练、完整 25-run 矩阵及新策略最终性能验收尚未执行。
