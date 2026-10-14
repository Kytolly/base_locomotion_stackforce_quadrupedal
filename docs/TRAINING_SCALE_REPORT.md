# 训练并行规模实测

> **历史测量边界：** 本文保留指定硬件、资产与当时配置下的规模探针结果。它用于比较该次 4096/6144/8192 测量，不能保证当前 E0 配置的 iteration 时间；近期约 5 s/iteration 的 run 应按自身配置和日志单独分析。

## 结论

RTX 5080 16 GB 的正式训练默认使用 4096 个环境。8192 环境需要显著放大 PhysX GPU broadphase buffer，显存余量过小且总吞吐低于 4096，不适合作为当前资产和接触配置的默认规模。

## 测试条件

- GPU：NVIDIA GeForce RTX 5080，16303 MiB；
- rollout：每环境 24 步，1 次 PPO iteration；
- 仿真：200 Hz，policy 50 Hz，8 类地形，5 个 contact sensor；
- 渲染：`launcher.viz=none`；
- W&B：offline；
- policy/critic：`[128, 128]`；
- seed：42。

## 结果

| 环境数 | PhysX broadphase 配置 | steps/s | iteration | GPU 峰值 | 判定 |
| ---: | --- | ---: | ---: | ---: | --- |
| 4096 | aggregate found/lost `2**26` | 4979 | 19.74 s | 未单独采样 | 默认 |
| 6144 | found/lost `2**22`；aggregate found/lost `2**27`；total aggregate `2**22` | 4065 | 36.27 s | 9940 MiB | 可运行但不优于 4096 |
| 8192 | found/lost `2**22`；aggregate found/lost `2**28`；total aggregate `2**23` | 2941 | 66.84 s | 15113 MiB | 仅压力测试 |

4096 的运行完成 98304 steps，`train/runtime/all_finite=1`，无 PhysX buffer error。8192 的最终运行完成 196608 steps，容量错误已消失，但只剩约 1.3 GiB 显存余量，且 iteration 时间是 4096 的 3.39 倍。

6144 的运行也完成 147456 steps，`train/runtime/all_finite=1` 且无 PhysX buffer error；其吞吐约为 4096 的 82%，所以不作为默认规模。

## 证据

- `logs/ppo_smoke_4096_physx.log`；
- `logs/ppo_smoke_6144_physx.log`；
- `logs/ppo_smoke_6144_physx_gpu.csv`；
- `logs/ppo_smoke_8192_physx_268m.log`；
- `logs/ppo_smoke_8192_physx_268m_gpu.csv`。

`base_link/base_link` 是同名 Mesh 叶节点。PhysX tensor view 在解析父刚体名称时仍会为该 Mesh 输出 lookup warning；该提示不改变动作维度、刚体绑定或上述 buffer 判定，但会放大启动日志，后续应作为传感器后端兼容性问题单独处理。

## E0-v2 吞吐优化（G5 完成）

> 2026-10-14：计时矩阵已完成。没有开关同时满足“至少提速 5%”和物理资格，因此 G5 的正确结论是保留当前配置；PASS 表示取舍完成，不表示获得了净提速。

方法：4096 env、seed 42、`b2_bisec_full`、foundation profile，每项只改一处，跑 200 iteration，取后 100 次 `Perf/collection time` 的均值（`python scripts/fix_plan_runs.py g5`）。保留条件：耗时降低 ≥ 5%，物理检查全部满足，reward 曲线无异常。结果写入 `output/fix_plan/G5/timing_matrix.json`，再由 `scripts/fix_plan_gate.py g5` 记录。

| 编号 | 开关（`configs/train/base_locomotion_e0_46d.yaml`） | collection time | 增益 | 物理检查 | 保留 |
| --- | --- | ---: | ---: | --- | --- |
| 基线 | E0-v2 当前配置 | 7.640 s | — | — | 保留 |
| T2 | `env.throughput.enabled_self_collisions: false` | 7.432 s | +2.7% | 未达到 5% 门槛，物理复核不再展开 | 否 |
| T3a | `solver_position_iteration_count: 4` | 5.326 s | +30.3% | closure 与高度 RMSE 均失败 | 否 |
| T3b | `solver_position_iteration_count: 6` | 5.896 s | +22.8% | closure 与高度 RMSE 均失败 | 否 |
| T4 | `env.throughput.foot_track_pose: false` | 7.655 s | -0.2% | 无提速 | 否 |
| T5 | 训练只开 core/safety/runtime/actuation/terrain，关闭面板 | 7.296 s | +4.5% | 未达到 5% 门槛 | 否 |
| T6 | `env.throughput.foot_contact_update_period: 0.02` | 6.663 s | +12.8% | 单项接触探针通过；合并复测 7.943 s，比相邻基线慢 4.0% | 拒绝 |
| T7 | `env.throughput.replicate_physics: true` | — | — | 与逐环境 contact-report 启动修改不兼容 | 拒绝 |

T1（课程等级跨阶段继承）不影响单 iteration 耗时，只减少低 level 的无效训练，已在编排器中实现。最终 `kept_overrides=[]`，没有把任何吞吐开关写成新默认值。不做的修改：增加 `num_envs`（见上文实测）、减少 rollout 步数或 PPO epoch（学习时间不到 3%）。完整证据见 `output/fix_plan/G5/timing_matrix.json` 和 `G5/gate.json`。
