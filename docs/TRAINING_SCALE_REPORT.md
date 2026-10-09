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
