# 动作范围修复与训练吞吐优化：执行计划

> **版本：** v3.7（2026-10-14）。
> - v1：只有分析和方向。
> - v2：增加按门控逐步执行到"可验收长训"的完整计划。
> - v3：舵机确认为 DS041MG；动作范围改为按舵机真实指令行程推导（§3A），力矩门 G2 据此更新。
> - v3.1：确认为 180° 版本；G4 写明具体代码位置。
> - v3.2：用户确认"固件零位 = 架空时腿伸直"（附照片），无需实测。§3A 给出逐关节范围；**发现仿真默认站姿在实机上不可达**，G4 增加默认站姿与高度指令的修改。
> - v3.3（2026-10-13）：按本计划完成代码、配置与门控工具；§7 记录每个门控的实现、命令和与原文的差异。
> - v3.4（2026-10-13）：用户确认 wiki 当前有效配置为 300°；重算 §3A 名义范围，原厂参考指令与训练候选站姿分离。180° 推导撤销。
> - v3.5（2026-10-13）：被动外膝占位限位由 ±1.57 rad 放宽到 ±π；G3 以 0.05 rad 网格建立逐腿连通可行域，并在动作端投影不可行组合。
> - v3.6（2026-10-14）：按本轮“快速定位动作范围问题”的范围，将 G6 明确拆成动作合同资格和策略质量观察；约 1k iteration 的 P0/P1 不再承担策略收敛验收，性能风险转入 G8/G9。
> - v3.7（2026-10-14）：G7 改为长训前机械净空资格；手写控制器的整套越阶结果保留为观察，不再用控制脚本质量代替机构能力判断。
> - v3.9（2026-10-15）：正式配置加入观测归一化、轮噪声 0.1、动作平滑与速度惩罚加强、前进核 σ 0.15。根因是观测未归一化导致策略不看指令；对照与证据见 [奖励与课程修复方案](REWARD_CURRICULUM_FIX_PLAN.md)。限速仍关闭。阶段门控阈值（Foundation `flat_timeout_rate >= 0.90`、unsafe 中止条件）尚未校准，长训可能会暂停或中止。
> - v3.8（2026-10-14）：T3 反向调整，位置迭代 8 → 32（用户确认）。8 次时闭环只在位置层收敛，PhysX 报告的关节速度含幽灵分量，伺服额外承担约 2.6 倍静力学负载；32 次与五连杆静力学一致。代价：4096 env 每 PPO iteration 采样 2.92 → 3.97 s（+36%；16 次为 3.30 s，+13%）。另：轮碰撞体统一为 32 边棱柱（已安装）；腿探索噪声改为站姿两侧物理 std ≤ 0.10 rad（原按较小半区间折算，较大一侧达 0.22–0.42 rad）；腿目标限速 `env.leg_target_max_step_rad` 已实现但正式配置保持关闭（用户决定：仅当回放仍显示无效高频指令、且不损害支撑/落脚/扰动恢复时再启用，并同步到真机）。见 [外髋静态内力诊断](OUTER_HIP_STATIC_FORCE_DIAGNOSIS.md)。
> **状态（2026-10-14）：** G0–G7 已记录 PASS；正式合同为 `g3_measured`。P1 动作饱和已降至验收线内且全程有限，短策略性能未验收；G7 已测得四腿自由基座支持下 72 mm 以上净空，但整套手写越阶仍失败并作为长训风险保留。正式长训尚未开始。
> **证据边界：** 文中数值来自下列 run、配置和资料；任何修改之后都必须按门控重新测量，不能沿用旧结论。

---

## 0. 目标与交付物

**目标：** 在 0.9 倍原有关节范围内，训练出一个新的 E0 长训 checkpoint，同时降低每次 iteration 的耗时。具体要求：

- 地形 level 不再卡在约 4；
- 动作饱和率接近验收线（≤ 5%）；
- 能完成 30 mm 台阶跨越；
- 输出完整的验收证据包。

**最终交付物（G9）：**

1. 冻结 checkpoint（`model_19999.pt`）及其 SHA-256、resolved config、W&B run 链接；
2. 固定 validation JSON（11 command × 11 地形，8 个 level，env 数为 64 的倍数）；
3. 泛化 suite（`evaluate_suite.py`）JSON；
4. Plateau/Washboard benchmark：nominal 与 randomized 各 5 个 seed；
5. 台阶探针 JSON：30 mm 与 40 mm 单台阶，以及 3×40 mm 台阶；
6. 各阶段视频（`record_e0_stage_videos.py`）；
7. 长训报告 `docs/E0_V2_LONG_RUN_REPORT.md`：按 `ACCEPTANCE_CRITERIA.md` 逐项写明 PASS/FAIL，并与 b0/b2 旧基线对比。

"可验收"指证据包完整、每个门控都有明确结论，交给你判断。它不等于预先承诺全部 PASS。

---

## 1. 现状依据（摘要）

| 项目 | 事实 | 来源 |
| --- | --- | --- |
| 腿动作 | 8 个 hip 位置动作，`scale=0.5`，`clip=±0.5 rad`，默认关节角 0；输出为 tanh 压缩的 Gaussian，raw ∈ (−1, 1) | `env.yaml`；`training/distribution.py` |
| URDF 限位 | Outer Hip `±1.57 rad`；Inner Hip `±π`（loop-cut 资产的占位值） | 资产 URDF |
| 执行器 | Hip `Kp=20`，`effort_limit_sim=0.392 N·m`，`velocity_limit_sim=8.73 rad/s`；误差超过约 0.02 rad 力矩即饱和 | `env.yaml`；资产 README |
| 舵机型号 | 8 个髋关节均为 GDW **DS041MG**（数字金属齿轮舵机，用户确认）。7.4 V 下：堵转力矩 4.0 kg·cm ≈ 0.392 N·m（servodatabase）或 5.0 kg·cm ≈ 0.49 N·m（零售页）；额定力矩 1.3 kg·cm ≈ 0.127 N·m；空载速度 0.12–0.13 s/60° ≈ 8.1–8.7 rad/s。标准版行程 180°，另有 270° 和 290° 版本 | [servodatabase](https://servodatabase.com/servo/gdw/ds041mg)、[零售页](https://aifitlab.com/products/gdw-ds041mg-servo-motor) |
| 舵机接口 | PCA9685 PWM，500–2500 μs 对应 0–300°；开环，无编码器；机械零位（U02）和机械限位（U04）尚未确定 | wiki `E-FW-001`、`M1-T01`、`M1-T02` |
| 原厂工作空间 | 五连杆 L1=L4=60 mm，L2=L3=100 mm，L5=40 mm；机身高度 70–130 mm；trot 抬腿 60 mm | wiki `E-FW-003` |
| 抬腿能力 | 悬空时 `±0.5 rad` 约 19 mm，`±1.2 rad` 约 38.5 mm；课程台阶 0.04–0.14 m | `STAIR_PROBE_RESULTS_2026-10-02.md` |
| 训练表现 | b0/b2 的 level 卡在 3.7–4.0；饱和率 27–38%；`leg_action_rms` 0.82（轮 0.63）；stairs 不安全终止 36–41% | TF events |
| 耗时 | 采样 7.0–7.8 s/iteration，学习 0.05–0.17 s；四阶段 20k iteration 约 41 h | TF events |
| 课程回落 | 每个阶段续训时 level 回到约 1（`max_init_terrain_level=1`），浪费约 1000 iteration/阶段 | TF events；`complex_env_cfg.py` |
| 规模 | 6144 和 8192 env 的吞吐都低于 4096 | `TRAINING_SCALE_REPORT.md` |

---

## 2. 不变量（全程遵守）

1. **范围规则：** 动作限位 = 舵机真实指令行程 × 0.9，以行程中心为基准收缩（例如 ±180° → ±162°），再换算到仿真关节坐标（推导见 §3A）。内外侧髋关节用同型号舵机，规则相同；但两者在仿真零位对应的舵机角不同，所以换算到仿真坐标后的区间不同，而且不对称。现有 URDF 中 Outer `±1.57`、Inner `±π` 的限位不再作为依据，改为 §3A 的结果。五连杆闭环和碰撞检查只用来剔除物理上不可达的姿态，不重新定义范围。
2. **一次只改一类东西：** 范围、执行器、吞吐三类修改分开验证，最后再合并。
3. **合同升级：** 改变动作语义会破坏 checkpoint 合同，`protocol_version` 升为 `E0-v2-46d`。46D 观测维度不变，但 previous action 的物理含义变了。旧的 b0/b2 checkpoint **不得**用于续训，只作为对照。
4. **不改的内容：** 观测、奖励权重、网络结构、PPO 超参数（例外：G4 中腿部探索噪声的物理等效调整）。这样效果可以归因到范围和执行器。
5. **证据目录：** 每个门控的产出放在 `output/fix_plan/G<n>/`，JSON 加 README，并记录 commit hash。
6. **失败处置：** 门控失败时先停下，写失败报告给你，不自行跳过。

---

## 3A. 按当前 300° 配置推导名义关节范围（v3.4）

### 已确认的前提

| 编号 | 前提 | 来源 |
| --- | --- | --- |
| A1 | DS041MG 当前配置为 300° 版本：500–2500 μs 名义对应 0–300°，即 0.15°/μs；物理端点未实测 | wiki `E-HW-007`；用户再次确认 |
| A2 | 固件零位（指令 180° + `servo_off`）= 架空时腿伸直，内外曲柄都竖直向下（$\alpha=\beta=90^\circ$，足端在髋轴下方 158 mm） | 用户确认（照片：RR 腿 ch7/ch8） |
| A3 | 固件与舵机配置均按 0–300° 换算，名义 1° 指令 = 1° 舵机角；零位脉宽 $p_{0,i}=1700+6.667\cdot\text{off}_i$ μs | `E-FW-001` 与 A1 |
| A4 | 安装方向：FR/RL 为 $\theta=270^\circ-\text{角}$，FL/RR 为 $\theta=90^\circ+\text{角}$；仿真零位为 $\alpha_0=154^\circ$、$\beta_0=6.5^\circ$（足端在髋轴下方 79 mm），由 URDF 推出 | `E-FW-003`；URDF |
| A5 | 机械限位：外侧髋两个方向几乎不受限；内侧髋朝一侧受限 | 用户确认 |

### 推导

1. 从零位出发，名义转角为 $r=0.15\,(p-p_0)$。0.9 行程 $p\in[600,2400]$ μs，即固件命令 15–285°；未计 trim 时 $r\in[-165^\circ,+105^\circ]$。零位不在行程中心，两侧余量不对称。A2 是训练坐标约定，不是外部机械零位实测。
2. 曲柄角：FR/RL 为 $90^\circ-r$，FL/RR 为 $90^\circ+r$。左右镜像安装，所以同一侧余量在左右两边落到相反的曲柄方向。
3. 换算到仿真关节：$q=s_i(\text{曲柄角}-\text{仿真零位曲柄角})$。$s_i$ 见下表；抬腿方向为 $\alpha\uparrow$、$\beta\downarrow$。

| 关节 | $+q$ 的效果 | $s_i$ |
| --- | --- | --- |
| FR/FL 外侧 | $\alpha$ 增大 | +1 |
| RL/RR 外侧 | $\alpha$ 减小 | −1 |
| FR/RL 内侧 | $\beta$ 减小 | −1 |
| FL/RR 内侧 | $\beta$ 增大 | +1 |

### 当前范围的含义

在 300° 名义范围下，仿真零位落在全部八个区间内；180° 推导中的“默认站姿不可达”结论不适用。该深蹲姿态的抬腿空间仍需检验。名义范围不是装机安全范围，G3 必须剔除机械干涉、闭环失败与无法跟踪的组合。

### 建议的新默认站姿

保留几何选定的训练候选站姿：$\alpha=128.4^\circ$、$\beta=66.6^\circ$，足端在髋轴下方约 137 mm，待 G3 验证稳定性。它不是原厂 reference command 的换算结果：`E-M1-002` 的开机指令在 300° 名义尺度下对应约 $154^\circ/51^\circ$、足端深度 110 mm。逐关节结果（rad）：

| 通道 | 关节 | 新默认 $q$ | 0.9 行程绝对区间 | 相对新默认的动作区间 | 抬腿方向余量 |
| --- | --- | ---: | --- | --- | ---: |
| ch1 | FR_Outer_Hip | −0.447 | [−3.019, +1.693] | [−2.572, +2.140] | 2.140 |
| ch2 | FR_Inner_Hip | −1.049 | [−4.250, +0.463] | [−3.201, +1.512] | 1.512 |
| ch3 | FL_Inner_Hip | +1.049 | [−1.510, +3.203] | [−2.559, +2.154] | 2.559 |
| ch4 | FL_Outer_Hip | −0.447 | [−3.875, +0.838] | [−3.428, +1.285] | 1.285 |
| ch5 | RL_Outer_Hip | +0.447 | [−1.710, +3.002] | [−2.157, +2.555] | 2.157 |
| ch6 | RL_Inner_Hip | −1.049 | [−4.250, +0.463] | [−3.201, +1.512] | 1.512 |
| ch7 | RR_Inner_Hip | +1.049 | [−1.283, +3.429] | [−2.332, +2.380] | 2.332 |
| ch8 | RR_Outer_Hip | +0.447 | [−0.576, +4.136] | [−1.023, +3.689] | 1.023 |

300° 范围下的抬腿上限尚未扫描，不沿用 180° 合同的 48–68 mm 估算；最终以 G3 的四腿最小实测抬腿为准。

### 影响与注意

- 绝对区间中超出机构可达的部分（例如曲柄转入机身），由 G3 的闭环和碰撞扫描剔除（对应 A5）。
- 新站姿比现在高约 58 mm，必须同步修改：初始机身高度、Decision 中的高度指令范围、高度相关的奖励和终止阈值。数值在 G3 中实测后确定，见 G4。
- 名义 PWM 逆映射为 $p=p_{0,i}+r/0.15$，与固件 300° 尺度一致；部署仍须单独确认物理零位、方向和安全端点。
- 本节完全依赖 A1–A4。G3.0 会在仿真中复核 A2 与 A4；任何一项不符都要回到本节重算。

## 3. 门控总览

```text
G0 基线与计时剖析 ─┐
G1 诊断指标 ───────┼─> G3 闭环可行域与 0.9 范围 ─> G4 动作合同修改 ─┐
G2 舵机规格与零位约定（已定，只需登记）──────────────────┤
G1 ─> G5 吞吐优化（与 G3/G4 并行）───────────────────────────────────┤
                                                                     v
                         G6 短探针对比 ─> G7 机构能力门 ─> G8 正式长训 ─> G9 交付验收
```

| 门控 | 预计耗时 | 是否需要你介入 |
| --- | --- | --- |
| G0 | 0.5 天 | 否 |
| G1 | 0.5–1 天 | 否 |
| G2 | 0.5 天（登记） | 否 |
| G3 | 1 天 | 否 |
| G4 | 1 天 | 确认合同升级 |
| G5 | 1–2 天（每项 200 iteration 计时） | 否 |
| G6 | 约 1 天（3 组 × 1500 iteration × 约 6–7 s） | 否 |
| G7 | 0.5 天 | 否 |
| G8 | 约 30–41 h 墙钟时间 | 阶段结束时可抽查 |
| G9 | 1 天 | **是**：最终验收 |

---

## 4. 逐门控执行说明

### G0：基线冻结与计时剖析

**目的：** 固定对照数据，弄清 7.4 s 的采样时间花在哪里。

**操作：**

1. 从 b0/b2 consolidation 的 TF events 导出对照表 `output/fix_plan/G0/baseline_metrics.json`，内容为最后 300 iteration 的均值：
   - terrain level；
   - 总饱和率；
   - `leg_action_rms`、`wheel_action_rms`；
   - 各地形族的 unsafe 与 timeout；
   - `Perf/*`。
2. 写临时剖析脚本 `scripts/profile_env_step.py`（不进入训练路径）。用 b2_bisec_full 配置，4096 env，零动作或已有 checkpoint 推理，跑 100 个控制步。在 `env.step` 内按段计时，每段前后执行 `torch.cuda.synchronize()`：
   - `sim.step` 与 `scene.update`（×4 physics substep）；
   - action、observation、reward、termination、curriculum、command 各 manager；
   - metric/recorder（`LocomotionStepRecorder` 与 training metrics）。
3. 输出 `output/fix_plan/G0/step_profile.json`，记录每段的毫秒数和占比。

**通过条件：** 各段之和与实测单步时间相差不超过 10%；基线表齐全。

**失败处置：** 剖析之和偏差过大时，改用 PyTorch profiler 重测；不影响 G1–G3 开始。

---

### G1：诊断指标

**目的：** 证明饱和主要来自腿，并且区分"行程不够"和"力矩不够"。

**操作：** 在 `evaluation/metric/`（training metric 的 actuation 与 runtime 组）中新增以下指标，并接入 W&B actuation 面板：

| 新键 | 定义 |
| --- | --- |
| `actuation/leg_saturation_rate` | 8 个腿维度中 \|raw\| > 0.95 的比例 |
| `actuation/wheel_saturation_rate` | 4 个轮维度中 \|raw\| > 0.95 的比例 |
| `actuation/leg_torque_saturation_rate` | 8 个 hip 中 \|applied τ\| ≥ 0.95·τmax 的比例 |
| `actuation/leg_tracking_error_rad` | hip 目标角与实际角之差的 RMS |
| `runtime/closure_residual_max_mm` | 4 个 closure joint 锚点距离的最大值；每 N 步采样一次以控制开销 |

另外补充单元测试 `tests/test_actuation_metrics.py`：用构造张量验证上面五个定义。

**验证：**

1. CPU 单 env 的 metrics smoke 通过，所有值有限；
2. 对 b0 `model_19999.pt` 跑一次 `validate_policy.py`（64 env，union_consolidation），导出新指标。

**通过条件：**

- 新键在 W&B 中出现且有限；
- 旧 checkpoint 上 `leg_saturation_rate` 明显高于 `wheel_saturation_rate`；
- 记录 `leg_torque_saturation_rate` 和 `closure_residual_max_mm` 的基线值。

**失败处置：** 如果腿饱和率不高于轮，说明诊断前提不成立，停下并报告，重新分析（可能是轮速度范围的问题）。

---

### G2：舵机规格与行程（型号已确认为 DS041MG）

**G2-A 力矩与速度（基本可以落定）：**

- 仿真的 `effort_limit_sim=0.392 N·m` 正好等于 DS041MG 在 7.4 V 下的**堵转力矩**（4.0 kg·cm，servodatabase）；`velocity_limit_sim=8.73 rad/s` 对应 0.12 s/60°，也与规格一致。所以仿真并没有低估力矩，力矩上限是真实的物理限制，**不能调高**。v2 中"有规格就跑 P2 组"的设想取消。
- 额定（持续）力矩只有约 0.127 N·m。静载估算中每个曲柄需要 0.1–0.26 N·m，所以三足支撑和抬腿时长时间接近或超过额定值，实机上会发热、掉力。
- **决策：**
  - `effort_limit_sim` 保持 0.392（以两个来源中较低的堵转值作为峰值），`velocity_limit_sim` 保持 8.73；
  - 新增 G1 指标 `actuation/leg_torque_over_rated_rate`（\|τ\| > 0.127 N·m 的时间比例）和 1 s 窗口内的 \|τ\| 均值；
  - 第一轮只记录这两个指标，不加惩罚。长训报告中把它们作为 sim2real 风险项列出。
- `stiffness=20` 和 `damping=0.5` 在 M3-G03 舵机标定之前保持不变，记为已知的 sim2real 债务。

**G2-B：** 在 wiki 登记 300° 配置、固件零位约定与 G3 安全过滤前提。G3.0 只复核仿真映射，不代替物理机械零位测量。可选实机参考指令照片不阻塞仿真，但不能把名义端点标为实测端点。

**产出：** 在 wiki 新增证据 `E-CAL-002-DS041MG规格与零位约定`（含规格来源、A1–A5 与照片）；仓库中引用该证据 ID。

**通过条件：** G2-A 已记录；A1–A5 已登记。

---

### G3：零位映射复核、闭环过滤、新站姿与抬腿实测

**目的：** 验证 §3A 的映射，剔除不可达姿态，得到最终的逐关节范围，并测出可实现的抬腿高度。

**操作：**

1. **G3.0 映射复核。** 写 `scripts/verify_servo_mapping.py`，在悬空 fixed-base 中依次执行三个姿态：
   - 腿伸直（$\alpha=\beta=90^\circ$）：足端应在髋轴下方 158 mm；
   - 新默认站姿（$128.4^\circ/66.6^\circ$）：足端应在髋轴下方约 137 mm；
   - 每个关节沿 $+q$ 走 0.1 rad：曲柄转向应与 §3A 的 $s_i$ 表一致。

   前两项要求 closure 残差 ≤ 1 mm、足端位置误差 ≤ 3 mm。不通过就停下，修正 §3A，**不进入后续步骤**。
2. **闭环与碰撞过滤。** 新增 `scripts/probe_action_range.py`，复用 `probe_stair_climb.py --fixed-base` 夹具。对每条腿的 (outer, inner) 在 §3A 区间内做二维网格（步长 0.05 rad），每个点 settle 1 s，记录：
   - closure 残差；
   - 自碰撞接触力；
   - 足端相对机身的位移；
   - 跟踪误差；
   - 是否出现 NaN。

   满足"残差 ≤ 1 mm、无自碰撞、跟踪误差 ≤ 0.05 rad、数值有限"的点为可行。
3. **最终范围。** 最终范围 = §3A 的 0.9 行程区间 ∩ 可行域：
   - 0.9 余量已经在行程上取过，**不再二次收缩**；
   - 优先取可行域内的逐关节 box（允许不对称）；
   - 如果 box 明显小于 §3A 区间的有用部分（尤其是抬腿方向），改用"box 裁剪 + 可行多边形投影"，投影表存为 `feasible_polygon.json`。
4. **新站姿实测。** 自由基座、平地，四条腿置于新默认站姿（§3A），以零动作站立 5 s，记录：
   - 机身高度（相对支撑面）；
   - 四轮接触；
   - 8 个 hip 的稳态力矩，以及它们相对 0.392 N·m 堵转值和 0.127 N·m 额定值的比例。

   通过条件：能稳定站立，hip 稳态力矩 < 0.392 N·m（最好低于额定值）。如果站立需要的力矩过大，把站姿沿伸直方向调整（足端降低 10 mm 一档），重新测量。
5. **抬腿实测。** 从新站姿出发，在最终范围内做悬空抬腿扫描，四条腿分别报告（RR 受 ch8 限制，预计最小）。
6. **原厂步态对照（参考，不作门控）。** 用 `record_firmware_reference_gait.py` 回放原厂 trot，对比抬腿幅度。

**产出：** `output/fix_plan/G3/{mapping_check.json, range_scan.json, final_ranges.json, feasible_polygon.json, stance.json, lift.json, README.md}`，并附可行域图。

**通过条件：**

- G3.0 全部通过；
- 最终范围内残差 ≤ 1 mm、无自碰撞、数值有限；
- 新站姿能稳定站立，hip 稳态力矩低于堵转值；
- 四条腿中最小的抬腿 ≥ 30 mm（300° 合同尚无实测余量结论）。

**失败处置：**

- 最小抬腿在 25–30 mm：可以继续，但课程台阶上限 = 0.8 × 实测值，并在报告中注明"台阶能力受舵机行程限制"；
- 最小抬腿 < 25 mm：停下报告。G2-A 已经确认力矩无法提高，所以剩余方向只有两个，需要你决定：
  - 放宽 0.9 规则（例如 0.95）；
  - 接受越障上限，把课程限制在可达高度内。

---

### G4：动作合同修改（`E0-v2-46d`）

**目的：** 把 G3 的范围和新站姿落实到配置中，并通过动作语义门。改变站姿和高度指令也属于 `E0-v2-46d` 合同的一部分。

**修改清单：**

| 文件 | 修改内容 |
| --- | --- |
| `mdp/action/policy.py` | 现有 `JointPositionActionCfg` 只支持对称的 `scale` 和 `clip`（`LEG_POSITION_SCALE=0.5`，`LEG_POSITION_TARGET_CLIP=±0.5`）。新增自定义 action term（如 `AsymmetricJointPositionAction`）：raw 0 → 默认姿态，raw +1 → 上限，raw −1 → 下限，逐关节分段线性映射；如果 G3 选了投影方案，在 `process_actions` 中加入多边形投影。轮子的动作项不变 |
| `env/robots/stackforce.py`、`mdp/events.py` 与资产 | 训练用的关节硬限位：在 `mdp/events.py` 增加一个 `startup` 事件，调用 `robot.write_joint_position_limit_to_sim`，把 8 个 hip 写成 G3 的最终范围（不需要重建 USD）。资产一致性：修正 URDF 中 Outer `±1.57`、Inner `±π` 的限位，用 `assets/.../scripts/build_asset.py` 和 `recover_closed_loops.py` 重建 closed USD，作为后续维护项，不阻塞训练。`soft_joint_pos_limit_factor` 保持 0.95；`_actuator` 的 0.392/8.73/20 保持不变（G2-A） |
| `env/robots/stackforce.py`（`init_state`） | **默认站姿改为 §3A 的新站姿**：`joint_pos` 改为逐关节值（例如 FR_Outer −0.447、FR_Inner −1.049 等，以 G3 复核后的值为准），被动膝关节给出闭环一致的初值；`init_state.pos` 的 z 改为 G3 实测站立高度 + 0.01 m。`use_default_offset=True` 不变，于是 raw 0 对应新站姿 |
| `mdp/policy/commands.py`、`configs/train/base_locomotion_e0_46d.yaml` | Decision 中的高度指令：名义值从 0.105 m 改为 G3 实测站立高度，范围按名义值 ±0.03 m 设定（需在最终范围内可达）；shaped height 同步修改 |
| `complex_env_cfg.py`（奖励与终止） | `low_body_height_margin` 的 `warning_height`（0.08）和 `base_height` 终止的 `minimum_height`（0.06）按新名义高度同比上移；`track_body_height` 的 std 不变 |
| `mdp/events.py`（`reset_root_to_default`） | 复位时根高度跟随新的 `init_state.pos` |
| `training/distribution.py` | 支持按维度初始化 `log_std`（腿与轮分别设）。腿的 `init_std` 按物理噪声保持不变：当前约 0.3×0.5 = 0.15 rad，新的半区间约 0.35–2.5 rad，按各关节较小一侧的半区间折算腿 `init_std`，使初始物理噪声约为 0.15 rad，轮保持 0.3；`min_std` 0.05 与 `max_std` 0.5 不变 |
| `agents/rsl_rl_ppo_cfg.py`、`configs/train/base_locomotion_e0_46d.yaml` | 新增 per-group std 配置项 |
| `env/terrains/complex.py`（第 91、97、105 行：`grid_height_range=(0.03,0.12)`、`step_height_range=(0.04,0.14)`；`continuous.py` 中的台阶段同样处理） | stairs 和 boxes 的起点改为 0.02 m，上限 = min(原上限, 0.8 × G3 实测抬腿)；level 插值方式不变 |
| `complex_env_cfg.py` | `protocol_version = "E0-v2-46d"` |
| `training/checkpoint.py` | 合同 payload 自动包含新的动作参数（确认旧 checkpoint 在新合同下 resume 会被拒绝） |
| `docs/POLICY_ACTION_REWARD.md`、`TRAINING_DESIGN.md` | 更新动作合同表与范围来源 |
| `tests/` | 新增测试：默认站姿下 closure 残差 ≤ 1 mm；raw = 0 时目标等于默认姿态；raw = ±1 时目标等于范围端点；投影后的点在多边形内；旧合同 checkpoint 被拒绝 |

**验证：**

1. 动作语义门：重跑 `scripts/probe_robot_action.py`，`--action-magnitude` 取 0.2、0.5、1.0，输出 `output/action-probes/e0v2_*.json`。逐通道核对目标值、符号、仅驱动目标关节、被动闭环响应、数值有限。
2. `pytest` 全部通过。
3. 单 env CPU smoke 和 64 env 零策略 validation smoke 通过。

**通过条件：** 满足 `ACCEPTANCE_CRITERIA.md` 中 "Action semantics" 一行；以上验证全部通过；你确认合同升级。

---

### G5：吞吐优化（G1 完成后，与 G3/G4 并行）

**计时规则：** 每次只改一项。4096 env、seed 42、b2_bisec_full 配置、foundation profile，跑 200 iteration，取后 100 次 `Perf/collection_time` 的均值。保留条件：

- 耗时降低 ≥ 5%；
- 下表"物理检查"列全部满足；
- 同条件下 reward 曲线不出现异常。

| 编号 | 修改 | 文件 | 物理检查 |
| --- | --- | --- | --- |
| T1 | **课程等级跨阶段继承。** 在 `scripts/e0_training_orchestrator.py` 中读取上一阶段最后 100 iteration 的 `train/terrain/level` 中位数 $m$，续训时加 override `env.scene.terrain.max_init_terrain_level=round(m)`。更好的做法是把每个 env 的 `terrain_levels` 存进 checkpoint 的 extra 字段并恢复 | `e0_training_orchestrator.py`；`training/checkpoint.py` 或 `mdp/curricula/terrain.py` | 续训后前 100 iteration level ≥ m−1；`allow_terrain_stage_resume` 的合同检查仍然生效 |
| T2 | **自碰撞精简。** `enabled_self_collisions` 设为 false，或只保留腿与机身、轮与机身的碰撞对 | `env/robots/stackforce.py` | 用 G3 扫描复核无腿间穿模；`base_collision` 率不变；站立高度误差不变 |
| T3 | **求解器迭代：** `solver_position_iteration_count` 从 8 降到 4（不行则试 6）。**v3.8：两项均未通过；改为升到 32**（常量 `SOLVER_POSITION_ITERATIONS`，YAML 必须一致，有测试），采样时间 +36% | 同上 | `closure_residual_max_mm` ≤ 1；机身高度 RMSE 变化 < 2 mm；v3.8 加：静止站立外髋 `kp*err` 与静力学一致 |
| T4 | **足端传感器：** 如果没有任何 reward 或 metric 读取 pose（全局搜索 `pose_w`、`quat_w`），把 `track_pose` 从 True 改为 False | `complex_env_cfg.py` | 指标键不缺失 |
| T5 | **训练指标裁剪：** 训练时只开 core/safety/runtime/actuation/terrain；command/support/motion/reward 组以及 `wandb_panels` 关闭。完整指标只在 validation 中开启 | `configs/train/base_locomotion_e0_46d.yaml` 的 `logging` | 课程所需的键（level、tracking rmse、progress）仍然产生 |
| T6 | **足端接触传感器 `update_period`** 从 dt 改为控制周期 0.02 s；`base_contact` 保持 dt | `complex_env_cfg.py` 的 `__post_init__` | 足端 air time 相关指标的数值合理（不出现 0 或异常跳变） |
| T7 | **`replicate_physics`：** 确认 False 是否为闭环资产或随机化所必需；不需要就改为 True | `complex_env_cfg.py` | 逐 env 的质量和材质随机化生效；closure 正常 |
| T8 | **运维：** 长训期间 GPU 上不运行其他训练；暂停 b2 | 运维 | 无 |

**不做的修改：** 增加 `num_envs`（已实测更慢）；减少 rollout 步数或 PPO epoch（学习时间不到 3%）。

**产出：** `output/fix_plan/G5/timing_matrix.json`（每项的耗时、增益、物理检查结果）；在 `TRAINING_SCALE_REPORT.md` 末尾追加"E0-v2 吞吐优化"一节。

**通过条件：** 合并配置下，`collection_time` 相对 G0 下降（不预设具体数值）；所有保留项的物理检查通过；T1 已实现。

---

### G6：短探针对比

**目的：** 用尽量少的训练证明新动作合同可训练、不会继续产生动作边界饱和或数值故障，并记录后续长训需要监控的风险。本门不要求约 1k iteration 的策略性能超过旧合同。

**配置：** 合并 G4 与 G5 的修改；b2_bisec_full；seed 42；foundation profile 加低台阶段（0.02–0.04 m）。原计划 1500 iteration；实际 P0/P1 均在约 1005 iteration 结束，已足够用于动作合同诊断，不用于策略收敛结论。

| 组 | 动作范围 | 执行器 | 说明 |
| --- | --- | --- | --- |
| P0 | 旧 ±0.5 rad | 旧 | 对照，使用 G5 吞吐配置，因此单独重跑 |
| P1 | G3 的 0.9 范围 | 旧 | 主要假设 |

**动作合同资格条件（P1 取最后 300 iteration 的均值）：**

| 指标 | 条件 |
| --- | --- |
| `actuation/action_saturation_rate` | ≤ 5% |
| `actuation/leg_saturation_rate` | < 10% |
| `base_collision`、`runtime/finite_rate` | base collision 不差于 P0；finite_rate = 1 |

低台阶 `unsafe_termination`、terrain level、动态 batch-maximum closure 残差、关节速度与力矩饱和仍完整记录，但属于**策略质量观察**，不决定本诊断门 PASS。这样只说明原问题“动作合同把输出卡在边界”已修复，不说明短训策略可部署或已具备台阶能力；这些指标必须在 G8 阶段门和 G9 固定评估中重新验收。

**分支处置：**

- **动作合同资格达标：** 进入以脚本动作验证机构能力的 G7，不追加短训。
- **动作饱和或有限值不达标：** 回到 G3/G4，不进入 G7。
- **动态 closure、速度/力矩饱和、unsafe 或 level 不理想：** 作为 G8 风险保留；G3/G4 的静态与动作探针仍须已经通过。长训阶段若持续超限，由阶段门暂停，不能用 G6 PASS 覆盖。

**产出：** `output/fix_plan/G6/{p0,p1,p2}_summary.json` 加 W&B 链接，以及对照图。

---

### G7：机械净空门（长训前）

**目的：** 在花 30–41 h 长训之前，确认新范围下四条腿在自由基座与实际执行器限制下均能产生超过 30 mm 的受支持净空。整套越阶还依赖时序、移重心和轮腿协调，属于策略/控制器质量，不由本门强制要求。

**操作：**

1. **脚本步态：** 在新范围下运行 `scripts/probe_stair_climb.py --step-height 0.03`（自由基座，`--five-phase` 或 `--support-search`），以及 `record_firmware_reference_gait.py --terrain-usd <30 mm 单台阶>`。
2. **策略探针：** 用 G6 的 P1 checkpoint 运行 `probe_stair_climb.py --checkpoint ... --step-height 0.03`。只记录，不作通过条件，因为 1500 iteration 的策略预期还不会走台阶。

**通过条件：** 同一个自由基座报告中，四条腿都存在满足以下条件的候选：目标轮卸载；至少两只其他轮持续承重；倾角 < 0.35 rad；机身相对抬腿和轮下持续净空都 ≥ 30 mm；无终止。脚本整套越阶与 P1 策略探针仍记录，但不作 G7 通过条件，也不能据 G7 PASS 宣称策略已经会跨台阶。

**失败处置：** 如果任一腿不能获得受支持净空，停下报告并回到 G2/G3；如果净空通过但整套脚本失败，把失败保留为 G8/G9 的控制风险，不继续为非交付的手写步态调参。

---

### G8：正式长训

**配置：**

- 实验：`b2_bisec_full`，seed 42，合同 `E0-v2-46d`；
- 四阶段：1000 / 5000 / 7000 / 7000 iteration；
- 合并 G4、G5、G6 选定的修改；
- 从 foundation 重新开始，不使用任何旧 checkpoint；
- 启动方式：`bash scripts/run_e0_training.sh --experiment b2_bisec_full --seed 42`（已加入 T1 的课程继承）；
- 如果 GPU 时间允许，再以同样配置串行跑 `b0_t1_ppo` 作为架构对照，不与 b2 并行。

**阶段结束检查（每阶段结束后自动或人工完成；任意一项不满足就暂停，评估后再续训）：**

| 阶段结束 | 检查项 |
| --- | --- |
| Foundation（999） | `finite_rate`=1；`leg_saturation_rate` < 15%；flat 的 timeout ≥ 90% |
| Expansion（5999） | level 显著高于旧基线同期（旧基线约 4）；续训后 level 没有回落到 1 附近（验证 T1）；stairs unsafe 低于旧基线 |
| Composition（12999） | level 继续上升或稳定在更高平台；`closure_residual_max_mm` ≤ 1；饱和率 < 10% |
| Consolidation（19999） | 训练指标稳定；进入 G9 |

**中止条件（任意时刻）：**

- `runtime/all_finite` 出现 0；
- 出现 PhysX buffer error；
- 连续 500 iteration 的 `unsafe_termination` > 30%，且呈上升趋势。

**预计耗时：** 20k × G5 后的单 iteration 时间。按 7.4 s 计约 41 h；T1 减少了无效的低 level 训练，但不减少 iteration 数。

**产出：** 四个阶段的 run 目录、orchestration log 与 manifest、W&B run 链接。

---

### G9：交付验收

**操作（全部使用冻结的 `model_19999.pt`）：**

| 项 | 命令 | 产出 |
| --- | --- | --- |
| 固定 validation | `scripts/validate_policy.py --checkpoint <ckpt> --num-envs 64 --terrain-level -1 --terrain-profile union_consolidation` | `output/fix_plan/G9/validation.json` |
| 泛化 suite | `scripts/evaluate_suite.py --checkpoint <ckpt> --config configs/evaluation/generalization.yaml` | `suite.json` |
| Benchmark | `scripts/run_benchmark.py`：Plateau 和 Washboard，nominal 与 randomized 各 5 个 seed | `benchmark_*.json` |
| 台阶探针 | `scripts/probe_stair_climb.py --checkpoint <ckpt>`：30 mm、40 mm 单台阶，3×40 mm 金字塔 | `stair_*.json` 及轨迹 |
| 视频 | `scripts/record_e0_stage_videos.py` | 各阶段与各地形的 mp4 |
| 报告 | 撰写 `docs/E0_V2_LONG_RUN_REPORT.md` | 逐项 PASS/FAIL、与 b0/b2 的对比表、已知局限 |

**报告必须写明：**

1. `ACCEPTANCE_CRITERIA.md` 中每条阈值的数值与 PASS/FAIL；
2. 相对旧基线的改善：level、饱和率、stairs unsafe、台阶跨越高度；
3. DS041MG 的规格与行程实测结果；超过额定力矩的时间比例；由此带来的 sim2real 限制；
4. 0.9 范围只是仿真范围，在 M1-T06 实机 `workspace_constraints` 完成之前，不是实机安全范围。

**验收：** 由你审阅证据包，决定接受、补训或重新设计。

---

## 5. 风险登记

| 风险 | 征兆 | 应对 |
| --- | --- | --- |
| 力矩才是真正瓶颈（DS041MG 堵转 0.392 N·m，额定 0.127 N·m） | G6 中力矩饱和不降 | 降低站立高度、限制课程；不调高力矩 |
| 300° 配置与实际端点不一致 | 物理端点未测 | 名义使用 0.15°/μs；部署前另测安全端点 |
| §3A 零位或符号推导有误 | G3.0 失败 | 修正推导后重做 G3 |
| 可行域内放不下足够大的 box | G3 的 box 明显小于 §3A 区间 | 投影方案 |
| 新站姿站立力矩接近堵转 | G3 第 4 步 | 站姿向伸直方向调整 |
| 新站姿抬高机身后，倾覆或侧翻风险上升 | G6 中 tilt 和 unsafe 上升 | 下调名义高度一档后重跑 |
| 新范围导致探索过大、动作抖动 | `action_rate` 和速度触限上升 | 腿 `init_std` 调整（只允许一次） |
| 自碰撞关闭后出现穿模支撑 | G5-T2 复核失败 | 保留必要的碰撞对 |
| 课程继承引入难度跳变 | 续训初期 unsafe 激增 | 改用 $m-1$ 作为初始最大 level |
| 长训仍然停在新平台 | Expansion 结束时 level 无明显提升 | 暂停，回到 G6/G7 分析，不硬跑完 |
| **腿目标限速的价值未经证实（开放决策）** | 见 §5.1 | 正式配置保持关闭；按 §5.1 的判据和评估方法决定 |

### 5.1 开放决策：腿目标限速（`env.leg_target_max_step_rad`）

**问题。** 限速把每个 20 ms 控制步的腿目标变化限制在 0.175 rad 以内，可以去掉舵机跟不上的无效高频指令，减轻力矩饱和、发热和齿轮冲击。代价是响应变慢，可能损害支撑、落脚和扰动恢复。目前**不知道前者是否值得后者**。

**现状。** 代码与测试已实现（`mdp/action/asymmetric.py`；`tests/test_leg_slew_limit.py`），全局默认关闭，**正式配置也关闭**（`leg_target_max_step_rad: null`）。如果启用，策略观测中的“上一动作”会改为限速后的命令，实机必须在 `setAngle` 之前做同样的限幅；现有固件没有这一层。

**已有证据，及其局限。**
- 确定性评估（关闭探索噪声，`output/diagnostics/noise_eval/`）：P1 与 200 iteration 冒烟两个策略，分别约 26% 和 24% 的控制步目标变化超过 0.175 rad；带噪声时约 60%。局限：这两个策略是在旧的大噪声下训练的（较大一侧物理噪声 0.22–0.42 rad），在新噪声下重新训练后，策略本身可能就变平滑。
- 40 iteration 检查（`output/diagnostics/sanity/compare.json`）：力矩饱和 0.65→0.29、速度超限 0.68→0.05、跟踪误差 0.54→0.16 rad。局限：这次同时改了降噪和限速，两者的作用**无法分开**；而且 40 iteration 不涉及落脚和扰动恢复能力。
- 2026-10-15 补充检查（`output/diagnostics/obs_check/compare.json`）：设置为降噪、关闭限速、actor 改用舵机模型观测，取第 20–40 次 iteration。力矩饱和 0.69、速度超限 0.58、跟踪误差 0.60 rad、`action_rate_rms` 0.46，与旧噪声下的冒烟（0.65 / 0.68 / 0.54 / 0.45）相当。上一次检查（降噪 + 限速）则为 0.29 / 0.05 / 0.16 / 0.34。由此推测，训练早期的改善主要来自限速，而不是降噪：策略均值本身就在发出大幅跳变的目标。局限：这次同时改了观测，与上一次不是严格的单变量对照；40 iteration 也只反映训练早期，训练后的策略可能自己学得更平滑。
- 0.175 rad/步按 DS041MG **空载**转速 8.73 rad/s 计算。带载时舵机更慢，所以这个限幅并不保守；舵机实际跟不上的指令，在限速后仍会存在。

**约定（2026-10-15 用户再次确认）：** 默认长训不启用限速。带限速的对照实验（[奖励与课程修复方案](REWARD_CURRICULUM_FIX_PLAN.md) 中的 D 组）只用于收集证据。另外，代码目前没有推扰事件和恢复指标，所以下面判据中的“扰动恢复”一项，在补上相应测量之前无法评估。

**采用判据。** 只有同时满足以下两条才启用，并作为仿真与实机共同的设计：
1. 在降噪、修正物理模型后重新训练的策略，回放中仍有明显比例的无效高频指令（目标单步变化超过舵机实际能力，且对应关节跟踪误差高、力矩饱和）；
2. 启用限速后，支撑、落脚与扰动恢复没有明显变差。

**评估方法。**
1. 同一配置分别在限速关闭和开启下训练相同 iteration 数（消融）。
2. 用 `scripts/eval_action_noise.py` 做确定性回放，比较：
   - 目标单步变化分布、超过 0.175 rad 的比例；
   - 力矩饱和率、速度超限率、腿跟踪误差；
   - 机身高度 RMSE、倾角；
   - 台阶和箱体地形的通过率与 unsafe termination；
   - 外力推扰后的恢复率和恢复时间。
3. 如有实机回放，同时比较舵机电流和温升。

## 6. 相关文档

- 分析依据：本文 §1；[台阶探针](STAIR_PROBE.md)、[2026-10-02 台阶结果](STAIR_PROBE_RESULTS_2026-10-02.md)、[并行规模实测](TRAINING_SCALE_REPORT.md)
- 合同与验收：[Policy、Action 与 Reward 合同](POLICY_ACTION_REWARD.md)、[Locomotion 验收标准](ACCEPTANCE_CRITERIA.md)、[E0 全实验与长训命令](E0_EXPERIMENT_COMMANDS.md)
- wiki：`E-FW-001`、`E-FW-003`、`E-VAL-002`、`M1-T01`、`M1-T02`、`M1-T05`、`M1-T06`、`M2-T04`、`M3-G03`

## 7. 实现与执行记录（v3.4）

本节只记录已经写进仓库的实现和运行方式；表中"证据"一栏的文件在运行后才会出现，目前都还没有。

### 7.1 动作合同的数据来源

- `mdp/action/e0_v2_servo_contract.json`：E0-v2 的唯一数据源（逐关节范围、新站姿、被动膝初值、高度、障碍上限、探索噪声）。由 `scripts/build_servo_contract.py` 生成；无参数时复现 §3A（`status: provisional_section_3A`），G3 之后用 `--final-ranges/--stance/--lift` 重建（`status: g3_measured`）。`--check` 用于确认文件与生成器一致。
- `env/robots/kinematics.py`：直接读取包内 URDF 的五连杆运动学（纯 NumPy）。离线复核结果：零位曲柄角 154.0°/6.5°、零位足端 79 mm、伸直 158 mm、新站姿 137 mm，§3A 的 `s_i` 符号表四条腿全部一致；新站姿的被动膝初值闭环残差为 0（`tests/test_servo_action_contract.py`）。
- 由几何推出的临时高度：名义 0.163 m（几何站高 0.1636 m 减 1 mm，与 E0-v1 的 0.105/0.106 m 关系相同），初始根高 0.168 m，预警 0.124 m、终止 0.093 m（按名义高度同比放大 0.08/0.06）。G3 第 4 步实测后由生成器替换。

### 7.2 与原文的差异（实现时的取舍）

| 项 | 原文 | 实现 | 原因 |
| --- | --- | --- | --- |
| 高度指令范围 | 名义 ±0.03 m | 下限名义 −0.03 m；上限 min(名义 +0.03, 伸直腿高度 −10 mm) = 0.175 m | 原文要求"需在最终范围内可达"；伸直腿只到 0.1846 m |
| T1 课程继承 | `max_init_terrain_level=round(m)` | 启动事件把每个 env 放进 `[m−1, m]` 等级带 | 原生参数是 `[0, m]` 均匀分配，会违反"前 100 iteration level ≥ m−1"的检查 |
| G1 饱和定义 | \|raw\| > 0.95 | 新键按 `>`；旧键 `action_saturation_rate` 仍按 `>=`，保持可比 | 不改旧指标的口径 |
| 1 s 力矩均值 | "1 s 窗口内的 \|τ\| 均值" | `actuation/leg_torque_1s_mean_max_nm`：每 episode 内 1 s 滑动均值的最大值（8 个 hip 取最大），日志取批最大 | 作为发热风险项，最坏窗口最有用 |
| G3 投影方案 | 必要时在 `process_actions` 做多边形投影 | 只输出 `feasible_polygon.json` 和 `projection_recommended`；投影本身未实现 | 是否需要由 G3 结果决定，先不引入 |
| G3 自碰撞判据 | 自碰撞接触力 | 机身与移动腿车轮的接触传感器；腿与腿之间的接触无法感知 | 场景只有这两类传感器；报告中写明 |
| G6 地形 | foundation 加低台阶段 | 新地形配比 `probe_low_steps` + `env.obstacle_heights` 把台阶/箱体限定在 0.02–0.04 m | P0、P1 必须看到同一地形 |
| T2 | 关闭或只保留部分碰撞对 | 只实现整体开关 | 碰撞对过滤需要改资产，先测整体开关 |
| W&B 分组 | 未提 | E0-v2 的组名加 `e0v2-` 前缀，实验目录改为 `base_locomotion_e0_v2_46d` | 避免与 E0-v1 的 run 混在同一组和同一目录 |

旧的 b0/b2 checkpoint：E0-v2 合同下续训会被合同哈希拒绝（测试覆盖）；评估脚本（`validate_policy.py`、`run_benchmark.py`、`probe_stair_climb.py`、stepping 脚本）根据 run 目录的 `resolved_training_config.yaml` 自动选择 `legacy_symmetric` 或 `servo_range_v2`。`base_locomotion_complex.yaml` 与 `base_locomotion_hybrid.yaml` 两条旧实验线固定为 `legacy_symmetric`。`scripts/rsl_rl/play*.py` 与 `record_policy_videos.py`、`record_e0_stage_videos.py` 未改，播放旧 checkpoint 时需手动设置 `action_contract`。

### 7.3 逐门控的实现与命令

所有命令在仓库根目录、`env_isaaclab` 中运行。门控结论统一由 `scripts/fix_plan_gate.py <gate>` 读取证据文件后写入 `output/fix_plan/G<n>/gate.json`（含证据文件 SHA-256 与 commit）；失败时记录原因并停止，不跳过。

2026-10-13 实测：G0 在 4096 env、20 步预热、100 步零动作采样下为 141.22 ms/step，分段合计偏差 0.43%；物理步 59.46 ms，recorder/训练指标 25.95 ms。该同步剖析不等同于训练日志的 collection time，不能代替 G5 计时。G1 的 64 env、11 场景旧 b0 验证见 `output/fix_plan/G1/b0_validation.json`；腿饱和率在 0.36–0.66，轮饱和率在 0.002–0.145。单环境 CPU smoke 通过；短时离线 W&B 训练的六项新键和有限值可在 `wandb/offline-run-20261013_174530-tahjy8ve` 及对应 TF events 中核对。旧 b0 的性能验收仍为 FAIL，不将其诊断基线误记为可用新策略。G2 已用 wiki `E-CAL-002` 重新核验 PASS。修正被动外膝占位限位后，G3.0 的伸直/站姿、闭环、符号全部通过；0.05 rad 扫描得到逐腿连通可行域，动作端只投影不可行组合，最小抬腿 73.5 mm，站立高度 0.163 m。G4 三档动作探针、CPU smoke、64-env 零策略流程和完整 pytest 均通过。

| 门控 | 实现 | 运行 | 证据 |
| --- | --- | --- | --- |
| G0 | `scripts/profile_env_step.py`（按段同步计时）；`scripts/export_tf_metrics.py`（TF events 尾部均值） | `python scripts/profile_env_step.py --headless`；`python scripts/export_tf_metrics.py --label b0 --run "<b0 consolidation run>" --label b2 --run "<b2 consolidation run>" --output output/fix_plan/G0/baseline_metrics.json`；`python scripts/fix_plan_gate.py g0` | `G0/step_profile.json`、`G0/baseline_metrics.json` |
| G1 | `evaluation/metric/actuation.py` + `locomotion.py`：`actuation/{leg,wheel}_saturation_rate`、`leg_torque_saturation_rate`、`leg_tracking_error_rad`、`leg_torque_over_rated_rate`、`leg_torque_1s_mean_max_nm`、`runtime/closure_residual_max_mm`（每 10 步采样）；`tests/test_actuation_metrics.py` | CPU smoke：`python scripts/verify_complex_env.py --device cpu --num_envs 1 --steps 5`；b0：`python scripts/validate_policy.py --checkpoint <b0 model_19999.pt> --num-envs 64 --terrain-profile union_consolidation --output output/fix_plan/G1/b0_validation.json`；`python scripts/fix_plan_gate.py g1 --report output/fix_plan/G1/b0_validation.json --metrics-smoke-passed --wandb-keys-seen` | `G1/b0_validation.json` |
| G2 | 力矩/速度上限保持 0.392 N·m / 8.73 rad/s；额定力矩指标见 G1 | wiki 登记 `E-CAL-002` 后：`python scripts/fix_plan_gate.py g2 --wiki-evidence /path/to/wiki/docs/evidence/E-CAL-002-DS041MG规格与零位约定.md` | gate.json |
| G3 | `scripts/verify_servo_mapping.py`（G3.0）；`scripts/probe_action_range.py --mode scan`（并行网格扫描、最终范围、抬腿）与 `--mode stance`（自由基座站立） | 依次运行三个脚本（`--headless`），然后 `python scripts/fix_plan_gate.py g3`；通过后 `python scripts/build_servo_contract.py --final-ranges output/fix_plan/G3/final_ranges.json --stance output/fix_plan/G3/stance.json --lift output/fix_plan/G3/lift.json`，并按 `servo_range.leg_init_std()` 更新 YAML 中的 `init_std` | `G3/{mapping_check,range_scan,final_ranges,feasible_polygon,stance,lift}.json` |
| G4 | `AsymmetricJointPositionAction`（raw 0→站姿、±1→范围端点）；启动事件写入舵机硬限位；新站姿与闭环一致的被动膝初值；高度指令/预警/终止随站姿缩放；障碍起点 0.02 m、上限 min(原上限, 0.8×实测抬腿)；逐维 `init_std`；`protocol_version: E0-v2-46d`；旧合同 checkpoint 续训被拒 | `pytest -q`；`python scripts/probe_robot_action.py --action-magnitude {0.2,0.5,1.0} --output output/action-probes/e0v2_{0.2,0.5,1.0}.json --headless`（每个通道正负两个方向）；CPU smoke 与 64 env 零策略 validation；`python scripts/fix_plan_gate.py g4 --action-probe output/action-probes/e0v2_*.json --pytest-passed --cpu-smoke-passed --validation-smoke-passed --confirm-contract-upgrade` | gate.json（`contract_upgrade_confirmed: true` 只能由你加参数写入） |
| G5 | YAML `env.throughput.*`（T2 自碰撞、T3 求解器迭代、T4 足端 pose、T6 足端传感器周期、T7 replicate_physics）；T5 用 `logging` 覆盖；T1 在编排器 | `python scripts/fix_plan_runs.py g5`（基线 + 每项 200 iteration）；人工物理检查用 `fix_plan_gate.py g5 --manual "T3a:<检查>=ok"` 记录；保留项合并后 `fix_plan_runs.py g5 --combined <overrides>`；最后把保留项写进 `base_locomotion_e0_46d.yaml` | `G5/timing_matrix.json` |
| G6 | `scripts/fix_plan_runs.py g6`：P0 = `legacy_symmetric` + 标量 0.3 std，P1 = E0-v2；`probe_low_steps` 地形；门控只判动作合同资格，策略质量作为 G8/G9 风险 | 已有 run 可用 `python scripts/fix_plan_runs.py g6 --only P0 P1 --existing-run P0=<run> P1=<run>` 汇总；`python scripts/fix_plan_gate.py g6` | `G6/{p0,p1}_summary.json`、gate.json 中的 `policy_observations` 与 `risks` |
| G7 | `probe_stair_climb.py` 支持 E0-v2：脚本动作仍是"站姿附近的 rad 偏移/0.5"，在进入环境前换算成非对称 raw；高度指令取名义高度。`record_firmware_reference_gait.py` 的原厂轨迹是绝对髋角（E0-v1 单位），E0-v2 下按同一绝对角换算并裁剪到新范围 | `python scripts/probe_stair_climb.py --step-height 0.03 --five-phase --output output/fix_plan/G7/five_phase_30mm.json --headless`（及 `--support-search`、`record_firmware_reference_gait.py`）；`python scripts/fix_plan_gate.py g7 --report output/fix_plan/G7/*.json` | `G7/*.json` |
| G8 | 编排器：T1 等级继承、阶段结束检查（`configs/experiments/e0_v2_stage_gates.yaml`，不满足则以退出码 3 暂停）、中止条件（PhysX buffer 错误、`all_finite` 为 0、unsafe > 30% 持续 500 iteration 且上升，退出码 4）；`scripts/run_e0_v2_long_run.sh` 启动前检查 G0–G7 的 gate.json、协议版本、GPU 空闲与"无旧 v2 run" | `bash scripts/run_e0_v2_long_run.sh [--with-b0-control]`；暂停后审阅 `logs/e0_orchestration/*_gate.json`，再 `--resume-after <stage>` | `logs/e0_orchestration/` |
| G9 | 未改；现有脚本已按 checkpoint 选择合同 | 按 G9 表执行 | — |

`python scripts/fix_plan_gate.py status` 列出各门控当前记录。
