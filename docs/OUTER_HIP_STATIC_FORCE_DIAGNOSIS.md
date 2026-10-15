# FL/RR 外髋静态内力诊断（2026-10-14）

## 结论

G3 站立中 FL/RR 外髋力矩偏高（73–75% 堵转），FR/RL 只有 19–23%。这不是舵机合同的映射错误，也不是同一条腿内外髋互相顶。原因是**四个轮子的碰撞体不一致**。

包内 closed USD 的四个轮子各用一个独立导出的网格，再做 `convexHull` 近似。PhysX 对每个网格烘焙出的凸包不同，静止时轮心离地面 31.4–33.0 mm（标称半径 33 mm），四个轮子不共面。左后（RL）轮心比右后（RR）轮心低 1.6 mm。机身由四点支撑，属于超静定结构，腿的刚度约 5–7 N/mm。四个轮心高度差出约 1.27 mm 的对角“跛桌”误差后，FL/RR 这条对角线就多承担了约 ±1.6 N 的载荷。

把四个轮子换成相同的碰撞体后，对角不平衡基本消失，四个外髋的受力也拉平了：

- 球体：各轮法向力 3.56–3.73 N，外髋 PhysX 力矩 0.121–0.130 N·m；
- 32 边棱柱：各轮法向力 3.49–3.81 N，外髋 PhysX 力矩 0.116–0.131 N·m。

外髋力矩的理论值可以用五连杆雅可比由静力学算出：每个轮子平均承担 3.65 N 时为 0.1295 N·m。上面两组 PhysX 实测值与它一致。

后来又发现第二个原因：位置迭代 8 次时闭环在速度层不收敛，伺服额外承担非物理内力。修复状态见文末“待决定”。

## 方法

脚本是 `scripts/diagnose_outer_hip_force.py`，每次只做几秒的仿真。动作为零，使用 E0-v2 舵机合同的默认站姿，平地，随机化全部关闭，结果写到 `output/diagnostics/outer_hip_force_<tag>.json`。每个关节记录以下四个量：

| 量 | 含义 |
|---|---|
| `target` / `actual` | 关节位置目标值与实际值，二者之差为 `err` |
| `kp*err` | 20 × 平均误差，即 PD 驱动律给出的静态力矩 |
| `torque_est` | Isaac Lab `applied_torque`。这是隐式执行器的估计值，G3 `stance.json` 用的就是它 |
| `proj` | PhysX `get_dof_projected_joint_forces` |

`proj` 的外髋值与雅可比静力学结果一致，所以下文以它为外髋力矩的主要依据。对内髋，`proj` 在所有工况下都恒为 ±0.0082 N·m，没有反映闭环传来的载荷，不可用。

**注意：** `fix_plan_common.Rig(suspended=True)` 每步把基座传送回原位，基座在步内其实处于自由落体，所以关节感受不到重力载荷。用它做的悬空测试等价于零重力。下文的“悬空”工况改用 `--fix-root`，在 PhysX 中真正固定基座。

## 结果

### 1. 悬空，固定基座，有重力，2 s（迭代 8 / 16 次）

| 关节 | FR | FL | RL | RR |
|---|---|---|---|---|
| 外髋 `proj` (N·m) | +0.0505 | +0.0505 | −0.0505 | −0.0505 |
| 外髋 `err` (rad) | +0.00191 | +0.00191 | −0.00191 | −0.00191 |
| 内髋 `err` (rad) | +0.00146 | −0.00146 | +0.00146 | −0.00146 |
| closure 残差 (mm) | 0.021 | 0.021 | 0.021 | 0.021 |

四条腿严格镜像，外髋约占堵转力矩的 13%。腿离地后，FL/RR 不再偏大。

每条腿内外髋力矩的符号，与雅可比静力学预测的符号一致，即 J^T·F 的符号（FL/RR 内髋轴取反）。腿内没有对抗。改用零重力和自由落体基座时，八个髋关节的力矩都约为 1e-6 N·m。

迭代次数从 8 增加到 16，结果不变（误差差异小于 1e-5 rad）。

### 2. 闭环一致性与合同审查

- 五连杆在矢状面内有 2 个自由度，由两个髋共同决定。所以内髋不是外髋的函数，“由外髋目标反推内髋”这个问题本身不存在。要检查的是两点：给定 (外髋, 内髋) 时，被动膝的初值能让闭环闭合；URDF 模型与仿真一致。
- URDF 五连杆模型在默认站姿下的 closure 残差为 0.000 mm（四条腿）。合同里的被动膝初值与仿真稳态值差在 0.003 rad 以内。仿真足端位置与模型的偏差 ≤ 0.0002 mm。
- 四条腿在默认站姿下的足端位置（基座系）为 (±97.30, ±86.545, −130.624) mm，四腿之间相差不超过 0.2 µm。
- `build_servo_contract.py` 中，`mounting_sign` 对 FL/RR 取 +1、对 FR/RL 取 −1，只影响 `lower`/`upper` 端点，不影响 `default`。默认站姿由 `hips_for_crank_angles` 直接从 URDF 推出，四腿统一。不存在只在对角两腿出现、由镜像翻转两次造成的错误。
- USD closure joint 的 `localPos1/localRot1` 在 FL/RR 与 FR/RL 之间相差 2.5e-7 m 和 7.5e-6 rad，可以忽略。

### 3. 落地站立，3 s，取最后 1 s 的平均值

| 工况 | 轮心高度 FR/FL/RL/RR (mm) | 轮法向力 FR/FL/RL/RR (N) | 外髋 `proj` FR/FL/RL/RR (N·m) | 外髋 `kp*err` (N·m) |
|---|---|---|---|---|
| 原资产，迭代 8 次 | 32.20 / 31.90 / 31.45 / 33.02 | 2.03 / **5.28** / 2.26 / **5.01** | 0.045 / **0.221** / 0.069 / **0.195** | 0.10 / **0.44（已饱和）** / 0.16 / **0.36** |
| 原资产，迭代 16 次 | — | 1.70 / 5.74 / 2.04 / 5.11 | 0.034 / 0.215 / 0.051 / 0.189 | 0.06 / 0.25 / 0.06 / 0.21 |
| 原资产，轮初始角 0.05 / 0.10 rad | — | 1.95 / 5.38 / 2.14 / 5.12；2.07 / 5.23 / 2.33 / 4.95 | 不变 | — |
| 原资产，地面摩擦 0.2 | — | 2.00 / 5.31 / 2.32 / 4.95 | 不变 | — |
| 原资产，hullVertexLimit=255 | — | 结果逐位相同（设置不生效） | — | — |
| **球形轮碰撞体** | 33.00 ×4 | 3.56 / 3.73 / 3.56 / 3.73 | 0.121 / 0.130 / 0.121 / 0.129 | 0.24 / 0.26 / 0.24 / 0.24 |
| **32 边棱柱轮碰撞体** | 33.00 / 33.01 / 33.16 / 33.14 | 3.64 / 3.65 / 3.49 / 3.81 | 0.124 / 0.124 / 0.116 / 0.131 | 0.24 / 0.25 / 0.23 / 0.24 |

对原资产的分析：

- 轮心在基座系内的四个高度不共面，对角误差为 (z_FR + z_RL) − (z_FL + z_RR) = 1.27 mm。刚体落在平面上，四个触点本应共面，所以这个误差只能来自四个轮子的有效半径不同。
- 法向力的偏差几乎全部落在扭转模态上：FR −1.61、FL +1.64、RL −1.38、RR +1.37 N，前后、左右两个方向的分量都接近 0。机身质心偏移只能产生前后或左右方向的载荷差，产生不了扭转，所以排除质心偏移。
- 轮初始角、摩擦系数、求解器迭代次数都不影响结果。可能的解释是，自由转动的轮子会滚到它凸包上最平的那个面停下，因此停止状态与初始相位无关。

### 4. 求解器迭代 16 次 vs 8 次

静态 closure 残差：迭代 8 次时最大 0.25 mm，16 次时最大 0.09 mm。对角载荷与外髋力矩基本不变（FL 0.221 → 0.215 N·m）。因此残差不是求解器容差造成的。P1 训练里的 11–16 mm 闭环残差，是训练过程中动态工况下整批次的最大值，与这个静态问题无关，需要单独分析。

## 第二个原因：闭环约束只在位置层保持（2026-10-14 追加）

轮子碰撞体修好后，迭代 8 次时外髋的 `kp*err` 仍约为静力学值的 2.6 倍。追查的结果如下。

**1. 闭环传力本身是对的。** 固定基座、足端施加已知的 3.65 N 竖直力：外髋 `kp*err` 的增量为 0.120 N·m，内髋为 0.091 N·m；雅可比静力学给出 0.130 和 0.094，分别吻合到 92% 和 97%。

**2. PhysX 报告的关节速度是“幽灵速度”。** 关节位置静止不动（按位置差分算出的速度为 1e-5 rad/s），但 PhysX 报告的速度不为零：

| 工况 | 外髋 | 内髋 | 被动膝 | 轮 |
|---|---|---|---|---|
| 悬空 | ±0.08 | ±0.11 | — | — |
| 落地站立 | ±0.31 | ±0.18 | ±0.48 | ±0.33 |

单位均为 rad/s。用这些速度经五连杆运动学推算，closure 点在以每秒 2.7 mm（悬空）到 8 mm（站立）的速度分离，再由迭代的位置修正每步拉回。也就是说，闭环只在位置层被维持，速度层并不满足。

后果：
- 伺服阻尼和轮阻尼作用在幽灵速度上。站立时每个轮子因此多出约 0.16–0.2 N·m 的阻尼力矩，经地面摩擦变成纵向内力。
- 为了平衡这些非物理的力，外髋多承担约 0.15 N·m，位置误差比静力学需要的大约 2.6 倍。
- Isaac `applied_torque` 中 kd·qd 一项被幽灵速度污染。
- 策略观测到的关节速度在静止时也不为零。

**3. 加大位置迭代次数能收敛（速度迭代从 1 改到 4 无效）。** 球形轮、站立：

| 位置迭代 | 静态 closure 残差 (mm) | closure 分离速度 (mm/s) | 外髋 `kp*err` (N·m) | 外髋报告速度 (rad/s) |
|---|---|---|---|---|
| 8 | 0.108 | 8.0 | 0.238 | 0.31 |
| 16 | 0.028 | 4.8 | 0.154 | 0.28 |
| 32 | 0.005 | 1.7 | 0.114 | 0.24 |
| 64 | 0.001 | 0.4 | 0.096 | 0.20 |

参考值：静力学 0.1295 N·m 减去腿自重项 0.038 N·m，预期约 0.091 N·m。迭代 64 次时外髋已与静力学一致，但报告的关节速度仍不为零。

另外，把 closure joint 从轮子（Foot_Link）改挂到外小腿上，结果完全不变，可以排除“closure 挂在轮子上”这一原因。

**4. 计时。** 条件：4096 环境，`b2_bisec_full`，零动作，100 个控制步，用 `scripts/profile_env_step.py` 测量，未训练。

| 位置迭代 | 每控制步 (ms) | 其中物理 (ms) | 每次 PPO 迭代的采样时间 (s) | 相对 8 次 |
|---|---|---|---|---|
| 8 | 121.7 | 55.8 | 2.92 | — |
| 16 | 137.4 | 70.9 | 3.30 | +13% |
| 32 | 165.3 | 99.2 | 3.97 | +36% |

## 轮子碰撞体修复（已实现，资产待安装）

- `assets/.../scripts/wheel_colliders.py` 是纯 pxr 实现。它从四个轮子网格测出半径（33.000 mm）和轴向范围，先确认四个轮子互为镜像，再禁用 `*_foot` 网格碰撞体，在每个 `*_Foot_Link` 下添加相同的 `wheel_collider`：32 边棱柱，64 个顶点，`convexHull`，平面到轮心 33 mm。
- `recover_closed_loops.py` 在导出 closed USD 之后调用这一步。
- 用重建后的 USD 与包内现有 USD 做了语义对比：组合后的机器人层级中，只多出 4 个 `wheel_collider`，4 个 `*_foot` 网格设了 `collisionEnabled=False`，其余逐属性一致。
- 包内的 `usd/stackforce_quadrupedal_wheeled_robot_closed.usda` 这次没能覆盖，因为写入被权限拦截了。需要手动安装。
- `tests/test_wheel_colliders.py`（pxr）检查两点：棱柱平面距轮心 33 mm；四个轮子只有同一种镜像棱柱碰撞体。它对重建产物通过，对当前包内资产失败（失败是预期的）。
- 仿真验收：`diagnose_outer_hip_force.py --phase stand --check-wheel-settle`。要求轮心高度极差 ≤ 0.3 mm、对角扭转 ≤ 10%，不满足时退出码为 1。

G3 第 4 步在重建资产上的结果。原资产为：轮心极差 1.571 mm，扭转 0.411，未通过。

| 位置迭代 | 轮心极差 (mm) | 扭转 | 轮法向力 FR/FL/RL/RR (N) | 外髋 `proj` (N·m) | 外髋 `kp*err` (N·m) | 内髋 `kp*err` (N·m) |
|---|---|---|---|---|---|---|
| 8 | 0.266 | 0.017 | 4.52 / 2.77 / 2.65 / 4.65 | 0.167 / 0.085 / 0.078 / 0.175 | 0.318 / 0.189 / 0.161 / 0.329 | 0.090 / 0.011 / 0.073 / 0.119 |
| 16 | 0.203 | 0.015 | 4.24 / 3.06 / 2.94 / 4.35 | 0.153 / 0.096 / 0.090 / 0.158 | 0.189 / 0.124 / 0.113 / 0.191 | 0.090 / 0.036 / 0.073 / 0.107 |
| 32 | 0.109 | 0.012 | 3.69 / 3.66 / 3.52 / 3.72 | 0.122 / 0.120 / 0.114 / 0.124 | 0.103 / 0.104 / 0.097 / 0.103 | 0.081 / 0.075 / 0.094 / 0.092 |

静力学参考：每个轮子 3.65 N 时，地面反力项外髋 0.1295、内髋 0.093 N·m；扣除腿自重后约 0.091 和 0.064 N·m。内髋的 `proj` 恒为 0.008，不可用。

- 迭代 8 次时，对角扭转已经消除，但仍有左右方向的偏载（右侧承担 63%）。轮子停在棱柱的不同相位上（轮心 32.91–33.17 mm），再加上幽灵速度产生的轮阻尼力，造成了这个偏载。
- 迭代 32 次时，四个轮子受力均匀，外髋约 0.10 N·m（26% 堵转），内髋约 0.09 N·m（23% 堵转），与静力学基本一致。

## 待决定

1. 安装重建后的 USD，即用重建产物覆盖包内的 closed USD。之后重跑 `pytest tests/test_wheel_colliders.py` 和 `--check-wheel-settle`。
2. ~~位置迭代次数~~：用户确认后已改为 32（`env/robots/stackforce.py` 的 `SOLVER_POSITION_ITERATIONS`，YAML 一致且有测试）。

   200 iteration 冒烟：P1 合同与覆盖参数，seed 42，候选棱柱轮资产，结果在 `output/diagnostics/smoke/compare.json`。与 P1 在 100–200 iteration 窗口对比：

   | 指标 | P1 | 冒烟 |
   |---|---|---|
   | closure_residual_max_mm | 6.44 | 0.24 |
   | 腿跟踪误差 (rad) | 0.673 | 0.432 |
   | 腿力矩饱和率 | 0.649 | 0.552 |
   | 超额定力矩率 | 0.790 | 0.618 |
   | 平均回合长度 | 395 | 604 |
   | 每 iteration 耗时 (s) | 5.66 | 8.75（+55%） |
3. G3 的力矩口径：改用 `kp*err`，前提是迭代足够、收敛；也可改用外髋的 PhysX 投影力。不要再用 `applied_torque`。

## 复现

```bash
cd scripts
PY=/home/kytolly/Utils/Anaconda/envs/env_isaaclab/bin/python
$PY diagnose_outer_hip_force.py --headless --phase air --fix-root [--solver-iters 16]
$PY diagnose_outer_hip_force.py --headless --phase stand --seconds 3 [--solver-iters 16] [--wheel-init-rad 0.1] [--friction 0.2]
$PY diagnose_outer_hip_force.py --headless --phase air0g --fix-root --foot-force-z 3.65   # 已知足端力 vs 静力学
$PY make_wheel_collider_variant.py --shape prism --output-root /tmp/asset_prism
SF_QUAD_ASSET_ROOT=/tmp/asset_prism $PY diagnose_outer_hip_force.py --headless --phase stand --seconds 3 --tag _prism
# 重建 closed USD（含统一轮碰撞体）到临时目录并验收
cd ../source/base_locomotion_stackforce_quadrupedal/base_locomotion_stackforce_quadrupedal/assets/robots/stackforce_quadrupedal_wheeled_robot
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME /home/kytolly/Library/isaacsim/python.sh \
  scripts/recover_closed_loops.py usd/stackforce_quadrupedal_wheeled_robot/stackforce_quadrupedal_wheeled_robot.usda \
  --output /tmp/asset_rebuilt/usd/stackforce_quadrupedal_wheeled_robot_closed.usda   # 退出时 Kit 段错误属既有现象，文件已写出
SF_QUAD_ASSET_ROOT=/tmp/asset_rebuilt $PY -m pytest -q tests/test_wheel_colliders.py
SF_QUAD_ASSET_ROOT=/tmp/asset_rebuilt $PY scripts/diagnose_outer_hip_force.py --headless --phase stand --seconds 3 --solver-iters 32 --check-wheel-settle
```
