# 实机观测与执行规格（E0-v2-46d，`policy_joint_observation: servo_model`）

本文写明：要部署这里训练的策略，固件或上位机每个控制步必须计算什么。仿真中的公式与本文一致，代码位置在每节注明。

适用范围：`configs/train/base_locomotion_e0_46d.yaml` 及继承它的配置，其中 `env.policy_joint_observation: servo_model`。用 `measured`（仿真关节真值）训练的旧 checkpoint 不能按本文部署，因为它们需要实机测不到的舵机位置反馈。

## 1. 时序

- 控制周期 20 ms（50 Hz）。仿真中 `sim.dt = 0.005`、`decimation = 4`。每个控制步：读传感器 → 组装观测 → 推理 → 换算输出 → 更新舵机估计 → 下发。
- 仿真中命令立即生效（`servo_model_delay_steps: 0`）。如果实测从下发到舵机开始转动有稳定的整步延迟，改用 `servo_model_delay_steps: N` 重新训练，实机按同样的 N 计算估计。

## 2. 观测（46 维，按顺序）

坐标系：机体 `base_link`，**x 向右、y 向前、z 向上**（URDF 中 FR 在 +x、前腿在 +y）。IMU 数据必须先转到这个坐标系。

| 序号 | 维数 | 内容 | 实机来源 |
|---|---|---|---|
| 0–3 | 4 | 指令：前进速度 m/s、侧移速度 m/s、偏航角速度 rad/s、机身高度 m | 遥控映射。范围与训练一致；变化要限速，见 `mdp/policy/commands.py` 的 `max_rate`、`ranges` |
| 4–6 | 3 | 机体角速度 rad/s | IMU 陀螺（MPU6050 输出 °/s，需换算），转到机体坐标系 |
| 7–9 | 3 | 投影重力：重力单位向量在机体系中的表示，水平静止时为 (0, 0, −1) | IMU 姿态估计（加速度计 + 陀螺融合） |
| 10–17 | 8 | 髋关节**估计**位置 − 站姿，顺序 FR/FL/RL/RR 外髋，然后 FR/FL/RL/RR 内髋 | §3 舵机模型 |
| 18–21 | 4 | 0（轮子转角不观测） | 常数 0 |
| 22–29 | 8 | 髋关节估计速度 rad/s（同上顺序） | §3 差分 |
| 30–33 | 4 | 轮子**指令**角速度 rad/s，顺序 FR/FL/RL/RR | 本步下发的轮速目标（§4），不用编码器 |
| 34–45 | 12 | 上一步动作，取策略输出 [-1, 1] 原值（8 腿 + 4 轮） | 本地保存。如果启用限速，存限速后折算回的原值（`joint_target_to_raw`） |

仿真对应代码：`mdp/observation/proprioception/terms.py` 的 `servo_model_joint_position` / `servo_model_joint_velocity`，以及 `mdp/observation/history/terms.py`。仿真对角速度和重力项分别加了 ±0.02、±0.01 的均匀噪声，对关节项分别加了 ±0.002、±0.02 的噪声。被动膝关节不进入 actor 观测。

## 3. 髋舵机位置估计（实机与仿真必须相同）

DS041MG 是开环 PWM 舵机，没有位置反馈，所以 actor 看到的是下面这个模型，而不是测量值。仿真代码：`mdp/action/servo_range.py` 的 `servo_model_step`，以及 `mdp/action/asymmetric.py`。

```c
// 每个控制步、每个髋关节；target 为本步下发的关节目标角 (rad)
const float STEP = 8.73f * 0.02f;          // 空载转速 × 控制周期 = 0.1746 rad
float prev = est[i];
float d = target_delayed[i] - est[i];      // delay = 0 时就是本步 target
if (d >  STEP) d =  STEP;
if (d < -STEP) d = -STEP;
est[i] = est[i] + d;
est_vel[i] = (est[i] - prev) / 0.02f;
obs_pos[i] = est[i] - stance[i];           // stance 取自舵机合同 joints.*.default
```

- 启动和每次复位时：`est = stance`，`est_vel = 0`。机器人必须先把舵机命令到站姿并保持住，再进入策略模式。
- 已知局限：这里用的是空载转速，带载时舵机更慢，估计会领先于真实位置。仿真里这个误差同样存在（仿真舵机受力矩和速度上限约束），所以策略训练时已经见过这种偏差，但误差大小与实机未必相同。

## 4. 动作换算

**腿（前 8 维）。** 合同文件为 `mdp/action/e0_v2_servo_contract.json`。
1. 分段线性映射：`raw ≥ 0` 时 `target = default + raw·(upper − default)`，否则 `target = default + raw·(default − lower)`；再限制在 `[lower, upper]` 内。
2. 不可行的外髋/内髋组合按合同中的 `projection` 网格投影（`servo_range.project_leg_target_grid`）。
3. 可选限速（默认关闭，见 §6）。
4. 关节角换算为舵机指令角（0–300° 刻度，脉宽 500–2500 µs）。换算链见 `scripts/build_servo_contract.py`：`crank = 90° + mounting_sign·(command − zero_command)`，其中 `zero_command = 180° + trim`；FL/RR 的 `mounting_sign` 为 +1，FR/RL 为 −1；每个关节的 trim 和通道见 `CHANNELS`。关节角与曲柄角之间的换算用 `env/robots/kinematics.py` 的 `hips_for_crank_angles`。指令角最后限制在 15–285°。
5. 把换算后的 `target` 送入 §3 的估计。

**轮（后 4 维）。** 目标角速度 = `raw × 20 rad/s`，限制在 ±20 rad/s（`mdp/action/policy.py`）。仿真中轮关节是阻尼 0.5、力矩上限 1 N·m 的速度驱动。实机需要换算成 BLDC 驱动的速度单位：后轮经 Serial2 的 `SF_BLDC`；前轮由主板通过 CAN 发给第二块板 `SF_serveo_control_device2`。轮半径 0.033 m。符号按 `motorStatus.*Dir` 对齐。

## 5. 现有固件的复用与新增

位置：`lib/firmware/SF_serveo_control/src/main.cpp`，ESP32-S3。

- **可复用**：舵机输出 `servos.setAngle`（量程 0–300°、500–2500 µs）、MPU6050、后轮 `SF_BLDC`、前轮 CAN 下发、PPM 遥控、`servo_off` 零位微调。
- **需新增**：
  - 策略模式下的 50 Hz 定时循环，替代手写步态 `trot()`、`inverseKinematics()` 和姿态 PID；
  - 策略推理（46→128→128→12 的 MLP，可在 ESP32-S3 上用浮点运行，或放在上位机经串口/WiFi 收发）；
  - 本文 §2–§4 的观测组装、舵机模型、动作换算；
  - 急停与遥控接管。
- **不需要**：舵机位置反馈和轮编码器（actor 不使用）。

## 6. 未决事项

- **腿目标限速（`env.leg_target_max_step_rad`）在正式配置中关闭。** 它的价值未经证实：减少无效抖动和执行器负担，是否值得付出支撑、落脚和扰动恢复上的响应损失，目前不清楚。证据、局限、判据和评估方法见 [修复计划 §5.1](ACTION_RANGE_THROUGHPUT_FIX_PLAN.md)。如果将来启用，实机必须在 §4 第 3 步做同样的每步限幅，观测中的上一动作也要按 §2 的说明保存。
- IMU 的坐标轴对齐与零偏标定、指令通道到四维指令的映射，需要在实机上实测确认。
- 轮速单位换算尚无实测：`sim2real_calibration/README.md` 记录了后轮反馈没有 SI 换算，前轮没有反馈。
