# E0 Curriculum Configurations

课程轴独立保存，避免把 motion、terrain 和 robustness 的门槛混成一个不可审计的配置：

- `terrain.yaml`：按地形难度和 episode 表现晋级/回退；
- `motion.yaml`：维护运动族覆盖、成功率和非零复习采样下限；
- `robustness.yaml`：记录 nominal、moderate、mixed/stress robustness bin。

这些文件是 `configs/train/base_locomotion_e0_46d.yaml` 中 `curriculum` 区块的版本化来源。三轴均不向 Actor 增加观测，Actor 合同保持 `4 + 30 + 12 = 46`。
