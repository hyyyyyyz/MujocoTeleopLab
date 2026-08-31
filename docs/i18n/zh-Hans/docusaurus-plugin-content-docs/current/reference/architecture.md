---
sidebar_position: 2
---

# 系统架构

本页定义 Teleopit 的运行时流程、仓库布局、支持的技术范围和公共入口。

## Pipeline

![Teleopit 运行时流程](/img/diagrams/architecture-pipeline-zh.svg)

全身运控主流程把 BVH 或 PICO 实时身体动作转换为时间对齐的 G1 参考。
`VelCmdObservationBuilder` 把参考动作与机器人状态组合起来，双输入 TemporalCNN
ONNX 运控器再输出 29 维关节偏移。同一套观测和运控器路径同时用于 MuJoCo 和真机 G1。

Pico 手部和主动视觉路径是可选的进程隔离 worker。它们复用同一个进程内
`PicoBridge` 接收器，不会向 167 维运控策略观测增加字段。手部或颈部故障不能停止
G1 身体控制。这些可选硬件路径只支持机载部署；外部主机 Pico 部署只支持全身控制。

主机高层策略部署与 Pico 运行时彼此独立。单独的主机环境接收 JPEG RGB、G1 实测关节
位置、O6 原始实测 readback、OpenNeck 实测角度，以及 observation 时刻的 source
reference root pose。身体、手部和颈部数组组成 43 维模型观测；session-local source
pose 只用于重建 source-relative root 输出。主机再通过严格的 ZeroMQ/msgpack 消息返回
canonical `float32[T,50]` action chunk。机载校验器和调度器把其中的身体部分转换为
36 维参考，交给现有 motion tracker；主机输出不能绕过 tracker，也不能直接成为电机
命令。

Teleopit 和主机环境共享语义数据和一份完全相同的 `hand_calibration.json`，但不会导入
对方的 Python 包。当前 client/server 代码和协议测试定义网络结构，因此协议变化时
两个仓库必须同步修改。

## 运行时边界

- 离线核心组件通过 `InProcessBus` 通信，不复制数组 payload。
- 真机机器人控制、参考生成、相机、录制、手部、颈部和高层策略客户端在可能阻塞或
  硬件故障影响 50 Hz 控制循环时使用进程隔离。
- 本地 sim2real worker 使用 localhost ZeroMQ 和共享内存视频环。
- 外部主机策略边界使用 msgpack 和非 pickle 的 float32 数组。
- 共享组件契约是在 `teleopit/interfaces.py` 中定义的 `typing.Protocol`。

## 仓库布局

```text
teleopit/                              — 核心推理和部署包
├── interfaces.py                     — 机器人、运控器、输入和重定向协议
├── pipeline.py                       — 轻量离线仿真 facade
├── runtime/                          — 配置/路径解析、工厂和 CLI 校验
├── configs/                          — Hydra 运行时配置
├── bus/                              — 进程内零拷贝发布/订阅
├── inputs/                           — BVH、PICO 和实时输入适配器
├── retargeting/gmr/                  — 自包含的全身 GMR 实现
├── controllers/                      — 观测构建器和 ONNX 策略运控器
├── robots/                           — MuJoCo 机器人适配器
├── sim/                              — 200 Hz PD / 50 Hz 策略仿真循环
├── sim2real/
│   ├── mp/                           — 进程 supervisor、IPC 和机器人控制状态机
│   ├── hands/                        — 可选 LinkerHand 驱动和输入映射
│   └── neck/                         — 可选 OpenNeck 映射和 worker
├── high_level_policy/                — 主机协议、坐标变换和 action 调度器
└── recording/                        — Sim2real 数据 schema 和录制 worker

train_mimic/                          — 训练包
├── app.py                            — 共享的训练/播放/benchmark 装配
├── tasks/tracking/                   — General-Tracking-G1 任务和 TemporalCNN 模型
├── data/                             — 数据集构建和动作加载
└── scripts/                          — 训练、播放、benchmark 和 ONNX 导出

scripts/                              — 面向用户的运行和维护入口
├── run/                              — 仿真、sim2real 和录制命令
├── setup/                            — 资源下载和硬件设置
├── render/                           — 离线视频渲染
├── view/                             — 录制数据检查
└── dev/                              — 校验和标定工具

third_party/                          — 可选硬件 SDK 和 somehand
tests/                                — 单元、协议和集成测试
```

## 技术规格

| 规格 | 支持值 |
|------|--------|
| 机器人 | 29 个执行关节的 Unitree G1 |
| 仿真器 | MuJoCo |
| 全身重定向 | GMR（General Motion Retargeting） |
| 策略 / PD 频率 | 50 Hz / 200 Hz |
| 训练任务 | `General-Tracking-G1` |
| 推理观测 | `velcmd_history`（167 维） |
| ONNX 签名 | 双输入：`obs`（167 维）+ `obs_history` |
| 策略动作 | 相对 `default_dof_pos` 的 29 维关节偏移 |
| Actor / critic | TemporalCNN（2048、1024、512、256、128） |
| 训练采样 | 默认 `rewind`；支持 `uniform`；播放使用 `start`；benchmark 固定精确 clip 并禁用 clip 末尾重采样 |
| 训练窗口 | `window_steps=[0]` |
| 分发动作数据 | 递归 minimal HDF5 `shard_*.h5` 文件 |
| 可选手部 | LinkerHand L6/O6，支持 gripper 或 PICO 手部姿态输入 |
| 可选主动视觉 | 使用物理角度的 OpenNeck yaw/pitch |
| 主机策略观测 | JPEG RGB + G1 关节位置（29 维）+ O6 原始 readback（12 维）+ OpenNeck 角度（2 维）；请求还携带相机时刻的 active reference root pose（7 维） |
| 主机策略动作 | `float32[T,50]`，30 Hz 源时间线，`T` 在 `[1,50]` 内 |
| 主机策略身体控制 | 36 维根部/关节参考，通过现有 50 Hz motion tracker |

## 约束

- `controller.policy_path` 必须显式提供，并指向现有文件。
- 离线 BVH 运行必须显式提供现有的 `input.bvh_file`。
- `viewers` 是唯一的 viewer 配置键。
- 观测定义必须与 ONNX 签名完全一致；启动时会直接失败，不会 pad 或 trim 数据。
- `default_dof_pos` 必须来自所选机器人的默认站立角度。
- sim2real 使用与仿真相同的双输入观测契约。
- 主机消息 envelope 或 schema 不匹配时会被拒绝，机器人保持在 `STANDING`。shape、
  有限值、session、sequence、四元数、时效性或安全检查失败时，会拒绝整个 action
  chunk。
- 主机 action 在机载侧校验、调度并限速；主机不能绕过 motion tracker 或发送 G1
  电机命令。
- 策略 entry 在一个主机 session 等待第一份有效 chunk 时保持为 `STANDING` 内部流程。
  该 chunk 会直接进入 `POLICY`，不执行候选参考对齐、entry Kp ramp 或第二次
  session/reset。50 Hz limiter 从 session 开始时捕获的机器人实测参考起步。
- chunk 边界和内部的根部、yaw 与关节参考时间跳变都会被接受，再由 50 Hz scheduler
  输出限速，从而保留录制的 pause/resume 转换。
- PICO 输入、RealSense 预览、录制、手部和颈部故障都是非关键故障；Unitree 遥控器
  和机器人控制循环仍然可用。

## 公共入口

支持的运行模式包括离线 sim2sim、离线 sim2real 回放、PICO sim2sim、PICO G1
sim2real，以及独立主机高层策略 G1 sim2real。

运行命令：

- `scripts/run/run_sim.py` — 离线 BVH 和 PICO 实时 sim2sim
- `scripts/run/run_sim2real.py` — BVH 或 PICO G1 sim2real
- `scripts/run/run_high_level_policy_sim2real.py` — 独立主机高层策略 G1 部署
- `scripts/run/record_pico_motion.py` — 从 PICO 录制重定向动作 clip
- `scripts/render/render_sim.py` — 渲染 mocap、重定向和 sim2sim 视频
- `scripts/view/view_recording.py` — 检查同步的 sim2real 录制数据

训练和数据命令：

- `train_mimic/scripts/train.py`、`play.py`、`benchmark.py`、`save_onnx.py`
- `train_mimic/scripts/data/build_dataset.py`
- `train_mimic/scripts/data/precompute_dataset.py`

公共 Python 接口：

- `teleopit/interfaces.py` 中的协议
- `TeleopPipeline`
- `VelCmdObservationBuilder`
- `RLPolicyController`
