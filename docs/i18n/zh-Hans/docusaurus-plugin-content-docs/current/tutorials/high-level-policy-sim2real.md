---
sidebar_position: 4
---

# 从遥操数据到模仿学习 / VLA 真机部署

本教程串联完整工作流：使用 Teleopit 录制 Pico 示教，在 `lerobot-teleopit` 中训练
ACT 或 GR00T N1.7 策略，再把训练结果运行到 Unitree G1 真机上。

```text
Pico 示教
  -> Teleopit v4 录制数据
  -> LeRobot Dataset
  -> ACT 或 GR00T checkpoint
  -> 主机策略服务
  -> Teleopit onboard motion tracker
  -> G1 + LinkerHand O6 + OpenNeck
```

Teleopit 负责录制和实时机器人控制；
[`lerobot-teleopit`](https://github.com/BotRunner64/lerobot-teleopit)
负责数据转换、模型训练和主机策略服务。两个仓库使用相互独立的 Python 环境；主机发送
reference motion，而不是 G1 电机命令。

## 开始之前

- [Unitree G1 VR 遥操作](pico-sim2real)已经可靠运行。
- Onboard 配置包含两只 LinkerHand O6、OpenNeck 和一台 RealSense RGB 相机。
  当前训练与部署路径要求这些硬件全部存在。
- Onboard 计算机已经按照[安装指南](../getting-started/installation)安装 recording、
  OpenNeck、LinkerHand 和 somehand 支持，并且存在
  `ckpt/track_g1_neck_o6.onnx`。
- 使用相同 G1 网络接口和底层 tracking policy 时，
  [独立站立测试](standalone-standing)已经稳定运行。
- 主机工作站已经单独按照
  [`lerobot-teleopit` 安装指南](https://github.com/BotRunner64/lerobot-teleopit#installation)
  准备完成。

:::danger 始终把 Unitree 遥控器拿在手中
动作异常时立即按 `L1+R1` 进入 `DAMPING`。确保机器人周围有足够的安全空间，
安排另一人随时准备扶住或停止机器人，并且不要同时运行两个可能向 G1 发送命令的程序。
:::

## 1. 录制并检查示教

在 G1 onboard 计算机上运行录制配置。下面的示例使用 Pico 手部姿态重定向；
如果示教需要使用手柄扳机，请改用 `hands.mode=gripper`。

```bash
python scripts/run/run_sim2real.py \
  --config-name sim2real_record \
  controller.policy_path=ckpt/track_g1_neck_o6.onnx \
  real_robot.network_interface=eth0 \
  hands.enabled=true \
  hands.driver=linkerhand_o6 \
  hands.mode=vr_hand_pose \
  neck.enabled=true \
  recording.output_dir=data/recordings/my_task \
  recording.task="pick up the object"
```

使用 G1 遥控器进入 `MOCAP` 或 `ARMS`，然后操作录制终端：

| 按键 | 动作 |
|------|------|
| `R` | RealSense 有新鲜画面后，开始一条 episode |
| `S` | 保存当前 episode |
| `D` | 丢弃当前 episode |
| `Q` | 关闭运行时 |

每个 dataset 只录制一项任务，并保持 `recording.task` 一致。只保存成功示教，
同时覆盖有意义的初始姿态、物体位置和执行速度变化。

训练前检查同步视频、实测状态和 reference：

```bash
python scripts/view/view_recording.py \
  --recording data/recordings/my_task
```

出现追踪丢失、相机中断或不安全 reference 时，应丢弃对应 episode。录制 schema
和恢复规则见[遥操数据集](../reference/resources/teleoperation-datasets)。

## 2. 将 Dataset 交给 `lerobot-teleopit`

把完整录制目录复制到主机，不要展开目录层级或修改其中的名称。典型 source 目录为：

```text
lerobot-teleopit/data/raw/my_task/
├── schema.json
├── episodes.jsonl
├── data/
└── videos/d435i_rgb/
```

当前转换器要求 Teleopit v4 dataset 包含 LinkerHand O6 和 OpenNeck 的 state/action
字段；缺失字段会被拒绝，不会自动补齐。如果同一任务录制在多个目录中，请在转换前使用
主机仓库的 `merge_raw_datasets.py` 工具合并。

后续所有主机命令都应在独立的 `lerobot-teleopit` 环境中运行。其
[数据转换与训练指南](https://github.com/BotRunner64/lerobot-teleopit/blob/main/docs/training-entrypoint.zh-CN.md)
包含依赖、合并选项、训练规模、多 GPU 设置和实验日志说明。

## 3. 在主机上转换并训练

最短的数据转换命令为：

```bash
python scripts/convert_dataset.py \
  --source data/raw/my_task \
  --output data/lerobot/my_task \
  --repo-id local/my_task \
  --workers 4
```

选择一种训练命令。训练 ACT：

```bash
python scripts/train_policy.py \
  --policy act \
  --dataset-root data/lerobot/my_task \
  --devices 0
```

训练 GR00T N1.7：

```bash
python scripts/train_policy.py \
  --policy groot \
  --dataset-root data/lerobot/my_task \
  --devices 0,1,2,3
```

追加 `--dry-run` 可以检查最终启动配置，而不实际开始训练。如果没有设置
`--output-dir`，训练结果会写入 `outputs/train/`。可部署产物是对应 run 下的
`checkpoints/last/pretrained_model/` 目录。

## 4. 使用 ReplayPolicy 验证真机链路

加载 learned checkpoint 前，先从主机回放一条已经录制的 episode：

```bash
python scripts/run_policy_server.py \
  --backend replay \
  --dataset-root data/lerobot/my_task \
  --repo-id local/my_task \
  --episode 0 \
  --start-frame 0 \
  --chunk-size 15 \
  --bind tcp://0.0.0.0:5555
```

只在可信的机器人网络上绑定 `0.0.0.0`。该服务没有身份验证，不得暴露到公网。

在 G1 onboard 计算机上启动 Teleopit 专用运行时。将 `HOST_IP` 替换为工作站地址，
并使用与 dataset 一致的任务描述：

```bash
python scripts/run/run_high_level_policy_sim2real.py \
  controller.policy_path=ckpt/track_g1_neck_o6.onnx \
  high_level_policy.endpoint=tcp://HOST_IP:5555 \
  high_level_policy.task="pick up the object" \
  real_robot.network_interface=eth0
```

进程启动后，机器人保持 `IDLE`。使用 Unitree 遥控器操作：

| 操作 | 动作 |
|------|------|
| `Start` | 进入 `STANDING` |
| `Y` | 创建策略 session；第一份有效 chunk 会进入 `POLICY` |
| `B` | 暂停，或在新鲜 chunk 可用后恢复 |
| `X` | 结束 session 并返回 `STANDING` |
| `L1+R1` | 立即进入 `DAMPING` |

ReplayPolicy 应当足够准确地重现录制 reference，以验证网络、action 约定和 onboard
执行链路。如果回放不正确，请在这里停止。Learned policy 无法修复录制、转换、坐标或
底层 tracking 问题。

## 5. 部署训练好的策略

按 `X` 让 G1 返回 `STANDING`，然后停止 ReplayPolicy。在主机上使用
`pretrained_model` 目录本身启动 learned-policy server：

```bash
python scripts/run_policy_server.py \
  --backend lerobot \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --device cuda \
  --bind tcp://0.0.0.0:5555
```

ACT 和 GR00T 使用相同的 server 命令。按 Unitree 遥控器 `Y` 创建新的策略 session。
开始时使用训练分布内熟悉的场景，并只允许幅度小、容易恢复的动作。

如果需要记录一次运行中交换的 observation 和 action，增加以下主机参数：

```bash
--record-dir outputs/policy-recordings
```

Teleopit 会在 50 Hz motion tracker 使用之前校验并限速每一份 plan。格式错误的输出
会被拒绝，不会被补齐或删减。主机、网络、相机或 action watchdog 故障会暂停 session
并保持最后一条命令，不会自动进入 `STANDING`。恢复故障链路后按 `B` 继续，或根据情况
使用 `X` 或 `L1+R1`。

## 常见问题

| 现象 | 检查项 |
|------|--------|
| 按 `Y` 后始终不进入 `POLICY` | 主机 IP 和防火墙、server 日志、新鲜的 RealSense 画面、两边匹配的代码版本，以及完全一致的 `hand_calibration.json` |
| `POLICY` 进入暂停 | 主机推理延迟、请求超时、相机/结果过期、action watchdog，或必要 worker 退出 |

模型 action 坐标和主机端行为见
[`lerobot-teleopit` Action Space 指南](https://github.com/BotRunner64/lerobot-teleopit/blob/main/docs/planar-relative-root-actions.zh-CN.md)。
Onboard 时序和安全设置见
[配置字段](../reference/configuration/fields#主机-high-level-policy独立-sim2real)。
