---
sidebar_position: 3
---

# 用 VR 遥操真实 G1

本教程会把已经在 MuJoCo 中验证过的 Pico 流程迁移到真实 Unitree G1。先确定 Teleopit
运行在哪里，再验证站立运控，最后才把机器人交给实时身体追踪。

:::danger 始终把 Unitree 遥控器拿在手里
动作异常时立即按 `L1+R1` 进入 `DAMPING`。清空机器人周围空间，并安排另一名人员
随时扶住或停止机器人。
:::

## 选择部署方式

### 外部主机部署：仅全身追踪

Teleopit 运行在工作站或笔记本电脑上，电脑通过网线连接 G1；Pico 头显需要能够通过
网络访问这台电脑。

这种部署方式只用于 G1 全身控制，请保持 LinkerHand、OpenNeck、RealSense 画面和
数据录制关闭。先查看连接 G1 的有线网卡：

```bash
ifconfig
```

在后面的命令中填写这块网卡的名称。本文以 `enp130s0` 为例。

### 机载电脑部署：完整具身能力

如果还需要 LinkerHand、OpenNeck、RealSense 画面或数据采集，请直接在 G1 机载电脑
上运行 Teleopit。Pico 头显需要能够访问机载电脑。

机载配置同时使用 O6 双手和 OpenNeck 时，请将底层运控策略设为
`controller.policy_path=ckpt/track_g1_neck_o6.onnx`。

G1 DDS 默认使用 `eth0`。除了网络接口和可选机载硬件配置之外，全身控制的配置和启动
命令与外部主机部署相同。

## 开始之前

请确认以下条件全部满足：

- [在仿真中进行 VR 遥操](pico-sim2sim)已经稳定运行；
- 已按照[安装](../getting-started/installation)安装 `pico4` 依赖并编译
  `g1_bridge_sdk`；
- 已准备好 `ckpt/track_g1.onnx`、机器人文件和 GMR 资源；
- 运行 Teleopit 的设备已经通过有线 DDS 网络连接 G1；
- 没有其他程序正在控制机器人。

## 1. 检查站立运控

先只检查机器人状态接收和策略频率，不发送电机命令。外部主机需要把 `enp130s0`
替换为 `ifconfig` 查到的有线网卡：

```bash
python scripts/run/standalone_standing.py \
    --policy ckpt/track_g1.onnx \
    --network-interface enp130s0 \
    --dry-run
```

在机载电脑上运行时，使用 `--network-interface eth0`。

Dry run 成功后，在确保硬件安全的情况下去掉 `--dry-run` 再运行一次：

```bash
python scripts/run/standalone_standing.py \
    --policy ckpt/track_g1.onnx \
    --network-interface enp130s0
```

站立运控不稳定时不要继续接入 Pico，请先按照
[单独测试站立运控](standalone-standing)排查。

## 2. 启动 Pico 真机遥操

外部主机示例：

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=ckpt/track_g1.onnx \
    real_robot.network_interface=enp130s0
```

机载电脑示例：

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=ckpt/track_g1.onnx \
    real_robot.network_interface=eth0
```

程序启动后不会立即让 Pico 接管机器人。

## 3. 按 G1 状态机操作

![Pico G1 控制状态机](/img/diagrams/pico-g1-state-machine-zh.svg)

图中的 **G1 遥控器**表示 Unitree 遥控器，**Pico 手柄**表示 VR 手柄。电脑键盘不负责
切换真机状态。

先按 **G1 遥控器** `Start` 进入 `STANDING`。等机器人站稳，以中立姿态站好，并确认
Pico 追踪有效。然后按 **G1 遥控器** `Y` 进入 `MOCAP`，从缓慢的小幅动作开始。需要
结束 VR 会话时，按 **G1 遥控器** `X` 返回 `STANDING`。

`MOCAP` 控制全身。`ARMS` 会让身体、腰和腿保持站立，只有双臂继续跟随。
`PAUSED` 保持当前参考姿态，恢复后回到暂停前的 `MOCAP` 或 `ARMS`。

进入 `MOCAP` 前，Teleopit 会连续检查多帧 Pico 数据。检查没有通过时，机器人会继续
停留在 `STANDING`。

:::tip 暂停和恢复
G1 遥控器 `B` 或 Pico 手柄 `A` 会暂停、恢复当前会话。恢复时请保持静止，并尽量接近
暂停时的姿态。需要结束会话时，请使用 G1 遥控器 `X`。
:::

如果 Pico 输入中断，全身运控会保持最后一个参考，G1 遥控器仍然可用。按 `X` 返回
`STANDING`，或按 `L1+R1` 进入 `DAMPING`；不要等待系统自动切换状态。

## 仅机载：LinkerHand

只有 LinkerHand 已连接到机载电脑时才需要本节。先按照
[安装](../getting-started/installation)安装手部依赖，再启用两路 CAN：

```bash
sudo /usr/sbin/ip link set can0 up type can bitrate 1000000
sudo /usr/sbin/ip link set can1 up type can bitrate 1000000
```

启动 G1 运控前，先单独测试双手：

```bash
python scripts/dev/test_linkerhand.py \
    --driver linkerhand_o6 \
    --hand-type both \
    --left-can can0 \
    --right-can can1
```

在真机启动命令后追加以下参数，即可启用 O6 手部姿态控制：

```text
hands.enabled=true
hands.driver=linkerhand_o6
hands.mode=vr_hand_pose
hands.linkerhand_o6.left_can=can0
hands.linkerhand_o6.right_can=can1
```

使用 `hands.mode=gripper` 时，需要按住对应手柄侧面的握持扳机键（grip）才会启用
该侧手部控制；保持按住后，再用食指扳机键（trigger）控制闭合程度。松开侧面握持
扳机键会让该侧手张开。LinkerHand L6 也受支持，对应参数为
`hands.linkerhand_l6.*`。

## 仅机载：OpenNeck

安装并校准 OpenNeck：

```bash
pip install -e '.[openneck]'
openneck calibrate
```

然后在真机启动命令后追加：

```text
neck.enabled=true
neck.port=/dev/ttyACM0
```

OpenNeck 会根据 Pico 头显相对操作者上身的运动转动，并复用已有的 Pico 接收程序。

## 仅机载：RealSense 画面

安装 `pyrealsense2`，再追加：

```text
input.video.enabled=true
input.video.device=<可选的-realsense-序列号>
```

相机画面会发送到头显。相机超时后会在后台重连，不会停止 Pico 追踪或 G1 运控。

## 仅机载：录制和查看数据

录制前必须能够收到新的 RealSense RGB 帧：

```bash
python scripts/run/run_sim2real.py \
    --config-name sim2real_record \
    controller.policy_path=ckpt/track_g1.onnx \
    real_robot.network_interface=eth0 \
    recording.task="向前走"
```

在终端按 `R` 开始一个 episode，按 `S` 保存，按 `D` 丢弃，按 `Q` 关闭程序。如果
一秒内没有收到新的相机帧，当前 episode 会被丢弃，但机器人运控会继续。视频恢复后
需要手动重新开始录制。

查看已保存的数据：

```bash
pip install -e '.[review]'
python scripts/view/view_recording.py \
    --recording data/recordings/sim2real_hdf5
```

查看器会同步显示相机视频、G1 实测与参考姿态，以及可选的手部和头部信号。字段说明
见[遥操数据集](../reference/resources/teleoperation-datasets)。

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| Arm 设备上 RealSense 无法使用 | 先运行 `pip uninstall pyrealsense2` 删除 PyPI wheel，再运行 `conda install -c conda-forge pyrealsense2` 安装 conda-forge 提供的 Arm 构建 |

## 其他 G1 工作流

- [单独测试站立运控](standalone-standing)
- [在 Unitree G1 上回放 BVH](bvh-sim2real)
- [从遥操数据到模仿学习 / VLA 真机部署](high-level-policy-sim2real)
