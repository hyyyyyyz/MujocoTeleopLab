---
sidebar_position: 4
---

# 配套项目

Teleopit 集成了四个职责明确的组件，分别负责机器人通信、灵巧手重定向、主动视觉和
PICO 数据传输。它们位于 `teleopit` Python 包之外，各自维护硬件协议和公共 API。

| 组件 | 源码地址 | 功能 | 在 Teleopit 中的用途 |
|------|----------|------|-----------------------|
| G1 Bridge SDK | [Teleopit 源码目录](https://github.com/BotRunner64/Teleopit/tree/master/third_party/g1_bridge_sdk) | 基于 Unitree SDK2、Cyclone DDS 和 pybind11 的原生 C++ bridge | 获取 G1 实时状态和遥控器输入、切换模式，并发送 200 Hz 底层命令 |
| somehand | [GitHub](https://github.com/BotRunner64/somehand) | 灵巧手动作重定向库 | 把 Pico 实时手部 landmark 映射为 LinkerHand L6/O6 目标 |
| OpenNeck | [GitHub](https://github.com/BotRunner64/OpenNeck) | 带标定的双轴颈部驱动 | 把物理 yaw/pitch 角度转换为安全的舵机命令 |
| PICO Bridge | [GitHub](https://github.com/BotRunner64/pico-bridge) | 传输 PICO 追踪和视频的头显应用与 Python 接收器 | 提供身体、手柄、手部和 HMD 帧，并可回传 RGB 视频 |

## G1 Bridge SDK

G1 Bridge SDK 直接维护在 Teleopit 的 `third_party/g1_bridge_sdk` 中，不是单独的
仓库。安装脚本会下载 [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)，
然后构建并安装本地 pybind11 扩展：

```bash
bash scripts/setup/setup_g1_bridge.sh
```

所有 DDS 发布和订阅都运行在原生 C++ 线程中。Teleopit 的 `UnitreeG1` 适配器通过
bridge 读取关节状态、基座方向、角速度和无线遥控器输入，并发送带逐关节 PD 增益的
29 关节位置目标。真机遥操、独立站立测试和主机高层策略部署都使用这个硬件边界。

## somehand

somehand 提供可配置的人手到机器人手部动作重定向。Teleopit 把兼容源码固定为
`third_party/somehand` Git submodule，并使用 0.3.0 的公共 `somehand.api`。

在 `hands.mode=vr_hand_pose` 下，Teleopit 把 PICO 的 26 关节手部状态转换为 21 个
landmark，调用 somehand 连续重定向，再把结果发送给 LinkerHand L6 或 O6。Pico
实时接收和 landmark 转换由 Teleopit 负责，不会启动 somehand 自带的 Pico 输入路径。

安装灵巧手依赖：

```bash
git submodule update --init --recursive
pip install -e third_party/linkerhand-python-sdk
pip install -e third_party/somehand
```

## OpenNeck

OpenNeck 负责双轴主动视觉云台的串口通信、角度到舵机 step 的转换和标定机械限位。
Teleopit 支持 OpenNeck 0.2.0 的物理角度 API，并调用 `move_deg()`；已经移除的
normalized 控制字段不兼容。

在 Pico 遥操中，Teleopit 计算 HMD 相对同帧 `Body.Spine3` 的旋转，应用配置的死区和
俯仰增益，再由非关键 neck worker 发送 yaw/pitch 角度。主机高层策略部署则发送
canonical action 中经过校验的颈部字段。

```bash
pip install -e '.[openneck]'
openneck calibrate
```

## PICO Bridge

PICO Bridge 同时包含头显应用和可导入的 Python PC 接收器。Teleopit 支持 0.2.1
版本，由 `pico4` extra 安装：

```bash
pip install -e '.[pico4]'
```

一个进程内 `PicoBridge` 实例为 Teleopit 提供全身、手柄、手部和独立 HMD 数据。
全身重定向、手部控制和 OpenNeck 共用这个接收器。启用视频后，Teleopit 还可以通过
`push_video_frame()` 把 MuJoCo 或 RealSense RGB 帧推回头显。

头显 APK 从 [PICO Bridge Releases](https://github.com/BotRunner64/pico-bridge/releases)
下载。
