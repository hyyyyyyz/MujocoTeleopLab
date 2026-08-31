---
sidebar_position: 2
---

# 在仿真中进行 VR 遥操

连接真实机器人之前，先用 Pico 控制仿真 G1。不要跳过这一步：头显、网络和身体追踪
问题都可以在这里解决，不会给硬件带来风险。

## 支持的头显

- Pico 4
- Pico 4 Ultra
- Pico 4 Ultra Enterprise
- Pico 4 Pro

所有设备都需要开启全身追踪，并使用支持当前身体追踪接口的 Pico 系统版本。

## 开始之前

你需要：

- 头显和运行 Teleopit 的电脑处于同一网络；
- 已安装 `pico4` 依赖并下载 `robots gmr ckpt bvh` 资源；
- [在仿真中运行运控](offline-sim2sim)已经正常。

## 1. 准备头显

1. 从 [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases)
   下载头显 APK。
2. 安装 APK：

   ```bash
   adb install pico-bridge.apk
   ```

3. 在头显中打开 pico-bridge。
4. 开启全身追踪。

Teleopit 使用 pico-bridge 0.2.1。接收程序会直接运行在 Teleopit 进程中，不需要再
启动一个单独的转发程序。

## 2. 检查电脑是否收到 Pico 数据

下面的诊断只打印身体帧和连接状态，不会启动机器人运控：

```bash
python scripts/dev/test_pico_bridge.py --no-video
```

轻微移动身体，确认终端持续收到新的有效帧。按 `Ctrl+C` 结束诊断。

如果自动发现选择了错误的网卡地址，显式指定头显能够访问的地址：

```bash
python scripts/dev/test_pico_bridge.py \
    --no-video \
    --bridge-advertise-ip=192.168.1.20
```

## 3. 启动仿真

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx
```

机器人会有意从 `STANDING` 开始；只有操作者主动切换后，实时身体追踪才会接管。

## 4. 按状态机操作

![Pico 仿真控制状态机](/img/diagrams/pico-sim-state-machine-zh.svg)

图中的**键盘**表示运行 Teleopit 的电脑键盘，**Pico 手柄**表示 VR 手柄。仿真过程
不使用 Unitree G1 遥控器。

以舒适的中立姿态站好，等待追踪稳定，再按**键盘** `Y` 进入 `MOCAP`。先从小幅慢动作
开始。按**键盘** `X` 结束 VR 会话并返回 `STANDING`；在任意状态按**键盘** `Q`
退出仿真。

`MOCAP` 控制全身。`ARMS` 会让身体、腰和腿保持站立，只有双臂继续跟随。
`PAUSED` 保持当前参考姿态，恢复后返回之前的 `MOCAP` 或 `ARMS`。

每次重新从 `STANDING` 进入 `MOCAP` 时，系统都会重新对齐实时根部姿态。操作者可以
在站立状态改变朝向，再重新进入 `MOCAP`。

:::tip 暂停不等于结束 VR 控制
键盘或 Pico 手柄 `A` 只会冻结并恢复当前动捕姿态。需要结束会话并回到站立时，
请按键盘 `X`。
:::

## 选择 Viewer 布局

Pico 仿真默认会打开动捕、重定向和物理仿真三个视图。不再需要全部视图时，可以减少窗口：

```bash
# 只看物理仿真结果
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx \
    viewers=sim2sim

# 不打开窗口
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx \
    viewers=none
```

## 可选：头显视频

把仿真的 `d435i_rgb` 相机画面发送回头显：

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx \
    input.video.enabled=true
```

使用 `input.video.source=test-pattern` 可以只检查视频链路。视频失败时预览会关闭，但
身体追踪和运控会继续运行。

## 网络参数

大部分网络只需要自动发现。诊断显示网络有问题时再使用这些参数：

```bash
# 向头显广播指定的电脑地址
input.bridge_advertise_ip=192.168.1.20

# 关闭自动发现并显式绑定
input.bridge_discovery=false
input.bridge_host=0.0.0.0
input.bridge_port=63901

# 延长等待第一帧身体数据的时间
input.pico4_timeout=30
```

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| 收不到身体帧 | 把 Pico 头显升级到最新可用的系统版本，重启头显，重新开启全身追踪，然后再次运行 `scripts/dev/test_pico_bridge.py --no-video` |

这条流程稳定后，再继续[用 VR 遥操真实 G1](pico-sim2real)。
