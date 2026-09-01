---
sidebar_position: 3
---

# 在 MuJoCo 中使用 XRoboToolkit 进行 Pico 遥操作

这是 Pico 到 MuJoCo G1 的工作流。它使用 XRoboToolkit PC Service 和头显应用获取追踪数据，再复用 Teleopit 的 G1 重定向与 MuJoCo 控制器；它**不会**连接真实机器人。

如需使用 Dex3 手进行桌面物体交互、摇杆移动和已发布的解耦全身控制器，请使用独立的[场景遥操作工作流](scene-teleop.md)。两套运行时不能同时启动。

## 一次性本地配置

当前工作区的忽略目录 `.tools/xrobotoolkit/` 已包含下载好的 PC Service 与 Pico APK。如需在本机重新创建这套配置，请运行：

```bash
bash scripts/setup/setup_xrobotoolkit.sh
```

该脚本下载官方 Ubuntu 24.04 x86_64 PC Service，在本地解包（不安装到系统），编译官方 `xrobotoolkit_sdk` Python 绑定并安装到 `.venv`。它需要 `c++`、`cmake` 和 `dpkg-deb`。

## 启动 PC Service

在项目根目录的终端 1 运行：

```bash
.tools/xrobotoolkit/service/opt/apps/roboticsservice/runService.sh
```

保持该终端运行。SDK 使用本地 TCP 60061；面向头显的服务在局域网 TCP 63901 上接受连接。

## 配置 Pico 头显

使用 ADB 安装 `.tools/xrobotoolkit/XRoboToolkit-PICO-1.1.1.apk`，或从 [XRoboToolkit Unity Client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1) 下载相同版本。

```bash
.tools/platform-tools/adb install -r .tools/xrobotoolkit/XRoboToolkit-PICO-1.1.1.apk
```

头显和 PC 必须连接同一局域网。在头显中：

1. 从未知来源打开 **XRoboToolkit**。
2. 选择发现到的 PC 地址，或输入 PC 的局域网 IPv4 地址后选择 **Connect**。状态必须变为 **WORKING**。
3. 在 **Tracking** 下开启 **Head** 和 **Controller**。
4. 在 **PICO Motion Tracker** 中选择 **Full-body**。先在 Pico 系统 Motion Tracker 应用中配对并校准两个脚踝追踪器。
5. 开启 **Send**。

## 在启动仿真前验证追踪

在终端 2 运行：

```bash
.venv/bin/python scripts/dev/test_xrobotoolkit.py --seconds 20
```

稍作移动。继续前，该命令必须报告递增的序列号和稳定帧率。

## 运行 MuJoCo G1 遥操作

在同一终端运行：

```bash
PICO_VIDEO_HOST=<Pico 当前 IPv4> \
.venv/bin/python scripts/run/run_sim.py \
  --config-dir "$PWD/local" \
  --config-name pico4_xrobotoolkit_mujoco \
  controller.policy_path=ckpt/track_g1.onnx
```

仿真器从 `STANDING` 启动，手柄交互遵循 SIMPLE 的流程。先等待机器人稳定站立 4--6 秒，并在 XRoboToolkit 头显应用中启用 **Controller**，然后直接使用 Pico 手柄：

| SIMPLE 手柄输入 | Teleopit 操作 |
| --- | --- |
| 按住左手 **Menu**（三横线按键）并扣下右手食指扳机 | 开始全身 `MOCAP`。这是边沿触发的启动方式：短按后松开两者。 |
| 同时扣住左右侧握把（左右手中指） | 安全结束当前动捕会话，返回 `STANDING`，并重新对齐参考。 |
| `B` | 保留给 XRoboToolkit Remote Vision 的双目/单目视图切换；Teleopit 不绑定此键。 |

PC Service 尚未收到至少一帧 Full-body 数据前，启动组合键会被忽略。Teleopit 直接跟踪身体动作：此 G1 sim2sim 工作流没有解耦 WBC 的导航或手部执行器栈，因此不会复制 SIMPLE 的摇杆行走/转向、`X` 下蹲和扳机控制夹爪。电脑键盘仍可作为可选备用方式：`Y` 启动动捕、`A` 暂停/恢复、`B` 切换仅手臂模式、`X` 返回站立、`Q` 退出。

## MuJoCo 第一人称视频

本地预设 `pico4_xrobotoolkit_mujoco` 会将模拟 G1 的 `d435i_rgb` 相机画面发送到头显。当前 XRoboToolkit 头显版本会直接在 TCP 12345 开放媒体监听，启动时请将 `PICO_VIDEO_HOST` 设置为头显当前的局域网 IPv4 地址。预设从环境变量读取地址，不再硬编码旧地址，因为 Pico 的 DHCP 租约可能在重连后变化。PC 会持续重连该监听，并把模拟的单目画面复制为 2560×720、60 FPS 的 H.264 双目流（单目渲染尺寸为 1280×720，与 SIMPLE 的 ZEDMINI 规格一致）。

如果启用了 UFW，只允许本地局域网访问 XRoboToolkit 服务端口（请将网段替换为你自己的局域网）。直连视频由 PC 主动连接头显，不需要放行入站 TCP 13579：

```bash
sudo ufw allow from 10.0.90.0/23 to any port 63901 proto tcp
```

在头显中打开 **Remote Vision**，选择 **ZEDMINI**（该名称用于选择 H.264 接收器），并点击 **Listen**。如果 Pico 重新通过 DHCP 获取了 IP，请修改启动命令中的 `PICO_VIDEO_HOST` 后重启仿真，不要继续使用上一次会话的地址。旧版头显中需要输入 PC 地址的流程仍通过 TCP 13579 支持，但不是此本地预设的默认流程。

如需打开三个标准调试窗口（mocap、retarget 和 sim2sim），请使用基础配置
`--config-name xrobotoolkit_sim`。该基础配置默认不启用视频；如果显式设置
`input.video.enabled=true` 且未设置 `input.video.direct_host`，Teleopit 才会通过 TCP
13579 回退到旧版 `OPEN_CAMERA` 协商。如果需要当前头显的 TCP 12345 直连 Remote Vision，
请使用上面的本地 `pico4_xrobotoolkit_mujoco` 预设。无窗口冒烟测试可增加
`viewers=none +num_steps=8`。

## 故障排除

| 现象 | 检查项 |
| --- | --- |
| PC Service 已启动但没有帧 | 头显界面显示 `WORKING`；已开启 Full-body 和 Send；两个脚踝追踪器已配对并校准。 |
| SDK 导入失败 | 在项目根目录重新运行 `bash scripts/setup/setup_xrobotoolkit.sh`。 |
| 服务无法绑定端口 60061 | 停止已有的 `RoboticsServiceProcess`；只能运行一个 PC Service 实例。 |
| 仿真等待数据 | 先运行 `scripts/dev/test_xrobotoolkit.py`，不要从 MuJoCo 开始排查。 |
| Remote Vision 显示空白白框 | 确认仿真以 `pico4_xrobotoolkit_mujoco` 启动、选择 `ZEDMINI`、点击 **Listen**，并将 `PICO_VIDEO_HOST` 设置为头显当前的局域网 IPv4 地址。 |
