---
sidebar_position: 4
---

# 使用 XRoboToolkit 进行桌面场景遥操作

此工作流用于基于物理仿真的 MuJoCo 桌面操作场景：包含 43 自由度 Unitree G1、两只 Dex3 手、平衡/行走策略、手臂 IK 与可碰撞物体。它与 [XRoboToolkit MuJoCo 教程](xrobotoolkit-sim2sim.md)中 29 自由度的全身追踪工作流相互独立。请勿同时启动两者，因为它们会争用 XRoboToolkit。
此工作流仅用于仿真，不会向真实 G1 发送命令。

## 包含内容

初始场景使用已发布的 `decoupled_wbc` G1 资源：

| 场景 | 内容 |
| --- | --- |
| `cube` | 桌子和一个可自由运动的小立方体，用于第一次抓取和放置测试。 |
| `bottle` | 桌子和一个可自由运动的瓶子。 |
| `box` | 桌子和一个较大的可自由运动的盒子。 |

场景运行时以 200 Hz 运行 MuJoCo 物理，以 50 Hz 运行已发布的 Balance/Walk 全身策略。43 个受控关节为 29 个身体关节与每只手 7 个 Dex3 关节。机器人和物体会一同重置，因此失败的抓取不需要重启应用。

## 一次性配置

先配置常规 XRoboToolkit PC Service 与其 Python 3.12 绑定：

```bash
bash scripts/setup/setup_xrobotoolkit.sh
```

然后创建上游 WBC、Pinocchio IK 与 43 自由度 MuJoCo 资源使用的独立 Python 3.10 环境：

```bash
bash scripts/setup/setup_scene_teleop.sh
```

若 `third_party/decoupled_wbc` 不存在，配置脚本会克隆它。该目录和 `.venv_scene` 是本地依赖，Git 有意忽略它们。
`.venv_scene` 只用于 Python 3.10 的 WBC/IK/MuJoCo 栈，不会安装主 `teleopit` 包；场景启动脚本会直接从仓库根目录导入代码。常规 `.venv` 仍用于 Python 3.12 的 XRoboToolkit 桥接器及 Teleopit 依赖。

## Pico 配置

1. 启动本地 PC Service：

   ```bash
   .tools/xrobotoolkit/service/opt/apps/roboticsservice/runService.sh
   ```

2. 在 Pico 的 **XRoboToolkit** 中连接 PC，直到状态显示 **WORKING**。
3. 开启 **Head**、**Controller** 与 **Send**。
4. 在 **Remote Vision** 中选择 **ZEDMINI** 并点击 **Listen**。记下 Pico 当前的局域网 IPv4 地址。

此场景工作流不需要全身追踪或脚踝追踪器；它仅使用 HMD 与两只控制器。

## 启动立方体场景

在项目根目录运行；若 Pico 地址变化，请替换该地址：

```bash
PICO_VIDEO_HOST=<Pico IPv4> bash scripts/run/start_scene_teleop.sh --scene cube
```

该启动脚本要求设置 `PICO_VIDEO_HOST`，不会静默回退到可能已经失效的旧地址。若只需本地 MuJoCo 窗口或无头检查、不需要 Pico Remote Vision，请显式关闭视频：

```bash
SCENE_NO_VIDEO=1 bash scripts/run/start_scene_teleop.sh --scene cube
```

启动器还会拒绝重复的场景/桥接进程，因为它们会争用同一个本地 XR 数据端口。重新启动前请先停止现有启动器。

这一条命令会启动两个本地进程：

1. `.venv` 读取 XRoboToolkit 控制器/HMD 数据，并仅发送到 localhost UDP 桥接。
2. `.venv_scene` 运行 43 自由度 WBC、MuJoCo 场景、本地 viewer 与 Remote Vision 发送器。

直连 Remote Vision 由 PC 主动连接 Pico 的 TCP 12345。场景采用 SIMPLE 的
ZEDMINI 媒体规格：先渲染单目 1280x720，再复制成并排的 2560x720 H.264
流，帧率为 60 FPS。若只是诊断机器负载，可以传入 `--video-fps 30`，正常头显
使用建议保留默认的 60 FPS。若启用防火墙，PC Service 仍需要与标准
XRoboToolkit 教程相同的局域网规则。

视频流变为 live 后，转动头部即可观察模拟场景。场景运行时会把 **Head** 提供的 HMD
姿态应用到 `scene_head_camera`；进程收到的第一份有效 HMD 姿态作为中立参考，之后只应用
相对于该参考的偏航/俯仰/滚转变化。HMD 平移会被有意忽略，因此相机会继续固定在 MuJoCo
躯干上。重启场景进程即可建立新的中立方向。

若需不打开渲染器的冒烟测试，运行：

```bash
.venv_scene/bin/python scripts/run/run_scene_teleop.py \
  --scene cube --headless --seconds 10 --no-video
```

如需更严格且确定性的验收检查，请运行控制器位姿、抓握、接触和物体运动冒烟测试：

```bash
.venv_scene/bin/python scripts/dev/smoke_scene_teleop.py
```

它必须报告站立根部高度、至少一个立方体接触体，以及至少 1 cm 的立方体位移。该命令用于首次在 Pico 操作前验证此立方体场景，或在更新场景控制栈后复查。

### Pico 充电期间运行

本地物理检查不需要 Pico。可以显式跳过 Python 3.12 的 XRoboToolkit 桥接器和出站视频线程，然后以无头方式运行独立的 MuJoCo 进程：

```bash
SCENE_NO_BRIDGE=1 SCENE_NO_VIDEO=1 \
bash scripts/run/start_scene_teleop.sh \
  --scene cube --headless --seconds 10 --no-realtime
```

这会检查场景加载、43 自由度执行器映射、WBC 初始化、PD 步进和干净退出；没有 XR 数据时会安全等待，因此不需要开机的头显或正在运行的 PC Service。上面的 `smoke_scene_teleop.py` 还会通过合成手柄序列驱动手臂 IK 和立方体接触，全程不打开网络套接字。

## Pico 手柄映射

该映射有意遵循 SIMPLE 的 Pico 解耦 WBC 交互。标明为边沿触发的控制在按下后应松开组合键。

| Pico 输入 | 场景操作 |
| --- | --- |
| 左手 **Menu** + 左手食指扳机 | 切换行走输入锁。锁定时平衡仍保持激活，机器人会持续站立。 |
| 左手 **Menu** + 右手食指扳机 | 切换手臂/手部遥操作；进入时校准手臂参考。 |
| 左摇杆 | 前后行走与横移。 |
| 右摇杆左右 | 转动行走参考。 |
| 左/右食指扳机 | 对应手的食指捏合手势。 |
| 食指扳机 + 侧握把 | 对应手的力量抓握手势。 |
| 按住 **X** | 降低目标基座高度。 |
| 按住 **Y** | 将目标基座高度升至站立高度。 |
| 两侧握把同时按住 | 重置机器人、物体、WBC 状态与手臂校准。 |
| **B** | 保留给 XRoboToolkit Remote Vision 的单目/双目切换。 |

机器人稳定站立后，如需行走先按一次 Menu + 左食指，再按一次 Menu + 右食指。启动手臂遥操作时请将两只控制器保持在舒适的中立位置：该位置会成为手臂参考。逐步把控制器移动到物体处，再使用手部手势闭合抓取。本地 MuJoCo viewer 仍可用于调试，Remote Vision 则提供头显视角。

## 选择或添加场景

使用一个已包含的场景名：

```bash
bash scripts/run/start_scene_teleop.sh --scene bottle
bash scripts/run/start_scene_teleop.sh --scene box
```

对于新任务，传入一个恰好暴露已发布的 43 个关节执行器名称（29 个 G1 身体关节 + 14 个 Dex3 手关节）的自定义 XML：

```bash
bash scripts/run/start_scene_teleop.sh --scene-xml /absolute/path/my_scene.xml
```

运行时会在启动时插入 `scene_head_camera`，源 XML 保持只读。请使自由物体保持动态且可碰撞；双侧握把提供常规 episode 重置动作。

## 故障排除

| 现象 | 检查 |
| --- | --- |
| `No XR packet` 或控制器无响应 | PC Service 与 Pico 必须为 `WORKING`；开启 Head、Controller 与 Send。不需要 Full-body。 |
| 终端显示 `waiting for bridge packets` | 场景进程未收到本地 Python 3.12 桥接数据。重启 `start_scene_teleop.sh`；正常时会显示 `Scene XR input: ... Hz`。 |
| WBC 导入或资源错误 | 重新运行 `bash scripts/setup/setup_scene_teleop.sh`；不要把 WBC 栈安装到常规 `.venv` 中。 |
| Remote Vision 为白色/空白 | 在监听打开前，终端会报告 `Scene Remote Vision waiting for Pico Listen`。在 Pico 中选择 ZEDMINI 并点击 Listen，再确认终端变为 `connected`/`live`；将 `PICO_VIDEO_HOST` 设置为头显当前 IP。 |
| PC Service 连接冲突 | 启动场景工作流前关闭旧的 29 自由度 `run_sim.py` 工作流。 |
| 机器人/物体状态不可用 | 同时按住 Pico 两侧握把以重置整个场景。 |
