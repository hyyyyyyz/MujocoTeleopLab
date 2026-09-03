---
sidebar_position: 1
slug: /
---

# MujocoTeleopLab

MujocoTeleopLab 是一个以仿真为先的研究工作区，面向 **PICO/XRoboToolkit
遥操作、桌面操作和 VLA 数据生成**。主要目标是 43 自由度 Unitree G1/Dex3
MuJoCo 场景：操作者可以控制机器人与物体交互，也可以使用 CuRobo 自动生成
具备碰撞约束的轨迹，用于数据采集。

仓库同时保留了兼容的 `teleopit` 全身动作重定向运行时，用于标准 G1 全身实验。
保留这个包名是为了兼容已有 API 和配置；项目本身以 MujocoTeleopLab 独立维护，
并非 Teleopit 官方发行版。

同一套运控策略会先在 MuJoCo 中运行。你可以先在仿真里确认动作和控制方式，再连接
真实机器人。

## 从这里开始

第一次使用 MujocoTeleopLab 时，建议按这个顺序：

1. 根据自己的目标[安装 MujocoTeleopLab](getting-started/installation)，并完成该页面最后的
   安装检查。
2. 从下面七条路径中选择一条继续。

| 我想做什么 | 对应教程 |
|------------|----------|
| 在 MuJoCo 中检查运控策略 | [在仿真中运行运控](tutorials/offline-sim2sim) |
| 不连接真机，先尝试 Pico VR 遥操 | [在仿真中进行 VR 遥操](tutorials/pico-sim2sim) |
| 使用 XRoboToolkit Pico 追踪和标准 G1 sim2sim 运控器 | [在 MuJoCo 中使用 XRoboToolkit 进行 Pico 遥操作](tutorials/xrobotoolkit-sim2sim) |
| 在 MuJoCo 中用 Pico、Dex3 和场景 WBC 操作桌面物体 | [使用 XRoboToolkit 进行桌面场景遥操作](tutorials/scene-teleop) |
| 使用 Pico VR 控制真实 G1 | [用 VR 遥操真实 G1](tutorials/pico-sim2real) |
| 训练并导出自己的运控策略 | [训练运控策略](tutorials/training) |

:::warning 连接真机之前
请先把 Pico 仿真遥操跑通。真机运行时始终把 Unitree 遥控器拿在手里；
`L1+R1` 是进入 `DAMPING` 的紧急停止方式。
:::

## 想了解实现细节？

主线教程只保留完成任务所需的内容。运行流程和技术规格见
[系统架构](reference/architecture)，下载文件与资源分组见
[资产](reference/resources/assets)，Hydra 参数见
[配置说明](reference/configuration/overview)。
