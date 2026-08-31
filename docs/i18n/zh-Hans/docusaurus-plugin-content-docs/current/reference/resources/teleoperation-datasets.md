---
sidebar_position: 3
---

# 遥操数据集

遥操数据集是手动录制的 sim2real episode。它同步保存 G1 状态、motion tracker
消费的参考动作、可选的手部和颈部命令，以及 RealSense RGB 视频。这个格式用于数据
检查和外部策略开发，不是运控训练使用的动作数据集。

## 录制 Episode

录制只支持带交互终端和新鲜 RealSense 画面的 Pico 机载 sim2real 部署：

```bash
pip install -e '.[recording]'
python scripts/run/run_sim2real.py --config-name sim2real_record \
    controller.policy_path=policy.onnx
```

手动配置的等价条件是 `recording.enabled=true`、`input.provider=pico4`、
`input.video.enabled=true` 和 `input.video.source=realsense`。

终端按 `R` 开始一条 episode，按 `S` 保存，按 `D` 丢弃，按 `Q` 关闭运行时。
`STANDING`、`MOCAP`、`ARMS` 和动捕暂停状态都可以录制。没有新鲜相机帧时不能开始
录制；录制过程中相机画面超过一秒未更新时，当前 episode 会被丢弃，但 Pico 输入和
G1 控制继续运行。视频恢复后不会自动重新开始录制。

## 数据集目录

录制程序写出的是一个可编辑数据集，而不是单个包含所有内容的 HDF5：

```text
data/recordings/sim2real_hdf5/
├── schema.json
├── episodes.jsonl
├── data/
│   └── episode_000000.h5
└── videos/
    └── d435i_rgb/
        └── episode_000000.mp4
```

`schema.json` 定义数据集 FPS、`robot_type`、`hand_type`、`neck_type`，以及每个字段的
shape、dtype、名称和分组。硬件类型必须与当前运行配置一致。

`episodes.jsonl` 是可编辑的 episode 清单。每一行把一条 episode 映射到对应 HDF5
和 MP4，并保存任务描述。任务文本不会写入 HDF5 attribute，因此修改任务描述不需要
重写帧数据。

## 帧字段

每个 HDF5 只包含按帧对齐的数组：

| 字段 | Shape | 含义 |
|------|-------|------|
| `frame_index` | scalar | 相机/动作帧序号 |
| `timestamp` | scalar | 单调时钟时间戳，单位为秒 |
| `observation.state` | `(68,)` | G1 关节状态、基座方向/角速度和投影重力 |
| `observation.state.hand` | `(12,)`，可选 | 左右 LinkerHand 硬件关节回读 |
| `observation.state.neck` | `(2,)`，可选 | 以度为单位的 OpenNeck 舵机 yaw/pitch 回读 |
| `observation.mode` | scalar | `STANDING`、`MOCAP`、`ARMS` 或动捕暂停状态码 |
| `action` | `(36,)` | motion tracker 使用的根部姿态和 29 关节参考 |
| `action.hand` | `(12,)`，可选 | 启用手部控制时的左右 LinkerHand 目标 |
| `action.neck` | `(2,)`，可选 | 经过机械限位后的 OpenNeck yaw/pitch 角度 |

`observation.state` 的顺序为 `joint_pos(29)`、`joint_vel(29)`、
`base_quat_wxyz(4)`、`base_ang_vel(3)` 和 `projected_gravity(3)`。
`observation.state.hand` 使用 LinkerHand SDK 的 0-255 关节数值，顺序是左手六个
通道，然后是右手六个通道。`observation.state.neck` 是 OpenNeck `read_deg()`
返回的 `[yaw_deg, pitch_deg]`。
`observation.mode` 使用 `standing=0`、`mocap=1`、`arms=2` 和 `pause=3`。
`action` 的结构是 `root_pos(3) + root_quat_wxyz(4) + reference_joint_pos(29)`。

相机 RGB 只保存在 MP4 sidecar 中，HDF5 不重复保存 raw image。只有启用对应硬件时，
才会出现可选 state 和 action 字段。

## 提交与恢复规则

录制器会先提交 HDF5 和视频文件，再向清单追加记录。进程中断后，未提交的 episode
会在下次录制进程启动时删除，也不会占用 episode 序号。已有 `schema.json` 与当前
配置不兼容时，只会停止非关键的录制进程，G1 控制会继续运行。

## 检查录制数据

```bash
python scripts/view/view_recording.py \
    --recording data/recordings/sim2real_hdf5
```

播放前，查看器会检查清单路径、HDF5 shape、dtype、有限值和 MP4 对齐。录制数据
不包含实测根部 XYZ，因此实测机器人会锚定到参考根部位置；这个格式无法评估全局
根部平移。
