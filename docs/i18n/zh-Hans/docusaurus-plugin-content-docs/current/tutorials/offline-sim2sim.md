---
sidebar_position: 1
---

# 在仿真中运行运控

本教程让训练好的运控策略在 MuJoCo 中复现一段动作。在接入 VR 或真实机器人之前，
先用它确认两个最基本的问题：

- 运控模型能否正常加载，并让 G1 保持稳定？
- 重定向后的机器人动作是否与原始动作一致？

## 开始之前

按照[安装说明](../getting-started/installation)安装基础依赖，并下载
`robots gmr ckpt bvh` 资源包。

## 1. 运行示例动作

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

最重要的是 `sim2sim` 窗口：它显示的是运控策略和物理仿真共同产生的 G1 动作，而不是
单纯的运动学目标。

| 按键 | 作用 |
|------|------|
| `Space` 或 `P` | 暂停或继续 |
| `R` | 从第一帧重新播放 |
| `Q` | 停止 |

机器人能够保持稳定，并大致跟上动作的节奏和姿态，就说明运行正常。少量跟踪误差是正常
的；摔倒、关节不动或朝向明显错误则不是。

## 2. 对比三个视图

动作异常时，打开全部视图可以判断问题从哪一步开始：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=all
```

| 视图 | 显示内容 |
|------|----------|
| `mocap` | 从 BVH 中读取的人体骨架 |
| `retarget` | GMR 生成的 G1 运动学目标 |
| `sim2sim` | 经过运控推理和 MuJoCo 物理后的 G1 |

如果 `mocap` 就不对，先检查 BVH 格式；如果 `mocap` 正常但 `retarget` 不对，检查动作
重定向；如果只有 `sim2sim` 不对，检查运控模型和观测配置。

也可以只打开需要的视图：

```bash
# 只看物理仿真结果
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=sim2sim

# 不打开窗口，适合服务器或时序测试
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=none
```

关闭所有已打开的 Viewer 后，仿真会自动结束。

## 3. 使用自己的 BVH

LAFAN1 格式：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=/path/to/motion.bvh \
    input.bvh_format=lafan1
```

`hc_mocap` 格式：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=/path/to/motion.bvh \
    input.bvh_format=hc_mocap
```

Teleopit 不会猜测未知的骨架布局。一个文件即使是合法 BVH，也可能需要先写适配器才能
作为支持的格式使用。

## 4. 保存视频

需要可重复的视频结果而不是交互窗口时：

```bash
MUJOCO_GL=egl python scripts/render/render_sim.py \
    --bvh data/sample_bvh/aiming1_subject1.bvh \
    --policy ckpt/track_g1.onnx
```

`hc_mocap` 输入需要再加 `--format hc_mocap`。渲染脚本会输出同步的 `mocap`、
`retarget` 和 `sim2sim` 视频。

## 常用播放参数

```bash
# 动作结束后保持最后姿态
playback.pause_on_end=true

# 运行 300 个仿真 step；0 表示不限制
num_steps=300

# 即使不打开 Viewer，也按照真实时间运行
realtime=true
```

完整字段见[配置说明](../reference/configuration/overview)。
