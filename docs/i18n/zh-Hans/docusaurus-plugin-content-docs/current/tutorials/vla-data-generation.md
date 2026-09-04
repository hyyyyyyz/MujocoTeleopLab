---
sidebar_position: 6
---

# 自动生成 VLA 数据

Teleopit 提供了一个 43-DOF 桌面场景自动数据生成流程。它只运行仿真，不需要 PICO、
XR bridge 或真实机器人。默认规划器是 CuRobo `MotionGen`：它从 MuJoCo 桌面和地面构建
碰撞世界，在关节空间规划接近/抓取/抬升/放置分段，再通过 43-DOF PD 回路执行，并记录
MuJoCo 相机帧、状态、目标动作、物体位姿和语言指令。

CuRobo 需要 CUDA 版 PyTorch 环境。缺少 CuRobo 时命令会明确失败；只有数据管线冒烟测试
才允许使用 `--planner scripted`，不能把它当作碰撞规划训练数据。

在 CUDA 工作站上可运行 `bash scripts/setup/setup_curobo.sh` 安装固定版本的 CuRobo。
默认 `.venv_scene` 环境在未安装 CuRobo 时仍可用于其他场景功能。

先安装并生成目标物体场景，再生成多个 episode：

```bash
.venv/bin/python scripts/setup/download_scene_object.py can
.venv_scene/bin/python scripts/setup/build_scene_with_object.py can
.venv_scene/bin/python scripts/run/generate_vla_scene_data.py \
  --scene can --planner curobo --episodes 10 --output-dir outputs/vla_scene_data
```

被忽略的输出目录包含 `schema.json`、`episodes.jsonl`、每次尝试的压缩 NPZ、图像目录和 MP4
诊断视频。`success` 必须同时满足：真实手指接触、物体抬升至少 3 cm、水平搬运至少 10 cm，
并在释放后回到桌面附近。失败尝试会保留用于排查，但训练前必须过滤掉失败 episode。

如果希望得到固定数量的有效数据，同时保留失败尝试，可以设置更大的尝试预算：

```bash
.venv_scene/bin/python scripts/run/generate_vla_scene_data.py \
  --scene cube --planner curobo \
  --successful-episodes 100 --episodes 140 \
  --output-dir outputs/vla_scene_data_curobo
```

如果在尝试预算内没有达到目标数量，命令会以状态码 2 退出。

规划器采用可替换接口。`CuroboSceneTrajectoryPlanner` 是生产后端；`ScriptedPickPlacePlanner`
仅用于管线测试。
