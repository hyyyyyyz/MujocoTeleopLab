---
sidebar_position: 1
---

# 安装 Teleopit

只安装你真正需要的部分。下面的命令都在仓库根目录执行，并要求 Python 3.10
或更高版本。

## 1. 获取代码

```bash
git clone https://github.com/BotRunner64/Teleopit.git
cd Teleopit
```

只有连接真实 G1 或使用 LinkerHand 时才需要 Git 子模块，相关命令放在本页后面。

## 2. 创建 Python 环境

下面三种方式任选一种，不要全部执行。

### uv

```bash
uv venv --python 3.10
source .venv/bin/activate
```

本页后续出现 `pip install` 时，也可以替换为 `uv pip install`。

### pip 和 venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Conda

```bash
conda create -n teleopit python=3.10
conda activate teleopit
```

Conda 负责创建环境；进入环境后，仍使用 `pip install` 安装 Teleopit。

## 3. 根据目标安装依赖

每个 extra 都包含 Teleopit 基础包。先安装与你当前目标对应的一项；以后可以在同一个
环境中继续安装其他 extra。

| 目标 | 安装命令 | 增加的内容 |
|------|----------|------------|
| 在 MuJoCo 中运行运控 | `pip install -e .` | 基础推理、GMR、MuJoCo 和 ONNX Runtime |
| 在仿真或 G1 上使用 Pico | `pip install -e '.[pico4]'` | Pico 接收与真机运行环境 |
| 不使用 Pico，在真实 G1 上回放 BVH | `pip install -e '.[sim2real]'` | G1 运行环境和 OpenCV |
| 训练运控策略 | `pip install -e '.[train]'` | mjlab、RSL-RL 和实验记录工具 |
| 录制 Pico 真机数据 | `pip install -e '.[recording]'` | Pico 运行环境和 MP4 写入依赖 |
| 查看已录制的数据 | `pip install -e '.[review]'` | OpenCV 和 MuJoCo/Viser 查看工具 |
| 使用 OpenNeck | `pip install -e '.[openneck]'` | Pico 运行环境和 OpenNeck 驱动 |
| 运行测试 | `pip install -e '.[dev]'` | pytest 和覆盖率工具 |

## 4. 下载对应资源

Python 包里不包含机器人 mesh、运控模型和动作数据。先安装默认的 ModelScope
下载工具：

```bash
pip install modelscope
```

再根据目标下载：

| 目标 | 命令 |
|------|------|
| 仿真、Pico VR 或 G1 推理 | `python scripts/setup/download_assets.py --only robots gmr ckpt bvh` |
| 使用已发布数据集训练 | `python scripts/setup/download_assets.py --only robots data` |
| 下载全部资源 | `python scripts/setup/download_assets.py` |

需要从 HuggingFace 下载时：

```bash
python scripts/setup/download_assets.py \
    --source huggingface \
    --only robots gmr ckpt bvh
```

推理资源包会把 `track_g1` 和 `track_g1_neck_o6` 两组 ONNX/checkpoint 放到 `ckpt/`，
并把 G1 模型文件、GMR 文件和示例 BVH 放到代码默认查找的位置。完整文件清单和资源分组
见[资产](../reference/resources/assets)。

## 5. 连接真实 G1 前的额外安装

在实际运行 Teleopit 的电脑上编译 C++ DDS bridge：

```bash
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

无论使用 Pico 还是真机 BVH 回放，都需要这个 bridge。如果编译失败或收不到机器人
状态，请查看[配套项目](../reference/companion-projects#g1-bridge-sdk)。

## 6. 可选硬件

### LinkerHand L6 或 O6

只有设置 `hands.enabled=true` 时才需要安装：

```bash
git submodule update --init --recursive
pip install -e third_party/linkerhand-python-sdk
pip install -e third_party/somehand
bash scripts/setup/download_somehand_assets.sh
```

### OpenNeck

`openneck` extra 已经包含 Pico 依赖。启用前先完成标定：

```bash
pip install -e '.[openneck]'
openneck calibrate
```

Teleopit 使用 OpenNeck 的角度接口，不支持旧版归一化标定字段。

### RealSense 录制或视频预览

启用 RealSense 时还需要单独安装 `pyrealsense2`。Arm 设备建议使用
conda-forge：

```bash
conda install -c conda-forge pyrealsense2
```

Pico 身体追踪本身不依赖 RealSense。

## 7. 检查安装结果

先检查 Teleopit 基础包：

```bash
python -c "import teleopit; print('teleopit OK')"
```

如果安装了 Pico 或训练依赖，再运行对应检查：

```bash
python -c "from pico_bridge import PicoBridge; print('Pico OK')"
python -c "import train_mimic.tasks; print('training OK')"
```

如果安装的是推理环境，并已经下载 `robots gmr ckpt bvh` 资源，最后运行一次示例仿真：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

MuJoCo 窗口能够打开，仿真 G1 能跟随示例动作，就说明安装完成。关闭窗口即可停止，
然后进入四条任务教程之一。
