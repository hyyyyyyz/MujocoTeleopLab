---
sidebar_position: 1
---

# 资产

Teleopit 的 Git 仓库只保存代码，不保存大型机器人 mesh、运控模型和动作数据。
[安装说明](../../getting-started/installation)给出了每种用户场景最短的下载命令；本页提供
完整文件清单和维护者说明。

## 不入库的内容

- `assets/robots/` — canonical 机器人 XML/mesh
- `teleopit/retargeting/gmr/assets/` — GMR 重定向资源、IK 配置和非 canonical 机器人描述
- `data/`、`ckpt/`、checkpoint、缓存等生成产物
- 演示媒体（`assets/demo.gif`、`assets/demo.mp4`）

## 资源清单

| 资源组 | 下载后的路径 | 用途 |
|--------|--------------|------|
| `ckpt` | `ckpt/track_g1.{onnx,pt}`、`ckpt/track_g1_neck_o6.{onnx,pt}` | 可直接运行的推理模型和对应 PyTorch checkpoint |
| `robots` | `assets/robots/` 下的机器人 XML 变体与 mesh | 训练、MuJoCo 推理、GMR 和数据集 FK |
| `gmr` | `teleopit/retargeting/gmr/assets/` | 动作重定向模型和 IK 配置 |
| `bvh` | `data/sample_bvh/*.bvh` | 安装检查和仿真教程使用的示例动作 |
| `data` | `data/datasets/<dataset>/shard_*.h5` | 用于分发的精简动作数据；训练前需要预计算 |

当前 G1 机器人资源包包括：

| 模型 XML | 配置 |
|----------|------|
| `assets/robots/unitree_g1/g1_29dof.xml` | 基础 G1 模型，也是默认值 |
| `assets/robots/unitree_g1/g1_29dof_dex3.xml` | 带 Dex3 手部几何和惯性参数的 G1 |
| `assets/robots/unitree_g1/g1_29dof_neck_o6.xml` | 带颈部主动视觉和 O6 手部模型的 G1 |

默认值不是模型白名单。训练可以通过 `--robot_xml` 选择其他与任务兼容的 XML。GMR
资源目录中的 XML 属于对应的重定向配置，与运行时机器人资源包是两套不同资源。基础
模型使用 `track_g1` 模型对，颈部加 O6 模型使用 `track_g1_neck_o6` 模型对。

## 远程仓库

### ModelScope（默认下载源）

| 仓库 | 类型 | 内容 |
|------|------|------|
| `BingqianWu/Teleopit-models` | model | checkpoint、GMR retargeting 资源、示例 BVH |
| `BingqianWu/Teleopit-datasets` | dataset | 训练/验证数据集 |

### HuggingFace（备选）

| 仓库 | 类型 | 内容 |
|------|------|------|
| `12e21/Teleopit-models` | model | checkpoint、GMR retargeting 资源、示例 BVH |
| `12e21/Teleopit-datasets` | dataset | 训练/验证数据集 |

### 资源组与仓库对应关系

| 组 | 仓库 | 远端路径 |
|----|------|---------|
| `ckpt` | Teleopit-models | `checkpoints/track_g1.{onnx,pt}`、`checkpoints/track_g1_neck_o6.{onnx,pt}` |
| `robots` | Teleopit-models | `archives/robot_assets.tar.gz` |
| `gmr` | Teleopit-models | `archives/gmr_assets.tar.gz` |
| `bvh` | Teleopit-models | `archives/sample_bvh.tar.gz` |
| `data` | Teleopit-datasets | `data/datasets/*/*.h5`（`lafan1`、`pico_record`、`seed`、`twist2`） |

## 下载行为

使用项目自带的下载脚本（默认从 ModelScope 下载）：

```bash
# 下载全部
python scripts/setup/download_assets.py

# 只下载推理必需的资源
python scripts/setup/download_assets.py --only robots gmr ckpt bvh

# 只下载训练数据
python scripts/setup/download_assets.py --only data

# 从 HuggingFace 下载
python scripts/setup/download_assets.py --source huggingface
```

下载后各资源的本地落点：

| 远端路径 | 本地路径 |
|---------|---------|
| `checkpoints/track_g1.onnx` | `ckpt/track_g1.onnx` |
| `checkpoints/track_g1.pt` | `ckpt/track_g1.pt` |
| `checkpoints/track_g1_neck_o6.onnx` | `ckpt/track_g1_neck_o6.onnx` |
| `checkpoints/track_g1_neck_o6.pt` | `ckpt/track_g1_neck_o6.pt` |
| `archives/robot_assets.tar.gz` | `assets/robots/`（自动解压） |
| `archives/gmr_assets.tar.gz` | `teleopit/retargeting/gmr/assets/`（自动解压） |
| `archives/sample_bvh.tar.gz` | `data/sample_bvh/`（自动解压） |
| `data/datasets/*/*.h5` | `data/datasets/` |

## 上传到 ModelScope

### 第一步：准备上传目录

```bash
python scripts/setup/prepare_modelscope_assets.py --only ckpt robots gmr bvh --clean
python scripts/setup/prepare_modelscope_assets.py --only data
```

产物在 `data/modelscope_upload/`。

### 第二步：上传到对应仓库

```bash
# 模型仓库
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/checkpoints checkpoints --sync
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/archives archives

# 数据集仓库
modelscope upload --repo-type dataset BingqianWu/Teleopit-datasets \
    data/modelscope_upload/data data
```

checkpoint 上传有意使用 `--sync`。它只会在远端 `checkpoints/` 目录内删除本地不存在的
旧模型名，不会影响 `archives/`。除非本地 staging 中包含所有需要保留的远端归档，否则
不要给归档上传命令添加 `--sync`。

### 第三步：打版本 tag

ModelScope 仅模型仓库支持 tag，数据集仓库不支持。

```bash
python - <<'EOF'
from modelscope.hub.api import HubApi
api = HubApi()
url = api.create_model_tag("BingqianWu/Teleopit-models", "vX.Y.Z")
print(url)
EOF
```

tag 与代码仓库的 Git tag 保持一致，方便追溯每个版本对应的模型。

## 上传到 HuggingFace

### 第一步：准备 staging 目录

```bash
# 准备全部（--clean 会清空旧的 staging 目录）
python scripts/setup/upload_hf_assets.py --dry-run --clean

# 只准备指定组
python scripts/setup/upload_hf_assets.py --only ckpt robots gmr bvh --dry-run
python scripts/setup/upload_hf_assets.py --only data --dry-run
```

`--dry-run` 只写 staging，不执行上传，可用于检查文件完整性。

### 第二步：执行上传

```bash
python scripts/setup/upload_hf_assets.py --only ckpt robots gmr bvh --clean
python scripts/setup/upload_hf_assets.py --only data --clean
```

:::warning
每次运行前建议加 `--clean`，否则 staging 目录可能残留上次遗留的文件，导致 `--only` 语义失效（旧资源被误带入上传）。
:::

### 第三步：打版本 tag

HuggingFace 模型仓库支持 tag（数据集仓库不支持）：

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("12e21/Teleopit-models", tag="vX.Y.Z", repo_type="model")
EOF
```

tag 与代码仓库的 Git tag 保持一致，方便追溯每个版本对应的模型。

## 提交前检查

推送代码前运行：

```bash
python scripts/dev/check_large_tracked_files.py
```

会拦截大二进制文件并检查已跟踪文件的体积上限。
