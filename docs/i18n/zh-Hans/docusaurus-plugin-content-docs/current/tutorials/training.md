---
sidebar_position: 4
---

# 训练运控策略

本教程从已下载的动作数据开始，最终得到可以在 Teleopit 仿真和真实 G1 上运行的
ONNX 运控模型。

常规训练流程默认使用 NVIDIA GPU。动作数据会在启动时全部加载到内存，因此合并数据集
越大，需要的内存和显存也越多。

## 开始之前

按照[安装说明](../getting-started/installation)完成：

- `train` 依赖；
- `robots data` 资源包。

检查训练包：

```bash
python -c "import train_mimic.tasks; print('training OK')"
```

## 1. 预处理已下载的数据集

下载的数据是便于分发的精简版本。训练需要另一个目录，其中提前计算好了关节速度和
身体运动学信息：

```bash
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets \
    --outdir data/datasets_precomputed \
    --jobs 8
```

下面所有训练、回放和 benchmark 命令都使用 `data/datasets_precomputed`。把原始
`data/datasets` 直接传给训练会报错，这不是支持的快捷方式。

自定义 BVH、PKL、NPZ 或 Pico 录制数据的处理方法见
[动作数据集](../reference/resources/motion-datasets)。

## 2. 选择机器人模型

使用 `--robot_xml` 指定训练所用的 MuJoCo 模型。如果省略该参数，默认使用：

```text
assets/robots/unitree_g1/g1_29dof.xml
```

当前 `robots` 资源包提供了以下可以直接使用的示例：

| 模型 XML | 配置 |
|----------|------|
| `assets/robots/unitree_g1/g1_29dof.xml` | 基础 G1 模型，也是默认值 |
| `assets/robots/unitree_g1/g1_29dof_dex3.xml` | 带 Dex3 手部几何和惯性参数的 G1 |
| `assets/robots/unitree_g1/g1_29dof_neck_o6.xml` | 带颈部主动视觉和 O6 手部模型的 G1 |

这个表只是当前资源包随附的模型示例，不是写死的模型白名单。只要关节和刚体定义与所选
训练任务配置及数据集兼容，也可以传入其他模型 XML。

下面“开始完整训练”的主命令会显式选择基础模型，便于直接复制。训练其他模型时，
替换 `--robot_xml` 后的路径即可。其他命令不再重复该参数；回放和 benchmark 会从
所选任务配置中加载机器人。

## 3. 先做短时间冒烟测试

开始长时间训练前，先确认数据集、仿真器和日志工具能够一起工作：

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets_precomputed
```

只要环境能够持续 step、终端输出 loss，并且
`logs/rsl_rl/g1_general_tracking/` 下生成新的运行目录，这项检查就通过了。

## 4. 开始完整训练

```bash
python train_mimic/scripts/train.py \
    --robot_xml assets/robots/unitree_g1/g1_29dof.xml \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

显存不足时降低 `--num_envs`。默认日志工具是 TensorBoard；需要时可使用
`--logger wandb` 或 `--logger swanlab`。

`--max_iterations` 表示继续训练多少次。例如从 `model_12000.pt` 恢复并设置
`--max_iterations 18000`，最终会训练到第 30000 次。

## 5. 在仿真中查看 checkpoint

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed
```

回放会从每段动作开头开始，并关闭训练噪声。导出前先用它排除明显不稳定的模型。

## 6. 运行 Benchmark

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed \
    --num_envs 32
```

Benchmark 会对每个长度足够的 clip 执行一次确定性的 10 秒 rollout，并报告：

- 平均关节位置误差（`MPJPE`）；
- 根部位置、旋转和速度误差；
- rollout 成功率。

结果会保存为文本摘要、JSON、逐 clip CSV 和逐 rollout CSV。

## 7. 导出 ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output ckpt/track_g1.onnx \
    --history_length 10
```

输出必须是包含 `obs` 和 `obs_history` 的双输入 TemporalCNN。Teleopit 会在启动时
检查 167D 观测签名，不兼容的导出文件会直接报错。

使用正常运行入口检查导出结果：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

## 扩展到多张 GPU

单机多卡：

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

这里的 `--num_envs` 是每张 GPU 的环境数量。

多机训练使用 `torchrun`：

```bash
torchrun \
    --nnodes=$PET_NNODES \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$PET_MASTER_ADDR \
    --master_port=$PET_MASTER_PORT \
    train_mimic/scripts/train.py \
    --num_envs 1024 \
    --max_iterations 1000 \
    --motion_file data/datasets_precomputed
```

这里的 `--num_envs` 是每个进程的环境数量，总数会随 world size 增长。

## 常见问题

| 现象 | 检查内容 |
|------|----------|
| Loader 提示数据集是 minimal 格式 | 运行 `precompute_dataset.py`，并使用它的输出目录 |
| 显存不足 | 降低 `--num_envs`，或使用更少的预计算数据 shard |
| 启动加载时内存不足 | 减少参与训练的 precomputed shard，或增加内存 |
| 训练速度异常缓慢 | 检查 PyTorch 是否识别 CUDA，并确认训练设备实际使用 CUDA GPU |

任务内部结构和模型维度见[系统架构](../reference/architecture)。
