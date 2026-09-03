# Docker GPU 环境

这里提供 MujocoTeleopLab 的隔离 CUDA 环境，面向带 NVIDIA GPU 的服务器，
例如 RTX 5090。容器使用 CUDA 12.8 用户态和主机 NVIDIA 驱动，通过 `nvidia-container-toolkit`
访问 GPU；驱动本身不打包进镜像。

## 主机准备

在服务器上确认驱动、Docker 和 NGC 网络访问已就绪：

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all --entrypoint nvidia-smi \
  nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04
```

如果最后一条失败，先安装/配置 `nvidia-container-toolkit` 或检查 NGC 网络，
不要继续构建项目镜像。
RTX 5090 的驱动版本应使用 NVIDIA 官方当前支持 Blackwell 的版本（建议 570 或更新）；CUDA 镜像的
用户态版本不要求与驱动版本完全相同，但驱动必须满足其最低兼容版本。

## 构建和检查

在仓库根目录执行：

```bash
docker compose -f docker/compose.yml build
docker compose -f docker/compose.yml run --rm mujoco-teleoplab check
```

构建时会固定安装 CuRobo revision 和 CUDA PyTorch（默认 CUDA 12.8 / PyTorch 2.7.1）。首次构建需要下载较大的 CUDA、
PyTorch 和 CuRobo 层，建议预留至少 20 GB 磁盘空间。

Compose 构建阶段使用 host 网络，以兼容部分 GPU 服务器上 Docker bridge
网络的 DNS 限制。

## 自动 VLA 数据生成

CuRobo 规划不依赖 PICO，可直接在服务器上运行：

```bash
# 先准备被 .gitignore 忽略的 G1/场景资产
docker compose -f docker/compose.yml run --rm mujoco-teleoplab \
  bash scripts/setup/setup_scene_teleop.sh

docker compose -f docker/compose.yml run --rm mujoco-teleoplab vla \
  --scene can --episodes 1 --output-dir outputs/vla_scene_data_curobo
```

容器会把当前仓库挂载到 `/workspace`，所以生成的数据和下载的资产仍保存在主机
目录中，并继续遵守 Git 忽略规则。

## MuJoCo 场景窗口

默认 `MUJOCO_GL=egl`，适合服务器上的离屏渲染和 VLA 采集。若服务器有桌面会话，
可以尝试把它改为 GLFW，并允许当前 X 会话访问 Docker：

```bash
xhost +local:docker
MUJOCO_GL=glfw docker compose -f docker/compose.yml run --rm mujoco-teleoplab \
  scene --scene cube --no-bridge --no-video
```

使用完请收回授权：`xhost -local:docker`。无桌面环境不要使用 `--no-headless` 或
依赖 GLFW 的窗口；VLA 生成应保持 EGL。

## PICO/XRoboToolkit 说明

容器隔离的是 MuJoCo/WBC/CuRobo 运行时。XRoboToolkit PC Service 和其 Python SDK
通常绑定主机桌面、USB/网络设备，建议继续在主机的 Python 3.12 `.venv` 运行
`scripts/run/run_scene_xr_bridge.py`，再让容器里的 scene 服务通过主机网络接收
UDP。若只做自动 VLA 生成，则不需要启动 PICO 或 XRoboToolkit。

## 常用命令

```bash
docker compose -f docker/compose.yml run --rm mujoco-teleoplab shell
docker compose -f docker/compose.yml run --rm mujoco-teleoplab python3.10 -m pytest -q
docker compose -f docker/compose.yml down
```
