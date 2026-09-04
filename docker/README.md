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

Compose 构建配置声明使用 host 网络；如果 Docker Compose/Buildx 版本仍未将
该设置传递给构建器，可使用下面的等价命令：

```bash
DOCKER_BUILDKIT=1 docker build --network=host \
  -t mujoco-teleoplab:cuda -f docker/Dockerfile .
```

Dockerfile 会在构建层内把 Ubuntu apt 源替换为服务器当前可达的固定 IP，并禁用
基础 CUDA 镜像中不需要的 NVIDIA apt 源，以绕过 BuildKit 的只读 DNS 挂载。

## 自动 VLA 数据生成

CuRobo 规划不依赖 PICO，可直接在服务器上运行：

```bash
# 先准备被 .gitignore 忽略的 G1/场景资产
docker compose -f docker/compose.yml run --rm mujoco-teleoplab \
  bash scripts/setup/setup_scene_teleop.sh

docker compose -f docker/compose.yml run --rm mujoco-teleoplab vla \
  --scene can --episodes 1 --output-dir outputs/vla_scene_data_curobo
```

重放并验收已采集的 episode（同时生成 JPEG 帧和 MP4）：

```bash
docker compose -f docker/compose.yml run --rm mujoco-teleoplab replay \
  --scene can \
  --episode outputs/vla_scene_data_curobo/episode_000000.npz \
  --render-dir outputs/vla_scene_replays/episode_000000
```

重放报告会写入 `replay_report.json`，包含状态最大误差、物体位姿误差、
物体位移和独立 `success` 判断。命令在 success 或状态重放不匹配时返回非零，
可直接用于批量筛选数据。

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

容器隔离的是 MuJoCo/WBC/CuRobo 运行时。PICO/XRoboToolkit bridge 也已经提供独立
的 Python 3.12 容器，因此不依赖 5090 宿主机的 Python、`.venv` 或 SDK 安装。
若只做自动 VLA 生成，则不需要启动 PICO 或 XRoboToolkit。

### 全 Docker bridge

使用独立的 `xrbridge` 容器：
它包含 Python 3.12、XRoboToolkit 原生 binding 和 PC Service；MuJoCo 场景仍在
CUDA/Python 3.10 容器中运行。两个服务使用 host 网络的 `127.0.0.1:17600` 通信：

```bash
PICO_VIDEO_HOST=10.0.90.191 \
DISPLAY="$DISPLAY" \
docker compose -f docker/compose.yml --profile teleop up --build \
  xrbridge scene-x11
```

通过 SSH X11 启动时，在远端 shell 中执行同一命令即可；确保远端
`$DISPLAY` 和 `$HOME/.Xauthority` 已由 `ssh -Y` 设置。PICO 仍需选择
`Head`、`Controller`、`Send`、`ZEDMINI`，最后点击 `Listen`。

首次构建 `mujoco-teleoplab:xrbridge` 会下载官方 PC Service（约 110 MB）并编译
原生 SDK，耗时比普通场景镜像更长。`scene-x11` 首次启动时还会在容器挂载的
`/workspace` 内自动创建 `.venv_scene` 并下载 `decoupled_wbc` 场景依赖；这些步骤
不使用宿主机 Python。批量 VLA 采集不需要启动 `scene-x11`，继续使用 EGL 模式即可。

## 常用命令

```bash
docker compose -f docker/compose.yml run --rm mujoco-teleoplab shell
docker compose -f docker/compose.yml run --rm mujoco-teleoplab python3.10 -m pytest -q
docker compose -f docker/compose.yml down
```
