# CUDA-enabled runtime for MujocoTeleopLab on NVIDIA GPUs (including RTX 5090).
# The host NVIDIA driver is supplied by nvidia-container-toolkit; it is not
# installed in the image.
# NGC is used instead of Docker Hub because many secured GPU servers allow
# access to nvcr.io while registry-1.docker.io is rate-limited or blocked.
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUROBO_REVISION=a56e0f06db9efb99232586db18b18b323cb22c47
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX_CUDA=cu128
# CuRobo builds CUDA extensions during image construction.  BuildKit does not
# expose the host GPU to `docker build`, so PyTorch cannot infer an architecture
# from visible devices.  RTX 5090 is Blackwell compute capability 12.0.
ARG TORCH_CUDA_ARCH_LIST=12.0

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

RUN sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|http://185.125.190.83/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|http://91.189.92.24/ubuntu|g' \
      -e 's|^deb .*developer.download.nvidia.com.*|# disabled: CUDA is provided by the base image|' \
      /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update -o Acquire::http::Pipeline-Depth=0 && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip python3-venv \
      build-essential git curl ca-certificates pkg-config \
      libglfw3 libgl1 libegl1 libgles2 libosmesa6 libx11-6 libxext6 \
      libxrender1 libxrandr2 libxi6 libxxf86vm1 libsm6 libice6 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel uv

WORKDIR /opt

# CuRobo is pinned so a rebuilt image does not silently change planner APIs.
RUN git clone --filter=blob:none https://github.com/NVlabs/curobo.git /opt/curobo && \
    git -C /opt/curobo fetch --depth 1 origin "${CUROBO_REVISION}" && \
    git -C /opt/curobo checkout --detach "${CUROBO_REVISION}" && \
    python3.10 -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX_CUDA}" && \
    # Build a regular wheel: this pinned commit has no release tag and its
    # editable metadata path is incompatible with modern setuptools_scm.
    python3.10 -m pip wheel --no-cache-dir --no-build-isolation --no-deps \
      -w /tmp/curobo-wheel /opt/curobo && \
    python3.10 -m pip install --no-cache-dir /tmp/curobo-wheel/*.whl && \
    rm -rf /tmp/curobo-wheel

COPY pyproject.toml README.md LICENSE /opt/mujocoteleoplab/
COPY teleopit /opt/mujocoteleoplab/teleopit
COPY train_mimic /opt/mujocoteleoplab/train_mimic
WORKDIR /opt/mujocoteleoplab
RUN python3.10 -m pip install --no-cache-dir -e ".[dev]"

COPY docker/entrypoint.sh /usr/local/bin/mujoco-teleop-entrypoint
RUN chmod +x /usr/local/bin/mujoco-teleop-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/mujoco-teleop-entrypoint"]
CMD ["shell"]
