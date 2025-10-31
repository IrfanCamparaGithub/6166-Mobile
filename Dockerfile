FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MEDIAPIPE_DISABLE_GPU=1 \
    GLOG_minloglevel=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git ninja-build cmake build-essential \
    ffmpeg libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir fastapi uvicorn[standard] python-multipart && \
    python3 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      torch==2.0.0+cu118 torchvision==0.15.1+cu118

COPY third_party /tmp/third_party
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" FORCE_CUDA=1
RUN python3 -m pip install --no-cache-dir \
      /tmp/third_party/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

ENV XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache \
    CUDA_CACHE_PATH=/workspace/.cache \
    MPLCONFIGDIR=/workspace/.cache/mplcache \
    OUTPUT_DIR=/workspace/.cache/output
RUN mkdir -p /workspace/.cache/mplcache /workspace/.cache/output && \
    chmod -R 777 /workspace/.cache

COPY . .


CMD ["bash","-lc","python3 -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
