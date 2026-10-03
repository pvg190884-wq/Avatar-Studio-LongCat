# Используем официальный образ RunPod с CUDA 12.4 и PyTorch
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libsndfile1 \
    wget \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/LongCat-Video

# Клонируем официальный репозиторий
RUN git clone --single-branch --branch main \
    https://github.com/meituan-longcat/LongCat-Video .

# Скачиваем вспомогательные скрипты из стороннего репозитория
RUN wget -O /tmp/longcat-setup.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/setup.py && \
    wget -O /tmp/patch_audio_condition.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/docker/patch_audio_condition.py && \
    wget -O /opt/LongCat-Video/download_weights.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/download_weights.py && \
    mkdir -p /opt/LongCat-Video/longcat_video/modules && \
    wget -O /opt/LongCat-Video/longcat_video/modules/quantization.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/docker/quantization.py

# Применяем патч
RUN python3 /tmp/patch_audio_condition.py run_demo_avatar_single_audio_to_video.py

# ДОБАВЛЕНО: путь к cuDNN 9 внутри venv (иначе import torch падает: libcudnn.so.9 not found)
ENV LD_LIBRARY_PATH=/opt/LongCat-Video/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# Создаем виртуальное окружение и устанавливаем зависимости
RUN python3.10 -m venv .venv \
    && .venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && .venv/bin/python -m pip install nvidia-cudnn-cu12 \
    && .venv/bin/python /tmp/longcat-setup.py install_requirements \
       --avatar --use-system-cuda --project-dir /opt/LongCat-Video \
    && .venv/bin/python -m pip install onnxruntime-gpu accelerate runpod mutagen

# Указываем путь к весам (они будут на Network Volume)
ENV WEIGHTS_DIR=/workspace/weights

# Копируем ваш handler.py
COPY handler.py /opt/LongCat-Video/handler.py

# Запускаем
CMD ["/opt/LongCat-Video/.venv/bin/python", "-u", "/opt/LongCat-Video/handler.py"]
