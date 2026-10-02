FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

RUN git clone --single-branch --branch main \
    https://github.com/meituan-longcat/LongCat-Video .

# ВАЖНО: setup.py (install_requirements helper), download_weights.py,
# quantization.py и patch_audio_condition.py НЕ входят в официальный
# репозиторий meituan-longcat/LongCat-Video — это решения из сторонней,
# независимо провалидированной сборки под RunPod/A40
# (github.com/pjain-github/longcat-video), которая реально столкнулась
# и справилась с проблемами апстрима:
#   - SIGKILL/exit -9 при наивной INT8-загрузке (стандартный загрузчик
#     сперва собирает модель в полной точности и не помещается в 48 ГБ
#     VRAM) — quantization.py использует meta device и загружает
#     safetensors-шарды по одному, без полновесной промежуточной копии
#   - баг в официальном run_demo_avatar_single_audio_to_video.py,
#     требующий патча (patch_audio_condition.py)
# Подключаем эти файлы напрямую из их репозитория на этапе сборки —
# привязано к ветке main, а не к конкретному commit SHA; для
# воспроизводимости в будущем стоит зафиксировать точный коммит, если
# автор внесёт несовместимые изменения.
RUN wget -O /tmp/longcat-setup.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/setup.py && \
    wget -O /tmp/patch_audio_condition.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/docker/patch_audio_condition.py && \
    wget -O /opt/LongCat-Video/download_weights.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/download_weights.py && \
    mkdir -p /opt/LongCat-Video/longcat_video/modules && \
    wget -O /opt/LongCat-Video/longcat_video/modules/quantization.py \
      https://raw.githubusercontent.com/pjain-github/longcat-video/main/docker/quantization.py

RUN python3 /tmp/patch_audio_condition.py run_demo_avatar_single_audio_to_video.py

RUN python3.10 -m venv .venv \
    && .venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && .venv/bin/python /tmp/longcat-setup.py install_requirements \
       --avatar --use-system-cuda --project-dir /opt/LongCat-Video \
    && .venv/bin/python -m pip install onnxruntime-gpu accelerate runpod mutagen

# Веса (~42 ГБ: ~21 ГБ Avatar-1.5 + ~22 ГБ базовой LongCat-Video) НЕ
# запекаются в образ — живут на постоянном Network Volume,
# подключаемом к эндпоинту в /workspace (см. README деплоя). Закачка —
# отдельный разовый шаг через обычный Pod на том же Volume, не внутри
# serverless-воркера (иначе первый же холодный старт упрётся в таймаут
# на скачивании 42 ГБ).
ENV WEIGHTS_DIR=/workspace/weights

COPY handler.py /opt/LongCat-Video/handler.py

CMD ["/opt/LongCat-Video/.venv/bin/python", "-u", "/opt/LongCat-Video/handler.py"]
