import os
import json
import glob
import time
import math
import base64
import shutil
import subprocess
import tempfile
import uuid

from mutagen import File as MutagenFile
import runpod

REPO_DIR = "/opt/LongCat-Video"
VENV_DIR = os.path.join(REPO_DIR, ".venv")
VENV_PY = os.path.join(VENV_DIR, "bin", "python")
VENV_TORCHRUN = os.path.join(VENV_DIR, "bin", "torchrun")

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/workspace/weights")
AVATAR_CKPT_DIR = os.path.join(WEIGHTS_DIR, "LongCat-Video-Avatar-1.5")
TMP_ROOT = "/workspace/tmp"

# Из README провалидированной сборки: один сегмент — 93 кадра при
# 25 fps ≈ 3.72 сек. --num_segments задаётся вручную (сам скрипт не
# подстраивает длину под длительность аудио автоматически) — считаем
# нужное число сегментов от реальной длительности входного аудио,
# чтобы не обрезать речь и не генерировать лишнее.
SECONDS_PER_SEGMENT = 3.72

# LongCat не использует отдельные pose-шаблоны, как EchoMimicV2 —
# стиль подачи (мимика/интонация) задаётся естественным текстовым
# промптом, который модель интерпретирует сама. Это тот же список
# эмоций, что и в UI Кейса 1; для Кейса 2 (автоопределение по аудио)
# бэкенд передаёт то же текстовое значение, что вернула модель
# распознавания эмоций.
EMOTION_TO_PROMPT_HINT = {
    "neutral": "speaking calmly and naturally to the camera",
    "happy": "speaking cheerfully with a warm, genuine smile",
    "sad": "speaking gently with a subdued, thoughtful expression",
    "angry": "speaking firmly with a serious, intense expression",
    "surprised": "speaking with an animated, surprised expression",
}
DEFAULT_PROMPT_HINT = EMOTION_TO_PROMPT_HINT["neutral"]


def ensure_weights_present():
    """Веса (~42 ГБ) должны уже лежать на подключённом Network Volume —
    закачиваются ОДИН РАЗ вручную через обычный Pod на этом же Volume
    (см. инструкцию деплоя), а не на холодном старте воркера. Если их
    нет — explicit ошибка с понятной подсказкой, а не попытка скачать
    42 ГБ посреди обработки запроса (упёрлось бы в execution timeout)."""
    marker = os.path.join(AVATAR_CKPT_DIR, "config.json")
    if not os.path.exists(marker):
        raise RuntimeError(
            f"Веса LongCat не найдены в {AVATAR_CKPT_DIR}. Запусти download_weights.py "
            "один раз через обычный Pod на этом же Network Volume перед первым запросом "
            "к serverless-эндпоинту — см. инструкцию деплоя."
        )


def get_audio_duration_seconds(audio_path: str) -> float:
    media = MutagenFile(audio_path)
    if media is None or media.info is None or not hasattr(media.info, "length"):
        return 15.0  # запасное значение, если не удалось определить длительность
    return float(media.info.length)


def build_prompt(emotion) -> str:
    hint = EMOTION_TO_PROMPT_HINT.get((emotion or "").lower(), DEFAULT_PROMPT_HINT)
    return f"A person {hint}, in a realistic, professional setting."


def run_longcat_inference(image_path: str, audio_path: str, emotion) -> tuple[str, str]:
    os.makedirs(TMP_ROOT, exist_ok=True)
    job_dir = tempfile.mkdtemp(prefix="longcat_job_", dir=TMP_ROOT)
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    audio_duration = get_audio_duration_seconds(audio_path)
    num_segments = max(1, math.ceil(audio_duration / SECONDS_PER_SEGMENT))

    input_json_path = os.path.join(job_dir, "input.json")
    input_data = {
        "prompt": build_prompt(emotion),
        "cond_image": image_path,
        "cond_audio": {"person1": audio_path},
    }
    # Промпт обязан быть в одну строку (см. troubleshooting в README
    # провалидированной сборки — многострочный промпт даёт
    # JSONDecodeError) — build_prompt() и так не вставляет переводы
    # строк, но на всякий случай подчищаем.
    input_data["prompt"] = input_data["prompt"].replace("\n", " ").strip()
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False)

    cmd = [
        VENV_TORCHRUN,
        "--nproc_per_node=1",
        "run_demo_avatar_single_audio_to_video.py",
        "--input_json", input_json_path,
        "--output_dir", output_dir,
        "--resolution", "480p",
        "--num_segments", str(num_segments),
        "--stage_1", "ai2v",
        "--checkpoint_dir", AVATAR_CKPT_DIR,
        "--model_type", "avatar-v1.5",
        "--use_int8",
        "--use_distill",
        # 0.70 — задокументированный в провалидированной сборке баланс
        # "естественно, но спокойнее" для 48 ГБ карт; дистиллированный
        # режим всё равно фиксирует text/audio guidance на 1.0 — эта
        # настройка снижает именно вес аудио-эмбеддинга, не скорость.
        "--audio_condition_scale", "0.70",
    ]

    start_time = time.time()
    result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LongCat-Video-Avatar inference упал: {result.stderr[-3000:]}")

    candidates = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
    candidates = [c for c in candidates if os.path.getmtime(c) >= start_time]
    if not candidates:
        raise RuntimeError("LongCat не создал видео — mp4-файл не найден после инференса")
    output_path = max(candidates, key=os.path.getmtime)
    return output_path, job_dir


def handler(event):
    inp = event.get("input", {})
    image_b64 = inp.get("image_base64")
    audio_b64 = inp.get("audio_base64")
    emotion = inp.get("emotion")  # None/отсутствует => нейтральный дефолт

    if not image_b64 or not audio_b64:
        return {"error": "image_base64 и audio_base64 обязательны"}

    try:
        ensure_weights_present()
    except Exception as e:
        return {"error": str(e)}

    os.makedirs(TMP_ROOT, exist_ok=True)
    request_id = uuid.uuid4().hex
    image_path = f"{TMP_ROOT}/{request_id}_ref.png"
    audio_path = f"{TMP_ROOT}/{request_id}_audio.wav"

    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    job_dir = None
    try:
        output_path, job_dir = run_longcat_inference(image_path, audio_path, emotion)
        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"video_base64": video_b64}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in (image_path, audio_path):
            if os.path.exists(p):
                os.remove(p)
        if job_dir and os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
