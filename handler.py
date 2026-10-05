import os
import re
import json
import glob
import time
import math
import base64
import shutil
import subprocess
import tempfile
import textwrap
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

# --- Режим экономии видеопамяти (чтобы генерация помещалась в 48 ГБ) ----
# LOW_VRAM=1 (по умолчанию): перед запуском демо-скрипта handler делает его
# копию с тремя безопасными правками (см. build_lowvram_script_text).
# LOW_VRAM=0 — запуск оригинального скрипта без правок.
LOW_VRAM = os.environ.get("LOW_VRAM", "1") == "1"
# Число GPU на воркере. 1 — обычный режим. 2 — запасной вариант из README
# LongCat (torchrun на 2 GPU + context parallel): для него в настройках
# эндпоинта нужно поставить GPU count = 2 и переменную NUM_GPUS=2.
NUM_GPUS = max(1, int(os.environ.get("NUM_GPUS", "1")))

BASE_DEMO_SCRIPT = "run_demo_avatar_single_audio_to_video.py"
LOWVRAM_DEMO_SCRIPT = "run_demo_avatar_single_audio_to_video_lowvram.py"

# Файлы, которые обязаны присутствовать в LongCat-Video-Avatar-1.5.
# Список составлен по реальной структуре папки на Network Volume:
# config.json лежит НЕ в корне, а в подпапках base_model_int8/ и
# whisper-large-v3/, поэтому проверка корневого config.json всегда
# падала с "Веса не найдены", даже когда все веса были на месте.
REQUIRED_WEIGHT_FILES = [
    os.path.join("base_model_int8", "config.json"),
    os.path.join("base_model_int8", "quantization_config.json"),
    os.path.join("base_model_int8", "quantized_model.safetensors.index.json"),
    os.path.join("lora", "dmd_lora.safetensors"),
    os.path.join("whisper-large-v3", "model.safetensors"),
    os.path.join("vocal_separator", "Kim_Vocal_2.onnx"),
]

# Параметры сегментов из демо-скрипта LongCat (avatar-v1.5): 93 кадра на
# сегмент, 13 кадров перекрытия, 25 fps. Первый сегмент даёт 93/25 = 3.72 с
# видео, каждый следующий — только (93-13)/25 = 3.2 с нового видео.
# Скрипт сам не подстраивает число сегментов под длительность аудио —
# --num_segments задаётся вручную, поэтому считаем его по этим цифрам.
FRAMES_PER_SEGMENT = 93
COND_FRAMES = 13
SAVE_FPS = 25
FIRST_SEGMENT_SECONDS = FRAMES_PER_SEGMENT / SAVE_FPS                  # 3.72
NEXT_SEGMENT_SECONDS = (FRAMES_PER_SEGMENT - COND_FRAMES) / SAVE_FPS   # 3.2

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


def segments_for_duration(audio_duration: float) -> int:
    """Сколько сегментов нужно, чтобы видео не короче аудио.

    Раньше было ceil(длительность / 3.72) — это неверно: после первого
    сегмента каждый следующий добавляет только 3.2 с, поэтому для части
    длительностей (например 11 или 14 с) конец речи обрезался."""
    if audio_duration <= FIRST_SEGMENT_SECONDS:
        return 1
    return 1 + math.ceil((audio_duration - FIRST_SEGMENT_SECONDS) / NEXT_SEGMENT_SECONDS)


# ---------------------------------------------------------------------------
# Правки демо-скрипта LongCat для экономии видеопамяти
# ---------------------------------------------------------------------------
# Что в штатном скрипте держится на GPU постоянно, хотя нужно на секунды:
#   * text encoder UMT5 (~11 ГБ в bf16) — нужен только на время кодирования
#     промпта, потом простаивает всю генерацию;
#   * аудио-энкодер whisper-large-v3 и vocal separator — нужны один раз,
#     чтобы построить аудио-эмбеддинг;
#   * KV-кэш условных кадров для 2-го и следующих сегментов
#     (offload_kv_cache=False зашито в скрипте).
# Математика генерации не меняется — видео получается тем же самым.

_TEXT_ENCODER_OFFLOAD = '''\
# [lowvram] text encoder нужен только на время кодирования промпта:
# между кодированиями держим его на CPU и освобождаем видеопамять.
pipe.text_encoder.to("cpu")
_lowvram_orig_encode_prompt = pipe.encode_prompt
def _lowvram_encode_prompt(*args, **kwargs):
    pipe.text_encoder.to(local_rank)
    try:
        return _lowvram_orig_encode_prompt(*args, **kwargs)
    finally:
        pipe.text_encoder.to("cpu")
        torch_gc()
pipe.encode_prompt = _lowvram_encode_prompt
torch_gc()
'''

_AUDIO_MODELS_FREE = '''\
# [lowvram] аудио-эмбеддинг построен — whisper и vocal separator больше не нужны.
try:
    audio_encoder.to("cpu")
except Exception:
    pass
try:
    del vocal_separator
except NameError:
    pass
__import__("gc").collect()
torch_gc()
'''


def build_lowvram_script_text(src: str):
    """Возвращает (новый_текст_скрипта, список_применённых_правок).
    Каждая правка применяется только если её якорь найден ровно как
    ожидается; иначе она пропускается, а оригинал остаётся рабочим."""
    applied = []

    # 1) KV-кэш условных кадров — на CPU вместо GPU.
    if "offload_kv_cache=False" in src:
        src = src.replace("offload_kv_cache=False", "offload_kv_cache=True")
        applied.append("kv_cache_offload")

    # 2) text encoder на CPU между кодированиями промпта.
    m = re.search(r"^(?P<i>[ \t]*)pipe\.to\(local_rank\)[ \t]*\n", src, re.M)
    if m:
        block = textwrap.indent(_TEXT_ENCODER_OFFLOAD, m.group("i"))
        src = src[:m.end()] + block + src[m.end():]
        applied.append("text_encoder_offload")

    # 3) после построения аудио-эмбеддинга освобождаем whisper и separator.
    m = re.search(r"^(?P<i>[ \t]*)os\.remove\(temp_vocal_path\)[ \t]*\n", src, re.M)
    if m:
        block = textwrap.indent(_AUDIO_MODELS_FREE, m.group("i"))
        src = src[:m.end()] + block + src[m.end():]
        applied.append("audio_models_free")

    return src, applied


def prepare_demo_script() -> str:
    """Имя демо-скрипта, который запускать через torchrun (относительно
    REPO_DIR). При любой проблеме — оригинальный скрипт."""
    if not LOW_VRAM:
        print("[lowvram] LOW_VRAM=0 — запуск оригинального скрипта")
        return BASE_DEMO_SCRIPT
    try:
        base_path = os.path.join(REPO_DIR, BASE_DEMO_SCRIPT)
        with open(base_path, "r", encoding="utf-8") as f:
            src = f.read()
        patched, applied = build_lowvram_script_text(src)
        if not applied:
            print("[lowvram] ни одна правка не применилась — запуск оригинального скрипта")
            return BASE_DEMO_SCRIPT
        compile(patched, LOWVRAM_DEMO_SCRIPT, "exec")  # проверка синтаксиса
        with open(os.path.join(REPO_DIR, LOWVRAM_DEMO_SCRIPT), "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"[lowvram] применены правки: {', '.join(applied)}")
        return LOWVRAM_DEMO_SCRIPT
    except Exception as e:
        print(f"[lowvram] не удалось подготовить скрипт ({e}) — запуск оригинального скрипта")
        return BASE_DEMO_SCRIPT


DEMO_SCRIPT = prepare_demo_script()


def ensure_weights_present():
    """Веса (~42 ГБ) должны уже лежать на подключённом Network Volume —
    закачиваются ОДИН РАЗ вручную через обычный Pod на этом же Volume
    (см. инструкцию деплоя), а не на холодном старте воркера. Если их
    нет — explicit ошибка с понятной подсказкой, а не попытка скачать
    42 ГБ посреди обработки запроса (упёрлось бы в execution timeout)."""
    missing = [
        rel for rel in REQUIRED_WEIGHT_FILES
        if not os.path.exists(os.path.join(AVATAR_CKPT_DIR, rel))
    ]
    if missing:
        raise RuntimeError(
            f"Веса LongCat не найдены или неполные в {AVATAR_CKPT_DIR}. "
            f"Отсутствуют файлы: {', '.join(missing)}. "
            "Запусти download_weights.py один раз через обычный Pod на этом же "
            "Network Volume перед первым запросом к serverless-эндпоинту — "
            "см. инструкцию деплоя."
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
    num_segments = segments_for_duration(audio_duration)

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
        f"--nproc_per_node={NUM_GPUS}",
        DEMO_SCRIPT,
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
    if NUM_GPUS > 1:
        cmd += ["--context_parallel_size", str(NUM_GPUS)]

    # Даже если переменную забыли добавить на эндпоинте — включаем
    # expandable_segments (меньше фрагментации памяти).
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, env=env)
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
