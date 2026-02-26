#!/usr/bin/env python3
"""
Диагностика скорости SDXL на RTX 5060 (Blackwell).
Запуск: python diagnose_speed.py

Собирает все замеры из чек-листа GPT для поиска причины 60 сек/step.
"""

import subprocess
import sys
import threading
import time

# Запуск из корня проекта: python diagnose_speed.py

import torch
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# СБОР ДАННЫХ: nvidia-smi во время нагрузки
# ─────────────────────────────────────────────────────────────────────────────

_nvidia_smi_log: list[str] = []
_nvidia_smi_stop = threading.Event()


def _nvidia_smi_loop():
    """Периодически читает nvidia-smi -q -d CLOCK в фоне."""
    while not _nvidia_smi_stop.is_set():
        try:
            kwargs = {"capture_output": True, "text": True, "timeout": 5}
            if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            out = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK"], **kwargs)
            if out.returncode == 0:
                _nvidia_smi_log.append(out.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        except Exception:
            pass
        _nvidia_smi_stop.wait(timeout=3)


def _extract_graphics_clock(log_lines: list[str]) -> str:
    """Извлечь Graphics clock из вывода nvidia-smi -q -d CLOCK."""
    text = "".join(log_lines)
    for line in text.split("\n"):
        if "Graphics" in line and ":" in line and "MHz" in line:
            return line.strip()
    return "Graphics clock: (not found in log)"


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 0: Базовая информация
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ДИАГНОСТИКА СКОРОСТИ SDXL — RTX 5060 Blackwell")
print("=" * 70)

print("\n--- ШАГ 0: PyTorch / CUDA / GPU ---")
# nvidia-smi (idle) для справки
try:
    r = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        for line in r.stdout.split("\n"):
            if "Graphics" in line and "MHz" in line:
                print("nvidia-smi (idle) Graphics:", line.strip().split(":")[-1].strip())
                break
except Exception:
    print("nvidia-smi: не удалось выполнить (запусти вручную: nvidia-smi -q -d CLOCK)")
print("PyTorch:", torch.__version__)
print("CUDA (PyTorch):", torch.version.cuda or "N/A")
print("cuDNN:", torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print("Device capability (sm_X.Y):", cap, "→ sm_" + str(cap[0]) + str(cap[1]))
    print("GPU:", name)
    print("VRAM:", f"{vram_gb:.2f} GB")
else:
    print("CUDA недоступна. Выход.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 6: Чистый SDXL (без ControlNet, без inpainting)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ШАГ 6: PURE SDXL (без ControlNet, без inpainting)")
print("=" * 70)

# Используем bf16 для Blackwell
dtype = torch.bfloat16 if cap[0] >= 12 else torch.float16
print(f"dtype: {dtype}")

print("Загрузка StableDiffusionXLPipeline...")
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

pure_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
).to("cuda")

# Запускаем nvidia-smi в фоне во время теста
_nvidia_smi_log.clear()
_nvidia_smi_stop.clear()
smi_thread = threading.Thread(target=_nvidia_smi_loop, daemon=True)
smi_thread.start()

print("Генерация: prompt='cat', 10 steps, 512x512...")
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    pure_pipe("a photograph of a cat", num_inference_steps=10, height=512, width=512)
torch.cuda.synchronize()
pure_time = time.time() - start

_nvidia_smi_stop.set()
smi_thread.join(timeout=2)

print(f"\n>>> PURE SDXL TIME: {pure_time:.2f} sec (10 steps)")
print(f">>> ≈ {pure_time/10:.2f} sec/step")
print(f">>> Ожидаемо: 10–40 сек total, 1–4 сек/step. Если 60+ сек/step — что-то сломано.")

# Частоты GPU во время генерации
gfx = _extract_graphics_clock(_nvidia_smi_log)
print(f"\n>>> Graphics clock (во время теста): {gfx}")
print(">>> RTX 5060 должна работать ~2000+ MHz. Если <1000 — карта задушена.")

# Освобождаем память перед загрузкой второго пайплайна
del pure_pipe
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 1–5, 8–11: Проверка нашего пайплайна (ControlNet Inpaint)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ШАГ 1–5: Состояние нашего пайплайна (ControlNet Inpaint)")
print("=" * 70)

# Загружаем наш пайплайн БЕЗ torch.compile, БЕЗ cpu_offload, БЕЗ attention_slicing
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import (
    StableDiffusionXLControlNetInpaintPipeline,
)

_controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=dtype,
)
controlnet = _controlnet.to("cuda")

our_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet,
    torch_dtype=dtype,
).to("cuda")

# Явно отключаем всё, что может замедлять (как в ШАГ 2)
if hasattr(our_pipe, "disable_attention_slicing"):
    our_pipe.disable_attention_slicing()
our_pipe.unet.gradient_checkpointing = False
if hasattr(our_pipe.vae, "gradient_checkpointing"):
    our_pipe.vae.gradient_checkpointing = False

# Проверка состояния (ШАГ 1)
print("\n--- Состояние pipe ---")
print("dtype UNet:", next(our_pipe.unet.parameters()).dtype)
print("dtype VAE:", next(our_pipe.vae.parameters()).dtype)
print("device UNet:", next(our_pipe.unet.parameters()).device)
print("is compiled:", hasattr(our_pipe.unet, "_orig_mod"))

# Attention processors (ШАГ 11)
attn_procs = our_pipe.unet.attn_processors
try:
    proc_type = type(list(attn_procs.values())[0]).__name__ if attn_procs else "unknown"
except (IndexError, TypeError):
    proc_type = str(attn_procs)[:80]
print("attn_processors type:", proc_type)
if "xformers" in str(proc_type).lower():
    print(">>> ВНИМАНИЕ: используется xformers — на Blackwell старый xformers может быть крайне медленным!")

# ШАГ 3: cpu offload не используем (мы загрузили .to("cuda"))

# Генерация тестового изображения и маски
print("\n--- Тест нашего пайплайна (768x768, 5 steps) ---")
h, w = 768, 768
img = np.zeros((h, w, 3), dtype=np.uint8) + 128
mask = np.zeros((h, w), dtype=np.uint8)
mask[200:400, 200:400] = 255  # маленький квадрат в центре

# Canny для control_image
import cv2
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 50, 150)
canny_rgb = np.stack([canny] * 3, axis=-1)
control_pil = Image.fromarray(canny_rgb)

img_pil = Image.fromarray(img)
mask_pil = Image.fromarray(mask)

_nvidia_smi_log.clear()
_nvidia_smi_stop.clear()
smi_thread = threading.Thread(target=_nvidia_smi_loop, daemon=True)
smi_thread.start()

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    our_pipe(
        "a ceramic mug",
        image=img_pil,
        mask_image=mask_pil,
        control_image=control_pil,
        num_inference_steps=5,
        height=h,
        width=w,
    )
torch.cuda.synchronize()
our_time = time.time() - start

_nvidia_smi_stop.set()
smi_thread.join(timeout=2)

print(f"\n>>> OUR PIPELINE TIME (5 steps): {our_time:.2f} sec")
print(f">>> ≈ {our_time/5:.2f} sec/step")
print(f">>> Ориентир: ~0.5–2 сек/step. Если 60+ сек/step — проблема в пайплайне/ControlNet.")

# ─────────────────────────────────────────────────────────────────────────────
# ИТОГОВАЯ СВОДКА
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ИТОГОВАЯ СВОДКА (скопируй и отправь в GPT)")
print("=" * 70)
print(f"""
Device capability: {cap}
GPU: {name}
PyTorch: {torch.__version__}
CUDA: {torch.version.cuda}

PURE SDXL TIME (10 steps, 512x512): {pure_time:.2f} sec
OUR PIPELINE TIME (5 steps, 768x768): {our_time:.2f} sec

dtype UNet: {next(our_pipe.unet.parameters()).dtype}
device UNet: {next(our_pipe.unet.parameters()).device}
is compiled: {hasattr(our_pipe.unet, "_orig_mod")}
attn_processors: {proc_type}

Graphics clock (during pure SDXL test): {gfx}
""")
print("=" * 70)
