# Video Object Replacer

AI-инструмент для замены объектов на видео: нейросеть находит объект, строит точную маску и закрашивает его новым контентом покадрово — с сохранением освещения, зернистости и плавности между кадрами.

## Как это работает

```
Видео → Извлечение кадров → Сегментация (SAM2)
      → Трекинг маски (Optical Flow)
      → ControlNet (Canny + Depth)
      → Inpainting SDXL
      → Temporal consistency
      → Цветовая гармонизация
      → Совпадение зернистости и motion blur
      → Сборка видео
```

### Три режима выбора объекта

| Режим | Описание |
|-------|----------|
| `auto` | Grounding DINO находит объект по тексту → показывает превью → пользователь подтверждает или уточняет кликом *(по умолчанию)* |
| `click` | Открывается первый кадр в окне, пользователь кликает на объект |
| `text` | Полностью автоматический, без GUI — только текстовый промпт |

---

## Требования к железу

| Режим | Минимум | Рекомендуется |
|-------|---------|---------------|
| CUDA (Linux / Windows) | GPU 8 GB VRAM | GPU 16+ GB VRAM |
| MPS (macOS Apple Silicon) | 16 GB RAM | 32 GB unified memory |
| CPU | 32 GB RAM | — (очень медленно) |

> SDXL-inpainting требует значительных ресурсов. На CPU обработка одного кадра может занимать несколько минут.

---

## Установка

### 1. Общие зависимости

#### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv git ffmpeg build-essential
```

#### macOS

```bash
# Установить Homebrew, если ещё нет:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.11 git ffmpeg
```

#### Windows

1. Установить [Python 3.11](https://www.python.org/downloads/) (отметить «Add to PATH»)
2. Установить [Git for Windows](https://git-scm.com/download/win)
3. Установить [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) и добавить в PATH:
   - Распаковать архив → скопировать путь к папке `bin`
   - `Пуск → Переменные среды → PATH → Добавить`

---

### 2. Клонировать репозиторий и создать виртуальное окружение

#### Linux / macOS

```bash
git clone https://github.com/your-username/video_changer.git
cd video_changer

python3.11 -m venv myvenv
source myvenv/bin/activate
```

#### Windows (cmd)

```bat
git clone https://github.com/your-username/video_changer.git
cd video_changer

python -m venv myvenv
myvenv\Scripts\activate.bat
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/your-username/video_changer.git
cd video_changer

python -m venv myvenv
myvenv\Scripts\Activate.ps1
```

> Если PowerShell выдаёт ошибку прав — выполни: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

### 3. Установить PyTorch

Выбери команду под своё железо:

#### CUDA 12.1 (Linux / Windows, NVIDIA GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 11.8 (Linux / Windows, старые GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### macOS (Apple Silicon M1/M2/M3 — MPS) и macOS Intel

```bash
pip install torch torchvision torchaudio
```

#### CPU (без GPU — только для тестирования)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Проверить, что PyTorch видит GPU:

```python
# Linux / Windows
python -c "import torch; print(torch.cuda.is_available())"  # должно быть True

# macOS Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"  # должно быть True
```

---

### 4. Установить остальные зависимости

```bash
pip install -r requirements.txt
```

> На Windows может упасть установка `controlnet_aux` из-за `onnxruntime`. Если так:
> ```bat
> pip install onnxruntime
> pip install -r requirements.txt
> ```

---

### 5. Скачать модели

#### SAM2 (Segment Anything Model 2)

```bash
# Linux / macOS
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O sam2_hiera_large.pt
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml -O sam2_hiera_l.yaml

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "sam2_hiera_large.pt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" -OutFile "sam2_hiera_l.yaml"
```

#### Grounding DINO (нужен для режимов `text` и `auto`)

```bash
# Linux / macOS
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -OutFile "groundingdino_swint_ogc.pth"
```

Конфиг Grounding DINO идёт внутри пакета `groundingdino-py`. Нужно узнать путь:

```bash
python -c "import groundingdino; import os; print(os.path.dirname(groundingdino.__file__))"
```

Затем передать путь в `--gdino-config`:

```
<путь_из_команды_выше>/config/GroundingDINO_SwinT_OGC.py
```

#### SDXL + ControlNet (скачиваются автоматически с HuggingFace)

Модели скачиваются при первом запуске (~15 GB суммарно). Нужен доступ к HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli login   # опционально, для приватных моделей
```

#### RAFT (опционально, для более точного optical flow)

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
# Скачать веса:
# Linux / macOS
./download_models.sh
# Windows: открыть download_models.sh и вручную скачать raft-things.pth
cd ..
```

При использовании добавлять в PYTHONPATH:

```bash
# Linux / macOS
export PYTHONPATH="$PYTHONPATH:$(pwd)/RAFT"

# Windows (cmd)
set PYTHONPATH=%PYTHONPATH%;%cd%\RAFT

# Windows (PowerShell)
$env:PYTHONPATH += ";$(Get-Location)\RAFT"
```

---

### 6. xformers (опционально — ускорение памяти на NVIDIA)

```bash
pip install xformers
```

> На macOS и CPU xformers не нужен.

---

## Структура файлов после установки

```
video_changer/
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── ui.py
│   ├── sam_load.py
│   ├── raft_load.py
│   └── multy_control_net.py
├── sam2_hiera_large.pt          ← скачать вручную
├── sam2_hiera_l.yaml            ← скачать вручную
├── groundingdino_swint_ogc.pth  ← скачать вручную
├── RAFT/                        ← опционально
├── myvenv/
├── requirements.txt
└── README.md
```

---

## Запуск

Все команды выполняются из корня проекта при активированном виртуальном окружении.

### Режим auto (рекомендуется)

```bash
python src/main.py \
  --input  input.mp4 \
  --output output.mp4 \
  --mode   auto \
  --detect-prompt "ceramic mug" \
  --prompt "ceramic mug with space galaxy design, photorealistic, studio lighting"
```

Программа откроет первый кадр видео, покажет что нашла, и предложит подтвердить или уточнить выделение.

### Режим click (ручной выбор объекта)

```bash
python src/main.py \
  --input  input.mp4 \
  --output output.mp4 \
  --mode   click \
  --prompt "mug with dragon print, hyperrealistic"
```

Нажми на объект в открывшемся окне → `Enter` чтобы подтвердить.

### Режим text (полностью автоматический, без GUI)

```bash
python src/main.py \
  --input         input.mp4 \
  --output        output.mp4 \
  --mode          text \
  --detect-prompt "mug" \
  --prompt        "red mug with golden logo"
```

### Максимальное качество

```bash
python src/main.py \
  --input          input.mp4 \
  --output         output.mp4 \
  --detect-prompt  "ceramic mug" \
  --prompt         "ceramic mug with dragon, hyperrealistic, 8k, studio lighting" \
  --multi-controlnet \
  --use-raft       \
  --raft-model     RAFT/models/raft-things.pth \
  --steps          40 \
  --guidance-scale 8.0 \
  --blend-alpha    0.8
```

### Принудительно указать устройство

```bash
# NVIDIA GPU
python src/main.py --device cuda ...

# macOS Apple Silicon
python src/main.py --device mps ...

# CPU (медленно)
python src/main.py --device cpu ...
```

---

## Все параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--input` | `input.mp4` | Входное видео |
| `--output` | `output.mp4` | Выходное видео |
| `--mode` | `auto` | Режим выбора: `click`, `text`, `auto` |
| `--detect-prompt` | *(первые 3 слова --prompt)* | Что найти на видео (для Grounding DINO) |
| `--prompt` | *(встроенный)* | Что сгенерировать вместо (для SDXL) |
| `--negative-prompt` | *(встроенный)* | Что исключить из генерации |
| `--device` | *(авто)* | `cuda` / `mps` / `cpu` |
| `--seed` | `42` | Фиксированный seed для воспроизводимости |
| `--steps` | `30` | Шаги диффузии на кадр (больше = лучше, медленнее) |
| `--guidance-scale` | `7.5` | CFG scale (выше = строже следует промпту) |
| `--blend-alpha` | `0.85` | Темпоральный блендинг (1.0 = без сглаживания) |
| `--multi-controlnet` | выкл | Canny + Depth ControlNet (выше качество) |
| `--use-raft` | выкл | RAFT вместо Farneback (точнее, нужен RAFT repo) |
| `--raft-model` | `RAFT/models/raft-things.pth` | Путь к весам RAFT |
| `--sam2-checkpoint` | `sam2_hiera_large.pt` | Путь к весам SAM2 |
| `--sam2-config` | `sam2_hiera_l.yaml` | Путь к конфигу SAM2 |
| `--gdino-config` | *(внутри пакета)* | Путь к конфигу Grounding DINO |
| `--gdino-checkpoint` | `groundingdino_swint_ogc.pth` | Путь к весам Grounding DINO |

---

## Управление в интерактивном окне

| Клавиша | Действие |
|---------|----------|
| `Left Click` | Поставить точку / кликнуть на объект |
| `Enter` | Подтвердить выбор |
| `R` | Сбросить / переделать |
| `F` | Уточнить кликом (в окне превью маски) |
| `Esc` | Отмена |

---

## Частые проблемы

### `CUDA out of memory`

Уменьши количество шагов и/или отключи multi-controlnet:

```bash
python src/main.py --steps 20 ...
# Или освободить VRAM перед запуском, закрыв другие программы
```

### Grounding DINO не находит объект

- Попробуй более простой текст: вместо `"white ceramic coffee mug"` → `"mug"` или `"cup"`
- Уменьши пороги через `--mode auto` и переопредели в коде: в `detector.py` параметры `box_threshold=0.35` (уменьши до `0.25`)
- Переключись на `--mode click`

### `ModuleNotFoundError: No module named 'groundingdino'`

```bash
pip install groundingdino-py
# Если не помогает — установить из исходников:
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..
```

### `No module named 'sam2'`

```bash
pip install -e git+https://github.com/facebookresearch/sam2.git#egg=SAM_2
```

### На macOS: `MPS backend out of memory`

```bash
# Ограничить использование памяти MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python src/main.py --device mps ...
```

### На Windows: ошибка при сборке видео через imageio

```bash
pip install imageio-ffmpeg
# Убедиться что ffmpeg установлен и доступен в PATH:
ffmpeg -version
```

### `RuntimeError: SAM2 checkpoint not found`

Убедись, что запускаешь скрипт из корневой папки проекта:

```bash
cd /path/to/video_changer
python src/main.py ...
```

---

## Зависимости (ключевые)

| Библиотека | Версия | Назначение |
|-----------|--------|-----------|
| PyTorch | 2.x | Основной DL фреймворк |
| diffusers | 0.36+ | SDXL Inpainting pipeline |
| SAM2 | latest | Сегментация объектов |
| Grounding DINO | 0.4+ | Детекция по тексту |
| controlnet_aux | 0.0.10 | Depth/Canny карты для ControlNet |
| opencv-python | 4.x | Обработка видео и UI |
| imageio + imageio-ffmpeg | 2.x | Сборка видео |
| RAFT | — | Точный optical flow (опционально) |
