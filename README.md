# Video Object Replacer

AI-инструмент для замены объектов на видео: нейросеть находит объект, строит точную маску и закрашивает его новым контентом покадрово — с сохранением освещения, зернистости и плавности между кадрами.

## Как это работает

```
Видео
  → Извлечение кадров
  → Детекция объекта: Grounding DINO сканирует N равномерных кадров,
    берёт детекцию с максимальным confidence (не первую найденную)
  → Начальная маска (SAM2 Image Predictor)
  → Трекинг маски по всему видео (SAM2 Video Predictor)
    + заполнение 1-2 кадровых пробелов (drop-out compensation)
  → Для кадров с объектом:
      → Анализ освещения сцены → авто-дополнение промпта
      → ControlNet (Canny [+ Depth])
      → SDXL Inpainting (мягкая маска + IP-Adapter референс)
      → Poisson seamless blending (бесшовный стык)
      → Temporal consistency (optical flow)
      → Синтетическая тень (по оценённому направлению света)
      → Цветовая гармонизация (LAB)
      → Совпадение зернистости и motion blur
  → Сборка видео
```

> Кадры без объекта копируются без обработки — диффузионная модель на них не запускается.

### Что делает результат реалистичным

| Техника | Что решает |
|---------|-----------|
| **IP-Adapter reference** | Все кадры генерируются с одинаковой текстурой/стилем (первый кадр → референс) |
| **Авто-анализ освещения** | SDXL получает описание яркости, цветовой температуры и контраста сцены |
| **Мягкая маска** | Gaussian feathering убирает резкий край маски до передачи в SDXL |
| **Poisson blending** | Математически бесшовный переход на границе (решение уравнения Пуассона) |
| **Синтетическая тень** | Тень под объектом в направлении, обратном источнику света |
| **Color harmonization** | Цветовая статистика (mean/std в LAB) выравнивается с окружением |
| **Grain matching** | Зернистость оригинала добавляется в inpainted регион |
| **Motion blur** | Граница маски размывается пропорционально optical flow |
| **Temporal consistency** | 15% предыдущего кадра смешивается с текущим → нет мерцания |

### Три режима выбора объекта

| Режим | Описание |
|-------|----------|
| `auto` | Grounding DINO находит объект по тексту → показывает превью → пользователь подтверждает или уточняет кликом *(по умолчанию)* |
| `click` | Открывается первый кадр в окне, пользователь кликает на объект |
| `text` | Полностью автоматический, без GUI — только текстовый промпт |

---

## Требования к железу

| Режим | VRAM / RAM | Флаг | Время/кадр |
|-------|-----------|------|-----------|
| CUDA, 16+ GB VRAM | полная загрузка | *(нет)* | ~5-10 сек |
| CUDA, 8 GB VRAM | CPU offload | `--cpu-offload` | ~15-25 сек |
| CUDA, 6 GB VRAM | Sequential offload | `--sequential-offload` | ~30-60 сек |
| MPS (Apple Silicon) | 16 GB unified | *(авто)* | ~3 мин |
| CPU | 32 GB RAM | *(авто)* | ~10-20 мин |

> **RTX 5060 / 3060 / 4060 (8 GB)**: добавь `--cpu-offload`. SDXL+ControlNet+IP-Adapter суммарно требуют ~8 GB, CPU offload снижает пиковое потребление до ~4 GB.
>
> **xformers** (только NVIDIA): установи `pip install xformers` для дополнительного снижения VRAM на ~15-20%.
>
> На macOS MPS SDXL обрабатывает ~3 мин/кадр. Для быстрой проверки используй `--steps 10`.

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

#### macOS (Apple Silicon M1/M2/M3/M4 — MPS) и macOS Intel

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

Все модели кладутся в **корень проекта** (рядом с `src/`).

#### SAM2.1 (Segment Anything Model 2)

```bash
# Linux / macOS
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O sam2_hiera_large.pt

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "sam2_hiera_large.pt"
```

Конфиг SAM2 идёт внутри пакета (устанавливается автоматически через `requirements.txt`).
По умолчанию используется путь `configs/sam2.1/sam2.1_hiera_l.yaml` — менять не нужно.

#### Grounding DINO (нужен для режимов `text` и `auto`)

```bash
# Linux / macOS
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -OutFile "groundingdino_swint_ogc.pth"
```

Конфиг Grounding DINO определяется автоматически из установленного пакета. Ничего дополнительно скачивать не нужно.

#### SDXL + ControlNet + IP-Adapter (скачиваются автоматически с HuggingFace)

Модели скачиваются при первом запуске (~16.5 GB суммарно). Нужен доступ к HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli login   # опционально, для приватных моделей
```

| Модель | Размер | Назначение |
|--------|--------|-----------|
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | ~6 GB | SDXL Inpainting |
| `diffusers/controlnet-canny-sdxl-1.0` | ~2.5 GB | ControlNet Canny |
| `diffusers/controlnet-depth-sdxl-1.0` | ~2.5 GB | ControlNet Depth (только с `--multi-controlnet`) |
| `h94/IP-Adapter` (sdxl_models) | ~1.5 GB | IP-Adapter (стилевой референс) |

> IP-Adapter скачивается автоматически при первом запуске. Отключить: `--no-ip-adapter`

#### RAFT (опционально, для более точного temporal blending)

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
# Linux / macOS
./download_models.sh
# Windows: открыть download_models.sh и вручную скачать raft-things.pth
cd ..
```

При использовании добавить в PYTHONPATH:

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
│   ├── main.py             — основной скрипт
│   ├── detector.py         — Grounding DINO + SAM2 (text→mask)
│   ├── sam_load.py         — SAM2 Image Predictor
│   ├── sam_video.py        — SAM2 Video Predictor (трекинг по всему видео)
│   ├── ui.py               — интерактивный OpenCV UI
│   ├── raft_load.py        — RAFT optical flow (опционально)
│   └── multy_control_net.py — Multi-ControlNet (Canny + Depth)
├── sam2_hiera_large.pt          ← скачать (шаг 5)
├── groundingdino_swint_ogc.pth  ← скачать (шаг 5)
├── RAFT/                        ← опционально (шаг 5)
├── videos/                      ← папка для входных/выходных видео
├── myvenv/
├── requirements.txt
└── README.md
```

---

## Запуск

Все команды выполняются из корня проекта при активированном виртуальном окружении.

### Быстрая проверка (мало шагов — для теста)

```bash
python src/main.py \
  --input  videos/input.mp4 \
  --output videos/output.mp4 \
  --mode   text \
  --detect-prompt "mug" \
  --prompt "ceramic mug with space galaxy design, photorealistic" \
  --steps  10
```

### Режим auto (рекомендуется)

```bash
python src/main.py \
  --input  videos/input.mp4 \
  --output videos/output.mp4 \
  --mode   auto \
  --detect-prompt "ceramic mug" \
  --prompt "ceramic mug with space galaxy design, photorealistic, studio lighting"
```

Программа покажет найденный объект в окне и предложит подтвердить или уточнить.

### Режим click (ручной выбор объекта)

```bash
python src/main.py \
  --input  videos/input.mp4 \
  --output videos/output.mp4 \
  --mode   click \
  --prompt "mug with dragon print, hyperrealistic"
```

Нажми на объект в открывшемся окне → `Enter` чтобы подтвердить.

### Режим text (полностью автоматический, без GUI)

```bash
python src/main.py \
  --input         videos/input.mp4 \
  --output        videos/output.mp4 \
  --mode          text \
  --detect-prompt "mug" \
  --prompt        "red mug with golden logo"
```

### Максимальное качество

```bash
python src/main.py \
  --input           videos/input.mp4 \
  --output          videos/output.mp4 \
  --mode            auto \
  --detect-prompt   "ceramic mug" \
  --prompt          "ceramic mug with dragon, hyperrealistic, 8k, studio lighting" \
  --multi-controlnet \
  --use-raft        \
  --raft-model      RAFT/models/raft-things.pth \
  --steps           40 \
  --guidance-scale  8.0 \
  --blend-alpha     0.8 \
  --ip-adapter-scale 0.65 \
  --shadow-opacity  0.4 \
  --feather-radius  16
```

### Быстрый тест без тяжёлых модулей

```bash
python src/main.py \
  --input          videos/input.mp4 \
  --output         videos/output.mp4 \
  --mode           text \
  --detect-prompt  "mug" \
  --prompt         "mug with galaxy print, photorealistic" \
  --steps          10 \
  --no-ip-adapter \
  --shadow-opacity 0
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

### Ввод / вывод

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--input` | `input.mp4` | Входное видео |
| `--output` | `output.mp4` | Выходное видео |
| `--device` | *(авто)* | `cuda` / `mps` / `cpu` |

### Выбор объекта

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--mode` | `auto` | Режим: `click`, `text`, `auto` |
| `--detect-prompt` | *(первые 3 слова --prompt)* | Что найти (для Grounding DINO) |
| `--box-threshold` | `0.35` | Порог уверенности bbox (ниже → больше детекций) |
| `--text-threshold` | `0.25` | Порог совпадения текста |
| `--scan-frames` | `30` | Кол-во равномерных зондов по всему видео; все сканируются, побеждает наибольший confidence |
| `--fill-gaps` | `2` | Заполнять пробелы маски длиной до N кадров (SAM2 drop-out) |

### Генерация (SDXL)

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--prompt` | *(встроенный)* | Что сгенерировать вместо объекта |
| `--negative-prompt` | *(встроенный)* | Что исключить из генерации |
| `--seed` | `42` | Фиксированный seed |
| `--steps` | `30` | Шаги диффузии (больше = лучше, медленнее) |
| `--guidance-scale` | `7.5` | CFG scale |
| `--feather-radius` | `12` | Размытие краёв маски перед inpainting (0 = жёсткий край) |

### Реализм и постобработка

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--no-ip-adapter` | выкл | Отключить IP-Adapter (по умолчанию **включён**) |
| `--ip-adapter-scale` | `0.55` | Сила влияния IP-Adapter референса (0–1) |
| `--no-light-analysis` | выкл | Отключить авто-анализ освещения сцены |
| `--shadow-opacity` | `0.35` | Непрозрачность тени (0 = без тени) |
| `--shadow-offset` | `10` | Смещение тени в пикселях |
| `--light-angle` | *(авто)* | Направление света в градусах (0=право, 90=верх, 180=лево) |
| `--no-poisson` | выкл | Отключить Poisson seamless blending |
| `--blend-alpha` | `0.85` | Темпоральный блендинг (1.0 = без сглаживания) |

### Управление памятью GPU

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--cpu-offload` | выкл | Модели перемещаются в CPU RAM по необходимости (~4 GB VRAM) |
| `--sequential-offload` | выкл | Один слой на GPU одновременно (~3 GB VRAM, медленнее) |

### ControlNet / Optical flow

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--multi-controlnet` | выкл | Canny + Depth ControlNet (лучше качество, медленнее) |
| `--use-raft` | выкл | RAFT optical flow (точнее Farneback, нужен RAFT repo) |
| `--raft-model` | `RAFT/models/raft-things.pth` | Путь к весам RAFT |

### Пути к моделям

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--sam2-checkpoint` | `sam2_hiera_large.pt` | Веса SAM2 |
| `--sam2-config` | `configs/sam2.1/sam2.1_hiera_l.yaml` | Конфиг SAM2 |
| `--gdino-config` | *(авто)* | Конфиг Grounding DINO (авто-определяется) |
| `--gdino-checkpoint` | `groundingdino_swint_ogc.pth` | Веса Grounding DINO |

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

На 8 GB GPU (RTX 3060/4060/5060) — включи CPU offload:
```bash
python src/main.py --cpu-offload ...
```

На 6 GB GPU — используй sequential offload (медленнее, но минимальные требования):
```bash
python src/main.py --sequential-offload ...
```

Также помогает: уменьшить шаги, отключить IP-Adapter и multi-controlnet:
```bash
python src/main.py --cpu-offload --steps 15 --no-ip-adapter ...
```

### Grounding DINO не находит объект

- Попробуй более простое слово: `"mug"`, `"cup"`, `"watch"`, `"bottle"` вместо длинных фраз
- Снизь пороги:
  ```bash
  --box-threshold 0.20 --text-threshold 0.15
  ```
- Увеличь диапазон сканирования кадров:
  ```bash
  --scan-frames 30
  ```
- Переключись на ручной режим: `--mode click`

### На macOS MPS всё очень медленно

Это нормально — Apple Silicon не имеет полноценной поддержки SDXL.
Для быстрой проверки результата:

```bash
python src/main.py --steps 10 ...
```

Также можно ограничить использование памяти:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python src/main.py --device mps ...
```

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

### `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`

Несовместимость `groundingdino-py` с `transformers >= 4.48`. Исправить:

```bash
pip install "transformers>=4.40.0,<4.48.0"
```

### `RuntimeError: Error(s) in loading state_dict for SAM2Base`

Несоответствие чекпоинта и конфига SAM2. Убедись, что скачал именно `sam2.1_hiera_large.pt` (не старый `sam2_hiera_large.pt`), и используй конфиг по умолчанию `configs/sam2.1/sam2.1_hiera_l.yaml`.

### `RuntimeError: SAM2 checkpoint not found`

Убедись, что запускаешь скрипт из корневой папки проекта:

```bash
cd /path/to/video_changer
python src/main.py ...
```

### IP-Adapter не загружается / ошибка при скачивании

IP-Adapter требует интернета при первом запуске (~1.5 GB). Если нет доступа или ошибка:
```bash
python src/main.py --no-ip-adapter ...
```

### Объект на всех кадрах выглядит по-разному (мерцание стиля)

IP-Adapter решает эту проблему. Убедись что он включён (по умолчанию), и попробуй увеличить его влияние:
```bash
--ip-adapter-scale 0.7
```

### Grounding DINO выбирает не тот объект

Используй более высокий порог чтобы отфильтровать слабые детекции:
```bash
--box-threshold 0.45
```
Или переключись на ручной режим: `--mode click`

### На Windows: ошибка при сборке видео через imageio

```bash
pip install imageio-ffmpeg
# Убедиться что ffmpeg установлен и доступен в PATH:
ffmpeg -version
```

---

## Зависимости (ключевые)

| Библиотека | Версия | Назначение |
|-----------|--------|-----------|
| PyTorch | 2.x | Основной DL фреймворк |
| diffusers | **0.33+** | SDXL Inpainting + IP-Adapter SDXL |
| SAM2 | latest (git) | Сегментация и трекинг объектов |
| Grounding DINO | 0.4+ | Детекция по текстовому промпту |
| transformers | 4.40–4.47 | Текстовые энкодеры (BERT для DINO, CLIP для SDXL) |
| controlnet_aux | 0.0.10 | Depth/Canny карты для ControlNet |
| opencv-python | 4.x | Обработка видео, Poisson blending, UI |
| imageio + imageio-ffmpeg | 2.x | Сборка видео |
| huggingface_hub | 0.23+ | Авто-загрузка IP-Adapter весов |
| RAFT | — | Точный optical flow (опционально) |

> **Примечание**: IP-Adapter веса (`h94/IP-Adapter`, ~1.5 GB) и модели SDXL/ControlNet (~15 GB) скачиваются автоматически с HuggingFace при первом запуске.
