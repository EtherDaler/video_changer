# video_changer — контекст для агента

## Назначение

Проект — **подмена или изменение объектов на видео по текстовому промпту**. Идея: не генерировать всю сцену, а найти объект, сгенерировать только его в **ROI** и склеить с исходным кадром.

## Пайплайн (актуальный: `video_changer_colab_v3.ipynb`)

1. **Grounding DINO** — сканирование кадров, bbox объекта по **`detect_prompt`**.
2. **SAM2** — маска на seed-кадре, затем **video tracking** масок; `fill_mask_gaps`.
3. **Shot cuts** — гистограммы HSV → границы планов (сброс части состояния между сценами).
4. **CLIP re-ID** (опционально) — при повторном появлении объекта после пропажи.
5. **Optical flow + IoU масок** — решение «нужна ли новая генерация» (на таких кадрах вызывается API).
6. **Диффузия** (`API_MODE`): **WAN** (Replicate), **nano** (Gen-API Nano Banana 2), **bria** (Gen-API Replace Item). Вход в API: ROI с апскейлом по **`roi_min_side_for_api`**, JPEG **`api_jpeg_quality`**, у Nano — ещё **`nano_resolution`**.
7. **Nano (v3 ноутбук)** — в API: **полный кадр** (даунскейл по **`nano_scene_max_side`**), **кроп ROI**, опционально **предыдущий сгенерированный кроп**; маска в API по умолчанию не отправляется. Склейка — **`composite_roi_bgr`** (как у Bria, с более широким feather). Устаревший режим с маской: **`nano_send_mask=True`**.
8. **Склейка** — `composite_roi_simple_bgr` / `composite_roi_bgr` из `src/v3_compositing.py`; temporal blend **`blend_alpha`**.

Локальный **`src/main.py`** — отдельный путь: SDXL Inpainting + ControlNet + IP-Adapter (см. README), не смешивать с описанием Colab v3 без необходимости.

## Главные артефакты

| Файл | Роль |
|------|------|
| [video_changer_colab_v3.ipynb](video_changer_colab_v3.ipynb) | Основной видео-пайплайн: DINO → SAM2 video → shot / flow / **WAN \| nano \| bria** → склейка. |
| [video_changer_colab.ipynb](video_changer_colab.ipynb) | Более ранний полный ноутбук (Nano + склейка); ориентир по идее. |
| [single_frame_object_replace_flux_fill_colab.ipynb](single_frame_object_replace_flux_fill_colab.ipynb) | Эталон **одного кадра**: ROI + API + compositing. |

| Модуль `src/` | Роль |
|---------------|------|
| `nano_genapi.py` | Nano Banana 2, Bria Replace Item, `call_inpaint_crop`, апскейл ROI. |
| `v3_nano_cutout.py` | (legacy) вырез PNG после Nano; основной путь v3 — без cutout. |
| `v3_compositing.py` | Склейка ROI с кадром. |
| `v3_notebook_compat.py` | Загрузка моделей, детекция, SAM2 video для ноутбука. |
| `replicate_wan.py` | WAN inpaint через Replicate. |

При правках **видео-ноутбука** не ломать без нужды: загрузка видео, SAM2 Image + Video, `fill_mask_gaps`, цикл «пустая маска → оригинал». Менять в первую очередь блок **замены на кадре** и конфиг.

## Параметры и секреты

- **Gen-API** (`nano`, `bria`): переменная **`GEN_API_KEY`** или поле в конфиге — **не коммитить** реальные ключи.
- **Replicate** (`wan`): **`REPLICATE_API_TOKEN`**.
- Промпты: **`detect_prompt`** (что ищем в исходном видео), **`replace_prompt`** (что подставить в ROI в API).

## Ограничения и заметки

- Трекинг: опционально **`EXPAND_REFLECTION_BBOX`** — расширение bbox по второй компоненте маски (отражение на глянце).
- **Bria**: маска в API — **SAM в ROI**, не полный белый прямоугольник.
- Не путать с устаревшим «SDXL на весь кадр» в старых описаниях — целевой путь для видео в ноутбуке: **ROI + Gen-API / WAN + склейка**.

## Стиль изменений в коде

- Минимальный дифф: только то, что относится к задаче.
- Ноутбуки — сохранять работоспособность ячеек по порядку (Colab/локально).
