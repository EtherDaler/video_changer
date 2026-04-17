"""Gen-API Nano Banana 2 — minimal client for video_changer_colab_v3.ipynb."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import requests
from PIL import Image

NANO_BANANA_URL = "https://api.gen-api.ru/api/v1/networks/nano-banana-2"
BRIA_REPLACE_ITEM_URL = "https://api.gen-api.ru/api/v1/functions/replace-item"
BRIA_FIBO_URL = "https://api.gen-api.ru/api/v1/networks/bria-fibo"
# Inpainting по маске в ROI (документация: https://gen-api.ru/model/flux/api , https://gen-api.ru/model/sdxl/api )
FLUX_NETWORK_URL = "https://api.gen-api.ru/api/v1/networks/flux"
SDXL_NETWORK_URL = "https://api.gen-api.ru/api/v1/networks/sdxl"

# Gen-API Flux (model ultra): только эти canvas — см. ответ сервера «Допустимые разрешения».
FLUX_GENAPI_ALLOWED_SIZES: List[Tuple[int, int]] = [
    (3136, 1344),
    (2048, 2048),
    (2752, 1536),
    (2368, 1792),
    (1344, 3136),
    (1536, 2752),
    (1792, 2368),
]


@dataclass(frozen=True)
class FluxLetterboxMeta:
    """Обратное преобразование: crop с холста Flux → исходный ROI."""

    tw: int
    th: int
    orig_w: int
    orig_h: int
    pad_x: int
    pad_y: int
    content_w: int
    content_h: int


def _flux_pick_allowed_size(ow: int, oh: int) -> Tuple[int, int]:
    """Выбирает (tw, th) из белого списка с ближайшим соотношением сторон к ow/oh."""
    if ow <= 0 or oh <= 0:
        return (2048, 2048)
    target_ar = ow / float(oh)
    best_tw, best_th = FLUX_GENAPI_ALLOWED_SIZES[0]
    best_score = float("inf")
    for tw, th in FLUX_GENAPI_ALLOWED_SIZES:
        ar = tw / float(th)
        score = abs(math.log(ar) - math.log(target_ar))
        if score < best_score - 1e-9:
            best_score = score
            best_tw, best_th = tw, th
        elif abs(score - best_score) <= 1e-9:
            # При равном аспекте — больший холст (чуть больше деталей).
            if tw * th > best_tw * best_th:
                best_tw, best_th = tw, th
    return best_tw, best_th


def _flux_letterbox_to_canvas(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    tw: int,
    th: int,
) -> Tuple[Image.Image, Image.Image, FluxLetterboxMeta]:
    """Вписывает ROI в tw×th (contain), поля — чёрные; маска в полях — 0."""
    ow, oh = roi_img_pil.size
    scale = min(tw / float(ow), th / float(oh))
    sw = max(1, int(round(ow * scale)))
    sh = max(1, int(round(oh * scale)))
    sw = min(sw, tw)
    sh = min(sh, th)
    img_s = roi_img_pil.resize((sw, sh), Image.Resampling.LANCZOS)
    mask_s = roi_mask_pil.resize((sw, sh), Image.Resampling.NEAREST)
    pad_x = (tw - sw) // 2
    pad_y = (th - sh) // 2
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    canvas.paste(img_s, (pad_x, pad_y))
    m_canvas = Image.new("L", (tw, th), 0)
    m_canvas.paste(mask_s, (pad_x, pad_y))
    meta = FluxLetterboxMeta(
        tw=tw,
        th=th,
        orig_w=ow,
        orig_h=oh,
        pad_x=pad_x,
        pad_y=pad_y,
        content_w=sw,
        content_h=sh,
    )
    return canvas, m_canvas, meta


def expand_bbox_for_genapi_context(
    inner_pb: Tuple[int, int, int, int],
    frame_shape: Tuple[int, ...],
    expand_frac: float,
) -> Tuple[int, int, int, int]:
    """
    Расширяет bbox для запроса к Bria / Fibo / Flux / SDXL (больше контекста вокруг объекта).
    `expand_frac` — доля от ширины/высоты inner: столько пикселей добавляется с каждой стороны (по каждой оси).
    """
    if expand_frac <= 0:
        return inner_pb
    x1, y1, x2, y2 = inner_pb
    H, W = int(frame_shape[0]), int(frame_shape[1])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return inner_pb
    dx = int(round(bw * expand_frac))
    dy = int(round(bh * expand_frac))
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(W, x2 + dx)
    ny2 = min(H, y2 + dy)
    if nx2 <= nx1 or ny2 <= ny1:
        return inner_pb
    return (nx1, ny1, nx2, ny2)


def _preprocess_mask_for_genapi(
    mask_gray: np.ndarray,
    dilate_px: int,
    blur_sigma: float,
) -> np.ndarray:
    """Дилатация + лёгкое Gaussian по маске только для входа в API (смягчить края объекта)."""
    if dilate_px <= 0 and blur_sigma <= 0:
        return mask_gray
    out = mask_gray.astype(np.uint8, copy=True)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)
    if blur_sigma > 0:
        out_f = out.astype(np.float32)
        out_f = cv2.GaussianBlur(out_f, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
        out = np.clip(np.round(out_f), 0, 255).astype(np.uint8)
    return out


def mask_for_genapi_inpaint(
    mask_gray: np.ndarray,
    *,
    dilate_px: int = 0,
    blur_sigma: float = 0.0,
    expand_frac: float = 0.0,
) -> np.ndarray:
    """
    Маска для моделей inpaint по маске (Bria / Flux / SDXL / WAN): белое = зона перерисовки.
    Узкая маска заставляет новый объект «втискиваться» в силуэт старого — дилатация даёт запас.

    ``expand_frac`` — доля от min(высота, ширина) ROI: эффективный радиус дилатации не меньше
    ``round(expand_frac * min(h,w))`` (масштабируется после апскейла ROI). Склейка в кадр идёт по
    исходной SAM-маске, не по этой маске.
    """
    m = mask_gray
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.uint8, copy=False)
    h, w = int(m.shape[0]), int(m.shape[1])
    eff = int(max(0, dilate_px))
    if expand_frac > 0:
        eff = max(eff, int(round(min(h, w) * float(expand_frac))))
    if eff <= 0 and blur_sigma <= 0:
        return m
    return _preprocess_mask_for_genapi(m, eff, blur_sigma)


def _flux_unletterbox_result(gen_pil: Image.Image, meta: FluxLetterboxMeta) -> Image.Image:
    """Вырезает область ROI на холсте и возвращает размер orig_w × orig_h."""
    gen_pil = gen_pil.convert("RGB")
    if gen_pil.size != (meta.tw, meta.th):
        gen_pil = gen_pil.resize((meta.tw, meta.th), Image.Resampling.LANCZOS)
    box = (
        meta.pad_x,
        meta.pad_y,
        meta.pad_x + meta.content_w,
        meta.pad_y + meta.content_h,
    )
    crop = gen_pil.crop(box)
    return crop.resize((meta.orig_w, meta.orig_h), Image.Resampling.LANCZOS)


def _ceil_to_multiple(n: int, m: int = 8) -> int:
    """Размеры для диффузии обычно кратны 8 (иначе 422 от валидатора)."""
    if m <= 1:
        return max(1, n)
    n = max(m, n)
    return ((n + m - 1) // m) * m


def _align_roi_pils_for_diffusion(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    multiple: int = 8,
    min_side: int = 0,
) -> Tuple[Image.Image, Image.Image, int, int]:
    """
    Подгоняет ROI и маску под одинаковые w×h, кратные `multiple`.
    Если `min_side` > 0 (Flux/SDXL на Gen-API: не меньше 512 по каждой оси) —
    равномерно увеличивает кроп, пока и ширина, и высота не станут >= min_side.
    """
    w, h = roi_img_pil.size
    if min_side > 0:
        s = max(min_side / float(w), min_side / float(h), 1.0)
        if s > 1.0:
            w = int(math.ceil(w * s))
            h = int(math.ceil(h * s))
            roi_img_pil = roi_img_pil.resize((w, h), Image.Resampling.LANCZOS)
            roi_mask_pil = roi_mask_pil.resize((w, h), Image.Resampling.NEAREST)
    nw, nh = _ceil_to_multiple(w, multiple), _ceil_to_multiple(h, multiple)
    if (nw, nh) != (w, h):
        roi_img_pil = roi_img_pil.resize((nw, nh), Image.Resampling.LANCZOS)
        roi_mask_pil = roi_mask_pil.resize((nw, nh), Image.Resampling.NEAREST)
    return roi_img_pil, roi_mask_pil, nw, nh


def _form_data_str(data: Dict[str, Union[str, int, float, bool]]) -> Dict[str, str]:
    """Gen-API multipart: все поля формы — строки; bool → «1»/«0» (как в типичном Laravel)."""
    out: Dict[str, str] = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = "1" if v else "0"
        else:
            out[k] = str(v)
    return out


def _post_genapi_network(
    url: str,
    api_key: str,
    data: Dict[str, Union[str, int, float, bool]],
    files: List[Tuple[str, Tuple[str, BytesIO, str]]],
    timeout: int = 120,
) -> dict:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.post(
        url,
        headers=headers,
        data=_form_data_str(data),
        files=files,
        timeout=timeout,
    )
    if not r.ok:
        try:
            detail: Any = r.json()
        except Exception:
            detail = r.text[:8000]
        raise RuntimeError(f"Gen-API HTTP {r.status_code} {url}: {detail}")
    return r.json()


def pil_to_bytes(
    pil_image: Image.Image, format: str = "JPEG", quality: int = 92
) -> BytesIO:
    buffer = BytesIO()
    if format.upper() == "JPEG" and pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    if format.upper() == "JPEG":
        pil_image.save(buffer, format=format, quality=int(quality))
    else:
        pil_image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def _pil_resize_max_side(pil_image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return pil_image
    w, h = pil_image.size
    m = max(w, h)
    if m <= max_side:
        return pil_image
    s = max_side / float(m)
    return pil_image.resize((int(round(w * s)), int(round(h * s))), Image.Resampling.LANCZOS)


def nano_banana_submit(
    roi_img_pil: Image.Image,
    roi_mask_pil: Optional[Image.Image],
    prompt: str,
    seed: int,
    api_key: str,
    resolution: str = "0.5K",
    reference_pil: Optional[Image.Image] = None,
    full_scene_pil: Optional[Image.Image] = None,
    api_jpeg_quality: int = 95,
) -> int:
    """
    Nano Banana 2: либо классика ROI+маска, либо (при full_scene_pil) сцена без маски:
    image_urls — (1) полная сцена, (2) кроп ROI, (3) опционально предыдущая генерация того же ROI.
    """
    q = int(api_jpeg_quality)
    files = []
    if full_scene_pil is not None:
        files.append(
            ("image_urls[]", ("scene.jpg", pil_to_bytes(full_scene_pil, quality=q), "image/jpeg")),
        )
    files.append(
        ("image_urls[]", ("roi_img.jpg", pil_to_bytes(roi_img_pil, quality=q), "image/jpeg")),
    )
    if roi_mask_pil is not None:
        files.append(
            ("image_urls[]", ("roi_mask.jpg", pil_to_bytes(roi_mask_pil, quality=q), "image/jpeg")),
        )
    if reference_pil is not None:
        files.append(
            ("image_urls[]", ("prev_gen.jpg", pil_to_bytes(reference_pil, quality=q), "image/jpeg"))
        )
    data = {"prompt": prompt, "seed": int(seed), "resolution": resolution}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.post(NANO_BANANA_URL, headers=headers, data=data, files=files, timeout=120)
    r.raise_for_status()
    j = r.json()
    rid = j.get("request_id")
    if rid is None:
        raise RuntimeError(f"nano-banana: нет request_id в ответе: {j}")
    return int(rid)


def bria_submit(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    prompt: str,
    seed: int,
    api_key: str,
    *,
    api_jpeg_quality: int = 95,
) -> int:
    # Именно такой multipart ожидает replace-item (см. gen-api.ru/function/replace-item/api).
    q = int(api_jpeg_quality)
    files = [
        ("image", ("roi_img.jpg", pil_to_bytes(roi_img_pil, quality=q), "image/jpeg")),
        ("mask", ("roi_mask.jpg", pil_to_bytes(roi_mask_pil, quality=q), "image/jpeg")),
    ]
    data = {"prompt": prompt, "seed": int(seed)}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.post(
        BRIA_REPLACE_ITEM_URL, headers=headers, data=data, files=files, timeout=120
    )
    r.raise_for_status()
    j = r.json()
    rid = j.get("request_id")
    if rid is None:
        raise RuntimeError(f"bria: нет request_id в ответе: {j}")
    return int(rid)


def bria_fibo_submit(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    prompt: str,
    seed: int,
    api_key: str,
    negative_prompt: str,
    api_jpeg_quality: int = 95,
) -> int:
    q = int(api_jpeg_quality)
    files = [
        ("image_url", ("roi_img.jpg", pil_to_bytes(roi_img_pil, quality=q), "image/jpeg")),
        ("mask_url", ("roi_mask.jpg", pil_to_bytes(roi_mask_pil, quality=q), "image/jpeg")),
    ]
    data = {"prompt": prompt, "seed": int(seed), "negative_prompt": negative_prompt, "model": "edit"}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.post(BRIA_FIBO_URL, headers=headers, data=data, files=files, timeout=120)
    r.raise_for_status()
    j = r.json()
    rid = j.get("request_id")
    if rid is None:
        raise RuntimeError(f"bria-fibo: нет request_id в ответе: {j}")
    return int(rid)


def flux_submit(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    prompt: str,
    seed: int,
    api_key: str,
    *,
    model: str = "inpainting ",
    num_inference_steps: int = 28,
    guidance_scale: float = 5.0,
    strength: float = 1.0,
    api_jpeg_quality: int = 95,
    translate_input: bool = False,
) -> Tuple[int, FluxLetterboxMeta]:
    """
    Gen-API Flux: только фиксированные разрешения холста (`FLUX_GENAPI_ALLOWED_SIZES`).
    ROI вписывается в холст (letterbox), после генерации нужен `_flux_unletterbox_result`.
    """
    tw, th = _flux_pick_allowed_size(roi_img_pil.size[0], roi_img_pil.size[1])
    img_canvas, mask_canvas, meta = _flux_letterbox_to_canvas(
        roi_img_pil, roi_mask_pil, tw, th
    )
    q = int(api_jpeg_quality)
    files = [
        ("image", ("roi_img.jpg", pil_to_bytes(img_canvas, quality=q), "image/jpeg")),
        ("mask", ("roi_mask.jpg", pil_to_bytes(mask_canvas, quality=q), "image/jpeg")),
    ]
    data: Dict[str, Union[str, int, float, bool]] = {
        "prompt": prompt,
        "seed": int(seed),
        "model": model,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": guidance_scale,
        "strength": strength,
        "translate_input": translate_input,
    }
    j = _post_genapi_network(FLUX_NETWORK_URL, api_key, data, files)
    rid = j.get("request_id")
    if rid is None:
        raise RuntimeError(f"flux: нет request_id в ответе: {j}")
    return int(rid), meta


def sdxl_submit(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    prompt: str,
    seed: int,
    api_key: str,
    *,
    negative_prompt: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    api_jpeg_quality: int = 95,
    translate_input: bool = False,
    upgrade_prompt: bool = False,
) -> int:
    """Gen-API SDXL: inpainting — `image_url`/`mask_url`, размеры кратны 8."""
    roi_img_pil, roi_mask_pil, nw, nh = _align_roi_pils_for_diffusion(
        roi_img_pil, roi_mask_pil, multiple=8, min_side=512
    )
    q = int(api_jpeg_quality)
    files = [
        ("image", ("roi_img.jpg", pil_to_bytes(roi_img_pil, quality=q), "image/jpeg")),
        ("mask", ("roi_mask.jpg", pil_to_bytes(roi_mask_pil, quality=q), "image/jpeg")),
    ]
    data: Dict[str, Union[str, int, float, bool]] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "width": nw,
        "height": nh,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": guidance_scale,
        "num_outputs": 1,
        "translate_input": translate_input,
        "upgrade_prompt": upgrade_prompt,
    }
    j = _post_genapi_network(SDXL_NETWORK_URL, api_key, data, files)
    rid = j.get("request_id")
    if rid is None:
        raise RuntimeError(f"sdxl: нет request_id в ответе: {j}")
    return int(rid)


def _extract_image_url_from_genapi_job(j: dict) -> Optional[str]:
    res: Any = j.get("result")
    if isinstance(res, list) and res:
        u = res[0]
        if isinstance(u, str) and u.startswith("http"):
            return u
        return None
    if isinstance(res, dict):
        for kk in ("0", 0, "1", "url", "image"):
            if kk in res:
                v = res[kk]
                if isinstance(v, str) and v.startswith("http"):
                    return v
                if isinstance(v, dict) and "url" in v:
                    u = v["url"]
                    if isinstance(u, str) and u.startswith("http"):
                        return u
        for v in res.values():
            if isinstance(v, str) and v.startswith("http"):
                return v
            if isinstance(v, dict) and "url" in v:
                u = v["url"]
                if isinstance(u, str) and u.startswith("http"):
                    return u
    fr = j.get("full_response")
    if isinstance(fr, dict):
        for kk in ("0", 0, "1", 1):
            if kk in fr:
                v = fr[kk]
                if isinstance(v, dict) and "url" in v:
                    u = v["url"]
                    if isinstance(u, str) and u.startswith("http"):
                        return u
                if isinstance(v, str) and v.startswith("http"):
                    return v
    return None


def _result_is_transient_unavailable(j: dict) -> bool:
    """Сообщения сервиса «модель недоступна, попробуйте позже» — повторить опрос."""
    res = j.get("result")
    if isinstance(res, list) and res and isinstance(res[0], str):
        t = res[0].lower()
        if any(
            x in t
            for x in (
                "недоступ",
                "unavailable",
                "попозже",
                "later",
                "try again",
                "retry",
                "overload",
                "перегруз",
            )
        ):
            return True
    if isinstance(res, str) and any(
        x in res.lower() for x in ("недоступ", "unavailable", "попозже", "try again")
    ):
        return True
    return False


def wait_request_image_url(
    request_id: int,
    api_key: str,
    poll_interval: float = 2.0,
    timeout_sec: float = 600.0,
) -> str:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    url = f"https://api.gen-api.ru/api/v1/request/get/{request_id}"
    t0 = time.time()
    st: Optional[str] = None
    transient_failures = 0
    while time.time() - t0 < timeout_sec:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        j = r.json()
        st = j.get("status")
        if st == "success":
            out = _extract_image_url_from_genapi_job(j)
            if out is not None:
                return out
            # success без URL ещё (например только structured_prompt) — ждём дальше
            time.sleep(poll_interval)
            continue
        if st in ("failed", "error", "canceled", "cancelled"):
            if _result_is_transient_unavailable(j) and transient_failures < 24:
                transient_failures += 1
                time.sleep(max(poll_interval, 5.0))
                continue
            err_txt = j.get("result")
            if isinstance(err_txt, list) and err_txt and isinstance(err_txt[0], str):
                raise RuntimeError(
                    f"Модель: статус {st}: {err_txt[0]} (полный ответ: {j})"
                )
            raise RuntimeError(f"Модель: статус {st}: {j}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Модель: таймаут {timeout_sec}s для request_id={request_id}, статус {st}")


def download_image_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def _strict_scene_preservation_suffix() -> str:
    """Общие ограничения: не трогать фон и стиль сцены вне маски."""
    return (
        " CRITICAL — Inpainting only: change pixels strictly INSIDE the white mask region. "
        "Outside the mask, the output MUST be pixel-identical to image (1): same background, table, hands, props, "
        "and every unmasked detail. Do NOT restyle, recolor, relight, denoise, sharpen, or recompose the whole crop. "
        "Do NOT replace the scene, change art direction, or \"improve\" the image globally. "
        "The new object must match the existing scene: same white balance, exposure, film/video grain, "
        "compression noise, depth of field, and color grading as the unmasked parts of (1). "
        "Seam only at the mask boundary; keep continuity with adjacent pixels."
    )


def build_replace_prompt(old_object: str, new_object: str) -> str:
    return (
        f"In image (1), replace the {old_object} with {new_object}. "
        "The new object must sit naturally in the masked area only. Photorealistic, consistent with the photo. "
        + _strict_scene_preservation_suffix()
    )


def build_nano_scene_replace_prompt(
    old_object: str,
    new_object: str,
    *,
    has_prev_generation: bool,
) -> str:
    """
    Nano без маски: (1) полная сцена, (2) кроп ROI, (3) опционально предыдущий кроп генерации.
    """
    base = (
        f"Image (1) is the full frame for context. Image (2) is the zoomed region to re-render. "
        f"In (2), replace the {old_object} with {new_object}. "
        "Photorealistic: match lighting, color grading, shadows, and any glossy or mirror reflection on the surface — "
        "the new object must show a plausible reflection consistent with the scene, same as the original would. "
        "Output must match image (2) dimensions and framing exactly (same crop) for seamless compositing into the video. "
        "Keep everything outside the object and its reflection in (2) aligned with (1); do not restyle the whole frame."
    )
    if has_prev_generation:
        base += (
            " Image (3) is the previous frame's generated crop for this segment — keep the same object identity "
            "(colors, materials, logos, proportions) while updating pose, illumination, and reflections for this frame."
        )
    return base


def build_bria_prompt(new_object: str) -> str:
    """
    Bria через Gen-API replace-item: длинные промпты с детальными ограничениями часто дают
    error «Параметры сформированы некорректно» — у модели жёсткий лимит/валидация текста.
    Держим короткую формулировку.
    """
    return (
        f"{new_object}. "
        "Photorealistic, match scene lighting and colors. "
        "Edit only the masked area."
    )


def _maybe_upscale_roi(
    crop_bgr: np.ndarray,
    mask_gray: np.ndarray,
    min_side: int,
    max_upscale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Увеличивает ROI до min_side по короткой стороне (не больше max_upscale раз)."""
    ch, cw = crop_bgr.shape[:2]
    mside = min(ch, cw)
    if min_side <= 0 or mside >= min_side:
        return crop_bgr, mask_gray
    s = min(float(min_side) / float(mside), max_upscale)
    nh, nw = int(round(ch * s)), int(round(cw * s))
    crop_bgr = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    mask_gray = cv2.resize(mask_gray, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return crop_bgr, mask_gray


def call_inpaint_crop(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    padded_bbox: Tuple[int, int, int, int],
    *,
    api_key: str,
    new_object_prompt: str,
    old_object_prompt: str = "",
    seed: int = 42,
    resolution: str = "0.5K",
    poll_interval: float = 2.0,
    timeout_sec: float = 600.0,
    rectangular_mask: bool = False,
    reference_pil: Optional[Image.Image] = None,
    roi_min_side_for_api: int = 0,
    roi_max_upscale: float = 2.5,
    api_jpeg_quality: int = 95,
    bria_backend: str = "replace_item",
    negative_prompt: str = "",
    nano_scene_max_side: int = 1920,
    nano_send_mask: bool = False,
    flux_model: str = "ultra",
    flux_num_inference_steps: int = 28,
    flux_guidance_scale: float = 5.0,
    flux_strength: float = 1.0,
    flux_translate_input: bool = False,
    sdxl_num_inference_steps: int = 25,
    sdxl_guidance_scale: float = 7.5,
    sdxl_translate_input: bool = False,
    sdxl_upgrade_prompt: bool = False,
    inner_composite_bbox: Optional[Tuple[int, int, int, int]] = None,
    genapi_mask_dilate_px: int = 0,
    genapi_mask_blur_sigma: float = 0.0,
    genapi_mask_expand_frac: float = 0.0,
    nano_align_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Inpaint bbox region via Nano Banana 2, Bria, Bria Fibo, Flux или SDXL (Gen-API). Returns (result_crop_bgr, bbox).

    reference_pil: для Nano — предыдущий успешный кроп генерации (цепочка идентичности).
    nano_scene_max_side: даунскейл полного кадра для Gen-API (Nano), 0 = без даунскейла.
    nano_send_mask: для Nano — если True, в API уходит и маска (устаревший режим inpaint).
    roi_min_side_for_api: если >0 и короткая сторона ROI меньше — апскейл перед отправкой в API.
    bria_backend: при old_object_prompt==\"\": \"replace_item\" | \"fibo\" | \"flux\" | \"sdxl\" (маска SAM в ROI).
    negative_prompt: для bria_fibo и sdxl (например cfg.negative_prompt).
    padded_bbox: кроп для API (может быть шире compositing-области).
    inner_composite_bbox: если задан — область склейки (обычно «узкий» pb); из ответа API вырезается
        соответствующий фрагмент и приводится к размеру inner. Если None — совпадает с padded_bbox.
    genapi_mask_dilate_px / genapi_mask_blur_sigma: только для маски в API (мягче край объекта).
    genapi_mask_expand_frac: доля от min(h,w) ROI — минимальный радиус дилатации маски для API
        (после апскейла); даёт модели больше места, чем узкий силуэт SAM. 0 = только dilate_px.
    nano_align_fn: для Nano (old_object_prompt не пустой): после скачивания вызывается
        ``fn(gen_bgr_HxW, target_w, target_h) -> gen_bgr`` — tight crop по объекту + resize под ROI,
        вместо растягивания всего ответа API на весь ROI.
    """
    fh, fw = frame_bgr.shape[:2]
    api_x1, api_y1, api_x2, api_y2 = padded_bbox
    api_x1, api_y1 = max(0, api_x1), max(0, api_y1)
    api_x2, api_y2 = min(fw, api_x2), min(fh, api_y2)
    if api_x2 <= api_x1 or api_y2 <= api_y1:
        raise ValueError("Invalid padded_bbox")

    if inner_composite_bbox is None:
        in_x1, in_y1, in_x2, in_y2 = api_x1, api_y1, api_x2, api_y2
    else:
        in_x1, in_y1, in_x2, in_y2 = inner_composite_bbox
        in_x1, in_y1 = max(0, in_x1), max(0, in_y1)
        in_x2, in_y2 = min(fw, in_x2), min(fh, in_y2)
    in_x1 = max(in_x1, api_x1)
    in_y1 = max(in_y1, api_y1)
    in_x2 = min(in_x2, api_x2)
    in_y2 = min(in_y2, api_y2)
    if in_x2 <= in_x1 or in_y2 <= in_y1:
        raise ValueError("inner_composite_bbox invalid relative to padded_bbox")

    aw0 = api_x2 - api_x1
    ah0 = api_y2 - api_y1

    crop_bgr = frame_bgr[api_y1:api_y2, api_x1:api_x2].copy()
    if old_object_prompt == "" and rectangular_mask:
        mask_gray = np.full((ah0, aw0), 255, dtype=np.uint8)
    else:
        m = mask_u8[api_y1:api_y2, api_x1:api_x2]
        if m.ndim == 2:
            mask_gray = m
        else:
            mask_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    crop_bgr, mask_gray = _maybe_upscale_roi(
        crop_bgr, mask_gray, roi_min_side_for_api, roi_max_upscale
    )

    mask_gray = mask_for_genapi_inpaint(
        mask_gray,
        dilate_px=genapi_mask_dilate_px,
        blur_sigma=genapi_mask_blur_sigma,
        expand_frac=genapi_mask_expand_frac,
    )

    nah, naw = int(crop_bgr.shape[0]), int(crop_bgr.shape[1])
    sx = naw / float(max(1, aw0))
    sy = nah / float(max(1, ah0))

    roi_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    roi_img_pil = Image.fromarray(roi_rgb)
    roi_mask_pil = Image.fromarray(mask_gray).convert("L")

    ref_submit: Optional[Image.Image] = None
    if reference_pil is not None:
        ref_submit = reference_pil.resize(roi_img_pil.size, Image.Resampling.LANCZOS)

    q = int(min(95, max(1, int(api_jpeg_quality))))
    flux_letterbox_meta: Optional[FluxLetterboxMeta] = None
    if old_object_prompt == "":
        prompt = build_bria_prompt(new_object_prompt)
        if bria_backend == "fibo":
            neg = negative_prompt.strip() if negative_prompt else "blurry, low quality, deformed"
            rid = bria_fibo_submit(
                roi_img_pil,
                roi_mask_pil,
                prompt,
                seed,
                api_key,
                negative_prompt=neg,
                api_jpeg_quality=q,
            )
        elif bria_backend == "flux":
            rid, flux_letterbox_meta = flux_submit(
                roi_img_pil,
                roi_mask_pil,
                prompt,
                seed,
                api_key,
                model=flux_model,
                num_inference_steps=flux_num_inference_steps,
                guidance_scale=flux_guidance_scale,
                strength=flux_strength,
                api_jpeg_quality=q,
                translate_input=flux_translate_input,
            )
        elif bria_backend == "sdxl":
            neg = negative_prompt.strip() if negative_prompt else (
                "blurry, low quality, deformed, watermark, text"
            )
            rid = sdxl_submit(
                roi_img_pil,
                roi_mask_pil,
                prompt,
                seed,
                api_key,
                negative_prompt=neg,
                num_inference_steps=sdxl_num_inference_steps,
                guidance_scale=sdxl_guidance_scale,
                api_jpeg_quality=q,
                translate_input=sdxl_translate_input,
                upgrade_prompt=sdxl_upgrade_prompt,
            )
        else:
            rid = bria_submit(
                roi_img_pil,
                roi_mask_pil,
                prompt,
                seed,
                api_key,
                api_jpeg_quality=q,
            )
    else:
        prompt = build_nano_scene_replace_prompt(
            old_object_prompt,
            new_object_prompt,
            has_prev_generation=reference_pil is not None,
        )
        full_scene_pil = None
        if not nano_send_mask:
            full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            full_scene_pil = _pil_resize_max_side(
                Image.fromarray(full_rgb), int(nano_scene_max_side)
            )
        mask_for_api = roi_mask_pil if nano_send_mask else None
        rid = nano_banana_submit(
            roi_img_pil,
            mask_for_api,
            prompt,
            seed,
            api_key,
            resolution=resolution,
            reference_pil=ref_submit,
            full_scene_pil=full_scene_pil,
            api_jpeg_quality=q,
        )
    out_url = wait_request_image_url(
        rid, api_key, poll_interval=poll_interval, timeout_sec=timeout_sec
    )
    gen = download_image_url(out_url)
    if flux_letterbox_meta is not None:
        gen = _flux_unletterbox_result(gen, flux_letterbox_meta)
    inner_w = in_x2 - in_x1
    inner_h = in_y2 - in_y1
    if nano_align_fn is not None and old_object_prompt != "":
        gen_bgr = cv2.cvtColor(np.array(gen.convert("RGB")), cv2.COLOR_RGB2BGR)
        gen_bgr = nano_align_fn(gen_bgr, inner_w, inner_h)
        return gen_bgr, (in_x1, in_y1, in_x2, in_y2)
    if gen.size != (naw, nah):
        gen = gen.resize((naw, nah), Image.Resampling.LANCZOS)
    uix1 = int(math.floor((in_x1 - api_x1) * sx))
    uiy1 = int(math.floor((in_y1 - api_y1) * sy))
    uix2 = int(math.ceil((in_x2 - api_x1) * sx))
    uiy2 = int(math.ceil((in_y2 - api_y1) * sy))
    uix1 = max(0, min(uix1, naw - 1))
    uiy1 = max(0, min(uiy1, nah - 1))
    uix2 = max(uix1 + 1, min(uix2, naw))
    uiy2 = max(uiy1 + 1, min(uiy2, nah))
    gen = gen.crop((uix1, uiy1, uix2, uiy2))
    gen = gen.resize((inner_w, inner_h), Image.Resampling.LANCZOS)
    gen_bgr = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
    return gen_bgr, (in_x1, in_y1, in_x2, in_y2)
