"""Gen-API Nano Banana 2 — minimal client for video_changer_colab_v3.ipynb."""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Tuple, Optional

import cv2
import numpy as np
import requests
from PIL import Image

NANO_BANANA_URL = "https://api.gen-api.ru/api/v1/networks/nano-banana-2"
BRIA_REPLACE_ITEM_URL = "https://api.gen-api.ru/api/v1/functions/replace-item"
BRIA_FIBO_URL = "https://api.gen-api.ru/api/v1/networks/bria-fibo"


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
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Inpaint bbox region via Nano Banana 2, Bria Replace Item или Bria Fibo. Returns (result_crop_bgr, padded_bbox).

    reference_pil: для Nano — предыдущий успешный кроп генерации (цепочка идентичности).
    nano_scene_max_side: даунскейл полного кадра для Gen-API (Nano), 0 = без даунскейла.
    nano_send_mask: для Nano — если True, в API уходит и маска (устаревший режим inpaint).
    roi_min_side_for_api: если >0 и короткая сторона ROI меньше — апскейл перед отправкой в API.
    bria_backend: при old_object_prompt==\"\": \"replace_item\" | \"fibo\" (промпт как у Bria — build_bria_prompt).
    negative_prompt: для bria_fibo (например cfg.negative_prompt).
    """
    x1, y1, x2, y2 = padded_bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid padded_bbox")

    cw0 = x2 - x1
    ch0 = y2 - y1

    crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
    if old_object_prompt == "" and rectangular_mask:
        mask_gray = np.full((ch0, cw0), 255, dtype=np.uint8)
    else:
        m = mask_u8[y1:y2, x1:x2]
        if m.ndim == 2:
            mask_gray = m
        else:
            mask_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    crop_bgr, mask_gray = _maybe_upscale_roi(
        crop_bgr, mask_gray, roi_min_side_for_api, roi_max_upscale
    )

    roi_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    roi_img_pil = Image.fromarray(roi_rgb)
    roi_mask_pil = Image.fromarray(mask_gray).convert("L")

    ref_submit: Optional[Image.Image] = None
    if reference_pil is not None:
        ref_submit = reference_pil.resize(roi_img_pil.size, Image.Resampling.LANCZOS)

    q = int(min(95, max(1, int(api_jpeg_quality))))
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
    gen = gen.resize((roi_img_pil.width, roi_img_pil.height), Image.Resampling.LANCZOS)
    gen_bgr = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
    gen_bgr = cv2.resize(gen_bgr, (cw0, ch0), interpolation=cv2.INTER_LANCZOS4)
    return gen_bgr, (x1, y1, x2, y2)
