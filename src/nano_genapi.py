"""Gen-API Nano Banana 2 — minimal client for video_changer_colab_v3.ipynb."""

from __future__ import annotations

import time
from io import BytesIO
from typing import Tuple, Optional

import cv2
import numpy as np
import requests
from PIL import Image

NANO_BANANA_URL = "https://api.gen-api.ru/api/v1/networks/nano-banana-2"
BRIA_REPLACE_ITEM_URL = "https://api.gen-api.ru/api/v1/functions/replace-item"


def pil_to_bytes(pil_image: Image.Image, format: str = "JPEG") -> BytesIO:
    buffer = BytesIO()
    if format.upper() == "JPEG" and pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def nano_banana_submit(
    roi_img_pil: Image.Image,
    roi_mask_pil: Image.Image,
    prompt: str,
    seed: int,
    api_key: str,
    resolution: str = "0.5K",
) -> int:
    files = [
        ("image_urls[]", ("roi_img.jpg", pil_to_bytes(roi_img_pil), "image/jpeg")),
        ("image_urls[]", ("roi_mask.jpg", pil_to_bytes(roi_mask_pil), "image/jpeg")),
    ]
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
    prev_img: Optional[Image.Image] = None,
) -> int:
    files = [
        ("image", ("roi_img.jpg", pil_to_bytes(roi_img_pil), "image/jpeg")),
        ("mask", ("roi_mask.jpg", pil_to_bytes(roi_mask_pil), "image/jpeg")),
    ]
    data = {"prompt": prompt, "seed": int(seed)}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.post(BRIA_REPLACE_ITEM_URL, headers=headers, data=data, files=files, timeout=120)
    r.raise_for_status()
    j = r.json()
    rid = j.get("request_id")
    if rid is None:
        raise RuntimeError(f"bria: нет request_id в ответе: {j}")
    return int(rid)


def wait_request_image_url(
    request_id: int,
    api_key: str,
    poll_interval: float = 2.0,
    timeout_sec: float = 600.0,
) -> str:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    url = f"https://api.gen-api.ru/api/v1/request/get/{request_id}"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        j = r.json()
        st = j.get("status")
        if st == "success":
            res = j.get("result")
            if isinstance(res, list) and res:
                return res[0]
            raise RuntimeError(f"success, но пустой result: {j}")
        if st in ("failed", "error", "canceled", "cancelled"):
            raise RuntimeError(f"Модель: статус {st}: {j}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Модель: таймаут {timeout_sec}s для request_id={request_id}, статус {st}")


def download_image_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def build_replace_prompt(old_object: str, new_object: str) -> str:
    return (
        f"Replace {old_object} to a {new_object}. "
        "New object should be in the mask box. Photorealistic. "
        "Match original scene lighting, perspective, depth of field, color and noise. "
        "Replace only the masked object and keep surroundings consistent."
    )

def build_draw_prompt(new_object: str) -> str:
    return f"Draw a {new_object} in the mask box. Photorealistic. Match original scene lighting, perspective, depth of field, color and noise."


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
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Inpaint bbox region via Nano Banana 2 or Bria Replace Item. Returns (result_crop_bgr, padded_bbox).
    """
    x1, y1, x2, y2 = padded_bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid padded_bbox")

    crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
    m = mask_u8[y1:y2, x1:x2]
    if m.ndim == 2:
        mask_gray = m
    else:
        mask_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    roi_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    roi_img_pil = Image.fromarray(roi_rgb)
    roi_mask_pil = Image.fromarray(mask_gray).convert("L")
    if old_object_prompt == "":    
        prompt = build_draw_prompt(new_object_prompt)
        rid = bria_submit(
            roi_img_pil, roi_mask_pil, prompt, seed, api_key
        )
    else:
        prompt = build_replace_prompt(old_object_prompt, new_object_prompt)
        rid = nano_banana_submit(
            roi_img_pil, roi_mask_pil, prompt, seed, api_key, resolution=resolution
        )
    out_url = wait_request_image_url(
        rid, api_key, poll_interval=poll_interval, timeout_sec=timeout_sec
    )
    gen = download_image_url(out_url)
    gen = gen.resize((roi_img_pil.width, roi_img_pil.height), Image.Resampling.LANCZOS)
    gen_bgr = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
    return gen_bgr, (x1, y1, x2, y2)
