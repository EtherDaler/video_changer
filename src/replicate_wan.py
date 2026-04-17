"""
Replicate API — WAN video inpainting (e.g. andreasjansson/wan-1.3b-inpaint).

Ожидаются поля input_video и mask_video (см. схему модели на replicate.com).
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Tuple

import cv2
import numpy as np
import urllib.request
from PIL import Image

from src.nano_genapi import mask_for_genapi_inpaint


def _frames_to_tmp_mp4(
    frames_bgr: list[np.ndarray],
    size_xy: Tuple[int, int],
    fps: float = 16.0,
) -> str:
    """frames_bgr: список BGR кадров; size_xy: (width, height)."""
    w, h = size_xy
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not out.isOpened():
        os.unlink(path)
        raise RuntimeError("VideoWriter не открылся для временного mp4")
    for f in frames_bgr:
        if f.shape[1] != w or f.shape[0] != h:
            f = cv2.resize(f, (w, h))
        out.write(f)
    out.release()
    return path


def _download_video_first_frame_bgr(url: str, target_wh: Tuple[int, int]) -> np.ndarray:
    with urllib.request.urlopen(url) as r:
        data = r.read()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Не удалось прочитать кадр из ответа WAN")
        tw, th = target_wh
        if frame.shape[1] != tw or frame.shape[0] != th:
            frame = cv2.resize(frame, (tw, th))
        return frame
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def call_wan_inpaint_replicate(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    padded_bbox: Tuple[int, int, int, int],
    *,
    prompt: str,
    negative_prompt: str,
    model: str,
    seed: int,
    sampling_steps: int,
    genapi_mask_dilate_px: int = 0,
    genapi_mask_blur_sigma: float = 0.0,
    genapi_mask_expand_frac: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Один «чанк» из одного кадра: временные mp4 → Replicate WAN inpaint.
    Возвращает (result_crop_bgr, padded_bbox).
    """
    import replicate

    x1, y1, x2, y2 = padded_bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid padded_bbox")

    crop = frame_bgr[y1:y2, x1:x2].copy()
    m = mask_u8[y1:y2, x1:x2]
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    mh, mw = m.shape[:2]
    cw, ch = x2 - x1, y2 - y1
    if (mw, mh) != (cw, ch):
        m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)

    m = mask_for_genapi_inpaint(
        m,
        dilate_px=genapi_mask_dilate_px,
        blur_sigma=genapi_mask_blur_sigma,
        expand_frac=genapi_mask_expand_frac,
    )

    mask_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    size_xy = (cw, ch)

    vid_path = _frames_to_tmp_mp4([crop], size_xy)
    mask_path = _frames_to_tmp_mp4([mask_bgr], size_xy)

    try:
        with open(vid_path, "rb") as vf, open(mask_path, "rb") as mf:
            out = replicate.run(
                model,
                input={
                    "input_video": vf,
                    "mask_video": mf,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "sampling_steps": sampling_steps,
                    "seed": seed,
                },
            )
    finally:
        try:
            os.unlink(vid_path)
        except OSError:
            pass
        try:
            os.unlink(mask_path)
        except OSError:
            pass

    url = None
    if isinstance(out, str) and out.startswith("http"):
        url = out
    elif isinstance(out, (list, tuple)) and out:
        u = out[0]
        url = u if isinstance(u, str) else getattr(u, "url", None) or str(u)
    elif hasattr(out, "url"):
        u = out.url
        url = u() if callable(u) else u
    else:
        url = str(out)

    if not url or not str(url).startswith("http"):
        raise RuntimeError(f"Неожиданный ответ Replicate WAN: {out!r}")
    url = str(url)

    result_bgr = _download_video_first_frame_bgr(url, size_xy)
    return result_bgr, (x1, y1, x2, y2)
