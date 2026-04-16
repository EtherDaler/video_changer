"""
После генерации Nano Banana: Grounding DINO + SAM2 на выходе модели → маска нового объекта,
вырез с альфой и вставка поверх исходного кадра (фон из оригинала).

Режим кэша: один tight BGRA на сегмент (до shot cut / потери объекта) — дальше только геометрия по SAM,
без повторных вызовов API.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.detector import detect_bbox, get_mask_from_bbox
from src.v3_compositing import composite_roi_simple_bgr


def resolve_nano_cutout_grounding_prompt(cfg) -> str:
    """
    Текст для Grounding DINO на кропе ПОСЛЕ Nano (короткий, 1–3 слова на англ.).
    Приоритет: nano_cutout_grounding_prompt → устар. nano_cutout_detect_prompt → replace_prompt.
    """
    g = getattr(cfg, "nano_cutout_grounding_prompt", None)
    if g is not None and str(g).strip():
        return str(g).strip()
    legacy = getattr(cfg, "nano_cutout_detect_prompt", None)
    if legacy is not None and str(legacy).strip():
        return str(legacy).strip()
    return str(getattr(cfg, "replace_prompt", "object")).strip()


def _mask_to_bbox(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    m = np.asarray(mask_u8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.float32)
    if m.max() <= 1.0:
        binm = m > 0.5
    else:
        binm = m > 127.0
    ys, xs = np.where(binm)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_foreground_area_px(mold_gray: np.ndarray) -> float:
    m = np.asarray(mold_gray)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.float32)
    if m.max() <= 1.0:
        return float((m > 0.5).sum())
    return float((m > 127.0).sum())


def _effective_target_size_and_center(
    mold_roi: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """
    По маске исходного объекта: площадь A_orig, bbox tw×th.
    Эффективный целевой прямоугольник Weff×Teff с тем же соотношением сторон, что и bbox, и площадью A_orig:
    Weff*Teff = A_orig, Weff/Teff = tw/th.
    Центр — центроид маски (в координатах ROI-кропа).
    """
    mold = mold_roi
    if mold.ndim == 3:
        mold = cv2.cvtColor(mold, cv2.COLOR_BGR2GRAY)
    bb = _mask_to_bbox(mold)
    if bb is None:
        return None
    tx1, ty1, tx2, ty2 = bb
    tw = float(tx2 - tx1)
    th = float(ty2 - ty1)
    if tw < 1.0 or th < 1.0:
        return None
    a_orig = _mask_foreground_area_px(mold)
    if a_orig < 1.0:
        a_orig = tw * th
    # Сохраняем aspect bbox, подгоняем площадь к A_orig
    weff = float(np.sqrt(a_orig * (tw / th)))
    teff = float(np.sqrt(a_orig * (th / tw)))
    m2 = mold.astype(np.float32)
    if m2.max() <= 1.0:
        ys, xs = np.where(m2 > 0.5)
    else:
        ys, xs = np.where(m2 > 127.0)
    if len(xs) == 0:
        cx = (tx1 + tx2) * 0.5
        cy = (ty1 + ty2) * 0.5
    else:
        cx = float(xs.mean())
        cy = float(ys.mean())
    return weff, teff, cx, cy


def _uniform_scale_center_paste(
    pw: int,
    ph: int,
    mold_roi: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """
    Единый масштаб s (без искажения), чтобы прямоугольник pw×ph после масштаба
    покрывал целевой Weff×Teff (площадь = площади маски, стороны как у bbox маски).
    Возвращает (s, cx, cy) в координатах ROI.
    """
    et = _effective_target_size_and_center(mold_roi)
    if et is None:
        return None
    weff, teff, cx, cy = et
    if pw < 1 or ph < 1:
        return None
    s = max(weff / float(pw), teff / float(ph))
    if s <= 0 or not np.isfinite(s):
        return None
    return s, cx, cy


def _feather_alpha(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = px * 2 + 1
    return cv2.GaussianBlur(alpha.astype(np.float32), (k, k), 0)


def mask_new_object_on_generated_crop(
    gen_crop_bgr: np.ndarray,
    gdino_model,
    sam_image_predictor,
    text_prompt: str,
    *,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> Optional[np.ndarray]:
    """
    Ищет новый объект по тексту на кропе от Nano, возвращает маску 0..255 того же размера.
    """
    rgb = cv2.cvtColor(gen_crop_bgr, cv2.COLOR_BGR2RGB)
    hit = detect_bbox(
        gdino_model,
        rgb,
        text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    if hit is None:
        return None
    bbox, _, _ = hit
    return get_mask_from_bbox(sam_image_predictor, rgb, bbox)


def mask_new_object_on_generated_crop_retry(
    gen_crop_bgr: np.ndarray,
    gdino_model,
    sam_image_predictor,
    prompt: str,
    cfg,
) -> Optional[np.ndarray]:
    """
    Несколько попыток DINO с порогами: на новых сценах/ином свете строгие пороги часто дают пропуск.
    """
    bt0 = float(getattr(cfg, "nano_cutout_box_threshold", cfg.box_threshold))
    tt0 = float(getattr(cfg, "nano_cutout_text_threshold", cfg.text_threshold))
    pairs: list[Tuple[float, float]] = [(bt0, tt0)]
    if getattr(cfg, "nano_cutout_retry_relaxed", True):
        pairs.append((max(0.22, bt0 * 0.55), max(0.2, tt0 * 0.55)))
        pairs.append((max(0.18, bt0 * 0.4), max(0.16, tt0 * 0.4)))
    for bt, tt in pairs:
        m = mask_new_object_on_generated_crop(
            gen_crop_bgr,
            gdino_model,
            sam_image_predictor,
            prompt,
            box_threshold=bt,
            text_threshold=tt,
            device=cfg.device,
        )
        if m is not None:
            return m
    return None


def _paste_scaled_cutout(
    frame_bgr: np.ndarray,
    gen_crop_bgr: np.ndarray,
    mask_new_roi: np.ndarray,
    mask_old_roi: np.ndarray,
    pb: Tuple[int, int, int, int],
    *,
    feather_px: int,
) -> np.ndarray:
    """
    Масштабирует вырез нового объекта с **единым** коэффициентом (без растяжения по осям):
    целевая площадь и соотношение сторон задаются по **маске** исходного объекта (см. _uniform_scale_center_paste).
    """
    x1, y1, x2, y2 = pb
    H, W = frame_bgr.shape[:2]

    mold = mask_old_roi
    if mold.ndim == 3:
        mold = cv2.cvtColor(mold, cv2.COLOR_BGR2GRAY)

    mnew = mask_new_roi.astype(np.float32)
    if mnew.max() <= 1.0:
        mnew = mnew * 255.0
    src = _mask_to_bbox(mnew.astype(np.uint8))
    if src is None:
        return frame_bgr
    bx1, by1, bx2, by2 = src
    pw, ph = bx2 - bx1, by2 - by1
    if pw < 2 or ph < 2:
        return frame_bgr

    patch = gen_crop_bgr[by1:by2, bx1:bx2].astype(np.float32)
    a = (mnew[by1:by2, bx1:bx2] / 255.0).clip(0.0, 1.0)
    a = _feather_alpha(a, feather_px)

    sc = _uniform_scale_center_paste(pw, ph, mold)
    if sc is None:
        return frame_bgr
    scale, cx, cy = sc
    nw = max(1, int(round(pw * scale)))
    nh = max(1, int(round(ph * scale)))
    patch_r = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    a_r = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)

    px0 = int(round(cx - nw * 0.5))
    py0 = int(round(cy - nh * 0.5))

    px = x1 + px0
    py = y1 + py0
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    alpha_canvas = np.zeros((H, W), dtype=np.float32)

    py1, py2 = max(0, py), min(H, py + nh)
    px1, px2 = max(0, px), min(W, px + nw)
    sy1 = py1 - py
    sy2 = sy1 + (py2 - py1)
    sx1 = px1 - px
    sx2 = sx1 + (px2 - px1)

    canvas[py1:py2, px1:px2] = patch_r[sy1:sy2, sx1:sx2]
    alpha_canvas[py1:py2, px1:px2] = a_r[sy1:sy2, sx1:sx2]

    base = frame_bgr.astype(np.float32)
    a3 = np.clip(alpha_canvas[..., None], 0.0, 1.0)
    out = base * (1.0 - a3) + canvas * a3
    return np.clip(out, 0, 255).astype(np.uint8)


def _paste_tight_bgra(
    frame_bgr: np.ndarray,
    tight_bgra: np.ndarray,
    mask_old_roi: np.ndarray,
    pb: Tuple[int, int, int, int],
    *,
    feather_px: int,
) -> np.ndarray:
    """tight_bgra: H×W×4, BGRA; масштаб единый, цель по площади/аспекту маски (см. _uniform_scale_center_paste)."""
    x1, y1, x2, y2 = pb
    H, W = frame_bgr.shape[:2]

    mold = mask_old_roi
    if mold.ndim == 3:
        mold = cv2.cvtColor(mold, cv2.COLOR_BGR2GRAY)

    patch = tight_bgra[:, :, :3].astype(np.float32)
    a = (tight_bgra[:, :, 3].astype(np.float32) / 255.0).clip(0.0, 1.0)
    a = _feather_alpha(a, feather_px)
    ph, pw = patch.shape[:2]
    if pw < 2 or ph < 2:
        return frame_bgr

    sc = _uniform_scale_center_paste(pw, ph, mold)
    if sc is None:
        return frame_bgr
    scale, cx, cy = sc
    nw = max(1, int(round(pw * scale)))
    nh = max(1, int(round(ph * scale)))
    patch_r = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    a_r = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)

    px0 = int(round(cx - nw * 0.5))
    py0 = int(round(cy - nh * 0.5))

    px = x1 + px0
    py = y1 + py0
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    alpha_canvas = np.zeros((H, W), dtype=np.float32)

    py1, py2 = max(0, py), min(H, py + nh)
    px1, px2 = max(0, px), min(W, px + nw)
    sy1 = py1 - py
    sy2 = sy1 + (py2 - py1)
    sx1 = px1 - px
    sx2 = sx1 + (px2 - px1)

    canvas[py1:py2, px1:px2] = patch_r[sy1:sy2, sx1:sx2]
    alpha_canvas[py1:py2, px1:px2] = a_r[sy1:sy2, sx1:sx2]

    base = frame_bgr.astype(np.float32)
    a3 = np.clip(alpha_canvas[..., None], 0.0, 1.0)
    out = base * (1.0 - a3) + canvas * a3
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_tight_rgba_cutout(
    gen_crop_bgr: np.ndarray,
    gdino_model,
    sam_image_predictor,
    cfg,
) -> Optional[np.ndarray]:
    """
    DINO+SAM на кропе Nano → tight BGRA (альфа по маске).
    Возвращает None, если детекция не удалась.
    """
    prompt = resolve_nano_cutout_grounding_prompt(cfg)
    mask_new = mask_new_object_on_generated_crop_retry(
        gen_crop_bgr, gdino_model, sam_image_predictor, prompt, cfg
    )
    if mask_new is None:
        return None
    bb = _mask_to_bbox(mask_new.astype(np.uint8))
    if bb is None:
        return None
    bx1, by1, bx2, by2 = bb
    bgr = gen_crop_bgr[by1:by2, bx1:bx2].copy()
    a = mask_new[by1:by2, bx1:bx2].astype(np.float32) / 255.0
    feather = int(getattr(cfg, "nano_cutout_feather_px", 4))
    a = _feather_alpha(a, feather).clip(0.0, 1.0)
    a_u8 = (a * 255.0).astype(np.uint8)
    return np.dstack([bgr, a_u8])


def composite_cached_cutout_on_frame(
    frame_bgr: np.ndarray,
    tight_bgra: np.ndarray,
    mask_full_u8: np.ndarray,
    pb: Tuple[int, int, int, int],
    cfg,
) -> np.ndarray:
    """Вставка закэшированного tight BGRA на текущую позицию объекта (маска SAM)."""
    x1, y1, x2, y2 = pb
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr.copy()

    mfull = mask_full_u8
    if mfull.ndim == 3:
        mfull = cv2.cvtColor(mfull, cv2.COLOR_BGR2GRAY)
    mold_roi = mfull[y1:y2, x1:x2].copy()
    feather = int(getattr(cfg, "nano_cutout_feather_px", 4))
    return _paste_tight_bgra(
        frame_bgr, tight_bgra, mold_roi, pb, feather_px=feather
    )


def composite_nano_cutout_or_fallback(
    frame_bgr: np.ndarray,
    gen_crop_bgr: np.ndarray,
    mask_full_u8: np.ndarray,
    pb: Tuple[int, int, int, int],
    cfg,
    gdino_model,
    sam_image_predictor,
) -> np.ndarray:
    """
    Пытается cutout+вставку; при ошибке детекции — обычная склейка ROI по маске.
    """
    x1, y1, x2, y2 = pb
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr.copy()

    mfull = mask_full_u8
    if mfull.ndim == 3:
        mfull = cv2.cvtColor(mfull, cv2.COLOR_BGR2GRAY)
    mold_roi = mfull[y1:y2, x1:x2].copy()

    prompt = resolve_nano_cutout_grounding_prompt(cfg)

    mask_new = mask_new_object_on_generated_crop_retry(
        gen_crop_bgr, gdino_model, sam_image_predictor, prompt, cfg
    )
    if mask_new is None:
        return composite_roi_simple_bgr(
            frame_bgr,
            gen_crop_bgr,
            mask_full_u8,
            pb,
            blend_mode="mask",
            mask_dilate=cfg.mask_dilate,
            mask_cover_extra_dilate=cfg.mask_cover_extra_dilate,
            edge_feather_px=cfg.composite_edge_feather_px,
        )

    feather = int(getattr(cfg, "nano_cutout_feather_px", 4))
    out = _paste_scaled_cutout(
        frame_bgr,
        gen_crop_bgr,
        mask_new,
        mold_roi,
        pb,
        feather_px=feather,
    )
    return out
