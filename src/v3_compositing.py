"""Склейка ROI с кадром — как в video_changer_colab (alpha_insert_smooth)."""

from __future__ import annotations

import cv2
import numpy as np


def to_mask255(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    return m


def dilate_mask_u8(mask255: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask255
    k = np.ones((px, px), np.uint8)
    return cv2.dilate(mask255.astype(np.uint8), k, iterations=1)


def dilate_mask_asymmetric(
    mask255: np.ndarray, px_h: int, px_v_top: int, px_v_bottom: int
) -> np.ndarray:
    if px_h <= 0 and px_v_top <= 0 and px_v_bottom <= 0:
        return mask255
    kh = px_h * 2 + 1
    kv = px_v_top + px_v_bottom + 1
    kernel = np.zeros((kv, kh), np.uint8)
    kernel[: px_v_top + 1, :] = 1
    kernel[px_v_top:, kh // 2] = 1
    return cv2.dilate(mask255.astype(np.uint8), kernel, iterations=1)


def refine_mask(mask: np.ndarray, dilate_px: int = 9, feather_px: int = 11):
    hard = to_mask255(mask)
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        hard = cv2.dilate(hard, k, iterations=1)
    soft = hard.copy()
    if feather_px > 0:
        ks = feather_px * 2 + 1
        soft = cv2.GaussianBlur(soft, (ks, ks), 0)
    return hard, soft


def poisson_insert_roi(
    base_img: np.ndarray,
    roi_gen: np.ndarray,
    roi_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    h, w = base_img.shape[:2]
    src = base_img.copy()
    src[y1:y2, x1:x2] = roi_gen
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = roi_mask
    if mask_full.max() == 0:
        return base_img
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
    out = cv2.seamlessClone(src_bgr, dst_bgr, mask_full, center, cv2.NORMAL_CLONE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def alpha_insert_smooth(
    base_img: np.ndarray,
    roi_gen: np.ndarray,
    roi_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    blend_dilate_px: int = 12,
    feather_ks: int = 63,
    edge_band_ks: int = 11,
    use_poisson: bool = False,
    add_patch_noise: bool = False,
) -> np.ndarray:
    """Дилатация маски, edge-band matching, широкое размытие альфы."""
    base = base_img.copy().astype(np.float32)
    target = base[y1:y2, x1:x2].astype(np.float32)
    patch = roi_gen.astype(np.float32)
    rm = roi_mask.astype(np.uint8)
    if blend_dilate_px > 0:
        rm = dilate_mask_u8(rm, blend_dilate_px)
    eb = max(3, int(edge_band_ks) | 1)
    edge_band = cv2.dilate(rm, np.ones((eb, eb), np.uint8), 1) - rm
    edge_band = edge_band.astype(bool)
    if edge_band.sum() > 0:
        src_edge_mean = patch[edge_band].mean()
        dst_edge_mean = target[edge_band].mean()
        patch += dst_edge_mean - src_edge_mean
    if add_patch_noise:
        noise = target - cv2.GaussianBlur(target, (7, 7), 0)
        noise_std = float(noise.std()) + 1e-6
        patch += np.random.normal(0, noise_std, patch.shape)
    target_blur = cv2.Laplacian(target, cv2.CV_32F).var()
    patch_blur = cv2.Laplacian(patch, cv2.CV_32F).var()
    if patch_blur > target_blur:
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
    if use_poisson:
        return poisson_insert_roi(
            base_img,
            np.clip(roi_gen, 0, 255).astype(np.uint8),
            rm,
            x1,
            y1,
            x2,
            y2,
        )
    fk = max(3, int(feather_ks) | 1)
    m = cv2.GaussianBlur(rm.astype(np.float32), (fk, fk), 0)
    alpha = np.clip(m / 255.0, 0, 1)[..., None]
    blended = patch * alpha + target * (1.0 - alpha)
    base[y1:y2, x1:x2] = blended
    return np.clip(base, 0, 255).astype(np.uint8)


def composite_roi_simple_bgr(
    frame_bgr: np.ndarray,
    gen_crop_bgr: np.ndarray,
    mask_full_u8: np.ndarray,
    pb: tuple,
    *,
    blend_mode: str = "mask",
    mask_dilate: int = 13,
    mask_cover_extra_dilate: int = 10,
    edge_feather_px: int = 3,
) -> np.ndarray:
    """
    Минимальная склейка: альфа по маске без edge-matching, широкого размытия и «улучшений» из colab.
    edge_feather_px: узкое размытие только на границе (0 = жёсткая маска).
    """
    x1, y1, x2, y2 = pb
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr.copy()

    hc, wc = y2 - y1, x2 - x1
    if gen_crop_bgr.shape[0] != hc or gen_crop_bgr.shape[1] != wc:
        gen_crop_bgr = cv2.resize(
            gen_crop_bgr, (wc, hc), interpolation=cv2.INTER_LANCZOS4
        )

    out = frame_bgr.copy()
    if blend_mode == "full_rect":
        roi_hard = np.full((hc, wc), 255, dtype=np.uint8)
    else:
        hard = to_mask255(mask_full_u8)
        if mask_dilate > 0:
            hard = cv2.dilate(
                hard, np.ones((mask_dilate, mask_dilate), np.uint8), iterations=1
            )
        if mask_cover_extra_dilate > 0:
            hard = dilate_mask_u8(hard, mask_cover_extra_dilate)
        roi_hard = hard[y1:y2, x1:x2].copy()

    if edge_feather_px > 0:
        k = edge_feather_px * 2 + 1
        alpha = (
            cv2.GaussianBlur(roi_hard.astype(np.float32), (k, k), 0) / 255.0
        )
    else:
        alpha = roi_hard.astype(np.float32) / 255.0
    alpha = np.clip(alpha[..., None], 0.0, 1.0)

    orig_crop = out[y1:y2, x1:x2].astype(np.float32)
    patch = gen_crop_bgr.astype(np.float32)
    blended = alpha * patch + (1.0 - alpha) * orig_crop
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def composite_roi_bgr(
    frame_bgr: np.ndarray,
    gen_crop_bgr: np.ndarray,
    mask_full_u8: np.ndarray,
    pb: tuple,
    *,
    blend_mode: str = "mask",
    mask_dilate: int = 13,
    mask_feather: int = 11,
    mask_cover_extra_dilate: int = 10,
    small_mask_area_ratio: float = 0.002,
    small_mask_boost_dilate: int = 8,
    blend_mask_dilate_px: int = 14,
    blend_dilate_px_bottom: int = 2,
    composite_feather_ks: int = 48,
    use_poisson_blend: bool = False,
    add_patch_noise: bool = False,
) -> np.ndarray:
    """
    blend_mode:
      - \"mask\" — маска из SAM, refine + дилаты (как основной ноутбук)
      - \"full_rect\" — весь ROI белый (после полнопрямоугольной Bria-генерации)
    """
    x1, y1, x2, y2 = pb
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr.copy()

    hc, wc = y2 - y1, x2 - x1
    if gen_crop_bgr.shape[0] != hc or gen_crop_bgr.shape[1] != wc:
        gen_crop_bgr = cv2.resize(gen_crop_bgr, (wc, hc), interpolation=cv2.INTER_LANCZOS4)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.cvtColor(gen_crop_bgr, cv2.COLOR_BGR2RGB)

    if blend_mode == "full_rect":
        roi_hard_blended = np.full((hc, wc), 255, dtype=np.uint8)
    else:
        hard_mask, _ = refine_mask(mask_full_u8, dilate_px=mask_dilate, feather_px=mask_feather)
        if mask_cover_extra_dilate > 0:
            hard_mask = dilate_mask_u8(hard_mask, mask_cover_extra_dilate)
        _ar = float(hard_mask.sum()) / (255.0 * h * w + 1e-6)
        if _ar < small_mask_area_ratio and small_mask_boost_dilate > 0:
            hard_mask = dilate_mask_u8(hard_mask, small_mask_boost_dilate)
        roi_hard = hard_mask[y1:y2, x1:x2].copy()
        roi_hard_blended = dilate_mask_asymmetric(
            roi_hard,
            px_h=blend_mask_dilate_px,
            px_v_top=blend_mask_dilate_px,
            px_v_bottom=blend_dilate_px_bottom,
        )

    out_rgb = alpha_insert_smooth(
        frame_rgb,
        gen_rgb,
        roi_hard_blended,
        x1,
        y1,
        x2,
        y2,
        blend_dilate_px=0,
        feather_ks=composite_feather_ks,
        edge_band_ks=11,
        use_poisson=use_poisson_blend,
        add_patch_noise=add_patch_noise,
    )
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def clear_frame_dir(directory: str) -> None:
    """Удаляет jpg/png в папке отладочных кадров перед новым прогоном."""
    import glob
    import os

    os.makedirs(directory, exist_ok=True)
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for p in glob.glob(os.path.join(directory, pattern)):
            try:
                os.remove(p)
            except OSError:
                pass
