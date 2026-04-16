"""
Compatibility layer for video_changer_colab_v3.ipynb.

Maps notebook API to actual src/detector.py and src/sam_video.py.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from src.detector import detect_bbox, get_mask_from_bbox, load_grounding_dino
from src.sam_load import build_sam2_predictor
from src.sam_video import build_video_predictor, fill_mask_gaps, track_masks_video


def resolve_model_path(repo_root: str, path: str) -> str:
    """
    Prefer models/ under repo root (same layout as video_changer_colab.ipynb).
    """
    if os.path.isabs(path) and os.path.exists(path):
        return path
    base = os.path.basename(path)
    for folder in (os.path.join(repo_root, "models"), repo_root):
        cand = os.path.join(folder, base)
        if os.path.exists(cand):
            return cand
    # legacy: path relative to cwd
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(repo_root, "models", base)


def load_models(
    gdino_config: str,
    gdino_checkpoint: str,
    device: str = "cuda",
):
    """Notebook expects (model, processor). GroundingDINO has no separate processor."""
    cfg = gdino_config.strip() if gdino_config else None
    if not cfg:
        cfg = None
    model = load_grounding_dino(cfg, gdino_checkpoint, device=device)
    return model, None


def detect_object_gdino(
    model,
    processor,
    frame_bgr: np.ndarray,
    prompt: str,
    *,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """Returns (bbox, confidence) or None."""
    del processor
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detect_bbox(
        model,
        frame_rgb,
        prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    if result is None:
        return None
    bbox, _phrase, conf = result
    return bbox, conf


def load_sam2_video_pair(
    checkpoint: str,
    config: str,
    device: str = "cuda",
):
    """Returns (video_predictor, image_predictor) for SAM2 video + seed mask."""
    vp = build_video_predictor(checkpoint, config, device=device)
    ip = build_sam2_predictor(checkpoint, config, device=device)
    return vp, ip


def expand_bbox_for_reflection(
    mask_u8: np.ndarray,
    bbox: Tuple[int, int, int, int],
    frame_hw: Tuple[int, int],
    *,
    min_secondary_area_ratio: float = 0.04,
    max_vertical_gap_ratio: float = 3.0,
    min_x_overlap_ratio: float = 0.12,
) -> Tuple[int, int, int, int]:
    """
    Если маска даёт второй связный компонент под/рядом с основным (типичное отражение на столе),
    расширяет bbox, чтобы в API попало и отражение.
    """
    h, w = frame_hw[0], frame_hw[1]
    x1, y1, x2, y2 = bbox
    if mask_u8 is None:
        return bbox
    m = mask_u8
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    binm = (m > 127).astype(np.uint8)
    if int(binm.sum()) < 16:
        return bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 2:
        return bbox

    y1c, y2c = max(0, y1), min(h, y2)
    x1c, x2c = max(0, x1), min(w, x2)
    if y2c <= y1c or x2c <= x1c:
        roi_l = labels
    else:
        roi_l = labels[y1c:y2c, x1c:x2c]
    inside = roi_l[roi_l > 0]
    if inside.size == 0:
        return bbox
    main_lab = int(np.argmax(np.bincount(inside.flatten())))
    if main_lab <= 0:
        return bbox

    mx1 = int(stats[main_lab, cv2.CC_STAT_LEFT])
    my1 = int(stats[main_lab, cv2.CC_STAT_TOP])
    mxw = int(stats[main_lab, cv2.CC_STAT_WIDTH])
    myh = int(stats[main_lab, cv2.CC_STAT_HEIGHT])
    mx2, my2 = mx1 + mxw, my1 + myh
    main_area = float(stats[main_lab, cv2.CC_STAT_AREA])

    out_x1, out_y1, out_x2, out_y2 = x1, y1, x2, y2
    for lab in range(1, num):
        if lab == main_lab:
            continue
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < min_secondary_area_ratio * max(main_area, 1.0):
            continue
        lx = int(stats[lab, cv2.CC_STAT_LEFT])
        ly = int(stats[lab, cv2.CC_STAT_TOP])
        lw = int(stats[lab, cv2.CC_STAT_WIDTH])
        lh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        lx2, ly2 = lx + lw, ly + lh
        x_ov = float(min(mx2, lx2) - max(mx1, lx))
        if x_ov < min_x_overlap_ratio * float(bw):
            continue
        gap_below = max(0, ly - my2)
        gap_above = max(0, my1 - ly2)
        if gap_below > max_vertical_gap_ratio * bh and gap_above > max_vertical_gap_ratio * bh:
            continue
        cy_m = centroids[main_lab][1]
        cy_s = centroids[lab][1]
        if abs(cy_s - cy_m) > 2.2 * bh:
            continue
        out_x1 = min(out_x1, lx)
        out_y1 = min(out_y1, ly)
        out_x2 = max(out_x2, lx2)
        out_y2 = max(out_y2, ly2)

    out_x1 = max(0, min(w, out_x1))
    out_y1 = max(0, min(h, out_y1))
    out_x2 = max(0, min(w, out_x2))
    out_y2 = max(0, min(h, out_y2))
    if out_x2 <= out_x1 or out_y2 <= out_y1:
        return bbox
    return (out_x1, out_y1, out_x2, out_y2)


def track_object_video(
    video_predictor,
    image_predictor,
    frames_bgr: list[np.ndarray],
    bbox: Tuple[int, int, int, int],
    fill_gaps: int = 2,
    expand_reflection: bool = True,
) -> tuple[list[np.ndarray], list[Optional[Tuple[int, int, int, int]]]]:
    """
    Track object through frames. frames_bgr: BGR uint8.
    Returns (masks_list, bboxes_list) per frame.
    """
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    seed = frames_rgb[0]
    initial_mask = get_mask_from_bbox(image_predictor, seed, bbox)
    masks = track_masks_video(
        video_predictor,
        frames_rgb,
        initial_mask,
        initial_frame_idx=0,
    )
    masks = fill_mask_gaps(masks, max_gap=fill_gaps)

    fh, fw = frames_rgb[0].shape[0], frames_rgb[0].shape[1]
    bboxes: list[Optional[Tuple[int, int, int, int]]] = []
    for m in masks:
        if m is None or not m.any():
            bboxes.append(None)
            continue
        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            bboxes.append(None)
            continue
        bb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if expand_reflection:
            bb = expand_bbox_for_reflection(m, bb, (fh, fw))
        bboxes.append(bb)
    return masks, bboxes
