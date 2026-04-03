"""
Compatibility layer for video_changer_colab_v3.ipynb.

Maps notebook API to actual src/detector.py and src/sam_video.py.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from detector import detect_bbox, get_mask_from_bbox, load_grounding_dino
from sam_load import build_sam2_predictor
from sam_video import build_video_predictor, fill_mask_gaps, track_masks_video


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


def track_object_video(
    video_predictor,
    image_predictor,
    frames_bgr: list[np.ndarray],
    bbox: Tuple[int, int, int, int],
    fill_gaps: int = 2,
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

    bboxes: list[Optional[Tuple[int, int, int, int]]] = []
    for m in masks:
        if m is None or not m.any():
            bboxes.append(None)
            continue
        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            bboxes.append(None)
            continue
        bboxes.append(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        )
    return masks, bboxes
