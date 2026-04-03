"""
frame_change_detector.py
========================
Efficient per-frame change detection for video object replacement pipelines.

Detects four categories of change between consecutive frames:
  1. Object geometric changes  (translation, rotation, scale)
  2. Background changes        (inside the bounding box, outside the mask)
  3. Occlusion changes         (new content covering the object)
  4. Lighting / appearance     (illumination shift on the object)

All detection is classical CV + NumPy — no deep learning required.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  –  tune every threshold here
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # ── Geometric ──────────────────────────────────────────────────────────
    # Minimum IoU drop between masks to flag a geometric change
    "iou_change_threshold": 0.85,

    # Centroid shift in pixels to flag translation
    "centroid_shift_threshold": 8.0,        # px

    # Rotation change in degrees to flag rotation
    "rotation_change_threshold": 5.0,       # degrees

    # Relative scale change of the bounding box area to flag scale change
    # e.g. 0.10 means a 10 % area change triggers regeneration
    "scale_change_threshold": 0.10,

    # ── Background (inside bbox, outside mask) ─────────────────────────────
    # Mean absolute pixel difference (0–255) in the background region
    "background_diff_threshold": 18.0,

    # Minimum fraction of background pixels that must be "active"
    # before we consider it a real background change (noise guard)
    "background_change_pixel_fraction": 0.05,

    # ── Occlusion ──────────────────────────────────────────────────────────
    # If the visible mask area shrinks by more than this fraction
    # of the previous mask area, we consider it an occlusion
    "occlusion_area_drop_threshold": 0.10,

    # ── Lighting ───────────────────────────────────────────────────────────
    # Mean absolute difference of pixel intensity INSIDE the mask (grayscale)
    "lighting_diff_threshold": 12.0,

    # Fraction of mask pixels that must exceed per-pixel threshold
    "lighting_change_pixel_fraction": 0.10,

    # Per-pixel intensity delta that counts as "changed"
    "lighting_per_pixel_threshold": 20,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two binary masks.

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        Binary (H, W) arrays with values in {0, 1}.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 1.0 if both masks are empty.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Return the (cx, cy) centroid of a binary mask.

    Uses the image moment M00 / M10 / M01 so it is consistent with
    OpenCV's contour moments.

    Returns (0, 0) for an empty mask (caller should guard against this).
    """
    m = cv2.moments(mask.astype(np.uint8))
    if m["m00"] == 0:
        return (0.0, 0.0)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (cx, cy)


def centroid_shift(mask_prev: np.ndarray, mask_curr: np.ndarray) -> float:
    """
    Euclidean distance between the centroids of two masks (pixels).
    """
    cx_p, cy_p = mask_centroid(mask_prev)
    cx_c, cy_c = mask_centroid(mask_curr)
    return float(np.hypot(cx_c - cx_p, cy_c - cy_p))


def estimate_rotation(
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
) -> float:
    """
    Approximate rotation of the object between frames (degrees).

    Strategy
    --------
    Fit an ellipse to each mask using image moments and compute the
    difference in orientation angles.  This is a fast O(N) approximation
    that handles roughly convex / symmetric objects well.

    For highly irregular or elongated objects consider using the
    minimum bounding rectangle angle from cv2.minAreaRect instead
    (uncomment the alternative path below).

    Returns 0.0 if either mask is too sparse to fit reliably.
    """
    def _orientation(mask: np.ndarray) -> float:
        m = cv2.moments(mask.astype(np.uint8))
        if m["m00"] < 10:
            return 0.0
        # Centralised second-order moments
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        # Angle of the principal axis
        angle = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))
        return float(angle)

    # ── Alternative: minimum bounding rectangle ──────────────────────────
    # def _orientation(mask):
    #     pts = cv2.findNonZero(mask.astype(np.uint8))
    #     if pts is None or len(pts) < 5:
    #         return 0.0
    #     _, _, angle = cv2.minAreaRect(pts)
    #     return float(angle)

    angle_prev = _orientation(mask_prev)
    angle_curr = _orientation(mask_curr)
    delta = angle_curr - angle_prev
    # Wrap to [-90, 90] (orientation is modulo 180°)
    delta = (delta + 90.0) % 180.0 - 90.0
    return abs(delta)


def bbox_scale_change(
    bbox_prev: Tuple[int, int, int, int],
    bbox_curr: Tuple[int, int, int, int],
) -> float:
    """
    Relative change in bounding-box area between two frames.

    Returns
    -------
    float
        |area_curr - area_prev| / area_prev.
        Returns 0.0 if area_prev is zero (degenerate box).
    """
    x1p, y1p, x2p, y2p = bbox_prev
    x1c, y1c, x2c, y2c = bbox_curr
    area_prev = max(0, x2p - x1p) * max(0, y2p - y1p)
    area_curr = max(0, x2c - x1c) * max(0, y2c - y1c)
    if area_prev == 0:
        return 0.0
    return abs(area_curr - area_prev) / float(area_prev)


def illumination_diff(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    mask: np.ndarray,
    per_pixel_threshold: int = 20,
) -> Tuple[float, float]:
    """
    Compare illumination inside *mask* between two frames.

    Parameters
    ----------
    frame_prev, frame_curr : np.ndarray
        BGR or RGB frames (H, W, 3).
    mask : np.ndarray
        Binary (H, W) mask — pixels inside the object.
    per_pixel_threshold : int
        Absolute intensity delta above which a pixel is considered changed.

    Returns
    -------
    mean_diff : float
        Mean absolute grayscale difference inside the mask.
    changed_fraction : float
        Fraction of mask pixels whose difference exceeds *per_pixel_threshold*.
    """
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mask_bool = mask.astype(bool)
    n_pixels  = mask_bool.sum()
    if n_pixels == 0:
        return 0.0, 0.0

    diff = np.abs(gray_curr[mask_bool] - gray_prev[mask_bool])
    mean_diff        = float(diff.mean())
    changed_fraction = float((diff > per_pixel_threshold).sum()) / float(n_pixels)
    return mean_diff, changed_fraction


def _extract_background_region(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    """
    Crop the bounding-box region and zero-out the object mask pixels,
    leaving only the background area inside the box.

    Returns a float32 (H_box, W_box, 3) array.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop_frame = frame[y1:y2, x1:x2].astype(np.float32)
    crop_mask  = mask[y1:y2, x1:x2].astype(bool)

    # Zero out the object itself so we only compare background pixels
    crop_frame[crop_mask] = 0.0
    return crop_frame


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def has_significant_change(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_mask:  np.ndarray,
    curr_mask:  np.ndarray,
    prev_bbox:  Tuple[int, int, int, int],
    curr_bbox:  Tuple[int, int, int, int],
    cfg:        dict | None = None,
    return_debug: bool = False,
) -> Union[bool, Tuple[bool, dict]]:
    """
    Decide whether the current frame needs regeneration.

    Parameters
    ----------
    prev_frame, curr_frame : np.ndarray
        BGR (or consistently RGB) frames, shape (H, W, 3), dtype uint8.
    prev_mask, curr_mask : np.ndarray
        Binary segmentation masks (H, W), values in {0, 1}.
    prev_bbox, curr_bbox : tuple
        Bounding boxes (x1, y1, x2, y2) in pixel coordinates.
    cfg : dict, optional
        Override any key from the module-level CFG dict.
    return_debug : bool
        If True, return (changed: bool, debug: dict).

    Returns
    -------
    bool
        True  → regenerate this frame.
        False → reuse previous generated result.
    Tuple[bool, dict]  (only when return_debug=True)
        Second element contains per-check metric and threshold values.
    """
    c = {**CFG, **(cfg or {})}  # merge caller overrides

    debug: dict = {
        "triggered_by": None,
        "checks": {},
    }

    changed = False

    # ── 1a. Geometric: IoU ────────────────────────────────────────────────
    iou = mask_iou(prev_mask, curr_mask)
    iou_flag = iou < c["iou_change_threshold"]
    debug["checks"]["mask_iou"] = {
        "value":     iou,
        "threshold": c["iou_change_threshold"],
        "triggered": iou_flag,
    }
    if iou_flag and not changed:
        debug["triggered_by"] = "mask_iou"
    changed = changed or iou_flag

    # ── 1b. Geometric: centroid translation ───────────────────────────────
    shift = centroid_shift(prev_mask, curr_mask)
    shift_flag = shift > c["centroid_shift_threshold"]
    debug["checks"]["centroid_shift_px"] = {
        "value":     shift,
        "threshold": c["centroid_shift_threshold"],
        "triggered": shift_flag,
    }
    if shift_flag and not changed:
        debug["triggered_by"] = "centroid_shift"
    changed = changed or shift_flag

    # ── 1c. Geometric: rotation ───────────────────────────────────────────
    rot = estimate_rotation(prev_mask, curr_mask)
    rot_flag = rot > c["rotation_change_threshold"]
    debug["checks"]["rotation_deg"] = {
        "value":     rot,
        "threshold": c["rotation_change_threshold"],
        "triggered": rot_flag,
    }
    if rot_flag and not changed:
        debug["triggered_by"] = "rotation"
    changed = changed or rot_flag

    # ── 1d. Geometric: scale (bbox area) ──────────────────────────────────
    scale = bbox_scale_change(prev_bbox, curr_bbox)
    scale_flag = scale > c["scale_change_threshold"]
    debug["checks"]["bbox_scale_change"] = {
        "value":     scale,
        "threshold": c["scale_change_threshold"],
        "triggered": scale_flag,
    }
    if scale_flag and not changed:
        debug["triggered_by"] = "bbox_scale"
    changed = changed or scale_flag

    # ── 2. Background changes inside the bounding box ─────────────────────
    # We use the *union* of prev/curr masks to suppress the object itself,
    # then compare the background region between the two frames.
    union_mask = np.logical_or(prev_mask.astype(bool), curr_mask.astype(bool)).astype(np.uint8)

    # Use the larger (union) bbox to ensure we capture the whole context
    bg_x1 = min(prev_bbox[0], curr_bbox[0])
    bg_y1 = min(prev_bbox[1], curr_bbox[1])
    bg_x2 = max(prev_bbox[2], curr_bbox[2])
    bg_y2 = max(prev_bbox[3], curr_bbox[3])
    bg_bbox = (bg_x1, bg_y1, bg_x2, bg_y2)

    bg_prev = _extract_background_region(prev_frame, bg_bbox, union_mask)
    bg_curr = _extract_background_region(curr_frame, bg_bbox, union_mask)

    # Only compare pixels that belong to the background in BOTH crops
    bg_mask_crop = ~union_mask[
        max(0, bg_y1):min(prev_frame.shape[0], bg_y2),
        max(0, bg_x1):min(prev_frame.shape[1], bg_x2),
    ].astype(bool)

    n_bg = bg_mask_crop.sum()
    if n_bg > 0:
        bg_diff_vals = np.abs(bg_curr[bg_mask_crop] - bg_prev[bg_mask_crop])
        bg_mean_diff = float(bg_diff_vals.mean())
        bg_pixel_frac = float((bg_diff_vals > c["lighting_per_pixel_threshold"]).any(axis=-1).mean())
    else:
        bg_mean_diff = 0.0
        bg_pixel_frac = 0.0

    bg_flag = (
        bg_mean_diff    > c["background_diff_threshold"] and
        bg_pixel_frac   > c["background_change_pixel_fraction"]
    )
    debug["checks"]["background_diff"] = {
        "mean_diff":     bg_mean_diff,
        "pixel_frac":    bg_pixel_frac,
        "diff_threshold":  c["background_diff_threshold"],
        "frac_threshold":  c["background_change_pixel_fraction"],
        "triggered":     bg_flag,
    }
    if bg_flag and not changed:
        debug["triggered_by"] = "background_change"
    changed = changed or bg_flag

    # ── 3. Occlusion: mask area drop ──────────────────────────────────────
    area_prev = float(prev_mask.sum())
    area_curr = float(curr_mask.sum())
    if area_prev > 0:
        area_drop = max(0.0, (area_prev - area_curr) / area_prev)
    else:
        area_drop = 0.0

    occ_flag = area_drop > c["occlusion_area_drop_threshold"]
    debug["checks"]["occlusion_area_drop"] = {
        "value":     area_drop,
        "threshold": c["occlusion_area_drop_threshold"],
        "triggered": occ_flag,
    }
    if occ_flag and not changed:
        debug["triggered_by"] = "occlusion"
    changed = changed or occ_flag

    # ── 4. Lighting / illumination inside the object mask ─────────────────
    # Use the *intersection* mask so we only compare pixels visible in both.
    inter_mask = np.logical_and(prev_mask.astype(bool), curr_mask.astype(bool)).astype(np.uint8)
    light_mean, light_frac = illumination_diff(
        prev_frame, curr_frame,
        inter_mask,
        per_pixel_threshold=c["lighting_per_pixel_threshold"],
    )
    light_flag = (
        light_mean > c["lighting_diff_threshold"] and
        light_frac > c["lighting_change_pixel_fraction"]
    )
    debug["checks"]["lighting_diff"] = {
        "mean_diff":     light_mean,
        "pixel_frac":    light_frac,
        "diff_threshold":  c["lighting_diff_threshold"],
        "frac_threshold":  c["lighting_change_pixel_fraction"],
        "triggered":     light_flag,
    }
    if light_flag and not changed:
        debug["triggered_by"] = "lighting"
    changed = changed or light_flag

    # ── Final decision ─────────────────────────────────────────────────────
    if return_debug:
        return changed, debug
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST  –  run with `python frame_change_detector.py`
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    H, W = 480, 640
    base_frame = rng.integers(60, 200, (H, W, 3), dtype=np.uint8)

    # Simple circular mask centred at (320, 240), radius 60
    def _circle_mask(cx, cy, r, h=H, w=W):
        ys, xs = np.ogrid[:h, :w]
        return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2).astype(np.uint8)

    prev_mask = _circle_mask(320, 240, 60)
    prev_bbox = (260, 180, 380, 300)
    prev_frame = base_frame.copy()

    # ── Test 1: identical frames → should NOT change
    changed, dbg = has_significant_change(
        prev_frame, prev_frame.copy(),
        prev_mask, prev_mask.copy(),
        prev_bbox, prev_bbox,
        return_debug=True,
    )
    print(f"[Test 1 – no change]   changed={changed}  triggered_by={dbg['triggered_by']}")
    assert not changed, "Expected False for identical frames"

    # ── Test 2: large centroid shift → SHOULD change
    curr_mask2 = _circle_mask(360, 280, 60)   # shifted by ~57 px
    curr_bbox2  = (300, 220, 420, 340)
    changed2, dbg2 = has_significant_change(
        prev_frame, prev_frame.copy(),
        prev_mask, curr_mask2,
        prev_bbox, curr_bbox2,
        return_debug=True,
    )
    print(f"[Test 2 – translation] changed={changed2}  triggered_by={dbg2['triggered_by']}")
    assert changed2, "Expected True for large translation"

    # ── Test 3: bright light burst on the object → SHOULD change
    curr_frame3 = prev_frame.copy()
    curr_frame3[prev_mask.astype(bool)] = np.clip(
        curr_frame3[prev_mask.astype(bool)].astype(np.int16) + 60, 0, 255
    ).astype(np.uint8)
    changed3, dbg3 = has_significant_change(
        prev_frame, curr_frame3,
        prev_mask, prev_mask.copy(),
        prev_bbox, prev_bbox,
        return_debug=True,
    )
    print(f"[Test 3 – lighting]    changed={changed3}  triggered_by={dbg3['triggered_by']}")
    assert changed3, "Expected True for large illumination change"

    # ── Test 4: background motion inside bbox → SHOULD change
    curr_frame4 = prev_frame.copy()
    x1, y1, x2, y2 = prev_bbox
    # Shift background patch by 20 px horizontally (simulate moving hand)
    bg_patch = prev_frame[y1:y2, x1:x2].copy()
    curr_frame4[y1:y2, x1:x2] = np.roll(bg_patch, shift=20, axis=1)
    # Restore the object pixels so only background changed
    curr_frame4[prev_mask.astype(bool)] = prev_frame[prev_mask.astype(bool)]
    changed4, dbg4 = has_significant_change(
        prev_frame, curr_frame4,
        prev_mask, prev_mask.copy(),
        prev_bbox, prev_bbox,
        return_debug=True,
    )
    print(f"[Test 4 – background]  changed={changed4}  triggered_by={dbg4['triggered_by']}")
    assert changed4, "Expected True for background change inside bbox"

    print("\nAll tests passed ✓")
