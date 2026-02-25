"""
Interactive OpenCV UI for object selection.

Keyboard shortcuts (shown in window title):
  Left-click  — place point / confirm click
  Enter       — accept current selection
  R           — redo / reset
  Esc         — cancel
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────

def _to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def _draw_point(bgr: np.ndarray, pt: tuple[int, int]) -> np.ndarray:
    out = bgr.copy()
    cv2.circle(out, pt, 10, (0, 255, 0), -1)
    cv2.circle(out, pt, 14, (255, 255, 255), 2)
    return out


def _draw_bbox(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    return out


def _overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blue-tinted overlay on the masked area."""
    overlay = frame_rgb.astype(np.float32).copy()
    mask_bool = mask > 128
    overlay[mask_bool] = overlay[mask_bool] * 0.45 + np.array([30, 120, 255]) * 0.55
    return overlay.clip(0, 255).astype(np.uint8)


def _put_hint(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    h = out.shape[0]
    cv2.rectangle(out, (0, h - 36), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────

def select_click_point(
    frame_rgb: np.ndarray,
    window_name: str = "Select object",
) -> Optional[tuple[int, int]]:
    """
    Show *frame_rgb* in a window and let the user left-click on the object.

    Returns (x, y) when the user presses Enter, or None on Esc.
    """
    state: dict = {"pt": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    bgr_orig = _to_bgr(frame_rgb)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    hint = "Left-click on object  |  Enter = confirm  |  R = reset  |  Esc = cancel"

    while True:
        display = bgr_orig.copy()
        if state["pt"]:
            display = _draw_point(display, state["pt"])
        cv2.imshow(window_name, _put_hint(display, hint))

        key = cv2.waitKey(30) & 0xFF
        if key == 13 and state["pt"]:   # Enter
            break
        elif key == ord("r"):
            state["pt"] = None
        elif key == 27:                 # Esc
            state["pt"] = None
            break

    cv2.destroyWindow(window_name)
    return state["pt"]


def confirm_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    extra_info: str = "",
    window_name: str = "Mask preview",
) -> str:
    """
    Show the mask overlaid on *frame_rgb*.

    Returns:
      "accept"  — user pressed Enter
      "redo"    — user pressed R (redo from scratch)
      "refine"  — user pressed F (refine by click)
      "cancel"  — user pressed Esc
    """
    overlay_rgb = _overlay_mask(frame_rgb, mask)
    bgr = _to_bgr(overlay_rgb)
    hint = (
        f"{extra_info}  |  " if extra_info else ""
    ) + "Enter=accept  R=redo  F=refine by click  Esc=cancel"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, _put_hint(bgr, hint))

    result = "cancel"
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:           # Enter
            result = "accept"
            break
        elif key == ord("r"):
            result = "redo"
            break
        elif key == ord("f"):
            result = "refine"
            break
        elif key == 27:         # Esc
            result = "cancel"
            break

    cv2.destroyWindow(window_name)
    return result


def show_bbox_preview(
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    phrase: str,
    window_name: str = "Detection preview",
) -> str:
    """
    Show the detected bounding box before SAM2 runs.

    Returns "accept", "redo", or "cancel".
    """
    bgr = _draw_bbox(_to_bgr(frame_rgb), bbox)
    hint = f'Detected: "{phrase}"  |  Enter=segment  R=redo text  Esc=cancel'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, _put_hint(bgr, hint))

    result = "cancel"
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:
            result = "accept"
            break
        elif key == ord("r"):
            result = "redo"
            break
        elif key == 27:
            result = "cancel"
            break

    cv2.destroyWindow(window_name)
    return result
