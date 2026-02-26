"""
SAM2 Video Predictor — tracks an object mask across all video frames.

Advantages over optical flow warping:
  • Handles occlusions (object temporarily hidden)
  • Handles reappearance after going off-screen
  • Handles fast motion and shape changes
  • Produces per-frame masks in one forward pass
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import torch

SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def build_video_predictor(
    checkpoint: str = SAM2_CHECKPOINT,
    config: str = SAM2_CONFIG,
    device: str = "cpu",
):
    """Build and return a SAM2VideoPredictor."""
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(config, checkpoint, device=device)


def track_masks_video(
    predictor,
    frames: list[np.ndarray],
    initial_mask: np.ndarray,
    initial_frame_idx: int = 0,
) -> list[np.ndarray]:
    """
    Propagate *initial_mask* (uint8, 0/255) through all *frames* using
    SAM2 VideoPredictor.

    Returns a list of uint8 masks (0/255), one per frame.
    Frames where the object is not detected get an empty (zero) mask.
    """
    n = len(frames)

    # SAM2 VideoPredictor loads from a directory of JPEG images.
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_frames(frames, tmpdir)

        with torch.inference_mode():
            state = predictor.init_state(tmpdir)
            predictor.reset_state(state)

            # Provide the initial mask on the seed frame.
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=initial_frame_idx,
                obj_id=1,
                mask=initial_mask > 128,  # bool array
            )

            tracked: list[np.ndarray | None] = [None] * n

            def _collect(gen) -> None:
                for out_idx, obj_ids, logits in gen:
                    if tracked[out_idx] is not None:
                        continue  # forward pass already set this frame
                    obj_ids_list = [int(x) for x in obj_ids]
                    if 1 not in obj_ids_list:
                        continue
                    ch = obj_ids_list.index(1)
                    binary = (logits[ch] > 0.0).squeeze(0).cpu().numpy()
                    tracked[out_idx] = binary.astype(np.uint8) * 255

            # Forward pass: seed frame → last frame
            _collect(predictor.propagate_in_video(state))

            # Backward pass: seed frame → first frame (only needed when seed > 0)
            if initial_frame_idx > 0:
                try:
                    _collect(predictor.propagate_in_video(state, reverse=True))
                except TypeError:
                    # Older SAM2 versions may not support reverse=True
                    pass

    # Fill frames where SAM2 found nothing with an empty mask.
    empty = np.zeros(initial_mask.shape, dtype=np.uint8)
    result = [m if m is not None else empty.copy() for m in tracked]
    return result


def fill_mask_gaps(masks: list[np.ndarray], max_gap: int = 2) -> list[np.ndarray]:
    """
    Fill isolated runs of empty masks that are surrounded by non-empty ones.

    This fixes SAM2 Video Predictor drop-outs where the model loses confidence
    for 1-2 frames even though the object is clearly still present.

    Args:
        masks:    List of uint8 masks (0/255), one per frame.
        max_gap:  Maximum length of an empty run to fill.
                  Runs longer than this are kept empty (object genuinely absent).
    """
    n = len(masks)
    filled = [m.copy() for m in masks]

    i = 0
    while i < n:
        if filled[i].any():
            i += 1
            continue

        # Find the extent of this empty run.
        j = i
        while j < n and not filled[j].any():
            j += 1
        gap_len = j - i

        # Fill only if the gap is short enough and has non-empty neighbours on both sides.
        has_left  = i > 0 and filled[i - 1].any()
        has_right = j < n and filled[j].any()
        if gap_len <= max_gap and has_left and has_right:
            for k in range(i, j):
                # Interpolate linearly between the bordering masks.
                left_mask  = filled[i - 1].astype(np.float32)
                right_mask = filled[j].astype(np.float32)
                t = (k - i + 1) / (gap_len + 1)
                blended = (1 - t) * left_mask + t * right_mask
                filled[k] = (blended > 127).astype(np.uint8) * 255

        i = j if gap_len > max_gap else j  # skip past the run
    return filled


# ──────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────

def _write_frames(frames: list[np.ndarray], directory: str) -> None:
    """Write RGB frames to *directory* as numbered JPEG files."""
    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = os.path.join(directory, f"{i:05d}.jpg")
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
