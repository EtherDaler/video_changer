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

            for out_idx, obj_ids, logits in predictor.propagate_in_video(state):
                if 1 not in obj_ids:
                    continue
                ch = list(obj_ids).index(1)
                binary = (logits[ch] > 0.0).squeeze(0).cpu().numpy()
                tracked[out_idx] = binary.astype(np.uint8) * 255

    # Fill frames where SAM2 found nothing with an empty mask.
    empty = np.zeros(initial_mask.shape, dtype=np.uint8)
    return [m if m is not None else empty.copy() for m in tracked]


# ──────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────

def _write_frames(frames: list[np.ndarray], directory: str) -> None:
    """Write RGB frames to *directory* as numbered JPEG files."""
    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = os.path.join(directory, f"{i:05d}.jpg")
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
