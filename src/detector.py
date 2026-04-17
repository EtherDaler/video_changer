"""
Text-based object detection: Grounding DINO → SAM2.

Flow:
  text prompt  →  Grounding DINO  →  bounding box
              →  SAM2 (box prompt)  →  precise mask
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────
# GROUNDING DINO
# ──────────────────────────────────────────────

GDINO_CHECKPOINT = "groundingdino_swint_ogc.pth"


def _gdino_config_path() -> str:
    """Auto-detect the config bundled inside the installed groundingdino package."""
    try:
        import groundingdino
        cfg = os.path.join(
            os.path.dirname(groundingdino.__file__),
            "config", "GroundingDINO_SwinT_OGC.py",
        )
        if os.path.exists(cfg):
            return cfg
    except ImportError:
        pass
    # Legacy fallback: cloned repo next to project root
    return "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"


def load_grounding_dino(
    config_path: str | None = None,
    checkpoint_path: str = GDINO_CHECKPOINT,
    device: str = "cuda",
):
    """Load Grounding DINO model."""
    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError(
            "groundingdino not found. Install with:\n"
            "  pip install groundingdino-py\n"
            "or from source: https://github.com/IDEA-Research/GroundingDINO"
        )
    if config_path is None:
        config_path = _gdino_config_path()
    return load_model(config_path, checkpoint_path, device=device)


def _preprocess_for_gdino(image_np: np.ndarray) -> torch.Tensor:
    """Apply Grounding DINO's standard image transform."""
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.fromarray(image_np))  # type: ignore[return-value]


def detect_bbox(
    model,
    image_np: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.82,
    text_threshold: float = 0.77,
    device: str = "cpu",
) -> tuple[tuple[int, int, int, int], str, float] | None:
    """
    Detect the best-matching bounding box for *text_prompt* in *image_np*.

    Returns (bbox, matched_phrase, confidence) where bbox is (x1, y1, x2, y2)
    in pixel coords, or None if nothing was found above the thresholds.
    """
    from groundingdino.util.inference import predict

    image_tensor = _preprocess_for_gdino(image_np)
    h, w = image_np.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if len(boxes) == 0:
        return None

    best = int(logits.argmax())
    cx, cy, bw, bh = boxes[best].tolist()  # normalised cxcywh

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))

    return (x1, y1, x2, y2), phrases[best], float(logits[best])


def detect_bboxes_multi(
    model,
    image_np: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.82,
    text_threshold: float = 0.77,
    device: str = "cpu",
    max_detections: int = 10,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """
    Все детекции Grounding DINO по *text_prompt* (по убыванию confidence), не более *max_detections*.
    Координаты bbox в пикселях, как у :func:`detect_bbox`.
    """
    from groundingdino.util.inference import predict

    image_tensor = _preprocess_for_gdino(image_np)
    h, w = image_np.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if len(boxes) == 0:
        return []

    scores = logits.detach().float().cpu().numpy() if hasattr(logits, "detach") else np.asarray(
        logits, dtype=np.float32
    )
    order = np.argsort(-scores)
    out: list[tuple[tuple[int, int, int, int], str, float]] = []
    n = min(len(boxes), int(max_detections))

    for k in range(n):
        i = int(order[k])
        cx, cy, bw, bh = boxes[i].tolist()
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        if isinstance(phrases, (list, tuple)):
            phrase = str(phrases[i])
        else:
            phrase = str(phrases)
        out.append(((x1, y1, x2, y2), phrase, float(scores[i])))
    return out


# ──────────────────────────────────────────────
# SAM2 WITH BOUNDING-BOX PROMPT
# ──────────────────────────────────────────────

def get_mask_from_bbox(
    predictor,
    image_np: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Run SAM2 with a bounding-box prompt.
    Returns a uint8 mask (0 / 255).
    """
    x1, y1, x2, y2 = bbox
    predictor.set_image(image_np)

    masks, _, _ = predictor.predict(
        box=np.array([x1, y1, x2, y2]),
        multimask_output=False,
    )
    return masks[0].astype(np.uint8) * 255


# ──────────────────────────────────────────────
# HIGH-LEVEL HELPER
# ──────────────────────────────────────────────

def text_to_mask(
    grounding_model,
    sam_predictor,
    image_np: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.82,
    text_threshold: float = 0.77,
    device: str = "cpu",
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str | None, float]:
    """
    End-to-end: text prompt → segmentation mask.

    Returns (mask, bbox, matched_phrase, confidence).
    mask/bbox/phrase are None and confidence is 0.0 if the object was not detected.
    """
    result = detect_bbox(
        grounding_model, image_np, text_prompt,
        box_threshold, text_threshold, device=device,
    )
    if result is None:
        return None, None, None, 0.0

    bbox, phrase, conf = result
    mask = get_mask_from_bbox(sam_predictor, image_np, bbox)
    return mask, bbox, phrase, conf
