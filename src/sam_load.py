import numpy as np

SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def build_sam2_predictor(
    checkpoint: str = SAM2_CHECKPOINT,
    config: str = SAM2_CONFIG,
    device: str = "cuda",
):
    """Build and return a SAM2 image predictor."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def get_mask_from_click(predictor, frame, x: int, y: int) -> np.ndarray:
    """Return a binary uint8 mask (0/255) for the object at (x, y)."""
    predictor.set_image(frame)

    masks, _, _ = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )

    return masks[0].astype(np.uint8) * 255
