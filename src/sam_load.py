import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)

def get_mask_from_click(frame, x, y):
    sam2_predictor.set_image(frame)

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    masks, scores, _ = sam2_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = masks[0].astype(np.uint8) * 255
    return mask