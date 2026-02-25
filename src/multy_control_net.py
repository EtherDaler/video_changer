import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    MultiControlNetModel,
)
from controlnet_aux import MidasDetector


def build_multi_controlnet_pipe(
    torch_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[StableDiffusionXLControlNetInpaintPipeline, MidasDetector]:
    """Build SDXL inpainting pipeline with Canny + Depth ControlNet."""

    depth_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")

    controlnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch_dtype,
    ).to(device)

    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch_dtype,
    ).to(device)

    multi_controlnet = MultiControlNetModel([controlnet_canny, controlnet_depth])

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=multi_controlnet,
        torch_dtype=torch_dtype,
    ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=True)

    return pipe, depth_detector
