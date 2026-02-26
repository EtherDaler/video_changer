import torch
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import (
    StableDiffusionXLControlNetInpaintPipeline,
)
from controlnet_aux import MidasDetector


def build_multi_controlnet_pipe(
    torch_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    cpu_offload: bool = False,
    sequential_offload: bool = False,
) -> tuple[StableDiffusionXLControlNetInpaintPipeline, MidasDetector]:
    """Build SDXL inpainting pipeline with Canny + Depth ControlNet."""

    depth_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")

    # With memory offloading the pipeline moves weights automatically;
    # do NOT call .to(device) on individual models in that case.
    move_to_device = not (cpu_offload or sequential_offload)

    controlnet_canny: ControlNetModel = ControlNetModel.from_pretrained(  # type: ignore[assignment]
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch_dtype,
    )
    if move_to_device:
        controlnet_canny = controlnet_canny.to(device)  # type: ignore[assignment]

    controlnet_depth: ControlNetModel = ControlNetModel.from_pretrained(  # type: ignore[assignment]
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch_dtype,
    )
    if move_to_device:
        controlnet_depth = controlnet_depth.to(device)  # type: ignore[assignment]

    multi_controlnet = MultiControlNetModel([controlnet_canny, controlnet_depth])

    pipe: StableDiffusionXLControlNetInpaintPipeline = (
        StableDiffusionXLControlNetInpaintPipeline.from_pretrained(  # type: ignore[assignment]
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=multi_controlnet,
            torch_dtype=torch_dtype,
        )
    )

    if sequential_offload:
        pipe.enable_sequential_cpu_offload()
        print("  Memory mode: sequential CPU offload (~3 GB VRAM)")
    elif cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  Memory mode: model CPU offload (~4 GB VRAM)")
    else:
        pipe = pipe.to(device)  # type: ignore[assignment]
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)

    return pipe, depth_detector
