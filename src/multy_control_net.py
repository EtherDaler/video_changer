from controlnet_aux import MidasDetector
from diffusers import MultiControlNetModel

depth_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")


controlnet_canny = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to("cuda")

controlnet_depth = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16
).to("cuda")

multi_controlnet = MultiControlNetModel(
    [controlnet_canny, controlnet_depth]
)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=multi_controlnet,
    torch_dtype=torch.float16
).to("cuda")