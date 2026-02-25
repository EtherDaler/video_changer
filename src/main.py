import torch
import cv2
import numpy as np
import imageio

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel
)
from PIL import Image


# ========================
# CONFIG
# ========================

device = "cuda"
torch_dtype = torch.float16
WIDTH = 1024
HEIGHT = 1024

prompt = "realistic printed photo on ceramic mug surface, matching lighting"
negative_prompt = "distorted object, unrealistic lighting, extra objects"


# ========================
# LOAD MODELS
# ========================

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch_dtype
).to(device)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet,
    torch_dtype=torch_dtype
).to(device)

pipe.enable_xformers_memory_efficient_attention()
pipe.set_progress_bar_config(disable=True)


# ========================
# VIDEO
# ========================

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frames.append(frame)

    cap.release()
    return frames


# ========================
# MASK (заглушка)
# ========================

def initial_mask(frame):
    # Временно — центр кадра
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (WIDTH//2, HEIGHT//2), 200, 255, -1)
    return mask


# ========================
# CONTROL IMAGE
# ========================

def make_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges)


# ========================
# OPTICAL FLOW
# ========================

def compute_flow(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray,
        frame2_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )
    return flow


def warp_mask(mask, flow):
    h, w = mask.shape

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST)
    return warped


# ========================
# INPAINT WITH TEMPORAL
# ========================

def inpaint_with_temporal(
    image_np,
    mask_np,
    previous_latents,
    seed=42
):
    image = Image.fromarray(image_np)
    mask = Image.fromarray(mask_np)

    control_image = make_canny(image_np)

    generator = torch.Generator(device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        control_image=control_image,
        guidance_scale=7.0,
        num_inference_steps=30,
        generator=generator,
        latents=previous_latents,
        output_type="latent"
    )

    latents = result.images

    decoded = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor
    )[0]

    image = pipe.image_processor.postprocess(decoded)[0]

    return np.array(image), latents


# ========================
# ASSEMBLE
# ========================

def assemble_video(frames, output_path, fps=24):
    imageio.mimsave(output_path, frames, fps=fps)


# ========================
# MAIN
# ========================

if __name__ == "__main__":

    frames = extract_frames("input.mp4")
    mask = initial_mask(frames[0])

    previous_latents = None
    processed = []

    for i in range(len(frames)):

        frame = frames[i]

        if i > 0:
            flow = compute_flow(frames[i-1], frame)
            mask = warp_mask(mask, flow)

        output, previous_latents = inpaint_with_temporal(
            frame,
            mask,
            previous_latents
        )

        processed.append(output)

    assemble_video(processed, "output.mp4")