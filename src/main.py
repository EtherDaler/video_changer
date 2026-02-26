"""
Video object replacement pipeline:
  Frame extraction → SAM2 segmentation → Mask tracking (optical flow)
  → ControlNet (Canny [+ Depth]) → SDXL Inpainting → Temporal consistency
  → Color harmonization → Grain + motion-blur match → Video assembly
"""

import argparse

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import (
    StableDiffusionXLControlNetInpaintPipeline,
)


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _auto_device()
# float16 is not supported on CPU; MPS works fine with float16 on modern macOS
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
WIDTH  = 1024
HEIGHT = 1024

# TF32 is faster than FP32 on Ampere/Ada/Blackwell for any float32 fallback ops
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True


def _cuda_sm() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of GPU 0, or None if not CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    return None


def _is_blackwell() -> bool:
    """True when running on NVIDIA Blackwell (sm_120, RTX 5000-series)."""
    sm = _cuda_sm()
    return sm is not None and sm[0] >= 12

DEFAULT_PROMPT = (
    "realistic printed photo on ceramic mug surface, matching lighting"
)
DEFAULT_NEG_PROMPT = (
    "distorted object, unrealistic lighting, extra objects, blurry"
)


# ──────────────────────────────────────────────
# PIPELINE LOADING
# ──────────────────────────────────────────────

def _configure_attention(pipe, device: str, attention_slicing: bool) -> None:
    """
    Choose the best attention backend for the current GPU.

    Strategy:
      • Blackwell (sm_120+, RTX 5060/5070/5080/5090):
          xformers pre-0.0.33 has no Blackwell CUTLASS kernels and causes slowdowns.
          We force PyTorch's native scaled_dot_product_attention (AttnProcessor2_0)
          which works out-of-the-box with PyTorch 2.7+cu128.

      • Other NVIDIA GPUs:
          Try xformers first (15-20% VRAM savings). If not installed, SDPA is used
          automatically by diffusers ≥0.20 with PyTorch 2.x.

      • attention_slicing: always applied on top, further chunked memory access.
    """
    if device != "cuda":
        if attention_slicing:
            pipe.enable_attention_slicing(1)
            print("  Attention slicing enabled")
        return

    if _is_blackwell():
        # On Blackwell: use PyTorch 2.x built-in SDPA (AttnProcessor2_0).
        # xformers < 0.0.33 lacks Blackwell CUTLASS kernels and may be slower
        # than native SDPA or cause CUDA errors.
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            print("  Attention: PyTorch SDPA (AttnProcessor2_0) — optimal for Blackwell")
        except Exception as e:
            print(f"  Attention: SDPA (AttnProcessor2_0 unavailable: {e})")
    else:
        # Non-Blackwell NVIDIA: prefer xformers for VRAM savings
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  Attention: xformers memory-efficient")
        except Exception:
            print("  Attention: PyTorch SDPA (xformers not installed)")

    if attention_slicing:
        pipe.enable_attention_slicing(1)
        print("  Attention slicing enabled")


def _apply_torch_compile(pipe) -> None:
    """
    JIT-compile the UNet (and optionally VAE decoder) with torch.compile.

    PyTorch 2.7 ships Triton 3.3 which adds Blackwell (sm_120) support, so
    torch.compile works correctly on RTX 5060+ when using cu128.

    NOTE: CUDA 12.8/12.9 has a known missing libnvptxcompiler.so issue that
    breaks JIT PTX compilation. If torch.compile fails, we fall back silently.
    Warm-up on first frame: 2-5 min. All subsequent frames run at native speed.
    """
    # On Blackwell use "default" mode — "reduce-overhead" uses CUDA graphs
    # which can trigger the JIT PTX bug on CUDA 12.8; "default" is safer.
    mode = "default" if _is_blackwell() else "reduce-overhead"
    try:
        print(f"  torch.compile: compiling UNet (mode={mode}, first frame = warm-up ~2-5 min)…")
        pipe.unet = torch.compile(pipe.unet, mode=mode, fullgraph=False)
        # VAE decode is also in the hot path; compile it too
        if hasattr(pipe, "vae"):
            pipe.vae.decode = torch.compile(pipe.vae.decode, mode=mode, fullgraph=False)
        print("  torch.compile ready")
    except Exception as e:
        print(f"  torch.compile skipped ({e})")


def load_pipeline(
    use_multi_controlnet: bool = False,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
    cpu_offload: bool = False,
    sequential_offload: bool = False,
    attention_slicing: bool = False,
    torch_compile: bool = False,
):
    """
    Load SDXL inpainting pipeline with Canny (+ Depth) ControlNet.

    Memory modes:
      default             — all weights on GPU (~8 GB VRAM needed)
      cpu_offload         — model layers swapped to CPU RAM on demand (~4 GB VRAM)
      sequential_offload  — one layer at a time on GPU (~3 GB VRAM, slower)

    attention_slicing uses a chunked attention code path which can be faster
    on GPU architectures where Flash/memory-efficient attention is not available.
    """
    if use_multi_controlnet:
        from multy_control_net import build_multi_controlnet_pipe
        pipe, depth_detector = build_multi_controlnet_pipe(
            dtype, device,
            cpu_offload=cpu_offload,
            sequential_offload=sequential_offload,
        )
        if torch_compile:
            _apply_torch_compile(pipe)
        return pipe, depth_detector

    controlnet: ControlNetModel = ControlNetModel.from_pretrained(  # type: ignore[assignment]
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=dtype,
    )

    # With cpu_offload we do NOT call .to(device) manually —
    # the offload hooks move layers automatically during inference.
    if not (cpu_offload or sequential_offload):
        controlnet = controlnet.to(device)  # type: ignore[assignment]

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    if sequential_offload:
        pipe.enable_sequential_cpu_offload()
        print("  Memory mode: sequential CPU offload (~3 GB VRAM)")
    elif cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  Memory mode: model CPU offload (~4 GB VRAM, slower PCIe transfers)")
    else:
        pipe = pipe.to(device)

    # VAE slicing: decode latent rows one at a time → less peak VRAM, no quality loss
    pipe.enable_vae_slicing()

    _configure_attention(pipe, device, attention_slicing)

    if torch_compile:
        _apply_torch_compile(pipe)

    pipe.set_progress_bar_config(disable=True)
    return pipe, None


# ──────────────────────────────────────────────
# FRAME EXTRACTION
# ──────────────────────────────────────────────

def extract_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Read all frames (RGB, resized to WIDTH×HEIGHT) and return (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frames.append(frame)

    cap.release()
    return frames, fps


# ──────────────────────────────────────────────
# CONTROL IMAGES
# ──────────────────────────────────────────────

def make_canny(image_np: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def make_depth(image_np: np.ndarray, depth_detector) -> Image.Image:
    return depth_detector(Image.fromarray(image_np))


# ──────────────────────────────────────────────
# OPTICAL FLOW & MASK WARPING
# ──────────────────────────────────────────────

def compute_flow_farneback(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    # OpenCV stubs incorrectly mark `flow` as non-optional; None is valid at runtime.
    no_initial_flow: np.ndarray = np.zeros((*g1.shape, 2), dtype=np.float32)
    flow: np.ndarray = cv2.calcOpticalFlowFarneback(
        g1, g2, no_initial_flow, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def _remap(src: np.ndarray, flow: np.ndarray, interp: int) -> np.ndarray:
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(src, map_x, map_y, interp)


def warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    return _remap(mask, flow, cv2.INTER_NEAREST)


def warp_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    return _remap(image, flow, cv2.INTER_LINEAR)


# ──────────────────────────────────────────────
# MASK FEATHERING
# ──────────────────────────────────────────────

def feather_mask(mask_np: np.ndarray, radius: int = 12) -> np.ndarray:
    """
    Soften hard mask edges with a Gaussian blur so SDXL inpainting blends
    into the surroundings rather than cutting out sharply.

    Returns a uint8 mask (0-255) with smooth falloff near boundaries.
    """
    if radius < 1:
        return mask_np
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask_np.astype(np.float32), (ksize, ksize), radius / 2)
    return blurred.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# SCENE LIGHT ANALYSIS
# ──────────────────────────────────────────────

def analyze_scene_lighting(image_np: np.ndarray, mask_np: np.ndarray) -> str:
    """
    Analyse the pixels surrounding the mask and return comma-separated
    lighting descriptor terms ready to append to the SDXL prompt.

    Three axes are measured independently:
      • Luminance  → bright / dark / medium scene
      • Warmth     → warm golden / cool blue / neutral lighting
      • Contrast   → high contrast / soft diffused / dramatic shadows
    """
    mask_bool = mask_np > 128
    surround = ~mask_bool

    if surround.sum() < 200:
        return ""

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0][surround]
    mean_L = float(L.mean())
    std_L = float(L.std())

    rgb = image_np[surround].astype(np.float32)
    warmth = float(rgb[:, 0].mean() - rgb[:, 2].mean())  # R − B channel delta

    terms: list[str] = []

    # Luminance
    if mean_L > 175:
        terms.append("bright well-lit scene")
    elif mean_L < 70:
        terms.append("dark scene, low-key lighting")
    else:
        terms.append("medium natural exposure")

    # Color temperature
    if warmth > 25:
        terms.append("warm golden-hour lighting")
    elif warmth < -25:
        terms.append("cool blue-tinted atmosphere")
    else:
        terms.append("neutral white-balanced light")

    # Contrast
    if std_L > 55:
        terms.append("high contrast, dramatic hard shadows")
    elif std_L < 20:
        terms.append("soft diffused light, gentle shadows")
    else:
        terms.append("moderate contrast")

    return ", ".join(terms)


def estimate_light_angle(image_np: np.ndarray, mask_np: np.ndarray) -> float:
    """
    Estimate the dominant light direction in the scene by looking at the
    brightness gradient around the object.

    Returns an angle in degrees (0 = right, 90 = top, 180 = left, 270 = bottom).
    Defaults to 315 (top-left) if estimation fails.
    """
    mask_bool = mask_np > 128
    surround = ~mask_bool
    if surround.sum() < 400:
        return 315.0

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    # Average gradient direction in the surrounding region
    mean_gx = float(gx[surround].mean())
    mean_gy = float(gy[surround].mean())

    angle_rad = float(np.arctan2(-mean_gy, mean_gx))
    return float(np.degrees(angle_rad)) % 360.0


# ──────────────────────────────────────────────
# SHADOW SYNTHESIS
# ──────────────────────────────────────────────

def synthesize_shadow(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    light_angle_deg: float = 315.0,
    opacity: float = 0.35,
    offset_px: int = 10,
    blur_radius: int = 18,
) -> np.ndarray:
    """
    Add a soft synthetic drop-shadow beneath the masked object.

    The shadow is cast in the direction *opposite* to the light source,
    blurred so it feathers naturally into the scene.

    Args:
        light_angle_deg: direction FROM which light arrives
                         (0 = right, 90 = top, 180 = left, 270 = bottom).
        opacity:         max darkness of the shadow (0–1).
        offset_px:       how far to displace the shadow.
        blur_radius:     Gaussian blur radius for soft edges.
    """
    mask_bool = (mask_np > 128).astype(np.uint8) * 255
    if not mask_bool.any():
        return image_np

    # Shadow falls opposite the light
    shadow_angle = np.radians(light_angle_deg + 180.0)
    dx = int(np.cos(shadow_angle) * offset_px)
    dy = int(-np.sin(shadow_angle) * offset_px)   # image y-axis is flipped

    h, w = mask_bool.shape
    M: np.ndarray = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shadow_mask: np.ndarray = cv2.warpAffine(mask_bool, M, (w, h))

    # Soften edges
    ksize = blur_radius * 2 + 1
    shadow_soft = cv2.GaussianBlur(
        shadow_mask.astype(np.float32), (ksize, ksize), blur_radius / 2
    )

    # Don't darken where the object itself sits (shadow hidden under object)
    shadow_soft[mask_np > 128] = 0.0
    shadow_alpha = (shadow_soft / 255.0 * opacity)[..., np.newaxis]

    result = image_np.astype(np.float32) * (1.0 - shadow_alpha)
    return result.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# INPAINTING  (with optional IP-Adapter)
# ──────────────────────────────────────────────

def load_ip_adapter(
    pipe,
    scale: float = 0.6,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
    cpu_offload: bool = False,
    sequential_offload: bool = False,
) -> bool:
    """
    Load IP-Adapter SDXL weights into *pipe* for appearance-guided generation.
    Returns True on success, False if unavailable (network / weight issue).

    diffusers adds pipe.image_encoder *after* the pipeline is already on the
    GPU, so it lands on CPU by default and causes a device-mismatch error.
    We fix this by explicitly moving it to the correct device, or by
    re-applying the offloading hooks when memory offloading is active.
    """
    try:
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        pipe.set_ip_adapter_scale(scale)

        if sequential_offload:
            # Re-apply hooks so the newly added image_encoder is also covered.
            pipe.enable_sequential_cpu_offload()
        elif cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            # No offloading: move image_encoder to the same device/dtype as the rest.
            if getattr(pipe, "image_encoder", None) is not None:
                pipe.image_encoder = pipe.image_encoder.to(device=device, dtype=dtype)

        print(f"  IP-Adapter loaded  (scale={scale})")
        return True
    except Exception as e:
        print(f"  IP-Adapter unavailable ({e}). Skipping.")
        return False


def inpaint_frame(
    pipe,
    image_np: np.ndarray,
    mask_np: np.ndarray,
    control_image,
    prompt: str,
    negative_prompt: str,
    seed: int = 42,
    steps: int = 30,
    guidance_scale: float = 7.5,
    feather_radius: int = 12,
    ip_adapter_image: Image.Image | None = None,
    extra_prompt_terms: str = "",
) -> np.ndarray:
    """Inpaint one frame; returns RGB numpy array."""
    # Auto-append scene lighting analysis to guide SDXL style matching
    full_prompt = prompt
    if extra_prompt_terms:
        full_prompt = f"{prompt}, {extra_prompt_terms}"

    # Feather mask edges so SDXL blends smoothly at boundaries
    soft_mask = feather_mask(mask_np, radius=feather_radius)

    image_pil = Image.fromarray(image_np)
    mask_pil  = Image.fromarray(soft_mask)
    generator = torch.Generator(DEVICE).manual_seed(seed)

    extra_kwargs: dict = {}
    if ip_adapter_image is not None:
        extra_kwargs["ip_adapter_image"] = ip_adapter_image

    # Per-step ASCII progress bar shown inline during diffusion
    _bar_width = 25
    def _step_cb(pipe, step_idx: int, timestep, kwargs: dict) -> dict:
        filled = int((step_idx + 1) / steps * _bar_width)
        bar = "█" * filled + "░" * (_bar_width - filled)
        print(f"\r    [{bar}] {step_idx + 1}/{steps} steps", end="", flush=True)
        return kwargs

    with torch.no_grad():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=control_image,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
            output_type="pil",
            callback_on_step_end=_step_cb,
            **extra_kwargs,
        )
    print()  # newline after step bar

    return np.array(result.images[0])


# ──────────────────────────────────────────────
# TEMPORAL CONSISTENCY
# ──────────────────────────────────────────────

def apply_temporal_consistency(
    current_np: np.ndarray,
    prev_result_np: np.ndarray | None,
    mask_np: np.ndarray,
    flow: np.ndarray | None,
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Blend current inpainting with the warped previous result inside the mask.
    Higher alpha → more weight on the current frame (less smoothing).
    """
    if prev_result_np is None or flow is None:
        return current_np

    warped_prev = warp_image(prev_result_np, flow)
    mask_f = (mask_np[..., np.newaxis] / 255.0).astype(np.float32)

    blended = (
        alpha * current_np.astype(np.float32)
        + (1.0 - alpha) * warped_prev.astype(np.float32)
    )
    result = current_np.astype(np.float32) * (1.0 - mask_f) + blended * mask_f
    return result.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# POISSON SEAMLESS BLENDING
# ──────────────────────────────────────────────

def poisson_blend(
    original_np: np.ndarray,
    inpainted_np: np.ndarray,
    mask_np: np.ndarray,
) -> np.ndarray:
    """
    Blend the inpainted region into the original frame using Poisson seamless
    cloning (cv2.seamlessClone). This is the same technique used in professional
    compositing: it equalises gradients at the boundary so the seam becomes
    mathematically invisible.

    Works best when the inpainted object has similar luminance to the scene.
    Falls back to the inpainted result if the mask is too small.
    """
    mask_bool = mask_np > 128
    if mask_bool.sum() < 100:
        return inpainted_np

    # cv2.seamlessClone expects BGR images and a binary uint8 mask
    src_bgr  = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2BGR)
    dst_bgr  = cv2.cvtColor(original_np,  cv2.COLOR_RGB2BGR)
    mask_u8  = (mask_bool.astype(np.uint8)) * 255

    # Centre of the mask bounding box — required by seamlessClone
    ys, xs = np.where(mask_bool)
    cx = int((xs.min() + xs.max()) / 2)
    cy = int((ys.min() + ys.max()) / 2)

    try:
        blended_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_u8, (cx, cy),
                                        cv2.NORMAL_CLONE)
        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        # seamlessClone can fail when the mask touches the image border
        return inpainted_np


# ──────────────────────────────────────────────
# COLOR HARMONIZATION
# ──────────────────────────────────────────────

def color_harmonize(
    original_np: np.ndarray,
    inpainted_np: np.ndarray,
    mask_np: np.ndarray,
) -> np.ndarray:
    """
    Shift the color statistics of the inpainted region to match the
    surrounding context of the original frame (works in LAB space).
    """
    mask_bool = mask_np > 128
    surround = ~mask_bool

    if mask_bool.sum() < 10 or surround.sum() < 10:
        return inpainted_np

    orig_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    inp_lab = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    result_lab = inp_lab.copy()

    for c in range(3):
        surr_vals = orig_lab[..., c][surround]
        inp_vals = inp_lab[..., c][mask_bool]

        surr_mean, surr_std = surr_vals.mean(), surr_vals.std() + 1e-6
        inp_mean, inp_std = inp_vals.mean(), inp_vals.std() + 1e-6

        adjusted = (inp_vals - inp_mean) / inp_std * surr_std + surr_mean
        result_lab[..., c][mask_bool] = adjusted.clip(0, 255)

    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ──────────────────────────────────────────────
# GRAIN MATCHING
# ──────────────────────────────────────────────

def match_grain(
    inpainted_np: np.ndarray,
    reference_np: np.ndarray,
    mask_np: np.ndarray,
    strength: float = 0.4,
) -> np.ndarray:
    """
    Estimate high-frequency grain from the original frame and add
    a proportional amount to the inpainted region.
    """
    blurred = cv2.GaussianBlur(reference_np.astype(np.float32), (5, 5), 0)
    grain = reference_np.astype(np.float32) - blurred

    mask_f = (mask_np[..., np.newaxis] / 255.0).astype(np.float32)
    result = inpainted_np.astype(np.float32) + grain * mask_f * strength
    return result.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# MOTION BLUR MATCHING
# ──────────────────────────────────────────────

def match_motion_blur(
    inpainted_np: np.ndarray,
    mask_np: np.ndarray,
    flow: np.ndarray,
    blur_scale: float = 0.25,
) -> np.ndarray:
    """
    Blur the edges of the inpainted region proportionally to the local
    optical-flow magnitude so the boundary doesn't look artificially sharp.
    """
    mask_bool = mask_np > 128
    if not mask_bool.any():
        return inpainted_np

    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_flow = float(flow_mag[mask_bool].mean())
    kernel_size = int(mean_flow * blur_scale)

    if kernel_size < 1:
        return inpainted_np

    kernel_size = min(kernel_size | 1, 15)  # force odd, cap at 15

    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=2)
    edge_mask = ((mask_dilated > 128) & ~mask_bool).astype(np.float32)[..., np.newaxis]

    blurred = cv2.GaussianBlur(inpainted_np, (kernel_size, kernel_size), 0)
    result = (
        inpainted_np.astype(np.float32) * (1.0 - edge_mask)
        + blurred.astype(np.float32) * edge_mask
    )
    return result.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# VIDEO ASSEMBLY
# ──────────────────────────────────────────────

def assemble_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    imageio.mimsave(output_path, frames, fps=fps)  # type: ignore[call-overload]


# ──────────────────────────────────────────────
# MASK SELECTION  (click / text / auto)
# ──────────────────────────────────────────────

def _build_sam(args) -> object:
    from sam_load import build_sam2_predictor
    return build_sam2_predictor(args.sam2_checkpoint, args.sam2_config, device=DEVICE)


def _select_mask(
    args: argparse.Namespace,
    frames: list[np.ndarray],
) -> tuple[np.ndarray, int]:
    """
    Return (initial_mask, seed_frame_idx) for the object the user wants to replace.

    mode=click  → interactive click on frame 0 → SAM2
    mode=text   → Grounding DINO scans first --scan-frames frames → SAM2
    mode=auto   → Grounding DINO → preview → [accept | refine by click | redo]
    """
    from ui import select_click_point, confirm_mask, show_bbox_preview
    from sam_load import get_mask_from_click
    from detector import load_grounding_dino, detect_bbox, get_mask_from_bbox

    first_frame = frames[0]

    # Derive detect_prompt from the generation prompt if not given
    detect_prompt = args.detect_prompt or " ".join(args.prompt.split()[:3])

    def _scan_indices(n_frames: int, n_probes: int) -> list[int]:
        """Return up to n_probes evenly-spaced frame indices across [0, n_frames)."""
        if n_probes >= n_frames:
            return list(range(n_frames))
        step = n_frames / n_probes
        return [int(i * step) for i in range(n_probes)]

    def _best_detection(
        gdino, indices: list[int], prompt: str
    ) -> tuple[int, np.ndarray, tuple, str, float] | None:
        """
        Scan all frames in *indices*, collect every detection above threshold,
        return (frame_idx, frame, bbox, phrase, confidence) for the globally
        highest-confidence hit — or None if nothing found.

        Scanning ALL sampled frames (not stopping at first) ensures we don't
        accidentally pick a false positive that appears earlier but with lower
        confidence than the real target later in the video.
        """
        best: tuple[float, int, np.ndarray, tuple, str] | None = None
        n = len(frames)
        for fi in indices:
            result = detect_bbox(
                gdino, frames[fi], prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=DEVICE,
            )
            if result is None:
                continue
            bbox, phrase, conf = result
            print(f"    frame {fi:4d}/{n}: '{phrase}'  conf={conf:.3f}")
            if best is None or conf > best[0]:
                best = (conf, fi, frames[fi], bbox, phrase)

        if best is None:
            return None
        conf, fi, frame, bbox, phrase = best
        return fi, frame, bbox, phrase, conf

    # ── click ──────────────────────────────────────────────────────────────
    if args.mode == "click":
        print("Click mode: select the object in the window…")
        sam = _build_sam(args)
        while True:
            pt = select_click_point(first_frame)
            if pt is None:
                raise SystemExit("Selection cancelled.")
            mask = get_mask_from_click(sam, first_frame, pt[0], pt[1])
            decision = confirm_mask(first_frame, mask, "click mode")
            if decision == "accept":
                break
            elif decision == "cancel":
                raise SystemExit("Selection cancelled.")
        del sam
        torch.cuda.empty_cache()
        return mask, 0

    # ── text ───────────────────────────────────────────────────────────────
    if args.mode == "text":
        indices = _scan_indices(len(frames), args.scan_frames)
        print(f"Text mode: detecting \"{detect_prompt}\" "
              f"(scanning {len(indices)} evenly-spaced frames, picking best confidence)…")
        gdino = load_grounding_dino(args.gdino_config, args.gdino_checkpoint, device=DEVICE)
        sam = _build_sam(args)

        hit = _best_detection(gdino, indices, detect_prompt)

        if hit is None:
            del gdino, sam
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"Grounding DINO could not find \"{detect_prompt}\" "
                f"in {len(indices)} sampled frames.\n"
                f"Tips:\n"
                f"  • Try a simpler word: \"mug\", \"cup\", \"watch\", \"bottle\"\n"
                f"  • Lower thresholds: --box-threshold 0.2 --text-threshold 0.15\n"
                f"  • Probe more frames: --scan-frames 60\n"
                f"  • Switch mode: --mode click"
            )

        seed_idx, seed_frame, bbox, phrase, conf = hit
        print(f"  ★ Best: \"{phrase}\" on frame {seed_idx}  confidence={conf:.3f}")
        if conf < 0.40:
            print(f"  ⚠ Low confidence ({conf:.3f}). If the wrong object was selected,")
            print(f"    try --box-threshold 0.40 or --mode click.")

        mask = get_mask_from_bbox(sam, seed_frame, bbox)
        del gdino, sam
        torch.cuda.empty_cache()
        return mask, seed_idx

    # ── auto ───────────────────────────────────────────────────────────────
    # auto = text detection (best confidence across all sampled frames), then user confirms
    indices = _scan_indices(len(frames), args.scan_frames)
    print(f"Auto mode: detecting \"{detect_prompt}\" "
          f"(scanning {len(indices)} evenly-spaced frames, picking best confidence)…")
    gdino = load_grounding_dino(args.gdino_config, args.gdino_checkpoint, device=DEVICE)
    sam = _build_sam(args)

    # Find the globally best-confidence detection across all sampled frames
    hit = _best_detection(gdino, indices, detect_prompt)
    if hit is not None:
        seed_idx, seed_frame, _, _, conf = hit
        print(f"  ★ Best detection on frame {seed_idx}  confidence={conf:.3f}")
    else:
        seed_idx = 0
        seed_frame = first_frame

    mask = None
    while mask is None:
        result = detect_bbox(
            gdino, seed_frame, detect_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=DEVICE,
        )

        if result is None:
            print(f"  Not found on frame {seed_idx}. Enter a new description (or leave empty to switch to click):")
            new_prompt = input("  detect-prompt> ").strip()
            if not new_prompt:
                break  # fall through to click fallback
            detect_prompt = new_prompt
            continue

        bbox, phrase, conf = result

        # Show bounding box for quick sanity check
        decision = show_bbox_preview(seed_frame, bbox, phrase)
        if decision == "cancel":
            raise SystemExit("Selection cancelled.")
        elif decision == "redo":
            print("  Enter a new description:")
            detect_prompt = input("  detect-prompt> ").strip() or detect_prompt
            continue

        # Bounding box accepted → run SAM2
        mask = get_mask_from_bbox(sam, seed_frame, bbox)

        # Show full mask preview
        decision = confirm_mask(seed_frame, mask, f'"{phrase}"')
        if decision == "accept":
            break
        elif decision == "cancel":
            raise SystemExit("Selection cancelled.")
        elif decision == "refine":
            pt = select_click_point(seed_frame)
            if pt:
                from sam_load import get_mask_from_click
                mask = get_mask_from_click(sam, seed_frame, pt[0], pt[1])
                decision2 = confirm_mask(seed_frame, mask, "refined")
                if decision2 == "accept":
                    break
                elif decision2 == "cancel":
                    raise SystemExit("Selection cancelled.")
            mask = None  # redo
        else:  # "redo"
            mask = None
            print("  Enter a new description:")
            detect_prompt = input("  detect-prompt> ").strip() or detect_prompt

    if mask is None:
        # Fallback: manual click
        print("  Falling back to click mode…")
        while True:
            pt = select_click_point(seed_frame)
            if pt is None:
                raise SystemExit("Selection cancelled.")
            from sam_load import get_mask_from_click
            mask = get_mask_from_click(sam, seed_frame, pt[0], pt[1])
            decision = confirm_mask(seed_frame, mask, "click fallback")
            if decision == "accept":
                break
            elif decision == "cancel":
                raise SystemExit("Selection cancelled.")

    del gdino, sam
    torch.cuda.empty_cache()
    return mask, seed_idx


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI video object replacement",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # I/O
    p.add_argument("--input",  default="input.mp4",  help="Input video path")
    p.add_argument("--output", default="output.mp4", help="Output video path")
    p.add_argument("--device", default=None,
                   help="Compute device: cuda | mps | cpu (auto-detected by default)")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None,
                   help=(
                       "Model weight dtype. Default: fp16 on GPU, fp32 on CPU. "
                       "bf16 is recommended for Blackwell (RTX 5060+) — native hardware "
                       "support, same speed as fp16 but more numerically stable."
                   ))

    # ── Selection mode ──────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["click", "text", "auto"],
        default="auto",
        help=(
            "How to select the object to replace:\n"
            "  click  — open first frame, left-click on the object\n"
            "  text   — describe the object (--detect-prompt), fully automatic\n"
            "  auto   — text detection first, then interactive confirm/refine"
        ),
    )
    p.add_argument(
        "--detect-prompt",
        default=None,
        metavar="TEXT",
        help=(
            "What object to find (used by 'text' and 'auto' modes).\n"
            "Example: --detect-prompt \"ceramic mug\"\n"
            "Defaults to the first two words of --prompt if not set."
        ),
    )
    p.add_argument("--box-threshold",  type=float, default=0.82,
                   help="Grounding DINO box confidence threshold (lower = more detections)")
    p.add_argument("--text-threshold", type=float, default=0.77,
                   help="Grounding DINO text similarity threshold (lower = more detections)")
    p.add_argument("--scan-frames",    type=int,   default=30,
                   help="How many evenly-spaced frames to probe across the whole video. All are scanned and the highest-confidence detection wins.")
    p.add_argument("--fill-gaps",      type=int,   default=2,
                   help="Fill isolated empty-mask runs of this length or shorter (fixes SAM2 single-frame drop-outs)")

    # Diffusion
    p.add_argument("--prompt",          default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance-scale",  type=float, default=7.5)
    p.add_argument("--blend-alpha",     type=float, default=0.85,
                   help="Temporal blend alpha (1.0 = no blending)")
    p.add_argument("--feather-radius",  type=int,   default=12,
                   help="Gaussian feather radius for mask edges before inpainting (0 = hard mask)")
    p.add_argument("--no-poisson",      action="store_true",
                   help="Disable Poisson seamless blending (faster but seams more visible)")

    # Realism: IP-Adapter, light analysis, shadow
    p.add_argument("--no-ip-adapter",   action="store_true",
                   help="Disable IP-Adapter reference (enabled by default; first inpainted frame guides all others)")
    p.add_argument("--ip-adapter-scale", type=float, default=0.55,
                   help="IP-Adapter influence strength (0.0–1.0, default 0.55)")
    p.add_argument("--no-light-analysis", action="store_true",
                   help="Disable automatic scene lighting analysis appended to prompt")
    p.add_argument("--shadow-opacity",  type=float, default=0.35,
                   help="Drop shadow opacity (0.0 = no shadow, 1.0 = fully opaque)")
    p.add_argument("--shadow-offset",   type=int,   default=10,
                   help="Shadow displacement in pixels")
    p.add_argument("--light-angle",     type=float, default=None,
                   help="Override estimated light direction in degrees (0=right, 90=top, 180=left)")

    # Resolution (lower = faster; SDXL works best at 768–1024)
    p.add_argument("--width",  type=int, default=1024,
                   help="Processing width in pixels (default 1024). Use 768 for ~3x speedup on slow GPUs.")
    p.add_argument("--height", type=int, default=1024,
                   help="Processing height in pixels (default 1024). Use 768 for ~3x speedup on slow GPUs.")

    # Memory management
    p.add_argument("--cpu-offload", action="store_true",
                   help="Enable model CPU offload (~4 GB VRAM). Recommended for 8 GB GPUs.")
    p.add_argument("--sequential-offload", action="store_true",
                   help="Enable sequential CPU offload (~3 GB VRAM, slower). For 6 GB GPUs.")
    p.add_argument("--attention-slicing", action="store_true",
                   help="Enable attention slicing (alternative attention code path, helps on new GPU architectures).")
    p.add_argument("--torch-compile", action="store_true",
                   help=(
                       "JIT-compile UNet with torch.compile (mode=reduce-overhead). "
                       "Generates GPU-specific kernels at runtime — essential for new "
                       "architectures like Blackwell (RTX 5060) where Flash Attention is "
                       "not pre-compiled in PyTorch. First frame is slow (2-5 min warm-up), "
                       "subsequent frames run at full speed."
                   ))

    # ControlNet / flow
    p.add_argument("--multi-controlnet", action="store_true",
                   help="Use Canny + Depth ControlNet")
    p.add_argument("--use-raft", action="store_true",
                   help="Use RAFT optical flow (more accurate, needs RAFT repo)")
    p.add_argument("--raft-model", default="RAFT/models/raft-things.pth")

    # Model paths
    p.add_argument("--sam2-checkpoint", default="sam2_hiera_large.pt")
    p.add_argument("--sam2-config",     default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--gdino-config", default=None,
                   help="Path to Grounding DINO config (auto-detected from installed package if not set)")
    p.add_argument("--gdino-checkpoint", default="groundingdino_swint_ogc.pth")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Override device / dtype if explicitly requested
    global DEVICE, DTYPE
    if args.device:
        DEVICE = args.device
        DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

    if args.dtype == "bf16":
        DTYPE = torch.bfloat16
    elif args.dtype == "fp32":
        DTYPE = torch.float32
    elif args.dtype == "fp16":
        DTYPE = torch.float16
    elif _is_blackwell() and DTYPE == torch.float16:
        # Auto-switch to bfloat16 on Blackwell — native hardware support,
        # avoids rare fp16 NaN issues, same throughput
        DTYPE = torch.bfloat16
        print("  Auto: switched to bfloat16 (native Blackwell hardware)")

    # Apply resolution override (must happen before frame extraction)
    global WIDTH, HEIGHT
    WIDTH  = args.width
    HEIGHT = args.height

    print(f"Device: {DEVICE}  |  dtype: {DTYPE}  |  resolution: {WIDTH}×{HEIGHT}")

    # CUDA diagnostics: GPU capability, PyTorch version, attention backend status
    if DEVICE == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        pt_ver = torch.__version__
        cuda_ver = torch.version.cuda or "?"

        if cap[0] >= 12:
            # Blackwell: needs PyTorch 2.7+cu128 for native support
            pt_major = int(pt_ver.split(".")[0])
            pt_minor = int(pt_ver.split(".")[1].split("+")[0].split("a")[0].split("b")[0])
            has_cu128 = "cu128" in pt_ver and (pt_major, pt_minor) >= (2, 7)
            has_cu130 = "cu130" in pt_ver  # nightly cu130 also fully supports Blackwell
            has_native = has_cu128 or has_cu130
            if has_native:
                attention_note = f"PyTorch {pt_ver} — Blackwell natively supported ✓"
            else:
                attention_note = (
                    f"PyTorch {pt_ver} does NOT support sm_{cap[0]}{cap[1]} natively — "
                    "all CUDA ops fall back → extremely slow (30+ min/frame). "
                    "FIX: pip uninstall torch torchvision torchaudio -y && "
                    "pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu130"
                )
        else:
            attention_note = "supported"

        print(f"  GPU: {name}  sm_{cap[0]}{cap[1]}  PyTorch {pt_ver}  CUDA {cuda_ver}")
        if cap[0] >= 12:
            print(f"  Blackwell status: {attention_note}")

        # VRAM advice: tell user whether --cpu-offload is needed
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # bf16 SDXL UNet ~2.6 GB, ControlNet ~1.25 GB, VAE ~0.15 GB ≈ 4.0 GB
            # ip-adapter adds ~0.5 GB; leave 2 GB headroom for activations
            recommended_min = 6.0
            if total_vram_gb >= recommended_min:
                if args.cpu_offload or args.sequential_offload:
                    print(
                        f"  VRAM: {total_vram_gb:.1f} GB — enough to run WITHOUT --cpu-offload. "
                        "CPU offload transfers weights over PCIe each step → much slower. "
                        "Try removing --cpu-offload for a large speed-up."
                    )
                else:
                    print(f"  VRAM: {total_vram_gb:.1f} GB — direct GPU mode (fastest) ✓")
            else:
                if not (args.cpu_offload or args.sequential_offload):
                    print(
                        f"  VRAM: {total_vram_gb:.1f} GB — low VRAM; add --cpu-offload if you get OOM errors"
                    )
                else:
                    print(f"  VRAM: {total_vram_gb:.1f} GB — CPU offload enabled")
        except Exception:
            pass

    # ── 1. Frame extraction (first — frees us to plan memory) ─────────────
    print(f"Extracting frames from '{args.input}'…")
    frames, fps = extract_frames(args.input)
    print(f"  {len(frames)} frames @ {fps:.2f} FPS")

    # ── 2. Initial mask via SAM2 image + GroundingDINO ────────────────────
    initial_mask, seed_frame_idx = _select_mask(args, frames)

    # ── 3. SAM2 Video Predictor — track mask through all frames ───────────
    print("Tracking object through all frames with SAM2 Video Predictor…")
    from sam_video import build_video_predictor, track_masks_video

    video_predictor = build_video_predictor(
        args.sam2_checkpoint, args.sam2_config, device=DEVICE
    )
    tracked_masks = track_masks_video(video_predictor, frames, initial_mask, initial_frame_idx=seed_frame_idx)
    del video_predictor
    torch.cuda.empty_cache()

    # Fill short drop-out gaps (SAM2 occasionally loses confidence for 1-2 frames).
    from sam_video import fill_mask_gaps
    tracked_masks = fill_mask_gaps(tracked_masks, max_gap=args.fill_gaps)
    print(f"  Tracking done. "
          f"Object present in {sum(m.any() for m in tracked_masks)}/{len(frames)} frames.")

    # ── 4. Diffusion pipeline (loaded after SAM2 is freed) ────────────────
    print("Loading diffusion pipeline…")
    pipe, depth_detector = load_pipeline(
        args.multi_controlnet,
        cpu_offload=args.cpu_offload,
        sequential_offload=args.sequential_offload,
        attention_slicing=args.attention_slicing,
        torch_compile=args.torch_compile,
    )

    # IP-Adapter: load weights so first inpainted frame can guide all others
    ip_adapter_ok = False
    if not args.no_ip_adapter:
        ip_adapter_ok = load_ip_adapter(
            pipe,
            scale=args.ip_adapter_scale,
            device=DEVICE,
            dtype=DTYPE,
            cpu_offload=args.cpu_offload,
            sequential_offload=args.sequential_offload,
        )

    # ── 5. RAFT (optional, for temporal blending only) ────────────────────
    raft_model = None
    if args.use_raft:
        try:
            from raft_load import load_raft
            raft_model = load_raft(args.raft_model)
            print("RAFT optical flow loaded.")
        except Exception as e:
            print(f"RAFT unavailable ({e}), falling back to Farneback.")

    # ── 6. Frame loop ─────────────────────────────────────────────────────
    import time

    prev_result: np.ndarray | None = None
    ip_reference: Image.Image | None = None   # set after first successful inpaint
    processed: list[np.ndarray] = []

    n_frames = len(frames)
    n_with_object = sum(m.any() for m in tracked_masks)
    frames_done = 0          # counts only frames that go through diffusion
    loop_start = time.time()

    for i, frame in enumerate(frames):
        mask = tracked_masks[i]

        # Skip frames where object is not visible — no diffusion needed
        if not mask.any():
            print(f"  [{i + 1}/{n_frames}] skip (no object)", flush=True)
            processed.append(frame.copy())
            prev_result = None
            continue

        t0 = time.time()

        # Optical flow — only for temporal blending
        if i > 0:
            if raft_model is not None:
                from raft_load import compute_flow_raft
                flow = compute_flow_raft(raft_model, frames[i - 1], frame)
            else:
                flow = compute_flow_farneback(frames[i - 1], frame)
        else:
            flow = None

        # Build ControlNet guidance image(s)
        canny = make_canny(frame)
        if args.multi_controlnet and depth_detector is not None:
            depth = make_depth(frame, depth_detector)
            control_image = [canny, depth]
        else:
            control_image = canny

        # ── Scene light analysis → auto-enrich prompt ────────────────────
        scene_lighting = ""
        if not args.no_light_analysis:
            scene_lighting = analyze_scene_lighting(frame, mask)

        # ── SDXL inpainting ──────────────────────────────────────────────
        # Once IP-Adapter is loaded the UNet *always* needs image_embeds.
        # For the first frame there is no reference yet, so we set scale=0
        # and pass the current frame as a neutral placeholder (zero influence).
        # From the second frame onward the real reference is used at full scale.
        if ip_adapter_ok and ip_reference is None:
            pipe.set_ip_adapter_scale(0.0)
            ip_placeholder: Image.Image | None = Image.fromarray(frame)
        else:
            if ip_adapter_ok:
                pipe.set_ip_adapter_scale(args.ip_adapter_scale)
            ip_placeholder = None

        print(f"  [{i + 1}/{n_frames}] inpainting…", end="  ", flush=True)
        inpainted = inpaint_frame(
            pipe, frame, mask, control_image,
            args.prompt, args.negative_prompt,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            feather_radius=args.feather_radius,
            ip_adapter_image=(ip_reference if ip_reference is not None else ip_placeholder) if ip_adapter_ok else None,
            extra_prompt_terms=scene_lighting,
        )

        # Store the very first inpainted result as the IP-Adapter reference
        if ip_adapter_ok and ip_reference is None:
            ip_reference = Image.fromarray(inpainted)

        # ── Post-processing ──────────────────────────────────────────────

        # Shadow synthesis (applied to original frame, before compositing)
        if args.shadow_opacity > 0:
            light_angle = args.light_angle
            if light_angle is None:
                light_angle = estimate_light_angle(frame, mask)
            frame_with_shadow = synthesize_shadow(
                frame, mask,
                light_angle_deg=light_angle,
                opacity=args.shadow_opacity,
                offset_px=args.shadow_offset,
            )
        else:
            frame_with_shadow = frame

        # Poisson seamless blending
        if not args.no_poisson:
            inpainted = poisson_blend(frame_with_shadow, inpainted, mask)

        # Temporal consistency
        inpainted = apply_temporal_consistency(
            inpainted, prev_result, mask, flow, alpha=args.blend_alpha
        )

        # Color harmonization
        inpainted = color_harmonize(frame, inpainted, mask)

        # Grain matching
        inpainted = match_grain(inpainted, frame, mask)

        # Motion blur at mask boundary
        if flow is not None:
            inpainted = match_motion_blur(inpainted, mask, flow)

        prev_result = inpainted.copy()
        processed.append(inpainted)
        torch.cuda.empty_cache()

        frames_done += 1
        elapsed = time.time() - t0
        total_elapsed = time.time() - loop_start
        avg_sec = total_elapsed / frames_done
        remaining = (n_with_object - frames_done) * avg_sec
        mins, secs = divmod(int(remaining), 60)
        print(f"{elapsed:.1f}s/frame  |  ETA {mins}m {secs:02d}s  "
              f"({frames_done}/{n_with_object} object frames done)", flush=True)

    # ── 7. Assembly ───────────────────────────────────────────────────────
    print(f"Assembling '{args.output}'…")
    assemble_video(processed, args.output, fps=fps)
    print("Done.")


if __name__ == "__main__":
    main()
