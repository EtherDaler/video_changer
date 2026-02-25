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
WIDTH = 1024
HEIGHT = 1024

DEFAULT_PROMPT = (
    "realistic printed photo on ceramic mug surface, matching lighting"
)
DEFAULT_NEG_PROMPT = (
    "distorted object, unrealistic lighting, extra objects, blurry"
)


# ──────────────────────────────────────────────
# PIPELINE LOADING
# ──────────────────────────────────────────────

def load_pipeline(
    use_multi_controlnet: bool = False,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
):
    """Load SDXL inpainting pipeline with Canny (+ Depth) ControlNet."""
    if use_multi_controlnet:
        from multy_control_net import build_multi_controlnet_pipe
        return build_multi_controlnet_pipe(dtype, device)

    controlnet: ControlNetModel = ControlNetModel.from_pretrained(  # type: ignore[assignment]
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=dtype,
    )
    controlnet = controlnet.to(device)  # type: ignore[assignment]

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

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
# INPAINTING
# ──────────────────────────────────────────────

def inpaint_frame(
    pipe,
    image_np: np.ndarray,
    mask_np: np.ndarray,
    control_image,
    prompt: str,
    negative_prompt: str,
    seed: int = 42,
) -> np.ndarray:
    """Inpaint one frame; returns RGB numpy array."""
    image_pil = Image.fromarray(image_np)
    mask_pil = Image.fromarray(mask_np)
    generator = torch.Generator(DEVICE).manual_seed(seed)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=control_image,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator,
            output_type="pil",
        )

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
    from detector import load_grounding_dino, detect_bbox, get_mask_from_bbox, text_to_mask

    first_frame = frames[0]

    # Derive detect_prompt from the generation prompt if not given
    detect_prompt = args.detect_prompt or " ".join(args.prompt.split()[:3])

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
        print(f"Text mode: detecting \"{detect_prompt}\" (scanning up to {args.scan_frames} frames)…")
        gdino = load_grounding_dino(args.gdino_config, args.gdino_checkpoint, device=DEVICE)
        sam = _build_sam(args)

        mask, bbox, phrase, seed_idx = None, None, None, 0
        scan_limit = min(args.scan_frames, len(frames))
        for fi in range(scan_limit):
            mask, bbox, phrase = text_to_mask(
                gdino, sam, frames[fi], detect_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=DEVICE,
            )
            if mask is not None:
                seed_idx = fi
                break

        del gdino, sam
        torch.cuda.empty_cache()

        if mask is None:
            raise RuntimeError(
                f"Grounding DINO could not find \"{detect_prompt}\" "
                f"in the first {scan_limit} frames.\n"
                f"Tips:\n"
                f"  • Try a simpler word: \"mug\", \"cup\", \"watch\", \"bottle\"\n"
                f"  • Lower thresholds: --box-threshold 0.2 --text-threshold 0.15\n"
                f"  • Scan more frames: --scan-frames 30\n"
                f"  • Switch mode: --mode click"
            )
        print(f"  Detected: \"{phrase}\" on frame {seed_idx} at {bbox}")
        return mask, seed_idx

    # ── auto ───────────────────────────────────────────────────────────────
    # auto = text detection first (scans first N frames), then user confirms / refines
    print(f"Auto mode: detecting \"{detect_prompt}\" (scanning up to {args.scan_frames} frames)…")
    gdino = load_grounding_dino(args.gdino_config, args.gdino_checkpoint, device=DEVICE)
    sam = _build_sam(args)

    # Find first frame where the object is detected
    scan_limit = min(args.scan_frames, len(frames))
    seed_idx = 0
    seed_frame = first_frame
    for fi in range(scan_limit):
        probe = detect_bbox(
            gdino, frames[fi], detect_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=DEVICE,
        )
        if probe is not None:
            seed_idx = fi
            seed_frame = frames[fi]
            break

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

        bbox, phrase = result

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
    p.add_argument("--box-threshold",  type=float, default=0.35,
                   help="Grounding DINO box confidence threshold (lower = more detections)")
    p.add_argument("--text-threshold", type=float, default=0.25,
                   help="Grounding DINO text similarity threshold (lower = more detections)")
    p.add_argument("--scan-frames",    type=int,   default=10,
                   help="In text/auto mode: scan first N frames if object not on frame 0")

    # Diffusion
    p.add_argument("--prompt",          default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance-scale",  type=float, default=7.5)
    p.add_argument("--blend-alpha",     type=float, default=0.85,
                   help="Temporal blend alpha (1.0 = no blending)")

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
    if args.device:
        global DEVICE, DTYPE
        DEVICE = args.device
        DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

    print(f"Device: {DEVICE}  |  dtype: {DTYPE}")

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
    print(f"  Tracking done. "
          f"Object present in {sum(m.any() for m in tracked_masks)}/{len(frames)} frames.")

    # ── 4. Diffusion pipeline (loaded after SAM2 is freed) ────────────────
    print("Loading diffusion pipeline…")
    pipe, depth_detector = load_pipeline(args.multi_controlnet)

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
    prev_result: np.ndarray | None = None
    processed: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        print(f"  Frame {i + 1}/{len(frames)}", end="\r", flush=True)

        # Use SAM2-tracked mask (replaces optical flow mask warping)
        mask = tracked_masks[i]

        # Skip frames where object is not visible — no diffusion needed
        if not mask.any():
            processed.append(frame.copy())
            prev_result = None  # reset temporal state on gap
            continue

        # Optical flow — only for temporal blending on frames with the object
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

        # SDXL inpainting
        inpainted = inpaint_frame(
            pipe, frame, mask, control_image,
            args.prompt, args.negative_prompt, seed=args.seed,
        )

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

    print()  # newline after \r progress

    # ── 7. Assembly ───────────────────────────────────────────────────────
    print(f"Assembling '{args.output}'…")
    assemble_video(processed, args.output, fps=fps)
    print("Done.")


if __name__ == "__main__":
    main()
