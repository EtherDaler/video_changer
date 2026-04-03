# -*- coding: utf-8 -*-
"""
Restore + apply temporal patch using json (no raw line JSON hacks).
Clears cell outputs to avoid huge base64 breaking the file again.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "video_changer_colab_restored.ipynb"
DST = ROOT / "video_changer_colab.ipynb"

GDINO_BLOCK = r'''

def detect_all_bboxes(
    model,
    image_np: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
):
    """Все боксы Grounding DINO: список ((x1,y1,x2,y2), phrase, confidence)."""
    import torchvision.transforms as T
    from groundingdino.util.inference import predict

    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(Image.fromarray(image_np))
    h, w = image_np.shape[:2]
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE,
    )
    out = []
    for i in range(len(boxes)):
        cx, cy, bw, bh = boxes[i].tolist()
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        out.append(((x1, y1, x2, y2), phrases[i], float(logits[i])))
    return out


def bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter + 1e-6
    return inter / union


def bbox_axis_overlap_foreign(my_bbox, other_bbox, same_iou_threshold=0.75) -> bool:
    """Чужой бокс пересекает наш, но это не дубликат того же объекта (высокий IoU)."""
    if bbox_iou_xyxy(my_bbox, other_bbox) >= same_iou_threshold:
        return False
    ax1, ay1, ax2, ay2 = my_bbox
    bx1, by1, bx2, by2 = other_bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return ix2 > ix1 and iy2 > iy1


def detect_foreign_bbox_overlaps_mask(
    gdino_model,
    frame_rgb: np.ndarray,
    cross_prompt: str,
    mask_bbox_xyxy,
    box_threshold: float,
    text_threshold: float,
) -> bool:
    """Детекции по cross_prompt: bbox пересекается с bbox маски (не дубликат)."""
    if gdino_model is None or not (cross_prompt or "").strip():
        return False
    try:
        all_b = detect_all_bboxes(
            gdino_model,
            frame_rgb,
            cross_prompt.strip(),
            box_threshold,
            text_threshold,
        )
    except Exception:
        return False
    for obox, _, _ in all_b:
        if bbox_axis_overlap_foreign(mask_bbox_xyxy, obox):
            return True
    return False

'''

STATE_OLD = """class NanoTemporalState:
    \"\"\"Кэш последнего сгенерированного ROI и метрик ключевого кадра (после full/edit).\"\"\"

    __slots__ = (\"cached_gen_pil\", \"key_metrics\")

    def __init__(self):
        self.cached_gen_pil = None
        self.key_metrics = None

    def reset(self):
        self.cached_gen_pil = None
        self.key_metrics = None"""

STATE_NEW = """class NanoTemporalState:
    \"\"\"Кэш ROI, метрик маски и ссылок на освещение/фон для покадрового сравнения.\"\"\"

    __slots__ = (
        \"cached_gen_pil\",
        \"key_metrics\",
        \"prev_object_light_mean\",
        \"prev_bg_ring_rgb\",
        \"bg_cross_sticky_remaining\",
        \"prev_frame_metrics\",
        \"prev_ctx_motion_small\",
    )

    def __init__(self):
        self.cached_gen_pil = None
        self.key_metrics = None
        self.prev_object_light_mean = None
        self.prev_bg_ring_rgb = None
        self.bg_cross_sticky_remaining = 0
        self.prev_frame_metrics = None
        self.prev_ctx_motion_small = None

    def reset(self):
        self.cached_gen_pil = None
        self.key_metrics = None
        self.prev_object_light_mean = None
        self.prev_bg_ring_rgb = None
        self.bg_cross_sticky_remaining = 0
        self.prev_frame_metrics = None
        self.prev_ctx_motion_small = None"""

STATE_NEW_PRE_STICKY = """class NanoTemporalState:
    \"\"\"Кэш ROI, метрик маски и ссылок на освещение/фон для покадрового сравнения.\"\"\"

    __slots__ = (
        \"cached_gen_pil\",
        \"key_metrics\",
        \"prev_object_light_mean\",
        \"prev_bg_ring_rgb\",
    )

    def __init__(self):
        self.cached_gen_pil = None
        self.key_metrics = None
        self.prev_object_light_mean = None
        self.prev_bg_ring_rgb = None

    def reset(self):
        self.cached_gen_pil = None
        self.key_metrics = None
        self.prev_object_light_mean = None
        self.prev_bg_ring_rgb = None"""

HELPERS = """

def object_luminance_mean(frame_rgb, hard_mask_255):
    m = (hard_mask_255 > 127).astype(bool)
    if m.sum() < 50:
        return None
    r = frame_rgb[..., 0].astype(np.float64)
    g = frame_rgb[..., 1].astype(np.float64)
    b = frame_rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return float(y[m].mean())


def bg_ring_rgb_mean(frame_rgb, hard_mask_255, dilate_px, erode_px):
    outer = dilate_mask_u8(hard_mask_255, dilate_px)
    ek = max(3, erode_px * 2 + 1)
    inner = cv2.erode(hard_mask_255, np.ones((ek, ek), np.uint8), iterations=1)
    ring = cv2.subtract(outer, inner)
    if ring.sum() < 100:
        return None
    pix = frame_rgb[ring > 127]
    return pix.mean(axis=0).astype(np.float64)


def context_bg_motion_mad(ctx_rgb, ctx_mask_255, prev_small, max_side=128):
    # Mean abs RGB diff on context background (mask==0), downscaled. Returns (mad, small_rgb).
    h, w = ctx_rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    nh = max(32, int(round(h * scale)))
    nw = max(32, int(round(w * scale)))
    small = cv2.resize(ctx_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(ctx_mask_255, (nw, nh), interpolation=cv2.INTER_NEAREST)
    bg = msmall.astype(np.float32) < 127.0
    if bg.sum() < 50:
        return 0.0, small.copy()
    if prev_small is None or prev_small.shape != small.shape:
        return 0.0, small.copy()
    diff = np.abs(small.astype(np.float32) - prev_small.astype(np.float32))
    mad = float(diff[bg].mean())
    return mad, small.copy()


def pick_regen_reason(mask_full, lighting, bg_ring, bg_motion, frame_motion, cross):
    if mask_full:
        return "mask_large"
    if lighting:
        return "lighting"
    if bg_ring:
        return "bg_ring"
    if bg_motion:
        return "bg_motion"
    if frame_motion:
        return "frame_motion"
    if cross:
        return "bg_cross"
    return "mask_large"


def format_viewpoint_hint(key: dict | None, cur: dict) -> str:
    # Suffix from mask/bbox vs last keyframe (metrics_delta_vs_key).
    if key is None:
        return ""
    d_pos, d_log, d_ar = metrics_delta_vs_key(key, cur)
    return (
        " Viewpoint vs last keyframe: "
        f"center Δ {d_pos:.3f} (normalized), log-area Δ {d_log:.3f}, aspect Δ {d_ar:.3f}. "
        "Match the current mask silhouette; do not stretch or squash the object along one axis; "
        "keep proportions consistent with the reference image."
    )

"""

# Вставка в уже пропатченные ячейки без format_viewpoint_hint (миграция).
FORMAT_VIEWPOINT_HINT = """

def format_viewpoint_hint(key: dict | None, cur: dict) -> str:
    # Suffix from mask/bbox vs last keyframe (metrics_delta_vs_key).
    if key is None:
        return ""
    d_pos, d_log, d_ar = metrics_delta_vs_key(key, cur)
    return (
        " Viewpoint vs last keyframe: "
        f"center Δ {d_pos:.3f} (normalized), log-area Δ {d_log:.3f}, aspect Δ {d_ar:.3f}. "
        "Match the current mask silhouette; do not stretch or squash the object along one axis; "
        "keep proportions consistent with the reference image."
    )

"""

EDIT_FN = """

def build_nano_edit_prompt(old_object: str, new_object: str, view_hint: str = "") -> str:
    tail = view_hint.strip()
    if tail:
        tail = " " + tail
    ref_roles = (
        "Image inputs: (1) current scene crop for this frame; (2) mask; (3) reference — your previous render "
        "of the replacement object composited on the same context canvas. "
        "You MUST treat image (3) as the binding identity reference: same colors, materials, patterns, logos, "
        "shape and proportions as in (3). "
        "Re-project that object into the pose, scale, rotation and foreshortening required by (1) and the mask; "
        "do not invent a different product. "
    )
    return (
        build_nano_replace_prompt(old_object, new_object)
        + ref_roles
        + "Adapt to the camera view and perspective in (1). "
        "Rotate and foreshorten with the scene; do not mirror or flip the handle side. "
        "Only change the masked object; keep pixels outside the mask as in (1)."
    ) + tail

"""

REGEN_FN = """

def build_nano_regen_with_reference_prompt(
    old_object: str,
    new_object: str,
    reason: str,
    view_hint: str = "",
) -> str:
    tail = view_hint.strip()
    if tail:
        tail = " " + tail
    base = build_nano_replace_prompt(old_object, new_object)
    ref = (
        "Image inputs: (1) current scene crop; (2) mask; (3) reference — the previous generation of the replacement "
        "object on this canvas (sent as the third image in image_urls). "
        "You MUST use (3) as the sole identity/appearance reference: same colors, materials, patterns, logos, "
        "lettering, proportions and style as in (3). "
        "Relight and reintegrate for this frame as needed, but do not redesign the object — match (3) for what the object is. "
    )
    if reason == "lighting":
        return (
            base
            + ref
            + "Scene lighting on the object and nearby surfaces has changed. "
            "Relight the object to match the current frame illumination, shadows, and highlights in the first image. "
            "Adjust specular and diffuse brightness to match the scene; do not change the object's design or identity."
        ) + tail
    if reason == "bg_ring":
        return (
            base
            + ref
            + "The visible surroundings around the object (outside the mask) changed in color or brightness. "
            "Match the new background and integration; keep the object visually identical to the third image."
        ) + tail
    if reason == "bg_cross":
        return (
            base
            + ref
            + "Another object or foreground element may overlap the region; scene composition behind the mask changed. "
            "Integrate the object into the current scene without changing identity, texture, or color from the third image."
        ) + tail
    if reason == "bg_motion":
        return (
            base
            + ref
            + "Background pixels outside the mask moved between frames (motion in the scene). "
            "Redraw the visible background and how the object sits in it; keep the object identical to the third image."
        ) + tail
    if reason == "frame_motion":
        return (
            base
            + ref
            + "The object mask bbox or shape changed quickly from the previous frame (motion / camera / pose). "
            "Match the current frame perspective and integration; keep the object identical to the third image."
        ) + tail
    return (
        base
        + ref
        + "The pose, size, or perspective of the object in the mask changed significantly. "
        "Adapt rotation, scale, and foreshortening to match the scene; do not mirror or flip the handle side. "
        "Keep the object's identity identical to the third image where possible."
    ) + tail

"""

# Full replace_object body — same as before (MARKER comments optional — omit to reduce noise)
REPLACE_FN = r'''def replace_object_roi_nano_temporal(
    frame_rgb,
    mask_uint8,
    state: NanoTemporalState,
    *,
    old_object_prompt,
    new_object_prompt,
    api_key,
    seed,
    roi_padding=72,
    context_padding=None,
    max_roi_side=896,
    mask_dilate=13,
    mask_feather=11,
    resolution="0.5K",
    use_match_domain=True,
    use_match_domain_on_reuse=False,
    poll_interval=2.0,
    timeout_sec=600,
    stable_pos=0.035,
    stable_ar=0.07,
    reuse_max_log_area=0.30,
    edit_max_pos=0.20,
    edit_max_log_area=0.45,
    edit_max_ar=0.14,
    full_regen_pos=0.35,
    full_regen_log_area=0.75,
    mask_cover_extra_dilate=8,
    small_mask_area_boost_dilate=6,
    small_mask_area_ratio=0.0015,
    blend_mask_dilate_px=12,
    blend_dilate_px_bottom=2,
    composite_feather_ks=48,
    use_poisson_blend=False,
    gdino_model=None,
    bg_cross_prompt="",
    bg_cross_box_threshold=0.35,
    bg_cross_text_threshold=0.25,
    bg_cross_sticky_frames=8,
    check_object_lighting=True,
    object_light_rel_delta=0.08,
    check_bg_ring=True,
    bg_ring_dilate_px=32,
    bg_ring_erode_px=10,
    bg_ring_rgb_delta=12.0,
    reuse_max_d_ar=0.025,
    check_bg_motion=True,
    motion_bg_mad_threshold=5.0,
    motion_context_max_side=128,
    check_frame_motion=True,
    frame_mv_pos=0.012,
    frame_mv_log=0.03,
    frame_mv_ar=0.015,
):
    """Reuse / edit / full; смена света/фона/движение/пересечение bbox — регенерация с ref и уточнённым промптом."""
    # См. вложенную _submit_full_with_optional_ref ниже (после _fetch_composite).
    h, w = frame_rgb.shape[:2]
    hard_mask, _ = refine_mask(mask_uint8, dilate_px=mask_dilate, feather_px=mask_feather)
    if mask_cover_extra_dilate > 0:
        hard_mask = dilate_mask_u8(hard_mask, mask_cover_extra_dilate)
    _area_ratio = float(hard_mask.sum()) / (255.0 * h * w + 1e-6)
    if _area_ratio < small_mask_area_ratio and small_mask_area_boost_dilate > 0:
        hard_mask = dilate_mask_u8(hard_mask, small_mask_area_boost_dilate)

    cur_metrics = mask_metrics_from_hard(hard_mask, h, w)
    mb = mask_bbox(hard_mask)
    if mb is None or not hard_mask.any() or cur_metrics is None:
        return frame_rgb.copy()

    frame_motion = False
    if check_frame_motion and state.prev_frame_metrics is not None:
        fp, fl, fa = metrics_delta_vs_key(state.prev_frame_metrics, cur_metrics)
        if fp >= frame_mv_pos or fl >= frame_mv_log or fa >= frame_mv_ar:
            frame_motion = True

    cur_light = object_luminance_mean(frame_rgb, hard_mask)
    cur_ring = bg_ring_rgb_mean(frame_rgb, hard_mask, bg_ring_dilate_px, bg_ring_erode_px)

    lighting_changed = False
    if check_object_lighting and state.prev_object_light_mean is not None and cur_light is not None:
        prev = state.prev_object_light_mean
        if prev > 1e-6 and abs(cur_light - prev) / prev >= object_light_rel_delta:
            lighting_changed = True

    bg_ring_changed = False
    if check_bg_ring and state.prev_bg_ring_rgb is not None and cur_ring is not None:
        if float(np.linalg.norm(cur_ring - state.prev_bg_ring_rgb)) >= bg_ring_rgb_delta:
            bg_ring_changed = True

    cross_overlap = detect_foreign_bbox_overlaps_mask(
        gdino_model,
        frame_rgb,
        bg_cross_prompt,
        mb,
        bg_cross_box_threshold,
        bg_cross_text_threshold,
    )
    if cross_overlap:
        state.bg_cross_sticky_remaining = max(
            state.bg_cross_sticky_remaining, bg_cross_sticky_frames
        )
    sticky_cross = state.bg_cross_sticky_remaining > 0

    ctx_pad = context_padding if context_padding is not None else roi_padding
    rx1, ry1, rx2, ry2 = expand_bbox(mb, w, h, pad=roi_padding)
    cx1, cy1, cx2, cy2 = expand_bbox(mb, w, h, pad=ctx_pad)

    roi_img = frame_rgb[ry1:ry2, rx1:rx2].copy()
    roi_hard = hard_mask[ry1:ry2, rx1:rx2].copy()
    ctx_img = frame_rgb[cy1:cy2, cx1:cx2].copy()
    ctx_mask = hard_mask[cy1:cy2, cx1:cx2].copy()

    motion_mad = 0.0
    ctx_motion_small = None
    if check_bg_motion:
        motion_mad, ctx_motion_small = context_bg_motion_mad(
            ctx_img,
            ctx_mask,
            state.prev_ctx_motion_small,
            motion_context_max_side,
        )
    motion_bg_changed = bool(
        check_bg_motion
        and state.prev_ctx_motion_small is not None
        and motion_mad >= motion_bg_mad_threshold
    )

    crw, crh = ctx_img.shape[1], ctx_img.shape[0]
    crw, crh = clamp_size_keep_aspect(crw, crh, max_side=max_roi_side)
    target_w = snap_to_multiple(crw, base=16, min_v=256)
    target_h = snap_to_multiple(crh, base=16, min_v=256)
    roi_img_pil = Image.fromarray(ctx_img).resize((target_w, target_h), Image.Resampling.LANCZOS)
    roi_mask_pil = Image.fromarray(ctx_mask).resize((target_w, target_h), Image.Resampling.NEAREST)

    _blend_kw = {
        "blend_dilate_px": 0,
        "feather_ks": composite_feather_ks,
        "edge_band_ks": 11,
        "use_poisson": use_poisson_blend,
    }
    roi_hard_blended = dilate_mask_asymmetric(
        roi_hard,
        px_h=blend_mask_dilate_px,
        px_v_top=blend_mask_dilate_px,
        px_v_bottom=blend_dilate_px_bottom,
    )

    def composite_from_cached(cached_pil: Image.Image):
        gen_roi_np = np.array(cached_pil.resize((roi_img.shape[1], roi_img.shape[0]), Image.Resampling.LANCZOS))
        if use_match_domain_on_reuse:
            gen_roi_np = match_domain(gen_roi_np, roi_img)
        return alpha_insert_smooth(
            frame_rgb.copy(), gen_roi_np, roi_hard_blended, rx1, ry1, rx2, ry2, **_blend_kw
        )

    def _fetch_composite(rid):
        return _nano_fetch_and_composite(
            frame_rgb,
            roi_img,
            roi_hard_blended,
            rx1,
            ry1,
            rx2,
            ry2,
            cx1,
            cy1,
            cx2,
            cy2,
            rid,
            api_key,
            use_match_domain,
            poll_interval,
            timeout_sec,
            blend_kw=_blend_kw,
        )

    # MARKER _submit_full_with_optional_ref
    def _submit_full_with_optional_ref(prompt_text):
        # Третье изображение в image_urls — прошлая генерация (identity), если кэш есть.
        ref = None
        if state.cached_gen_pil is not None:
            ref = paste_narrow_ref_on_context_canvas(
                roi_img_pil,
                state.cached_gen_pil,
                rx1,
                ry1,
                rx2,
                ry2,
                cx1,
                cy1,
                cx2,
                cy2,
            )
        rid = nano_banana_submit(
            roi_img_pil,
            roi_mask_pil,
            prompt_text,
            seed,
            api_key,
            resolution,
            reference_pil=ref,
        )
        return _fetch_composite(rid)

    def _finalize_frame(out_img):
        if cur_light is not None:
            state.prev_object_light_mean = cur_light
        if cur_ring is not None:
            state.prev_bg_ring_rgb = cur_ring.copy()
        if state.bg_cross_sticky_remaining > 0:
            state.bg_cross_sticky_remaining -= 1
        state.prev_frame_metrics = cur_metrics
        if ctx_motion_small is not None:
            state.prev_ctx_motion_small = ctx_motion_small
        return out_img

    # --- Перегенерация: key_metrics = якорь только после full (первый кадр, mask_full|env_force, fallback full).
    #     Ветка edit обновляет только cached_gen_pil — иначе ракурс «схлопывается» к прошлому кадру.
    #     reuse: только composite без API. Порядок проверок: mask_full|env_force → reuse → edit → full.
    key = state.key_metrics
    if key is None:
        prompt = build_nano_replace_prompt(old_object_prompt, new_object_prompt)
        out, pil_cache = _submit_full_with_optional_ref(prompt)
        state.cached_gen_pil = pil_cache
        state.key_metrics = cur_metrics
        return _finalize_frame(out)

    d_pos, d_log, d_ar = metrics_delta_vs_key(key, cur_metrics)
    mask_full = d_pos >= full_regen_pos or d_log >= full_regen_log_area

    view_hint = format_viewpoint_hint(key, cur_metrics)
    env_force = bool(
        lighting_changed
        or bg_ring_changed
        or cross_overlap
        or sticky_cross
        or motion_bg_changed
        or frame_motion
    )
    reason = pick_regen_reason(
        mask_full,
        lighting_changed,
        bg_ring_changed,
        motion_bg_changed,
        frame_motion,
        cross_overlap or sticky_cross,
    )

    if mask_full or env_force:
        use_ref = state.cached_gen_pil is not None
        if use_ref:
            prompt = build_nano_regen_with_reference_prompt(
                old_object_prompt, new_object_prompt, reason, view_hint=view_hint
            )
        else:
            prompt = build_nano_replace_prompt(old_object_prompt, new_object_prompt)
        out, pil_cache = _submit_full_with_optional_ref(prompt)
        state.cached_gen_pil = pil_cache
        state.key_metrics = cur_metrics
        return _finalize_frame(out)

    if (
        state.cached_gen_pil is not None
        and d_pos < stable_pos
        and d_ar < reuse_max_d_ar
        and d_log < reuse_max_log_area
    ):
        return _finalize_frame(composite_from_cached(state.cached_gen_pil))

    if (
        state.cached_gen_pil is not None
        and d_pos < edit_max_pos
        and d_log < edit_max_log_area
        and d_ar < edit_max_ar
    ):
        ref = paste_narrow_ref_on_context_canvas(
            roi_img_pil,
            state.cached_gen_pil,
            rx1,
            ry1,
            rx2,
            ry2,
            cx1,
            cy1,
            cx2,
            cy2,
        )
        prompt = build_nano_edit_prompt(
            old_object_prompt, new_object_prompt, view_hint=view_hint
        )
        rid = nano_banana_submit(
            roi_img_pil,
            roi_mask_pil,
            prompt,
            seed,
            api_key,
            resolution,
            reference_pil=ref,
        )
        out, pil_cache = _fetch_composite(rid)
        state.cached_gen_pil = pil_cache
        return _finalize_frame(out)

    use_ref = state.cached_gen_pil is not None
    if use_ref:
        prompt = build_nano_regen_with_reference_prompt(
            old_object_prompt, new_object_prompt, "mask_large", view_hint=view_hint
        )
    else:
        prompt = build_nano_replace_prompt(old_object_prompt, new_object_prompt)
    out, pil_cache = _submit_full_with_optional_ref(prompt)
    state.cached_gen_pil = pil_cache
    state.key_metrics = cur_metrics
    return _finalize_frame(out)
'''


def cell_to_str(cell):
    s = cell.get("source", "")
    if isinstance(s, list):
        return "".join(s)
    return s


def str_to_cell_source(s: str):
    """Split into lines for ipynb source array."""
    lines = s.splitlines(keepends=True)
    if not lines:
        return []
    return lines


def patch_pipeline_cell(src: str) -> str:
    if "def detect_all_bboxes" not in src:
        src = src.replace(
            "    return (x1, y1, x2, y2), phrases[best], confidence\n\n\n# ── SAM2 Image Predictor",
            "    return (x1, y1, x2, y2), phrases[best], confidence\n" + GDINO_BLOCK + "\n# ── SAM2 Image Predictor",
        )
    if "prev_object_light_mean" not in src:
        src = src.replace(STATE_OLD, STATE_NEW)
    elif "bg_cross_sticky_remaining" not in src and STATE_NEW_PRE_STICKY in src:
        src = src.replace(STATE_NEW_PRE_STICKY, STATE_NEW, 1)
    if "prev_frame_metrics" not in src and '"bg_cross_sticky_remaining"' in src:
        src = src.replace(
            '        "bg_cross_sticky_remaining",\n    )',
            '        "bg_cross_sticky_remaining",\n        "prev_frame_metrics",\n        "prev_ctx_motion_small",\n    )',
            1,
        )
    if "self.prev_frame_metrics" not in src and "self.bg_cross_sticky_remaining = 0" in src:
        src = src.replace(
            "        self.bg_cross_sticky_remaining = 0\n\n    def reset(self):",
            "        self.bg_cross_sticky_remaining = 0\n        self.prev_frame_metrics = None\n        self.prev_ctx_motion_small = None\n\n    def reset(self):",
            1,
        )
        src = src.replace(
            "        self.bg_cross_sticky_remaining = 0\n\n\n\ndef mask_metrics_from_hard",
            "        self.bg_cross_sticky_remaining = 0\n        self.prev_frame_metrics = None\n        self.prev_ctx_motion_small = None\n\n\n\ndef mask_metrics_from_hard",
            1,
        )
    if "def object_luminance_mean" not in src:
        src = src.replace(
            "    return d_pos, d_log, d_ar\n\n\ndef nano_banana_submit",
            "    return d_pos, d_log, d_ar\n" + HELPERS + "\ndef nano_banana_submit",
            1,
        )
    elif "def format_viewpoint_hint" not in src and "def pick_regen_reason" in src:
        src = src.replace(
            '    return "mask_large"\n\n\ndef nano_banana_submit',
            '    return "mask_large"\n' + FORMAT_VIEWPOINT_HINT + "\ndef nano_banana_submit",
            1,
        )
    if "def context_bg_motion_mad" not in src and "def bg_ring_rgb_mean" in src:
        src = src.replace(
            "    return pix.mean(axis=0).astype(np.float64)\n\n\ndef pick_regen_reason",
            "    return pix.mean(axis=0).astype(np.float64)\n\n"
            "def context_bg_motion_mad(ctx_rgb, ctx_mask_255, prev_small, max_side=128):\n"
            "    # Mean abs RGB diff on context background (mask==0), downscaled. Returns (mad, small_rgb).\n"
            "    h, w = ctx_rgb.shape[:2]\n"
            "    scale = min(1.0, float(max_side) / float(max(h, w)))\n"
            "    nh = max(32, int(round(h * scale)))\n"
            "    nw = max(32, int(round(w * scale)))\n"
            "    small = cv2.resize(ctx_rgb, (nw, nh), interpolation=cv2.INTER_AREA)\n"
            "    msmall = cv2.resize(ctx_mask_255, (nw, nh), interpolation=cv2.INTER_NEAREST)\n"
            "    bg = msmall.astype(np.float32) < 127.0\n"
            "    if bg.sum() < 50:\n"
            "        return 0.0, small.copy()\n"
            "    if prev_small is None or prev_small.shape != small.shape:\n"
            "        return 0.0, small.copy()\n"
            "    diff = np.abs(small.astype(np.float32) - prev_small.astype(np.float32))\n"
            "    mad = float(diff[bg].mean())\n"
            "    return mad, small.copy()\n\n\n"
            "def pick_regen_reason",
            1,
        )
    if (
        "def pick_regen_reason(mask_full, lighting, bg_ring, cross):" in src
        and "frame_motion" not in src
    ):
        src = src.replace(
            'def pick_regen_reason(mask_full, lighting, bg_ring, cross):\n'
            '    if mask_full:\n'
            '        return "mask_large"\n'
            '    if lighting:\n'
            '        return "lighting"\n'
            '    if bg_ring:\n'
            '        return "bg_ring"\n'
            '    if cross:\n'
            '        return "bg_cross"\n'
            '    return "mask_large"',
            'def pick_regen_reason(mask_full, lighting, bg_ring, bg_motion, frame_motion, cross):\n'
            '    if mask_full:\n'
            '        return "mask_large"\n'
            '    if lighting:\n'
            '        return "lighting"\n'
            '    if bg_ring:\n'
            '        return "bg_ring"\n'
            '    if bg_motion:\n'
            '        return "bg_motion"\n'
            '    if frame_motion:\n'
            '        return "frame_motion"\n'
            '    if cross:\n'
            '        return "bg_cross"\n'
            '    return "mask_large"',
            1,
        )
    pat_edit_regen = re.compile(
        r"def build_nano_edit_prompt\([\s\S]*?\n\n(?=def dilate_mask_asymmetric)",
        re.MULTILINE,
    )
    if pat_edit_regen.search(src):
        src = pat_edit_regen.sub(
            EDIT_FN.strip() + "\n\n" + REGEN_FN.strip() + "\n\n",
            src,
            count=1,
        )
    old_nb_retry = (
        "    r = requests.post(NANO_BANANA_URL, headers=headers, data=data, files=files, timeout=120)\n"
        "    if r.status_code >= 400 and reference_pil is not None:\n"
        "        r = requests.post(\n"
        "            NANO_BANANA_URL,\n"
        "            headers=headers,\n"
        "            data=data,\n"
        "            files=files[:2],\n"
        "            timeout=120,\n"
        "        )\n"
        "    r.raise_for_status()\n"
    )
    new_nb_retry = (
        "    r = requests.post(NANO_BANANA_URL, headers=headers, data=data, files=files, timeout=120)\n"
        "    r.raise_for_status()\n"
    )
    if old_nb_retry in src:
        src = src.replace(old_nb_retry, new_nb_retry, 1)
    # replace replace_object function (non-greedy through closing print)
    pat = re.compile(
        r"def replace_object_roi_nano_temporal\([\s\S]*?\nprint\(\"✅ Все функции готовы!\"\)",
    )
    m = pat.search(src)
    if not m:
        raise RuntimeError("replace_object_roi_nano_temporal block not found")
    src = pat.sub(
        REPLACE_FN.rstrip() + "\nprint(\"✅ Все функции готовы!\")",
        src,
        count=1,
    )
    return src


def patch_params_cell(src: str) -> str:
    if "MOTION_BG_MAD_THRESHOLD" not in src and "BG_RING_RGB_DELTA = 12.0" in src:
        src = src.replace(
            "BG_RING_RGB_DELTA = 12.0\n\n# API\n",
            "BG_RING_RGB_DELTA = 12.0\n\n"
            "# Покадровое движение (фон контекста вне маски; метрики маски между кадрами)\n"
            "CHECK_BG_MOTION = True\n"
            "MOTION_BG_MAD_THRESHOLD = 5.0\n"
            "MOTION_CONTEXT_MAX_SIDE = 128\n"
            "CHECK_FRAME_MOTION = True\n"
            "FRAME_MV_POS = 0.012\n"
            "FRAME_MV_LOG = 0.03\n"
            "FRAME_MV_AR = 0.015\n\n"
            "# API\n",
            1,
        )
    if "COMPOSITE_FEATHER_KS = 71" in src:
        src = src.replace("COMPOSITE_FEATHER_KS = 71", "COMPOSITE_FEATHER_KS = 55", 1)
    if "TEMP_REUSE_MAX_D_AR" not in src and "TEMP_REUSE_MAX_LOG_AREA = 0.30" in src:
        src = src.replace(
            "TEMP_REUSE_MAX_LOG_AREA = 0.30\n",
            "TEMP_REUSE_MAX_LOG_AREA = 0.30\n"
            "TEMP_REUSE_MAX_D_AR = 0.025  # жёстче TEMP_STABLE_AR: меньше растяжения при reuse cached ROI\n",
            1,
        )
    if "BG_CROSS_STICKY_FRAMES" not in src and "BG_CROSS_TEXT_THRESHOLD = 0.25" in src:
        src = src.replace(
            "BG_CROSS_TEXT_THRESHOLD = 0.25\n",
            "BG_CROSS_TEXT_THRESHOLD = 0.25\n"
            "BG_CROSS_STICKY_FRAMES = 8  # после пересечения bbox ещё N кадров реген с ref\n",
            1,
        )
    if "BG_CROSS_PROMPT" in src:
        return src
    old = "TEMP_FULL_LOG_AREA = 0.75\n\n# API\n"
    new = (
        "TEMP_FULL_LOG_AREA = 0.75\n\n"
        "# Регенерация: свет / фон / пересечение bbox (Grounding DINO)\n"
        'BG_CROSS_PROMPT = ""  # напр. hand . person . face — пусто = без DINO на кадре\n'
        "BG_CROSS_BOX_THRESHOLD = 0.35\n"
        "BG_CROSS_TEXT_THRESHOLD = 0.25\n"
        "BG_CROSS_STICKY_FRAMES = 8  # после пересечения bbox ещё N кадров реген с ref\n"
        "CHECK_OBJECT_LIGHTING = True\n"
        "OBJECT_LIGHT_REL_DELTA = 0.08\n"
        "CHECK_BG_RING = True\n"
        "BG_RING_DILATE_PX = 32\n"
        "BG_RING_ERODE_PX = 10\n"
        "BG_RING_RGB_DELTA = 12.0\n\n"
        "# Покадровое движение (фон контекста вне маски; метрики маски между кадрами)\n"
        "CHECK_BG_MOTION = True\n"
        "MOTION_BG_MAD_THRESHOLD = 5.0\n"
        "MOTION_CONTEXT_MAX_SIDE = 128\n"
        "CHECK_FRAME_MOTION = True\n"
        "FRAME_MV_POS = 0.012\n"
        "FRAME_MV_LOG = 0.03\n"
        "FRAME_MV_AR = 0.015\n\n"
        "# API\n"
    )
    if old not in src:
        raise RuntimeError("params anchor not found")
    return src.replace(old, new, 1)


def patch_step6_cell(src: str) -> str:
    if "check_bg_motion=CHECK_BG_MOTION" not in src and "bg_ring_rgb_delta=BG_RING_RGB_DELTA" in src:
        ins = (
            "        bg_ring_rgb_delta=BG_RING_RGB_DELTA,\n"
            "        check_bg_motion=CHECK_BG_MOTION,\n"
            "        motion_bg_mad_threshold=MOTION_BG_MAD_THRESHOLD,\n"
            "        motion_context_max_side=MOTION_CONTEXT_MAX_SIDE,\n"
            "        check_frame_motion=CHECK_FRAME_MOTION,\n"
            "        frame_mv_pos=FRAME_MV_POS,\n"
            "        frame_mv_log=FRAME_MV_LOG,\n"
            "        frame_mv_ar=FRAME_MV_AR,\n"
        )
        src = src.replace("        bg_ring_rgb_delta=BG_RING_RGB_DELTA,\n", ins, 1)
    if "reuse_max_d_ar=TEMP_REUSE_MAX_D_AR" not in src and "reuse_max_log_area=TEMP_REUSE_MAX_LOG_AREA" in src:
        src = src.replace(
            "        reuse_max_log_area=TEMP_REUSE_MAX_LOG_AREA,\n",
            "        reuse_max_log_area=TEMP_REUSE_MAX_LOG_AREA,\n"
            "        reuse_max_d_ar=TEMP_REUSE_MAX_D_AR,\n",
            1,
        )
    if "gdino_cross" in src:
        if "bg_cross_sticky_frames=BG_CROSS_STICKY_FRAMES" not in src and "bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD" in src:
            src = src.replace(
                "        bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD,\n",
                "        bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD,\n"
                "        bg_cross_sticky_frames=BG_CROSS_STICKY_FRAMES,\n",
                1,
            )
        return src
    old = "temporal_state = NanoTemporalState()\nprev_mask_any = False\n\nfor i, frame in enumerate(frames):"
    new = (
        "temporal_state = NanoTemporalState()\nprev_mask_any = False\n\n"
        "gdino_cross = None\n"
        'if (BG_CROSS_PROMPT or "").strip():\n'
        '    print("  Загружаю Grounding DINO для проверки пересечений (BG_CROSS_PROMPT)...")\n'
        "    gdino_cross = load_grounding_dino(GDINO_CHECKPOINT)\n\n\n"
        "for i, frame in enumerate(frames):"
    )
    if old not in src:
        raise RuntimeError("step6 loop anchor not found")
    src = src.replace(old, new, 1)

    old_call = (
        "        composite_feather_ks=COMPOSITE_FEATHER_KS,\n"
        "        use_poisson_blend=USE_POISSON_BLEND,\n"
        "    )\n"
    )
    new_call = (
        "        composite_feather_ks=COMPOSITE_FEATHER_KS,\n"
        "        use_poisson_blend=USE_POISSON_BLEND,\n"
        "        gdino_model=gdino_cross,\n"
        "        bg_cross_prompt=BG_CROSS_PROMPT,\n"
        "        bg_cross_box_threshold=BG_CROSS_BOX_THRESHOLD,\n"
        "        bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD,\n"
        "        bg_cross_sticky_frames=BG_CROSS_STICKY_FRAMES,\n"
        "        check_object_lighting=CHECK_OBJECT_LIGHTING,\n"
        "        object_light_rel_delta=OBJECT_LIGHT_REL_DELTA,\n"
        "        check_bg_ring=CHECK_BG_RING,\n"
        "        bg_ring_dilate_px=BG_RING_DILATE_PX,\n"
        "        bg_ring_erode_px=BG_RING_ERODE_PX,\n"
        "        bg_ring_rgb_delta=BG_RING_RGB_DELTA,\n"
        "        reuse_max_d_ar=TEMP_REUSE_MAX_D_AR,\n"
        "        check_bg_motion=CHECK_BG_MOTION,\n"
        "        motion_bg_mad_threshold=MOTION_BG_MAD_THRESHOLD,\n"
        "        motion_context_max_side=MOTION_CONTEXT_MAX_SIDE,\n"
        "        check_frame_motion=CHECK_FRAME_MOTION,\n"
        "        frame_mv_pos=FRAME_MV_POS,\n"
        "        frame_mv_log=FRAME_MV_LOG,\n"
        "        frame_mv_ar=FRAME_MV_AR,\n"
        "    )\n"
    )
    if old_call not in src:
        raise RuntimeError("step6 call anchor not found")
    src = src.replace(old_call, new_call, 1)

    old_end = (
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n\n"
        "total_time = time.time() - start_time\n"
    )
    new_end = (
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n\n"
        "if gdino_cross is not None:\n"
        "    del gdino_cross\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n\n"
        "total_time = time.time() - start_time\n"
    )
    if old_end not in src:
        raise RuntimeError("step6 cleanup anchor not found")
    src = src.replace(old_end, new_end, 1)
    if "bg_cross_sticky_frames=BG_CROSS_STICKY_FRAMES" not in src and "bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD" in src:
        src = src.replace(
            "        bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD,\n",
            "        bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD,\n"
            "        bg_cross_sticky_frames=BG_CROSS_STICKY_FRAMES,\n",
            1,
        )
    return src


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        cell["outputs"] = []
        cell["execution_count"] = None

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell_to_str(cell)
        if "def replace_object_roi_nano_temporal" in src and "def extract_frames" in src:
            cell["source"] = str_to_cell_source(patch_pipeline_cell(src))
            print("patched pipeline cell")
        elif (
            "TEMP_STABLE_POS" in src
            and "GEN_API_KEY" in src
            and "TEMP_FULL_LOG_AREA = 0.75\n\n# API\n" in src
        ):
            cell["source"] = str_to_cell_source(patch_params_cell(src))
            print("patched params cell")
        elif "Шаг 6:" in src and "replace_object_roi_nano_temporal" in src:
            cell["source"] = str_to_cell_source(patch_step6_cell(src))
            print("patched step 6 cell")

    # scrub key if leaked
    t = json.dumps(nb, ensure_ascii=False, indent=1)
    t = t.replace(
        'GEN_API_KEY = os.environ.get("GEN_API_KEY", "sk-CoYciF7amUAe4VUMv0NkRezxssBY71pIkBr4QoZwzqSWFpuyPG00Bifckml7").strip()',
        'GEN_API_KEY = os.environ.get("GEN_API_KEY", "").strip()',
    )
    nb = json.loads(t)

    out = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    DST.write_text(out, encoding="utf-8")
    json.loads(DST.read_text(encoding="utf-8"))
    print("wrote", DST, "bytes", DST.stat().st_size)
    print("OK valid JSON")


if __name__ == "__main__":
    main()
