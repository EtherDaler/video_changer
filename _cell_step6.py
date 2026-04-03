# ── Шаг 6: Обработка кадров (Nano Banana 2 + склейка ROI) ─────────────
import time

if not GEN_API_KEY:
    raise RuntimeError(
        "GEN_API_KEY пустой. Задай переменную окружения GEN_API_KEY или вставь ключ в ячейке параметров."
    )

print("=" * 60)
print("Шаг 6: замена по кадрам (кэш / масштаб / edit с ref / full)")
print("=" * 60)
print(f"  OLD → NEW в промпте: '{OLD_OBJECT_PROMPT[:40]}...' → '{NEW_OBJECT_PROMPT[:40]}...'")
print(f"  Всего кадров: {len(frames)}  |  С объектом: {visible_count}\n")

processed = []
preview_frames = []
start_time = time.time()
temporal_state = NanoTemporalState()
prev_mask_any = False

gdino_cross = None
if (BG_CROSS_PROMPT or "").strip():
    print("  Загружаю Grounding DINO для проверки пересечений (BG_CROSS_PROMPT)...")
    gdino_cross = load_grounding_dino(GDINO_CHECKPOINT)


for i, frame in enumerate(frames):
    mask = tracked_masks[i]

    if not mask.any():
        processed.append(frame.copy())
        prev_mask_any = False
        temporal_state.reset()
        print(f"  Кадр {i+1:3d}/{len(frames)} — объект не виден, пропускаю")
        continue

    if not prev_mask_any:
        temporal_state.reset()
    prev_mask_any = True

    frame_seed = int(SEED) + i
    inpainted = replace_object_roi_nano_temporal(
        frame,
        mask,
        temporal_state,
        old_object_prompt=OLD_OBJECT_PROMPT,
        new_object_prompt=NEW_OBJECT_PROMPT,
        api_key=GEN_API_KEY,
        seed=frame_seed,
        roi_padding=ROI_PADDING,
        context_padding=CONTEXT_PADDING,
        max_roi_side=MAX_ROI_SIDE,
        mask_dilate=MASK_DILATE,
        mask_feather=MASK_FEATHER,
        resolution=NANO_RESOLUTION,
        use_match_domain=USE_MATCH_DOMAIN,
        use_match_domain_on_reuse=USE_MATCH_DOMAIN_ON_REUSE,
        poll_interval=NANO_POLL_INTERVAL,
        timeout_sec=NANO_TIMEOUT_SEC,
        stable_pos=TEMP_STABLE_POS,
        stable_ar=TEMP_STABLE_AR,
        reuse_max_log_area=TEMP_REUSE_MAX_LOG_AREA,
        reuse_max_d_ar=TEMP_REUSE_MAX_D_AR,
        edit_max_pos=TEMP_EDIT_MAX_POS,
        edit_max_log_area=TEMP_EDIT_MAX_LOG_AREA,
        edit_max_ar=TEMP_EDIT_MAX_AR,
        full_regen_pos=TEMP_FULL_POS,
        full_regen_log_area=TEMP_FULL_LOG_AREA,
        mask_cover_extra_dilate=MASK_COVER_EXTRA_DILATE,
        small_mask_area_ratio=SMALL_MASK_AREA_RATIO,
        small_mask_area_boost_dilate=SMALL_MASK_AREA_BOOST_DILATE,
        blend_mask_dilate_px=BLEND_MASK_DILATE_PX,
        composite_feather_ks=COMPOSITE_FEATHER_KS,
        use_poisson_blend=USE_POISSON_BLEND,
        gdino_model=gdino_cross,
        bg_cross_prompt=BG_CROSS_PROMPT,
        bg_cross_box_threshold=BG_CROSS_BOX_THRESHOLD,
        bg_cross_text_threshold=BG_CROSS_TEXT_THRESHOLD,
        bg_cross_sticky_frames=BG_CROSS_STICKY_FRAMES,
        check_object_lighting=CHECK_OBJECT_LIGHTING,
        object_light_rel_delta=OBJECT_LIGHT_REL_DELTA,
        check_bg_ring=CHECK_BG_RING,
        bg_ring_dilate_px=BG_RING_DILATE_PX,
        bg_ring_erode_px=BG_RING_ERODE_PX,
        bg_ring_rgb_delta=BG_RING_RGB_DELTA,
        #reuse_max_d_ar=TEMP_REUSE_MAX_D_AR,
        check_bg_motion=CHECK_BG_MOTION,
        motion_bg_mad_threshold=MOTION_BG_MAD_THRESHOLD,
        motion_context_max_side=MOTION_CONTEXT_MAX_SIDE,
        check_frame_motion=CHECK_FRAME_MOTION,
        frame_mv_pos=FRAME_MV_POS,
        frame_mv_log=FRAME_MV_LOG,
        frame_mv_ar=FRAME_MV_AR,
    )
    processed.append(inpainted)

    if i < 5 or i % 10 == 0:
        preview_frames.append((i, frame.copy(), inpainted.copy()))

    elapsed = time.time() - start_time
    frames_done = i + 1
    fps_proc = frames_done / elapsed
    eta = (len(frames) - frames_done) / max(fps_proc, 0.001)
    print(f"  Кадр {frames_done:3d}/{len(frames)} ✓  [{elapsed:.0f}s прошло, ~{eta:.0f}s осталось]")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if gdino_cross is not None:
    del gdino_cross
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

total_time = time.time() - start_time
print(f"\n✅ Обработка завершена за {total_time/60:.1f} минут!")
