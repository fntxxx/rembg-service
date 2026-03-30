import io
import time
from typing import Optional

import numpy as np
from fastapi import HTTPException
from PIL import Image
from rembg import remove

from app.core.config import (
    DEFAULT_MAX_SIDE,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_REJECT_EDGE_QUALITY,
    DEFAULT_REJECT_LOW_CONFIDENCE,
    MAX_OUTPUT_SIDE,
)
from app.core.session import get_session
from app.domain.alpha_pipeline import (
    apply_alpha_curve,
    collect_alpha_stats,
    compute_bbox_ratios,
)
from app.domain.edge_decontaminate import (
    decontaminate_edge_rgb,
    estimate_background_rgb,
)
from app.domain.rejection import evaluate_rejection
from app.schemas.responses import (
    build_remove_bg_success_data,
    raise_gateway_error,
    raise_rejection_error,
)


def run_remove(in_bytes: bytes, model_name: str, quality: str) -> bytes:
    session = get_session(model_name)

    remove_kwargs = {}
    if quality == "high":
        remove_kwargs = {
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 2,
        }

    return remove(in_bytes, session=session, **remove_kwargs)


def process_remove_bg(raw: bytes):
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    max_side = DEFAULT_MAX_SIDE
    quality = DEFAULT_QUALITY
    model: Optional[str] = DEFAULT_MODEL
    reject_low_confidence = DEFAULT_REJECT_LOW_CONFIDENCE
    reject_edge_quality = DEFAULT_REJECT_EDGE_QUALITY

    timing = {}
    request_started_at = time.perf_counter()

    timing["fallback_used"] = False
    timing["fallback_sec"] = 0.0

    try:
        resize_started_at = time.perf_counter()
        original_full = Image.open(io.BytesIO(raw)).convert("RGBA")

        img = original_full.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        in_bytes = buf.getvalue()

        timing["resize_sec"] = round(time.perf_counter() - resize_started_at, 4)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    model_name = model or DEFAULT_MODEL
    actual_model = model_name
    try:
        get_session(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        base_remove_started_at = time.perf_counter()
        out_bytes = run_remove(in_bytes, model_name, quality)
        timing["base_remove_sec"] = round(time.perf_counter() - base_remove_started_at, 4)
    except Exception as e:
        raise_gateway_error(
            code="REMBG_EXECUTION_FAILED",
            message="去背引擎執行失敗。",
            details={
                "cause": str(e),
                "model": model_name,
                "quality": quality,
            },
        )

    try:
        postprocess_started_at = time.perf_counter()

        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        alpha = out_img.getchannel("A")

        alpha_stats = collect_alpha_stats(alpha)
        alpha_mean = alpha_stats["alpha_mean"]
        foreground_ratio = alpha_stats["foreground_ratio"]
        used_fallback = False

        alpha, _curve_debug = apply_alpha_curve(
            alpha=alpha,
            used_fallback=used_fallback,
            foreground_ratio=foreground_ratio,
            alpha_mean=alpha_mean,
        )

        debug_alpha_np = np.array(alpha, dtype=np.float32)
        debug_mid_alpha_ratio = float(((debug_alpha_np >= 10) & (debug_alpha_np <= 200)).mean())
        debug_high_alpha_ratio = float((debug_alpha_np >= 160).mean())
        final_alpha_stats = collect_alpha_stats(alpha)
        final_edge_metrics = {
            "edge_band_ratio": final_alpha_stats["edge_band_ratio"],
            "edge_band_mid_ratio": final_alpha_stats["edge_band_mid_ratio"],
            "edge_band_low_ratio": final_alpha_stats["edge_band_low_ratio"],
        }

        bbox_width_ratio, bbox_height_ratio, debug_alpha_np = compute_bbox_ratios(alpha)

        print(
            ">>> AFTER POSTPROCESS <<<",
            {
                "used_fallback": used_fallback,
                "foreground_ratio": round(float((debug_alpha_np >= 8).mean()), 6),
                "alpha_mean": round(float(debug_alpha_np.mean()), 4),
                "mid_alpha_ratio": round(debug_mid_alpha_ratio, 6),
                "high_alpha_ratio": round(debug_high_alpha_ratio, 6),
                "edge_band_ratio": round(final_edge_metrics["edge_band_ratio"], 6),
                "edge_band_mid_ratio": round(final_edge_metrics["edge_band_mid_ratio"], 6),
                "edge_band_low_ratio": round(final_edge_metrics["edge_band_low_ratio"], 6),
            },
        )

        foreground_ratio = float((debug_alpha_np >= 8).mean())
        alpha_mean = float(debug_alpha_np.mean())

        rejection = evaluate_rejection(
            foreground_ratio=foreground_ratio,
            alpha_mean=alpha_mean,
            debug_mid_alpha_ratio=debug_mid_alpha_ratio,
            debug_high_alpha_ratio=debug_high_alpha_ratio,
            bbox_width_ratio=bbox_width_ratio,
            bbox_height_ratio=bbox_height_ratio,
            final_edge_metrics=final_edge_metrics,
            reject_low_confidence=reject_low_confidence,
            reject_edge_quality=reject_edge_quality,
        )

        final_should_reject = rejection["final_should_reject"]
        final_reject_reason = rejection["final_reject_reason"]
        large_garment_edge_bad = rejection["large_garment_edge_bad"]
        low_height_object_edge_bad = rejection["low_height_object_edge_bad"]

        if final_should_reject:
            timing["postprocess_sec"] = round(time.perf_counter() - postprocess_started_at, 4)
            timing["total_sec"] = round(time.perf_counter() - request_started_at, 4)
            raise_rejection_error(
                final_reject_reason=final_reject_reason,
                foreground_ratio=foreground_ratio,
                alpha_mean=alpha_mean,
                debug_mid_alpha_ratio=debug_mid_alpha_ratio,
                debug_high_alpha_ratio=debug_high_alpha_ratio,
                bbox_width_ratio=bbox_width_ratio,
                bbox_height_ratio=bbox_height_ratio,
                final_edge_metrics=final_edge_metrics,
                edge_quality_low_candidate=rejection["edge_quality_low_candidate"],
            )

        target_size = original_full.size
        w, h = target_size
        longest = max(w, h)

        if longest > MAX_OUTPUT_SIDE:
            scale = MAX_OUTPUT_SIDE / float(longest)
            target_size = (int(w * scale), int(h * scale))

        original_resized = original_full.resize(target_size, Image.LANCZOS)
        alpha = alpha.resize(target_size, Image.LANCZOS)

        original_resized_rgb = original_resized.convert("RGB")
        bg_rgb = estimate_background_rgb(original_resized_rgb)

        decontaminated_rgb = decontaminate_edge_rgb(
            original_rgb=original_resized_rgb,
            alpha=alpha,
            bg_rgb=bg_rgb,
            edge_alpha_min=20,
            edge_alpha_max=200,
            restore_strength=0.85,
        )

        merged = decontaminated_rgb.convert("RGBA")
        merged.putalpha(alpha)

        final_buf = io.BytesIO()
        merged.save(final_buf, format="PNG")
        final_bytes = final_buf.getvalue()

        timing["postprocess_sec"] = round(time.perf_counter() - postprocess_started_at, 4)
        timing["total_sec"] = round(time.perf_counter() - request_started_at, 4)
    except HTTPException:
        raise
    except Exception as e:
        raise_gateway_error(
            code="POSTPROCESS_FAILED",
            message="去背後處理失敗。",
            details={
                "cause": str(e),
                "model": model_name,
                "quality": quality,
            },
        )

    print(
        "[remove-bg timing]",
        {
            "model": model_name,
            "quality": quality,
            "max_side": max_side,
            "resize_sec": timing["resize_sec"],
            "base_remove_sec": timing["base_remove_sec"],
            "fallback_used": timing["fallback_used"],
            "fallback_sec": timing["fallback_sec"],
            "postprocess_sec": timing["postprocess_sec"],
            "total_sec": timing["total_sec"],
            "foreground_ratio": round(foreground_ratio, 6),
            "alpha_mean": round(alpha_mean, 4),
            "edge_band_ratio": round(final_edge_metrics["edge_band_ratio"], 6),
            "edge_band_mid_ratio": round(final_edge_metrics["edge_band_mid_ratio"], 6),
            "edge_band_low_ratio": round(final_edge_metrics["edge_band_low_ratio"], 6),
            "edge_quality_low_candidate": bool(
                large_garment_edge_bad or low_height_object_edge_bad
            ),
        },
    )

    final_edge_candidate = bool(large_garment_edge_bad or low_height_object_edge_bad)

    return build_remove_bg_success_data(
        image_bytes=final_bytes,
        actual_model=actual_model,
        fallback_used=timing["fallback_used"],
        final_edge_candidate=final_edge_candidate,
        final_edge_metrics=final_edge_metrics,
        output_width=merged.width,
        output_height=merged.height,
    )
