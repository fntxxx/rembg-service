import io
import time
from typing import Optional

import numpy as np
from fastapi import HTTPException
from PIL import Image
from rembg import remove

from app.core.config import DEFAULT_MODEL, MAX_OUTPUT_SIDE
from app.core.session import get_session
from app.domain.alpha_pipeline import (
    apply_alpha_curve,
    collect_alpha_stats,
    compute_bbox_ratios,
    # apply_fallback_blend,
    # should_use_fallback,
)
from app.domain.edge_decontaminate import (
    decontaminate_edge_rgb,
    estimate_background_rgb,
)
from app.domain.metrics import compute_edge_quality_metrics
from app.domain.rejection import evaluate_rejection
from app.schemas.responses import build_reject_payload, build_success_headers


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


def process_remove_bg(
    raw: bytes,
    max_side: int,
    quality: str,
    model: Optional[str],
    reject_low_confidence: bool,
    reject_edge_quality: bool,
):
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    timing = {}
    request_started_at = time.perf_counter()

    timing["fallback_used"] = False
    timing["fallback_sec"] = 0.0

    # ---- Resize to control cost ----
    try:
        resize_started_at = time.perf_counter()

        # 真正的原圖（最後保色、輸出用）
        original_full = Image.open(io.BytesIO(raw)).convert("RGBA")

        # 工作圖（縮小後給 rembg / alpha pipeline 用）
        img = original_full.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        in_bytes = buf.getvalue()

        timing["resize_sec"] = round(time.perf_counter() - resize_started_at, 4)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # ---- Choose model ----
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
        return {
            "kind": "json",
            "status_code": 502,
            "content": {
                "error": "rembg failed",
                "detail": str(e),
                "model": model_name,
                "quality": quality,
            },
        }

    try:
        postprocess_started_at = time.perf_counter()

        # 用去背結果的 alpha，但保留原圖 RGB，避免衣物本體顏色漂移
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        alpha = out_img.getchannel("A")

        # ---- Adaptive alpha routing ----
        alpha_stats = collect_alpha_stats(alpha)
        alpha_mean = alpha_stats["alpha_mean"]
        foreground_ratio = alpha_stats["foreground_ratio"]
        mid_alpha_ratio = alpha_stats["mid_alpha_ratio"]
        low_alpha_ratio = alpha_stats["low_alpha_ratio"]
        high_alpha_ratio = alpha_stats["high_alpha_ratio"]
        edge_band_ratio = alpha_stats["edge_band_ratio"]
        edge_band_mid_ratio = alpha_stats["edge_band_mid_ratio"]
        edge_band_low_ratio = alpha_stats["edge_band_low_ratio"]
        used_fallback = False

        # fallback 目前整段停用，固定只走 base model（isnet-general-use）
        # print(">>> BEFORE FALLBACK CHECK <<<", model_name, quality, foreground_ratio, alpha_mean)

        # if should_use_fallback(model_name, quality, foreground_ratio, alpha_mean):
        #     print(">>> USING FALLBACK <<<", foreground_ratio, alpha_mean)
        #     fallback_model = "u2net"
        #     fallback_quality = "fast"

        #     try:
        #         fallback_started_at = time.perf_counter()
        #         fallback_bytes = run_remove(in_bytes, fallback_model, fallback_quality)
        #         fallback_img = Image.open(io.BytesIO(fallback_bytes)).convert("RGBA")
        #         timing["fallback_sec"] = round(time.perf_counter() - fallback_started_at, 4)
        #         timing["fallback_used"] = True
        #     except Exception as e:
        #         return {
        #             "kind": "json",
        #             "status_code": 502,
        #             "content": {
        #                 "error": "fallback rembg failed",
        #                 "detail": str(e),
        #                 "model": fallback_model,
        #                 "quality": fallback_quality,
        #             },
        #         }

        #     fallback_alpha = fallback_img.getchannel("A")
        #     alpha = apply_fallback_blend(alpha, fallback_alpha)

        #     used_fallback = True
        #     actual_model = f"{model_name}+fallback"

        #     alpha_stats = collect_alpha_stats(alpha)
        #     alpha_mean = alpha_stats["alpha_mean"]
        #     foreground_ratio = alpha_stats["foreground_ratio"]
        #     mid_alpha_ratio = alpha_stats["mid_alpha_ratio"]
        #     low_alpha_ratio = alpha_stats["low_alpha_ratio"]
        #     high_alpha_ratio = alpha_stats["high_alpha_ratio"]
        #     edge_band_ratio = alpha_stats["edge_band_ratio"]
        #     edge_band_mid_ratio = alpha_stats["edge_band_mid_ratio"]
        #     edge_band_low_ratio = alpha_stats["edge_band_low_ratio"]

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

            return {
                "kind": "json",
                "status_code": 422,
                "content": build_reject_payload(
                    final_reject_reason=final_reject_reason,
                    foreground_ratio=foreground_ratio,
                    alpha_mean=alpha_mean,
                    debug_mid_alpha_ratio=debug_mid_alpha_ratio,
                    debug_high_alpha_ratio=debug_high_alpha_ratio,
                    bbox_width_ratio=bbox_width_ratio,
                    bbox_height_ratio=bbox_height_ratio,
                    final_edge_metrics=final_edge_metrics,
                    edge_quality_low_candidate=rejection["edge_quality_low_candidate"],
                ),
            }

        # alpha 還在小圖

        # 👉 resize alpha 到原圖
        target_size = original_full.size

        # 👉 限制最大輸出尺寸
        w, h = target_size
        longest = max(w, h)

        if longest > MAX_OUTPUT_SIDE:
            scale = MAX_OUTPUT_SIDE / float(longest)
            target_size = (int(w * scale), int(h * scale))

        # resize 原圖 + alpha 同步
        original_resized = original_full.resize(target_size, Image.LANCZOS)
        alpha = alpha.resize(target_size, Image.LANCZOS)

        # -------------------------------------------------
        # 邊界去污染：
        # - 主體內部 RGB 不動
        # - 只處理 alpha 過渡帶，降低白邊 / 灰邊
        # -------------------------------------------------
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
    except Exception as e:
        return {
            "kind": "json",
            "status_code": 502,
            "content": {
                "error": "postprocess failed",
                "detail": str(e),
                "model": model_name,
                "quality": quality,
            },
        }

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
        }
    )

    final_edge_candidate = bool(
        large_garment_edge_bad or low_height_object_edge_bad
    )

    return {
        "kind": "binary",
        "content": final_bytes,
        "headers": build_success_headers(
            fallback_used=timing["fallback_used"],
            actual_model=actual_model,
            final_edge_candidate=final_edge_candidate,
            final_edge_metrics=final_edge_metrics,
        ),
    }
