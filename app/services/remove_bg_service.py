import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
from rembg import remove

from app.core.config import (
    DEFAULT_MAX_SIDE,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_REJECT_EDGE_QUALITY,
    DEFAULT_REJECT_LOW_CONFIDENCE,
    FINAL_OUTPUT_LONGEST_SIDE,
)
from app.core.error_codes import ErrorCode
from app.core.exceptions import ApiError
from app.core.session import get_session
from app.domain.alpha_pipeline import (
    ALPHA_FOREGROUND_THRESHOLD,
    ALPHA_HIGH_THRESHOLD,
    ALPHA_MID_MAX,
    ALPHA_MID_MIN,
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
    build_remove_bg_success_headers,
    raise_gateway_error,
    raise_rejection_error,
)


@dataclass(frozen=True)
class RemoveBgConfig:
    max_side: int = DEFAULT_MAX_SIDE
    quality: str = DEFAULT_QUALITY
    model_name: str = DEFAULT_MODEL
    reject_low_confidence: bool = DEFAULT_REJECT_LOW_CONFIDENCE
    reject_edge_quality: bool = DEFAULT_REJECT_EDGE_QUALITY


@dataclass
class RemoveBgTiming:
    resize_sec: float = 0.0
    base_remove_sec: float = 0.0
    fallback_used: bool = False
    fallback_sec: float = 0.0
    postprocess_sec: float = 0.0
    total_sec: float = 0.0


@dataclass(frozen=True)
class PostprocessResult:
    image_bytes: bytes
    foreground_ratio: float
    alpha_mean: float
    final_edge_metrics: dict[str, float]
    final_edge_candidate: bool


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


def _load_and_prepare_input(raw: bytes, max_side: int) -> tuple[Image.Image, bytes, float]:
    if not raw:
        raise ApiError(
            status_code=400,
            code=ErrorCode.EMPTY_FILE,
            message="Empty file",
            details=None,
        )

    try:
        resize_started_at = time.perf_counter()
        original_full = Image.open(io.BytesIO(raw)).convert("RGBA")

        img = original_full.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        resize_sec = round(time.perf_counter() - resize_started_at, 4)
        return original_full, buf.getvalue(), resize_sec
    except Exception as exc:
        raise ApiError(
            status_code=422,
            code=ErrorCode.INVALID_IMAGE,
            message="Invalid image",
            details=None,
        ) from exc


def _debug_alpha_metrics(alpha_np: np.ndarray) -> tuple[float, float]:
    mid_alpha_ratio = float(((alpha_np >= ALPHA_MID_MIN) & (alpha_np <= ALPHA_MID_MAX)).mean())
    high_alpha_ratio = float((alpha_np >= ALPHA_HIGH_THRESHOLD).mean())
    return mid_alpha_ratio, high_alpha_ratio


def _resize_original_to_alpha_size(original_full: Image.Image, alpha: Image.Image) -> tuple[Image.Image, Image.Image]:
    target_size = alpha.size
    if target_size[0] <= 0 or target_size[1] <= 0:
        return original_full.copy(), alpha

    if original_full.size == target_size:
        return original_full.copy(), alpha

    return original_full.resize(target_size, Image.LANCZOS), alpha


def _compute_alpha_crop_box(alpha: Image.Image) -> tuple[int, int, int, int] | None:
    alpha_np = np.array(alpha, dtype=np.uint8)
    foreground_mask = alpha_np > 0
    ys, xs = np.where(foreground_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    left = int(xs.min())
    upper = int(ys.min())
    right = int(xs.max()) + 1
    lower = int(ys.max()) + 1

    if right <= left or lower <= upper:
        return None

    return left, upper, right, lower


def _crop_to_alpha_bbox(image: Image.Image, alpha: Image.Image) -> Image.Image:
    crop_box = _compute_alpha_crop_box(alpha)
    if crop_box is None:
        return image

    return image.crop(crop_box)


def _resize_image_to_longest_side(image: Image.Image, target_longest_side: int) -> Image.Image:
    if target_longest_side <= 0:
        return image

    width, height = image.size
    longest = max(width, height)

    if width <= 0 or height <= 0 or longest <= 0:
        return image

    if longest == target_longest_side:
        return image

    scale = target_longest_side / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    return image.resize((resized_width, resized_height), Image.LANCZOS)


def _build_final_png(original_full: Image.Image, alpha: Image.Image) -> bytes:
    original_resized, resized_alpha = _resize_original_to_alpha_size(original_full, alpha)

    original_resized_rgb = original_resized.convert("RGB")
    bg_rgb = estimate_background_rgb(original_resized_rgb)

    decontaminated_rgb = decontaminate_edge_rgb(
        original_rgb=original_resized_rgb,
        alpha=resized_alpha,
        bg_rgb=bg_rgb,
        edge_alpha_min=20,
        edge_alpha_max=200,
        restore_strength=0.85,
    )

    merged = decontaminated_rgb.convert("RGBA")
    merged.putalpha(resized_alpha)

    merged = _crop_to_alpha_bbox(merged, resized_alpha)
    merged = _resize_image_to_longest_side(merged, FINAL_OUTPUT_LONGEST_SIDE)

    final_buf = io.BytesIO()
    merged.save(final_buf, format="PNG")
    return final_buf.getvalue()


def _postprocess_remove_output(
    *,
    out_bytes: bytes,
    original_full: Image.Image,
    timing: RemoveBgTiming,
    request_started_at: float,
    config: RemoveBgConfig,
) -> PostprocessResult:
    postprocess_started_at = time.perf_counter()

    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    alpha = out_img.getchannel("A")

    alpha_stats = collect_alpha_stats(alpha)
    alpha_mean = alpha_stats["alpha_mean"]
    foreground_ratio = alpha_stats["foreground_ratio"]

    alpha, _curve_debug = apply_alpha_curve(
        alpha=alpha,
        used_fallback=False,
        foreground_ratio=foreground_ratio,
        alpha_mean=alpha_mean,
    )

    debug_alpha_np = np.array(alpha, dtype=np.float32)
    debug_mid_alpha_ratio, debug_high_alpha_ratio = _debug_alpha_metrics(debug_alpha_np)

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
            "used_fallback": False,
            "foreground_ratio": round(float((debug_alpha_np >= ALPHA_FOREGROUND_THRESHOLD).mean()), 6),
            "alpha_mean": round(float(debug_alpha_np.mean()), 4),
            "mid_alpha_ratio": round(debug_mid_alpha_ratio, 6),
            "high_alpha_ratio": round(debug_high_alpha_ratio, 6),
            "edge_band_ratio": round(final_edge_metrics["edge_band_ratio"], 6),
            "edge_band_mid_ratio": round(final_edge_metrics["edge_band_mid_ratio"], 6),
            "edge_band_low_ratio": round(final_edge_metrics["edge_band_low_ratio"], 6),
        },
    )

    foreground_ratio = float((debug_alpha_np >= ALPHA_FOREGROUND_THRESHOLD).mean())
    alpha_mean = float(debug_alpha_np.mean())

    rejection = evaluate_rejection(
        foreground_ratio=foreground_ratio,
        alpha_mean=alpha_mean,
        debug_mid_alpha_ratio=debug_mid_alpha_ratio,
        debug_high_alpha_ratio=debug_high_alpha_ratio,
        bbox_width_ratio=bbox_width_ratio,
        bbox_height_ratio=bbox_height_ratio,
        final_edge_metrics=final_edge_metrics,
        reject_low_confidence=config.reject_low_confidence,
        reject_edge_quality=config.reject_edge_quality,
    )

    if rejection["final_should_reject"]:
        timing.postprocess_sec = round(time.perf_counter() - postprocess_started_at, 4)
        timing.total_sec = round(time.perf_counter() - request_started_at, 4)
        raise_rejection_error(
            final_reject_reason=rejection["final_reject_reason"],
            foreground_ratio=foreground_ratio,
            alpha_mean=alpha_mean,
            debug_mid_alpha_ratio=debug_mid_alpha_ratio,
            debug_high_alpha_ratio=debug_high_alpha_ratio,
            bbox_width_ratio=bbox_width_ratio,
            bbox_height_ratio=bbox_height_ratio,
            final_edge_metrics=final_edge_metrics,
            edge_quality_low_candidate=rejection["edge_quality_low_candidate"],
        )

    final_bytes = _build_final_png(original_full, alpha)

    timing.postprocess_sec = round(time.perf_counter() - postprocess_started_at, 4)
    timing.total_sec = round(time.perf_counter() - request_started_at, 4)

    return PostprocessResult(
        image_bytes=final_bytes,
        foreground_ratio=foreground_ratio,
        alpha_mean=alpha_mean,
        final_edge_metrics=final_edge_metrics,
        final_edge_candidate=bool(rejection["edge_quality_low_candidate"]),
    )


def process_remove_bg(raw: bytes):
    config = RemoveBgConfig()
    timing = RemoveBgTiming()
    request_started_at = time.perf_counter()

    original_full, in_bytes, timing.resize_sec = _load_and_prepare_input(raw, config.max_side)

    try:
        get_session(config.model_name)
    except ValueError as exc:
        raise ApiError(
            status_code=500,
            code=ErrorCode.SERVER_MISCONFIGURATION,
            message="服務發生未預期錯誤。",
            details={"reason": "unsupported_model", "model": config.model_name},
        ) from exc

    try:
        base_remove_started_at = time.perf_counter()
        out_bytes = run_remove(in_bytes, config.model_name, config.quality)
        timing.base_remove_sec = round(time.perf_counter() - base_remove_started_at, 4)
    except Exception as exc:
        raise_gateway_error(
            code=ErrorCode.REMBG_EXECUTION_FAILED,
            message="去背引擎執行失敗。",
            details={
                "cause": str(exc),
                "model": config.model_name,
                "quality": config.quality,
            },
        )

    try:
        postprocess_result = _postprocess_remove_output(
            out_bytes=out_bytes,
            original_full=original_full,
            timing=timing,
            request_started_at=request_started_at,
            config=config,
        )
    except ApiError:
        raise
    except Exception as exc:
        raise_gateway_error(
            code=ErrorCode.POSTPROCESS_FAILED,
            message="去背後處理失敗。",
            details={
                "cause": str(exc),
                "model": config.model_name,
                "quality": config.quality,
            },
        )

    print(
        "[remove-bg timing]",
        {
            "model": config.model_name,
            "quality": config.quality,
            "max_side": config.max_side,
            "resize_sec": timing.resize_sec,
            "base_remove_sec": timing.base_remove_sec,
            "fallback_used": timing.fallback_used,
            "fallback_sec": timing.fallback_sec,
            "postprocess_sec": timing.postprocess_sec,
            "total_sec": timing.total_sec,
            "foreground_ratio": round(postprocess_result.foreground_ratio, 6),
            "alpha_mean": round(postprocess_result.alpha_mean, 4),
            "edge_band_ratio": round(postprocess_result.final_edge_metrics["edge_band_ratio"], 6),
            "edge_band_mid_ratio": round(
                postprocess_result.final_edge_metrics["edge_band_mid_ratio"], 6
            ),
            "edge_band_low_ratio": round(
                postprocess_result.final_edge_metrics["edge_band_low_ratio"], 6
            ),
            "edge_quality_low_candidate": postprocess_result.final_edge_candidate,
        },
    )

    return {
        "image_bytes": postprocess_result.image_bytes,
        "headers": build_remove_bg_success_headers(
            actual_model=config.model_name,
            fallback_used=timing.fallback_used,
            final_edge_candidate=postprocess_result.final_edge_candidate,
            final_edge_metrics=postprocess_result.final_edge_metrics,
        ),
    }