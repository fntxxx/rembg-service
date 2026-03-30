import base64
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from app.core.config import (
    DEFAULT_MAX_SIDE,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_REJECT_EDGE_QUALITY,
    DEFAULT_REJECT_LOW_CONFIDENCE,
)
from app.core.exceptions import ApiError


_STATUS_CODE_TO_ERROR_CODE = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "UNPROCESSABLE_ENTITY",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_SERVER_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


def build_error_envelope(
    *,
    code: str,
    message: str,
    details: Any = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
    }


def build_success_envelope(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "data": data,
    }


def build_reject_error_details(
    final_reject_reason: str,
    foreground_ratio: float,
    alpha_mean: float,
    debug_mid_alpha_ratio: float,
    debug_high_alpha_ratio: float,
    bbox_width_ratio: float,
    bbox_height_ratio: float,
    final_edge_metrics: Dict[str, float],
    edge_quality_low_candidate: bool,
) -> Dict[str, object]:
    return {
        "reason": final_reject_reason,
        "metrics": {
            "foreground_ratio": round(float(foreground_ratio), 6),
            "alpha_mean": round(float(alpha_mean), 4),
            "mid_alpha_ratio": round(float(debug_mid_alpha_ratio), 6),
            "high_alpha_ratio": round(float(debug_high_alpha_ratio), 6),
            "bbox_width_ratio": round(float(bbox_width_ratio), 6),
            "bbox_height_ratio": round(float(bbox_height_ratio), 6),
            "edge_band_ratio": round(float(final_edge_metrics["edge_band_ratio"]), 6),
            "edge_band_mid_ratio": round(float(final_edge_metrics["edge_band_mid_ratio"]), 6),
            "edge_band_low_ratio": round(float(final_edge_metrics["edge_band_low_ratio"]), 6),
            "edge_quality_low_candidate": bool(edge_quality_low_candidate),
        },
    }


def build_remove_bg_success_data(
    *,
    image_bytes: bytes,
    actual_model: str,
    fallback_used: bool,
    final_edge_candidate: bool,
    final_edge_metrics: Dict[str, float],
    output_width: int,
    output_height: int,
) -> Dict[str, Any]:
    return {
        "image": {
            "filename": "removed_bg.png",
            "mime_type": "image/png",
            "base64": base64.b64encode(image_bytes).decode("utf-8"),
            "width": output_width,
            "height": output_height,
        },
        "model": actual_model,
        "fallback_used": fallback_used,
        "edge_quality_low_candidate": final_edge_candidate,
        "metrics": {
            "edge_band_ratio": round(float(final_edge_metrics["edge_band_ratio"]), 6),
            "edge_band_mid_ratio": round(float(final_edge_metrics["edge_band_mid_ratio"]), 6),
            "edge_band_low_ratio": round(float(final_edge_metrics["edge_band_low_ratio"]), 6),
        },
        "processing": {
            "max_side": DEFAULT_MAX_SIDE,
            "quality": DEFAULT_QUALITY,
            "reject_low_confidence": DEFAULT_REJECT_LOW_CONFIDENCE,
            "reject_edge_quality": DEFAULT_REJECT_EDGE_QUALITY,
        },
    }


def build_http_exception_error(exc: HTTPException) -> Dict[str, Any]:
    details = exc.detail if isinstance(exc.detail, (dict, list)) else None
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    code = _STATUS_CODE_TO_ERROR_CODE.get(exc.status_code, "HTTP_ERROR")
    return build_error_envelope(code=code, message=message, details=details)


def build_validation_error(exc: RequestValidationError) -> Dict[str, Any]:
    sanitized_errors = []
    for err in exc.errors():
        sanitized_errors.append(
            {
                "loc": list(err.get("loc", [])),
                "msg": err.get("msg"),
                "type": err.get("type"),
            }
        )

    return build_error_envelope(
        code="REQUEST_VALIDATION_ERROR",
        message="請求參數驗證失敗。",
        details={"errors": sanitized_errors},
    )


def build_unexpected_error() -> Dict[str, Any]:
    return build_error_envelope(
        code="INTERNAL_SERVER_ERROR",
        message="服務發生未預期錯誤。",
        details=None,
    )


def build_api_error_response(exc: ApiError) -> Dict[str, Any]:
    return build_error_envelope(
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


def build_service_info_data() -> Dict[str, Any]:
    return {
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "endpoints": {
            "service_info": "/service-info",
            "health": "/health",
            "healthz": "/healthz",
            "warmup": "/warmup",
            "remove_bg": "POST /remove-bg",
        },
        "processing_defaults": {
            "max_side": DEFAULT_MAX_SIDE,
            "quality": DEFAULT_QUALITY,
            "reject_low_confidence": DEFAULT_REJECT_LOW_CONFIDENCE,
            "reject_edge_quality": DEFAULT_REJECT_EDGE_QUALITY,
        },
    }


def build_health_data(model_warmed: bool) -> Dict[str, Any]:
    return {
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": model_warmed,
    }


def build_warmup_data(model_warmed: bool) -> Dict[str, Any]:
    return {
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": model_warmed,
        "warmed_by": "warmup_endpoint",
    }


def raise_rejection_error(
    *,
    final_reject_reason: str,
    foreground_ratio: float,
    alpha_mean: float,
    debug_mid_alpha_ratio: float,
    debug_high_alpha_ratio: float,
    bbox_width_ratio: float,
    bbox_height_ratio: float,
    final_edge_metrics: Dict[str, float],
    edge_quality_low_candidate: bool,
) -> None:
    raise ApiError(
        status_code=422,
        code="LOW_CONFIDENCE_MASK",
        message="背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
        details=build_reject_error_details(
            final_reject_reason=final_reject_reason,
            foreground_ratio=foreground_ratio,
            alpha_mean=alpha_mean,
            debug_mid_alpha_ratio=debug_mid_alpha_ratio,
            debug_high_alpha_ratio=debug_high_alpha_ratio,
            bbox_width_ratio=bbox_width_ratio,
            bbox_height_ratio=bbox_height_ratio,
            final_edge_metrics=final_edge_metrics,
            edge_quality_low_candidate=edge_quality_low_candidate,
        ),
    )


def raise_gateway_error(*, code: str, message: str, details: Optional[Dict[str, Any]]) -> None:
    raise ApiError(
        status_code=502,
        code=code,
        message=message,
        details=details,
    )
