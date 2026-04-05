from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class SuccessEnvelope(BaseModel, Generic[T]):
    ok: Literal[True] = Field(..., json_schema_extra={"example": True})
    data: T


class ErrorInfo(BaseModel):
    code: str = Field(..., json_schema_extra={"example": "BAD_REQUEST"})
    message: str = Field(..., json_schema_extra={"example": "Invalid image"})
    details: Optional[Any] = Field(default=None, json_schema_extra={"example": None})


class ErrorEnvelope(BaseModel):
    ok: Literal[False] = Field(..., json_schema_extra={"example": False})
    error: ErrorInfo


class EndpointMap(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service_info": "/service-info",
                "health": "/health",
                "healthz": "/healthz",
                "warmup": "/warmup",
                "remove_bg": "POST /remove-bg",
            }
        }
    )

    service_info: str
    health: str
    healthz: str
    warmup: str
    remove_bg: str


class ProcessingDefaults(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_side": 512,
                "quality": "fast",
                "reject_low_confidence": True,
                "reject_edge_quality": True,
            }
        }
    )

    max_side: int
    quality: str
    reject_low_confidence: bool
    reject_edge_quality: bool


class ServiceInfoData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service": "rembg-service",
                "model": "isnet-general-use",
                "endpoints": EndpointMap.model_config["json_schema_extra"]["example"],
                "processing_defaults": ProcessingDefaults.model_config["json_schema_extra"]["example"],
            }
        }
    )

    service: str
    model: str
    endpoints: EndpointMap
    processing_defaults: ProcessingDefaults


class HealthData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service": "rembg-service",
                "model": "isnet-general-use",
                "model_warmed": True,
            }
        }
    )

    service: str
    model: str
    model_warmed: bool


class WarmupData(HealthData):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service": "rembg-service",
                "model": "isnet-general-use",
                "model_warmed": True,
                "warmed_by": "warmup_endpoint",
            }
        }
    )

    warmed_by: str


class RemoveBgMetrics(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "foreground_ratio": 0.286412,
                "alpha_mean": 62.1843,
                "mid_alpha_ratio": 0.038211,
                "high_alpha_ratio": 0.241933,
                "bbox_width_ratio": 0.624512,
                "bbox_height_ratio": 0.918274,
                "edge_band_ratio": 0.017223,
                "edge_band_mid_ratio": 0.009121,
                "edge_band_low_ratio": 0.004388,
                "edge_quality_low_candidate": True,
            }
        }
    )

    foreground_ratio: float
    alpha_mean: float
    mid_alpha_ratio: float
    high_alpha_ratio: float
    bbox_width_ratio: float
    bbox_height_ratio: float
    edge_band_ratio: float
    edge_band_mid_ratio: float
    edge_band_low_ratio: float
    edge_quality_low_candidate: bool


class RejectDetails(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reason": "complex_background_low_confidence",
                "metrics": RemoveBgMetrics.model_config["json_schema_extra"]["example"],
            }
        }
    )

    reason: str
    metrics: RemoveBgMetrics


class ValidationErrorItem(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorDetails(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "errors": [
                    {
                        "loc": ["body", "file"],
                        "msg": "Field required",
                        "type": "missing",
                    }
                ]
            }
        }
    )

    errors: list[ValidationErrorItem]


ServiceInfoResponse = SuccessEnvelope[ServiceInfoData]
HealthResponse = SuccessEnvelope[HealthData]
WarmupResponse = SuccessEnvelope[WarmupData]


class RejectErrorInfo(BaseModel):
    code: Literal["LOW_CONFIDENCE_MASK"] = Field(..., json_schema_extra={"example": "LOW_CONFIDENCE_MASK"})
    message: str = Field(
        ...,
        json_schema_extra={"example": "背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。"},
    )
    details: RejectDetails


class RemoveBgRejectedResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "error": {
                    "code": "LOW_CONFIDENCE_MASK",
                    "message": "背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
                    "details": RejectDetails.model_config["json_schema_extra"]["example"],
                },
            }
        }
    )

    ok: Literal[False] = False
    error: RejectErrorInfo


class RequestValidationErrorInfo(BaseModel):
    code: Literal["REQUEST_VALIDATION_ERROR"] = Field(
        ...,
        json_schema_extra={"example": "REQUEST_VALIDATION_ERROR"},
    )
    message: str = Field(..., json_schema_extra={"example": "請求參數驗證失敗。"})
    details: ValidationErrorDetails


class RequestValidationErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "請求參數驗證失敗。",
                    "details": ValidationErrorDetails.model_config["json_schema_extra"]["example"],
                },
            }
        }
    )

    ok: Literal[False] = False
    error: RequestValidationErrorInfo


GenericErrorResponse = ErrorEnvelope
