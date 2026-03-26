from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    service_info: str = Field(
        ...,
        title="Service Info Endpoint",
        description="服務基礎資訊端點，用於查看服務名稱、預設模型與主要 API 路徑。",
        example="/service-info",
    )
    health: str = Field(
        ...,
        title="Health Endpoint",
        description="一般健康檢查端點。",
        example="/health",
    )
    healthz: str = Field(
        ...,
        title="Lightweight Health Endpoint",
        description="輕量健康檢查端點。可供容器 healthcheck 使用。",
        example="/healthz",
    )
    warmup: str = Field(
        ...,
        title="Warmup Endpoint",
        description="warmup API 路徑，用於初始化預設模型 session。",
        example="/warmup",
    )
    remove_bg: str = Field(
        ...,
        title="Background Removal Endpoint",
        description="去背 API 路徑與方法提示。實際呼叫時請使用 POST 並帶入 file 檔案。",
        example="POST /remove-bg",
    )


class ServiceInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service": "rembg-service",
                "model": "isnet-general-use",
                "endpoints": {
                    "service_info": "/service-info",
                    "health": "/health",
                    "healthz": "/healthz",
                    "warmup": "/warmup",
                    "remove_bg": "POST /remove-bg",
                },
            }
        }
    )

    service: str = Field(
        ...,
        title="Service Name",
        description="目前服務名稱。",
        example="rembg-service",
    )
    model: str = Field(
        ...,
        title="Default Model",
        description="目前服務預設使用的 rembg 模型名稱。",
        example="isnet-general-use",
    )
    endpoints: EndpointMap = Field(
        ...,
        title="Endpoint Map",
        description="主要可用 API 路徑清單。",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "service": "rembg-service",
                "model": "isnet-general-use",
                "model_warmed": True,
            }
        }
    )

    ok: Literal[True] = Field(
        ...,
        title="Service Healthy",
        description="固定為 true，表示服務可正常回應。",
        example=True,
    )
    service: str = Field(
        ...,
        title="Service Name",
        description="目前服務名稱。",
        example="rembg-service",
    )
    model: str = Field(
        ...,
        title="Default Model",
        description="目前服務預設使用的 rembg 模型名稱。",
        example="isnet-general-use",
    )
    model_warmed: bool = Field(
        ...,
        title="Model Warmed",
        description="預設模型 session 是否已完成初始化。",
        example=True,
    )


class WarmupResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "service": "rembg-service",
                "model": "isnet-general-use",
                "model_warmed": True,
                "warmed_by": "warmup_endpoint",
            }
        }
    )

    ok: Literal[True] = Field(
        ...,
        title="Warmup Success",
        description="固定為 true，表示 warmup 已完成。",
        example=True,
    )
    service: str = Field(
        ...,
        title="Service Name",
        description="目前服務名稱。",
        example="rembg-service",
    )
    model: str = Field(
        ...,
        title="Default Model",
        description="本次 warmup 使用的模型名稱。",
        example="isnet-general-use",
    )
    model_warmed: bool = Field(
        ...,
        title="Model Warmed",
        description="warmup 完成後，預設模型 session 是否已載入。",
        example=True,
    )
    warmed_by: str = Field(
        ...,
        title="Warmup Trigger Source",
        description="觸發 warmup 的來源識別字串。",
        example="warmup_endpoint",
    )


class RejectMetrics(BaseModel):
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

    foreground_ratio: float = Field(
        ...,
        title="Foreground Ratio",
        description="alpha 大於等於基準值的前景像素比例，用於估計主體覆蓋範圍。",
        example=0.286412,
    )
    alpha_mean: float = Field(
        ...,
        title="Alpha Mean",
        description="整張 alpha mask 的平均值，用於估計整體前景信心。",
        example=62.1843,
    )
    mid_alpha_ratio: float = Field(
        ...,
        title="Mid Alpha Ratio",
        description="中間透明度區段的比例，用於判斷邊界是否過於模糊。",
        example=0.038211,
    )
    high_alpha_ratio: float = Field(
        ...,
        title="High Alpha Ratio",
        description="高透明度區段比例，用於估計穩定前景範圍。",
        example=0.241933,
    )
    bbox_width_ratio: float = Field(
        ...,
        title="Bounding Box Width Ratio",
        description="前景外接框寬度占整張圖寬度的比例。",
        example=0.624512,
    )
    bbox_height_ratio: float = Field(
        ...,
        title="Bounding Box Height Ratio",
        description="前景外接框高度占整張圖高度的比例。",
        example=0.918274,
    )
    edge_band_ratio: float = Field(
        ...,
        title="Edge Band Ratio",
        description="邊界帶區域整體比例。",
        example=0.017223,
    )
    edge_band_mid_ratio: float = Field(
        ...,
        title="Edge Band Mid Ratio",
        description="邊界帶中間透明度區段比例。",
        example=0.009121,
    )
    edge_band_low_ratio: float = Field(
        ...,
        title="Edge Band Low Ratio",
        description="邊界帶低透明度區段比例。",
        example=0.004388,
    )
    edge_quality_low_candidate: bool = Field(
        ...,
        title="Edge Quality Low Candidate",
        description="是否被判定為邊界品質偏低候選。",
        example=True,
    )


class RemoveBgRejectedResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "code": "LOW_CONFIDENCE_MASK",
                "message": "背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
                "reason": "low_confidence_mask",
                "metrics": RejectMetrics.model_config["json_schema_extra"]["example"],
            }
        }
    )

    ok: Literal[False] = Field(
        ...,
        title="Removal Rejected",
        description="固定為 false，表示此次請求未產出去背 PNG，而是回傳拒絕結果。",
        example=False,
    )
    code: str = Field(
        ...,
        title="Reject Code",
        description="服務定義的拒絕代碼。用於前端或測試腳本識別拒絕類型。",
        example="LOW_CONFIDENCE_MASK",
    )
    message: str = Field(
        ...,
        title="Reject Message",
        description="給前端或使用者顯示的簡短拒絕說明。",
        example="背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
    )
    reason: str = Field(
        ...,
        title="Reject Reason",
        description="內部判斷用的拒絕原因 key。",
        example="low_confidence_mask",
    )
    metrics: RejectMetrics = Field(
        ...,
        title="Reject Metrics",
        description="此次拒絕時輸出的 alpha 與邊界品質評估指標。",
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Empty file",
            }
        }
    )

    detail: str = Field(
        ...,
        title="Error Detail",
        description="錯誤詳細訊息。",
        example="Empty file",
    )