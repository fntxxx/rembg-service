from typing import Optional

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from app.core.config import DEFAULT_MODEL
from app.core.session import is_model_warmed, warmup_models
from app.schemas.api_docs import (
    ErrorResponse,
    HealthResponse,
    RemoveBgRejectedResponse,
    ServiceInfoResponse,
    WarmupResponse,
)
from app.services.remove_bg_service import process_remove_bg

router = APIRouter()


@router.get(
    "/service-info",
    response_model=ServiceInfoResponse,
    summary="取得服務基礎資訊",
    description="回傳服務名稱、預設模型，以及目前主要可用 API 路徑。",
    response_description="服務基礎資訊與端點清單。",
    tags=["system"],
    operation_id="getServiceInfo",
)
def service_info():
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
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="取得服務健康狀態",
    description="回傳服務基本狀態、預設模型，以及預設模型是否已完成 warmup。",
    response_description="服務健康檢查結果。",
    tags=["system"],
    operation_id="getHealth",
)
def health():
    return {
        "ok": True,
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": is_model_warmed(DEFAULT_MODEL),
    }


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="取得輕量健康檢查狀態",
    description="提供給容器 healthcheck 使用的輕量端點，不會主動觸發新的推論請求。",
    response_description="輕量健康檢查結果。",
    tags=["system"],
    operation_id="getHealthz",
)
def healthz():
    return {
        "ok": True,
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": is_model_warmed(DEFAULT_MODEL),
    }


@router.get(
    "/warmup",
    response_model=WarmupResponse,
    summary="執行模型 warmup",
    description="初始化預設 rembg 模型 session，讓服務在正式接收去背請求前先完成預熱。",
    response_description="warmup 執行結果。",
    tags=["system"],
    operation_id="runWarmup",
)
def warmup():
    warmup_models()
    return {
        "ok": True,
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": is_model_warmed(DEFAULT_MODEL),
        "warmed_by": "warmup_endpoint",
    }


@router.post(
    "/remove-bg",
    responses={
        200: {
            "description": "去背成功，回傳透明背景 PNG 檔案。",
            "content": {
                "image/png": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    }
                }
            },
            "headers": {
                "X-RemoveBg-Fallback-Used": {
                    "description": "是否使用 fallback 流程。此版本預期固定為 false。",
                    "schema": {"type": "string", "example": "false"},
                },
                "X-RemoveBg-Model": {
                    "description": "實際使用的模型名稱。",
                    "schema": {"type": "string", "example": "isnet-general-use"},
                },
                "X-Edge-Quality-Candidate": {
                    "description": "是否被標記為邊界品質偏低候選。",
                    "schema": {"type": "string", "example": "false"},
                },
                "X-Edge-Band-Ratio": {
                    "description": "邊界帶整體比例。",
                    "schema": {"type": "string", "example": "0.017223"},
                },
                "X-Edge-Band-Mid-Ratio": {
                    "description": "邊界帶中間透明度比例。",
                    "schema": {"type": "string", "example": "0.009121"},
                },
                "X-Edge-Band-Low-Ratio": {
                    "description": "邊界帶低透明度比例。",
                    "schema": {"type": "string", "example": "0.004388"},
                },
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "輸入檔案為空、不是有效圖片，或查詢參數不合法。",
            "content": {
                "application/json": {
                    "examples": {
                        "empty_file": {
                            "summary": "Empty file",
                            "value": {"detail": "Empty file"},
                        },
                        "invalid_image": {
                            "summary": "Invalid image",
                            "value": {"detail": "Invalid image"},
                        },
                        "unsupported_model": {
                            "summary": "Unsupported model",
                            "value": {"detail": "Unsupported model: unknown-model"},
                        },
                    }
                }
            },
        },
        422: {
            "model": RemoveBgRejectedResponse,
            "description": "圖片已成功進入處理流程，但因遮罩低信心或邊界品質不足而被拒絕。",
            "content": {
                "application/json": {
                    "examples": {
                        "low_confidence_mask": {
                            "summary": "Rejected due to low confidence mask",
                            "value": RemoveBgRejectedResponse.model_config["json_schema_extra"]["example"],
                        },
                        "missing_file_field": {
                            "summary": "Missing multipart file field",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "file"],
                                        "msg": "Field required",
                                        "type": "missing",
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
        502: {
            "description": "底層 rembg 或後處理階段失敗。",
            "content": {
                "application/json": {
                    "examples": {
                        "rembg_failed": {
                            "summary": "rembg execution failed",
                            "value": {
                                "error": "rembg failed",
                                "detail": "session initialization failed",
                                "model": "isnet-general-use",
                                "quality": "fast",
                            },
                        },
                        "postprocess_failed": {
                            "summary": "postprocess failed",
                            "value": {
                                "error": "postprocess failed",
                                "detail": "cannot merge alpha",
                                "model": "isnet-general-use",
                                "quality": "fast",
                            },
                        },
                    }
                }
            },
        },
    },
    summary="上傳圖片進行去背",
    description=(
        "上傳單張圖片進行去背處理。"
        "請使用 `multipart/form-data`，並以 `file` 作為欄位名稱。"
        "\n\n"
        "成功時回傳 `image/png`。"
        "若遮罩信心不足或邊界品質不佳，則回傳 `422 application/json` 的拒絕結果。"
    ),
    response_description="去背成功的 PNG 檔案，或低信心拒絕結果。",
    tags=["background-removal"],
    operation_id="removeBackground",
)
async def remove_bg(
    file: UploadFile = File(
        ...,
        description=(
            "要進行去背的單張圖片檔案。"
            "請使用 multipart/form-data 上傳，欄位名稱必須是 `file`。"
            "支援的實際格式依 Pillow 可讀取格式為準。"
        ),
    ),
    max_side: int = Query(
        512,
        ge=256,
        le=2048,
        description="送入去背流程前的工作圖最長邊尺寸。數值越大，通常品質較高但耗時也較長。",
        example=512,
    ),
    quality: str = Query(
        "fast",
        pattern="^(fast|high)$",
        description="去背品質模式。`fast` 為較快設定，`high` 會啟用 alpha matting 相關參數。",
        example="fast",
    ),
    model: Optional[str] = Query(
        None,
        description=(
            "可選的 rembg 模型名稱。未提供時使用服務預設模型。"
            "目前允許值以服務設定為準，例如 `isnet-general-use`、`u2net`、`u2netp`。"
        ),
        example="isnet-general-use",
    ),
    reject_low_confidence: bool = Query(
        True,
        description="是否啟用低信心遮罩拒絕判斷。啟用後，低品質去背結果可能直接改為回傳 422 JSON。",
        example=True,
    ),
    reject_edge_quality: bool = Query(
        True,
        description="是否啟用邊界品質拒絕判斷。啟用後，邊界過差的結果可能直接改為回傳 422 JSON。",
        example=True,
    ),
):
    raw = await file.read()

    result = process_remove_bg(
        raw=raw,
        max_side=max_side,
        quality=quality,
        model=model,
        reject_low_confidence=reject_low_confidence,
        reject_edge_quality=reject_edge_quality,
    )

    if result["kind"] == "json":
        return JSONResponse(
            status_code=result["status_code"],
            content=result["content"],
        )

    return Response(
        content=result["content"],
        media_type="image/png",
        headers=result["headers"],
    )