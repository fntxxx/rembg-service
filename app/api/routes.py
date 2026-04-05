from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response

from app.core.auth import require_internal_api_token
from app.core.config import DEFAULT_MODEL
from app.core.file_validation import validate_upload_file_for_remove_bg
from app.core.session import is_model_warmed, warmup_models
from app.schemas.api_docs import (
    GenericErrorResponse,
    HealthResponse,
    RemoveBgRejectedResponse,
    RequestValidationErrorResponse,
    ServiceInfoResponse,
    WarmupResponse,
)
from app.schemas.responses import (
    build_health_data,
    build_service_info_data,
    build_success_envelope,
    build_warmup_data,
)
from app.services.remove_bg_service import process_remove_bg

router = APIRouter()

COMMON_ERROR_RESPONSES = {
    400: {
        "model": GenericErrorResponse,
        "description": "一般請求錯誤，例如空檔案、無效圖片或不支援的輸入。",
        "content": {
            "application/json": {
                "example": {
                    "ok": False,
                    "error": {
                        "code": "BAD_REQUEST",
                        "message": "Invalid image",
                        "details": None,
                    },
                }
            }
        },
    },
    401: {
        "model": GenericErrorResponse,
        "description": "缺少、格式錯誤或無效的 Bearer Token。",
        "content": {
            "application/json": {
                "example": {
                    "ok": False,
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "缺少或無效的 API Token。",
                        "details": {"reason": "invalid_api_token"},
                    },
                }
            }
        },
    },
    422: {
        "description": "422 可能是請求驗證錯誤，或圖片進入處理流程後因遮罩低信心 / 邊界品質不足而被拒絕。",
        "content": {
            "application/json": {
                "examples": {
                    "request_validation_error": {
                        "summary": "Missing multipart file field",
                        "value": RequestValidationErrorResponse.model_config["json_schema_extra"]["example"],
                    },
                    "low_confidence_mask": {
                        "summary": "Rejected due to low confidence mask",
                        "value": RemoveBgRejectedResponse.model_config["json_schema_extra"]["example"],
                    },
                }
            }
        },
    },
    500: {
        "model": GenericErrorResponse,
        "description": "服務發生未預期錯誤。",
        "content": {
            "application/json": {
                "example": {
                    "ok": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "服務發生未預期錯誤。",
                        "details": None,
                    },
                }
            }
        },
    },
}


def _build_health_response():
    return build_success_envelope(build_health_data(is_model_warmed(DEFAULT_MODEL)))


@router.get(
    "/service-info",
    response_model=ServiceInfoResponse,
    dependencies=[Depends(require_internal_api_token)],
    summary="取得服務基礎資訊",
    description="回傳服務名稱、預設模型、主要 API 路徑，以及伺服器端固定處理策略。",
    response_description="統一 success envelope 的服務資訊。",
    tags=["system"],
    operation_id="getServiceInfo",
    responses=COMMON_ERROR_RESPONSES,
)
def service_info():
    return build_success_envelope(build_service_info_data())


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="取得服務健康狀態",
    description="回傳服務基本狀態、預設模型，以及預設模型是否已完成 warmup。",
    response_description="統一 success envelope 的健康檢查結果。",
    tags=["system"],
    operation_id="getHealth",
    responses=COMMON_ERROR_RESPONSES,
)
def health():
    return _build_health_response()


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="取得輕量健康檢查狀態",
    description="提供給容器 healthcheck 使用的輕量端點，不會主動觸發新的推論請求。",
    response_description="統一 success envelope 的輕量健康檢查結果。",
    tags=["system"],
    operation_id="getHealthz",
    responses=COMMON_ERROR_RESPONSES,
)
def healthz():
    return _build_health_response()


@router.get(
    "/warmup",
    response_model=WarmupResponse,
    dependencies=[Depends(require_internal_api_token)],
    summary="執行模型 warmup",
    description="初始化預設 rembg 模型 session，讓服務在正式接收去背請求前先完成預熱。",
    response_description="統一 success envelope 的 warmup 執行結果。",
    tags=["system"],
    operation_id="runWarmup",
    responses=COMMON_ERROR_RESPONSES,
)
def warmup():
    warmup_models()
    return build_success_envelope(build_warmup_data(is_model_warmed(DEFAULT_MODEL)))


@router.post(
    "/remove-bg",
    dependencies=[Depends(require_internal_api_token)],
    responses={
        **COMMON_ERROR_RESPONSES,
        200: {
            "description": "去背成功，直接回傳透明背景 PNG 檔案。",
            "content": {
                "image/png": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    }
                }
            },
        },
        502: {
            "model": GenericErrorResponse,
            "description": "底層去背引擎或後處理階段失敗。",
            "content": {
                "application/json": {
                    "examples": {
                        "rembg_failed": {
                            "summary": "rembg execution failed",
                            "value": {
                                "ok": False,
                                "error": {
                                    "code": "REMBG_EXECUTION_FAILED",
                                    "message": "去背引擎執行失敗。",
                                    "details": {
                                        "cause": "session initialization failed",
                                        "model": "isnet-general-use",
                                        "quality": "fast",
                                    },
                                },
                            },
                        },
                        "postprocess_failed": {
                            "summary": "postprocess failed",
                            "value": {
                                "ok": False,
                                "error": {
                                    "code": "POSTPROCESS_FAILED",
                                    "message": "去背後處理失敗。",
                                    "details": {
                                        "cause": "cannot merge alpha",
                                        "model": "isnet-general-use",
                                        "quality": "fast",
                                    },
                                },
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
        "成功時直接回傳 `image/png`。"
        "若遮罩信心不足或邊界品質不足，則維持 `422 application/json` 錯誤 envelope。"
        "僅接受副檔名白名單：`.jpg`、`.jpeg`、`.png`、`.webp`、`.avif`、`.heic`、`.heif`。"
        "request 若有帶 `content_type`，也必須落在白名單：`image/jpeg`、`image/png`、`image/webp`、`image/avif`、`image/heic`、`image/heif`。"
    ),
    response_description="去背成功時回傳 PNG；失敗時維持既有 JSON 錯誤格式。",
    tags=["background-removal"],
    operation_id="removeBackground",
)
async def remove_bg(
    file: UploadFile = File(
        ...,
        description=(
            "要進行去背的單張圖片檔案。"
            "請使用 multipart/form-data 上傳，欄位名稱必須是 `file`。"
            "僅接受 `.jpg`、`.jpeg`、`.png`、`.webp`、`.avif`、`.heic`、`.heif`。"
            "若 request 有帶 `content_type`，也只接受 `image/jpeg`、`image/png`、`image/webp`、`image/avif`、`image/heic`、`image/heif`。"
            "`svg` / `image/svg+xml` 明確禁止。"
        ),
    ),
):
    validate_upload_file_for_remove_bg(file)
    raw = await file.read()
    result = process_remove_bg(raw=raw)
    return Response(
        content=result["image_bytes"],
        media_type="image/png",
        headers=result["headers"],
    )