from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pillow_heif

from app.api.routes import router
from app.core.exceptions import ApiError
from app.core.image_policy import REMOVE_BG_RULES_DESCRIPTION
from app.core.session import warmup_models
from app.schemas.responses import (
    build_api_error_response,
    build_http_exception_error,
    build_unexpected_error,
    build_validation_error,
)

def _register_heif_plugins() -> None:
    register_heif_opener = getattr(pillow_heif, "register_heif_opener", None)
    if callable(register_heif_opener):
        register_heif_opener()

    register_avif_opener = getattr(pillow_heif, "register_avif_opener", None)
    if callable(register_avif_opener):
        register_avif_opener()


_register_heif_plugins()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup_models()
    yield


app = FastAPI(
    title="rembg-service",
    summary="去背 API，提供健康檢查、模型 warmup 與單張圖片去背處理。",
    description=(
        "此服務部署於 Hugging Face Spaces，Swagger UI 直接顯示於 `/`。"
        "主要用途為上傳單張圖片後，成功時直接回傳去背後的 PNG，失敗時維持統一 JSON 錯誤 contract。"
        "\n\n"
        "- Swagger UI：`/`\n"
        "- OpenAPI schema：`/openapi.json`\n"
        "- 檔案上傳方式：`multipart/form-data`\n"
        "- 上傳欄位名稱：`file`\n"
        "- 成功回應：`200 image/png`\n"
        "- 錯誤回應：`4XX/5XX application/json`，格式為 `ok + error`\n"
        f"- {REMOVE_BG_RULES_DESCRIPTION}\n"
        "- 固定處理策略已內建於伺服器端，不需再帶 query 參數"
    ),
    version="1.1.1",
    docs_url="/",
    redoc_url=None,
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "system",
            "description": "服務狀態、啟動預熱與基礎資訊相關端點。",
        },
        {
            "name": "background-removal",
            "description": "圖片去背處理端點。成功時回傳 PNG，失敗時使用既有 JSON 錯誤 envelope。",
        },
    ],
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(ApiError)
async def api_error_handler(_request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content=build_api_error_response(exc),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=build_validation_error(exc),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=build_http_exception_error(exc),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=500,
        content=build_unexpected_error(),
    )
