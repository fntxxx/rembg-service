from fastapi import FastAPI

from app.api.routes import router
from app.core.session import warmup_models

app = FastAPI(
    title="rembg-service",
    summary="去背 API，提供健康檢查、模型 warmup 與單張圖片去背處理。",
    description=(
        "此服務部署於 Hugging Face Spaces，Swagger UI 直接顯示於 `/`。"
        "主要用途為上傳單張圖片後，回傳去背後的 PNG，或在低信心情況下回傳 JSON 拒絕結果。"
        "\n\n"
        "- Swagger UI：`/`\n"
        "- OpenAPI schema：`/openapi.json`\n"
        "- 檔案上傳方式：`multipart/form-data`\n"
        "- 上傳欄位名稱：`file`\n"
        "- 成功回應：`image/png`\n"
        "- 低信心拒絕：`422 application/json`"
    ),
    version="1.0.0",
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
            "description": "圖片去背處理端點。成功時回傳 PNG，拒絕時回傳 JSON。",
        },
    ],
)
app.include_router(router)


@app.on_event("startup")
def startup_event():
    # 啟動時先完成模型預熱；如果模型壞掉或下載檔有問題，
    # 讓服務直接啟動失敗，比啟動成功後第一筆 request 才爆更容易發現。
    warmup_models()