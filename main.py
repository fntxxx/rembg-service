import io
import os
import threading
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response, JSONResponse
from PIL import Image

from rembg import remove, new_session

app = FastAPI()

# ---- Session cache (lazy load) ----
_sessions: Dict[str, object] = {}
_sessions_lock = threading.Lock()

# 預設模型：一般物件（衣物平拍/掛拍）通常更乾淨
DEFAULT_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")  # 可設回 u2netp

ALLOWED_MODELS = {
    "u2netp",
    "u2net",
    "isnet-general-use",
}

def get_session(model_name: str):
    """
    Lazy load:
    - /health 不會觸發載入模型
    - 第一次 /remove-bg 才初始化指定模型 session
    - 以 model_name 做 cache（同模型不會重複載入）
    """
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    s = _sessions.get(model_name)
    if s is not None:
        return s

    with _sessions_lock:
        s = _sessions.get(model_name)
        if s is None:
            _sessions[model_name] = new_session(model_name)
        return _sessions[model_name]

@app.get("/")
def root():
    return {
        "service": "rembg-service",
        "endpoints": {
            "health": "/health",
            "remove_bg": "POST /remove-bg"
        }
    }

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    max_side: int = Query(768, ge=256, le=2048),
    quality: str = Query("fast", pattern="^(fast|high)$"),
    model: Optional[str] = Query(None),
):
    # ---- Basic validation ----
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # ---- Resize to control cost ----
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

    # encode resized image to bytes (keeps pipeline deterministic)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    in_bytes = buf.getvalue()

    # ---- Choose model + session (lazy) ----
    model_name = model or DEFAULT_MODEL
    try:
        session = get_session(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ---- Quality switch (alpha matting) ----
    # 衣物照常見問題：毛邊/白邊/暈光 -> alpha matting 通常改善顯著
    remove_kwargs = {}
    if quality == "high":
        remove_kwargs = {
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 12,  # 衣物邊界可稍大，通常更乾淨
        }

    try:
        out_bytes = remove(in_bytes, session=session, **remove_kwargs)
    except Exception as e:
        # 讓上游錯誤更好診斷
        return JSONResponse(
            status_code=502,
            content={"error": "rembg failed", "detail": str(e), "model": model_name, "quality": quality},
        )

    return Response(content=out_bytes, media_type="image/png")