from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from PIL import Image
from rembg import remove, new_session

import io
import threading

app = FastAPI()

# 方便 Next.js demo 串接；若你之後要鎖網域再調整 allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Lazy-load rembg session (u2netp) ----
_session = None
_session_lock = threading.Lock()


def get_session():
    """
    Lazy load：
    - /health 不會觸發載入模型
    - 第一次 /remove-bg 才初始化 u2netp session
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = new_session("u2netp")
    return _session


def _load_image_from_upload(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # 強制解碼
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="檔案不是有效圖片或圖片已損毀")


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    # max_side <= 0 代表不縮圖
    if max_side <= 0:
        return img

    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # 用 LANCZOS 縮圖品質較好
    return img.resize((new_w, new_h), Image.LANCZOS)


def _to_png_bytes(img: Image.Image) -> bytes:
    # 避免一些模式造成輸出問題，統一轉 RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    # 刻意不呼叫 get_session()，避免 Render health check 期間載入模型
    return {"ok": True}


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    max_side: int = Query(0, ge=0, le=4096),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="檔案內容為空")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="檔案大小超過 12MB")

    session = get_session()

    try:
        # 不縮圖：走最快路徑（直接 remove 原始 bytes）
        if max_side <= 0:
            out_bytes = await run_in_threadpool(lambda: remove(data, session=session))
            return Response(content=out_bytes, media_type="image/png")

        # 需要縮圖：才解碼 → 縮放 → 轉 PNG bytes → remove
        img = _load_image_from_upload(data)
        img = _resize_max_side(img, max_side)
        png_bytes = _to_png_bytes(img)

        out_bytes = await run_in_threadpool(lambda: remove(png_bytes, session=session))
        return Response(content=out_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "remove-bg failed", "detail": str(e)},
        )