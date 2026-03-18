import io
import os
import threading
import numpy as np
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageFilter

from rembg import remove, new_session

app = FastAPI()

# ---- Session cache (lazy load) ----
_sessions: Dict[str, object] = {}
_sessions_lock = threading.Lock()

# 預設模型：第一輪 benchmark 顯示 apparel 以 u2netp 表現較佳
DEFAULT_MODEL = os.getenv("REMBG_MODEL", "u2netp")

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
    max_side: int = Query(512, ge=256, le=2048),
    quality: str = Query("fast", pattern="^(fast|high)$"),
    model: Optional[str] = Query(None),
):
    # ---- Basic validation ----
    # 不把 content-type 當硬門檻：有些上游/瀏覽器會送 application/octet-stream
    # 以 PIL 實際能否解碼為準（下面 Image.open 失敗就會回 400）

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # ---- Resize to control cost ----
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        original_full = img.copy()

        original_buf = io.BytesIO()
        original_full.save(original_buf, format="PNG")
        original_full_bytes = original_buf.getvalue()
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

    # ---- Choose model ----
    model_name = model or DEFAULT_MODEL
    try:
        get_session(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        out_bytes = run_remove(in_bytes, model_name, quality)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": "rembg failed", "detail": str(e), "model": model_name, "quality": quality},
        )

    try:
        # 用去背結果的 alpha，但保留原圖 RGB，避免衣物本體顏色漂移
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")

        if out_img.size != original_full.size:
            out_img = out_img.resize(original_full.size, Image.LANCZOS)

        original_rgba = original_full.convert("RGBA")

        alpha = out_img.getchannel("A")

        # ---- Adaptive alpha routing ----
        alpha_data = list(alpha.getdata())
        total_pixels = len(alpha_data)

        alpha_mean = sum(alpha_data) / total_pixels

        # 把很淡的半透明邊緣先排除，避免 foreground_ratio 被毛邊灌高
        foreground_pixels = sum(1 for p in alpha_data if p >= 8)
        foreground_ratio = foreground_pixels / total_pixels

        used_fallback = False

        print(">>> BEFORE FALLBACK CHECK <<<", model_name, quality, foreground_ratio, alpha_mean)

        # ---- Targeted fallback for sparse / pale hard cases ----
        # 只攔極少數低 alpha、低前景占比案例，避免影響 apparel 主線速度
        if (
            model_name == "u2netp"
            and quality == "fast"
            and (
                foreground_ratio <= 0.20
                or alpha_mean <= 50
            )
        ):
            print(">>> USING FALLBACK <<<", foreground_ratio, alpha_mean)
            fallback_model = "isnet-general-use"
            fallback_quality = "high"

            try:
                fallback_bytes = run_remove(original_full_bytes, fallback_model, fallback_quality)
                fallback_img = Image.open(io.BytesIO(fallback_bytes)).convert("RGBA")
            except Exception as e:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": "fallback rembg failed",
                        "detail": str(e),
                        "model": fallback_model,
                        "quality": fallback_quality,
                    },
                )

            if fallback_img.size != original_full.size:
                fallback_img = fallback_img.resize(original_full.size, Image.LANCZOS)

            fallback_alpha = fallback_img.getchannel("A")
            base_alpha = alpha  # 原本 u2netp 的 alpha

            fallback_np = np.array(fallback_alpha, dtype=np.float32)
            base_np = np.array(base_alpha, dtype=np.float32)

            # 混合（保留 edge + 補主體）
            alpha_np = fallback_np * 0.72 + base_np * 0.28

            alpha_np = np.clip(alpha_np, 0, 255)
            alpha = Image.fromarray(alpha_np.astype(np.uint8))

            alpha_data = list(alpha.getdata())
            total_pixels = len(alpha_data)
            alpha_mean = sum(alpha_data) / total_pixels
            foreground_pixels = sum(1 for p in alpha_data if p >= 8)
            foreground_ratio = foreground_pixels / total_pixels

            model_name = fallback_model
            quality = fallback_quality
            used_fallback = True
            original_rgba = original_full.convert("RGBA")

        # 先做「分流」，再套不同 curve
        # 目標：
        # - apparel：前景集中，可較積極，去掉灰霧
        # - footwear：主體小、邊界碎，最保守，避免缺角
        # - flatlay / others：中間值，保守處理

        if used_fallback:
            alpha_np = np.array(alpha, dtype=np.float32)

            alpha_np = np.where(alpha_np < 10, 0, alpha_np)
            alpha_np = np.clip(alpha_np, 0, 255).astype(np.uint8)

            alpha = Image.fromarray(alpha_np).filter(ImageFilter.GaussianBlur(radius=1.2))

            debug_alpha = np.array(alpha, dtype=np.uint8)
            debug_alpha = np.where(debug_alpha < 15, 0, debug_alpha).astype(np.uint8)
            debug_alpha_mean = float(debug_alpha.mean())
            debug_foreground_ratio = float((debug_alpha >= 8).mean())
            debug_mid_alpha_ratio = float(((debug_alpha >= 10) & (debug_alpha <= 200)).mean())

            print(
                ">>> AFTER FALLBACK CURVE <<<",
                debug_foreground_ratio,
                debug_alpha_mean,
                debug_mid_alpha_ratio,
            )

            alpha = Image.fromarray(debug_alpha)
        else:
            if foreground_ratio >= 0.42 and alpha_mean >= 95:
                alpha = alpha.point(lambda p: int(min(255, max(0, (p - 30) * 1.8))))
            elif foreground_ratio <= 0.26 or alpha_mean <= 60:
                alpha = alpha.point(lambda p: int(min(255, max(0, (p - 58) * 1.05))))
            else:
                alpha = alpha.point(lambda p: int(min(255, max(0, (p - 48) * 1.18))))

            # 👉 新增條件（只在「非高密度主體」才 blur）
            if foreground_ratio <= 0.45:
                alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))

        merged = original_rgba.copy()
        merged.putalpha(alpha)

        final_buf = io.BytesIO()
        merged.save(final_buf, format="PNG")
        final_bytes = final_buf.getvalue()
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": "postprocess failed",
                "detail": str(e),
                "model": model_name,
                "quality": quality,
            },
        )

    return Response(content=final_bytes, media_type="image/png")
