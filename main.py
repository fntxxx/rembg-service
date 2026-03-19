import io
import os
import time
import threading
import numpy as np
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageFilter

from rembg import remove, new_session

app = FastAPI()

@app.on_event("startup")
def warmup_models():
    # 預載常用模型，避免第一次 request 卡住
    get_session("u2netp")   # base
    get_session("u2net")    # fallback

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

    timing = {}
    request_started_at = time.perf_counter()

    timing["fallback_used"] = False
    timing["fallback_sec"] = 0.0

    # ---- Resize to control cost ----
    try:
        resize_started_at = time.perf_counter()

        # 真正的原圖（最後保色、輸出用）
        original_full = Image.open(io.BytesIO(raw)).convert("RGBA")

        # 工作圖（縮小後給 rembg / alpha pipeline 用）
        img = original_full.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        in_bytes = buf.getvalue()

        timing["resize_sec"] = round(time.perf_counter() - resize_started_at, 4)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # ---- Choose model ----
    model_name = model or DEFAULT_MODEL
    actual_model = model_name
    try:
        get_session(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        base_remove_started_at = time.perf_counter()
        out_bytes = run_remove(in_bytes, model_name, quality)
        timing["base_remove_sec"] = round(time.perf_counter() - base_remove_started_at, 4)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": "rembg failed", "detail": str(e), "model": model_name, "quality": quality},
        )

    try:
        postprocess_started_at = time.perf_counter()

        # 用去背結果的 alpha，但保留原圖 RGB，避免衣物本體顏色漂移
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")

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

        if (
            model_name == "u2netp"
            and quality == "fast"
            and (
                foreground_ratio <= 0.20
                or alpha_mean <= 50
            )
        ):
            print(">>> USING FALLBACK <<<", foreground_ratio, alpha_mean)
            fallback_model = "u2net"
            fallback_quality = "fast"

            try:
                fallback_started_at = time.perf_counter()
                fallback_bytes = run_remove(in_bytes, fallback_model, fallback_quality)
                fallback_img = Image.open(io.BytesIO(fallback_bytes)).convert("RGBA")
                timing["fallback_sec"] = round(time.perf_counter() - fallback_started_at, 4)
                timing["fallback_used"] = True
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

            fallback_alpha = fallback_img.getchannel("A")
            base_alpha = alpha  # 原本 u2netp 的 alpha

            fallback_np = np.array(fallback_alpha, dtype=np.float32)
            base_np = np.array(base_alpha, dtype=np.float32)

            # 只在 base 很弱的地方才讓 fallback 介入，避免污染已經正確的主體邊界
            weak_base_mask = base_np < 50
            very_weak_mask = base_np < 25
            mask = (fallback_np > 10) & weak_base_mask

            alpha_np = base_np.copy()
            blended = fallback_np[mask] * 0.78 + base_np[mask] * 0.22

            cap = np.where(base_np[mask] < 25, base_np[mask] + 120, base_np[mask] + 80)
            alpha_np[mask] = np.minimum(blended, cap)

            force_mask = (fallback_np > 10) & (base_np < 15)

            # 只保留比較連續的弱區，避免把鞋底雜訊顆粒直接拉亮
            force_alpha = np.zeros_like(alpha_np, dtype=np.float32)
            force_alpha[force_mask] = fallback_np[force_mask] * 1.1

            force_img = Image.fromarray(np.clip(force_alpha, 0, 255).astype(np.uint8))
            force_img = force_img.filter(ImageFilter.GaussianBlur(radius=0.6))
            force_np = np.array(force_img, dtype=np.float32)

            alpha_np = np.maximum(alpha_np, force_np)

            alpha_np = np.clip(alpha_np, 0, 255)
            alpha = Image.fromarray(alpha_np.astype(np.uint8))

            alpha_data = list(alpha.getdata())
            total_pixels = len(alpha_data)
            alpha_mean = sum(alpha_data) / total_pixels
            foreground_pixels = sum(1 for p in alpha_data if p >= 8)
            foreground_ratio = foreground_pixels / total_pixels

            used_fallback = True
            actual_model = f"{model_name}+fallback"
            original_rgba = original_full.convert("RGBA")

        # 先做「分流」，再套不同 curve
        # 目標：
        # - apparel：前景集中，可較積極，去掉灰霧
        # - footwear：主體小、邊界碎，最保守，避免缺角
        # - flatlay / others：中間值，保守處理

        if used_fallback:
            alpha_np = np.array(alpha, dtype=np.float32)
            mid_alpha_ratio = float(((alpha_np >= 10) & (alpha_np <= 200)).mean())

            # 針對低 + 中低 alpha 做「收斂」，但避免傷到鞋底主體
            low_mask = alpha_np < 30
            mid_mask = (alpha_np >= 30) & (alpha_np < 80)

            alpha_np = np.clip(alpha_np, 0, 255)
            alpha = Image.fromarray(alpha_np.astype(np.uint8))

            # 只在中低前景占比時做更輕的 blur，避免鞋底邊緣被再度抹薄
            if (
                0.14 <= foreground_ratio <= 0.24
                and alpha_mean > 60
                and mid_alpha_ratio < 0.08
            ):
                alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.3))

            debug_alpha = np.array(alpha, dtype=np.uint8)
            debug_alpha_mean = float(debug_alpha.mean())
            debug_foreground_ratio = float((debug_alpha >= 8).mean())
            debug_mid_alpha_ratio = float(((debug_alpha >= 10) & (debug_alpha <= 200)).mean())

            print(
                ">>> AFTER FALLBACK CURVE <<<",
                debug_foreground_ratio,
                debug_alpha_mean,
                debug_mid_alpha_ratio,
            )
        else:
            if foreground_ratio >= 0.42 and alpha_mean >= 95:
                alpha = alpha.point(lambda p: int(min(255, max(0, (p - 30) * 1.8))))
            elif foreground_ratio <= 0.26 or alpha_mean <= 60:
                alpha = alpha.point(lambda p: int(min(255, max(0, (p - 58) * 1.05))))
            else:
                alpha = alpha.point(lambda p: int(min(255, max(0, (p - 48) * 1.18))))

            if alpha_mean < 60:
                # 低對比，禁止 blur
                pass
            elif 0.08 <= foreground_ratio <= 0.35:
                alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))

        # alpha 還在小圖

        # 👉 resize alpha 到原圖
        target_size = original_full.size

        # 👉 限制最大輸出尺寸
        max_output_side = 1024

        w, h = target_size
        longest = max(w, h)

        if longest > max_output_side:
            scale = max_output_side / float(longest)
            target_size = (int(w * scale), int(h * scale))

        # resize 原圖 + alpha 同步
        original_resized = original_full.resize(target_size, Image.LANCZOS)
        alpha = alpha.resize(target_size, Image.LANCZOS)

        merged = original_resized.copy()
        merged.putalpha(alpha)

        final_buf = io.BytesIO()
        merged.save(final_buf, format="PNG")
        final_bytes = final_buf.getvalue()

        timing["postprocess_sec"] = round(time.perf_counter() - postprocess_started_at, 4)
        timing["total_sec"] = round(time.perf_counter() - request_started_at, 4)
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

    print(
        "[remove-bg timing]",
        {
            "model": model_name,
            "quality": quality,
            "max_side": max_side,
            "resize_sec": timing["resize_sec"],
            "base_remove_sec": timing["base_remove_sec"],
            "fallback_used": timing["fallback_used"],
            "fallback_sec": timing["fallback_sec"],
            "postprocess_sec": timing["postprocess_sec"],
            "total_sec": timing["total_sec"],
            "foreground_ratio": round(foreground_ratio, 6),
            "alpha_mean": round(alpha_mean, 4),
        }
    )

    return Response(
        content=final_bytes,
        media_type="image/png",
        headers={
            "X-RemoveBg-Fallback-Used": "true" if timing["fallback_used"] else "false",
            "X-RemoveBg-Model": actual_model,
        }
    )
