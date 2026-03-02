from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from rembg import remove
import io

app = FastAPI()


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


@app.on_event("startup")
def warmup():
    # 讓初始化成本在啟動時吃掉，並對齊「直接 remove(bytes)」最快路徑
    dummy = Image.new("RGB", (64, 64), (255, 255, 255))
    buf = io.BytesIO()
    dummy.save(buf, format="JPEG", quality=80)
    _ = remove(buf.getvalue())


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    max_side: int = Query(0, ge=0, le=4096),
):
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="檔案大小超過 12MB")

    # 不縮圖：走最快路徑（直接 remove 原始 bytes）
    if max_side <= 0:
        out_bytes = remove(data)
        return Response(content=out_bytes, media_type="image/png")

    # 需要縮圖：才解碼 → 縮放 → 轉 PNG bytes → remove
    img = _load_image_from_upload(data)
    img = _resize_max_side(img, max_side)
    out_bytes = remove(_to_png_bytes(img))
    return Response(content=out_bytes, media_type="image/png")


def _to_png_bytes(img: Image.Image) -> bytes:
    # 避免一些模式造成輸出問題，統一轉 RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()