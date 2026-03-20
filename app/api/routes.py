from typing import Optional

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from app.core.config import DEFAULT_MODEL
from app.core.session import is_model_warmed
from app.services.remove_bg_service import process_remove_bg

router = APIRouter()


@router.get("/")
def root():
    return {
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "endpoints": {
            "health": "/health",
            "healthz": "/healthz",
            "remove_bg": "POST /remove-bg",
        },
    }


@router.get("/health")
def health():
    return {
        "ok": True,
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": is_model_warmed(DEFAULT_MODEL),
    }


@router.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "rembg-service",
        "model": DEFAULT_MODEL,
        "model_warmed": is_model_warmed(DEFAULT_MODEL),
    }


@router.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    max_side: int = Query(512, ge=256, le=2048),
    quality: str = Query("fast", pattern="^(fast|high)$"),
    model: Optional[str] = Query(None),
    reject_low_confidence: bool = Query(True),
    reject_edge_quality: bool = Query(True),
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