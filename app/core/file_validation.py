from pathlib import Path

from fastapi import HTTPException, UploadFile

ALLOWED_IMAGE_EXTENSIONS = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".heic",
    ".heif",
})

ALLOWED_IMAGE_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/avif",
    "image/heic",
    "image/heif",
})


def validate_image_filename(filename: str | None) -> None:
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid image")

    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid image")


def validate_image_content_type(content_type: str | None) -> None:
    if content_type is None:
        return

    normalized = content_type.strip().lower()
    if not normalized:
        return

    if normalized not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image")


def validate_upload_file_for_remove_bg(file: UploadFile) -> None:
    validate_image_filename(file.filename)
    validate_image_content_type(file.content_type)
