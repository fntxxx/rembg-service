from pathlib import Path

from fastapi import UploadFile

from app.core.error_codes import ErrorCode
from app.core.exceptions import ApiError
from app.core.image_policy import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    ALLOWED_IMAGE_EXTENSIONS,
    REJECTED_IMAGE_CONTENT_TYPES,
    REJECTED_IMAGE_EXTENSIONS,
)


def _raise_unsupported_media_type() -> None:
    raise ApiError(
        status_code=415,
        code=ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        message="Unsupported media type",
        details=None,
    )


def validate_image_filename(filename: str | None) -> None:
    if not filename:
        _raise_unsupported_media_type()

    suffix = Path(filename).suffix.lower()
    if not suffix or suffix in REJECTED_IMAGE_EXTENSIONS or suffix not in ALLOWED_IMAGE_EXTENSIONS:
        _raise_unsupported_media_type()


def validate_image_content_type(content_type: str | None) -> None:
    if content_type is None:
        return

    normalized = content_type.strip().lower()
    if not normalized:
        return

    if normalized in REJECTED_IMAGE_CONTENT_TYPES or normalized not in ALLOWED_IMAGE_CONTENT_TYPES:
        _raise_unsupported_media_type()


def validate_upload_file_for_remove_bg(file: UploadFile) -> None:
    validate_image_filename(file.filename)
    validate_image_content_type(file.content_type)
