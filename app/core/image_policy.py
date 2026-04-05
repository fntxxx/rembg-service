from __future__ import annotations

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

REJECTED_IMAGE_EXTENSIONS = frozenset({".svg"})
REJECTED_IMAGE_CONTENT_TYPES = frozenset({"image/svg+xml"})

_ALLOWED_EXTENSION_TEXT = "、".join(f"`{extension}`" for extension in sorted(ALLOWED_IMAGE_EXTENSIONS))
_ALLOWED_CONTENT_TYPE_TEXT = "、".join(f"`{content_type}`" for content_type in sorted(ALLOWED_IMAGE_CONTENT_TYPES))
_REJECTED_TEXT = "`svg` / `image/svg+xml`"

REMOVE_BG_RULES_DESCRIPTION = (
    f"僅接受副檔名白名單：{_ALLOWED_EXTENSION_TEXT}。"
    f"request 若有帶 `content_type`，也必須落在白名單：{_ALLOWED_CONTENT_TYPE_TEXT}。"
    f"{_REJECTED_TEXT} 明確禁止。"
)

REMOVE_BG_FILE_FIELD_DESCRIPTION = (
    "要進行去背的單張圖片檔案。"
    "請使用 multipart/form-data 上傳，欄位名稱必須是 `file`。"
    f"僅接受 {_ALLOWED_EXTENSION_TEXT}。"
    f"若 request 有帶 `content_type`，也只接受 {_ALLOWED_CONTENT_TYPE_TEXT}。"
    f"{_REJECTED_TEXT} 明確禁止。"
)
