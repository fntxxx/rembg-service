import os

# 預設模型：目前服務預設以 isnet-general-use 跑
DEFAULT_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")

ALLOWED_MODELS = {
    "u2netp",
    "u2net",
    "isnet-general-use",
}

MAX_OUTPUT_SIDE = 1024
