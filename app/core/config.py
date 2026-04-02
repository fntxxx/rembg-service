import os

SERVICE_NAME = "rembg-service"

# 預設模型：目前服務預設以 isnet-general-use 跑
DEFAULT_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")

ALLOWED_MODELS = {
    "u2netp",
    "u2net",
    "isnet-general-use",
}

# 對外 API 不再暴露這些固定策略參數，由伺服器端統一套用。
DEFAULT_MAX_SIDE = 512
DEFAULT_QUALITY = "fast"
DEFAULT_REJECT_LOW_CONFIDENCE = True
DEFAULT_REJECT_EDGE_QUALITY = True

FINAL_OUTPUT_LONGEST_SIDE = 512