import os
from pathlib import Path

from dotenv import load_dotenv

# 只在本機開發時載入
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

SERVICE_NAME = "rembg-service"

DEFAULT_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")

ALLOWED_MODELS = {
    "u2netp",
    "u2net",
    "isnet-general-use",
}

DEFAULT_MAX_SIDE = 512
DEFAULT_QUALITY = "fast"
DEFAULT_REJECT_LOW_CONFIDENCE = True
DEFAULT_REJECT_EDGE_QUALITY = True

FINAL_OUTPUT_LONGEST_SIDE = 512

INTERNAL_API_TOKEN_ENV_NAME = "INTERNAL_API_TOKEN"


def get_internal_api_token() -> str:
    return os.getenv(INTERNAL_API_TOKEN_ENV_NAME, "")