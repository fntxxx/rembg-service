import threading
from typing import Dict

from rembg import new_session

from app.core.config import ALLOWED_MODELS, DEFAULT_MODEL

# ---- Session cache (lazy load) ----
_sessions: Dict[str, object] = {}
_sessions_lock = threading.Lock()


def get_session(model_name: str):
    """
    Lazy load:
    - /healthz 不會觸發載入模型
    - 第一次 /remove-bg 才初始化指定模型 session
    - 以 model_name 做 cache（同模型不會重複載入）
    """
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    s = _sessions.get(model_name)
    if s:
        return s

    with _sessions_lock:
        s = _sessions.get(model_name)
        if not s:
            _sessions[model_name] = new_session(model_name)
        return _sessions[model_name]


def warmup_models():
    # 預載目前正式使用的 base model
    get_session(DEFAULT_MODEL)


def is_model_warmed(model_name: str) -> bool:
    return model_name in _sessions