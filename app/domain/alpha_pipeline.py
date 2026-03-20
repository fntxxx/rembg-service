from typing import Dict, Tuple

import numpy as np
from PIL import Image

from app.domain.metrics import compute_edge_quality_metrics


def collect_alpha_stats(alpha: Image.Image) -> Dict[str, float]:
    alpha_data = list(alpha.getdata())
    total_pixels = len(alpha_data)

    alpha_mean = sum(alpha_data) / total_pixels

    # 把很淡的半透明邊緣先排除，避免 foreground_ratio 被毛邊灌高
    foreground_pixels = sum(1 for p in alpha_data if p >= 8)
    foreground_ratio = foreground_pixels / total_pixels

    alpha_np_base = np.array(alpha, dtype=np.float32)

    mid_alpha_ratio = float(((alpha_np_base >= 10) & (alpha_np_base <= 200)).mean())
    low_alpha_ratio = float(((alpha_np_base >= 8) & (alpha_np_base < 30)).mean())
    high_alpha_ratio = float((alpha_np_base >= 160).mean())

    edge_metrics = compute_edge_quality_metrics(alpha)

    return {
        "alpha_mean": alpha_mean,
        "foreground_ratio": foreground_ratio,
        "mid_alpha_ratio": mid_alpha_ratio,
        "low_alpha_ratio": low_alpha_ratio,
        "high_alpha_ratio": high_alpha_ratio,
        "edge_band_ratio": edge_metrics["edge_band_ratio"],
        "edge_band_mid_ratio": edge_metrics["edge_band_mid_ratio"],
        "edge_band_low_ratio": edge_metrics["edge_band_low_ratio"],
    }


def should_use_fallback(model_name: str, quality: str, foreground_ratio: float, alpha_mean: float) -> bool:
    return False


def apply_fallback_blend(alpha: Image.Image, fallback_alpha: Image.Image) -> Image.Image:
    # fallback 目前停用，暫不進行 blend
    return alpha


def apply_alpha_curve(
    alpha: Image.Image,
    used_fallback: bool,
    foreground_ratio: float,
    alpha_mean: float,
) -> Tuple[Image.Image, Dict[str, float]]:
    alpha_np = np.array(alpha, dtype=np.float32)

    # 記錄修正前指標
    before_mid_alpha_ratio = float(((alpha_np >= 10) & (alpha_np <= 200)).mean())
    before_high_alpha_ratio = float((alpha_np >= 220).mean())

    # -------------------------------------------------
    # 核心想法：
    # 1. 很低 alpha 的霧邊直接收掉，避免白底稀釋
    # 2. 中高 alpha 往上推，讓主體內部更接近實心
    # 3. 不把全部邊界硬切成 0/255，保留少量柔邊
    # -------------------------------------------------

    # A. 很淡的外圍霧邊：直接收掉
    very_low_mask = alpha_np < 20
    alpha_np[very_low_mask] = 0.0

    # B. 低 alpha 邊緣：大幅壓縮，縮窄 soft edge
    low_mask = (alpha_np >= 20) & (alpha_np < 80)
    alpha_np[low_mask] = np.maximum(0.0, (alpha_np[low_mask] - 20.0) * 0.85)

    # C. 中 alpha 區：往上推，避免主體發灰
    mid_mask = (alpha_np >= 80) & (alpha_np < 160)
    alpha_np[mid_mask] = np.minimum(255.0, alpha_np[mid_mask] * 1.18 + 8.0)

    # D. 高 alpha 區：再推高，讓主體核心更扎實
    high_mask = (alpha_np >= 160) & (alpha_np < 235)
    alpha_np[high_mask] = np.minimum(255.0, alpha_np[high_mask] * 1.10 + 10.0)

    # E. 核心區直接接近實心
    core_mask = alpha_np >= 235
    alpha_np[core_mask] = 255.0

    alpha_np = np.clip(alpha_np, 0, 255).astype(np.uint8)
    alpha = Image.fromarray(alpha_np)

    alpha_np = np.array(alpha, dtype=np.float32)

    # -----------------------------
    # core boost（關鍵）
    # -----------------------------
    boost_core_mask = alpha_np >= 120

    # 把主體往不透明推
    alpha_np[boost_core_mask] = np.clip(
        alpha_np[boost_core_mask] * 1.4 + 30,
        0,
        255
    )

    # 保證核心接近實心
    alpha_np[alpha_np > 220] = 255

    alpha = Image.fromarray(alpha_np.astype(np.uint8))
    after_alpha_np = np.array(alpha, dtype=np.uint8)

    return alpha, {
        "before_mid_alpha_ratio": before_mid_alpha_ratio,
        "before_high_alpha_ratio": before_high_alpha_ratio,
        "after_mid_alpha_ratio": float(((after_alpha_np >= 10) & (after_alpha_np <= 200)).mean()),
        "after_high_alpha_ratio": float((after_alpha_np >= 220).mean()),
    }


def compute_bbox_ratios(alpha: Image.Image) -> Tuple[float, float, np.ndarray]:
    debug_alpha_np = np.array(alpha, dtype=np.float32)
    bbox_fg = debug_alpha_np >= 8
    ys, xs = np.where(bbox_fg)

    bbox_width_ratio = 0.0
    bbox_height_ratio = 0.0
    if len(xs) > 0 and len(ys) > 0:
        bbox_width_ratio = float((xs.max() - xs.min() + 1) / debug_alpha_np.shape[1])
        bbox_height_ratio = float((ys.max() - ys.min() + 1) / debug_alpha_np.shape[0])

    return bbox_width_ratio, bbox_height_ratio, debug_alpha_np
