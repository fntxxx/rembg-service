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


def apply_alpha_curve(alpha: Image.Image, used_fallback: bool, foreground_ratio: float, alpha_mean: float) -> Tuple[Image.Image, Dict[str, float]]:
    # fallback 目前停用；以下分支暫時不會進入
    if used_fallback:
        alpha_np = np.array(alpha, dtype=np.float32)
        mid_alpha_ratio = float(((alpha_np >= 10) & (alpha_np <= 200)).mean())

        # 針對低 + 中低 alpha 做真正的收斂，避免整體發灰
        low_mask = alpha_np < 30
        mid_mask = (alpha_np >= 30) & (alpha_np < 80)
        high_mask = alpha_np >= 80

        # 很低 alpha：明顯收斂，避免霧狀外擴
        alpha_np[low_mask] = np.maximum(0.0, alpha_np[low_mask] * 1.05)

        # 中低 alpha：輕收斂，避免把主體整體壓太薄
        alpha_np[mid_mask] = np.minimum(255.0, alpha_np[mid_mask] * 1.08 + 2.0)

        alpha_np[high_mask] = np.minimum(255.0, alpha_np[high_mask] * 1.06 + 4.0)

        alpha_np = np.clip(alpha_np, 0, 255)
        alpha = Image.fromarray(alpha_np.astype(np.uint8))

        # fallback 路徑先停用 blur，先優先解決 fade / 發灰
        # 之後若邊界真的過硬，再回來針對特定條件加回極輕 blur

        debug_alpha = np.array(alpha, dtype=np.uint8)
        debug_alpha_mean = float(debug_alpha.mean())
        debug_foreground_ratio = float((debug_alpha >= 8).mean())
        debug_mid_alpha_ratio = float(((debug_alpha >= 10) & (debug_alpha <= 200)).mean())

        print(
            ">>> AFTER FALLBACK CURVE <<<",
            debug_foreground_ratio,
            debug_alpha_mean,
            debug_mid_alpha_ratio,
        )

        return alpha, {
            "mid_alpha_ratio": mid_alpha_ratio,
        }

    alpha_np = np.array(alpha, dtype=np.float32)

    if foreground_ratio >= 0.42 and alpha_mean >= 95:
        # 主體大且扎實：可以稍微去霧，但不要太激進
        low_mask = alpha_np < 40
        mid_mask = (alpha_np >= 40) & (alpha_np < 120)
        high_mask = alpha_np >= 140
        core_mask = alpha_np >= 180
        high_non_core_mask = high_mask & (~core_mask)

        alpha_np[low_mask] = np.maximum(0.0, (alpha_np[low_mask] - 10.0) * 0.95)
        alpha_np[mid_mask] = np.maximum(0.0, (alpha_np[mid_mask] - 10.0) * 1.12)

        # 核心區強 boost
        alpha_np[core_mask] = np.minimum(255.0, alpha_np[core_mask] * 1.08 + 6.0)

        # 非核心但高 alpha，輕微 boost
        alpha_np[high_non_core_mask] = np.minimum(255.0, alpha_np[high_non_core_mask] * 1.02)

    elif foreground_ratio <= 0.20 or alpha_mean <= 50:
        # 小主體 / 低 alpha：最保守，優先避免整體變淡
        low_mask = alpha_np < 35
        mid_mask = (alpha_np >= 50) & (alpha_np < 120)
        high_mask = alpha_np >= 110

        alpha_np[low_mask] = np.maximum(0.0, (alpha_np[low_mask] - 6.0) * 0.95)
        alpha_np[mid_mask] = np.maximum(0.0, (alpha_np[mid_mask] - 6.0) * 0.98)
        alpha_np[high_mask] = alpha_np[high_mask]

    else:
        # 中間型：輕微收斂，但保留主體厚度
        low_mask = alpha_np < 35
        mid_mask = (alpha_np >= 35) & (alpha_np < 115)
        high_mask = alpha_np >= 115

        alpha_np[low_mask] = np.maximum(0.0, (alpha_np[low_mask] - 8.0) * 0.93)
        alpha_np[mid_mask] = np.maximum(0.0, (alpha_np[mid_mask] - 8.0) * 1.0)
        alpha_np[high_mask] = np.minimum(255.0, alpha_np[high_mask] * 1.02)

    alpha_np = np.clip(alpha_np, 0, 255)
    alpha = Image.fromarray(alpha_np.astype(np.uint8))

    # 非 fallback 路徑第一輪先完全停用 blur，避免把 alpha 再抹薄
    # 先觀察 fade 指標是否明顯下降
    return alpha, {}


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
