from typing import Tuple

import numpy as np
from PIL import Image


def estimate_background_rgb(image: Image.Image) -> Tuple[int, int, int]:
    """
    從四個角落估一個背景色。
    適合你的商品圖場景：背景乾淨、接近純白 / 淺灰。
    """
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]

    patch_h = max(8, int(h * 0.05))
    patch_w = max(8, int(w * 0.05))

    patches = [
        rgb[0:patch_h, 0:patch_w],                 # 左上
        rgb[0:patch_h, w - patch_w:w],            # 右上
        rgb[h - patch_h:h, 0:patch_w],            # 左下
        rgb[h - patch_h:h, w - patch_w:w],        # 右下
    ]

    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    median_rgb = np.median(samples, axis=0)

    return tuple(int(round(v)) for v in median_rgb)


def decontaminate_edge_rgb(
    original_rgb: Image.Image,
    alpha: Image.Image,
    bg_rgb: Tuple[int, int, int],
    edge_alpha_min: int = 20,
    edge_alpha_max: int = 200,
    restore_strength: float = 0.85,
) -> Image.Image:
    """
    只處理邊界帶，不動主體核心區顏色。

    原理：
    observed = fg * a + bg * (1 - a)

    在 edge band 內估回較乾淨的 fg，降低白邊 / 灰邊。
    """
    rgb_np = np.array(original_rgb.convert("RGB"), dtype=np.float32)
    alpha_np = np.array(alpha.convert("L"), dtype=np.float32)

    result = rgb_np.copy()

    edge_mask = (alpha_np >= edge_alpha_min) & (alpha_np < edge_alpha_max)
    if not np.any(edge_mask):
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")

    a = alpha_np / 255.0
    a_safe = np.clip(a, 1e-4, 1.0)

    bg = np.zeros_like(rgb_np, dtype=np.float32)
    bg[..., 0] = bg_rgb[0]
    bg[..., 1] = bg_rgb[1]
    bg[..., 2] = bg_rgb[2]

    # 邊界去污染：把背景混入量扣回來
    restored = (rgb_np - bg * (1.0 - a_safe[..., None])) / a_safe[..., None]
    restored = np.clip(restored, 0, 255)

    # 只在 edge band 內逐步混合，避免修太重
    band_width = max(1.0, float(edge_alpha_max - edge_alpha_min))
    edge_strength = np.clip((alpha_np - edge_alpha_min) / band_width, 0.0, 1.0)

    # 稍微偏保守：靠外側修弱一點，靠內側修強一點
    edge_strength = np.power(edge_strength, 1.25) * restore_strength

    result[edge_mask] = (
        rgb_np[edge_mask] * (1.0 - edge_strength[edge_mask, None])
        + restored[edge_mask] * edge_strength[edge_mask, None]
    )

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")