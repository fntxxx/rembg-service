from typing import Dict

import numpy as np
from PIL import Image, ImageFilter


def compute_edge_quality_metrics(alpha: Image.Image) -> Dict[str, float]:
    alpha_np_base = np.array(alpha, dtype=np.float32)

    solid_mask = alpha_np_base >= 160
    mid_mask = (alpha_np_base >= 30) & (alpha_np_base < 160)
    low_mask = (alpha_np_base >= 8) & (alpha_np_base < 30)
    fg_mask = alpha_np_base >= 8

    solid_img = Image.fromarray((solid_mask.astype(np.uint8) * 255))
    solid_dilate = solid_img.filter(ImageFilter.MaxFilter(5))
    solid_erode = solid_img.filter(ImageFilter.MinFilter(5))

    solid_dilate_np = np.array(solid_dilate) > 0
    solid_erode_np = np.array(solid_erode) > 0
    edge_band_mask = solid_dilate_np & (~solid_erode_np)

    edge_band_pixels = int(edge_band_mask.sum())
    fg_pixels = int(fg_mask.sum())

    edge_band_ratio = float(edge_band_pixels / max(fg_pixels, 1))
    edge_band_mid_ratio = float((mid_mask & edge_band_mask).sum() / max(edge_band_pixels, 1))
    edge_band_low_ratio = float((low_mask & edge_band_mask).sum() / max(edge_band_pixels, 1))

    return {
        "edge_band_ratio": edge_band_ratio,
        "edge_band_mid_ratio": edge_band_mid_ratio,
        "edge_band_low_ratio": edge_band_low_ratio,
        "edge_band_pixels": float(edge_band_pixels),
        "fg_pixels": float(fg_pixels),
    }
