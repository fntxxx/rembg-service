
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("REMOVE_BG_API_URL", "http://127.0.0.1:7860/remove-bg")
DEFAULT_DATASET_DIR = Path(r"D:\DevData\remove_bg_testset")
REPORT_FILE = "test_remove_bg_color_diff_report.json"
TIMEOUT_SECONDS = 120
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif"}

# 預設只留目前最有 ROI 的組合，避免整批回歸被舊策略干擾
TEST_CONFIGS = [
    {
        "name": "isnet_fast_512",
        "params": {
            "model": "isnet-general-use",
            "quality": "fast",
            "max_side": 512,
        },
    },
]
CONFIG_MAP = {config["name"]: config for config in TEST_CONFIGS}

ALPHA_THRESHOLD = 8
CORE_ALPHA_THRESHOLD = 220
EDGE_ALPHA_MIN = 10
EDGE_ALPHA_MAX = 200
WASHOUT_DELTA_THRESHOLD = 12.0
VISUAL_FADE_BG_RGB = (245, 245, 245)
VISUAL_FADE_CORE_ALPHA_MIN = 160
VISUAL_FADE_LIGHTNESS_SHIFT_THRESHOLD = 8.0
VISUAL_FADE_RATIO_THRESHOLD = 0.35


@dataclass
class TestCase:
    file_path: Path


_original_cache: dict[str, Image.Image] = {}
_resized_original_cache: dict[tuple[str, int], Image.Image] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="去背色差／coverage 問題測試腳本")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help="測試集根目錄，預設為 D:\\DevData\\remove_bg_testset",
    )
    parser.add_argument(
        "--subset",
        action="append",
        help="只跑指定 subset，可重複傳入多次；subset 名稱為相對於 dataset_dir 的資料夾路徑",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIG_MAP.keys()),
        action="append",
        help="只跑指定 config，可重複傳入多次",
    )
    parser.add_argument(
        "--file",
        action="append",
        help="只跑指定檔名，可重複傳入多次，例如 --file white_shirt_01.jpg",
    )
    parser.add_argument(
        "--report-file",
        default=REPORT_FILE,
        help="輸出的 report 檔名",
    )
    parser.add_argument(
        "--save-output-dir",
        default=None,
        help="若有指定，會把 API 回傳 PNG 與比對預覽圖存到此資料夾",
    )
    parser.add_argument(
        "--save-only-flagged",
        action="store_true",
        help="搭配 --save-output-dir 使用，只存有問題 flag 的案例圖片",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="報表中列出最嚴重案例的數量",
    )
    parser.add_argument(
        "--compare-mode",
        choices=["resized", "original"],
        default="resized",
        help="resized: 用服務縮圖後原圖比對；original: 用原尺寸原圖縮到輸出尺寸後比對",
    )
    return parser.parse_args()


def iter_test_cases(dataset_dir: Path, file_filters: set[str] | None = None) -> list[TestCase]:
    cases: list[TestCase] = []
    for file_path in sorted(dataset_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_filters and file_path.name not in file_filters:
            continue
        cases.append(TestCase(file_path=file_path))
    return cases


def has_supported_images(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
    return False


def discover_subsets(dataset_dir: Path) -> list[tuple[str, Path]]:
    """
    自動找出所有「直接包含圖片檔」的資料夾，視為一個 subset。
    - 若 dataset root 自己就有圖片，subset 名稱記為 "."
    - 巢狀資料夾則用相對路徑，例如:
      tops/shirts
      shoes/sneakers
    """
    subsets: list[tuple[str, Path]] = []

    if has_supported_images(dataset_dir):
        subsets.append((".", dataset_dir))

    for dir_path in sorted(p for p in dataset_dir.rglob("*") if p.is_dir()):
        if has_supported_images(dir_path):
            rel_path = dir_path.relative_to(dataset_dir).as_posix()
            subsets.append((rel_path, dir_path))

    return subsets


def get_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def parse_bool_header(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def parse_float_header(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_response_meta(headers: requests.structures.CaseInsensitiveDict[str]) -> dict[str, Any]:
    fallback_header = headers.get("X-RemoveBg-Fallback-Used")
    model_header = headers.get("X-RemoveBg-Model")
    edge_candidate_header = headers.get("X-Edge-Quality-Candidate")
    edge_band_ratio_header = headers.get("X-Edge-Band-Ratio")
    edge_band_mid_ratio_header = headers.get("X-Edge-Band-Mid-Ratio")
    edge_band_low_ratio_header = headers.get("X-Edge-Band-Low-Ratio")

    return {
        "fallback_used": parse_bool_header(fallback_header),
        "actual_model": model_header,
        "edge_quality_candidate": parse_bool_header(edge_candidate_header),
        "edge_band_ratio": parse_float_header(edge_band_ratio_header),
        "edge_band_mid_ratio": parse_float_header(edge_band_mid_ratio_header),
        "edge_band_low_ratio": parse_float_header(edge_band_low_ratio_header),
        "raw_headers": {
            "X-RemoveBg-Fallback-Used": fallback_header,
            "X-RemoveBg-Model": model_header,
            "X-Edge-Quality-Candidate": edge_candidate_header,
            "X-Edge-Band-Ratio": edge_band_ratio_header,
            "X-Edge-Band-Mid-Ratio": edge_band_mid_ratio_header,
            "X-Edge-Band-Low-Ratio": edge_band_low_ratio_header,
        },
    }


def try_parse_json_bytes(raw: bytes) -> dict[str, Any] | None:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def call_remove_bg_api(
    session: requests.Session,
    file_path: Path,
    params: dict[str, Any],
) -> tuple[bytes, float, int, str, dict[str, Any]]:
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, get_mime_type(file_path))}
        started_at = time.perf_counter()
        response = session.post(API_URL, params=params, files=files, timeout=TIMEOUT_SECONDS)
        elapsed_sec = time.perf_counter() - started_at
    response_meta = extract_response_meta(response.headers)
    return response.content, elapsed_sec, response.status_code, response.headers.get("Content-Type", ""), response_meta


def resize_like_service(original: Image.Image, max_side: int) -> Image.Image:
    img = original.convert("RGBA")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def get_original(original_path: Path) -> Image.Image:
    cache_key = str(original_path)
    cached = _original_cache.get(cache_key)
    if cached is not None:
        return cached.copy()
    original = Image.open(original_path).convert("RGBA")
    _original_cache[cache_key] = original.copy()
    return original


def get_resized_original(original_path: Path, max_side: int) -> Image.Image:
    cache_key = (str(original_path), max_side)
    cached = _resized_original_cache.get(cache_key)
    if cached is not None:
        return cached.copy()
    original = get_original(original_path)
    resized_original = resize_like_service(original, max_side)
    _resized_original_cache[cache_key] = resized_original.copy()
    return resized_original


def get_compare_image(original_path: Path, max_side: int, output_size: tuple[int, int], compare_mode: str) -> Image.Image:
    if compare_mode == "original":
        compare_base = get_original(original_path)
    else:
        compare_base = get_resized_original(original_path, max_side)

    if compare_base.size != output_size:
        compare_base = compare_base.resize(output_size, Image.LANCZOS)
    return compare_base


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb)
    r = linear[:, 0]
    g = linear[:, 1]
    b = linear[:, 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    epsilon = 216 / 24389
    kappa = 24389 / 27

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16) / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)
    return np.stack([l, a, b2], axis=1)


def delta_e76(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    lab_a = rgb_to_lab(rgb_a)
    lab_b = rgb_to_lab(rgb_b)
    return np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=1))


def composite_rgba_on_bg(rgba: np.ndarray, bg_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
    bg = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    composite = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(composite, 0.0, 255.0).astype(np.float32)


def summarize_visual_fade_metrics(
    compare_rgba: np.ndarray,
    output_rgba: np.ndarray,
    mask: np.ndarray,
    bg_rgb: tuple[int, int, int],
) -> dict[str, Any] | None:
    count = int(mask.sum())
    if count == 0:
        return None

    compare_composite = composite_rgba_on_bg(compare_rgba, bg_rgb)
    output_composite = composite_rgba_on_bg(output_rgba, bg_rgb)

    orig_pixels = compare_composite[mask].astype(np.float32)
    out_pixels = output_composite[mask].astype(np.float32)

    orig_lightness = orig_pixels.mean(axis=1)
    out_lightness = out_pixels.mean(axis=1)
    lightness_shift = out_lightness - orig_lightness
    delta_e = delta_e76(orig_pixels, out_pixels)

    return {
        "pixels": count,
        "bg_rgb": list(bg_rgb),
        "mean_lightness_shift": round(float(lightness_shift.mean()), 4),
        "max_lightness_shift": round(float(lightness_shift.max()), 4),
        "fade_ratio": round(float(np.mean(lightness_shift >= VISUAL_FADE_LIGHTNESS_SHIFT_THRESHOLD)), 6),
        "mean_delta_e": round(float(delta_e.mean()), 4),
        "p95_delta_e": round(float(np.percentile(delta_e, 95)), 4),
    }


def summarize_mask_metrics(orig_rgb: np.ndarray, out_rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    count = int(mask.sum())
    if count == 0:
        return None

    orig_pixels = orig_rgb[mask].astype(np.float32)
    out_pixels = out_rgb[mask].astype(np.float32)
    diff = out_pixels - orig_pixels
    abs_diff = np.abs(diff)
    delta_e = delta_e76(orig_pixels, out_pixels)

    lightness_shift = float(np.mean(out_pixels.mean(axis=1) - orig_pixels.mean(axis=1)))
    washout_ratio = float(np.mean((out_pixels.mean(axis=1) - orig_pixels.mean(axis=1)) >= WASHOUT_DELTA_THRESHOLD))

    return {
        "pixels": count,
        "mean_abs_rgb_diff": round(float(abs_diff.mean()), 4),
        "max_abs_rgb_diff": round(float(abs_diff.max()), 4),
        "mean_channel_diff": {
            "r": round(float(diff[:, 0].mean()), 4),
            "g": round(float(diff[:, 1].mean()), 4),
            "b": round(float(diff[:, 2].mean()), 4),
        },
        "mean_delta_e": round(float(delta_e.mean()), 4),
        "p95_delta_e": round(float(np.percentile(delta_e, 95)), 4),
        "mean_lightness_shift": round(lightness_shift, 4),
        "washout_ratio": round(washout_ratio, 6),
    }


def compute_bbox_metrics(mask: np.ndarray) -> dict[str, Any]:
    h, w = mask.shape
    fg_pixels = int(mask.sum())
    total = int(h * w)
    if fg_pixels == 0:
        return {
            "bbox": None,
            "bbox_width_ratio": 0.0,
            "bbox_height_ratio": 0.0,
            "left_half_fg_ratio": 0.0,
            "right_half_fg_ratio": 0.0,
            "top_half_fg_ratio": 0.0,
            "bottom_half_fg_ratio": 0.0,
            "left_right_relative_drop": 0.0,
            "top_bottom_relative_drop": 0.0,
        }

    ys, xs = np.where(mask)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    mid_x = w // 2
    mid_y = h // 2
    left_pixels = int(mask[:, :mid_x].sum())
    right_pixels = int(mask[:, mid_x:].sum())
    top_pixels = int(mask[:mid_y, :].sum())
    bottom_pixels = int(mask[mid_y:, :].sum())

    left_ratio = left_pixels / total if total else 0.0
    right_ratio = right_pixels / total if total else 0.0
    top_ratio = top_pixels / total if total else 0.0
    bottom_ratio = bottom_pixels / total if total else 0.0

    return {
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "bbox_width_ratio": round((x1 - x0 + 1) / w, 6) if w else 0.0,
        "bbox_height_ratio": round((y1 - y0 + 1) / h, 6) if h else 0.0,
        "left_half_fg_ratio": round(left_ratio, 6),
        "right_half_fg_ratio": round(right_ratio, 6),
        "top_half_fg_ratio": round(top_ratio, 6),
        "bottom_half_fg_ratio": round(bottom_ratio, 6),
        "left_right_relative_drop": round(abs(left_ratio - right_ratio) / max(left_ratio + right_ratio, 1e-6), 6),
        "top_bottom_relative_drop": round(abs(top_ratio - bottom_ratio) / max(top_ratio + bottom_ratio, 1e-6), 6),
    }


def compute_balance_flags(coverage: dict[str, Any]) -> dict[str, bool]:
    left = coverage["left_half_fg_ratio"]
    right = coverage["right_half_fg_ratio"]
    top = coverage["top_half_fg_ratio"]
    bottom = coverage["bottom_half_fg_ratio"]

    left_right_drop = abs(left - right)
    top_bottom_drop = abs(top - bottom)

    return {
        "possible_left_right_imbalance": left_right_drop >= 0.08,
        "possible_top_bottom_imbalance": top_bottom_drop >= 0.12,
        "possible_left_right_relative_imbalance": coverage["left_right_relative_drop"] >= 0.35,
        "possible_top_bottom_relative_imbalance": coverage["top_bottom_relative_drop"] >= 0.35,
        "possible_narrow_mask": coverage["bbox_width_ratio"] <= 0.2,
        "possible_short_mask": coverage["bbox_height_ratio"] <= 0.2,
    }


def draw_bbox_overlay(image: Image.Image, bbox: dict[str, int] | None, color: tuple[int, int, int, int], width: int = 3) -> None:
    if not bbox:
        return
    draw = ImageDraw.Draw(image)
    for offset in range(width):
        draw.rectangle(
            [bbox["x0"] - offset, bbox["y0"] - offset, bbox["x1"] + offset, bbox["y1"] + offset],
            outline=color,
        )


def build_preview_image(compare_img: Image.Image, output: Image.Image, alpha: np.ndarray, bbox: dict[str, int] | None) -> Image.Image:
    compare_rgba = compare_img.convert("RGBA")
    output_rgba = output.convert("RGBA")

    white_bg = Image.new("RGBA", output_rgba.size, (255, 255, 255, 255))
    checker = Image.new("RGBA", output_rgba.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(checker)
    step = 24
    for y in range(0, checker.height, step):
        for x in range(0, checker.width, step):
            if ((x // step) + (y // step)) % 2 == 0:
                draw.rectangle([x, y, x + step - 1, y + step - 1], fill=(230, 230, 230, 255))

    composed = Image.alpha_composite(checker, output_rgba)
    alpha_img = Image.fromarray(alpha.astype(np.uint8), mode="L")
    alpha_rgb = ImageOps.colorize(alpha_img, black="black", white="white").convert("RGBA")

    # 在 compare 與 alpha panel 上畫 bbox，人工檢查 coverage 更直觀
    draw_bbox_overlay(compare_rgba, bbox, (255, 0, 0, 255), width=2)
    draw_bbox_overlay(alpha_rgb, bbox, (255, 0, 0, 255), width=2)

    panels = [compare_rgba, composed, Image.alpha_composite(white_bg, output_rgba), alpha_rgb]
    labels = ["compare_bbox", "output_checker", "output_white", "alpha_bbox"]

    width, height = compare_rgba.size
    header_h = 28
    canvas = Image.new("RGBA", (width * 2, (height + header_h) * 2), (250, 250, 250, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, panel in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x = col * width
        y = row * (height + header_h)
        draw.rectangle([x, y, x + width - 1, y + header_h - 1], fill=(240, 240, 240, 255))
        draw.text((x + 8, y + 6), labels[idx], fill=(20, 20, 20, 255))
        canvas.alpha_composite(panel, (x, y + header_h))

    return canvas


def analyze_output(
    original_path: Path,
    output_bytes: bytes,
    max_side: int,
    compare_mode: str,
) -> tuple[dict[str, Any], Image.Image, Image.Image, np.ndarray]:
    output = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    original_img = get_original(original_path)
    compare_img = get_compare_image(original_path, max_side=max_side, output_size=output.size, compare_mode=compare_mode)

    compare_np = np.array(compare_img, dtype=np.uint8)
    out_np = np.array(output, dtype=np.uint8)

    alpha = out_np[:, :, 3]
    fg_mask = alpha >= ALPHA_THRESHOLD
    core_mask = alpha >= CORE_ALPHA_THRESHOLD
    edge_mask = (alpha >= EDGE_ALPHA_MIN) & (alpha <= EDGE_ALPHA_MAX)

    total_pixels = int(alpha.shape[0] * alpha.shape[1])
    fg_pixels = int(fg_mask.sum())
    fg_ratio = fg_pixels / total_pixels if total_pixels else 0.0

    alpha_flat = alpha.flatten()
    mid_alpha_pixels = int(np.sum((alpha_flat >= EDGE_ALPHA_MIN) & (alpha_flat <= EDGE_ALPHA_MAX)))
    mid_alpha_ratio = mid_alpha_pixels / total_pixels if total_pixels else 0.0

    compare_rgb = compare_np[:, :, :3]
    out_rgb = out_np[:, :, :3]

    overall_metrics = summarize_mask_metrics(compare_rgb, out_rgb, fg_mask)
    core_metrics = summarize_mask_metrics(compare_rgb, out_rgb, core_mask)
    edge_metrics = summarize_mask_metrics(compare_rgb, out_rgb, edge_mask)
    visual_fade_metrics = summarize_visual_fade_metrics(
        compare_np,
        out_np,
        fg_mask,
        bg_rgb=VISUAL_FADE_BG_RGB,
    )
    alpha_fade_metrics = summarize_visual_fade_metrics(
        compare_np,
        out_np,
        alpha >= VISUAL_FADE_CORE_ALPHA_MIN,
        bg_rgb=VISUAL_FADE_BG_RGB,
    )
    coverage = compute_bbox_metrics(fg_mask)
    balance_flags = compute_balance_flags(coverage)

    orig_w, orig_h = original_img.size
    out_w, out_h = output.size
    width_scale_ratio = out_w / orig_w if orig_w else 0.0
    height_scale_ratio = out_h / orig_h if orig_h else 0.0
    output_scale_ratio = max(width_scale_ratio, height_scale_ratio)

    analysis = {
        "compare_mode": compare_mode,
        "width": int(output.width),
        "height": int(output.height),
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "output_width": int(out_w),
        "output_height": int(out_h),
        "width_scale_ratio": round(width_scale_ratio, 6),
        "height_scale_ratio": round(height_scale_ratio, 6),
        "output_scale_ratio": round(output_scale_ratio, 6),
        "foreground_pixels": fg_pixels,
        "foreground_ratio": round(fg_ratio, 6),
        "avg_alpha": round(float(alpha.mean()), 4),
        "mid_alpha_ratio": round(float(mid_alpha_ratio), 6),
        "overall": overall_metrics,
        "core": core_metrics,
        "edge": edge_metrics,
        "visual_fade": visual_fade_metrics,
        "alpha_fade": alpha_fade_metrics,
        "coverage": coverage,
        "flags": {
            "possible_empty_mask": fg_ratio < 0.01,
            "possible_tiny_mask": fg_ratio < 0.05,
            "possible_over_soft_edge": mid_alpha_ratio > 0.12,
            "possible_core_color_shift": (core_metrics or {}).get("mean_delta_e", 0) >= 6.0,
            "possible_edge_washout": (edge_metrics or {}).get("washout_ratio", 0) >= 0.2,
            "possible_visual_fade": (visual_fade_metrics or {}).get("mean_lightness_shift", 0) >= VISUAL_FADE_LIGHTNESS_SHIFT_THRESHOLD,
            "possible_alpha_fade": (alpha_fade_metrics or {}).get("fade_ratio", 0) >= VISUAL_FADE_RATIO_THRESHOLD,
            **balance_flags,
        },
    }
    return analysis, compare_img, output, alpha


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def has_any_problem_flag(flags: dict[str, Any]) -> bool:
    return any(bool(v) for v in flags.values())


def count_problem_flags(flags: dict[str, Any]) -> int:
    return sum(1 for v in flags.values() if bool(v))


def apply_response_meta_flags(item: dict[str, Any]) -> None:
    if not item.get("ok"):
        return
    analysis = item.get("analysis") or {}
    flags = analysis.get("flags")
    if not isinstance(flags, dict):
        return
    response_meta = item.get("response_meta") or {}
    flags["missing_fallback_header"] = response_meta.get("fallback_used") is None
    flags["missing_actual_model_header"] = not response_meta.get("actual_model")


def build_fallback_readable_summary(summary: dict[str, Any], subset_name: str, config_name: str) -> dict[str, Any]:
    return {
        "subset": subset_name,
        "config": config_name,
        "header_available_count": summary.get("fallback_header_available_count"),
        "fallback_used_count": summary.get("fallback_used_count"),
        "fallback_used_rate": summary.get("fallback_used_rate"),
        "missing_fallback_header_count": summary.get("missing_fallback_header_count"),
        "missing_actual_model_header_count": summary.get("missing_actual_model_header_count"),
    }


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    coverage = analysis["coverage"]
    response_meta = item.get("response_meta") or {}
    return {
        "file": item["file"],
        "elapsed_sec": item["elapsed_sec"],
        "foreground_ratio": analysis["foreground_ratio"],
        "core_mean_delta_e": (analysis.get("core") or {}).get("mean_delta_e"),
        "edge_washout_ratio": (analysis.get("edge") or {}).get("washout_ratio"),
        "visual_fade_lightness_shift": (analysis.get("visual_fade") or {}).get("mean_lightness_shift"),
        "visual_fade_ratio": (analysis.get("visual_fade") or {}).get("fade_ratio"),
        "alpha_fade_lightness_shift": (analysis.get("alpha_fade") or {}).get("mean_lightness_shift"),
        "alpha_fade_ratio": (analysis.get("alpha_fade") or {}).get("fade_ratio"),
        "original_width": analysis["original_width"],
        "original_height": analysis["original_height"],
        "output_width": analysis["output_width"],
        "output_height": analysis["output_height"],
        "output_scale_ratio": analysis["output_scale_ratio"],
        "bbox_width_ratio": coverage["bbox_width_ratio"],
        "bbox_height_ratio": coverage["bbox_height_ratio"],
        "left_half_fg_ratio": coverage["left_half_fg_ratio"],
        "right_half_fg_ratio": coverage["right_half_fg_ratio"],
        "left_right_relative_drop": coverage["left_right_relative_drop"],
        "top_half_fg_ratio": coverage["top_half_fg_ratio"],
        "bottom_half_fg_ratio": coverage["bottom_half_fg_ratio"],
        "top_bottom_relative_drop": coverage["top_bottom_relative_drop"],
        "mid_alpha_ratio": analysis["mid_alpha_ratio"],
        "fallback_used": response_meta.get("fallback_used"),
        "actual_model": response_meta.get("actual_model"),
        "missing_fallback_header": analysis["flags"].get("missing_fallback_header"),
        "missing_actual_model_header": analysis["flags"].get("missing_actual_model_header"),
        "problem_flag_count": count_problem_flags(analysis["flags"]),
        "flags": analysis["flags"],
        "saved_output_png": item.get("saved_output_png"),
        "saved_preview_png": item.get("saved_preview_png"),
        "edge_quality_candidate": response_meta.get("edge_quality_candidate"),
        "edge_band_ratio": response_meta.get("edge_band_ratio"),
        "edge_band_mid_ratio": response_meta.get("edge_band_mid_ratio"),
        "edge_band_low_ratio": response_meta.get("edge_band_low_ratio"),
    }


def classify_review_case(item: dict[str, Any], elapsed_threshold: float) -> set[str]:
    analysis = item["analysis"]
    flags = analysis["flags"]
    response_meta = item.get("response_meta") or {}

    kinds: set[str] = set()
    if response_meta.get("fallback_used") is True:
        kinds.add("fallback")
    if (
        flags.get("possible_core_color_shift")
        or flags.get("possible_edge_washout")
        or flags.get("possible_visual_fade")
        or flags.get("possible_alpha_fade")
    ):
        kinds.add("color_shift")
    if (
        flags.get("possible_left_right_imbalance")
        or flags.get("possible_top_bottom_imbalance")
        or flags.get("possible_left_right_relative_imbalance")
        or flags.get("possible_top_bottom_relative_imbalance")
        or flags.get("possible_narrow_mask")
        or flags.get("possible_short_mask")
        or flags.get("possible_empty_mask")
        or flags.get("possible_tiny_mask")
    ):
        kinds.add("imbalance")
    if item["elapsed_sec"] >= elapsed_threshold:
        kinds.add("slow")
    if flags.get("missing_fallback_header") or flags.get("missing_actual_model_header"):
        kinds.add("missing_header")
    if not kinds:
        kinds.add("other")
    return kinds


def pick_diverse_review_cases(success_items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not success_items or top_k <= 0:
        return []

    elapsed_threshold = float(np.percentile([x["elapsed_sec"] for x in success_items], 90))
    review_candidates = [
        x for x in success_items
        if (x.get("response_meta") or {}).get("fallback_used") is True
        or has_any_problem_flag(x["analysis"]["flags"])
        or x["elapsed_sec"] >= elapsed_threshold
    ]

    def rank_key(x: dict[str, Any]) -> tuple[Any, ...]:
        return (
            1 if (x.get("response_meta") or {}).get("fallback_used") is True else 0,
            count_problem_flags(x["analysis"]["flags"]),
            (x["analysis"].get("visual_fade") or {}).get("mean_lightness_shift", 0),
            (x["analysis"].get("visual_fade") or {}).get("fade_ratio", 0),
            (x["analysis"].get("core") or {}).get("mean_delta_e", 0),
            (x["analysis"].get("edge") or {}).get("washout_ratio", 0),
            x["elapsed_sec"],
        )

    sorted_candidates = sorted(review_candidates, key=rank_key, reverse=True)

    # 類型配額：至少讓 fallback / color_shift / imbalance / slow 各有機會進榜
    quotas = ["fallback", "color_shift", "imbalance", "slow"]
    chosen: list[dict[str, Any]] = []
    chosen_files: set[str] = set()

    for quota_name in quotas:
        for item in sorted_candidates:
            if item["file"] in chosen_files:
                continue
            kinds = classify_review_case(item, elapsed_threshold)
            if quota_name in kinds:
                chosen.append(item)
                chosen_files.add(item["file"])
                break

    for item in sorted_candidates:
        if len(chosen) >= top_k:
            break
        if item["file"] in chosen_files:
            continue
        chosen.append(item)
        chosen_files.add(item["file"])

    return chosen[:top_k]


def summarize_items(items: list[dict[str, Any]], top_k: int, subset_name: str, config_name: str) -> dict[str, Any]:
    success_items = [x for x in items if x["ok"]]
    rejected_items = [x for x in items if not x["ok"] and x.get("rejected")]
    failed_items = [x for x in items if not x["ok"] and not x.get("rejected")]

    elapsed_list = [x["elapsed_sec"] for x in items]

    p95_elapsed_sec = None
    if elapsed_list:
        p95_elapsed_sec = float(np.percentile(elapsed_list, 95))

    fg_ratio_list = [x["analysis"]["foreground_ratio"] for x in success_items if x.get("analysis")]
    core_de_list = [
        x["analysis"]["core"]["mean_delta_e"]
        for x in success_items
        if x.get("analysis", {}).get("core")
    ]
    edge_washout_list = [
        x["analysis"]["edge"]["washout_ratio"]
        for x in success_items
        if x.get("analysis", {}).get("edge")
    ]
    visual_fade_shift_list = [
        x["analysis"]["visual_fade"]["mean_lightness_shift"]
        for x in success_items
        if x.get("analysis", {}).get("visual_fade")
    ]
    visual_fade_ratio_list = [
        x["analysis"]["visual_fade"]["fade_ratio"]
        for x in success_items
        if x.get("analysis", {}).get("visual_fade")
    ]

    alpha_fade_shift_list = [
        x["analysis"]["alpha_fade"]["mean_lightness_shift"]
        for x in success_items
        if x.get("analysis", {}).get("alpha_fade")
    ]
    alpha_fade_ratio_list = [
        x["analysis"]["alpha_fade"]["fade_ratio"]
        for x in success_items
        if x.get("analysis", {}).get("alpha_fade")
    ]

    worst_core = sorted(
        [x for x in success_items if x.get("analysis", {}).get("core")],
        key=lambda x: x["analysis"]["core"]["mean_delta_e"],
        reverse=True,
    )[:top_k]
    worst_edge = sorted(
        [x for x in success_items if x.get("analysis", {}).get("edge")],
        key=lambda x: x["analysis"]["edge"]["washout_ratio"],
        reverse=True,
    )[:top_k]
    worst_visual_fade = sorted(
        [x for x in success_items if x.get("analysis", {}).get("visual_fade")],
        key=lambda x: (
            x["analysis"]["visual_fade"]["mean_lightness_shift"],
            x["analysis"]["visual_fade"]["fade_ratio"],
        ),
        reverse=True,
    )[:top_k]
    worst_alpha_fade = sorted(
        [x for x in success_items if x.get("analysis", {}).get("alpha_fade")],
        key=lambda x: (
            x["analysis"]["alpha_fade"]["fade_ratio"],
            x["analysis"]["alpha_fade"]["mean_lightness_shift"],
        ),
        reverse=True,
    )[:top_k]
    worst_imbalance = sorted(
        success_items,
        key=lambda x: x["analysis"]["coverage"]["left_right_relative_drop"],
        reverse=True,
    )[:top_k]
    worst_top_bottom = sorted(
        success_items,
        key=lambda x: x["analysis"]["coverage"]["top_bottom_relative_drop"],
        reverse=True,
    )[:top_k]
    worst_elapsed = sorted(
        success_items,
        key=lambda x: x["elapsed_sec"],
        reverse=True,
    )[:top_k]
    edge_candidate_known_items = [
        x for x in success_items
        if (x.get("response_meta") or {}).get("edge_quality_candidate") is not None
    ]
    edge_candidate_count = sum(
        1 for x in edge_candidate_known_items
        if (x.get("response_meta") or {}).get("edge_quality_candidate") is True
    )

    fallback_known_items = [x for x in success_items if (x.get("response_meta") or {}).get("fallback_used") is not None]
    fallback_used_count = sum(1 for x in fallback_known_items if (x.get("response_meta") or {}).get("fallback_used") is True)
    flagged_count = sum(1 for x in success_items if has_any_problem_flag(x["analysis"]["flags"]))
    missing_fallback_header_count = sum(
        1 for x in success_items if x["analysis"]["flags"].get("missing_fallback_header")
    )
    missing_actual_model_header_count = sum(
        1 for x in success_items if x["analysis"]["flags"].get("missing_actual_model_header")
    )

    recommended_review = pick_diverse_review_cases(success_items, top_k=top_k)

    summary = {
        "total": len(items),
        "success": len(success_items),
        "failed": len(failed_items),
        "flagged": flagged_count,
        "avg_elapsed_sec": round(safe_ratio(sum(elapsed_list), len(elapsed_list)), 4),
        "p95_elapsed_sec": round(p95_elapsed_sec, 4) if p95_elapsed_sec is not None else None,
        "max_elapsed_sec": round(max(elapsed_list), 4) if elapsed_list else 0.0,
        "min_elapsed_sec": round(min(elapsed_list), 4) if elapsed_list else 0.0,
        "avg_foreground_ratio": round(safe_ratio(sum(fg_ratio_list), len(fg_ratio_list)), 6) if fg_ratio_list else None,
        "avg_core_mean_delta_e": round(safe_ratio(sum(core_de_list), len(core_de_list)), 4) if core_de_list else None,
        "avg_edge_washout_ratio": round(safe_ratio(sum(edge_washout_list), len(edge_washout_list)), 6) if edge_washout_list else None,
        "avg_visual_fade_lightness_shift": round(safe_ratio(sum(visual_fade_shift_list), len(visual_fade_shift_list)), 4) if visual_fade_shift_list else None,
        "avg_visual_fade_ratio": round(safe_ratio(sum(visual_fade_ratio_list), len(visual_fade_ratio_list)), 6) if visual_fade_ratio_list else None,
        "avg_alpha_fade_lightness_shift": round(safe_ratio(sum(alpha_fade_shift_list), len(alpha_fade_shift_list)), 4) if alpha_fade_shift_list else None,
        "avg_alpha_fade_ratio": round(safe_ratio(sum(alpha_fade_ratio_list), len(alpha_fade_ratio_list)), 6) if alpha_fade_ratio_list else None,
        "fallback_header_available_count": len(fallback_known_items),
        "fallback_used_count": fallback_used_count,
        "fallback_used_rate": round(safe_ratio(fallback_used_count, len(fallback_known_items)), 6) if fallback_known_items else None,
        "missing_fallback_header_count": missing_fallback_header_count,
        "missing_actual_model_header_count": missing_actual_model_header_count,
        "worst_core_color_shift": [compact_item(x) for x in worst_core],
        "worst_edge_washout": [compact_item(x) for x in worst_edge],
        "worst_visual_fade": [compact_item(x) for x in worst_visual_fade],
        "worst_alpha_fade": [compact_item(x) for x in worst_alpha_fade],
        "worst_left_right_imbalance": [compact_item(x) for x in worst_imbalance],
        "worst_top_bottom_imbalance": [compact_item(x) for x in worst_top_bottom],
        "worst_elapsed_cases": [compact_item(x) for x in worst_elapsed],
        "recommended_review_cases": [compact_item(x) for x in recommended_review],
        "rejected": len(rejected_items),
        "reject_rate": round(safe_ratio(len(rejected_items), len(items)), 6),
        "reject_reasons": {
            reason: sum(1 for x in rejected_items if x.get("reject_reason") == reason)
            for reason in sorted({x.get("reject_reason") for x in rejected_items if x.get("reject_reason")})
        },
        "edge_candidate_header_available_count": len(edge_candidate_known_items),
        "edge_quality_candidate_count": edge_candidate_count,
        "edge_quality_candidate_rate": round(
            safe_ratio(edge_candidate_count, len(edge_candidate_known_items)), 6
        ) if edge_candidate_known_items else None,
    }
    summary["fallback_readable_summary"] = build_fallback_readable_summary(summary, subset_name, config_name)
    return summary


def save_artifacts(
    save_root: Path,
    subset_name: str,
    config_name: str,
    case: TestCase,
    output_bytes: bytes,
    preview: Image.Image,
) -> tuple[str, str]:
    subset_folder = "root" if subset_name == "." else subset_name
    subset_dir = save_root / subset_folder / config_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    stem = case.file_path.stem
    output_path = subset_dir / f"{stem}__output.png"
    preview_path = subset_dir / f"{stem}__preview.png"
    output_path.write_bytes(output_bytes)
    preview.save(preview_path)
    return str(output_path), str(preview_path)


def should_save_artifacts(item: dict[str, Any], save_only_flagged: bool) -> bool:
    if not save_only_flagged:
        return True
    if not item.get("ok"):
        return False
    analysis = item.get("analysis") or {}
    flags = analysis.get("flags") or {}
    response_meta = item.get("response_meta") or {}
    if response_meta.get("fallback_used") is True:
        return True
    tracked_flags = {
        "possible_empty_mask",
        "possible_tiny_mask",
        "possible_over_soft_edge",
        "possible_core_color_shift",
        "possible_edge_washout",
        "possible_visual_fade",
        "possible_alpha_fade",
        "possible_left_right_imbalance",
        "possible_top_bottom_imbalance",
        "possible_left_right_relative_imbalance",
        "possible_top_bottom_relative_imbalance",
        "possible_narrow_mask",
        "possible_short_mask",
        "missing_fallback_header",
        "missing_actual_model_header",
    }
    return any(bool(flags.get(name)) for name in tracked_flags)


def run_one_config(
    session: requests.Session,
    subset_name: str,
    cases: list[TestCase],
    config: dict[str, Any],
    save_output_dir: Path | None,
    save_only_flagged: bool,
    top_k: int,
    compare_mode: str,
) -> dict[str, Any]:
    config_name = config["name"]
    params = config["params"]

    print("\n" + "=" * 80)
    print(f"Subset: {subset_name}")
    print(f"Config: {config_name}")
    print(f"Compare: {compare_mode}")
    print(f"Params: {json.dumps(params, ensure_ascii=False)}")
    print("=" * 80)

    items: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] 測試中：{case.file_path.name}")
        try:
            output_bytes, elapsed_sec, status_code, content_type, response_meta = call_remove_bg_api(session, case.file_path, params)
            if status_code != 200:
                error_json = try_parse_json_bytes(output_bytes)
                reject_code = None
                reject_reason = None
                reject_metrics = None

                if status_code == 422 and isinstance(error_json, dict):
                    reject_code = error_json.get("code")
                    reject_reason = error_json.get("reason")
                    reject_metrics = error_json.get("metrics")

                item = {
                    "file": case.file_path.name,
                    "file_path": str(case.file_path),
                    "ok": False,
                    "status_code": status_code,
                    "content_type": content_type,
                    "elapsed_sec": round(elapsed_sec, 4),
                    "response_meta": response_meta,
                    "error": output_bytes.decode("utf-8", errors="ignore"),
                    "error_json": error_json,
                    "rejected": status_code == 422 and reject_code == "LOW_CONFIDENCE_MASK",
                    "reject_code": reject_code,
                    "reject_reason": reject_reason,
                    "reject_metrics": reject_metrics,
                }

                if item["rejected"]:
                    print(
                        f"  [REJECT] status={status_code}, elapsed={elapsed_sec:.4f}s, "
                        f"reason={reject_reason}, metrics={reject_metrics}"
                    )
                else:
                    print(f"  [FAIL] status={status_code}, elapsed={elapsed_sec:.4f}s")
            else:
                analysis, compare_img, output_img, alpha = analyze_output(
                    original_path=case.file_path,
                    output_bytes=output_bytes,
                    max_side=params["max_side"],
                    compare_mode=compare_mode,
                )
                preview = build_preview_image(compare_img, output_img, alpha, analysis["coverage"]["bbox"])
                item = {
                    "file": case.file_path.name,
                    "file_path": str(case.file_path),
                    "ok": True,
                    "status_code": status_code,
                    "content_type": content_type,
                    "elapsed_sec": round(elapsed_sec, 4),
                    "params": params,
                    "analysis": analysis,
                    "response_meta": response_meta,
                    "output_size_bytes": len(output_bytes),
                }
                apply_response_meta_flags(item)
                if save_output_dir is not None and should_save_artifacts(item, save_only_flagged):
                    saved_output_png, saved_preview_png = save_artifacts(
                        save_output_dir, subset_name, config_name, case, output_bytes, preview
                    )
                    item["saved_output_png"] = saved_output_png
                    item["saved_preview_png"] = saved_preview_png

                coverage = analysis["coverage"]
                core_de = (analysis.get("core") or {}).get("mean_delta_e")
                problem_flag_count = count_problem_flags(analysis["flags"])
                visual_fade_shift = (analysis.get("visual_fade") or {}).get("mean_lightness_shift")
                visual_fade_ratio = (analysis.get("visual_fade") or {}).get("fade_ratio")
                alpha_fade_shift = (analysis.get("alpha_fade") or {}).get("mean_lightness_shift")
                alpha_fade_ratio = (analysis.get("alpha_fade") or {}).get("fade_ratio")
                print(
                    f"  [PASS] elapsed={elapsed_sec:.4f}s, fg_ratio={analysis['foreground_ratio']}, "
                    f"core_delta_e={core_de}, visual_fade_shift={visual_fade_shift}, "
                    f"visual_fade_ratio={visual_fade_ratio}, alpha_fade_shift={alpha_fade_shift}, "
                    f"alpha_fade_ratio={alpha_fade_ratio}, left_fg={coverage['left_half_fg_ratio']}, "
                    f"right_fg={coverage['right_half_fg_ratio']}, lr_rel={coverage['left_right_relative_drop']}, "
                    f"mid_alpha={analysis['mid_alpha_ratio']}, problem_flags={problem_flag_count}, "
                    f"fallback={response_meta.get('fallback_used')}, actual_model={response_meta.get('actual_model')}, "
                    f"edge_candidate={response_meta.get('edge_quality_candidate')}, "
                    f"edge_band_ratio={response_meta.get('edge_band_ratio')}, "
                    f"edge_band_mid_ratio={response_meta.get('edge_band_mid_ratio')}, "
                    f"edge_band_low_ratio={response_meta.get('edge_band_low_ratio')}, "
                )
        except requests.RequestException as e:
            item = {
                "file": case.file_path.name,
                "file_path": str(case.file_path),
                "ok": False,
                "status_code": None,
                "elapsed_sec": 0.0,
                "error": f"request_error: {e}",
            }
            print(f"  [FAIL] request_error: {e}")
        except Exception as e:
            item = {
                "file": case.file_path.name,
                "file_path": str(case.file_path),
                "ok": False,
                "status_code": None,
                "elapsed_sec": 0.0,
                "error": f"unexpected_error: {e}",
            }
            print(f"  [FAIL] unexpected_error: {e}")

        items.append(item)

    summary = summarize_items(items, top_k=top_k, subset_name=subset_name, config_name=config_name)
    print("\n--- Summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return {
        "config_name": config_name,
        "params": params,
        "summary": summary,
        "items": items,
    }


def build_subsets(dataset_dir: Path, subset_filters: set[str] | None) -> list[tuple[str, Path]]:
    all_subsets = discover_subsets(dataset_dir)

    if not subset_filters:
        return all_subsets

    normalized_filters = {s.replace("\\", "/").strip() for s in subset_filters}
    return [(name, path) for name, path in all_subsets if name in normalized_filters]


def build_config_overview(config_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in config_reports:
        summary = report.get("summary") or {}
        rows.append(
            {
                "subset": report.get("subset"),
                "config": report.get("config_name"),
                "success": summary.get("success"),
                "fallback_header_available_count": summary.get("fallback_header_available_count"),
                "fallback_used_count": summary.get("fallback_used_count"),
                "fallback_used_rate": summary.get("fallback_used_rate"),
                "missing_fallback_header_count": summary.get("missing_fallback_header_count"),
                "missing_actual_model_header_count": summary.get("missing_actual_model_header_count"),
                "avg_elapsed_sec": summary.get("avg_elapsed_sec"),
                "p95_elapsed_sec": summary.get("p95_elapsed_sec"),
                "max_elapsed_sec": summary.get("max_elapsed_sec"),
                "rejected": summary.get("rejected"),
                "reject_rate": summary.get("reject_rate"),
                "edge_quality_candidate_count": summary.get("edge_quality_candidate_count"),
                "edge_quality_candidate_rate": summary.get("edge_quality_candidate_rate"),
            }
        )
    return rows


def get_next_version(base_dir: Path, base_name: str) -> int:
    version = 1
    while True:
        report_candidate = base_dir / f"{base_name}_report_v{version}.json"
        output_candidate = base_dir / f"{base_name}_outputs_v{version}"
        if not report_candidate.exists() and not output_candidate.exists():
            return version
        version += 1


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_name = dataset_dir.name
    auto_version = get_next_version(Path.cwd(), dataset_name)

    if not dataset_dir.exists():
        print(f"[ERROR] 找不到測試集資料夾：{dataset_dir}")
        return 1

    subset_filters = set(args.subset) if args.subset else None
    file_filters = set(args.file) if args.file else None
    selected_configs = [CONFIG_MAP[name] for name in (args.config or CONFIG_MAP.keys())]
    subsets = build_subsets(dataset_dir, subset_filters)

    if args.save_output_dir:
        save_output_dir = Path(args.save_output_dir).resolve()
    else:
        save_output_dir = Path.cwd() / f"{dataset_name}_outputs_v{auto_version}"

    total_cases = 0
    subset_cases_map: dict[str, list[TestCase]] = {}
    for subset_name, subset_path in subsets:
        cases = iter_test_cases(subset_path, file_filters=file_filters)
        subset_cases_map[subset_name] = cases
        total_cases += len(cases)

    if total_cases == 0:
        print(f"[ERROR] 找不到可測試圖片：{dataset_dir}")
        return 1

    print("=" * 80)
    print("去背色差／coverage 問題測試")
    print(f"API_URL      : {API_URL}")
    print(f"DATASET_DIR  : {dataset_dir}")
    print(f"TOTAL_CASES  : {total_cases}")
    display_subset_names = [("root" if name == "." else name) for name, _ in subsets]
    print(f"SUBSETS      : {', '.join(display_subset_names)}")
    print(f"CONFIGS      : {', '.join(config['name'] for config in selected_configs)}")
    print(f"COMPARE_MODE : {args.compare_mode}")
    print(f"FILE_FILTERS : {', '.join(sorted(file_filters)) if file_filters else '(none)'}")
    print(f"SAVE_DIR     : {save_output_dir if save_output_dir else '(none)'}")
    print(f"SAVE_FLAGGED : {args.save_only_flagged}")
    print("=" * 80)

    script_started_at = time.perf_counter()
    config_reports = []

    with requests.Session() as session:
        for subset_name, _subset_path in subsets:
            print(f"\n===== Subset: {subset_name} =====")
            cases = subset_cases_map[subset_name]
            if not cases:
                print(f"[WARN] 空資料夾或沒有符合篩選條件的圖片：{subset_name}")
                continue
            for config in selected_configs:
                result = run_one_config(
                    session=session,
                    subset_name=subset_name,
                    cases=cases,
                    config=config,
                    save_output_dir=save_output_dir,
                    save_only_flagged=args.save_only_flagged,
                    top_k=args.top_k,
                    compare_mode=args.compare_mode,
                )
                result["subset"] = subset_name
                config_reports.append(result)

    total_elapsed_sec = time.perf_counter() - script_started_at
    report = {
        "api_url": API_URL,
        "dataset_dir": str(dataset_dir),
        "total_cases": total_cases,
        "total_configs": len(selected_configs),
        "script_elapsed_sec": round(total_elapsed_sec, 4),
        "filters": {
            "subset": sorted(subset_filters) if subset_filters else None,
            "config": [config["name"] for config in selected_configs],
            "file": sorted(file_filters) if file_filters else None,
        },
        "compare_mode": args.compare_mode,
        "save_output_dir": str(save_output_dir) if save_output_dir else None,
        "save_only_flagged": args.save_only_flagged,
        "fallback_used_rate_by_subset_config": build_config_overview(config_reports),
        "configs": config_reports,
    }

    if args.report_file == REPORT_FILE:
        report_path = Path.cwd() / f"{dataset_name}_report_v{auto_version}.json"
    else:
        report_path = Path(args.report_file).resolve()

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("全部測試完成")
    print("=" * 80)
    print(f"script_elapsed_sec: {total_elapsed_sec:.4f}")
    print(f"report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
