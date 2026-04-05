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
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("REMOVE_BG_API_URL", "http://127.0.0.1:7860/remove-bg")
DEFAULT_DATASET_DIR = Path(r"D:\DevData\remove_bg_testset")
REPORT_FILE = "test_remove_bg_benchmark_report.json"
TIMEOUT_SECONDS = 120

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif"}

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


@dataclass
class TestCase:
    file_path: Path


_resized_original_cache: dict[tuple[str, int], Image.Image] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="去背服務 benchmark 腳本")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help="測試集根目錄，預設為 D:\\DevData\\remove_bg_testset",
    )
    parser.add_argument(
        "--subset",
        choices=["apparel", "footwear", "flatlay"],
        action="append",
        help="只跑指定 subset，可重複傳入多次",
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
        help="只跑指定檔名，可重複傳入多次，例如 --file light_beige_shoes_01.jpg",
    )
    parser.add_argument(
        "--report-file",
        default=REPORT_FILE,
        help="輸出的 report 檔名",
    )
    return parser.parse_args()


def iter_test_cases(dataset_dir: Path, file_filters: set[str] | None = None) -> list[TestCase]:
    cases: list[TestCase] = []
    for file_path in sorted(dataset_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_filters and file_path.name not in file_filters:
            continue
        cases.append(TestCase(file_path=file_path))
    return cases


def get_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def call_remove_bg_api(session: requests.Session, file_path: Path, params: dict[str, Any]) -> tuple[bytes, float, int]:
    with file_path.open("rb") as f:
        files = {
            "file": (file_path.name, f, get_mime_type(file_path)),
        }
        started_at = time.perf_counter()
        response = session.post(
            API_URL,
            params=params,
            files=files,
            timeout=TIMEOUT_SECONDS,
        )
        elapsed_sec = time.perf_counter() - started_at

    return response.content, elapsed_sec, response.status_code


def resize_like_service(original: Image.Image, max_side: int) -> Image.Image:
    img = original.convert("RGBA")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def get_resized_original(original_path: Path, max_side: int) -> Image.Image:
    cache_key = (str(original_path), max_side)
    cached = _resized_original_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    original = Image.open(original_path).convert("RGBA")
    resized_original = resize_like_service(original, max_side)
    _resized_original_cache[cache_key] = resized_original.copy()
    return resized_original


def analyze_output(original_path: Path, output_bytes: bytes, max_side: int) -> dict[str, Any]:
    resized_original = get_resized_original(original_path, max_side)
    output = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    if resized_original.size != output.size:
        resized_original = resized_original.resize(output.size, Image.LANCZOS)

    orig_np = np.array(resized_original, dtype=np.int16)
    out_np = np.array(output, dtype=np.int16)

    alpha = out_np[:, :, 3]
    fg_mask = alpha >= ALPHA_THRESHOLD

    total_pixels = int(alpha.shape[0] * alpha.shape[1])
    fg_pixels = int(fg_mask.sum())
    fg_ratio = fg_pixels / total_pixels if total_pixels else 0.0

    alpha_flat = alpha.flatten()
    mid_alpha_pixels = int(np.sum((alpha_flat >= 10) & (alpha_flat <= 200)))
    mid_alpha_ratio = mid_alpha_pixels / total_pixels if total_pixels else 0.0

    if fg_pixels == 0:
        return {
            "width": output.width,
            "height": output.height,
            "foreground_pixels": 0,
            "foreground_ratio": round(fg_ratio, 6),
            "avg_alpha": round(float(alpha.mean()), 4),
            "mid_alpha_ratio": round(float(mid_alpha_ratio), 6),
        }

    return {
        "width": output.width,
        "height": output.height,
        "foreground_pixels": fg_pixels,
        "foreground_ratio": round(fg_ratio, 6),
        "avg_alpha": round(float(alpha.mean()), 4),
        "mid_alpha_ratio": round(float(mid_alpha_ratio), 6),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    success_items = [x for x in items if x["ok"]]
    failed_items = [x for x in items if not x["ok"]]

    elapsed_list = [x["elapsed_sec"] for x in items]
    fg_ratio_list = [
        x["analysis"]["foreground_ratio"]
        for x in success_items
        if x.get("analysis", {}).get("foreground_ratio") is not None
    ]

    return {
        "total": len(items),
        "success": len(success_items),
        "failed": len(failed_items),
        "avg_elapsed_sec": round(safe_ratio(sum(elapsed_list), len(elapsed_list)), 4),
        "max_elapsed_sec": round(max(elapsed_list), 4) if elapsed_list else 0.0,
        "min_elapsed_sec": round(min(elapsed_list), 4) if elapsed_list else 0.0,
        "avg_foreground_ratio": round(safe_ratio(sum(fg_ratio_list), len(fg_ratio_list)), 6) if fg_ratio_list else None,
    }


def run_one_config(
    session: requests.Session,
    subset_name: str,
    cases: list[TestCase],
    config: dict[str, Any],
) -> dict[str, Any]:
    config_name = config["name"]
    params = config["params"]

    print("\n" + "=" * 80)
    print(f"Subset: {subset_name}")
    print(f"Config: {config_name}")
    print(f"Params: {json.dumps(params, ensure_ascii=False)}")
    print("=" * 80)

    items: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] 測試中：{case.file_path.name}")

        try:
            output_bytes, elapsed_sec, status_code = call_remove_bg_api(session, case.file_path, params)

            if status_code != 200:
                item = {
                    "file": case.file_path.name,
                    "file_path": str(case.file_path),
                    "ok": False,
                    "status_code": status_code,
                    "elapsed_sec": round(elapsed_sec, 4),
                    "error": output_bytes.decode("utf-8", errors="ignore"),
                }
                print(f"  [FAIL] status={status_code}, elapsed={elapsed_sec:.4f}s")
            else:
                analysis = analyze_output(
                    original_path=case.file_path,
                    output_bytes=output_bytes,
                    max_side=params["max_side"],
                )
                item = {
                    "file": case.file_path.name,
                    "file_path": str(case.file_path),
                    "ok": True,
                    "status_code": status_code,
                    "elapsed_sec": round(elapsed_sec, 4),
                    "params": params,
                    "analysis": analysis,
                    "output_size_bytes": len(output_bytes),
                }
                print(
                    f"  [PASS] elapsed={elapsed_sec:.4f}s, "
                    f"fg_ratio={analysis['foreground_ratio']}, "
                    f"avg_alpha={analysis['avg_alpha']}, "
                    f"mid_alpha={analysis['mid_alpha_ratio']}"
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

    summary = summarize_items(items)

    print("\n--- Summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return {
        "config_name": config_name,
        "params": params,
        "summary": summary,
        "items": items,
    }


def build_subsets(dataset_dir: Path, subset_filters: set[str] | None) -> list[tuple[str, Path]]:
    all_subsets = [
        ("apparel", dataset_dir / "apparel"),
        ("footwear", dataset_dir / "footwear"),
        ("flatlay", dataset_dir / "flatlay"),
    ]
    if not subset_filters:
        return all_subsets
    return [(name, path) for name, path in all_subsets if name in subset_filters]


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    if not dataset_dir.exists():
        print(f"[ERROR] 找不到測試集資料夾：{dataset_dir}")
        return 1

    subset_filters = set(args.subset) if args.subset else None
    file_filters = set(args.file) if args.file else None
    selected_configs = [CONFIG_MAP[name] for name in (args.config or CONFIG_MAP.keys())]
    subsets = build_subsets(dataset_dir, subset_filters)

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
    print("去背服務效能／品質基準測試")
    print(f"API_URL      : {API_URL}")
    print(f"DATASET_DIR  : {dataset_dir}")
    print(f"TOTAL_CASES  : {total_cases}")
    print(f"SUBSETS      : {', '.join(name for name, _ in subsets)}")
    print(f"CONFIGS      : {', '.join(config['name'] for config in selected_configs)}")
    print(f"FILE_FILTERS : {', '.join(sorted(file_filters)) if file_filters else '(none)'}")
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
                result = run_one_config(session, subset_name, cases, config)
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
        "configs": config_reports,
    }

    report_path = Path(args.report_file).resolve()
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("全部測試完成")
    print("=" * 80)
    print(f"script_elapsed_sec: {total_elapsed_sec:.4f}")
    print(f"report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
