from typing import Dict


def build_reject_payload(
    final_reject_reason: str,
    foreground_ratio: float,
    alpha_mean: float,
    debug_mid_alpha_ratio: float,
    debug_high_alpha_ratio: float,
    bbox_width_ratio: float,
    bbox_height_ratio: float,
    final_edge_metrics: Dict[str, float],
    edge_quality_low_candidate: bool,
) -> Dict[str, object]:
    return {
        "ok": False,
        "code": "LOW_CONFIDENCE_MASK",
        "message": "背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
        "reason": final_reject_reason,
        "metrics": {
            "foreground_ratio": round(float(foreground_ratio), 6),
            "alpha_mean": round(float(alpha_mean), 4),
            "mid_alpha_ratio": round(float(debug_mid_alpha_ratio), 6),
            "high_alpha_ratio": round(float(debug_high_alpha_ratio), 6),
            "bbox_width_ratio": round(float(bbox_width_ratio), 6),
            "bbox_height_ratio": round(float(bbox_height_ratio), 6),
            "edge_band_ratio": round(float(final_edge_metrics["edge_band_ratio"]), 6),
            "edge_band_mid_ratio": round(float(final_edge_metrics["edge_band_mid_ratio"]), 6),
            "edge_band_low_ratio": round(float(final_edge_metrics["edge_band_low_ratio"]), 6),
            "edge_quality_low_candidate": bool(edge_quality_low_candidate),
        },
    }


def build_success_headers(
    fallback_used: bool,
    actual_model: str,
    final_edge_candidate: bool,
    final_edge_metrics: Dict[str, float],
) -> Dict[str, str]:
    return {
        "X-RemoveBg-Fallback-Used": "true" if fallback_used else "false",
        "X-RemoveBg-Model": actual_model,
        "X-Edge-Quality-Candidate": "true" if final_edge_candidate else "false",
        "X-Edge-Band-Ratio": f"{final_edge_metrics['edge_band_ratio']:.6f}",
        "X-Edge-Band-Mid-Ratio": f"{final_edge_metrics['edge_band_mid_ratio']:.6f}",
        "X-Edge-Band-Low-Ratio": f"{final_edge_metrics['edge_band_low_ratio']:.6f}",
    }
