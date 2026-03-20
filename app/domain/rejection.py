from typing import Dict


def evaluate_rejection(
    foreground_ratio: float,
    alpha_mean: float,
    debug_mid_alpha_ratio: float,
    debug_high_alpha_ratio: float,
    bbox_width_ratio: float,
    bbox_height_ratio: float,
    final_edge_metrics: Dict[str, float],
    reject_low_confidence: bool,
    reject_edge_quality: bool,
) -> Dict[str, object]:
    final_should_reject = False
    final_reject_reason = None

    large_garment_edge_bad = (
        foreground_ratio >= 0.30
        and bbox_width_ratio >= 0.75
        and bbox_height_ratio >= 0.70
        and final_edge_metrics["edge_band_mid_ratio"] >= 0.27
        and final_edge_metrics["edge_band_low_ratio"] >= 0.055
    )

    low_height_object_edge_bad = (
        foreground_ratio >= 0.18
        and bbox_width_ratio >= 0.75
        and bbox_height_ratio <= 0.45
        and final_edge_metrics["edge_band_ratio"] >= 0.22
        and final_edge_metrics["edge_band_mid_ratio"] >= 0.32
        and final_edge_metrics["edge_band_low_ratio"] >= 0.065
    )

    if reject_low_confidence:
        if reject_edge_quality and (
            large_garment_edge_bad or low_height_object_edge_bad
        ):
            final_should_reject = True
            final_reject_reason = "edge_quality_low"

        elif (
            foreground_ratio >= 0.18
            and foreground_ratio <= 0.55
            and debug_mid_alpha_ratio >= 0.18
            and debug_high_alpha_ratio <= 0.22
            and bbox_width_ratio >= 0.78
        ):
            final_should_reject = True
            final_reject_reason = "complex_background_low_confidence"

        elif foreground_ratio <= 0.08 and alpha_mean <= 45:
            final_should_reject = True
            final_reject_reason = "nearly_empty_mask"

    return {
        "final_should_reject": final_should_reject,
        "final_reject_reason": final_reject_reason,
        "large_garment_edge_bad": large_garment_edge_bad,
        "low_height_object_edge_bad": low_height_object_edge_bad,
        "edge_quality_low_candidate": bool(
            large_garment_edge_bad or low_height_object_edge_bad
        ),
    }
