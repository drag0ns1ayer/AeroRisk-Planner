from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from configs.config import SimulationConfig


EMPTY_LOCAL_HAZARD_SUMMARY: dict[str, float] = {
    "hazard_need": 0.0,
    "base_hazard_need": 0.0,
    "max_danger": 0.0,
    "mean_danger": 0.0,
    "forward_danger": 0.0,
    "nearest_closeness": 0.0,
    "nearest_forward_alignment": 0.0,
    "trend_need": 0.0,
    "delta_max_danger": 0.0,
    "delta_forward_danger": 0.0,
    "delta_nearest_closeness": 0.0,
    "gradual_warning": 0.0,
    "risk_membrane_wall_ahead": 0.0,
    "risk_membrane_no_escape_gap": 0.0,
    "risk_membrane_front_blocked_width_deg": 0.0,
    "risk_membrane_best_gap_angle_deg": 0.0,
    "risk_membrane_best_gap_width_deg": 0.0,
    "risk_membrane_best_gap_side": 0.0,
    "risk_membrane_max_extended_risk": 0.0,
}


def empty_local_hazard_summary() -> dict[str, float]:
    return dict(EMPTY_LOCAL_HAZARD_SUMMARY)


def local_hazard_memory_item(summary: dict[str, float]) -> dict[str, float]:
    return {
        "max_danger": float(summary.get("max_danger", 0.0)),
        "forward_danger": float(summary.get("forward_danger", 0.0)),
        "nearest_closeness": float(summary.get("nearest_closeness", 0.0)),
    }


def local_hazard_history_features(
    current: dict[str, float],
    history: Sequence[dict[str, Any]],
    config: SimulationConfig,
) -> dict[str, float]:
    if not history:
        return {
            "trend_need": 0.0,
            "delta_max_danger": 0.0,
            "delta_forward_danger": 0.0,
            "delta_nearest_closeness": 0.0,
        }

    oldest = history[0]
    delta_max = float(current["max_danger"] - float(oldest["max_danger"]))
    delta_forward = float(current["forward_danger"] - float(oldest["forward_danger"]))
    delta_near = float(current["nearest_closeness"] - float(oldest["nearest_closeness"]))
    positive_signal = max(0.0, delta_forward) + 0.75 * max(0.0, delta_max) + 0.50 * max(0.0, delta_near)
    threshold = float(max(config.v25_local_hazard_trend_threshold, 1e-6))
    trend_need = float(np.clip(positive_signal / (4.0 * threshold), 0.0, 1.0))
    return {
        "trend_need": trend_need,
        "delta_max_danger": delta_max,
        "delta_forward_danger": delta_forward,
        "delta_nearest_closeness": delta_near,
    }


def local_hazard_gradual_warning(summary: dict[str, float], config: SimulationConfig) -> float:
    if float(summary.get("base_hazard_need", summary.get("hazard_need", 0.0))) >= float(
        config.v25_expert_activation_hazard
    ):
        return 0.0
    if max(float(summary.get("max_danger", 0.0)), float(summary.get("forward_danger", 0.0))) >= float(
        config.v25_expert_hard_risk_threshold
    ):
        return 0.0
    if float(summary.get("trend_need", 0.0)) < float(config.v25_expert_trend_warning_need):
        return 0.0
    if float(summary.get("delta_forward_danger", 0.0)) < float(config.v25_expert_trend_forward_delta):
        return 0.0
    return 1.0
