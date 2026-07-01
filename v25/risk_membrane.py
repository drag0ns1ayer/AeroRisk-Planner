from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from configs.config import SimulationConfig


RISK_MEMBRANE_ZERO_SUMMARY: dict[str, float] = {
    "risk_membrane_wall_ahead": 0.0,
    "risk_membrane_no_escape_gap": 0.0,
    "risk_membrane_front_blocked_width_deg": 0.0,
    "risk_membrane_best_gap_angle_deg": 0.0,
    "risk_membrane_best_gap_width_deg": 0.0,
    "risk_membrane_best_gap_side": 0.0,
    "risk_membrane_max_extended_risk": 0.0,
}


def empty_risk_membrane_summary() -> dict[str, float]:
    return dict(RISK_MEMBRANE_ZERO_SUMMARY)


def wrap_angle_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def compute_risk_membrane_summary(
    *,
    origin_xy: np.ndarray,
    heading_deg: float,
    current_time_s: float,
    sample_points: Sequence[np.ndarray],
    danger_at: Callable[[float, float, float], float],
    config: SimulationConfig,
) -> dict[str, float]:
    """
    Build a local risk membrane from sparse circular observer samples.

    The function is intentionally snapshot-based: it consumes only current
    observations and a danger callback at the supplied observation time.
    """
    angle_min = float(config.v25_risk_membrane_angle_min_deg)
    angle_max = float(config.v25_risk_membrane_angle_max_deg)
    angle_step = float(config.v25_risk_membrane_angle_step_deg)
    angle_bins = np.arange(angle_min, angle_max + 0.5 * angle_step, angle_step, dtype=float)
    radius_m = float(config.v25_radar_radius_m)
    radial_bins = np.linspace(
        radius_m / max(int(config.v25_risk_membrane_radial_bins), 1),
        radius_m,
        int(config.v25_risk_membrane_radial_bins),
        dtype=float,
    )
    if len(angle_bins) == 0 or len(radial_bins) == 0:
        return empty_risk_membrane_summary()

    origin = np.asarray(origin_xy, dtype=float)
    observations: list[tuple[float, float, float]] = []
    for point in sample_points:
        offset = np.asarray(point, dtype=float) - origin
        dist = float(np.linalg.norm(offset))
        if dist < 1e-6:
            continue
        bearing = wrap_angle_deg(math.degrees(math.atan2(float(offset[1]), float(offset[0]))) - float(heading_deg))
        if bearing < angle_min - angle_step or bearing > angle_max + angle_step:
            continue
        danger = float(np.clip(danger_at(float(point[0]), float(point[1]), float(current_time_s)), 0.0, 1.0))
        if danger > 1e-6:
            observations.append((dist, bearing, danger))

    extended = np.zeros((len(radial_bins), len(angle_bins)), dtype=float)
    lambda_r = max(float(config.v25_risk_membrane_lambda_r_m), 1e-6)
    lambda_theta = max(float(config.v25_risk_membrane_lambda_theta_deg), 1e-6)
    for obs_dist, obs_bearing, danger in observations:
        for r_idx, r_center in enumerate(radial_bins):
            radial_decay = math.exp(-abs(float(r_center) - obs_dist) / lambda_r)
            for a_idx, angle_center in enumerate(angle_bins):
                angular_decay = math.exp(-abs(float(angle_center) - obs_bearing) / lambda_theta)
                extended[r_idx, a_idx] = max(extended[r_idx, a_idx], danger * radial_decay * angular_decay)

    angular_risk = np.max(extended, axis=0) if len(extended) else np.zeros(len(angle_bins), dtype=float)
    blocked = angular_risk >= float(config.v25_risk_membrane_block_threshold)
    front_mask = np.abs(angle_bins) <= float(config.v25_risk_membrane_front_window_deg)
    front_blocked_width = float(np.sum(blocked & front_mask) * angle_step)
    wall_ahead = front_blocked_width >= float(config.v25_risk_membrane_wall_width_deg)

    gaps: list[tuple[float, float, float]] = []
    start_idx: int | None = None
    for idx, is_blocked in enumerate(blocked):
        if not is_blocked and start_idx is None:
            start_idx = idx
        if (is_blocked or idx == len(blocked) - 1) and start_idx is not None:
            end_idx = idx - 1 if is_blocked else idx
            width = float((end_idx - start_idx + 1) * angle_step)
            center = float(0.5 * (angle_bins[start_idx] + angle_bins[end_idx]))
            if width >= float(config.v25_risk_membrane_min_gap_width_deg):
                gaps.append((center, width, float(abs(center))))
            start_idx = None

    if gaps:
        best_gap = min(gaps, key=lambda item: (item[2] - 0.02 * item[1], item[2]))
        best_gap_angle = float(best_gap[0])
        best_gap_width = float(best_gap[1])
        no_escape_gap = 0.0
    else:
        best_gap_angle = 0.0
        best_gap_width = 0.0
        no_escape_gap = 1.0 if wall_ahead else 0.0

    return {
        "risk_membrane_wall_ahead": float(bool(wall_ahead)),
        "risk_membrane_no_escape_gap": float(no_escape_gap),
        "risk_membrane_front_blocked_width_deg": front_blocked_width,
        "risk_membrane_best_gap_angle_deg": best_gap_angle,
        "risk_membrane_best_gap_width_deg": best_gap_width,
        "risk_membrane_best_gap_side": float(np.sign(best_gap_angle)),
        "risk_membrane_max_extended_risk": float(np.max(extended)) if extended.size else 0.0,
    }


def risk_membrane_action(local_hazard: dict[str, float], config: SimulationConfig) -> tuple[np.ndarray, str] | None:
    if float(local_hazard.get("risk_membrane_wall_ahead", 0.0)) <= 0.0:
        return None

    if float(local_hazard.get("risk_membrane_no_escape_gap", 0.0)) > 0.0:
        return (
            np.array([0.0, float(config.v25_expert_pre_emergency_speed_action), 0.0], dtype=float),
            "pre_emergency_slow",
        )

    gap_angle = float(local_hazard.get("risk_membrane_best_gap_angle_deg", 0.0))
    if abs(gap_angle) < 1e-6:
        return None

    heading_action = float(
        np.clip(
            gap_angle / max(float(config.rl_heading_delta_max_deg), 1e-6),
            -float(config.v25_expert_band_avoid_heading_limit),
            float(config.v25_expert_band_avoid_heading_limit),
        )
    )
    return (
        np.array([heading_action, float(config.v25_expert_band_avoid_speed_action), 0.0], dtype=float),
        "band_avoidance",
    )
