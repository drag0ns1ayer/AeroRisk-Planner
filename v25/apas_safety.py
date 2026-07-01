from __future__ import annotations

from typing import Callable

import numpy as np

from configs.config import SimulationConfig


def empty_segment_probe() -> dict[str, float | bool]:
    return {"max_risk_bonus": 0.0, "destructive_core_hit": False}


def default_apas_info(candidate_index: int = 0, intervened: bool = False) -> dict[str, float | int | bool]:
    return {
        "apas_intervened": bool(intervened),
        "apas_heading_offset_deg": 0.0,
        "apas_speed_reduction_mps": 0.0,
        "apas_agl_increment_m": 0.0,
        "apas_candidate_index": int(candidate_index),
        "apas_segment_rejections": 0,
        "apas_endpoint_rejections": 0,
        "apas_no_valid_candidate": False,
        "apas_segment_max_risk_bonus": 0.0,
        "apas_segment_core_hit": False,
    }


def probe_random_layer_segment(
    *,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    probe_time_s: float,
    sample_count: int,
    risk_bonus_at: Callable[[float, float, float], float],
    core_hit_at: Callable[[float, float, float], bool],
) -> dict[str, float | bool]:
    sample_count = max(1, int(sample_count))
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)

    max_risk_bonus = 0.0
    core_hit = False
    for idx in range(sample_count + 1):
        alpha = idx / sample_count
        point = (1.0 - alpha) * start + alpha * end
        x = float(point[0])
        y = float(point[1])
        max_risk_bonus = max(max_risk_bonus, float(risk_bonus_at(x, y, float(probe_time_s))))
        core_hit = core_hit or bool(core_hit_at(x, y, float(probe_time_s)))

    return {"max_risk_bonus": float(max_risk_bonus), "destructive_core_hit": bool(core_hit)}


def segment_probe_is_safe(
    segment_probe: dict,
    *,
    enabled: bool,
    risk_threshold: float,
) -> bool:
    if not bool(enabled):
        return True
    segment_core_hit = bool(segment_probe.get("destructive_core_hit", False))
    segment_high_risk = float(segment_probe.get("max_risk_bonus", 0.0)) > float(risk_threshold)
    return not (segment_core_hit or segment_high_risk)


def apas_candidate_score(segment_probe: dict, transition: dict, candidate_index: int) -> tuple[float, float, float, int]:
    return (
        1.0 if bool(segment_probe.get("destructive_core_hit", False)) else 0.0,
        float(segment_probe.get("max_risk_bonus", 0.0)),
        float(transition.get("p_crash", 0.0)),
        int(candidate_index),
    )


def build_apas_candidate_info(
    *,
    candidate_index: int,
    heading_offset_deg: float,
    desired_airspeed_mps: float,
    test_speed_mps: float,
    desired_agl_m: float,
    test_agl_m: float,
    segment_rejections: int,
    endpoint_rejections: int,
    segment_probe: dict,
) -> dict[str, float | int | bool]:
    return {
        "apas_intervened": int(candidate_index) > 0,
        "apas_heading_offset_deg": float(heading_offset_deg),
        "apas_speed_reduction_mps": float(max(0.0, float(desired_airspeed_mps) - float(test_speed_mps))),
        "apas_agl_increment_m": float(float(test_agl_m) - float(desired_agl_m)),
        "apas_candidate_index": int(candidate_index),
        "apas_segment_rejections": int(segment_rejections),
        "apas_endpoint_rejections": int(endpoint_rejections),
        "apas_no_valid_candidate": False,
        "apas_segment_max_risk_bonus": float(segment_probe.get("max_risk_bonus", 0.0)),
        "apas_segment_core_hit": bool(segment_probe.get("destructive_core_hit", False)),
    }


def generate_apas_candidates(
    *,
    desired_heading_deg: float,
    desired_airspeed_mps: float,
    desired_agl_m: float,
    min_clearance_agl: float,
    max_clearance_agl: float,
    config: SimulationConfig,
    wrap_angle: Callable[[float], float],
) -> list[dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []
    candidate_index = 0
    for agl_increment in config.v25_apas_agl_increments_m:
        test_agl = float(
            np.clip(
                float(desired_agl_m) + float(agl_increment),
                float(min_clearance_agl),
                float(max_clearance_agl),
            )
        )
        for heading_offset in config.v25_apas_heading_offsets_deg:
            test_heading = float(wrap_angle(float(desired_heading_deg) + float(heading_offset)))
            test_speed = float(desired_airspeed_mps)
            while test_speed >= float(config.rl_speed_min) - 1e-9:
                candidates.append(
                    {
                        "candidate_index": int(candidate_index),
                        "heading_deg": float(test_heading),
                        "airspeed_mps": float(test_speed),
                        "agl_m": float(test_agl),
                        "heading_offset_deg": float(heading_offset),
                    }
                )
                candidate_index += 1
                test_speed -= float(config.v25_apas_speed_decrement_mps)
    return candidates
