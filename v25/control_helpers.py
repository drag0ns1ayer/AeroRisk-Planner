from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from configs.config import SimulationConfig


def advance_waypoint_index(
    *,
    path: Sequence[Sequence[float]],
    current_idx: int,
    pos_xy: np.ndarray,
    refresh_radius_m: float,
    stale_skip_enabled: bool,
    stale_min_advance: int,
    stale_corridor_m: float,
) -> tuple[int, int]:
    """
    Advance a path waypoint index with conservative stale-waypoint skipping.

    Returns ``(new_idx, stale_skip_delta)``. The delta is positive only when the
    projection-style stale skip advances beyond the normal waypoint-radius
    rule.
    """
    if not path:
        return int(current_idx), 0

    idx = int(np.clip(current_idx, 0, len(path) - 1))
    pos = np.asarray(pos_xy, dtype=float)

    while idx < len(path) - 1:
        wp_xy = np.asarray(path[idx][:2], dtype=float)
        if float(np.linalg.norm(pos - wp_xy)) < float(refresh_radius_m):
            idx += 1
        else:
            break

    if not bool(stale_skip_enabled) or idx >= len(path) - 1:
        return idx, 0

    distances = [float(np.linalg.norm(pos - np.asarray(point[:2], dtype=float))) for point in path]
    nearest_idx = int(np.argmin(distances))
    nearest_dist = float(distances[nearest_idx])
    min_advance = int(max(1, stale_min_advance))
    corridor_m = float(max(0.0, stale_corridor_m))
    old_idx = int(idx)

    if nearest_idx >= old_idx + min_advance and nearest_dist <= corridor_m:
        idx = min(nearest_idx, len(path) - 1)
        return idx, max(0, idx - old_idx)

    return idx, 0


def default_do_no_harm_info() -> dict[str, Any]:
    return {
        "do_no_harm_active": False,
        "do_no_harm_reason": "none",
        "do_no_harm_recent_progress_sum": 0.0,
        "do_no_harm_recent_risk_delta": 0.0,
        "do_no_harm_recent_apas_interventions": 0,
        "do_no_harm_recent_segment_rejections": 0,
        "do_no_harm_recent_no_valid_candidates": 0,
    }


def evaluate_do_no_harm_gate(
    *,
    raw_action: np.ndarray,
    history: Sequence[dict[str, Any]],
    cooldown_steps_remaining: int,
    local_hazard: dict[str, float],
    config: SimulationConfig,
) -> tuple[np.ndarray, dict[str, Any], int, bool, bool, bool]:
    """
    Evaluate the residual rollback gate without mutating environment state.

    Returns:
        gated_action, info, next_cooldown, event_started, suppressed_step,
        cooldown_suppressed_step.
    """
    action = np.asarray(raw_action, dtype=float).copy()
    info = default_do_no_harm_info()

    if not bool(getattr(config, "v25_do_no_harm_gate_enabled", True)):
        info["do_no_harm_reason"] = "disabled"
        return action, info, int(cooldown_steps_remaining), False, False, False

    upper_layer_active = float(np.linalg.norm(action)) >= float(config.v25_do_no_harm_min_raw_action_norm)
    cooldown = int(max(0, cooldown_steps_remaining))

    if cooldown > 0:
        next_cooldown = cooldown - 1
        if upper_layer_active:
            info.update(do_no_harm_active=True, do_no_harm_reason="cooldown")
            return np.zeros_like(action), info, next_cooldown, False, True, True
        info["do_no_harm_reason"] = "cooldown_idle"
        return action, info, next_cooldown, False, False, False

    window = int(max(1, config.v25_do_no_harm_window_steps))
    if not upper_layer_active or len(history) < window:
        info["do_no_harm_reason"] = "inactive" if not upper_layer_active else "insufficient_history"
        return action, info, 0, False, False, False

    recent = list(history)[-window:]
    progress_sum = float(sum(float(item["progress_m"]) for item in recent))
    risk_delta = float(float(recent[-1]["p_crash"]) - float(recent[0]["p_crash"]))
    apas_interventions = int(sum(int(item["apas_intervened"]) for item in recent))
    segment_rejections = int(sum(int(item["apas_segment_rejections"]) for item in recent))
    no_valid_candidates = int(sum(int(item["apas_no_valid_candidate"]) for item in recent))
    info.update(
        do_no_harm_recent_progress_sum=progress_sum,
        do_no_harm_recent_risk_delta=risk_delta,
        do_no_harm_recent_apas_interventions=apas_interventions,
        do_no_harm_recent_segment_rejections=segment_rejections,
        do_no_harm_recent_no_valid_candidates=no_valid_candidates,
    )

    bad_progress = progress_sum <= float(config.v25_do_no_harm_min_progress_sum_m)
    risk_not_improving = risk_delta >= -float(config.v25_do_no_harm_risk_drop_eps)
    apas_struggling = (
        apas_interventions >= int(config.v25_do_no_harm_min_apas_interventions)
        or segment_rejections >= int(config.v25_do_no_harm_min_segment_rejections)
        or no_valid_candidates >= int(config.v25_do_no_harm_min_no_valid_candidates)
    )
    slow_into_risk_wall = (
        risk_not_improving
        and progress_sum <= float(config.v25_do_no_harm_slow_trap_max_progress_sum_m)
        and float(action[1]) <= float(config.v25_do_no_harm_slow_trap_speed_action)
        and (
            float(local_hazard.get("risk_membrane_no_escape_gap", 0.0)) >= 0.5
            or float(local_hazard.get("risk_membrane_wall_ahead", 0.0)) >= 0.5
            or float(local_hazard.get("forward_danger", 0.0)) >= float(config.v25_expert_hard_risk_threshold)
        )
    )

    if bad_progress and risk_not_improving and apas_struggling:
        info.update(do_no_harm_active=True, do_no_harm_reason="bad_progress_risk_not_improving")
        return (
            np.zeros_like(action),
            info,
            int(config.v25_do_no_harm_cooldown_steps),
            True,
            True,
            False,
        )

    if slow_into_risk_wall:
        info.update(do_no_harm_active=True, do_no_harm_reason="slow_into_risk_wall")
        return (
            np.zeros_like(action),
            info,
            int(config.v25_do_no_harm_cooldown_steps),
            True,
            True,
            False,
        )

    info["do_no_harm_reason"] = "clear"
    return action, info, 0, False, False, False


def compute_evaluation_costs(
    *,
    action: np.ndarray,
    action_delta: float,
    apas_info: dict[str, Any],
    expert_mode: str,
    base_energy_step_j: float,
    config: SimulationConfig,
) -> dict[str, float]:
    """
    Compute reporting-only maneuver energy and safety intervention burden.

    These metrics do not affect the executed action. They make comparisons
    fairer by making APAS and emergency behavior visible as extra burden rather
    than treating last-resort safety interventions as free.
    """
    action_arr = np.asarray(action, dtype=float)
    mode = str(expert_mode)

    expert_mode_burden = 0.0
    if mode in ("cautious", "cautious_trend"):
        expert_mode_burden = float(config.v25_eval_expert_cautious_burden)
    elif mode in ("avoiding", "band_avoidance"):
        expert_mode_burden = float(config.v25_eval_expert_avoiding_burden)
    elif mode in ("emergency", "pre_emergency_slow"):
        expert_mode_burden = float(config.v25_eval_expert_emergency_burden)
    elif mode == "recovering":
        expert_mode_burden = float(config.v25_eval_expert_recovering_burden)

    residual_maneuver_extra_energy_j = (
        float(config.v25_eval_maneuver_heading_energy_j) * abs(float(action_arr[0])) ** 2
        + float(config.v25_eval_maneuver_speed_energy_j) * abs(float(action_arr[1])) ** 2
        + float(config.v25_eval_maneuver_agl_energy_j) * abs(float(action_arr[2])) ** 2
        + float(config.v25_eval_maneuver_action_delta_energy_j) * (float(action_delta) ** 2)
    )

    apas_maneuver_extra_energy_j = 0.0
    if bool(apas_info.get("apas_intervened", False)):
        apas_maneuver_extra_energy_j = (
            float(config.v25_eval_apas_fixed_energy_j)
            + float(config.v25_eval_apas_heading_energy_j_per_deg)
            * abs(float(apas_info.get("apas_heading_offset_deg", 0.0)))
            + float(config.v25_eval_apas_speed_reduction_energy_j_per_mps2)
            * (max(0.0, float(apas_info.get("apas_speed_reduction_mps", 0.0))) ** 2)
            + float(config.v25_eval_apas_agl_energy_j_per_m)
            * max(0.0, float(apas_info.get("apas_agl_increment_m", 0.0)))
        )

    maneuver_extra_energy_j = float(residual_maneuver_extra_energy_j + apas_maneuver_extra_energy_j)
    safety_intervention_burden = (
        expert_mode_burden
        + float(config.v25_eval_apas_intervention_burden)
        * float(bool(apas_info.get("apas_intervened", False)))
        + float(config.v25_eval_apas_segment_rejection_burden)
        * float(apas_info.get("apas_segment_rejections", 0))
        + float(config.v25_eval_apas_no_valid_burden)
        * float(bool(apas_info.get("apas_no_valid_candidate", False)))
    )
    adjusted_energy_step_j = (
        float(base_energy_step_j)
        + maneuver_extra_energy_j
        + safety_intervention_burden * float(config.v25_eval_burden_energy_equivalent_j)
    )

    return {
        "expert_mode_burden": float(expert_mode_burden),
        "residual_maneuver_extra_energy_j": float(residual_maneuver_extra_energy_j),
        "apas_maneuver_extra_energy_j": float(apas_maneuver_extra_energy_j),
        "maneuver_extra_energy_j": float(maneuver_extra_energy_j),
        "safety_intervention_burden": float(safety_intervention_burden),
        "adjusted_energy_step_j": float(adjusted_energy_step_j),
    }
