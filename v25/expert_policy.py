from __future__ import annotations

import numpy as np

from configs.config import SimulationConfig


def expert_candidate_actions(
    config: SimulationConfig,
    *,
    emergency: bool = False,
    mild: bool = False,
) -> list[np.ndarray]:
    if emergency:
        headings = config.v25_expert_emergency_heading_actions
        speeds = config.v25_expert_emergency_speed_actions
        agls = config.v25_expert_emergency_agl_actions
    elif mild:
        headings = config.v25_expert_mild_heading_actions
        speeds = config.v25_expert_mild_speed_actions
        agls = config.v25_expert_mild_agl_actions
    else:
        headings = config.v25_expert_heading_actions
        speeds = config.v25_expert_speed_actions
        agls = config.v25_expert_agl_actions
    return [
        np.array([heading_action, speed_action, agl_action], dtype=float)
        for heading_action in headings
        for speed_action in speeds
        for agl_action in agls
    ]


def select_expert_action_from_evaluations(
    *,
    zero_eval: dict,
    normal_evaluations: list[dict],
    emergency_evaluations: list[dict],
    gradual_warning: bool,
    config: SimulationConfig,
) -> tuple[np.ndarray, str]:
    zero_action = np.zeros(3, dtype=float)
    candidate_mode = "cautious_trend" if gradual_warning else "avoiding"
    improvement_threshold = (
        float(config.v25_expert_trend_risk_improvement_threshold)
        if gradual_warning
        else float(config.v25_expert_risk_improvement_threshold)
    )

    safe_normal = [entry for entry in normal_evaluations if int(entry["hard_violation_count"]) == 0]
    if safe_normal:
        best = min(safe_normal, key=lambda entry: float(entry["score"]))
        if int(zero_eval["hard_violation_count"]) == 0:
            risk_improvement = float(zero_eval["max_risk"]) - float(best["max_risk"])
            if risk_improvement < improvement_threshold:
                return zero_action, "cautious_trend" if gradual_warning else "cautious"
        return np.asarray(best["action"], dtype=float), candidate_mode

    safe_emergency = [entry for entry in emergency_evaluations if int(entry["hard_violation_count"]) == 0]
    pool = safe_emergency if safe_emergency else emergency_evaluations
    best = min(
        pool,
        key=lambda entry: (
            int(entry["hard_violation_count"]),
            float(entry["max_risk"]),
            float(entry["score"]),
        ),
    )
    return np.asarray(best["action"], dtype=float), "emergency"


def score_expert_rollout_step(
    *,
    rollout_step: int,
    action: np.ndarray,
    p_crash: float,
    path_error_m: float,
    power_w: float,
    progress_m: float,
    progress_shortfall_m: float,
    core_or_high_risk: bool,
    collision_or_overload: bool,
    config: SimulationConfig,
) -> tuple[float, int]:
    step_weight = 1.0 + 0.20 * int(rollout_step)
    score = step_weight * (
        float(config.v25_expert_risk_gain) * (float(p_crash) ** 2)
        + float(config.v25_expert_path_error_gain) * float(path_error_m)
        + float(config.v25_expert_power_gain) * max(0.0, float(power_w) - float(config.base_power))
        + float(config.v25_expert_action_gain) * float(np.linalg.norm(action))
        + float(config.v25_expert_progress_gain) * float(progress_shortfall_m)
        - float(config.v25_expert_progress_gain) * float(progress_m)
    )

    hard_violation_count = 0
    if bool(core_or_high_risk):
        hard_violation_count += 1
        score += float(config.v25_expert_core_penalty) * step_weight
    if bool(collision_or_overload):
        hard_violation_count += 1
        score += float(config.v25_expert_core_penalty) * step_weight

    return float(score), int(hard_violation_count)


def finalize_expert_rollout_score(
    *,
    score: float,
    final_path_error_m: float,
    final_goal_dist_m: float,
    initial_goal_dist_m: float,
    hard_violation_count: int,
    config: SimulationConfig,
) -> float:
    terminal_goal_shortfall = max(0.0, float(final_goal_dist_m) - float(initial_goal_dist_m))
    total = (
        float(score)
        + float(config.v25_expert_final_path_error_gain) * float(final_path_error_m)
        + float(config.v25_expert_final_progress_gain) * terminal_goal_shortfall
    )
    if int(hard_violation_count):
        total += float(config.v25_expert_hard_constraint_penalty) * int(hard_violation_count)
    return float(total)
