from __future__ import annotations

from typing import Any


V25_EPISODE_METRIC_DEFAULTS: dict[str, int | float] = {
    "episode_do_no_harm_events": 0,
    "episode_do_no_harm_suppressed_steps": 0,
    "episode_do_no_harm_cooldown_steps": 0,
    "episode_disturbance_max": 0.0,
    "episode_disturbance_sum": 0.0,
    "episode_disturbance_steps": 0,
    "episode_residual_action_sum": 0.0,
    "episode_residual_heading_abs_sum": 0.0,
    "episode_residual_speed_abs_sum": 0.0,
    "episode_residual_agl_abs_sum": 0.0,
    "episode_action_delta_sum": 0.0,
    "episode_intervention_need_sum": 0.0,
    "episode_unneeded_residual_sum": 0.0,
    "episode_needed_residual_sum": 0.0,
    "episode_apas_interventions": 0,
    "episode_apas_segment_rejections": 0,
    "episode_apas_no_valid_candidates": 0,
    "episode_stale_waypoint_skips": 0,
    "episode_stale_waypoint_skip_delta": 0,
    "episode_destructive_core_hits": 0,
    "episode_unproductive_residual_cost_sum": 0.0,
    "episode_progress_shortfall_sum": 0.0,
    "episode_apas_intervention_cost_sum": 0.0,
    "episode_speed_residual_cost_sum": 0.0,
    "episode_residual_gate_sum": 0.0,
    "episode_local_hazard_need_sum": 0.0,
    "episode_local_hazard_cost_sum": 0.0,
    "episode_local_hazard_trend_need_sum": 0.0,
    "episode_local_hazard_forward_delta_sum": 0.0,
    "episode_local_hazard_positive_trend_steps": 0,
    "episode_eval_maneuver_extra_energy_j": 0.0,
    "episode_eval_safety_intervention_burden": 0.0,
    "episode_eval_adjusted_energy_j": 0.0,
    "episode_expert_normal_steps": 0,
    "episode_expert_cautious_steps": 0,
    "episode_expert_cautious_trend_steps": 0,
    "episode_expert_avoiding_steps": 0,
    "episode_expert_emergency_steps": 0,
    "episode_expert_band_avoidance_steps": 0,
    "episode_expert_pre_emergency_slow_steps": 0,
    "episode_expert_recovering_steps": 0,
    "episode_expert_rejoin_actions": 0,
    "episode_expert_rejoin_attempts": 0,
    "episode_expert_rejoin_rejected": 0,
    "episode_replans": 0,
    "episode_replan_successes": 0,
    "episode_replan_failures": 0,
    "episode_replan_to_rejoin_successes": 0,
    "episode_replan_to_goal_successes": 0,
    "episode_replan_path_drift_triggers": 0,
    "episode_replan_low_progress_triggers": 0,
    "episode_replan_no_valid_triggers": 0,
}


V25_RUNTIME_TRACKER_DEFAULTS: dict[str, int | str] = {
    "replan_cooldown_steps_remaining": 0,
    "consecutive_replan_low_progress_steps": 0,
    "apas_no_valid_linger_steps": 0,
    "last_replan_event": "none",
    "consecutive_avoiding_steps": 0,
    "consecutive_recovering_steps": 0,
    "consecutive_cautious_steps": 0,
    "consecutive_low_progress_steps": 0,
}


def reset_v25_episode_metrics(target: Any) -> None:
    for name, value in V25_EPISODE_METRIC_DEFAULTS.items():
        setattr(target, name, value)


def reset_v25_runtime_trackers(target: Any) -> None:
    for name, value in V25_RUNTIME_TRACKER_DEFAULTS.items():
        setattr(target, name, value)
