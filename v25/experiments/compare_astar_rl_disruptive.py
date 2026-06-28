from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("stable_baselines3 is required for RL evaluation.") from exc

from configs.config import SimulationConfig
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from v25.rl_env_disruptive import GuidedDroneEnvV25


Point2D = Tuple[float, float]


@dataclass
class EpisodeTask:
    seed: int
    start_xy: Point2D
    goal_xy: Point2D


def _path_length_xyz(points: List[Tuple[float, float, float]]) -> float:
    if points is None or len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        p0 = np.array(points[i - 1], dtype=np.float64)
        p1 = np.array(points[i], dtype=np.float64)
        total += float(np.linalg.norm(p1 - p0))
    return float(total)


def _goal_range_by_stage(config: SimulationConfig, stage: int) -> Tuple[float, float]:
    if stage == 1:
        return config.rl_goal_min_stage1_m, config.rl_goal_max_stage1_m
    if stage == 2:
        return config.rl_goal_min_stage2_m, config.rl_goal_max_stage2_m
    if stage == 3:
        return config.rl_goal_min_stage3_m, config.rl_goal_max_stage3_m
    return config.rl_goal_min_stage4_m, config.rl_goal_max_stage4_m


def _build_stack(config: SimulationConfig):
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    planner = AStarPlanner(config, estimator, physics)
    return map_manager, estimator, physics, planner


def _is_safe_xy(config: SimulationConfig, estimator: StateEstimator, x: float, y: float) -> bool:
    if estimator.map.is_in_nfz(x, y, nfz_list_km=config.nfz_list_km):
        return False
    z = estimator.get_altitude(x, y) + config.takeoff_altitude_agl
    p_crash, _ = estimator.get_risk(x, y, z, v_ground=max(config.drone_speed, 1.0), t_s=0.0)
    return p_crash <= config.rl_safe_spawn_risk_threshold


def sample_task_for_seed(config: SimulationConfig, seed: int, max_trials: int = 180) -> EpisodeTask:
    rng = np.random.default_rng(seed)
    _, estimator, _, planner = _build_stack(config)
    min_x, max_x, min_y, max_y = estimator.get_bounds()
    goal_min, goal_max = _goal_range_by_stage(config, config.curriculum_stage)

    for _ in range(max_trials):
        start_x = float(rng.uniform(min_x + config.rl_spawn_margin_m, max_x - config.rl_spawn_margin_m))
        start_y = float(rng.uniform(min_y + config.rl_spawn_margin_m, max_y - config.rl_spawn_margin_m))
        if not _is_safe_xy(config, estimator, start_x, start_y):
            continue

        start_xy = (start_x, start_y)
        for _ in range(max_trials):
            dist = float(rng.uniform(goal_min, goal_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            goal_x = float(np.clip(
                start_x + dist * np.cos(angle),
                min_x + config.rl_goal_margin_m,
                max_x - config.rl_goal_margin_m,
            ))
            goal_y = float(np.clip(
                start_y + dist * np.sin(angle),
                min_y + config.rl_goal_margin_m,
                max_y - config.rl_goal_margin_m,
            ))
            if not _is_safe_xy(config, estimator, goal_x, goal_y):
                continue

            goal_xy = (goal_x, goal_y)
            path = planner.search(start_xy, goal_xy, start_time_s=0.0)
            if path is not None and len(path) >= 2:
                return EpisodeTask(seed=seed, start_xy=start_xy, goal_xy=goal_xy)

    raise RuntimeError(f"Unable to sample a valid start/goal for seed={seed} within {max_trials} trials.")


def _evaluate_in_shared_true_world(
    config: SimulationConfig,
    task: EpisodeTask,
    model=None,
    enable_apas: bool = False,
    use_expert: bool = False,
) -> Dict[str, float]:
    run_cfg = copy.deepcopy(config)
    run_cfg.wind_seed = int(task.seed)
    run_cfg.enable_single_agent_gusts = False
    run_cfg.enable_random_gusts = False
    run_cfg.planner_time_mode = "4d"
    run_cfg.rl_enable_apas = bool(enable_apas)

    env = GuidedDroneEnvV25(run_cfg)
    obs, _ = env.reset(
        seed=int(task.seed),
        options={"start_xy": task.start_xy, "goal_xy": task.goal_xy},
    )
    terminated = False
    truncated = False
    final_info = {"is_success": False, "terminated_reason": "unknown"}
    path_distance_m = 0.0
    prev_pos = np.array(env.current_pos, dtype=np.float64)

    while not (terminated or truncated):
        if use_expert:
            action = env.local_avoidance_expert_action()
        elif model is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, final_info = env.step(action)
        curr_pos = np.array(env.current_pos, dtype=np.float64)
        path_distance_m += float(np.linalg.norm(curr_pos - prev_pos))
        prev_pos = curr_pos

    energy_used_j = float(run_cfg.battery_capacity_j - float(final_info.get("energy_remaining_j", env.energy_remaining)))
    avg_risk = float(np.mean(env.telemetry_risk)) if env.telemetry_risk else 0.0
    peak_risk = float(env.telemetry_max_p_crash) if env.telemetry_risk else 0.0
    peak_power = float(np.max(env.telemetry_power_w)) if env.telemetry_power_w else 0.0
    terminated_reason = final_info.get("terminated_reason", "unknown")

    return {
        "success": bool(final_info.get("is_success", False)),
        "terminated_reason": str(terminated_reason),
        "overload": bool(terminated_reason == "overload"),
        "energy_used_j": energy_used_j,
        "avg_risk": avg_risk,
        "peak_risk": peak_risk,
        "peak_power_w": peak_power,
        "mission_time_s": float(env.current_time),
        "path_distance_m": float(path_distance_m),
        "storm_hits": 0,
        "pulse_hits": 0,
        "overload_events": int(1 if terminated_reason == "overload" else 0),
        "high_risk_events": int(1 if terminated_reason == "storm_risk_too_high" else 0),
        "disturbance_steps": int(getattr(env, "episode_disturbance_steps", 0)),
        "disturbance_max": float(getattr(env, "episode_disturbance_max", 0.0)),
        "disturbance_mean": float(
            float(getattr(env, "episode_disturbance_sum", 0.0)) / max(env.current_step, 1)
        ),
        "episode_disturbance_max": float(getattr(env, "episode_disturbance_max", 0.0)),
        "episode_disturbance_mean": float(
            float(getattr(env, "episode_disturbance_sum", 0.0)) / max(env.current_step, 1)
        ),
        "episode_disturbance_steps": int(getattr(env, "episode_disturbance_steps", 0)),
        "episode_autonomy_bonus_sum": 0.0,
        "episode_residual_action_sum": float(getattr(env, "episode_residual_action_sum", 0.0)),
        "episode_residual_heading_abs_sum": float(getattr(env, "episode_residual_heading_abs_sum", 0.0)),
        "episode_residual_speed_abs_sum": float(getattr(env, "episode_residual_speed_abs_sum", 0.0)),
        "episode_residual_agl_abs_sum": float(getattr(env, "episode_residual_agl_abs_sum", 0.0)),
        "episode_action_delta_sum": float(getattr(env, "episode_action_delta_sum", 0.0)),
        "episode_intervention_need_mean": float(
            getattr(env, "episode_intervention_need_sum", 0.0) / max(env.current_step, 1)
        ),
        "episode_unneeded_residual_sum": float(getattr(env, "episode_unneeded_residual_sum", 0.0)),
        "episode_needed_residual_sum": float(getattr(env, "episode_needed_residual_sum", 0.0)),
        "episode_apas_interventions": int(getattr(env, "episode_apas_interventions", 0)),
        "episode_apas_segment_rejections": int(getattr(env, "episode_apas_segment_rejections", 0)),
        "episode_apas_no_valid_candidates": int(getattr(env, "episode_apas_no_valid_candidates", 0)),
        "episode_stale_waypoint_skips": int(getattr(env, "episode_stale_waypoint_skips", 0)),
        "episode_stale_waypoint_skip_delta": int(getattr(env, "episode_stale_waypoint_skip_delta", 0)),
        "episode_destructive_core_hits": int(getattr(env, "episode_destructive_core_hits", 0)),
        "episode_unproductive_residual_cost_sum": float(
            getattr(env, "episode_unproductive_residual_cost_sum", 0.0)
        ),
        "episode_progress_shortfall_mean": float(
            getattr(env, "episode_progress_shortfall_sum", 0.0) / max(env.current_step, 1)
        ),
        "episode_apas_intervention_cost_sum": float(getattr(env, "episode_apas_intervention_cost_sum", 0.0)),
        "episode_speed_residual_cost_sum": float(getattr(env, "episode_speed_residual_cost_sum", 0.0)),
        "episode_residual_gate_mean": float(getattr(env, "episode_residual_gate_sum", 0.0) / max(env.current_step, 1)),
        "episode_local_hazard_need_mean": float(
            getattr(env, "episode_local_hazard_need_sum", 0.0) / max(env.current_step, 1)
        ),
        "episode_local_hazard_cost_sum": float(getattr(env, "episode_local_hazard_cost_sum", 0.0)),
        "episode_local_hazard_trend_need_mean": float(
            getattr(env, "episode_local_hazard_trend_need_sum", 0.0) / max(env.current_step, 1)
        ),
        "episode_local_hazard_forward_delta_mean": float(
            getattr(env, "episode_local_hazard_forward_delta_sum", 0.0) / max(env.current_step, 1)
        ),
        "episode_local_hazard_positive_trend_steps": int(
            getattr(env, "episode_local_hazard_positive_trend_steps", 0)
        ),
        "episode_eval_maneuver_extra_energy_j": float(
            getattr(env, "episode_eval_maneuver_extra_energy_j", 0.0)
        ),
        "episode_eval_safety_intervention_burden": float(
            getattr(env, "episode_eval_safety_intervention_burden", 0.0)
        ),
        "episode_eval_adjusted_energy_j": float(getattr(env, "episode_eval_adjusted_energy_j", 0.0)),
        "episode_expert_normal_steps": int(getattr(env, "episode_expert_normal_steps", 0)),
        "episode_expert_cautious_steps": int(getattr(env, "episode_expert_cautious_steps", 0)),
        "episode_expert_cautious_trend_steps": int(getattr(env, "episode_expert_cautious_trend_steps", 0)),
        "episode_expert_avoiding_steps": int(getattr(env, "episode_expert_avoiding_steps", 0)),
        "episode_expert_emergency_steps": int(getattr(env, "episode_expert_emergency_steps", 0)),
        "episode_expert_band_avoidance_steps": int(getattr(env, "episode_expert_band_avoidance_steps", 0)),
        "episode_expert_pre_emergency_slow_steps": int(getattr(env, "episode_expert_pre_emergency_slow_steps", 0)),
        "episode_expert_recovering_steps": int(getattr(env, "episode_expert_recovering_steps", 0)),
        "episode_expert_rejoin_actions": int(getattr(env, "episode_expert_rejoin_actions", 0)),
        "episode_expert_rejoin_attempts": int(getattr(env, "episode_expert_rejoin_attempts", 0)),
        "episode_expert_rejoin_rejected": int(getattr(env, "episode_expert_rejoin_rejected", 0)),
        "episode_replans": int(getattr(env, "episode_replans", 0)),
        "episode_replan_successes": int(getattr(env, "episode_replan_successes", 0)),
        "episode_replan_failures": int(getattr(env, "episode_replan_failures", 0)),
        "episode_replan_to_rejoin_successes": int(getattr(env, "episode_replan_to_rejoin_successes", 0)),
        "episode_replan_to_goal_successes": int(getattr(env, "episode_replan_to_goal_successes", 0)),
        "episode_replan_path_drift_triggers": int(getattr(env, "episode_replan_path_drift_triggers", 0)),
        "episode_replan_low_progress_triggers": int(getattr(env, "episode_replan_low_progress_triggers", 0)),
        "episode_replan_no_valid_triggers": int(getattr(env, "episode_replan_no_valid_triggers", 0)),
    }


def evaluate_astar_only(config: SimulationConfig, task: EpisodeTask, enable_apas: bool = False) -> Dict[str, float]:
    return _evaluate_in_shared_true_world(config, task, model=None, enable_apas=enable_apas)


def evaluate_astar_plus_expert(
    config: SimulationConfig,
    task: EpisodeTask,
    enable_apas: bool = False,
) -> Dict[str, float]:
    return _evaluate_in_shared_true_world(config, task, model=None, enable_apas=enable_apas, use_expert=True)


def _load_ppo_model(model_path: str):
    p = Path(model_path)
    if p.suffix != ".zip":
        zip_candidate = p.with_suffix(".zip")
        if zip_candidate.exists():
            p = zip_candidate
    if not p.exists():
        raise FileNotFoundError(f"RL model not found: {model_path}")
    load_path = str(p)[:-4] if str(p).endswith(".zip") else str(p)
    model = PPO.load(load_path, device="cpu")
    expected_obs = 31 + GuidedDroneEnvV25._sensor_feature_count(SimulationConfig())
    model_obs = int(np.prod(model.observation_space.shape))
    if model_obs != expected_obs:
        raise ValueError(
            f"Model observation size is {model_obs}, but upgraded v2.5 requires {expected_obs}. "
            "Retrain the v2.5 policy before comparison."
        )
    return model


def evaluate_astar_plus_rl(config: SimulationConfig, task: EpisodeTask, model, enable_apas: bool = True) -> Dict[str, float]:
    return _evaluate_in_shared_true_world(config, task, model=model, enable_apas=enable_apas)


def summarize_rows(rows: List[Dict[str, object]], method: str) -> Dict[str, float]:
    sub = [r for r in rows if r["method"] == method]
    n = max(len(sub), 1)
    success_rate = float(sum(1 for r in sub if bool(r["success"])) / n)
    overload_rate = float(sum(1 for r in sub if bool(r["overload"])) / n)
    energy_mean = float(np.mean([float(r["energy_used_j"]) for r in sub])) if sub else 0.0
    avg_risk_mean = float(np.mean([float(r["avg_risk"]) for r in sub])) if sub else 0.0
    peak_risk_mean = float(np.mean([float(r["peak_risk"]) for r in sub])) if sub else 0.0
    peak_power_mean = float(np.mean([float(r["peak_power_w"]) for r in sub])) if sub else 0.0
    mission_time_mean = float(np.mean([float(r["mission_time_s"]) for r in sub])) if sub else 0.0
    path_distance_mean = float(np.mean([float(r["path_distance_m"]) for r in sub])) if sub else 0.0
    storm_hits_mean = float(np.mean([float(r["storm_hits"]) for r in sub])) if sub else 0.0
    pulse_hits_mean = float(np.mean([float(r["pulse_hits"]) for r in sub])) if sub else 0.0
    overload_events_mean = float(np.mean([float(r["overload_events"]) for r in sub])) if sub else 0.0
    high_risk_events_mean = float(np.mean([float(r["high_risk_events"]) for r in sub])) if sub else 0.0
    disturbance_steps_mean = float(np.mean([float(r["disturbance_steps"]) for r in sub])) if sub else 0.0
    disturbance_max_mean = float(np.mean([float(r["disturbance_max"]) for r in sub])) if sub else 0.0
    disturbance_mean_mean = float(np.mean([float(r["disturbance_mean"]) for r in sub])) if sub else 0.0
    episode_disturbance_max_mean = float(np.mean([float(r["episode_disturbance_max"]) for r in sub])) if sub else 0.0
    episode_disturbance_mean_mean = float(np.mean([float(r["episode_disturbance_mean"]) for r in sub])) if sub else 0.0
    episode_disturbance_steps_mean = float(np.mean([float(r["episode_disturbance_steps"]) for r in sub])) if sub else 0.0
    episode_autonomy_bonus_sum_mean = float(np.mean([float(r["episode_autonomy_bonus_sum"]) for r in sub])) if sub else 0.0
    episode_residual_action_sum_mean = float(np.mean([float(r["episode_residual_action_sum"]) for r in sub])) if sub else 0.0
    episode_residual_heading_abs_sum_mean = float(np.mean([float(r["episode_residual_heading_abs_sum"]) for r in sub])) if sub else 0.0
    episode_residual_speed_abs_sum_mean = float(np.mean([float(r["episode_residual_speed_abs_sum"]) for r in sub])) if sub else 0.0
    episode_residual_agl_abs_sum_mean = float(np.mean([float(r["episode_residual_agl_abs_sum"]) for r in sub])) if sub else 0.0
    episode_action_delta_sum_mean = float(np.mean([float(r["episode_action_delta_sum"]) for r in sub])) if sub else 0.0
    episode_intervention_need_mean = float(np.mean([float(r["episode_intervention_need_mean"]) for r in sub])) if sub else 0.0
    episode_unneeded_residual_sum_mean = float(np.mean([float(r["episode_unneeded_residual_sum"]) for r in sub])) if sub else 0.0
    episode_needed_residual_sum_mean = float(np.mean([float(r["episode_needed_residual_sum"]) for r in sub])) if sub else 0.0
    episode_apas_interventions_mean = float(np.mean([float(r["episode_apas_interventions"]) for r in sub])) if sub else 0.0
    episode_apas_segment_rejections_mean = (
        float(np.mean([float(r.get("episode_apas_segment_rejections", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_apas_no_valid_candidates_mean = (
        float(np.mean([float(r.get("episode_apas_no_valid_candidates", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_stale_waypoint_skips_mean = (
        float(np.mean([float(r.get("episode_stale_waypoint_skips", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_stale_waypoint_skip_delta_mean = (
        float(np.mean([float(r.get("episode_stale_waypoint_skip_delta", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_destructive_core_hits_mean = float(np.mean([float(r["episode_destructive_core_hits"]) for r in sub])) if sub else 0.0
    episode_unproductive_residual_cost_sum_mean = (
        float(np.mean([float(r["episode_unproductive_residual_cost_sum"]) for r in sub])) if sub else 0.0
    )
    episode_progress_shortfall_mean = (
        float(np.mean([float(r["episode_progress_shortfall_mean"]) for r in sub])) if sub else 0.0
    )
    episode_apas_intervention_cost_sum_mean = (
        float(np.mean([float(r["episode_apas_intervention_cost_sum"]) for r in sub])) if sub else 0.0
    )
    episode_speed_residual_cost_sum_mean = (
        float(np.mean([float(r["episode_speed_residual_cost_sum"]) for r in sub])) if sub else 0.0
    )
    episode_residual_gate_mean = (
        float(np.mean([float(r["episode_residual_gate_mean"]) for r in sub])) if sub else 0.0
    )
    episode_local_hazard_need_mean = (
        float(np.mean([float(r["episode_local_hazard_need_mean"]) for r in sub])) if sub else 0.0
    )
    episode_local_hazard_cost_sum_mean = (
        float(np.mean([float(r["episode_local_hazard_cost_sum"]) for r in sub])) if sub else 0.0
    )
    episode_local_hazard_trend_need_mean = (
        float(np.mean([float(r.get("episode_local_hazard_trend_need_mean", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_local_hazard_forward_delta_mean = (
        float(np.mean([float(r.get("episode_local_hazard_forward_delta_mean", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_local_hazard_positive_trend_steps_mean = (
        float(np.mean([float(r.get("episode_local_hazard_positive_trend_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_eval_maneuver_extra_energy_j_mean = (
        float(np.mean([float(r.get("episode_eval_maneuver_extra_energy_j", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_eval_safety_intervention_burden_mean = (
        float(np.mean([float(r.get("episode_eval_safety_intervention_burden", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_eval_adjusted_energy_j_mean = (
        float(np.mean([float(r.get("episode_eval_adjusted_energy_j", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_normal_steps_mean = (
        float(np.mean([float(r.get("episode_expert_normal_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_cautious_steps_mean = (
        float(np.mean([float(r.get("episode_expert_cautious_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_cautious_trend_steps_mean = (
        float(np.mean([float(r.get("episode_expert_cautious_trend_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_avoiding_steps_mean = (
        float(np.mean([float(r.get("episode_expert_avoiding_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_emergency_steps_mean = (
        float(np.mean([float(r.get("episode_expert_emergency_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_band_avoidance_steps_mean = (
        float(np.mean([float(r.get("episode_expert_band_avoidance_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_pre_emergency_slow_steps_mean = (
        float(np.mean([float(r.get("episode_expert_pre_emergency_slow_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_recovering_steps_mean = (
        float(np.mean([float(r.get("episode_expert_recovering_steps", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_rejoin_actions_mean = (
        float(np.mean([float(r.get("episode_expert_rejoin_actions", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_rejoin_attempts_mean = (
        float(np.mean([float(r.get("episode_expert_rejoin_attempts", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_expert_rejoin_rejected_mean = (
        float(np.mean([float(r.get("episode_expert_rejoin_rejected", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replans_mean = (
        float(np.mean([float(r.get("episode_replans", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_successes_mean = (
        float(np.mean([float(r.get("episode_replan_successes", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_failures_mean = (
        float(np.mean([float(r.get("episode_replan_failures", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_to_rejoin_successes_mean = (
        float(np.mean([float(r.get("episode_replan_to_rejoin_successes", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_to_goal_successes_mean = (
        float(np.mean([float(r.get("episode_replan_to_goal_successes", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_path_drift_triggers_mean = (
        float(np.mean([float(r.get("episode_replan_path_drift_triggers", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_low_progress_triggers_mean = (
        float(np.mean([float(r.get("episode_replan_low_progress_triggers", 0.0)) for r in sub])) if sub else 0.0
    )
    episode_replan_no_valid_triggers_mean = (
        float(np.mean([float(r.get("episode_replan_no_valid_triggers", 0.0)) for r in sub])) if sub else 0.0
    )
    return {
        "episodes": len(sub),
        "success_rate": success_rate,
        "overload_rate": overload_rate,
        "energy_used_j_mean": energy_mean,
        "avg_risk_mean": avg_risk_mean,
        "peak_risk_mean": peak_risk_mean,
        "peak_power_w_mean": peak_power_mean,
        "mission_time_s_mean": mission_time_mean,
        "path_distance_m_mean": path_distance_mean,
        "storm_hits_mean": storm_hits_mean,
        "pulse_hits_mean": pulse_hits_mean,
        "overload_events_mean": overload_events_mean,
        "high_risk_events_mean": high_risk_events_mean,
        "disturbance_steps_mean": disturbance_steps_mean,
        "disturbance_max_mean": disturbance_max_mean,
        "disturbance_mean_mean": disturbance_mean_mean,
        "episode_disturbance_max_mean": episode_disturbance_max_mean,
        "episode_disturbance_mean_mean": episode_disturbance_mean_mean,
        "episode_disturbance_steps_mean": episode_disturbance_steps_mean,
        "episode_autonomy_bonus_sum_mean": episode_autonomy_bonus_sum_mean,
        "episode_residual_action_sum_mean": episode_residual_action_sum_mean,
        "episode_residual_heading_abs_sum_mean": episode_residual_heading_abs_sum_mean,
        "episode_residual_speed_abs_sum_mean": episode_residual_speed_abs_sum_mean,
        "episode_residual_agl_abs_sum_mean": episode_residual_agl_abs_sum_mean,
        "episode_action_delta_sum_mean": episode_action_delta_sum_mean,
        "episode_intervention_need_mean": episode_intervention_need_mean,
        "episode_unneeded_residual_sum_mean": episode_unneeded_residual_sum_mean,
        "episode_needed_residual_sum_mean": episode_needed_residual_sum_mean,
        "episode_apas_interventions_mean": episode_apas_interventions_mean,
        "episode_apas_segment_rejections_mean": episode_apas_segment_rejections_mean,
        "episode_apas_no_valid_candidates_mean": episode_apas_no_valid_candidates_mean,
        "episode_stale_waypoint_skips_mean": episode_stale_waypoint_skips_mean,
        "episode_stale_waypoint_skip_delta_mean": episode_stale_waypoint_skip_delta_mean,
        "episode_destructive_core_hits_mean": episode_destructive_core_hits_mean,
        "episode_unproductive_residual_cost_sum_mean": episode_unproductive_residual_cost_sum_mean,
        "episode_progress_shortfall_mean": episode_progress_shortfall_mean,
        "episode_apas_intervention_cost_sum_mean": episode_apas_intervention_cost_sum_mean,
        "episode_speed_residual_cost_sum_mean": episode_speed_residual_cost_sum_mean,
        "episode_residual_gate_mean": episode_residual_gate_mean,
        "episode_local_hazard_need_mean": episode_local_hazard_need_mean,
        "episode_local_hazard_cost_sum_mean": episode_local_hazard_cost_sum_mean,
        "episode_local_hazard_trend_need_mean": episode_local_hazard_trend_need_mean,
        "episode_local_hazard_forward_delta_mean": episode_local_hazard_forward_delta_mean,
        "episode_local_hazard_positive_trend_steps_mean": episode_local_hazard_positive_trend_steps_mean,
        "episode_eval_maneuver_extra_energy_j_mean": episode_eval_maneuver_extra_energy_j_mean,
        "episode_eval_safety_intervention_burden_mean": episode_eval_safety_intervention_burden_mean,
        "episode_eval_adjusted_energy_j_mean": episode_eval_adjusted_energy_j_mean,
        "episode_expert_normal_steps_mean": episode_expert_normal_steps_mean,
        "episode_expert_cautious_steps_mean": episode_expert_cautious_steps_mean,
        "episode_expert_cautious_trend_steps_mean": episode_expert_cautious_trend_steps_mean,
        "episode_expert_avoiding_steps_mean": episode_expert_avoiding_steps_mean,
        "episode_expert_emergency_steps_mean": episode_expert_emergency_steps_mean,
        "episode_expert_band_avoidance_steps_mean": episode_expert_band_avoidance_steps_mean,
        "episode_expert_pre_emergency_slow_steps_mean": episode_expert_pre_emergency_slow_steps_mean,
        "episode_expert_recovering_steps_mean": episode_expert_recovering_steps_mean,
        "episode_expert_rejoin_actions_mean": episode_expert_rejoin_actions_mean,
        "episode_expert_rejoin_attempts_mean": episode_expert_rejoin_attempts_mean,
        "episode_expert_rejoin_rejected_mean": episode_expert_rejoin_rejected_mean,
        "episode_replans_mean": episode_replans_mean,
        "episode_replan_successes_mean": episode_replan_successes_mean,
        "episode_replan_failures_mean": episode_replan_failures_mean,
        "episode_replan_to_rejoin_successes_mean": episode_replan_to_rejoin_successes_mean,
        "episode_replan_to_goal_successes_mean": episode_replan_to_goal_successes_mean,
        "episode_replan_path_drift_triggers_mean": episode_replan_path_drift_triggers_mean,
        "episode_replan_low_progress_triggers_mean": episode_replan_low_progress_triggers_mean,
        "episode_replan_no_valid_triggers_mean": episode_replan_no_valid_triggers_mean,
    }


def compare_against_astar(rows: List[Dict[str, object]], method: str) -> Dict[str, object]:
    astar_by_episode = {int(r["episode"]): r for r in rows if r["method"] == "astar_only"}
    method_by_episode = {int(r["episode"]): r for r in rows if r["method"] == method}
    shared_episodes = sorted(set(astar_by_episode) & set(method_by_episode))

    rescued = []
    harmed = []
    both_success = []
    both_fail = []
    for episode in shared_episodes:
        astar_row = astar_by_episode[episode]
        method_row = method_by_episode[episode]
        astar_success = bool(astar_row["success"])
        method_success = bool(method_row["success"])
        seed = int(method_row["seed"])
        if method_success and not astar_success:
            rescued.append(seed)
        elif astar_success and not method_success:
            harmed.append(seed)
        elif astar_success and method_success:
            both_success.append(seed)
        else:
            both_fail.append(seed)

    n = max(len(shared_episodes), 1)
    return {
        "method": method,
        "baseline_method": "astar_only",
        "episodes": len(shared_episodes),
        "rescued_count": len(rescued),
        "harmed_count": len(harmed),
        "both_success_count": len(both_success),
        "both_fail_count": len(both_fail),
        "net_rescue_count": len(rescued) - len(harmed),
        "rescued_rate": float(len(rescued) / n),
        "harmed_rate": float(len(harmed) / n),
        "rescued_seeds": rescued,
        "harmed_seeds": harmed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare zero-residual A* baselines, Expert, and RL under the "
            "predictable-planning + disruptive-execution setup."
        )
    )
    parser.add_argument("--rl-model-path", required=True, help="Path to PPO model (with or without .zip suffix).")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes to compare.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed. Episode seed = seed + episode_idx.")
    parser.add_argument("--curriculum-stage", type=int, default=4, help="Curriculum stage used for distance range.")
    parser.add_argument(
        "--stress",
        choices=("normal", "hard", "extreme", "fragile"),
        default="normal",
        help="Hidden random-layer stress level used for robustness comparisons.",
    )
    parser.add_argument(
        "--rl-apas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the v2.5 true-dynamics APAS layer for A*+RL only (default: true).",
    )
    parser.add_argument(
        "--astar-apas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the v2.5 true-dynamics APAS layer for the zero-residual A* baseline.",
    )
    parser.add_argument(
        "--expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate the hand-written local avoidance expert on the same tasks.",
    )
    parser.add_argument(
        "--expert-apas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable APAS for the A*+Expert baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "compare_astar_rl_disruptive",
        help="Directory for raw CSV and summary JSON outputs.",
    )
    parser.add_argument(
        "--task-sampling-trials",
        type=int,
        default=180,
        help="Max trials used to sample one valid start/goal task.",
    )
    parser.add_argument(
        "--planner-max-steps",
        type=int,
        default=30000,
        help="A* search budget used during comparison task sampling and execution.",
    )
    parser.add_argument(
        "--apas-segment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable swept-segment core/risk checks inside APAS (default: true).",
    )
    parser.add_argument(
        "--replan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable v2.5 replan-to-rejoin recovery after stalled/invalid local progress (default: false).",
    )
    parser.add_argument(
        "--stale-waypoint-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable projection-based stale waypoint skipping for all controllers (default: true).",
    )
    parser.add_argument(
        "--risk-membrane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the expert's local risk membrane / band-avoidance observer (default: true).",
    )
    return parser


def _controller_label(method: str, *, apas: bool, stale_wp: bool, risk_membrane: bool) -> str:
    parts = [method]
    if apas:
        parts.append("APAS")
    if stale_wp:
        parts.append("waypoint-skip")
    if risk_membrane and "Expert" in method:
        parts.append("risk-membrane")
    return " + ".join(parts)


def main() -> int:
    args = build_parser().parse_args()
    model = _load_ppo_model(args.rl_model_path)

    config = SimulationConfig()
    config.curriculum_stage = int(args.curriculum_stage)
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"
    config.max_steps = int(args.planner_max_steps)
    config.v25_disruption_stress_level = str(args.stress)
    config.v25_apas_segment_check_enabled = bool(args.apas_segment_check)
    config.v25_replan_enabled = bool(args.replan)
    config.v25_stale_waypoint_skip_enabled = bool(args.stale_waypoint_skip)
    config.v25_risk_membrane_enabled = bool(args.risk_membrane)

    astar_label = _controller_label(
        "A* zero-residual",
        apas=bool(args.astar_apas),
        stale_wp=bool(args.stale_waypoint_skip),
        risk_membrane=False,
    )
    expert_label = _controller_label(
        "A* + Expert",
        apas=bool(args.expert_apas),
        stale_wp=bool(args.stale_waypoint_skip),
        risk_membrane=bool(args.risk_membrane),
    )
    rl_label = _controller_label(
        "A* + RL",
        apas=bool(args.rl_apas),
        stale_wp=bool(args.stale_waypoint_skip),
        risk_membrane=False,
    )

    rows: List[Dict[str, object]] = []
    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed + ep)
        task_cfg = copy.deepcopy(config)
        task_cfg.wind_seed = ep_seed
        task_cfg.max_steps = int(args.planner_max_steps)
        task_cfg.v25_disruption_stress_level = str(args.stress)
        print(f"[episode {ep:03d}] sampling seed={ep_seed}", flush=True)
        task = sample_task_for_seed(task_cfg, seed=ep_seed, max_trials=int(args.task_sampling_trials))

        astar_metrics = evaluate_astar_only(config, task, enable_apas=bool(args.astar_apas))
        expert_metrics = (
            evaluate_astar_plus_expert(config, task, enable_apas=bool(args.expert_apas))
            if bool(args.expert)
            else None
        )
        rl_metrics = evaluate_astar_plus_rl(config, task, model, enable_apas=bool(args.rl_apas))

        rows.append(
            {
                "episode": ep,
                "seed": ep_seed,
                "method": "astar_only",
                "controller_label": astar_label,
                "apas_enabled": bool(args.astar_apas),
                "stale_waypoint_skip_enabled": bool(args.stale_waypoint_skip),
                "risk_membrane_enabled": False,
                "start_x": task.start_xy[0],
                "start_y": task.start_xy[1],
                "goal_x": task.goal_xy[0],
                "goal_y": task.goal_xy[1],
                **astar_metrics,
            }
        )
        if expert_metrics is not None:
            rows.append(
                {
                    "episode": ep,
                    "seed": ep_seed,
                    "method": "astar_plus_expert",
                    "controller_label": expert_label,
                    "apas_enabled": bool(args.expert_apas),
                    "stale_waypoint_skip_enabled": bool(args.stale_waypoint_skip),
                    "risk_membrane_enabled": bool(args.risk_membrane),
                    "start_x": task.start_xy[0],
                    "start_y": task.start_xy[1],
                    "goal_x": task.goal_xy[0],
                    "goal_y": task.goal_xy[1],
                    **expert_metrics,
                }
            )
        rows.append(
            {
                "episode": ep,
                "seed": ep_seed,
                "method": "astar_plus_rl",
                "controller_label": rl_label,
                "apas_enabled": bool(args.rl_apas),
                "stale_waypoint_skip_enabled": bool(args.stale_waypoint_skip),
                "risk_membrane_enabled": False,
                "start_x": task.start_xy[0],
                "start_y": task.start_xy[1],
                "goal_x": task.goal_xy[0],
                "goal_y": task.goal_xy[1],
                **rl_metrics,
            }
        )

        print(
            f"[episode {ep:03d}] seed={ep_seed} "
            f"{astar_label} success={astar_metrics['success']} reason={astar_metrics['terminated_reason']} | "
            + (
                f"{expert_label} success={expert_metrics['success']} reason={expert_metrics['terminated_reason']} | "
                if expert_metrics is not None
                else ""
            )
            +
            f"{rl_label} success={rl_metrics['success']} reason={rl_metrics['terminated_reason']}",
            flush=True,
        )

    astar_summary = summarize_rows(rows, "astar_only")
    expert_summary = summarize_rows(rows, "astar_plus_expert") if bool(args.expert) else None
    rl_summary = summarize_rows(rows, "astar_plus_rl")
    expert_vs_astar = compare_against_astar(rows, "astar_plus_expert") if bool(args.expert) else None
    rl_vs_astar = compare_against_astar(rows, "astar_plus_rl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = output_dir / f"compare_raw_{timestamp}.csv"
    summary_json_path = output_dir / f"compare_summary_{timestamp}.json"

    with raw_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "meta": {
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "curriculum_stage": int(args.curriculum_stage),
            "stress": str(args.stress),
            "rl_model_path": str(args.rl_model_path),
            "rl_apas": bool(args.rl_apas),
            "astar_apas": bool(args.astar_apas),
            "expert": bool(args.expert),
            "expert_apas": bool(args.expert_apas),
            "apas_segment_check": bool(args.apas_segment_check),
            "replan": bool(args.replan),
            "stale_waypoint_skip": bool(args.stale_waypoint_skip),
            "risk_membrane": bool(args.risk_membrane),
            "planner_max_steps": int(args.planner_max_steps),
        },
        "controller_labels": {
            "astar_only": astar_label,
            **({"astar_plus_expert": expert_label} if expert_summary is not None else {}),
            "astar_plus_rl": rl_label,
        },
        "zero_residual_astar_baseline": astar_summary,
        "astar_only": astar_summary,
        **({"astar_plus_expert": expert_summary} if expert_summary is not None else {}),
        "astar_plus_rl": rl_summary,
        **({"expert_vs_astar": expert_vs_astar} if expert_vs_astar is not None else {}),
        "rl_vs_astar": rl_vs_astar,
        "raw_csv": str(raw_csv_path),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Comparison Summary ===")
    print(f"{astar_label}: {astar_summary}")
    if expert_summary is not None:
        print(f"{expert_label}: {expert_summary}")
        print(f"Expert vs {astar_label}: {expert_vs_astar}")
    print(f"{rl_label}: {rl_summary}")
    print(f"RL vs {astar_label}:     {rl_vs_astar}")
    print(f"raw_csv:    {raw_csv_path}")
    print(f"summary:    {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
