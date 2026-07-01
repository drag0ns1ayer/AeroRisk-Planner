from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from v25.experiments.compare_astar_rl_disruptive import sample_task_for_seed
from v25.experiments.diagnose_local_risk_history import _local_oracle_diagnostics
from v25.rl_env_disruptive import GuidedDroneEnvV25


def _parse_seeds(values: Iterable[str]) -> List[int]:
    seeds: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    return seeds


def _make_env(config: SimulationConfig, seed: int, start_xy, goal_xy, enable_apas: bool) -> GuidedDroneEnvV25:
    run_cfg = copy.deepcopy(config)
    run_cfg.wind_seed = int(seed)
    run_cfg.enable_single_agent_gusts = False
    run_cfg.enable_random_gusts = False
    run_cfg.planner_time_mode = "4d"
    run_cfg.rl_enable_apas = bool(enable_apas)
    env = GuidedDroneEnvV25(run_cfg)
    env.reset(seed=int(seed), options={"start_xy": start_xy, "goal_xy": goal_xy})
    return env


def _method_settings(method: str) -> tuple[str, bool]:
    if method == "astar":
        return "astar", False
    if method == "astar_apas":
        return "astar", True
    if method == "expert":
        return "expert", False
    if method == "expert_apas":
        return "expert", True
    if method == "rl":
        return "rl", False
    if method == "rl_apas":
        return "rl", True
    raise ValueError(f"Unsupported method: {method}")


def _load_ppo_model(model_path: str | None):
    if not model_path:
        return None
    path = Path(model_path)
    if path.suffix == ".zip":
        load_path = str(path)[:-4]
    else:
        load_path = str(path)
    return PPO.load(load_path, device="cpu")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _dist_to_current_waypoint(env: GuidedDroneEnvV25, pos_xy) -> float:
    if not getattr(env, "global_astar_path", None):
        return 0.0
    wp = env._path_point(int(env.current_wp_idx))
    return float(np.linalg.norm(np.asarray(pos_xy, dtype=float) - wp[:2]))


def _run_method(
    config: SimulationConfig,
    seed: int,
    method: str,
    last_steps: int,
    sampling_trials: int,
    rl_model=None,
) -> tuple[list[dict], dict]:
    task_cfg = copy.deepcopy(config)
    task_cfg.wind_seed = int(seed)
    task = sample_task_for_seed(task_cfg, seed=int(seed), max_trials=int(sampling_trials))
    controller, enable_apas = _method_settings(method)
    if controller == "rl" and rl_model is None:
        raise ValueError(f"Method {method} requires --rl-model-path.")
    env = _make_env(config, seed, task.start_xy, task.goal_xy, enable_apas=enable_apas)
    obs, _ = env._get_obs(), {}

    trace_buffer: deque[dict] = deque(maxlen=int(last_steps))
    terminated = False
    truncated = False
    final_info = {"is_success": False, "terminated_reason": "unknown"}
    prev_goal_dist = float(np.linalg.norm(env.goal_pos[:2] - env.current_pos[:2]))
    path_distance_m = 0.0
    prev_pos = np.asarray(env.current_pos, dtype=float).copy()

    while not (terminated or truncated):
        diag = _local_oracle_diagnostics(env)
        pre_path_error = float(env._estimate_local_path_error(env.current_pos[:2]))
        pre_goal_dist = float(np.linalg.norm(env.goal_pos[:2] - env.current_pos[:2]))
        pre_wp_idx = int(env.current_wp_idx)
        pre_nearest_path_idx = int(env._nearest_path_index(env.current_pos[:2]))
        pre_dist_to_wp = _dist_to_current_waypoint(env, env.current_pos[:2])
        pre_time = float(env.current_time)
        pre_pos = np.asarray(env.current_pos, dtype=float).copy()

        if controller == "expert":
            action = env.local_avoidance_expert_action()
            proposed_mode = str(getattr(env, "last_expert_mode", "unknown"))
        elif controller == "rl":
            action, _ = rl_model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            proposed_mode = "rl_policy"
        else:
            action = np.zeros(3, dtype=np.float32)
            proposed_mode = "inactive"

        obs, _, terminated, truncated, final_info = env.step(action)

        curr_pos = np.asarray(env.current_pos, dtype=float)
        step_distance = float(np.linalg.norm(curr_pos - prev_pos))
        path_distance_m += step_distance
        prev_pos = curr_pos.copy()
        post_goal_dist = float(final_info.get("goal_dist_m", np.linalg.norm(env.goal_pos[:2] - env.current_pos[:2])))
        progress_m = float(prev_goal_dist - post_goal_dist)
        prev_goal_dist = post_goal_dist
        post_nearest_path_idx = int(env._nearest_path_index(env.current_pos[:2]))
        post_dist_to_wp = _dist_to_current_waypoint(env, env.current_pos[:2])

        trace_buffer.append(
            {
                "seed": int(seed),
                "method": method,
                "step": int(env.current_step),
                "pre_time_s": pre_time,
                "post_time_s": float(env.current_time),
                "pre_x_m": float(pre_pos[0]),
                "pre_y_m": float(pre_pos[1]),
                "pre_z_m": float(pre_pos[2]),
                "post_x_m": float(env.current_pos[0]),
                "post_y_m": float(env.current_pos[1]),
                "post_z_m": float(env.current_pos[2]),
                "step_distance_m": step_distance,
                "progress_m": progress_m,
                "pre_goal_dist_m": pre_goal_dist,
                "post_goal_dist_m": post_goal_dist,
                "pre_path_error_m": pre_path_error,
                "post_path_error_m": _safe_float(final_info.get("path_error_m", 0.0)),
                "pre_wp_idx": pre_wp_idx,
                "post_wp_idx": int(env.current_wp_idx),
                "wp_idx_delta": int(env.current_wp_idx) - pre_wp_idx,
                "pre_nearest_path_idx": pre_nearest_path_idx,
                "post_nearest_path_idx": post_nearest_path_idx,
                "nearest_path_idx_delta": post_nearest_path_idx - pre_nearest_path_idx,
                "pre_dist_to_wp_m": pre_dist_to_wp,
                "post_dist_to_wp_m": post_dist_to_wp,
                "dist_to_wp_delta_m": pre_dist_to_wp - post_dist_to_wp,
                "action_heading": float(action[0]),
                "action_speed": float(action[1]),
                "action_agl": float(action[2]),
                "proposed_expert_mode": proposed_mode,
                "executed_expert_mode": str(final_info.get("expert_mode", proposed_mode)),
                "controller_mode": proposed_mode,
                "p_crash": _safe_float(final_info.get("p_crash", 0.0)),
                "power_w": _safe_float(final_info.get("power_w", 0.0)),
                "airspeed_mps": _safe_float(final_info.get("airspeed_mps", 0.0)),
                "ground_speed_mps": _safe_float(final_info.get("ground_speed_mps", 0.0)),
                "base_heading_deg": _safe_float(final_info.get("base_heading_deg", 0.0)),
                "base_airspeed_mps": _safe_float(final_info.get("base_airspeed_mps", 0.0)),
                "base_agl_m": _safe_float(final_info.get("base_agl_m", 0.0)),
                "commanded_heading_deg": _safe_float(final_info.get("commanded_heading_deg", 0.0)),
                "commanded_airspeed_mps": _safe_float(final_info.get("commanded_airspeed_mps", 0.0)),
                "commanded_agl_m": _safe_float(final_info.get("commanded_agl_m", 0.0)),
                "do_no_harm_active": bool(final_info.get("do_no_harm_active", False)),
                "do_no_harm_reason": str(final_info.get("do_no_harm_reason", "none")),
                "do_no_harm_recent_progress_sum": _safe_float(
                    final_info.get("do_no_harm_recent_progress_sum", 0.0)
                ),
                "do_no_harm_recent_risk_delta": _safe_float(final_info.get("do_no_harm_recent_risk_delta", 0.0)),
                "do_no_harm_recent_apas_interventions": int(
                    final_info.get("do_no_harm_recent_apas_interventions", 0)
                ),
                "do_no_harm_recent_segment_rejections": int(
                    final_info.get("do_no_harm_recent_segment_rejections", 0)
                ),
                "do_no_harm_recent_no_valid_candidates": int(
                    final_info.get("do_no_harm_recent_no_valid_candidates", 0)
                ),
                "do_no_harm_cooldown_steps_remaining": int(
                    final_info.get("do_no_harm_cooldown_steps_remaining", 0)
                ),
                "local_hazard_need": _safe_float(final_info.get("local_hazard_need", 0.0)),
                "local_hazard_max_danger": _safe_float(final_info.get("local_hazard_max_danger", 0.0)),
                "local_hazard_forward_danger": _safe_float(final_info.get("local_hazard_forward_danger", 0.0)),
                "local_hazard_trend_need": _safe_float(final_info.get("local_hazard_trend_need", 0.0)),
                "risk_membrane_wall_ahead": _safe_float(final_info.get("risk_membrane_wall_ahead", 0.0)),
                "risk_membrane_no_escape_gap": _safe_float(final_info.get("risk_membrane_no_escape_gap", 0.0)),
                "risk_membrane_front_blocked_width_deg": _safe_float(
                    final_info.get("risk_membrane_front_blocked_width_deg", 0.0)
                ),
                "risk_membrane_best_gap_angle_deg": _safe_float(
                    final_info.get("risk_membrane_best_gap_angle_deg", 0.0)
                ),
                "risk_membrane_best_gap_width_deg": _safe_float(
                    final_info.get("risk_membrane_best_gap_width_deg", 0.0)
                ),
                "risk_membrane_max_extended_risk": _safe_float(
                    final_info.get("risk_membrane_max_extended_risk", 0.0)
                ),
                "obs_max_danger": float(diag.get("max_danger", 0.0)),
                "obs_forward_danger": float(diag.get("forward_danger", 0.0)),
                "obs_nearest_closeness": float(diag.get("nearest_closeness", 0.0)),
                "obs_max_sample_distance_m": float(diag.get("max_sample_distance_m", 0.0)),
                "obs_max_sample_bearing_deg": float(diag.get("max_sample_bearing_deg", 0.0)),
                "apas_intervened": bool(final_info.get("apas_intervened", False)),
                "apas_heading_offset_deg": _safe_float(final_info.get("apas_heading_offset_deg", 0.0)),
                "apas_speed_reduction_mps": _safe_float(final_info.get("apas_speed_reduction_mps", 0.0)),
                "apas_agl_increment_m": _safe_float(final_info.get("apas_agl_increment_m", 0.0)),
                "apas_segment_rejections": int(final_info.get("apas_segment_rejections", 0)),
                "apas_endpoint_rejections": int(final_info.get("apas_endpoint_rejections", 0)),
                "apas_no_valid_candidate": bool(final_info.get("apas_no_valid_candidate", False)),
                "apas_segment_max_risk_bonus": _safe_float(final_info.get("apas_segment_max_risk_bonus", 0.0)),
                "apas_segment_core_hit": bool(final_info.get("apas_segment_core_hit", False)),
                "replan_triggered": bool(final_info.get("replan_triggered", False)),
                "replan_success": bool(final_info.get("replan_success", False)),
                "replan_reason": str(final_info.get("replan_reason", "none")),
                "last_replan_event": str(final_info.get("last_replan_event", "none")),
                "consecutive_low_progress_steps": int(final_info.get("consecutive_low_progress_steps", 0)),
                "consecutive_replan_low_progress_steps": int(
                    final_info.get("consecutive_replan_low_progress_steps", 0)
                ),
                "apas_no_valid_linger_steps": int(final_info.get("apas_no_valid_linger_steps", 0)),
                "episode_stale_waypoint_skips": int(final_info.get("episode_stale_waypoint_skips", 0)),
                "episode_stale_waypoint_skip_delta": int(final_info.get("episode_stale_waypoint_skip_delta", 0)),
                "terminated_after_step": bool(terminated or truncated),
                "terminated_reason_after_step": str(final_info.get("terminated_reason", "")) if terminated or truncated else "",
            }
        )

    success = bool(final_info.get("is_success", False))
    trace_rows = [] if success else list(trace_buffer)
    reason = str(final_info.get("terminated_reason", "unknown"))
    summary = {
        "seed": int(seed),
        "method": method,
        "success": success,
        "terminated_reason": reason,
        "steps": int(env.current_step),
        "mission_time_s": float(env.current_time),
        "path_distance_m": float(path_distance_m),
        "energy_used_j": float(config.battery_capacity_j - _safe_float(final_info.get("energy_remaining_j", env.energy_remaining))),
        "avg_risk": float(np.mean(env.telemetry_risk)) if env.telemetry_risk else 0.0,
        "peak_risk": float(env.telemetry_max_p_crash) if env.telemetry_risk else 0.0,
        "last_trace_rows": len(trace_rows),
        "apas_interventions": int(getattr(env, "episode_apas_interventions", 0)),
        "apas_segment_rejections": int(getattr(env, "episode_apas_segment_rejections", 0)),
        "apas_no_valid_candidates": int(getattr(env, "episode_apas_no_valid_candidates", 0)),
        "stale_waypoint_skips": int(getattr(env, "episode_stale_waypoint_skips", 0)),
        "stale_waypoint_skip_delta": int(getattr(env, "episode_stale_waypoint_skip_delta", 0)),
        "replans": int(getattr(env, "episode_replans", 0)),
        "replan_successes": int(getattr(env, "episode_replan_successes", 0)),
        "replan_failures": int(getattr(env, "episode_replan_failures", 0)),
        "replan_path_drift_triggers": int(getattr(env, "episode_replan_path_drift_triggers", 0)),
        "replan_low_progress_triggers": int(getattr(env, "episode_replan_low_progress_triggers", 0)),
        "replan_no_valid_triggers": int(getattr(env, "episode_replan_no_valid_triggers", 0)),
        "eval_maneuver_extra_energy_j": float(getattr(env, "episode_eval_maneuver_extra_energy_j", 0.0)),
        "eval_safety_intervention_burden": float(getattr(env, "episode_eval_safety_intervention_burden", 0.0)),
        "eval_adjusted_energy_j": float(getattr(env, "episode_eval_adjusted_energy_j", 0.0)),
        "expert_band_avoidance_steps": int(getattr(env, "episode_expert_band_avoidance_steps", 0)),
        "expert_pre_emergency_slow_steps": int(getattr(env, "episode_expert_pre_emergency_slow_steps", 0)),
        "do_no_harm_events": int(getattr(env, "episode_do_no_harm_events", 0)),
        "do_no_harm_suppressed_steps": int(getattr(env, "episode_do_no_harm_suppressed_steps", 0)),
        "do_no_harm_cooldown_steps": int(getattr(env, "episode_do_no_harm_cooldown_steps", 0)),
        "trace_included": not success,
    }
    return trace_rows, summary


def _summarize(summaries: list[dict]) -> dict:
    payload = {}
    methods = sorted({str(row["method"]) for row in summaries})
    for method in methods:
        rows = [row for row in summaries if str(row["method"]) == method]
        n = max(len(rows), 1)
        failures = [row for row in rows if not bool(row["success"])]
        payload[method] = {
            "episodes": len(rows),
            "success_rate": sum(1 for row in rows if bool(row["success"])) / n,
            "failure_count": len(failures),
            "terminated_reasons": dict(Counter(str(row["terminated_reason"]) for row in rows)),
            "traced_failures": sum(1 for row in rows if bool(row.get("trace_included", False))),
            "apas_interventions_mean": sum(float(row.get("apas_interventions", 0.0)) for row in rows) / n,
            "apas_no_valid_candidates_mean": sum(float(row.get("apas_no_valid_candidates", 0.0)) for row in rows) / n,
            "stale_waypoint_skips_mean": sum(float(row.get("stale_waypoint_skips", 0.0)) for row in rows) / n,
            "stale_waypoint_skip_delta_mean": sum(
                float(row.get("stale_waypoint_skip_delta", 0.0)) for row in rows
            )
            / n,
            "expert_band_avoidance_steps_mean": sum(
                float(row.get("expert_band_avoidance_steps", 0.0)) for row in rows
            )
            / n,
            "expert_pre_emergency_slow_steps_mean": sum(
                float(row.get("expert_pre_emergency_slow_steps", 0.0)) for row in rows
            )
            / n,
            "do_no_harm_events_mean": sum(float(row.get("do_no_harm_events", 0.0)) for row in rows) / n,
            "do_no_harm_suppressed_steps_mean": sum(
                float(row.get("do_no_harm_suppressed_steps", 0.0)) for row in rows
            )
            / n,
            "do_no_harm_cooldown_steps_mean": sum(
                float(row.get("do_no_harm_cooldown_steps", 0.0)) for row in rows
            )
            / n,
            "replans_mean": sum(float(row.get("replans", 0.0)) for row in rows) / n,
            "eval_adjusted_energy_j_mean": sum(float(row.get("eval_adjusted_energy_j", 0.0)) for row in rows) / n,
        }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record final-step traces for failed v2.5 episodes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("astar", "expert", "rl", "astar_apas", "expert_apas", "rl_apas"),
        default=["astar_apas", "expert_apas"],
    )
    parser.add_argument(
        "--rl-model-path",
        default=None,
        help="PPO model path used by rl/rl_apas methods, with or without .zip suffix.",
    )
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--planner-max-steps", type=int, default=15000)
    parser.add_argument("--task-sampling-trials", type=int, default=180)
    parser.add_argument("--last-steps", type=int, default=50)
    parser.add_argument(
        "--apas-segment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable swept-segment APAS checks (default: true).",
    )
    parser.add_argument(
        "--replan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable experimental replan-to-rejoin recovery (default: false).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "failure_traces")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = _parse_seeds(args.seeds) if args.seeds else [int(args.seed + idx) for idx in range(int(args.episodes))]
    rl_model = _load_ppo_model(args.rl_model_path) if any(str(m).startswith("rl") for m in args.methods) else None

    config = SimulationConfig()
    config.curriculum_stage = int(args.curriculum_stage)
    config.max_steps = int(args.planner_max_steps)
    config.v25_disruption_stress_level = str(args.stress)
    config.v25_sensor_mode = "circle_oracle"
    config.v25_apas_segment_check_enabled = bool(args.apas_segment_check)
    config.v25_replan_enabled = bool(args.replan)
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"

    all_traces: list[dict] = []
    summaries: list[dict] = []
    for seed in seeds:
        for method in args.methods:
            print(f"[failure-trace] seed={seed} method={method}", flush=True)
            trace_rows, summary = _run_method(
                config=config,
                seed=int(seed),
                method=str(method),
                last_steps=int(args.last_steps),
                sampling_trials=int(args.task_sampling_trials),
                rl_model=rl_model,
            )
            summaries.append(summary)
            all_traces.extend(trace_rows)
            print(
                f"  success={summary['success']} reason={summary['terminated_reason']} "
                f"trace_rows={summary['last_trace_rows']}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_csv_path = output_dir / f"failure_trace_episodes_{timestamp}.csv"
    trace_csv_path = output_dir / f"failure_trace_steps_{timestamp}.csv"
    summary_json_path = output_dir / f"failure_trace_summary_{timestamp}.json"

    if summaries:
        with episode_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    if all_traces:
        with trace_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_traces[0].keys()))
            writer.writeheader()
            writer.writerows(all_traces)

    payload = {
        "meta": {
            "seeds": seeds,
            "methods": list(args.methods),
            "curriculum_stage": int(args.curriculum_stage),
            "stress": str(args.stress),
            "last_steps": int(args.last_steps),
            "apas_segment_check": bool(args.apas_segment_check),
            "replan": bool(args.replan),
        },
        "summary": _summarize(summaries),
        "episode_csv": str(episode_csv_path),
        "trace_csv": str(trace_csv_path),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Failure Trace Summary ===")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"episode_csv: {episode_csv_path}")
    print(f"trace_csv:   {trace_csv_path}")
    print(f"summary:     {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
