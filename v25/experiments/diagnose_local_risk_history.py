from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from v25.experiments.compare_astar_rl_disruptive import sample_task_for_seed
from v25.rl_env_disruptive import GuidedDroneEnvV25


def _angle_wrap_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return float(angle)


def _local_oracle_diagnostics(env: GuidedDroneEnvV25) -> Dict[str, float]:
    features = env._circle_oracle_features()
    origin = np.asarray(env.current_pos[:2], dtype=float)
    heading_deg = float(env.current_heading)
    radius_m = float(env.config.v25_radar_radius_m)

    max_danger = -1.0
    max_dist = radius_m
    max_bearing = 0.0
    weighted_vec = np.zeros(2, dtype=float)
    weighted_sum = 0.0
    danger_samples = 0

    if env.disruptions is not None:
        for point in env._circle_oracle_sample_points():
            offset = np.asarray(point - origin, dtype=float)
            dist = float(np.linalg.norm(offset))
            risk_bonus = float(env.disruptions.risk_bonus_at(float(point[0]), float(point[1]), env.current_time))
            storm_danger = float(
                env.disruptions.destructive_storm.danger_at(float(point[0]), float(point[1]), env.current_time)
            )
            danger = max(risk_bonus, storm_danger)
            if danger > max_danger:
                max_danger = danger
                max_dist = dist
                max_bearing = _angle_wrap_deg(math.degrees(math.atan2(offset[1], offset[0])) - heading_deg)
            if danger > 1e-6:
                weighted_vec += offset * danger
                weighted_sum += danger
                danger_samples += 1

    if weighted_sum > 1e-6 and np.linalg.norm(weighted_vec) > 1e-6:
        centroid_vec = weighted_vec / weighted_sum
        centroid_bearing = _angle_wrap_deg(math.degrees(math.atan2(centroid_vec[1], centroid_vec[0])) - heading_deg)
        centroid_dist = float(np.linalg.norm(centroid_vec))
    else:
        centroid_bearing = 0.0
        centroid_dist = radius_m

    return {
        "max_danger": float(features[0]),
        "mean_danger": float(features[1]),
        "nearest_closeness": float(features[2]),
        "nearest_forward": float(features[3]),
        "nearest_left": float(features[4]),
        "centroid_forward": float(features[5]),
        "centroid_left": float(features[6]),
        "mean_wind_forward": float(features[7]),
        "mean_wind_left": float(features[8]),
        "max_wind_forward": float(features[9]),
        "max_wind_left": float(features[10]),
        "forward_danger": float(features[11]),
        "max_sample_danger": float(max(max_danger, 0.0)),
        "max_sample_distance_m": float(max_dist),
        "max_sample_bearing_deg": float(max_bearing),
        "centroid_distance_m": float(centroid_dist),
        "centroid_bearing_deg": float(centroid_bearing),
        "danger_sample_count": int(danger_samples),
    }


def _make_env(config: SimulationConfig, seed: int, start_xy, goal_xy) -> GuidedDroneEnvV25:
    run_cfg = copy.deepcopy(config)
    run_cfg.wind_seed = int(seed)
    run_cfg.enable_single_agent_gusts = False
    run_cfg.enable_random_gusts = False
    run_cfg.planner_time_mode = "4d"
    env = GuidedDroneEnvV25(run_cfg)
    env.reset(seed=int(seed), options={"start_xy": start_xy, "goal_xy": goal_xy})
    return env


def _run_method(config: SimulationConfig, seed: int, method: str, max_steps: int, sampling_trials: int):
    task_cfg = copy.deepcopy(config)
    task_cfg.wind_seed = int(seed)
    task = sample_task_for_seed(task_cfg, seed=int(seed), max_trials=int(sampling_trials))
    env = _make_env(config, seed, task.start_xy, task.goal_xy)

    rows = []
    terminated = False
    truncated = False
    final_info = {"is_success": False, "terminated_reason": "unknown"}
    previous_diag = None

    while not (terminated or truncated) and env.current_step < max_steps:
        diag = _local_oracle_diagnostics(env)
        if method == "expert":
            action = env.local_avoidance_expert_action()
            expert_mode = str(getattr(env, "last_expert_mode", "unknown"))
        elif method == "astar":
            action = np.zeros(3, dtype=np.float32)
            expert_mode = "inactive"
        else:
            raise ValueError(f"Unsupported method: {method}")

        delta = {}
        if previous_diag is None:
            for key in ("max_danger", "forward_danger", "nearest_closeness", "max_sample_bearing_deg"):
                delta[f"delta_{key}"] = 0.0
        else:
            delta["delta_max_danger"] = diag["max_danger"] - previous_diag["max_danger"]
            delta["delta_forward_danger"] = diag["forward_danger"] - previous_diag["forward_danger"]
            delta["delta_nearest_closeness"] = diag["nearest_closeness"] - previous_diag["nearest_closeness"]
            delta["delta_max_sample_bearing_deg"] = _angle_wrap_deg(
                diag["max_sample_bearing_deg"] - previous_diag["max_sample_bearing_deg"]
            )

        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "step": int(env.current_step),
                "time_s": float(env.current_time),
                "x_m": float(env.current_pos[0]),
                "y_m": float(env.current_pos[1]),
                "z_m": float(env.current_pos[2]),
                "heading_deg": float(env.current_heading),
                "wp_idx": int(env.current_wp_idx),
                "action_heading": float(action[0]),
                "action_speed": float(action[1]),
                "action_agl": float(action[2]),
                "expert_mode": expert_mode,
                "expert_rejoin_actions_so_far": int(getattr(env, "episode_expert_rejoin_actions", 0)),
                "expert_rejoin_attempts_so_far": int(getattr(env, "episode_expert_rejoin_attempts", 0)),
                "expert_rejoin_rejected_so_far": int(getattr(env, "episode_expert_rejoin_rejected", 0)),
                **diag,
                **delta,
            }
        )
        previous_diag = diag
        _, _, terminated, truncated, final_info = env.step(action)

    summary = {
        "seed": int(seed),
        "method": method,
        "success": bool(final_info.get("is_success", False)),
        "terminated_reason": str(final_info.get("terminated_reason", "unknown")),
        "steps": int(env.current_step),
        "max_danger_peak": float(max((r["max_danger"] for r in rows), default=0.0)),
        "forward_danger_peak": float(max((r["forward_danger"] for r in rows), default=0.0)),
        "nearest_closeness_peak": float(max((r["nearest_closeness"] for r in rows), default=0.0)),
        "positive_forward_trend_steps": int(sum(1 for r in rows if r["delta_forward_danger"] > 0.02)),
        "positive_max_trend_steps": int(sum(1 for r in rows if r["delta_max_danger"] > 0.02)),
        "bearing_motion_mean_abs_deg": float(
            np.mean([abs(float(r["delta_max_sample_bearing_deg"])) for r in rows[1:]]) if len(rows) > 1 else 0.0
        ),
        "expert_normal_steps": int(getattr(env, "episode_expert_normal_steps", 0)),
        "expert_cautious_steps": int(getattr(env, "episode_expert_cautious_steps", 0)),
        "expert_cautious_trend_steps": int(getattr(env, "episode_expert_cautious_trend_steps", 0)),
        "expert_avoiding_steps": int(getattr(env, "episode_expert_avoiding_steps", 0)),
        "expert_emergency_steps": int(getattr(env, "episode_expert_emergency_steps", 0)),
        "expert_recovering_steps": int(getattr(env, "episode_expert_recovering_steps", 0)),
        "expert_rejoin_actions": int(getattr(env, "episode_expert_rejoin_actions", 0)),
        "expert_rejoin_attempts": int(getattr(env, "episode_expert_rejoin_attempts", 0)),
        "expert_rejoin_rejected": int(getattr(env, "episode_expert_rejoin_rejected", 0)),
    }
    return rows, summary


def _parse_seeds(values: Iterable[str]) -> List[int]:
    seeds: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose whether v2.5 local random-layer observations contain useful history/trend signals."
    )
    parser.add_argument("--seeds", nargs="+", default=["43", "59", "72", "42"], help="Seeds or comma-separated seeds.")
    parser.add_argument("--methods", nargs="+", choices=("astar", "expert"), default=["astar", "expert"])
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--planner-max-steps", type=int, default=15000)
    parser.add_argument("--task-sampling-trials", type=int, default=180)
    parser.add_argument("--max-episode-steps", type=int, default=350)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "local_risk_history")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)

    config = SimulationConfig()
    config.curriculum_stage = int(args.curriculum_stage)
    config.max_steps = int(args.planner_max_steps)
    config.v25_disruption_stress_level = str(args.stress)
    config.v25_sensor_mode = "circle_oracle"
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"

    all_rows = []
    summaries = []
    for seed in seeds:
        for method in args.methods:
            print(f"[diagnose] seed={seed} method={method}", flush=True)
            rows, summary = _run_method(
                config=config,
                seed=seed,
                method=method,
                max_steps=int(args.max_episode_steps),
                sampling_trials=int(args.task_sampling_trials),
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"  success={summary['success']} reason={summary['terminated_reason']} "
                f"max={summary['max_danger_peak']:.3f} forward={summary['forward_danger_peak']:.3f} "
                f"trend_steps={summary['positive_forward_trend_steps']}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = output_dir / f"local_risk_history_raw_{timestamp}.csv"
    summary_json_path = output_dir / f"local_risk_history_summary_{timestamp}.json"

    if all_rows:
        with raw_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    payload = {
        "meta": {
            "seeds": seeds,
            "methods": list(args.methods),
            "curriculum_stage": int(args.curriculum_stage),
            "stress": str(args.stress),
            "max_episode_steps": int(args.max_episode_steps),
        },
        "summaries": summaries,
        "raw_csv": str(raw_csv_path),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"raw_csv: {raw_csv_path}")
    print(f"summary: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
