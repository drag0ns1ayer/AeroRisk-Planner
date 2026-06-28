from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

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


def _storm_state(env: GuidedDroneEnvV25, pos_xyz: np.ndarray | None = None, t_s: float | None = None) -> dict:
    if env.disruptions is None:
        return {
            "storm_active": False,
            "storm_center_x": "",
            "storm_center_y": "",
            "storm_radius_m": 0.0,
            "storm_core_radius_m": 0.0,
            "storm_halo_radius_m": 0.0,
            "dist_to_center_m": "",
            "core_margin_m": "",
            "halo_margin_m": "",
            "exact_danger": 0.0,
            "exact_risk_bonus": 0.0,
            "exact_core_hit": False,
        }

    storm = env.disruptions.destructive_storm
    pos = np.asarray(env.current_pos if pos_xyz is None else pos_xyz, dtype=float)
    time_s = float(env.current_time if t_s is None else t_s)
    center = storm.center_at(time_s)
    if center is None:
        return {
            "storm_active": False,
            "storm_center_x": "",
            "storm_center_y": "",
            "storm_radius_m": float(storm.radius_m),
            "storm_core_radius_m": float(storm.core_radius_m),
            "storm_halo_radius_m": float(storm.halo_radius_m),
            "dist_to_center_m": "",
            "core_margin_m": "",
            "halo_margin_m": "",
            "exact_danger": 0.0,
            "exact_risk_bonus": 0.0,
            "exact_core_hit": False,
        }

    dist = float(np.linalg.norm(pos[:2] - np.asarray(center, dtype=float)))
    return {
        "storm_active": True,
        "storm_center_x": float(center[0]),
        "storm_center_y": float(center[1]),
        "storm_radius_m": float(storm.radius_m),
        "storm_core_radius_m": float(storm.core_radius_m),
        "storm_halo_radius_m": float(storm.halo_radius_m),
        "dist_to_center_m": dist,
        "core_margin_m": float(dist - storm.core_radius_m),
        "halo_margin_m": float(dist - storm.halo_radius_m),
        "exact_danger": float(storm.danger_at(float(pos[0]), float(pos[1]), time_s)),
        "exact_risk_bonus": float(env.disruptions.risk_bonus_at(float(pos[0]), float(pos[1]), time_s)),
        "exact_core_hit": bool(env.disruptions.core_hit(float(pos[0]), float(pos[1]), time_s)),
    }


def _sample_coverage(env: GuidedDroneEnvV25, t_s: float | None = None) -> dict:
    if env.disruptions is None:
        return {
            "sample_core_hits": 0,
            "sample_danger_hits": 0,
            "sample_min_core_margin_m": "",
            "sample_min_center_dist_m": "",
        }

    storm = env.disruptions.destructive_storm
    time_s = float(env.current_time if t_s is None else t_s)
    center = storm.center_at(time_s)
    if center is None:
        return {
            "sample_core_hits": 0,
            "sample_danger_hits": 0,
            "sample_min_core_margin_m": "",
            "sample_min_center_dist_m": "",
        }

    core_hits = 0
    danger_hits = 0
    min_dist = float("inf")
    for point in env._circle_oracle_sample_points():
        x = float(point[0])
        y = float(point[1])
        dist = float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(center, dtype=float)))
        min_dist = min(min_dist, dist)
        core_hits += int(env.disruptions.core_hit(x, y, time_s))
        danger_hits += int(storm.danger_at(x, y, time_s) >= 0.05)

    return {
        "sample_core_hits": int(core_hits),
        "sample_danger_hits": int(danger_hits),
        "sample_min_core_margin_m": float(min_dist - storm.core_radius_m),
        "sample_min_center_dist_m": float(min_dist),
    }


def _segment_core_probe(
    env: GuidedDroneEnvV25,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    t_s: float,
    samples: int,
) -> dict:
    if env.disruptions is None:
        return {"segment_core_hit": False, "segment_min_core_margin_m": "", "segment_max_danger": 0.0}

    storm = env.disruptions.destructive_storm
    center = storm.center_at(float(t_s))
    if center is None:
        return {"segment_core_hit": False, "segment_min_core_margin_m": "", "segment_max_danger": 0.0}

    core_hit = False
    min_margin = float("inf")
    max_danger = 0.0
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    for idx in range(max(1, int(samples)) + 1):
        alpha = idx / max(1, int(samples))
        point = (1.0 - alpha) * start + alpha * end
        x = float(point[0])
        y = float(point[1])
        dist = float(np.linalg.norm(point[:2] - np.asarray(center, dtype=float)))
        min_margin = min(min_margin, dist - float(storm.core_radius_m))
        max_danger = max(max_danger, float(storm.danger_at(x, y, float(t_s))))
        core_hit = core_hit or bool(env.disruptions.core_hit(x, y, float(t_s)))

    return {
        "segment_core_hit": bool(core_hit),
        "segment_min_core_margin_m": float(min_margin),
        "segment_max_danger": float(max_danger),
    }


def _method_flags(method: str) -> tuple[str, bool]:
    if method == "astar":
        return "astar", False
    if method == "expert":
        return "expert", False
    if method == "astar_apas":
        return "astar", True
    if method == "expert_apas":
        return "expert", True
    raise ValueError(f"Unsupported method: {method}")


def _run_method(
    config: SimulationConfig,
    seed: int,
    method: str,
    max_steps: int,
    sampling_trials: int,
    segment_samples: int,
) -> tuple[list[dict], dict]:
    controller, enable_apas = _method_flags(method)
    task_cfg = copy.deepcopy(config)
    task_cfg.wind_seed = int(seed)
    task = sample_task_for_seed(task_cfg, seed=int(seed), max_trials=int(sampling_trials))
    env = _make_env(config, seed, task.start_xy, task.goal_xy, enable_apas=enable_apas)

    rows = []
    terminated = False
    truncated = False
    final_info = {"is_success": False, "terminated_reason": "unknown"}

    while not (terminated or truncated) and env.current_step < max_steps:
        pre_pos = np.asarray(env.current_pos, dtype=float).copy()
        pre_time = float(env.current_time)
        observer = _local_oracle_diagnostics(env)
        pre_storm = _storm_state(env, pre_pos, pre_time)
        coverage = _sample_coverage(env, pre_time)

        if controller == "expert":
            action = env.local_avoidance_expert_action()
            expert_mode = str(getattr(env, "last_expert_mode", "unknown"))
        else:
            action = np.zeros(3, dtype=np.float32)
            expert_mode = "inactive"

        _, _, terminated, truncated, step_info = env.step(action)
        final_info = step_info
        post_pos = np.asarray(env.current_pos, dtype=float).copy()
        post_time = float(env.current_time)
        post_storm = _storm_state(env, post_pos, post_time)
        segment = _segment_core_probe(env, pre_pos, post_pos, post_time, samples=segment_samples)

        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "step": int(env.current_step),
                "time_s": pre_time,
                "post_time_s": post_time,
                "x_m": float(pre_pos[0]),
                "y_m": float(pre_pos[1]),
                "post_x_m": float(post_pos[0]),
                "post_y_m": float(post_pos[1]),
                "action_heading": float(action[0]),
                "action_speed": float(action[1]),
                "action_agl": float(action[2]),
                "expert_mode": expert_mode,
                "terminated_after_step": bool(terminated or truncated),
                "terminated_reason_after_step": str(final_info.get("terminated_reason", "")) if terminated or truncated else "",
                "apas_intervened": bool(step_info.get("apas_intervened", False)),
                "apas_segment_rejections": int(step_info.get("apas_segment_rejections", 0)),
                "apas_endpoint_rejections": int(step_info.get("apas_endpoint_rejections", 0)),
                "apas_no_valid_candidate": bool(step_info.get("apas_no_valid_candidate", False)),
                "apas_segment_max_risk_bonus": float(step_info.get("apas_segment_max_risk_bonus", 0.0)),
                "apas_segment_core_hit": bool(step_info.get("apas_segment_core_hit", False)),
                "maneuver_extra_energy_j": float(step_info.get("maneuver_extra_energy_j", 0.0)),
                "safety_intervention_burden": float(step_info.get("safety_intervention_burden", 0.0)),
                "adjusted_energy_step_j": float(step_info.get("adjusted_energy_step_j", 0.0)),
                "replan_triggered": bool(step_info.get("replan_triggered", False)),
                "replan_success": bool(step_info.get("replan_success", False)),
                "replan_reason": str(step_info.get("replan_reason", "none")),
                "last_replan_event": str(step_info.get("last_replan_event", "none")),
                **{f"obs_{key}": value for key, value in observer.items()},
                **{f"pre_{key}": value for key, value in pre_storm.items()},
                **{f"post_{key}": value for key, value in post_storm.items()},
                **coverage,
                **segment,
            }
        )

    max_observed_danger = max(
        (max(float(row["obs_max_danger"]), float(row["obs_forward_danger"])) for row in rows),
        default=0.0,
    )
    max_exact_danger = max((float(row["pre_exact_danger"]) for row in rows), default=0.0)
    min_pre_core_margin = min(
        (float(row["pre_core_margin_m"]) for row in rows if row["pre_core_margin_m"] != ""),
        default=float("inf"),
    )
    min_post_core_margin = min(
        (float(row["post_core_margin_m"]) for row in rows if row["post_core_margin_m"] != ""),
        default=float("inf"),
    )
    max_sample_danger_hits = max((int(row["sample_danger_hits"]) for row in rows), default=0)
    max_sample_core_hits = max((int(row["sample_core_hits"]) for row in rows), default=0)
    final_reason = str(final_info.get("terminated_reason", "unknown"))

    if final_reason == "destructive_storm_core" and max_observed_danger < 0.05:
        if max_exact_danger >= 0.05 and max_sample_danger_hits == 0:
            suspected_cause = "observer_sampling_miss"
        elif min_pre_core_margin > 0.0 and min_post_core_margin <= 0.0:
            suspected_cause = "one_step_core_entry"
        else:
            suspected_cause = "late_or_unobserved_core"
    elif final_reason == "destructive_storm_core":
        suspected_cause = "warning_seen_but_not_avoided"
    else:
        suspected_cause = "no_core_failure"

    summary = {
        "seed": int(seed),
        "method": method,
        "success": bool(final_info.get("is_success", False)),
        "terminated_reason": final_reason,
        "steps": int(env.current_step),
        "max_observed_danger": float(max_observed_danger),
        "max_exact_danger": float(max_exact_danger),
        "min_pre_core_margin_m": "" if min_pre_core_margin == float("inf") else float(min_pre_core_margin),
        "min_post_core_margin_m": "" if min_post_core_margin == float("inf") else float(min_post_core_margin),
        "max_sample_danger_hits": int(max_sample_danger_hits),
        "max_sample_core_hits": int(max_sample_core_hits),
        "apas_interventions": int(getattr(env, "episode_apas_interventions", 0)),
        "apas_segment_rejections": int(getattr(env, "episode_apas_segment_rejections", 0)),
        "apas_no_valid_candidates": int(getattr(env, "episode_apas_no_valid_candidates", 0)),
        "eval_maneuver_extra_energy_j": float(getattr(env, "episode_eval_maneuver_extra_energy_j", 0.0)),
        "eval_safety_intervention_burden": float(getattr(env, "episode_eval_safety_intervention_burden", 0.0)),
        "eval_adjusted_energy_j": float(getattr(env, "episode_eval_adjusted_energy_j", 0.0)),
        "replans": int(getattr(env, "episode_replans", 0)),
        "replan_successes": int(getattr(env, "episode_replan_successes", 0)),
        "replan_failures": int(getattr(env, "episode_replan_failures", 0)),
        "replan_to_rejoin_successes": int(getattr(env, "episode_replan_to_rejoin_successes", 0)),
        "replan_to_goal_successes": int(getattr(env, "episode_replan_to_goal_successes", 0)),
        "sensor_blind_core_like": bool(final_reason == "destructive_storm_core" and max_observed_danger < 0.05),
        "suspected_cause": suspected_cause,
    }
    return rows, summary


def _summarize(summaries: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        by_method[str(row["method"])].append(row)

    payload = {}
    for method, rows in sorted(by_method.items()):
        n = max(len(rows), 1)
        core_failures = [row for row in rows if row["terminated_reason"] == "destructive_storm_core"]
        blind = [row for row in rows if bool(row["sensor_blind_core_like"])]
        payload[method] = {
            "episodes": len(rows),
            "success_rate": sum(1 for row in rows if bool(row["success"])) / n,
            "destructive_core_failures": len(core_failures),
            "sensor_blind_core_like": len(blind),
            "apas_interventions_mean": sum(float(row.get("apas_interventions", 0.0)) for row in rows) / n,
            "apas_segment_rejections_mean": sum(float(row.get("apas_segment_rejections", 0.0)) for row in rows) / n,
            "apas_no_valid_candidates_mean": sum(float(row.get("apas_no_valid_candidates", 0.0)) for row in rows) / n,
            "eval_maneuver_extra_energy_j_mean": sum(
                float(row.get("eval_maneuver_extra_energy_j", 0.0)) for row in rows
            )
            / n,
            "eval_safety_intervention_burden_mean": sum(
                float(row.get("eval_safety_intervention_burden", 0.0)) for row in rows
            )
            / n,
            "eval_adjusted_energy_j_mean": sum(float(row.get("eval_adjusted_energy_j", 0.0)) for row in rows) / n,
            "replans_mean": sum(float(row.get("replans", 0.0)) for row in rows) / n,
            "replan_successes_mean": sum(float(row.get("replan_successes", 0.0)) for row in rows) / n,
            "replan_failures_mean": sum(float(row.get("replan_failures", 0.0)) for row in rows) / n,
            "replan_to_rejoin_successes_mean": sum(
                float(row.get("replan_to_rejoin_successes", 0.0)) for row in rows
            )
            / n,
            "replan_to_goal_successes_mean": sum(float(row.get("replan_to_goal_successes", 0.0)) for row in rows)
            / n,
            "terminated_reasons": dict(Counter(str(row["terminated_reason"]) for row in rows)),
            "suspected_causes": dict(Counter(str(row["suspected_cause"]) for row in core_failures)),
        }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose why destructive storm core hits can be sensor-blind in v2.5."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("astar", "expert", "astar_apas", "expert_apas"),
        default=["astar", "expert"],
    )
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--planner-max-steps", type=int, default=15000)
    parser.add_argument("--task-sampling-trials", type=int, default=180)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--segment-samples", type=int, default=32)
    apas_segment_group = parser.add_mutually_exclusive_group()
    apas_segment_group.add_argument("--apas-segment-check", dest="apas_segment_check", action="store_true", default=True)
    apas_segment_group.add_argument("--no-apas-segment-check", dest="apas_segment_check", action="store_false")
    parser.add_argument(
        "--replan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable v2.5 replan-to-rejoin recovery during diagnosis (default: false).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "sensor_blind_core")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seeds:
        seeds = _parse_seeds(args.seeds)
    else:
        seeds = [int(args.seed + idx) for idx in range(int(args.episodes))]

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

    all_core_rows = []
    summaries = []
    for seed in seeds:
        for method in args.methods:
            print(f"[sensor-blind-core] seed={seed} method={method}", flush=True)
            rows, summary = _run_method(
                config=config,
                seed=int(seed),
                method=str(method),
                max_steps=int(args.max_episode_steps),
                sampling_trials=int(args.task_sampling_trials),
                segment_samples=int(args.segment_samples),
            )
            summaries.append(summary)
            if summary["terminated_reason"] == "destructive_storm_core":
                all_core_rows.extend(rows)
            print(
                f"  success={summary['success']} reason={summary['terminated_reason']} "
                f"blind={summary['sensor_blind_core_like']} cause={summary['suspected_cause']} "
                f"max_obs={summary['max_observed_danger']:.3f}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"sensor_blind_core_summary_{timestamp}.json"
    episode_csv_path = output_dir / f"sensor_blind_core_episodes_{timestamp}.csv"
    trace_csv_path = output_dir / f"sensor_blind_core_traces_{timestamp}.csv"

    if summaries:
        with episode_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    if all_core_rows:
        with trace_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_core_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_core_rows)

    payload = {
        "meta": {
            "seeds": seeds,
            "methods": list(args.methods),
            "curriculum_stage": int(args.curriculum_stage),
            "stress": str(args.stress),
            "segment_samples": int(args.segment_samples),
            "apas_segment_check": bool(args.apas_segment_check),
            "replan": bool(args.replan),
        },
        "summary": _summarize(summaries),
        "episode_csv": str(episode_csv_path),
        "core_trace_csv": str(trace_csv_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Sensor Blind Core Summary ===")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"episode_csv:    {episode_csv_path}")
    print(f"core_trace_csv: {trace_csv_path}")
    print(f"summary:        {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
