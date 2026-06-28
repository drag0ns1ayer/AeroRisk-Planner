from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


ABLATIONS = (
    {
        "method": "astar_raw",
        "label": "A* raw",
        "use_expert": False,
        "apas": False,
        "stale_waypoint_skip": False,
        "risk_membrane": False,
    },
    {
        "method": "astar_waypoint_skip",
        "label": "A* + waypoint-skip",
        "use_expert": False,
        "apas": False,
        "stale_waypoint_skip": True,
        "risk_membrane": False,
    },
    {
        "method": "astar_apas",
        "label": "A* + APAS",
        "use_expert": False,
        "apas": True,
        "stale_waypoint_skip": False,
        "risk_membrane": False,
    },
    {
        "method": "astar_apas_waypoint_skip",
        "label": "A* + APAS + waypoint-skip",
        "use_expert": False,
        "apas": True,
        "stale_waypoint_skip": True,
        "risk_membrane": False,
    },
    {
        "method": "expert_apas_waypoint_skip",
        "label": "A* + Expert + APAS + waypoint-skip",
        "use_expert": True,
        "apas": True,
        "stale_waypoint_skip": True,
        "risk_membrane": True,
    },
)


def _make_initial_env(base_config: SimulationConfig, task) -> GuidedDroneEnvV25:
    config = copy.deepcopy(base_config)
    config.wind_seed = int(task.seed)
    config.rl_enable_apas = False
    env = GuidedDroneEnvV25(config)
    env.reset(seed=int(task.seed), options={"start_xy": task.start_xy, "goal_xy": task.goal_xy})
    return env


def _run_variant(initial_env: GuidedDroneEnvV25, variant: dict) -> Dict[str, float]:
    env = copy.deepcopy(initial_env)
    env.config.rl_enable_apas = bool(variant["apas"])
    env.config.v25_stale_waypoint_skip_enabled = bool(variant["stale_waypoint_skip"])
    env.config.v25_risk_membrane_enabled = bool(variant["risk_membrane"])

    terminated = False
    truncated = False
    final_info = {"is_success": False, "terminated_reason": "unknown"}
    prev_pos = env.current_pos.copy()
    path_distance_m = 0.0

    while not (terminated or truncated):
        if bool(variant["use_expert"]):
            action = env.local_avoidance_expert_action()
        else:
            action = [0.0, 0.0, 0.0]
        _, _, terminated, truncated, final_info = env.step(action)
        curr_pos = env.current_pos.copy()
        path_distance_m += float(((curr_pos - prev_pos) ** 2).sum() ** 0.5)
        prev_pos = curr_pos

    steps = max(int(env.current_step), 1)
    energy_used_j = float(env.config.battery_capacity_j - float(final_info.get("energy_remaining_j", env.energy_remaining)))
    avg_risk = float(np.mean(env.telemetry_risk)) if env.telemetry_risk else 0.0
    peak_risk = float(env.telemetry_max_p_crash) if env.telemetry_risk else 0.0
    peak_power = float(np.max(env.telemetry_power_w)) if env.telemetry_power_w else 0.0
    return {
        "success": bool(final_info.get("is_success", False)),
        "failure_reason": final_info.get("terminated_reason"),
        "terminated_reason": final_info.get("terminated_reason"),
        "mission_time_s": float(env.current_time),
        "total_energy_used_j": energy_used_j,
        "energy_used_j": energy_used_j,
        "avg_risk": avg_risk,
        "peak_risk": peak_risk,
        "peak_power_w": peak_power,
        "path_distance_m": float(path_distance_m),
        "episode_apas_interventions": int(getattr(env, "episode_apas_interventions", 0)),
        "episode_apas_segment_rejections": int(getattr(env, "episode_apas_segment_rejections", 0)),
        "episode_apas_no_valid_candidates": int(getattr(env, "episode_apas_no_valid_candidates", 0)),
        "episode_stale_waypoint_skips": int(getattr(env, "episode_stale_waypoint_skips", 0)),
        "episode_stale_waypoint_skip_delta": int(getattr(env, "episode_stale_waypoint_skip_delta", 0)),
        "episode_eval_adjusted_energy_j": float(getattr(env, "episode_eval_adjusted_energy_j", 0.0)),
        "episode_eval_safety_intervention_burden": float(
            getattr(env, "episode_eval_safety_intervention_burden", 0.0)
        ),
        "episode_expert_band_avoidance_steps": int(getattr(env, "episode_expert_band_avoidance_steps", 0)),
        "episode_expert_emergency_steps": int(getattr(env, "episode_expert_emergency_steps", 0)),
        "episode_expert_recovering_steps": int(getattr(env, "episode_expert_recovering_steps", 0)),
        "episode_destructive_core_hits": int(getattr(env, "episode_destructive_core_hits", 0)),
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1))


def summarize_ablation_rows(rows: List[Dict[str, object]], method: str) -> Dict[str, float]:
    sub = [row for row in rows if row["method"] == method]
    n = len(sub)
    reasons: dict[str, int] = {}
    for row in sub:
        reason = str(row.get("terminated_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "episodes": n,
        "success_rate": _mean(sub, "success"),
        "energy_used_j_mean": _mean(sub, "energy_used_j"),
        "eval_adjusted_energy_j_mean": _mean(sub, "episode_eval_adjusted_energy_j"),
        "safety_intervention_burden_mean": _mean(sub, "episode_eval_safety_intervention_burden"),
        "avg_risk_mean": _mean(sub, "avg_risk"),
        "peak_risk_mean": _mean(sub, "peak_risk"),
        "mission_time_s_mean": _mean(sub, "mission_time_s"),
        "path_distance_m_mean": _mean(sub, "path_distance_m"),
        "apas_interventions_mean": _mean(sub, "episode_apas_interventions"),
        "apas_segment_rejections_mean": _mean(sub, "episode_apas_segment_rejections"),
        "apas_no_valid_candidates_mean": _mean(sub, "episode_apas_no_valid_candidates"),
        "stale_waypoint_skips_mean": _mean(sub, "episode_stale_waypoint_skips"),
        "destructive_core_hits_mean": _mean(sub, "episode_destructive_core_hits"),
        "expert_band_avoidance_steps_mean": _mean(sub, "episode_expert_band_avoidance_steps"),
        "expert_emergency_steps_mean": _mean(sub, "episode_expert_emergency_steps"),
        "terminated_reasons": reasons,
    }


def _write_outputs(
    rows: List[Dict[str, object]],
    raw_csv_path: Path,
    summary_json_path: Path,
    args: argparse.Namespace,
    complete: bool,
) -> None:
    if not rows:
        return
    with raw_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = {str(v["method"]): summarize_ablation_rows(rows, str(v["method"])) for v in ABLATIONS}
    labels = {str(v["method"]): str(v["label"]) for v in ABLATIONS}
    payload = {
        "complete": bool(complete),
        "config": {
            "episodes_requested": int(args.episodes),
            "rows_completed": len(rows),
            "seed": int(args.seed),
            "curriculum_stage": int(args.curriculum_stage),
            "stress": str(args.stress),
            "planner_max_steps": int(args.planner_max_steps),
            "env_max_steps": int(args.env_max_steps),
            "apas_segment_check": bool(args.apas_segment_check),
            "replan": False,
        },
        "labels": labels,
        "summaries": summaries,
        "raw_csv": str(raw_csv_path),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final v2.5 controller-layer ablation.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--planner-max-steps", type=int, default=15000)
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=900,
        help="Episode step limit used during execution; keep this separate from planner search budget.",
    )
    parser.add_argument("--task-sampling-trials", type=int, default=180)
    parser.add_argument(
        "--apas-segment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable swept-segment APAS checks (default: true).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "final_ablation_v25",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    base_config = SimulationConfig()
    base_config.curriculum_stage = int(args.curriculum_stage)
    base_config.max_steps = int(args.env_max_steps)
    base_config.v25_disruption_stress_level = str(args.stress)
    base_config.v25_apas_segment_check_enabled = bool(args.apas_segment_check)
    base_config.v25_replan_enabled = False
    base_config.enable_single_agent_gusts = False
    base_config.enable_random_gusts = False
    base_config.planner_time_mode = "4d"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = output_dir / f"final_ablation_raw_{timestamp}.csv"
    summary_json_path = output_dir / f"final_ablation_summary_{timestamp}.json"

    rows: List[Dict[str, object]] = []
    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        task_cfg = copy.deepcopy(base_config)
        task_cfg.wind_seed = ep_seed
        task_cfg.max_steps = int(args.planner_max_steps)
        print(f"[episode {ep:03d}] sampling seed={ep_seed}", flush=True)
        task = sample_task_for_seed(task_cfg, seed=ep_seed, max_trials=int(args.task_sampling_trials))
        initial_env = _make_initial_env(base_config, task)

        for variant in ABLATIONS:
            metrics = _run_variant(initial_env, variant)
            row = {
                "episode": ep,
                "seed": ep_seed,
                "method": str(variant["method"]),
                "controller_label": str(variant["label"]),
                "apas_enabled": bool(variant["apas"]),
                "stale_waypoint_skip_enabled": bool(variant["stale_waypoint_skip"]),
                "risk_membrane_enabled": bool(variant["risk_membrane"]),
                "start_x": task.start_xy[0],
                "start_y": task.start_xy[1],
                "goal_x": task.goal_xy[0],
                "goal_y": task.goal_xy[1],
                **metrics,
            }
            rows.append(row)
            _write_outputs(rows, raw_csv_path, summary_json_path, args, complete=False)
            print(
                f"  {variant['label']}: success={metrics['success']} reason={metrics['terminated_reason']}",
                flush=True,
            )

    _write_outputs(rows, raw_csv_path, summary_json_path, args, complete=True)
    summaries = {str(v["method"]): summarize_ablation_rows(rows, str(v["method"])) for v in ABLATIONS}

    print("\n=== Final Ablation Summary ===")
    for variant in ABLATIONS:
        method = str(variant["method"])
        print(f"{variant['label']}: {summaries[method]}")
    print(f"raw_csv: {raw_csv_path}")
    print(f"summary: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
