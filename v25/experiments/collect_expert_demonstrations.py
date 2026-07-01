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


def _make_env(config: SimulationConfig, task) -> GuidedDroneEnvV25:
    run_cfg = copy.deepcopy(config)
    run_cfg.wind_seed = int(task.seed)
    run_cfg.enable_single_agent_gusts = False
    run_cfg.enable_random_gusts = False
    run_cfg.planner_time_mode = "4d"
    run_cfg.rl_enable_apas = True
    run_cfg.v25_stale_waypoint_skip_enabled = True
    run_cfg.v25_risk_membrane_enabled = True
    env = GuidedDroneEnvV25(run_cfg)
    return env


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _run_episode(
    config: SimulationConfig,
    seed: int,
    episode: int,
    sampling_trials: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], dict]:
    task_cfg = copy.deepcopy(config)
    task_cfg.wind_seed = int(seed)
    task_cfg.max_steps = int(getattr(config, "v25_demo_planner_max_steps", config.max_steps))
    task = sample_task_for_seed(task_cfg, seed=int(seed), max_trials=int(sampling_trials))

    env = _make_env(config, task)
    obs, _ = env.reset(seed=int(seed), options={"start_xy": task.start_xy, "goal_xy": task.goal_xy})

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    step_rows: list[dict] = []
    terminated = False
    truncated = False
    final_info: dict = {"is_success": False, "terminated_reason": "unknown"}

    while not (terminated or truncated):
        expert_action = env.local_avoidance_expert_action()
        expert_mode = str(getattr(env, "last_expert_mode", "unknown"))
        expert_active = bool(getattr(env, "last_expert_active", False))
        pre_time_s = float(env.current_time)
        pre_step = int(env.current_step)
        pre_wp_idx = int(getattr(env, "current_wp_idx", 0))

        observations.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(np.asarray(expert_action, dtype=np.float32).copy())

        obs, _, terminated, truncated, final_info = env.step(expert_action)

        step_rows.append(
            {
                "episode": int(episode),
                "seed": int(seed),
                "step": pre_step,
                "time_s": pre_time_s,
                "expert_mode": expert_mode,
                "expert_active": expert_active,
                "action_heading": float(expert_action[0]),
                "action_speed": float(expert_action[1]),
                "action_agl": float(expert_action[2]),
                "apas_intervened": bool(final_info.get("apas_intervened", False)),
                "apas_no_valid_candidate": bool(final_info.get("apas_no_valid_candidate", False)),
                "apas_heading_offset_deg": _safe_float(final_info.get("apas_heading_offset_deg", 0.0)),
                "apas_speed_reduction_mps": _safe_float(final_info.get("apas_speed_reduction_mps", 0.0)),
                "apas_agl_increment_m": _safe_float(final_info.get("apas_agl_increment_m", 0.0)),
                "apas_segment_rejections": int(final_info.get("apas_segment_rejections", 0)),
                "current_wp_idx": pre_wp_idx,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "is_success": bool(final_info.get("is_success", False)),
                "terminated_reason": str(final_info.get("terminated_reason", "running")),
            }
        )

    episode_summary = {
        "episode": int(episode),
        "seed": int(seed),
        "steps": int(env.current_step),
        "success": bool(final_info.get("is_success", False)),
        "terminated_reason": str(final_info.get("terminated_reason", "unknown")),
        "mission_time_s": float(env.current_time),
        "energy_used_j": float(config.battery_capacity_j - float(final_info.get("energy_remaining_j", env.energy_remaining))),
        "episode_apas_interventions": int(getattr(env, "episode_apas_interventions", 0)),
        "episode_apas_segment_rejections": int(getattr(env, "episode_apas_segment_rejections", 0)),
        "episode_apas_no_valid_candidates": int(getattr(env, "episode_apas_no_valid_candidates", 0)),
        "episode_stale_waypoint_skips": int(getattr(env, "episode_stale_waypoint_skips", 0)),
        "episode_expert_band_avoidance_steps": int(getattr(env, "episode_expert_band_avoidance_steps", 0)),
        "episode_expert_emergency_steps": int(getattr(env, "episode_expert_emergency_steps", 0)),
        "episode_eval_adjusted_energy_j": float(getattr(env, "episode_eval_adjusted_energy_j", 0.0)),
        "start_x": float(task.start_xy[0]),
        "start_y": float(task.start_xy[1]),
        "goal_x": float(task.goal_xy[0]),
        "goal_y": float(task.goal_xy[1]),
    }
    return observations, actions, step_rows, episode_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Expert demonstrations from A* + Expert + APAS + waypoint-skip v2.5 rollouts."
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--planner-max-steps", type=int, default=15000)
    parser.add_argument("--env-max-steps", type=int, default=900)
    parser.add_argument("--task-sampling-trials", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("v25") / "artifacts" / "expert_demos")
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = SimulationConfig()
    config.curriculum_stage = int(args.curriculum_stage)
    config.max_steps = int(args.env_max_steps)
    config.v25_demo_planner_max_steps = int(args.planner_max_steps)
    config.v25_disruption_stress_level = str(args.stress)
    config.v25_apas_segment_check_enabled = True
    config.v25_replan_enabled = False
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = output_dir / f"expert_demo_{timestamp}.npz"
    step_csv_path = output_dir / f"expert_demo_steps_{timestamp}.csv"
    episode_csv_path = output_dir / f"expert_demo_episodes_{timestamp}.csv"
    summary_path = output_dir / f"expert_demo_summary_{timestamp}.json"

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_step_rows: List[dict] = []
    episode_rows: List[dict] = []

    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        print(f"[episode {ep:03d}] seed={ep_seed}", flush=True)
        obs_rows, action_rows, step_rows, episode_summary = _run_episode(
            config=config,
            seed=ep_seed,
            episode=ep,
            sampling_trials=int(args.task_sampling_trials),
        )
        keep = bool(episode_summary["success"]) or not bool(args.success_only)
        if keep:
            all_obs.extend(obs_rows)
            all_actions.extend(action_rows)
            all_step_rows.extend(step_rows)
        episode_summary["kept_for_training"] = bool(keep)
        episode_rows.append(episode_summary)
        print(
            f"  success={episode_summary['success']} reason={episode_summary['terminated_reason']} "
            f"steps={episode_summary['steps']} kept={keep}",
            flush=True,
        )

    observations = np.asarray(all_obs, dtype=np.float32)
    actions = np.asarray(all_actions, dtype=np.float32)
    np.savez_compressed(
        npz_path,
        observations=observations,
        actions=actions,
        action_low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    if all_step_rows:
        with step_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_step_rows)

    with episode_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    success_rate = float(np.mean([float(row["success"]) for row in episode_rows])) if episode_rows else 0.0
    kept_episodes = int(sum(1 for row in episode_rows if row["kept_for_training"]))
    summary = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "curriculum_stage": int(args.curriculum_stage),
        "stress": str(args.stress),
        "success_only": bool(args.success_only),
        "success_rate": success_rate,
        "kept_episodes": kept_episodes,
        "transition_count": int(len(observations)),
        "observation_dim": int(observations.shape[1]) if observations.ndim == 2 else 0,
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
        "npz": str(npz_path),
        "step_csv": str(step_csv_path),
        "episode_csv": str(episode_csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Expert Demonstration Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
