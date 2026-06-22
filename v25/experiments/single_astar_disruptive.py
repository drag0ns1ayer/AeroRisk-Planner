from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from models.mission_models import MissionResult, SimulationState
from simulation.mission_executor import MissionExecutor
from v25.disruptions import DisruptionLayerV25, build_disruption_layer_v25
from v25.rl_env_disruptive import GuidedDroneEnvV25


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


class DisruptedSingleAStarExecutor(MissionExecutor):
    """
    Planner sees predictable environment.
    Runtime execution overlays disruptive disturbances.
    """

    def __init__(
        self,
        config: SimulationConfig,
        estimator: StateEstimator,
        physics: PhysicsEngine,
        battery_manager: BatteryManager,
        planner: AStarPlanner,
        disruptions: DisruptionLayerV25,
    ) -> None:
        super().__init__(config, estimator, physics, battery_manager, planner)
        self.disruptions = disruptions
        self.overload_events = 0
        self.high_risk_events = 0
        self.storm_hits = 0
        self.pulse_hits = 0
        self.disturbance_steps = 0
        self.disturbance_severity_sum = 0.0
        self.disturbance_severity_max = 0.0

    def _advance_along_path(
        self,
        state: SimulationState,
        path_xyz: List[Point3D],
        delta_t_s: float,
        time_history_s: List[float],
        power_history_w: List[float],
        risk_history: List[float],
    ) -> SimulationState:
        remaining_distance_m = self.config.cruise_speed_mps * delta_t_s
        if remaining_distance_m <= 0:
            return state

        current_pos = np.array(state.position_xyz, dtype=float)
        new_traveled_path = list(state.traveled_path_xyz)
        total_energy_used_j = state.total_energy_used_j
        remaining_energy_j = state.remaining_energy_j
        current_time_s = state.current_time_s
        execution_path = [tuple(current_pos)] + list(path_xyz[1:])

        for i in range(len(execution_path) - 1):
            p0 = np.array(execution_path[i], dtype=float)
            p1_full = np.array(execution_path[i + 1], dtype=float)
            seg_vec = p1_full - p0
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len <= 1e-9:
                continue

            if remaining_distance_m >= seg_len:
                p1 = p1_full
            else:
                ratio = remaining_distance_m / seg_len
                p1 = p0 + ratio * seg_vec

            midpoint = 0.5 * (p0 + p1)
            horizontal_vec = p1[:2] - p0[:2]
            base_wind_2d = self.estimator.get_wind(
                float(midpoint[0]), float(midpoint[1]), float(midpoint[2]), current_time_s
            )

            storm_vec = self.disruptions.storm_mutation.wind_at(float(midpoint[0]), float(midpoint[1]), current_time_s)
            pulse_vec = self.disruptions.headwind_pulse.wind_at(
                float(midpoint[0]), float(midpoint[1]), current_time_s, horizontal_vec
            )
            if float(np.linalg.norm(storm_vec)) > 1e-6:
                self.storm_hits += 1
            if float(np.linalg.norm(pulse_vec)) > 1e-6:
                self.pulse_hits += 1

            disturbed_wind_2d, severity = self.disruptions.sample(
                base_wind_2d=base_wind_2d,
                x=float(midpoint[0]),
                y=float(midpoint[1]),
                t_s=current_time_s,
                travel_dir_xy=horizontal_vec,
            )
            self.disturbance_severity_sum += float(severity)
            self.disturbance_severity_max = max(self.disturbance_severity_max, float(severity))
            if float(severity) > 1e-9:
                self.disturbance_steps += 1
            wind_3d = np.array([disturbed_wind_2d[0], disturbed_wind_2d[1], 0.0], dtype=float)

            seg_energy_j, seg_time_s, seg_power_w = self.physics.estimate_segment_energy(
                p0_xyz=p0,
                p1_xyz=p1,
                wind_velocity_xyz=wind_3d,
                cruise_speed_mps=self.config.cruise_speed_mps,
            )
            next_time_s = current_time_s + seg_time_s
            v_ground = max(np.linalg.norm(p1 - p0) / max(seg_time_s, 1e-6), 1.0)
            p_crash_base, _ = self.estimator.get_risk(
                float(p1[0]), float(p1[1]), float(p1[2]), float(v_ground), next_time_s
            )
            p_crash = min(1.0, p_crash_base + self.disruptions.risk_bonus(severity))

            time_history_s.append(next_time_s)
            power_history_w.append(float(seg_power_w))
            risk_history.append(float(p_crash))

            if seg_power_w > self.config.max_power * self.config.rl_overload_power_ratio:
                self.overload_events += 1
                return SimulationState(
                    current_time_s=current_time_s,
                    position_xyz=tuple(current_pos),
                    remaining_energy_j=remaining_energy_j,
                    traveled_path_xyz=new_traveled_path,
                    replans_count=state.replans_count,
                    total_energy_used_j=total_energy_used_j,
                    is_mission_failed=True,
                    failure_reason="overload_under_disruption",
                )
            if p_crash > self.config.rl_terminate_risk_threshold:
                self.high_risk_events += 1
                return SimulationState(
                    current_time_s=current_time_s,
                    position_xyz=tuple(current_pos),
                    remaining_energy_j=remaining_energy_j,
                    traveled_path_xyz=new_traveled_path,
                    replans_count=state.replans_count,
                    total_energy_used_j=total_energy_used_j,
                    is_mission_failed=True,
                    failure_reason="risk_spike_under_disruption",
                )
            if not self.battery_manager.can_consume(remaining_energy_j, seg_energy_j):
                return SimulationState(
                    current_time_s=current_time_s,
                    position_xyz=tuple(current_pos),
                    remaining_energy_j=remaining_energy_j,
                    traveled_path_xyz=new_traveled_path,
                    replans_count=state.replans_count,
                    total_energy_used_j=total_energy_used_j,
                    is_mission_failed=True,
                    failure_reason="battery_depleted",
                )

            remaining_energy_j = self.battery_manager.consume_energy(remaining_energy_j, seg_energy_j)
            total_energy_used_j += seg_energy_j
            current_time_s = next_time_s
            current_pos = p1
            new_traveled_path.append(tuple(current_pos))

            if remaining_distance_m < seg_len:
                break
            remaining_distance_m -= seg_len

        return SimulationState(
            current_time_s=current_time_s,
            position_xyz=tuple(current_pos),
            remaining_energy_j=remaining_energy_j,
            traveled_path_xyz=new_traveled_path,
            replans_count=state.replans_count,
            total_energy_used_j=total_energy_used_j,
            is_goal_reached=self._check_goal_reached(tuple(current_pos), (path_xyz[-1][0], path_xyz[-1][1])),
            is_mission_failed=False,
            failure_reason=None,
        )

    def _plan_once(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        use_wind: bool,
        start_time_s: float = 0.0,
    ) -> Optional[List[Point3D]]:
        # Keep this experiment focused on predictable-planning + disruptive-execution.
        if not use_wind:
            return None
        return super()._plan_once(start_xy, goal_xy, use_wind, start_time_s)


def build_start_goal(config: SimulationConfig, estimator: StateEstimator) -> Tuple[Point2D, Point2D]:
    min_x, max_x, min_y, max_y = estimator.get_bounds()
    start_xy = (min_x + config.start_offset_x, min_y + config.start_offset_y)
    goal_xy = (
        min_x + (max_x - min_x) * config.goal_offset_factor_x,
        min_y + (max_y - min_y) * config.goal_offset_factor_y,
    )
    return start_xy, goal_xy


def summarize_result(result: MissionResult, executor: DisruptedSingleAStarExecutor) -> dict:
    avg_power = float(np.mean(result.power_history_w)) if result.power_history_w else 0.0
    peak_power = float(np.max(result.power_history_w)) if result.power_history_w else 0.0
    avg_risk = float(np.mean(result.risk_history)) if result.risk_history else 0.0
    peak_risk = float(np.max(result.risk_history)) if result.risk_history else 0.0
    return {
        "success": result.success,
        "failure_reason": result.failure_reason,
        "mission_time_s": result.total_mission_time_s,
        "total_energy_used_j": result.total_energy_used_j,
        "total_replans": result.total_replans,
        "avg_power_w": avg_power,
        "peak_power_w": peak_power,
        "avg_risk": avg_risk,
        "peak_risk": peak_risk,
        "disturbance_hits": {
            "storm_mutation_hits": executor.storm_hits,
            "headwind_pulse_hits": executor.pulse_hits,
            "overload_events": executor.overload_events,
            "high_risk_events": executor.high_risk_events,
            "disturbance_steps": executor.disturbance_steps,
            "disturbance_max": executor.disturbance_severity_max,
            "disturbance_mean": (
                executor.disturbance_severity_sum / max(len(result.time_history_s), 1)
            ),
        },
    }


def run_single_astar_disruptive(seed: int, output_dir: Path) -> Path:
    config = SimulationConfig()
    config.wind_seed = seed
    config.curriculum_stage = 3
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"

    env = GuidedDroneEnvV25(config)
    _, reset_info = env.reset(seed=seed)
    start_xy = (float(env.current_pos[0]), float(env.current_pos[1]))
    goal_xy = (float(env.goal_pos[0]), float(env.goal_pos[1]))
    terminated = False
    truncated = False
    final_info = reset_info
    path_distance_m = 0.0
    previous_pos = np.asarray(env.current_pos, dtype=float).copy()
    while not (terminated or truncated):
        _, _, terminated, truncated, final_info = env.step(np.zeros(3, dtype=np.float32))
        path_distance_m += float(np.linalg.norm(np.asarray(env.current_pos, dtype=float) - previous_pos))
        previous_pos = np.asarray(env.current_pos, dtype=float).copy()

    summary = {
        "success": bool(final_info.get("is_success", False)),
        "failure_reason": None if final_info.get("is_success", False) else final_info.get("terminated_reason", "unknown"),
        "mission_time_s": float(env.current_time),
        "total_energy_used_j": float(config.battery_capacity_j - env.energy_remaining),
        "path_distance_m": path_distance_m,
        "avg_power_w": float(np.mean(env.telemetry_power_w)) if env.telemetry_power_w else 0.0,
        "peak_power_w": float(np.max(env.telemetry_power_w)) if env.telemetry_power_w else 0.0,
        "avg_risk": float(np.mean(env.telemetry_risk)) if env.telemetry_risk else 0.0,
        "peak_risk": float(env.telemetry_max_p_crash),
        "random_layer_seed": int(reset_info["random_layer_seed"]),
        "disturbance_steps": int(env.episode_disturbance_steps),
        "disturbance_max": float(env.episode_disturbance_max),
        "disturbance_mean": float(env.episode_disturbance_sum / max(env.current_step, 1)),
        "residual_action_sum": float(env.episode_residual_action_sum),
    }

    print("\n=== A* Baseline (Shared true world, zero residual) ===")
    print(f"success: {summary['success']}")
    print(f"failure_reason: {summary['failure_reason']}")
    print(f"mission_time_s: {summary['mission_time_s']:.1f}")
    print(f"total_energy_used_j: {summary['total_energy_used_j']:.1f}")
    print(f"path_distance_m: {summary['path_distance_m']:.1f}")
    print(f"avg_power_w: {summary['avg_power_w']:.1f}")
    print(f"peak_power_w: {summary['peak_power_w']:.1f}")
    print(f"avg_risk: {summary['avg_risk']:.4f}")
    print(f"peak_risk: {summary['peak_risk']:.4f}")
    print(f"random_layer_seed: {summary['random_layer_seed']}")
    print(
        "disturbance: "
        f"steps={summary['disturbance_steps']}, "
        f"max={summary['disturbance_max']:.4f}, "
        f"mean={summary['disturbance_mean']:.4f}"
    )
    print(f"residual_action_sum: {summary['residual_action_sum']:.1f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"single_astar_disruptive_{timestamp}.json"
    payload = {
        "seed": seed,
        "summary": summary,
        "start_xy": {"x": start_xy[0], "y": start_xy[1]},
        "goal_xy": {"x": goal_xy[0], "y": goal_xy[1]},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_summary: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-drone A* experiment with execution-only disruptive disturbances."
    )
    parser.add_argument("--seed", type=int, default=37, help="Random seed for wind/storm baseline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "single_astar_disruptive",
        help="Directory for summary outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_single_astar_disruptive(seed=args.seed, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
