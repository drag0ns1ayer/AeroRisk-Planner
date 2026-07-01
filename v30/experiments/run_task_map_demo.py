from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from v30.mission_map import ChargingStation, InspectionPoint, MissionMap
from v30.segment_executor import AStarSegmentExecutor
from v30.task_executor import SimpleTaskExecutor


def build_demo_map(start_xy: tuple[float, float] = (0.0, 0.0), distance_scale: float = 1.0) -> MissionMap:
    sx, sy = start_xy
    s = float(distance_scale)
    return MissionMap(
        name="v30_demo_task_map",
        start_xy=(sx, sy),
        home_xy=(sx, sy),
        inspection_points=[
            InspectionPoint(id="tower-a", xy=(sx + s * 600.0, sy + s * 100.0), priority=3.0, service_time_s=45.0, risk_value=0.20),
            InspectionPoint(id="ridge-b", xy=(sx + s * 1200.0, sy - s * 200.0), priority=2.0, service_time_s=60.0, risk_value=0.35),
            InspectionPoint(id="valve-c", xy=(sx + s * 300.0, sy + s * 900.0), priority=4.0, service_time_s=30.0, risk_value=0.10),
        ],
        charging_stations=[
            ChargingStation(id="charge-west", xy=(sx + s * 100.0, sy + s * 100.0), charge_rate_j_per_s=4000.0),
            ChargingStation(id="charge-east", xy=(sx + s * 900.0, sy + s * 100.0), charge_rate_j_per_s=3500.0),
        ],
    )


def build_real_astar_segment_executor(config: SimulationConfig) -> tuple[AStarSegmentExecutor, tuple[float, float]]:
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery_manager = BatteryManager(config)
    planner = AStarPlanner(config, estimator, physics)
    min_x, _, min_y, _ = estimator.get_bounds()
    start_xy = (min_x + 100.0, min_y + 100.0)
    return AStarSegmentExecutor(config, estimator, physics, battery_manager, planner), start_xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight v3.0 semantic task-map demo.")
    parser.add_argument("--output-dir", default="results/v30_task_map_demo")
    parser.add_argument("--real-astar", action="store_true", help="Execute each semantic leg through the legacy A* stack.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig()
    segment_executor = None
    start_xy = (0.0, 0.0)
    distance_scale = 1.0
    if args.real_astar:
        config.max_replans = 8
        config.max_mission_time_s = 1800.0
        config.mission_update_interval_s = 60.0
        segment_executor, start_xy = build_real_astar_segment_executor(config)
        distance_scale = 0.18

    mission_map = build_demo_map(start_xy=start_xy, distance_scale=distance_scale)
    mission_map.save_json(output_dir / "mission_map.json")

    executor = SimpleTaskExecutor(config, segment_executor=segment_executor)
    result = executor.execute(mission_map)
    summary = {
        "success": result.success,
        "completed_inspections": result.completed_inspections,
        "charging_visits": result.charging_visits,
        "total_time_s": result.total_time_s,
        "total_energy_used_j": result.total_energy_used_j,
        "remaining_energy_j": result.remaining_energy_j,
        "failure_reason": result.failure_reason,
        "events": [
            {
                "time_s": event.time_s,
                "kind": event.kind,
                "target_id": event.target_id,
                "position_xy": event.position_xy,
                "remaining_energy_j": event.remaining_energy_j,
                "detail": event.detail,
            }
            for event in result.events
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== V3.0 Task Map Demo ===")
    print(f"success: {summary['success']}")
    print(f"completed_inspections: {summary['completed_inspections']}")
    print(f"charging_visits: {summary['charging_visits']}")
    print(f"total_time_s: {summary['total_time_s']:.1f}")
    print(f"total_energy_used_j: {summary['total_energy_used_j']:.1f}")
    print(f"mission_map: {output_dir / 'mission_map.json'}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
