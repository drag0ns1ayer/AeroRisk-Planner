from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SimulationConfig
from v25.disruptions import build_disruption_layer_v25
from v30.experiments.run_task_map_demo import (
    build_demo_map,
    build_real_astar_segment_executor,
    offset_mission_map,
)
from v30.mission_map import MissionMap
from v30.segment_executor import V25GuidedSegmentExecutor
from v30.task_executor import SimpleTaskExecutor
from v30.visualization import (
    generate_wind_trajectory_gif,
    plot_mission_elevation_profile,
    plot_mission_terrain_map,
    plot_wind_map,
)


def _route_end_xy(mission_map: MissionMap) -> tuple[float, float]:
    if mission_map.inspection_points:
        return mission_map.inspection_points[-1].xy
    if mission_map.home_xy is not None:
        return mission_map.home_xy
    return mission_map.start_xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Render v3.0 task-map wind fields, trajectories, and GIF.")
    parser.add_argument("--mission-map", default="", help="Path to a v3.0 mission map JSON.")
    parser.add_argument("--relative-map", action="store_true")
    parser.add_argument("--map-scale", type=float, default=0.18)
    parser.add_argument("--output-dir", default="results/v30_visuals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stress", choices=["normal", "hard", "extreme", "fragile"], default="fragile")
    parser.add_argument("--gif-frames", type=int, default=45)
    parser.add_argument("--max-replans", type=int, default=30)
    parser.add_argument("--max-mission-time", type=float, default=3600.0)
    parser.add_argument("--mission-update-interval", type=float, default=60.0)
    parser.add_argument(
        "--control-mode",
        choices=["legacy_astar", "v25_astar", "v25_expert", "v25_rl"],
        default="legacy_astar",
        help="Segment execution stack used for each v3.0 task leg.",
    )
    parser.add_argument("--rl-model-path", default="", help="PPO model path for --control-mode v25_rl.")
    parser.add_argument("--no-apas", action="store_true", help="Disable APAS for v25_* segment modes.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig()
    config.max_replans = int(args.max_replans)
    config.max_mission_time_s = float(args.max_mission_time)
    config.mission_update_interval_s = float(args.mission_update_interval)
    config.v25_stress_level = args.stress
    base_segment_executor, start_xy = build_real_astar_segment_executor(config)
    estimator = base_segment_executor.estimator
    segment_executor = base_segment_executor
    if args.control_mode.startswith("v25_"):
        model = None
        if args.control_mode == "v25_rl":
            try:
                from stable_baselines3 import PPO
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("stable_baselines3 is required for --control-mode v25_rl") from exc
            if not args.rl_model_path:
                raise ValueError("--rl-model-path is required for --control-mode v25_rl")
            model = PPO.load(args.rl_model_path, device="cpu")
        segment_executor = V25GuidedSegmentExecutor(
            config=config,
            mode=args.control_mode.replace("v25_", ""),
            model=model,
            enable_apas=not bool(args.no_apas),
            seed=int(args.seed),
        )

    if args.mission_map:
        mission_map = MissionMap.load_json(args.mission_map)
        if args.relative_map:
            mission_map = offset_mission_map(mission_map, start_xy, scale=float(args.map_scale))
    else:
        mission_map = build_demo_map(start_xy=start_xy, distance_scale=float(args.map_scale))

    disruptions = build_disruption_layer_v25(
        mission_map.start_xy,
        _route_end_xy(mission_map),
        config=config,
        seed=int(args.seed),
    )
    executor = SimpleTaskExecutor(config, segment_executor=segment_executor)
    result = executor.execute(mission_map)

    mission_map.save_json(output_dir / "mission_map.json")
    summary = {
        "success": result.success,
        "completed_inspections": result.completed_inspections,
        "charging_visits": result.charging_visits,
        "returned_home": result.returned_home,
        "final_position_xy": result.final_position_xy,
        "total_time_s": result.total_time_s,
        "total_energy_used_j": result.total_energy_used_j,
        "remaining_energy_j": result.remaining_energy_j,
        "failure_reason": result.failure_reason,
        "path_points": len(result.actual_path_xyz),
        "control_mode": args.control_mode,
        "apas_enabled": not bool(args.no_apas) if args.control_mode.startswith("v25_") else False,
        "map_loaded_from_file": bool(getattr(estimator.map, "map_loaded_from_file", False)),
        "map_source_path": str(getattr(estimator.map, "map_source_path", "")),
        "map_load_error": str(getattr(estimator.map, "map_load_error", "")),
        "dem_min_m": float(estimator.map.dem.min()),
        "dem_max_m": float(estimator.map.dem.max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_mission_terrain_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "mission_terrain_trajectory.png",
        title="V3.0 Task Mission on Bernese Terrain",
        full_map=True,
    )
    plot_mission_terrain_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "mission_terrain_zoom.png",
        title="V3.0 Task Mission Detail View",
        full_map=False,
    )
    plot_wind_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        output_path=output_dir / "observable_wind_field.png",
        include_random_layer=False,
        include_trajectory=False,
        title="Observable / Predictable Wind Field",
        full_map=True,
    )
    plot_wind_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "true_wind_field.png",
        include_random_layer=True,
        include_trajectory=False,
        title="Observable + Random Layer Wind Field",
        full_map=True,
    )
    plot_wind_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        output_path=output_dir / "observable_wind_trajectory.png",
        include_random_layer=False,
        include_trajectory=True,
        title="Trajectory on Observable / Predictable Wind Field",
        full_map=True,
    )
    plot_wind_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "true_wind_trajectory.png",
        include_random_layer=True,
        include_trajectory=True,
        title="Trajectory on Observable + Random Layer Wind Field",
        full_map=True,
    )
    plot_mission_elevation_profile(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        output_path=output_dir / "mission_elevation_profile.png",
    )
    generate_wind_trajectory_gif(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "true_wind_trajectory.gif",
        frames=int(args.gif_frames),
    )

    print("=== V3.0 Visual Outputs ===")
    for name in [
        "mission_terrain_trajectory.png",
        "mission_terrain_zoom.png",
        "observable_wind_field.png",
        "true_wind_field.png",
        "observable_wind_trajectory.png",
        "true_wind_trajectory.png",
        "mission_elevation_profile.png",
        "true_wind_trajectory.gif",
        "summary.json",
    ]:
        print(f"{name}: {output_dir / name}")


if __name__ == "__main__":
    main()
