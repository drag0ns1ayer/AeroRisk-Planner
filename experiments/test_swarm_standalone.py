from __future__ import annotations

from pathlib import Path

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.animation_builder import MissionAnimator
from utils.visualizer_core import Visualizer

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


def load_rl_model():
    if not PPO:
        print("[WARN] stable_baselines3 is not installed. RL runs will be skipped.")
        return None
    model_path = Path("models/ppo_drone_stage3_obs31_run1_best/best_model.zip")
    if not model_path.exists():
        print(f"[WARN] RL model not found: {model_path}. RL runs will be skipped.")
        return None
    print("[INFO] Loading RL model weights...")
    return PPO.load(str(model_path), device="cpu")


def run_experiment(mode_name: str, disturbance_enabled: bool = False, rl_model=None):
    variant_name = f"{mode_name}_{'gust' if disturbance_enabled else 'clean'}"
    print("\n" + "=" * 70)
    print(
        f"[RUN] Swarm matrix case: master={mode_name.upper()} "
        f"weather={'gust' if disturbance_enabled else 'clean'}"
    )
    print("=" * 70)

    config = SimulationConfig()
    config.wind_seed = 37
    config.curriculum_stage = 3
    config.enable_storms = True
    config.storm_count = 3
    config.enable_support_shield_mode = True

    if disturbance_enabled:
        config.enable_random_gusts = True
        config.gust_trigger_prob = 0.02
        config.gust_duration_s = 8.0
        config.gust_min_speed_mps = 4.0
        config.gust_max_speed_mps = 8.0
        config.gust_obs_noise_std = 0.01 if mode_name == "rl" else 0.0

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)

    swarm_executor = SwarmMissionExecutor(
        config,
        estimator,
        physics,
        battery,
        master_mode=mode_name,
        rl_model=rl_model,
    )

    start_xy = (-8000.0, -8000.0)
    goal_xy = (6000.0, 7500.0)
    mission_result = swarm_executor.execute_mission(start_xy, goal_xy)

    print("-" * 40)
    print(f"[{variant_name.upper()}] success: {mission_result.success}")
    print(f"[{variant_name.upper()}] failure_reason: {mission_result.failure_reason or 'None'}")
    print(f"[{variant_name.upper()}] mission_time_s: {mission_result.total_mission_time_s:.1f}")
    print(f"[{variant_name.upper()}] energy_kj: {mission_result.total_energy_used_j / 1000:.1f}")
    print(f"[{variant_name.upper()}] replans: {mission_result.total_replans}")
    print("-" * 40)

    output_dir = Path(f"results/swarm_test_{variant_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = Visualizer(config, estimator)
    vis.plot_swarm_execution(
        mission_result=mission_result,
        start_xy=start_xy,
        goal_xy=goal_xy,
        save_dir=str(output_dir),
    )
    vis.plot_swarm_elevation_profile(
        mission_result=mission_result,
        save_dir=str(output_dir),
    )

    animator = MissionAnimator(config, estimator)
    animator.generate_swarm_gif(
        mission_result=mission_result,
        start_xy=start_xy,
        goal_xy=goal_xy,
        filename=str(output_dir / f"swarm_dynamic_{variant_name}.gif"),
    )
    print(f"[DONE] Saved outputs to: {output_dir}")


def main() -> None:
    run_experiment("astar", disturbance_enabled=False, rl_model=None)
    run_experiment("astar", disturbance_enabled=True, rl_model=None)

    rl_model = load_rl_model()
    if rl_model is not None:
        run_experiment("rl", disturbance_enabled=False, rl_model=rl_model)
        run_experiment("rl", disturbance_enabled=True, rl_model=rl_model)


if __name__ == "__main__":
    main()

