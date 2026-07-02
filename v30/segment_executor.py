from __future__ import annotations

import copy
from typing import Literal

import numpy as np

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from models.mission_models import SimulationState
from simulation.mission_executor import MissionExecutor
from v25.rl_env_disruptive import GuidedDroneEnvV25
from v30.mission_map import Point2D
from v30.task_executor import SegmentExecutionResult


V25ControlMode = Literal["astar", "expert", "rl"]


class AStarSegmentExecutor:
    """
    Adapter from v3.0 semantic tasks to the legacy single-leg A* executor.

    The legacy MissionExecutor owns useful path planning and path advancement
    behavior, but its public execute_mission starts each run from full battery
    and t=0. This adapter keeps the useful internals while accepting v3.0's
    current mission time and remaining energy.
    """

    def __init__(
        self,
        config: SimulationConfig,
        estimator: StateEstimator,
        physics: PhysicsEngine,
        battery_manager: BatteryManager,
        planner: AStarPlanner,
    ):
        self.config = config
        self.estimator = estimator
        self.legacy_executor = MissionExecutor(
            config=config,
            estimator=estimator,
            physics=physics,
            battery_manager=battery_manager,
            planner=planner,
        )

    def execute_leg(
        self,
        start_xy: Point2D,
        goal_xy: Point2D,
        start_time_s: float,
        remaining_energy_j: float,
    ) -> SegmentExecutionResult:
        start_z = self.estimator.get_altitude(float(start_xy[0]), float(start_xy[1])) + self.config.takeoff_altitude_agl
        state = SimulationState(
            current_time_s=float(start_time_s),
            position_xyz=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
            remaining_energy_j=float(remaining_energy_j),
            traveled_path_xyz=[(float(start_xy[0]), float(start_xy[1]), float(start_z))],
        )
        start_energy = float(remaining_energy_j)
        segment_start_time = float(start_time_s)
        time_history_s: list[float] = []
        power_history_w: list[float] = []
        risk_history: list[float] = []

        while True:
            if self.legacy_executor._check_goal_reached(state.position_xyz, goal_xy):
                state.is_goal_reached = True
                break
            if state.replans_count >= self.config.max_replans:
                state.is_mission_failed = True
                state.failure_reason = "maximum replans exceeded"
                break
            if state.current_time_s >= self.config.max_mission_time_s:
                state.is_mission_failed = True
                state.failure_reason = "maximum mission time exceeded"
                break

            current_xy = (float(state.position_xyz[0]), float(state.position_xyz[1]))
            planned_path = self.legacy_executor._plan_once(
                current_xy,
                goal_xy,
                use_wind=True,
                start_time_s=state.current_time_s,
            )
            if not planned_path or len(planned_path) < 2:
                state.is_mission_failed = True
                state.failure_reason = "planner failed to find a path"
                break

            state = self.legacy_executor._advance_along_path(
                state=state,
                path_xyz=planned_path,
                delta_t_s=self.config.mission_update_interval_s,
                time_history_s=time_history_s,
                power_history_w=power_history_w,
                risk_history=risk_history,
            )
            state.replans_count += 1
            if state.is_mission_failed:
                break

        return SegmentExecutionResult(
            success=bool(state.is_goal_reached and not state.is_mission_failed),
            end_position_xy=(float(state.position_xyz[0]), float(state.position_xyz[1])),
            elapsed_time_s=max(0.0, float(state.current_time_s) - segment_start_time),
            energy_used_j=max(0.0, start_energy - float(state.remaining_energy_j)),
            remaining_energy_j=float(state.remaining_energy_j),
            failure_reason=state.failure_reason,
            path_xyz=[tuple(map(float, point)) for point in state.traveled_path_xyz],
        )


class V25GuidedSegmentExecutor:
    """
    Adapter from v3.0 semantic task legs to the v2.5 residual-control stack.

    Modes:
    - astar: zero residual around A* reference, optionally protected by APAS.
    - expert: v2.5 local expert / risk membrane / do-no-harm stack.
    - rl: PPO residual policy around A*, optionally protected by APAS.
    """

    def __init__(
        self,
        config: SimulationConfig,
        mode: V25ControlMode = "expert",
        model=None,
        enable_apas: bool = True,
        seed: int = 42,
    ):
        self.config = config
        self.mode = mode
        self.model = model
        self.enable_apas = bool(enable_apas)
        self.seed = int(seed)
        self.leg_index = 0

    def execute_leg(
        self,
        start_xy: Point2D,
        goal_xy: Point2D,
        start_time_s: float,
        remaining_energy_j: float,
    ) -> SegmentExecutionResult:
        run_cfg = copy.deepcopy(self.config)
        run_cfg.rl_enable_apas = bool(self.enable_apas)
        run_cfg.enable_single_agent_gusts = False
        run_cfg.enable_random_gusts = False
        run_cfg.planner_time_mode = "4d"
        run_cfg.wind_seed = int(self.seed + self.leg_index * 1009)

        env = GuidedDroneEnvV25(run_cfg)
        obs, _ = env.reset(
            seed=int(run_cfg.wind_seed),
            options={"start_xy": start_xy, "goal_xy": goal_xy},
        )
        env.energy_remaining = float(remaining_energy_j)

        start_energy = float(remaining_energy_j)
        path_xyz: list[tuple[float, float, float]] = [tuple(map(float, env.current_pos))]
        terminated = False
        truncated = False
        final_info = {"is_success": False, "terminated_reason": "unknown"}

        while not (terminated or truncated):
            if self.mode == "expert":
                action = env.local_avoidance_expert_action()
            elif self.mode == "rl":
                if self.model is None:
                    raise ValueError("V25GuidedSegmentExecutor mode='rl' requires a PPO-like model.")
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = np.zeros(3, dtype=np.float32)

            obs, _, terminated, truncated, final_info = env.step(action)
            path_xyz.append(tuple(map(float, env.current_pos)))

        self.leg_index += 1
        success = bool(final_info.get("is_success", False))
        reason = None if success else str(final_info.get("terminated_reason", "unknown"))
        remaining = float(final_info.get("energy_remaining_j", env.energy_remaining))
        return SegmentExecutionResult(
            success=success,
            end_position_xy=(float(env.current_pos[0]), float(env.current_pos[1])),
            elapsed_time_s=max(0.0, float(env.current_time)),
            energy_used_j=max(0.0, start_energy - remaining),
            remaining_energy_j=remaining,
            failure_reason=reason,
            path_xyz=path_xyz,
        )
