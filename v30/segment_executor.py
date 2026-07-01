from __future__ import annotations

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from models.mission_models import SimulationState
from simulation.mission_executor import MissionExecutor
from v30.mission_map import Point2D
from v30.task_executor import SegmentExecutionResult


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
