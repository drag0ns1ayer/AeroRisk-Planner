from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from v30.mission_map import MissionMap, Point2D
from v30.task_scheduler import GreedyTaskScheduler, ScheduledTarget, SchedulerState, euclidean_distance_m


@dataclass
class TaskExecutionEvent:
    time_s: float
    kind: str
    target_id: str | None
    position_xy: Point2D
    remaining_energy_j: float
    detail: str = ""


@dataclass
class TaskExecutionResult:
    success: bool
    completed_inspections: list[str] = field(default_factory=list)
    skipped_inspections: list[str] = field(default_factory=list)
    charging_visits: int = 0
    returned_home: bool = False
    total_time_s: float = 0.0
    total_energy_used_j: float = 0.0
    final_position_xy: Point2D = (0.0, 0.0)
    remaining_energy_j: float = 0.0
    failure_reason: str | None = None
    actual_path_xyz: list[tuple[float, float, float]] = field(default_factory=list)
    events: list[TaskExecutionEvent] = field(default_factory=list)


@dataclass
class SegmentExecutionResult:
    success: bool
    end_position_xy: Point2D
    elapsed_time_s: float
    energy_used_j: float
    remaining_energy_j: float
    failure_reason: str | None = None
    path_xyz: list[tuple[float, float, float]] = field(default_factory=list)


class SegmentExecutor(Protocol):
    def execute_leg(
        self,
        start_xy: Point2D,
        goal_xy: Point2D,
        start_time_s: float,
        remaining_energy_j: float,
    ) -> SegmentExecutionResult:
        ...


class SimpleTaskExecutor:
    """
    Lightweight semantic mission executor for v3.0.

    It does not replace the v2.5 flight stack. It validates the higher-level
    task loop: choose target, spend travel energy/time, perform inspection or
    charging, and repeat.
    """

    def __init__(
        self,
        config: SimulationConfig,
        scheduler: GreedyTaskScheduler | None = None,
        battery_manager: BatteryManager | None = None,
        segment_executor: SegmentExecutor | None = None,
        return_home: bool = True,
    ):
        self.config = config
        self.battery_manager = battery_manager or BatteryManager(config)
        self.scheduler = scheduler or GreedyTaskScheduler(config, battery_manager=self.battery_manager)
        self.segment_executor = segment_executor
        self.return_home = bool(return_home)

    def execute(self, mission_map: MissionMap, max_steps: int = 200) -> TaskExecutionResult:
        state = SchedulerState(
            position_xy=mission_map.start_xy,
            current_time_s=0.0,
            remaining_energy_j=float(self.config.battery_capacity_j),
        )
        result = TaskExecutionResult(
            success=False,
            final_position_xy=state.position_xy,
            remaining_energy_j=state.remaining_energy_j,
            actual_path_xyz=[(state.position_xy[0], state.position_xy[1], 0.0)],
        )

        for _ in range(max_steps):
            target = self.scheduler.choose_next(mission_map, state)
            if target.kind == "finished":
                if self._return_home_if_needed(mission_map, state, result):
                    result.success = True
                    result.failure_reason = None
                break
            if target.kind == "failed" or target.xy is None:
                result.failure_reason = target.reason
                break

            if not self._consume_travel(state, target, result):
                if result.failure_reason is None:
                    result.failure_reason = "battery depleted while traveling"
                break

            if target.kind == "inspection":
                self._perform_inspection(mission_map, state, target, result)
            elif target.kind == "charging":
                self._perform_charging(mission_map, state, target, result)
            else:
                result.failure_reason = f"unsupported target kind: {target.kind}"
                break

            if state.current_time_s > float(self.config.max_mission_time_s):
                result.failure_reason = "maximum mission time exceeded"
                break
        else:
            result.failure_reason = "maximum task steps exceeded"

        result.total_time_s = float(state.current_time_s)
        result.final_position_xy = state.position_xy
        result.remaining_energy_j = float(state.remaining_energy_j)
        return result

    def _return_home_if_needed(
        self,
        mission_map: MissionMap,
        state: SchedulerState,
        result: TaskExecutionResult,
    ) -> bool:
        if not self.return_home or mission_map.home_xy is None:
            return True

        home_xy = mission_map.home_xy
        if euclidean_distance_m(state.position_xy, home_xy) <= 1.0:
            result.returned_home = True
            result.events.append(
                TaskExecutionEvent(
                    time_s=float(state.current_time_s),
                    kind="return_home_done",
                    target_id="home",
                    position_xy=state.position_xy,
                    remaining_energy_j=float(state.remaining_energy_j),
                    detail="already at home",
                )
            )
            return True

        target = ScheduledTarget(
            kind="return_home",
            target_id="home",
            xy=home_xy,
            reason="return to home after all inspections",
            estimated_energy_j=float(self.scheduler.energy_estimator(state.position_xy, home_xy)),
            estimated_time_s=float(self.scheduler.time_estimator(state.position_xy, home_xy)),
            service_time_s=0.0,
        )
        if not self._consume_travel(state, target, result):
            if result.failure_reason is None:
                result.failure_reason = "failed to return home"
            return False

        result.returned_home = True
        result.events.append(
            TaskExecutionEvent(
                time_s=float(state.current_time_s),
                kind="return_home_done",
                target_id="home",
                position_xy=state.position_xy,
                remaining_energy_j=float(state.remaining_energy_j),
                detail="mission closed at home",
            )
        )
        return True

    def _consume_travel(
        self,
        state: SchedulerState,
        target: ScheduledTarget,
        result: TaskExecutionResult,
    ) -> bool:
        assert target.xy is not None
        if self.segment_executor is not None:
            segment = self.segment_executor.execute_leg(
                start_xy=state.position_xy,
                goal_xy=target.xy,
                start_time_s=state.current_time_s,
                remaining_energy_j=state.remaining_energy_j,
            )
            if not segment.success:
                result.failure_reason = segment.failure_reason
                return False
            state.position_xy = segment.end_position_xy
            state.current_time_s += float(segment.elapsed_time_s)
            state.remaining_energy_j = float(segment.remaining_energy_j)
            result.total_energy_used_j += float(segment.energy_used_j)
            if segment.path_xyz:
                if result.actual_path_xyz and result.actual_path_xyz[-1] == segment.path_xyz[0]:
                    result.actual_path_xyz.extend(segment.path_xyz[1:])
                else:
                    result.actual_path_xyz.extend(segment.path_xyz)
            else:
                result.actual_path_xyz.append((state.position_xy[0], state.position_xy[1], 0.0))
            result.events.append(
                TaskExecutionEvent(
                    time_s=float(state.current_time_s),
                    kind=f"arrive_{target.kind}",
                    target_id=target.target_id,
                    position_xy=state.position_xy,
                    remaining_energy_j=float(state.remaining_energy_j),
                    detail=f"segment_executor: {target.reason}",
                )
            )
            return True

        travel_energy = float(target.estimated_energy_j)
        if not self.battery_manager.can_consume(state.remaining_energy_j, travel_energy):
            return False
        state.remaining_energy_j = self.battery_manager.consume_energy(state.remaining_energy_j, travel_energy)
        state.current_time_s += float(target.estimated_time_s)
        state.position_xy = target.xy
        result.total_energy_used_j += travel_energy
        result.actual_path_xyz.append((state.position_xy[0], state.position_xy[1], 0.0))
        result.events.append(
            TaskExecutionEvent(
                time_s=float(state.current_time_s),
                kind=f"arrive_{target.kind}",
                target_id=target.target_id,
                position_xy=state.position_xy,
                remaining_energy_j=float(state.remaining_energy_j),
                detail=target.reason,
            )
        )
        return True

    def _perform_inspection(
        self,
        mission_map: MissionMap,
        state: SchedulerState,
        target: ScheduledTarget,
        result: TaskExecutionResult,
    ) -> None:
        assert target.target_id is not None
        point = mission_map.get_inspection(target.target_id)
        mission_map.mark_done(point.id)
        state.current_time_s += float(target.service_time_s)
        result.completed_inspections.append(point.id)
        result.events.append(
            TaskExecutionEvent(
                time_s=float(state.current_time_s),
                kind="inspection_done",
                target_id=point.id,
                position_xy=state.position_xy,
                remaining_energy_j=float(state.remaining_energy_j),
                detail=f"priority={point.priority}",
            )
        )

    def _perform_charging(
        self,
        mission_map: MissionMap,
        state: SchedulerState,
        target: ScheduledTarget,
        result: TaskExecutionResult,
    ) -> None:
        assert target.target_id is not None
        station = next(station for station in mission_map.charging_stations if station.id == target.target_id)
        state.current_time_s += float(target.service_time_s)
        target_energy = float(self.config.battery_capacity_j) * float(station.target_soc)
        missing_energy = max(0.0, target_energy - state.remaining_energy_j)
        charge_time = missing_energy / max(float(station.charge_rate_j_per_s), 1e-6)
        state.current_time_s += charge_time
        state.remaining_energy_j = min(float(self.config.battery_capacity_j), target_energy)
        result.charging_visits += 1
        result.events.append(
            TaskExecutionEvent(
                time_s=float(state.current_time_s),
                kind="charging_done",
                target_id=station.id,
                position_xy=state.position_xy,
                remaining_energy_j=float(state.remaining_energy_j),
                detail=f"charged_j={missing_energy:.1f}",
            )
        )
