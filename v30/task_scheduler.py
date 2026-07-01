from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from v30.mission_map import ChargingStation, InspectionPoint, MissionMap, Point2D


TargetKind = Literal["inspection", "charging", "finished", "failed"]


@dataclass
class TaskSchedulerWeights:
    priority_gain: float = 1000.0
    energy_cost: float = 1.0e-3
    time_cost: float = 0.2
    risk_cost: float = 350.0
    deadline_lateness_cost: float = 0.6


@dataclass
class SchedulerState:
    position_xy: Point2D
    current_time_s: float
    remaining_energy_j: float


@dataclass
class ScheduledTarget:
    kind: TargetKind
    target_id: str | None = None
    xy: Point2D | None = None
    score: float = 0.0
    reason: str = ""
    estimated_energy_j: float = 0.0
    estimated_time_s: float = 0.0


EnergyEstimator = Callable[[Point2D, Point2D], float]
TimeEstimator = Callable[[Point2D, Point2D], float]


def euclidean_distance_m(a: Point2D, b: Point2D) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def default_energy_estimator(config: SimulationConfig) -> EnergyEstimator:
    # Conservative but cheap scheduler estimate. Real execution can later call
    # A*/Physics; the scheduler only needs an ordering and feasibility screen.
    energy_per_m = float(config.battery_capacity_j) / 100000.0

    def estimate(start_xy: Point2D, goal_xy: Point2D) -> float:
        return euclidean_distance_m(start_xy, goal_xy) * energy_per_m

    return estimate


def default_time_estimator(config: SimulationConfig) -> TimeEstimator:
    speed = max(float(config.cruise_speed_mps), 1e-6)

    def estimate(start_xy: Point2D, goal_xy: Point2D) -> float:
        return euclidean_distance_m(start_xy, goal_xy) / speed

    return estimate


class GreedyTaskScheduler:
    """
    First-pass v3.0 scheduler.

    It chooses one next semantic target at a time. The rule is intentionally
    simple: inspect the best feasible point; if none is safe with battery
    reserve and a post-task escape route, go charge.
    """

    def __init__(
        self,
        config: SimulationConfig,
        battery_manager: BatteryManager | None = None,
        weights: TaskSchedulerWeights | None = None,
        energy_estimator: EnergyEstimator | None = None,
        time_estimator: TimeEstimator | None = None,
    ):
        self.config = config
        self.battery_manager = battery_manager or BatteryManager(config)
        self.weights = weights or TaskSchedulerWeights()
        self.energy_estimator = energy_estimator or default_energy_estimator(config)
        self.time_estimator = time_estimator or default_time_estimator(config)

    def choose_next(self, mission_map: MissionMap, state: SchedulerState) -> ScheduledTarget:
        pending = mission_map.pending_inspections()
        if not pending:
            return ScheduledTarget(kind="finished", reason="all inspections completed")

        best_inspection = self._best_feasible_inspection(mission_map, state, pending)
        if best_inspection is not None:
            return best_inspection

        best_charger = self._best_feasible_charger(mission_map, state)
        if best_charger is not None:
            return best_charger

        return ScheduledTarget(kind="failed", reason="no feasible inspection or charging station")

    def _best_feasible_inspection(
        self,
        mission_map: MissionMap,
        state: SchedulerState,
        pending: list[InspectionPoint],
    ) -> ScheduledTarget | None:
        candidates: list[ScheduledTarget] = []
        for point in pending:
            leg_energy = self.energy_estimator(state.position_xy, point.xy)
            leg_time = self.time_estimator(state.position_xy, point.xy)
            escape_energy = self._nearest_escape_energy(point.xy, mission_map)
            required_energy = leg_energy + escape_energy
            if not self.battery_manager.can_consume(state.remaining_energy_j, required_energy):
                continue

            finish_time = state.current_time_s + leg_time + max(float(point.service_time_s), 0.0)
            deadline_lateness = (
                max(0.0, finish_time - float(point.deadline_s)) if point.deadline_s is not None else 0.0
            )
            score = (
                self.weights.priority_gain * float(point.priority)
                - self.weights.energy_cost * leg_energy
                - self.weights.time_cost * (leg_time + max(float(point.service_time_s), 0.0))
                - self.weights.risk_cost * float(point.risk_value)
                - self.weights.deadline_lateness_cost * deadline_lateness
            )
            candidates.append(
                ScheduledTarget(
                    kind="inspection",
                    target_id=point.id,
                    xy=point.xy,
                    score=float(score),
                    reason="best feasible inspection",
                    estimated_energy_j=float(leg_energy),
                    estimated_time_s=float(leg_time + max(float(point.service_time_s), 0.0)),
                )
            )

        if not candidates:
            return None
        return max(candidates, key=lambda item: item.score)

    def _best_feasible_charger(self, mission_map: MissionMap, state: SchedulerState) -> ScheduledTarget | None:
        candidates: list[ScheduledTarget] = []
        for charger in mission_map.available_chargers():
            leg_energy = self.energy_estimator(state.position_xy, charger.xy)
            if not self.battery_manager.can_consume(state.remaining_energy_j, leg_energy):
                continue
            leg_time = self.time_estimator(state.position_xy, charger.xy)
            candidates.append(
                ScheduledTarget(
                    kind="charging",
                    target_id=charger.id,
                    xy=charger.xy,
                    score=-float(leg_energy),
                    reason="charge before next inspection",
                    estimated_energy_j=float(leg_energy),
                    estimated_time_s=float(leg_time + max(float(charger.docking_time_s), 0.0)),
                )
            )

        if not candidates:
            return None
        return max(candidates, key=lambda item: item.score)

    def _nearest_escape_energy(self, from_xy: Point2D, mission_map: MissionMap) -> float:
        escape_targets = [station.xy for station in mission_map.available_chargers()]
        if mission_map.home_xy is not None:
            escape_targets.append(mission_map.home_xy)
        if not escape_targets:
            return 0.0
        return min(self.energy_estimator(from_xy, target_xy) for target_xy in escape_targets)
