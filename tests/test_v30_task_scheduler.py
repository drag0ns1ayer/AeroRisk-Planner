import unittest

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from v30.mission_map import ChargingStation, InspectionPoint, InspectionStatus, MissionMap
from v30.segment_executor import AStarSegmentExecutor
from v30.task_executor import SegmentExecutionResult, SimpleTaskExecutor
from v30.task_scheduler import GreedyTaskScheduler, SchedulerState, TaskSchedulerWeights


class V30TaskSchedulerTests(unittest.TestCase):
    def test_mission_map_json_roundtrip(self):
        mission_map = MissionMap(
            name="demo",
            start_xy=(0.0, 0.0),
            home_xy=(0.0, 0.0),
            inspection_points=[
                InspectionPoint(
                    id="tower-a",
                    xy=(100.0, 0.0),
                    priority=2.0,
                    service_time_s=45.0,
                    risk_value=0.2,
                    deadline_s=600.0,
                )
            ],
            charging_stations=[ChargingStation(id="charge-1", xy=(50.0, 50.0), charge_rate_j_per_s=3000.0)],
        )

        with self.subTest("json roundtrip"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "mission.json"
                mission_map.save_json(path)
                loaded = MissionMap.load_json(path)

        self.assertEqual(loaded.name, "demo")
        self.assertEqual(loaded.start_xy, (0.0, 0.0))
        self.assertEqual(loaded.inspection_points[0].id, "tower-a")
        self.assertEqual(loaded.inspection_points[0].status, InspectionStatus.PENDING)
        self.assertEqual(loaded.charging_stations[0].charge_rate_j_per_s, 3000.0)

    def test_scheduler_prefers_high_priority_feasible_inspection(self):
        cfg = SimulationConfig()
        cfg.battery_capacity_j = 1_000_000.0
        scheduler = GreedyTaskScheduler(
            cfg,
            weights=TaskSchedulerWeights(priority_gain=1000.0, energy_cost=0.0, time_cost=0.0, risk_cost=0.0),
            energy_estimator=lambda a, b: 1000.0,
            time_estimator=lambda a, b: 10.0,
        )
        mission_map = MissionMap(
            start_xy=(0.0, 0.0),
            home_xy=(0.0, 0.0),
            inspection_points=[
                InspectionPoint(id="low", xy=(100.0, 0.0), priority=1.0),
                InspectionPoint(id="high", xy=(500.0, 0.0), priority=3.0),
            ],
        )

        target = scheduler.choose_next(
            mission_map,
            SchedulerState(position_xy=(0.0, 0.0), current_time_s=0.0, remaining_energy_j=900_000.0),
        )

        self.assertEqual(target.kind, "inspection")
        self.assertEqual(target.target_id, "high")

    def test_scheduler_goes_to_charger_when_inspections_are_not_battery_feasible(self):
        cfg = SimulationConfig()
        cfg.battery_capacity_j = 100_000.0
        cfg.reserve_energy_ratio = 0.2
        battery = BatteryManager(cfg)
        scheduler = GreedyTaskScheduler(
            cfg,
            battery_manager=battery,
            energy_estimator=lambda a, b: 90_000.0 if b == (1000.0, 0.0) else 10_000.0,
            time_estimator=lambda a, b: 10.0,
        )
        mission_map = MissionMap(
            start_xy=(0.0, 0.0),
            home_xy=(0.0, 0.0),
            inspection_points=[InspectionPoint(id="far-task", xy=(1000.0, 0.0), priority=10.0)],
            charging_stations=[ChargingStation(id="charge-1", xy=(50.0, 0.0))],
        )

        target = scheduler.choose_next(
            mission_map,
            SchedulerState(position_xy=(0.0, 0.0), current_time_s=0.0, remaining_energy_j=60_000.0),
        )

        self.assertEqual(target.kind, "charging")
        self.assertEqual(target.target_id, "charge-1")

    def test_scheduler_reports_finished_when_no_pending_inspections(self):
        cfg = SimulationConfig()
        mission_map = MissionMap(
            start_xy=(0.0, 0.0),
            inspection_points=[InspectionPoint(id="done", xy=(1.0, 1.0), status=InspectionStatus.DONE)],
        )
        scheduler = GreedyTaskScheduler(cfg)

        target = scheduler.choose_next(
            mission_map,
            SchedulerState(position_xy=(0.0, 0.0), current_time_s=0.0, remaining_energy_j=cfg.battery_capacity_j),
        )

        self.assertEqual(target.kind, "finished")

    def test_simple_task_executor_completes_feasible_inspections(self):
        cfg = SimulationConfig()
        cfg.battery_capacity_j = 1_000_000.0
        cfg.max_mission_time_s = 3600.0
        scheduler = GreedyTaskScheduler(
            cfg,
            energy_estimator=lambda a, b: 1000.0,
            time_estimator=lambda a, b: 10.0,
        )
        executor = SimpleTaskExecutor(cfg, scheduler=scheduler)
        mission_map = MissionMap(
            start_xy=(0.0, 0.0),
            home_xy=(0.0, 0.0),
            inspection_points=[
                InspectionPoint(id="p1", xy=(100.0, 0.0), priority=1.0),
                InspectionPoint(id="p2", xy=(200.0, 0.0), priority=2.0),
            ],
        )

        result = executor.execute(mission_map)

        self.assertTrue(result.success)
        self.assertEqual(set(result.completed_inspections), {"p1", "p2"})
        self.assertGreater(result.total_energy_used_j, 0.0)

    def test_simple_task_executor_charges_before_far_task(self):
        cfg = SimulationConfig()
        cfg.battery_capacity_j = 100_000.0
        cfg.reserve_energy_ratio = 0.2
        cfg.max_mission_time_s = 3600.0
        task_points = {(100.0, 0.0), (200.0, 0.0)}
        scheduler = GreedyTaskScheduler(
            cfg,
            energy_estimator=lambda a, b: 40_000.0 if b in task_points else 5_000.0,
            time_estimator=lambda a, b: 10.0,
        )
        executor = SimpleTaskExecutor(cfg, scheduler=scheduler)
        mission_map = MissionMap(
            start_xy=(0.0, 0.0),
            home_xy=(0.0, 0.0),
            inspection_points=[
                InspectionPoint(id="first", xy=(100.0, 0.0), priority=5.0),
                InspectionPoint(id="second", xy=(200.0, 0.0), priority=4.0),
            ],
            charging_stations=[
                ChargingStation(id="charge-1", xy=(50.0, 0.0), charge_rate_j_per_s=10_000.0, target_soc=1.0)
            ],
        )

        result = executor.execute(mission_map)

        self.assertTrue(result.success)
        self.assertEqual(set(result.completed_inspections), {"first", "second"})
        self.assertGreaterEqual(result.charging_visits, 1)

    def test_simple_task_executor_can_use_segment_executor_adapter(self):
        class FakeSegmentExecutor:
            def __init__(self):
                self.calls = []

            def execute_leg(self, start_xy, goal_xy, start_time_s, remaining_energy_j):
                self.calls.append((start_xy, goal_xy, start_time_s, remaining_energy_j))
                return SegmentExecutionResult(
                    success=True,
                    end_position_xy=goal_xy,
                    elapsed_time_s=12.0,
                    energy_used_j=500.0,
                    remaining_energy_j=remaining_energy_j - 500.0,
                )

        cfg = SimulationConfig()
        segment_executor = FakeSegmentExecutor()
        scheduler = GreedyTaskScheduler(
            cfg,
            energy_estimator=lambda a, b: 1000.0,
            time_estimator=lambda a, b: 10.0,
        )
        executor = SimpleTaskExecutor(cfg, scheduler=scheduler, segment_executor=segment_executor)
        mission_map = MissionMap(
            start_xy=(0.0, 0.0),
            home_xy=(0.0, 0.0),
            inspection_points=[InspectionPoint(id="p1", xy=(100.0, 0.0), service_time_s=5.0)],
        )

        result = executor.execute(mission_map)

        self.assertTrue(result.success)
        self.assertEqual(len(segment_executor.calls), 1)
        self.assertEqual(result.total_time_s, 17.0)
        self.assertEqual(result.total_energy_used_j, 500.0)

    def test_astar_segment_executor_runs_short_leg(self):
        cfg = SimulationConfig()
        cfg.max_replans = 3
        cfg.max_mission_time_s = 120.0
        cfg.mission_update_interval_s = 30.0
        cfg.cruise_speed_mps = 20.0
        cfg.battery_capacity_j = 500_000.0

        map_manager = MapManager(cfg)
        wind_model = WindModelFactory.create("slope", cfg, bounds=map_manager.get_bounds())
        estimator = StateEstimator(map_manager, wind_model, cfg)
        physics = PhysicsEngine(cfg)
        battery = BatteryManager(cfg)
        planner = AStarPlanner(cfg, estimator, physics)
        segment_executor = AStarSegmentExecutor(cfg, estimator, physics, battery, planner)

        min_x, _, min_y, _ = estimator.get_bounds()
        start_xy = (min_x + 100.0, min_y + 100.0)
        goal_xy = (min_x + 250.0, min_y + 250.0)

        result = segment_executor.execute_leg(
            start_xy=start_xy,
            goal_xy=goal_xy,
            start_time_s=20.0,
            remaining_energy_j=cfg.battery_capacity_j,
        )

        self.assertTrue(result.success, result.failure_reason)
        self.assertGreaterEqual(result.elapsed_time_s, 0.0)
        self.assertLessEqual(result.remaining_energy_j, cfg.battery_capacity_j)
