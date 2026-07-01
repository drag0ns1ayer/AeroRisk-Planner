import unittest

import numpy as np

from configs.config import SimulationConfig
from v25.apas_safety import build_apas_candidate_info, generate_apas_candidates, probe_random_layer_segment
from v25.control_helpers import compute_evaluation_costs
from v25.disruptions import build_disruption_layer_v25
from v25.episode_metrics import reset_v25_episode_metrics, reset_v25_runtime_trackers
from v25.expert_policy import expert_candidate_actions, select_expert_action_from_evaluations
from v25.risk_membrane import compute_risk_membrane_summary, risk_membrane_action
from v25.rl_env_disruptive import (
    GuidedDroneEnvV25,
    compute_intervention_need,
    compute_residual_control_cost,
    compute_residual_gate,
)
from v25.true_world_dynamics import TrueWorldDynamicsV25, VehicleStateV25


class V25TrueWorldTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()

    def test_random_layer_is_seeded(self):
        a = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        b = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        c = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=8)

        self.assertTrue(np.allclose(a.storm_mutation.center_xy_t0, b.storm_mutation.center_xy_t0))
        self.assertAlmostEqual(a.storm_mutation.strength_mps, b.storm_mutation.strength_mps)
        self.assertFalse(np.allclose(a.storm_mutation.center_xy_t0, c.storm_mutation.center_xy_t0))
        self.assertGreaterEqual(len(a.local_wind_regions), self.config.v25_local_wind_region_count_min)
        self.assertLessEqual(len(a.local_wind_regions), self.config.v25_local_wind_region_count_max)

    def test_random_layer_is_controller_independent(self):
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        pulse = layer.headwind_pulse
        t_s = 0.5 * (pulse.start_time_s + pulse.end_time_s)
        wind_a = pulse.wind_at(pulse.center_xy[0], pulse.center_xy[1], t_s, np.array([1.0, 0.0]))
        wind_b = pulse.wind_at(pulse.center_xy[0], pulse.center_xy[1], t_s, np.array([0.0, 1.0]))
        self.assertTrue(np.allclose(wind_a, wind_b))

    def test_hard_stress_adds_hidden_wind_regions(self):
        self.config.v25_disruption_stress_level = "hard"
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        self.assertGreaterEqual(len(layer.local_wind_regions), 4)

    def test_fragile_stress_places_hidden_wind_near_route(self):
        self.config.v25_disruption_stress_level = "fragile"
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        route_y_distances = [abs(float(region.center_xy[1])) for region in layer.local_wind_regions]
        self.assertLess(max(route_y_distances), 500.0)

    def test_destructive_storm_core_is_dangerous(self):
        self.config.v25_destructive_storm_probability = 1.0
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        storm = layer.destructive_storm
        t_s = storm.trigger_time_s
        center = storm.center_at(t_s)
        self.assertIsNotNone(center)
        self.assertTrue(layer.core_hit(float(center[0]), float(center[1]), t_s))
        self.assertGreater(layer.risk_bonus_at(float(center[0]), float(center[1]), t_s), 0.5)

    def test_destructive_storm_has_warning_halo_outside_core(self):
        self.config.v25_destructive_storm_probability = 1.0
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        storm = layer.destructive_storm
        t_s = storm.trigger_time_s
        center = storm.center_at(t_s)
        self.assertIsNotNone(center)

        direction = np.array([1.0, 0.0], dtype=float)
        halo_radius = 0.5 * (storm.core_radius_m + storm.halo_radius_m)
        halo_point = center + direction * halo_radius

        self.assertFalse(layer.core_hit(float(halo_point[0]), float(halo_point[1]), t_s))
        self.assertGreaterEqual(
            storm.danger_at(float(halo_point[0]), float(halo_point[1]), t_s),
            self.config.v25_destructive_storm_halo_danger_floor,
        )
        self.assertGreater(layer.risk_bonus_at(float(halo_point[0]), float(halo_point[1]), t_s), 0.0)

    def test_circular_radar_reports_sector_hazard(self):
        self.config.v25_destructive_storm_probability = 1.0
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        storm = layer.destructive_storm
        t_s = storm.trigger_time_s
        center = storm.center_at(t_s)
        origin = center - np.array([0.5 * self.config.v25_radar_radius_m, 0.0])
        sector_values = [
            layer.radar_sector(
                origin_xy=origin,
                heading_deg=0.0,
                sector_idx=i,
                sector_count=self.config.v25_radar_sectors,
                radius_m=self.config.v25_radar_radius_m,
                t_s=t_s,
            )
            for i in range(self.config.v25_radar_sectors)
        ]
        self.assertGreater(max(value[2] for value in sector_values), 0.5)

    def test_v25_observation_matches_configured_sensor_size(self):
        config = SimulationConfig()
        config.curriculum_stage = 1
        env = GuidedDroneEnvV25(config)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape[0], 31 + GuidedDroneEnvV25._sensor_feature_count(config))

    def test_circle_oracle_reports_local_danger(self):
        self.config.v25_sensor_mode = "circle_oracle"
        self.config.v25_destructive_storm_probability = 1.0
        layer = build_disruption_layer_v25((0.0, 0.0), (4000.0, 0.0), config=self.config, seed=7)
        storm = layer.destructive_storm
        t_s = storm.trigger_time_s
        center = storm.center_at(t_s)
        self.assertIsNotNone(center)

        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.disruptions = layer
        env.current_pos = np.array(
            [center[0] - 0.5 * self.config.v25_radar_radius_m, center[1], 100.0],
            dtype=float,
        )
        env.current_heading = 0.0
        env.current_time = t_s

        features = env._circle_oracle_features()
        self.assertEqual(len(features), GuidedDroneEnvV25.CIRCLE_ORACLE_FEATURES)
        self.assertGreater(features[0], 0.5)
        self.assertGreater(features[2], 0.0)

    def test_circle_oracle_samples_front_arc_densely(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env.current_heading = 0.0

        points = env._circle_oracle_sample_points()
        offsets = [point - env.current_pos[:2] for point in points[1:]]
        front_count = sum(
            1
            for offset in offsets
            if np.linalg.norm(offset) > 1e-6
            and offset[0] / np.linalg.norm(offset) > np.cos(np.radians(60.0))
        )

        self.assertEqual(len(points), self.config.v25_circle_oracle_samples)
        self.assertGreaterEqual(front_count, 60)

    def test_local_hazard_history_reports_positive_trend(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.local_hazard_history = [
            {"max_danger": 0.10, "forward_danger": 0.04, "nearest_closeness": 0.20}
        ]
        current = {
            "max_danger": 0.24,
            "forward_danger": 0.18,
            "nearest_closeness": 0.40,
        }

        trend = env._local_hazard_history_features(current)

        self.assertGreater(trend["trend_need"], 0.0)
        self.assertGreater(trend["delta_max_danger"], 0.0)
        self.assertGreater(trend["delta_forward_danger"], 0.0)
        self.assertGreater(trend["delta_nearest_closeness"], 0.0)

    def test_gradual_warning_requires_trend_before_hard_danger(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        summary = {
            "hazard_need": self.config.v25_expert_activation_hazard * 0.5,
            "base_hazard_need": self.config.v25_expert_activation_hazard * 0.5,
            "max_danger": 0.20,
            "forward_danger": 0.16,
            "nearest_closeness": 0.40,
            "trend_need": self.config.v25_expert_trend_warning_need + 0.05,
            "delta_forward_danger": self.config.v25_expert_trend_forward_delta + 0.01,
        }

        self.assertEqual(env._local_hazard_gradual_warning(summary), 1.0)

        summary["base_hazard_need"] = self.config.v25_expert_activation_hazard + 0.01
        self.assertEqual(env._local_hazard_gradual_warning(summary), 0.0)

    def test_reset_clears_local_hazard_history(self):
        self.config.curriculum_stage = 1
        env = GuidedDroneEnvV25(self.config)
        env.local_hazard_history.append(
            {"max_danger": 1.0, "forward_danger": 1.0, "nearest_closeness": 1.0}
        )

        env.reset(seed=42)

        self.assertEqual(len(env.local_hazard_history), 0)

    def test_episode_metric_reset_helpers_clear_key_v25_counters(self):
        class Dummy:
            pass

        target = Dummy()
        target.episode_apas_interventions = 99
        target.episode_eval_adjusted_energy_j = 123.0
        target.episode_replans = 3
        target.replan_cooldown_steps_remaining = 7
        target.last_replan_event = "path_drift"

        reset_v25_episode_metrics(target)
        reset_v25_runtime_trackers(target)

        self.assertEqual(target.episode_apas_interventions, 0)
        self.assertEqual(target.episode_eval_adjusted_energy_j, 0.0)
        self.assertEqual(target.episode_replans, 0)
        self.assertEqual(target.replan_cooldown_steps_remaining, 0)
        self.assertEqual(target.last_replan_event, "none")

    def test_true_wind_changes_ground_track(self):
        dynamics = TrueWorldDynamicsV25(self.config)
        state = VehicleStateV25(
            position_xyz=np.array([0.0, 0.0, 100.0]),
            heading_deg=0.0,
            airspeed_mps=10.0,
        )
        calm = dynamics.advance(state, 0.0, 10.0, 100.0, np.zeros(2), 2.0)
        crosswind = dynamics.advance(state, 0.0, 10.0, 100.0, np.array([0.0, 5.0]), 2.0)

        self.assertAlmostEqual(calm["position_xyz"][1], 0.0)
        self.assertAlmostEqual(crosswind["position_xyz"][1], 10.0)
        self.assertGreater(np.linalg.norm(crosswind["position_xyz"] - calm["position_xyz"]), 0.0)

    def test_turn_and_acceleration_are_limited(self):
        dynamics = TrueWorldDynamicsV25(self.config)
        state = VehicleStateV25(
            position_xyz=np.array([0.0, 0.0, 100.0]),
            heading_deg=0.0,
            airspeed_mps=5.0,
        )
        result = dynamics.advance(state, 180.0, 20.0, 100.0, np.zeros(2), 1.0)

        self.assertLessEqual(abs(result["heading_deg"]), self.config.v25_max_turn_rate_deg_s)
        self.assertLessEqual(result["airspeed_mps"] - state.airspeed_mps, self.config.v25_max_accel_mps2)

    def test_intervention_need_rises_with_observed_error(self):
        calm = compute_intervention_need(0.0, 0.0, 0.0, self.config)
        disturbed = compute_intervention_need(12.0, 0.0, 0.0, self.config)
        off_path = compute_intervention_need(0.0, 0.0, 1200.0, self.config)
        fully_needed = compute_intervention_need(12.0, 20.0, 1200.0, self.config)
        self.assertEqual(calm, 0.0)
        self.assertGreater(disturbed, off_path)
        self.assertEqual(fully_needed, 1.0)

    def test_intervention_need_uses_local_hazard(self):
        calm = compute_intervention_need(0.0, 0.0, 0.0, self.config, local_hazard_need=0.0)
        hazardous = compute_intervention_need(0.0, 0.0, 0.0, self.config, local_hazard_need=1.0)

        self.assertEqual(calm, 0.0)
        self.assertGreater(hazardous, calm)
        self.assertGreater(compute_residual_gate(hazardous, self.config), compute_residual_gate(calm, self.config))

    def test_residual_action_cost_is_higher_when_unneeded(self):
        action = np.array([0.5, 0.0, 0.0])
        previous_action = np.zeros(3)
        calm_cost, _, _ = compute_residual_control_cost(action, previous_action, 0.0, self.config)
        needed_cost, _, _ = compute_residual_control_cost(action, previous_action, 1.0, self.config)
        self.assertGreater(calm_cost, needed_cost)

    def test_residual_gate_expands_with_intervention_need(self):
        calm_gate = compute_residual_gate(0.0, self.config)
        mid_gate = compute_residual_gate(0.5, self.config)
        high_gate = compute_residual_gate(1.0, self.config)

        self.assertAlmostEqual(calm_gate, self.config.v25_residual_gate_min_scale)
        self.assertGreater(mid_gate, calm_gate)
        self.assertEqual(high_gate, 1.0)

    def test_local_avoidance_expert_prefers_safer_turn(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env.current_heading = 0.0
        env.current_airspeed = 12.0
        env.current_time = 0.0
        env.dt = self.config.rl_dt
        env.current_wp_idx = 0
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.last_ground_velocity_xy = np.array([12.0, 0.0], dtype=float)
        env.goal_pos = np.array([1000.0, 0.0, 100.0], dtype=float)
        env.min_clearance_agl = self.config.rl_min_clearance_agl
        env.max_clearance_agl = self.config.rl_max_clearance_agl
        env._astar_base_command = lambda: {
            "heading_deg": 0.0,
            "airspeed_mps": 12.0,
            "agl_m": 50.0,
            "desired_ground_velocity_xy": np.array([12.0, 0.0], dtype=float),
        }
        env._measured_residual_wind = lambda: np.zeros(2, dtype=float)
        env._tracking_velocity_error = lambda base_command: np.zeros(2, dtype=float)
        env._estimate_local_path_error = lambda pos_xy: 0.0
        env._local_hazard_summary = lambda: {
            "hazard_need": 1.0,
            "max_danger": 1.0,
            "forward_danger": 1.0,
            "nearest_closeness": 1.0,
            "nearest_forward_alignment": 1.0,
        }
        env._current_nfz_list = lambda: []

        class FakeMap:
            def is_collision(self, *args, **kwargs):
                return False

        class FakeEstimator:
            map = FakeMap()

        env.estimator = FakeEstimator()

        def fake_transition(heading, speed, agl, random_layer_time_s=None):
            straight_risk = 0.9 if abs(heading) < 1e-6 else 0.05
            return {
                "position_xyz": np.array([speed, 0.0, 100.0], dtype=float),
                "heading_deg": heading,
                "airspeed_mps": speed,
                "ground_velocity_xyz": np.array([speed, 0.0, 0.0], dtype=float),
                "p_crash": straight_risk,
                "power_w": self.config.base_power,
                "destructive_core_hit": abs(heading) < 1e-6,
            }

        env._simulate_true_transition = fake_transition
        straight_score = env._score_expert_rollout(np.array([0.0, 0.0, 0.0], dtype=float))
        turn_score = env._score_expert_rollout(np.array([0.65, 0.0, 0.0], dtype=float))
        action = env.local_avoidance_expert_action()

        self.assertGreater(straight_score, turn_score + self.config.v25_expert_hard_constraint_penalty)
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        self.assertNotAlmostEqual(float(action[0]), 0.0)

    def test_expert_rollout_penalizes_terminal_path_error(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env.current_heading = 0.0
        env.current_airspeed = 12.0
        env.current_time = 0.0
        env.dt = self.config.rl_dt
        env.current_wp_idx = 0
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.last_ground_velocity_xy = np.array([12.0, 0.0], dtype=float)
        env.goal_pos = np.array([1000.0, 0.0, 100.0], dtype=float)
        env.min_clearance_agl = self.config.rl_min_clearance_agl
        env.max_clearance_agl = self.config.rl_max_clearance_agl
        env._astar_base_command = lambda: {
            "heading_deg": 0.0,
            "airspeed_mps": 12.0,
            "agl_m": 50.0,
            "desired_ground_velocity_xy": np.array([12.0, 0.0], dtype=float),
        }
        env._current_nfz_list = lambda: []

        class FakeMap:
            def is_collision(self, *args, **kwargs):
                return False

        class FakeEstimator:
            map = FakeMap()

        env.estimator = FakeEstimator()
        env._estimate_local_path_error = lambda pos_xy: abs(float(pos_xy[1]))

        def fake_transition(heading, speed, agl, random_layer_time_s=None):
            lateral = 250.0 if abs(heading) > 1e-6 else 0.0
            return {
                "position_xyz": np.array([speed, lateral, 100.0], dtype=float),
                "heading_deg": heading,
                "airspeed_mps": speed,
                "ground_velocity_xyz": np.array([speed, 0.0, 0.0], dtype=float),
                "p_crash": 0.0,
                "power_w": self.config.base_power,
                "destructive_core_hit": False,
            }

        env._simulate_true_transition = fake_transition
        straight_score = env._score_expert_rollout(np.array([0.0, 0.0, 0.0], dtype=float))
        detour_score = env._score_expert_rollout(np.array([0.65, 0.0, 0.0], dtype=float))

        self.assertGreater(detour_score, straight_score)

    def test_expert_rollout_freezes_random_layer_time(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env.current_heading = 0.0
        env.current_airspeed = 12.0
        env.current_time = 123.0
        env.dt = self.config.rl_dt
        env.current_wp_idx = 0
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.last_ground_velocity_xy = np.array([12.0, 0.0], dtype=float)
        env.goal_pos = np.array([1000.0, 0.0, 100.0], dtype=float)
        env.min_clearance_agl = self.config.rl_min_clearance_agl
        env.max_clearance_agl = self.config.rl_max_clearance_agl
        env._astar_base_command = lambda: {
            "heading_deg": 0.0,
            "airspeed_mps": 12.0,
            "agl_m": 50.0,
            "desired_ground_velocity_xy": np.array([12.0, 0.0], dtype=float),
        }
        env._current_nfz_list = lambda: []
        env._estimate_local_path_error = lambda pos_xy: 0.0

        class FakeMap:
            def is_collision(self, *args, **kwargs):
                return False

        class FakeEstimator:
            map = FakeMap()

        env.estimator = FakeEstimator()
        seen_times = []

        def fake_transition(heading, speed, agl, random_layer_time_s=None):
            seen_times.append((float(env.current_time), random_layer_time_s))
            return {
                "position_xyz": np.array([speed, 0.0, 100.0], dtype=float),
                "heading_deg": heading,
                "airspeed_mps": speed,
                "ground_velocity_xyz": np.array([speed, 0.0, 0.0], dtype=float),
                "p_crash": 0.0,
                "power_w": self.config.base_power,
                "destructive_core_hit": False,
            }

        env._simulate_true_transition = fake_transition
        env._score_expert_rollout(np.array([0.0, 0.0, 0.0], dtype=float))

        self.assertEqual(len(seen_times), self.config.v25_expert_rollout_horizon)
        self.assertTrue(any(current_time > 123.0 for current_time, _ in seen_times))
        self.assertTrue(all(random_time == 123.0 for _, random_time in seen_times))

    def test_segment_probe_detects_core_between_endpoints(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.current_time = 77.0

        class FakeDisruptions:
            def risk_bonus_at(self, x, y, t_s):
                return 0.95 if 45.0 <= x <= 55.0 and abs(y) <= 5.0 and t_s == 77.0 else 0.0

            def core_hit(self, x, y, t_s):
                return bool(45.0 <= x <= 55.0 and abs(y) <= 5.0 and t_s == 77.0)

        env.disruptions = FakeDisruptions()
        probe = env._probe_random_layer_segment(
            np.array([0.0, 0.0, 100.0], dtype=float),
            np.array([100.0, 0.0, 100.0], dtype=float),
            random_layer_time_s=77.0,
            samples=20,
        )

        self.assertTrue(probe["destructive_core_hit"])
        self.assertGreater(probe["max_risk_bonus"], self.config.rl_terminate_risk_threshold)

    def test_segment_probe_helper_detects_midpoint_core(self):
        probe = probe_random_layer_segment(
            start_xyz=np.array([0.0, 0.0, 100.0], dtype=float),
            end_xyz=np.array([100.0, 0.0, 100.0], dtype=float),
            probe_time_s=0.0,
            sample_count=4,
            risk_bonus_at=lambda x, y, t: 0.8 if abs(x - 50.0) < 1e-6 else 0.0,
            core_hit_at=lambda x, y, t: abs(x - 50.0) < 1e-6,
        )

        self.assertTrue(probe["destructive_core_hit"])
        self.assertEqual(probe["max_risk_bonus"], 0.8)

    def test_apas_candidate_info_records_intervention_cost_terms(self):
        info = build_apas_candidate_info(
            candidate_index=3,
            heading_offset_deg=15.0,
            desired_airspeed_mps=14.0,
            test_speed_mps=10.0,
            desired_agl_m=80.0,
            test_agl_m=110.0,
            segment_rejections=2,
            endpoint_rejections=1,
            segment_probe={"max_risk_bonus": 0.4, "destructive_core_hit": False},
        )

        self.assertTrue(info["apas_intervened"])
        self.assertEqual(info["apas_speed_reduction_mps"], 4.0)
        self.assertEqual(info["apas_agl_increment_m"], 30.0)
        self.assertEqual(info["apas_segment_rejections"], 2)
        self.assertEqual(info["apas_segment_max_risk_bonus"], 0.4)

    def test_apas_candidate_generation_preserves_first_nominal_candidate(self):
        candidates = generate_apas_candidates(
            desired_heading_deg=10.0,
            desired_airspeed_mps=14.0,
            desired_agl_m=80.0,
            min_clearance_agl=30.0,
            max_clearance_agl=200.0,
            config=self.config,
            wrap_angle=lambda angle: ((angle + 180.0) % 360.0) - 180.0,
        )

        self.assertGreater(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_index"], 0)
        self.assertEqual(candidates[0]["heading_deg"], 10.0)
        self.assertEqual(candidates[0]["airspeed_mps"], 14.0)
        self.assertEqual(candidates[0]["agl_m"], 80.0)
        self.assertLess(candidates[1]["airspeed_mps"], candidates[0]["airspeed_mps"])

    def test_expert_uses_recovering_mode_when_low_risk_but_off_path(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env._astar_base_command = lambda: {
            "heading_deg": 0.0,
            "airspeed_mps": 12.0,
            "agl_m": 50.0,
            "desired_ground_velocity_xy": np.array([12.0, 0.0], dtype=float),
        }
        env._measured_residual_wind = lambda: np.zeros(2, dtype=float)
        env._tracking_velocity_error = lambda base_command: np.zeros(2, dtype=float)
        env._estimate_local_path_error = lambda pos_xy: self.config.v25_expert_recovery_path_error_m + 1.0
        env._local_hazard_summary = lambda: {
            "hazard_need": 0.0,
            "max_danger": 0.0,
            "forward_danger": 0.0,
            "nearest_closeness": 0.0,
            "nearest_forward_alignment": 0.0,
        }

        action = env.local_avoidance_expert_action()

        self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
        self.assertEqual(env.last_expert_mode, "recovering")
        self.assertTrue(env.last_expert_active)

    def test_rejoin_does_not_trigger_from_cautious_by_default(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.consecutive_avoiding_steps = 0
        env.consecutive_cautious_steps = self.config.v25_expert_rejoin_max_cautious_steps
        env.consecutive_low_progress_steps = 0

        should_rejoin = env._should_rejoin(
            {
                "hazard_need": self.config.v25_expert_activation_hazard + 0.01,
                "max_danger": self.config.v25_expert_hard_risk_threshold - 0.01,
            },
            path_error_m=0.0,
        )

        self.assertFalse(should_rejoin)

    def test_rejoin_can_experimentally_trigger_after_long_cautious_loitering(self):
        env = object.__new__(GuidedDroneEnvV25)
        self.config.v25_expert_rejoin_from_cautious = True
        env.config = self.config
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.consecutive_avoiding_steps = 0
        env.consecutive_cautious_steps = self.config.v25_expert_rejoin_max_cautious_steps
        env.consecutive_low_progress_steps = 0

        should_rejoin = env._should_rejoin(
            {
                "hazard_need": self.config.v25_expert_activation_hazard + 0.01,
                "max_danger": self.config.v25_expert_hard_risk_threshold - 0.01,
            },
            path_error_m=0.0,
        )

        self.assertTrue(should_rejoin)

    def test_rejoin_does_not_trigger_from_stalled_progress_by_default(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.consecutive_avoiding_steps = 0
        env.consecutive_cautious_steps = 0
        env.consecutive_low_progress_steps = self.config.v25_expert_rejoin_max_low_progress_steps

        should_rejoin = env._should_rejoin(
            {
                "hazard_need": self.config.v25_expert_activation_hazard + 0.01,
                "max_danger": self.config.v25_expert_hard_risk_threshold - 0.01,
            },
            path_error_m=0.0,
        )

        self.assertFalse(should_rejoin)

    def test_rejoin_does_not_override_hard_danger(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.consecutive_avoiding_steps = self.config.v25_expert_rejoin_max_avoiding_steps
        env.consecutive_cautious_steps = self.config.v25_expert_rejoin_max_cautious_steps
        env.consecutive_low_progress_steps = self.config.v25_expert_rejoin_max_low_progress_steps

        should_rejoin = env._should_rejoin(
            {
                "hazard_need": self.config.v25_expert_hard_risk_threshold,
                "max_danger": self.config.v25_expert_hard_risk_threshold,
            },
            path_error_m=self.config.v25_expert_recovery_path_error_m + 1.0,
        )

        self.assertFalse(should_rejoin)

    def test_replan_trigger_waits_for_low_risk_when_path_drifted(self):
        env = object.__new__(GuidedDroneEnvV25)
        self.config.v25_replan_enabled = True
        env.config = self.config
        env.global_astar_path = [(0.0, 0.0, 100.0), (1000.0, 0.0, 100.0)]
        env.episode_replans = 0
        env.replan_cooldown_steps_remaining = 0
        env.current_step = self.config.v25_replan_min_step
        env.apas_no_valid_linger_steps = 0
        env.consecutive_replan_low_progress_steps = 0

        hard_reason = env._replan_trigger_reason(
            {
                "hazard_need": self.config.v25_replan_hard_risk_threshold,
                "max_danger": self.config.v25_replan_hard_risk_threshold,
            },
            path_error_m=self.config.v25_replan_path_error_m + 1.0,
        )
        low_risk_reason = env._replan_trigger_reason(
            {"hazard_need": 0.0, "max_danger": 0.0},
            path_error_m=self.config.v25_replan_path_error_m + 1.0,
        )

        self.assertIsNone(hard_reason)
        self.assertEqual(low_risk_reason, "path_drift")

    def test_replan_to_rejoin_splices_new_prefix_with_original_suffix(self):
        env = object.__new__(GuidedDroneEnvV25)
        self.config.v25_replan_enabled = True
        self.config.v25_replan_rejoin_lookahead_wps = 1
        env.config = self.config
        env.current_pos = np.array([20.0, 50.0, 100.0], dtype=float)
        env.current_time = 10.0
        env.current_wp_idx = 1
        env.global_astar_path = [
            (0.0, 0.0, 100.0),
            (100.0, 0.0, 100.0),
            (200.0, 0.0, 100.0),
            (300.0, 0.0, 100.0),
        ]
        env.goal_pos = np.array([300.0, 0.0, 100.0], dtype=float)
        env.episode_replan_to_rejoin_successes = 0
        env.episode_replan_to_goal_successes = 0

        class FakePlanner:
            def search(self, start_xy, goal_xy, start_time_s=0.0):
                if np.allclose(goal_xy, (200.0, 0.0)):
                    return [(20.0, 50.0, 100.0), (200.0, 0.0, 100.0)]
                return None

        env.planner = FakePlanner()

        success = env._try_replan_to_rejoin("path_drift")

        self.assertTrue(success)
        self.assertEqual(env.episode_replan_to_rejoin_successes, 1)
        self.assertEqual(env.global_astar_path[-1], (300.0, 0.0, 100.0))
        self.assertEqual(env.current_wp_idx, 1)
        self.assertEqual(env.last_replan_event, "path_drift:to_rejoin")

    def test_expert_switches_to_emergency_when_normal_candidates_are_invalid(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        normal_action = np.array([0.0, 0.0, 0.0], dtype=float)
        emergency_action = np.array([1.0, -1.0, 0.0], dtype=float)

        def fake_candidates(emergency=False, mild=False):
            return [emergency_action] if emergency else [normal_action]

        def fake_evaluate(action):
            is_emergency = np.allclose(action, emergency_action)
            return {
                "action": np.asarray(action, dtype=float),
                "score": 10.0 if is_emergency else 1.0,
                "hard_violation_count": 0 if is_emergency else 1,
                "max_risk": 0.1 if is_emergency else 0.9,
                "final_path_error": 0.0,
                "final_goal_dist": 0.0,
            }

        env._expert_candidate_actions = fake_candidates
        env._evaluate_expert_rollout = fake_evaluate

        action, mode = env._select_expert_action()

        self.assertEqual(mode, "emergency")
        self.assertTrue(np.allclose(action, emergency_action))

    def test_expert_stays_cautious_when_zero_action_is_safe_without_clear_improvement(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        zero_action = np.array([0.0, 0.0, 0.0], dtype=float)
        candidate_action = np.array([0.65, 0.0, 0.0], dtype=float)

        def fake_candidates(emergency=False, mild=False):
            return [] if emergency else [zero_action, candidate_action]

        def fake_evaluate(action):
            is_zero = np.allclose(action, zero_action)
            return {
                "action": np.asarray(action, dtype=float),
                "score": 1.0 if is_zero else 0.5,
                "hard_violation_count": 0,
                "max_risk": 0.30 if is_zero else 0.27,
                "final_path_error": 0.0,
                "final_goal_dist": 0.0,
            }

        env._expert_candidate_actions = fake_candidates
        env._evaluate_expert_rollout = fake_evaluate

        action, mode = env._select_expert_action()

        self.assertEqual(mode, "cautious")
        self.assertTrue(np.allclose(action, zero_action))

    def test_gradual_warning_uses_mild_candidates_and_cautious_trend_mode(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        zero_action = np.array([0.0, 0.0, 0.0], dtype=float)
        mild_action = np.array([0.35, 0.0, 0.0], dtype=float)
        calls = []

        def fake_candidates(emergency=False, mild=False):
            calls.append((emergency, mild))
            if emergency:
                return [np.array([1.0, -1.0, 0.0], dtype=float)]
            return [zero_action, mild_action] if mild else [np.array([0.65, 0.0, 0.0], dtype=float)]

        def fake_evaluate(action):
            is_zero = np.allclose(action, zero_action)
            is_mild = np.allclose(action, mild_action)
            return {
                "action": np.asarray(action, dtype=float),
                "score": 1.0 if is_zero else 0.5,
                "hard_violation_count": 0,
                "max_risk": 0.30 if is_zero else (0.25 if is_mild else 0.10),
                "final_path_error": 0.0,
                "final_goal_dist": 0.0,
            }

        env._expert_candidate_actions = fake_candidates
        env._evaluate_expert_rollout = fake_evaluate

        action, mode = env._select_expert_action(gradual_warning=True)

        self.assertEqual(mode, "cautious_trend")
        self.assertTrue(np.allclose(action, mild_action))
        self.assertIn((False, True), calls)

    def test_expert_candidate_helper_uses_distinct_action_sets(self):
        normal = expert_candidate_actions(self.config)
        mild = expert_candidate_actions(self.config, mild=True)
        emergency = expert_candidate_actions(self.config, emergency=True)

        self.assertGreater(len(normal), 0)
        self.assertGreater(len(mild), 0)
        self.assertGreater(len(emergency), 0)
        self.assertTrue(any(abs(float(action[0])) < 1.0 for action in mild))
        self.assertTrue(any(float(action[1]) < 0.0 for action in emergency))

    def test_expert_selection_helper_falls_back_to_emergency_when_normal_unsafe(self):
        emergency_action = np.array([1.0, -1.0, 0.0], dtype=float)
        action, mode = select_expert_action_from_evaluations(
            zero_eval={
                "action": np.zeros(3, dtype=float),
                "score": 0.0,
                "hard_violation_count": 1,
                "max_risk": 0.9,
            },
            normal_evaluations=[
                {
                    "action": np.array([0.5, 0.0, 0.0], dtype=float),
                    "score": 1.0,
                    "hard_violation_count": 1,
                    "max_risk": 0.8,
                }
            ],
            emergency_evaluations=[
                {
                    "action": emergency_action,
                    "score": 5.0,
                    "hard_violation_count": 0,
                    "max_risk": 0.2,
                }
            ],
            gradual_warning=False,
            config=self.config,
        )

        self.assertEqual(mode, "emergency")
        self.assertTrue(np.allclose(action, emergency_action))

    def test_apas_searches_for_a_safer_residual_command(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.config.v25_apas_segment_check_enabled = False
        env.min_clearance_agl = self.config.rl_min_clearance_agl
        env.max_clearance_agl = self.config.rl_max_clearance_agl
        calls = []

        def fake_transition(heading, speed, agl):
            calls.append((heading, speed, agl))
            return {
                "safe": len(calls) >= 2,
                "position_xyz": np.array([float(len(calls)), 0.0, 100.0], dtype=float),
                "p_crash": 0.0,
                "power_w": self.config.base_power,
                "destructive_core_hit": False,
            }

        env._simulate_true_transition = fake_transition
        env._transition_is_safe_for_apas = lambda transition: transition["safe"]

        transition, info = env._simulate_apas_true_transition(10.0, 10.0, 50.0)
        self.assertTrue(transition["safe"])
        self.assertTrue(info["apas_intervened"])
        self.assertGreater(info["apas_speed_reduction_mps"], 0.0)

    def test_apas_segment_filter_rejects_core_crossing_candidate(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.config.v25_apas_segment_check_enabled = True
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env.current_time = 77.0
        env.min_clearance_agl = self.config.rl_min_clearance_agl
        env.max_clearance_agl = self.config.rl_max_clearance_agl

        calls = []

        def fake_transition(heading, speed, agl):
            calls.append(heading)
            if len(calls) == 1:
                return {
                    "position_xyz": np.array([100.0, 0.0, 100.0], dtype=float),
                    "p_crash": 0.0,
                    "power_w": self.config.base_power,
                    "destructive_core_hit": False,
                }
            return {
                "position_xyz": np.array([0.0, 100.0, 100.0], dtype=float),
                "p_crash": 0.0,
                "power_w": self.config.base_power,
                "destructive_core_hit": False,
            }

        def fake_segment_probe(start_xyz, end_xyz, random_layer_time_s=None, samples=None):
            if abs(float(end_xyz[0]) - 100.0) < 1e-6:
                return {"max_risk_bonus": 0.9, "destructive_core_hit": True}
            return {"max_risk_bonus": 0.0, "destructive_core_hit": False}

        class FakeMap:
            def is_collision(self, *args, **kwargs):
                return False

        class FakeEstimator:
            map = FakeMap()

        env.estimator = FakeEstimator()
        env._current_nfz_list = lambda: []
        env._simulate_true_transition = fake_transition
        env._probe_random_layer_segment = fake_segment_probe

        transition, info = env._simulate_apas_true_transition(10.0, 10.0, 50.0)

        self.assertTrue(info["apas_intervened"])
        self.assertEqual(info["apas_segment_rejections"], 1)
        self.assertFalse(info["apas_no_valid_candidate"])
        self.assertTrue(np.allclose(transition["position_xyz"], np.array([0.0, 100.0, 100.0])))

    def test_stale_waypoint_skip_advances_to_downstream_nearest_point(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.global_astar_path = [
            (0.0, 0.0, 100.0),
            (100.0, 0.0, 100.0),
            (200.0, 0.0, 100.0),
            (300.0, 0.0, 100.0),
        ]
        env.current_wp_idx = 0
        env.episode_stale_waypoint_skips = 0
        env.episode_stale_waypoint_skip_delta = 0
        env._suppress_waypoint_skip_metrics = False

        env._refresh_wp_index(np.array([205.0, 20.0], dtype=float))

        self.assertEqual(env.current_wp_idx, 2)
        self.assertEqual(env.episode_stale_waypoint_skips, 1)
        self.assertEqual(env.episode_stale_waypoint_skip_delta, 2)

    def test_stale_waypoint_skip_respects_corridor_radius(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.global_astar_path = [
            (0.0, 0.0, 100.0),
            (100.0, 0.0, 100.0),
            (200.0, 0.0, 100.0),
            (300.0, 0.0, 100.0),
        ]
        env.current_wp_idx = 0
        env.episode_stale_waypoint_skips = 0
        env.episode_stale_waypoint_skip_delta = 0
        env._suppress_waypoint_skip_metrics = False
        self.config.v25_stale_waypoint_corridor_m = 50.0

        env._refresh_wp_index(np.array([205.0, 120.0], dtype=float))

        self.assertEqual(env.current_wp_idx, 0)
        self.assertEqual(env.episode_stale_waypoint_skips, 0)

    def test_evaluation_costs_make_apas_interventions_nonfree(self):
        costs = compute_evaluation_costs(
            action=np.zeros(3, dtype=float),
            action_delta=0.0,
            apas_info={
                "apas_intervened": True,
                "apas_heading_offset_deg": 10.0,
                "apas_speed_reduction_mps": 2.0,
                "apas_agl_increment_m": 5.0,
                "apas_segment_rejections": 3,
                "apas_no_valid_candidate": False,
            },
            expert_mode="inactive",
            base_energy_step_j=1000.0,
            config=self.config,
        )

        self.assertGreater(costs["maneuver_extra_energy_j"], 0.0)
        self.assertGreater(costs["safety_intervention_burden"], self.config.v25_eval_apas_intervention_burden)
        self.assertGreater(costs["adjusted_energy_step_j"], 1000.0)

    def test_evaluation_costs_include_expert_mode_burden(self):
        normal = compute_evaluation_costs(
            action=np.zeros(3, dtype=float),
            action_delta=0.0,
            apas_info={"apas_intervened": False},
            expert_mode="inactive",
            base_energy_step_j=1000.0,
            config=self.config,
        )
        emergency = compute_evaluation_costs(
            action=np.zeros(3, dtype=float),
            action_delta=0.0,
            apas_info={"apas_intervened": False},
            expert_mode="emergency",
            base_energy_step_j=1000.0,
            config=self.config,
        )

        self.assertEqual(normal["safety_intervention_burden"], 0.0)
        self.assertEqual(emergency["safety_intervention_burden"], self.config.v25_eval_expert_emergency_burden)
        self.assertGreater(emergency["adjusted_energy_step_j"], normal["adjusted_energy_step_j"])

    def test_risk_membrane_detects_front_wall_and_side_gap(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.config.v25_sensor_mode = "circle_oracle"
        env.current_pos = np.array([0.0, 0.0, 100.0], dtype=float)
        env.current_heading = 0.0
        env.current_time = 0.0

        sample_points = []
        for angle_deg in (-30.0, -15.0, 0.0, 15.0, 30.0):
            rad = np.radians(angle_deg)
            sample_points.append(500.0 * np.array([np.cos(rad), np.sin(rad)], dtype=float))
        env._circle_oracle_sample_points = lambda: sample_points

        class FakeStorm:
            def danger_at(self, *args):
                return 0.0

        class FakeDisruptions:
            destructive_storm = FakeStorm()

            def risk_bonus_at(self, x, y, t_s):
                return 1.0

        env.disruptions = FakeDisruptions()

        summary = env._risk_membrane_summary()

        self.assertEqual(summary["risk_membrane_wall_ahead"], 1.0)
        self.assertEqual(summary["risk_membrane_no_escape_gap"], 0.0)
        self.assertGreaterEqual(summary["risk_membrane_best_gap_width_deg"], self.config.v25_risk_membrane_min_gap_width_deg)

    def test_risk_membrane_helper_detects_front_wall(self):
        sample_points = []
        for angle_deg in (-30.0, -15.0, 0.0, 15.0, 30.0):
            rad = np.radians(angle_deg)
            sample_points.append(500.0 * np.array([np.cos(rad), np.sin(rad)], dtype=float))

        summary = compute_risk_membrane_summary(
            origin_xy=np.array([0.0, 0.0], dtype=float),
            heading_deg=0.0,
            current_time_s=0.0,
            sample_points=sample_points,
            danger_at=lambda x, y, t: 1.0,
            config=self.config,
        )

        self.assertEqual(summary["risk_membrane_wall_ahead"], 1.0)
        self.assertEqual(summary["risk_membrane_no_escape_gap"], 0.0)
        self.assertGreater(summary["risk_membrane_max_extended_risk"], 0.0)

    def test_risk_membrane_action_steers_toward_best_gap(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        local_hazard = {
            "risk_membrane_wall_ahead": 1.0,
            "risk_membrane_no_escape_gap": 0.0,
            "risk_membrane_best_gap_angle_deg": 45.0,
            "risk_membrane_best_gap_width_deg": 45.0,
        }

        action, mode = env._risk_membrane_action(local_hazard)

        self.assertEqual(mode, "band_avoidance")
        self.assertGreater(action[0], 0.0)
        self.assertLess(action[1], 0.0)

    def test_risk_membrane_action_helper_slows_when_no_escape_gap(self):
        action, mode = risk_membrane_action(
            {
                "risk_membrane_wall_ahead": 1.0,
                "risk_membrane_no_escape_gap": 1.0,
                "risk_membrane_best_gap_angle_deg": 0.0,
            },
            self.config,
        )

        self.assertEqual(mode, "pre_emergency_slow")
        self.assertEqual(action[0], 0.0)
        self.assertLess(action[1], 0.0)


if __name__ == "__main__":
    unittest.main()
