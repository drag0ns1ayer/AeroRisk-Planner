import unittest

import numpy as np

from configs.config import SimulationConfig
from v25.disruptions import build_disruption_layer_v25
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

    def test_expert_switches_to_emergency_when_normal_candidates_are_invalid(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        normal_action = np.array([0.0, 0.0, 0.0], dtype=float)
        emergency_action = np.array([1.0, -1.0, 0.0], dtype=float)

        def fake_candidates(emergency=False):
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

        def fake_candidates(emergency=False):
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

    def test_apas_searches_for_a_safer_residual_command(self):
        env = object.__new__(GuidedDroneEnvV25)
        env.config = self.config
        env.min_clearance_agl = self.config.rl_min_clearance_agl
        env.max_clearance_agl = self.config.rl_max_clearance_agl
        calls = []

        def fake_transition(heading, speed, agl):
            calls.append((heading, speed, agl))
            return {"safe": len(calls) >= 2}

        env._simulate_true_transition = fake_transition
        env._transition_is_safe_for_apas = lambda transition: transition["safe"]

        transition, info = env._simulate_apas_true_transition(10.0, 10.0, 50.0)
        self.assertTrue(transition["safe"])
        self.assertTrue(info["apas_intervened"])
        self.assertGreater(info["apas_speed_reduction_mps"], 0.0)


if __name__ == "__main__":
    unittest.main()
