from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from configs.config import SimulationConfig
from rl_env.drone_env import GuidedDroneEnv
from v25.disruptions import DisruptionLayerV25, build_disruption_layer_v25
from v25.true_world_dynamics import TrueWorldDynamicsV25, VehicleStateV25


def compute_intervention_need(
    residual_wind_mps: float,
    tracking_error_mps: float,
    path_error_m: float,
    config: SimulationConfig,
    local_hazard_need: float = 0.0,
) -> float:
    """Return a causal 0..1 estimate of how much residual control is needed."""
    wind_need = np.clip(residual_wind_mps / config.v25_intervention_wind_scale_mps, 0.0, 1.0)
    tracking_need = np.clip(tracking_error_mps / config.v25_intervention_tracking_scale_mps, 0.0, 1.0)
    path_need = np.clip(path_error_m / config.v25_intervention_path_scale_m, 0.0, 1.0)
    hazard_need = np.clip(local_hazard_need, 0.0, 1.0)
    hazard_gain = float(np.clip(config.v25_intervention_hazard_gain, 0.0, 1.0))
    motion_need = float(np.clip(0.55 * wind_need + 0.30 * tracking_need + 0.15 * path_need, 0.0, 1.0))
    return float(np.clip(motion_need + (1.0 - motion_need) * hazard_gain * hazard_need, 0.0, 1.0))


def compute_residual_control_cost(
    action: np.ndarray,
    previous_action: np.ndarray,
    intervention_need: float,
    config: SimulationConfig,
) -> tuple[float, float, float]:
    """Return total, magnitude, and smoothness costs for a residual action."""
    magnitude = float(np.linalg.norm(action))
    action_delta = float(np.linalg.norm(action - previous_action))
    magnitude_gain = (
        config.v25_reward_residual_gain_true
        + config.v25_reward_calm_residual_gain_true * (1.0 - intervention_need)
    )
    total = magnitude_gain * magnitude + config.v25_reward_action_smoothness_gain_true * action_delta
    return float(total), magnitude, action_delta


def compute_residual_gate(intervention_need: float, config: SimulationConfig) -> float:
    """Return the residual action scale allowed by the current intervention need."""
    need = float(np.clip(intervention_need, 0.0, 1.0))
    min_scale = float(np.clip(config.v25_residual_gate_min_scale, 0.0, 1.0))
    power = max(float(config.v25_residual_gate_power), 1e-6)
    return float(min_scale + (1.0 - min_scale) * (need ** power))


class GuidedDroneEnvV25(GuidedDroneEnv):
    """
    Predictable A* planning plus residual RL control in a shared true world.

    A* only sees the predictable estimator. Execution uses predictable wind plus
    a seeded hidden disturbance layer. RL receives noisy local sensor readings,
    never the disturbance parameters or exact severity.
    """

    RADAR_FEATURES_PER_SECTOR = 4
    CIRCLE_ORACLE_FEATURES = 12
    SENSOR_BASE_FEATURES = 6
    SENSOR_FEATURES = SENSOR_BASE_FEATURES + 8 * RADAR_FEATURES_PER_SECTOR

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31 + self._sensor_feature_count(config),),
            dtype=np.float32,
        )
        self.disruptions: Optional[DisruptionLayerV25] = None
        self.dynamics = TrueWorldDynamicsV25(self.config)
        self.sensor_rng = np.random.default_rng(self.config.wind_seed)
        self.current_airspeed = float(self.config.drone_speed)
        self.last_measured_residual_wind = np.zeros(2, dtype=float)
        self.last_ground_velocity_xy = np.zeros(2, dtype=float)
        self.last_disturbance_severity = 0.0
        self.previous_action = np.zeros(3, dtype=float)
        self.last_expert_mode = "normal"
        self.last_expert_active = False
        self._suppress_waypoint_skip_metrics = False
        self.local_hazard_history = deque(maxlen=int(max(1, self.config.v25_local_hazard_history_steps)))
        self.do_no_harm_history = deque(maxlen=int(max(1, self.config.v25_do_no_harm_window_steps)))
        self.do_no_harm_cooldown_steps_remaining = 0
        self.last_do_no_harm_active = False
        self.last_do_no_harm_reason = "none"
        self.episode_do_no_harm_events = 0
        self.episode_do_no_harm_suppressed_steps = 0
        self.episode_do_no_harm_cooldown_steps = 0

        self.episode_disturbance_max = 0.0
        self.episode_disturbance_sum = 0.0
        self.episode_disturbance_steps = 0
        self.episode_residual_action_sum = 0.0
        self.episode_residual_heading_abs_sum = 0.0
        self.episode_residual_speed_abs_sum = 0.0
        self.episode_residual_agl_abs_sum = 0.0
        self.episode_action_delta_sum = 0.0
        self.episode_intervention_need_sum = 0.0
        self.episode_unneeded_residual_sum = 0.0
        self.episode_needed_residual_sum = 0.0
        self.episode_apas_interventions = 0
        self.episode_apas_segment_rejections = 0
        self.episode_apas_no_valid_candidates = 0
        self.episode_stale_waypoint_skips = 0
        self.episode_stale_waypoint_skip_delta = 0
        self.episode_destructive_core_hits = 0
        self.episode_unproductive_residual_cost_sum = 0.0
        self.episode_progress_shortfall_sum = 0.0
        self.episode_apas_intervention_cost_sum = 0.0
        self.episode_speed_residual_cost_sum = 0.0
        self.episode_residual_gate_sum = 0.0
        self.episode_local_hazard_need_sum = 0.0
        self.episode_local_hazard_cost_sum = 0.0
        self.episode_local_hazard_trend_need_sum = 0.0
        self.episode_local_hazard_forward_delta_sum = 0.0
        self.episode_local_hazard_positive_trend_steps = 0
        self.episode_eval_maneuver_extra_energy_j = 0.0
        self.episode_eval_safety_intervention_burden = 0.0
        self.episode_eval_adjusted_energy_j = 0.0
        self.episode_expert_normal_steps = 0
        self.episode_expert_cautious_steps = 0
        self.episode_expert_cautious_trend_steps = 0
        self.episode_expert_avoiding_steps = 0
        self.episode_expert_emergency_steps = 0
        self.episode_expert_band_avoidance_steps = 0
        self.episode_expert_pre_emergency_slow_steps = 0
        self.episode_expert_recovering_steps = 0
        self.episode_expert_rejoin_actions = 0
        self.episode_expert_rejoin_attempts = 0
        self.episode_expert_rejoin_rejected = 0
        self.episode_replans = 0
        self.episode_replan_successes = 0
        self.episode_replan_failures = 0
        self.episode_replan_to_rejoin_successes = 0
        self.episode_replan_to_goal_successes = 0
        self.episode_replan_path_drift_triggers = 0
        self.episode_replan_low_progress_triggers = 0
        self.episode_replan_no_valid_triggers = 0
        self.replan_cooldown_steps_remaining = 0
        self.consecutive_replan_low_progress_steps = 0
        self.apas_no_valid_linger_steps = 0
        self.last_replan_event = "none"
        self.consecutive_avoiding_steps = 0
        self.consecutive_recovering_steps = 0
        self.consecutive_cautious_steps = 0
        self.consecutive_low_progress_steps = 0

    @classmethod
    def _sensor_feature_count(cls, config: SimulationConfig) -> int:
        if getattr(config, "v25_sensor_mode", "sector_radar") == "circle_oracle":
            return cls.SENSOR_BASE_FEATURES + cls.CIRCLE_ORACLE_FEATURES
        return cls.SENSOR_BASE_FEATURES + int(config.v25_radar_sectors) * cls.RADAR_FEATURES_PER_SECTOR

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # super().reset() dynamically calls _get_obs(), which tolerates the
        # not-yet-created disturbance layer.
        _, info = super().reset(seed=seed, options=options)
        episode_seed = int(self.episode_wind_seed if seed is None else seed)
        self.sensor_rng = np.random.default_rng(episode_seed + 7919)
        self.disruptions = build_disruption_layer_v25(
            start_xy=(float(self.current_pos[0]), float(self.current_pos[1])),
            goal_xy=(float(self.goal_pos[0]), float(self.goal_pos[1])),
            nominal_speed_mps=float(max(self.config.drone_speed, self.config.rl_speed_min)),
            config=self.config,
            seed=episode_seed,
        )
        self.current_airspeed = float(
            np.clip(self.config.drone_speed, self.config.rl_speed_min, self.config.rl_speed_max)
        )
        self.last_measured_residual_wind = np.zeros(2, dtype=float)
        self.last_ground_velocity_xy = np.asarray(
            self._astar_base_command()["desired_ground_velocity_xy"],
            dtype=float,
        )
        self.last_disturbance_severity = 0.0
        self.previous_action = np.zeros(3, dtype=float)
        self.last_expert_mode = "normal"
        self.last_expert_active = False
        self._suppress_waypoint_skip_metrics = False
        self.local_hazard_history = deque(maxlen=int(max(1, self.config.v25_local_hazard_history_steps)))
        self.do_no_harm_history = deque(maxlen=int(max(1, self.config.v25_do_no_harm_window_steps)))
        self.do_no_harm_cooldown_steps_remaining = 0
        self.last_do_no_harm_active = False
        self.last_do_no_harm_reason = "none"
        self.episode_do_no_harm_events = 0
        self.episode_do_no_harm_suppressed_steps = 0
        self.episode_do_no_harm_cooldown_steps = 0
        self.episode_disturbance_max = 0.0
        self.episode_disturbance_sum = 0.0
        self.episode_disturbance_steps = 0
        self.episode_residual_action_sum = 0.0
        self.episode_residual_heading_abs_sum = 0.0
        self.episode_residual_speed_abs_sum = 0.0
        self.episode_residual_agl_abs_sum = 0.0
        self.episode_action_delta_sum = 0.0
        self.episode_intervention_need_sum = 0.0
        self.episode_unneeded_residual_sum = 0.0
        self.episode_needed_residual_sum = 0.0
        self.episode_apas_interventions = 0
        self.episode_apas_segment_rejections = 0
        self.episode_apas_no_valid_candidates = 0
        self.episode_stale_waypoint_skips = 0
        self.episode_stale_waypoint_skip_delta = 0
        self.episode_destructive_core_hits = 0
        self.episode_unproductive_residual_cost_sum = 0.0
        self.episode_progress_shortfall_sum = 0.0
        self.episode_apas_intervention_cost_sum = 0.0
        self.episode_speed_residual_cost_sum = 0.0
        self.episode_residual_gate_sum = 0.0
        self.episode_local_hazard_need_sum = 0.0
        self.episode_local_hazard_cost_sum = 0.0
        self.episode_local_hazard_trend_need_sum = 0.0
        self.episode_local_hazard_forward_delta_sum = 0.0
        self.episode_local_hazard_positive_trend_steps = 0
        self.episode_eval_maneuver_extra_energy_j = 0.0
        self.episode_eval_safety_intervention_burden = 0.0
        self.episode_eval_adjusted_energy_j = 0.0
        self.episode_expert_normal_steps = 0
        self.episode_expert_cautious_steps = 0
        self.episode_expert_cautious_trend_steps = 0
        self.episode_expert_avoiding_steps = 0
        self.episode_expert_emergency_steps = 0
        self.episode_expert_band_avoidance_steps = 0
        self.episode_expert_pre_emergency_slow_steps = 0
        self.episode_expert_recovering_steps = 0
        self.episode_expert_rejoin_actions = 0
        self.episode_expert_rejoin_attempts = 0
        self.episode_expert_rejoin_rejected = 0
        self.episode_replans = 0
        self.episode_replan_successes = 0
        self.episode_replan_failures = 0
        self.episode_replan_to_rejoin_successes = 0
        self.episode_replan_to_goal_successes = 0
        self.episode_replan_path_drift_triggers = 0
        self.episode_replan_low_progress_triggers = 0
        self.episode_replan_no_valid_triggers = 0
        self.replan_cooldown_steps_remaining = 0
        self.consecutive_replan_low_progress_steps = 0
        self.apas_no_valid_linger_steps = 0
        self.last_replan_event = "none"
        self.consecutive_avoiding_steps = 0
        self.consecutive_recovering_steps = 0
        self.consecutive_cautious_steps = 0
        self.consecutive_low_progress_steps = 0
        info["v25_mode"] = "predictable_astar_shared_true_world_residual_rl"
        info["random_layer_seed"] = episode_seed
        return self._get_obs(), info

    def _predictable_wind(self, x: float, y: float, z: float, t_s: float) -> np.ndarray:
        return np.asarray(self.estimator.get_wind(x, y, z, t_s=t_s), dtype=float)

    def _disturbance_at(
        self,
        x: float,
        y: float,
        t_s: float,
        travel_dir_xy: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self.disruptions is None:
            return np.zeros(2, dtype=float), 0.0
        return self.disruptions.disturbance_at(x, y, t_s, travel_dir_xy)

    def _true_wind(
        self,
        x: float,
        y: float,
        z: float,
        t_s: float,
        travel_dir_xy: np.ndarray,
        disturbance_time_s: float | None = None,
    ) -> tuple[np.ndarray, float]:
        predictable = self._predictable_wind(x, y, z, t_s)
        residual, severity = self._disturbance_at(
            x,
            y,
            t_s if disturbance_time_s is None else float(disturbance_time_s),
            travel_dir_xy,
        )
        return predictable + residual, severity

    def _measured_residual_wind(self) -> np.ndarray:
        travel = self.last_ground_velocity_xy
        if np.linalg.norm(travel) < 1e-6:
            rad = math.radians(self.current_heading)
            travel = np.array([math.cos(rad), math.sin(rad)], dtype=float)
        residual, _ = self._disturbance_at(
            float(self.current_pos[0]),
            float(self.current_pos[1]),
            float(self.current_time),
            travel,
        )
        noise = self.sensor_rng.normal(0.0, self.config.v25_sensor_wind_noise_std_mps, size=2)
        return residual + noise

    def _refresh_wp_index(self, pos_xy: np.ndarray) -> None:
        """
        Advance the reference waypoint, including a conservative stale-waypoint
        skip for v2.5 execution.

        The base rule only advances when the vehicle enters the current
        waypoint radius. Under wind and APAS corrections the vehicle can miss a
        waypoint while still tracking the path corridor; then it keeps chasing a
        point behind it and times out. This projection-style skip only jumps
        forward when the nearest path point is clearly downstream and the
        vehicle remains close to the planned corridor.
        """
        if not getattr(self, "global_astar_path", None):
            return

        while self.current_wp_idx < len(self.global_astar_path) - 1:
            wp = self._path_point(self.current_wp_idx)
            if np.linalg.norm(np.asarray(pos_xy, dtype=float) - wp[:2]) < self.config.rl_waypoint_refresh_radius_m:
                self.current_wp_idx += 1
            else:
                break

        if not bool(getattr(self.config, "v25_stale_waypoint_skip_enabled", True)):
            return
        if self.current_wp_idx >= len(self.global_astar_path) - 1:
            return

        pos = np.asarray(pos_xy, dtype=float)
        distances = [
            float(np.linalg.norm(pos - np.asarray(point[:2], dtype=float)))
            for point in self.global_astar_path
        ]
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])
        min_advance = int(self.config.v25_stale_waypoint_min_advance)
        corridor_m = float(self.config.v25_stale_waypoint_corridor_m)
        old_idx = int(self.current_wp_idx)

        if nearest_idx >= old_idx + min_advance and nearest_dist <= corridor_m:
            self.current_wp_idx = min(nearest_idx, len(self.global_astar_path) - 1)
            if not bool(getattr(self, "_suppress_waypoint_skip_metrics", False)):
                self.episode_stale_waypoint_skips = int(getattr(self, "episode_stale_waypoint_skips", 0)) + 1
                self.episode_stale_waypoint_skip_delta = int(
                    getattr(self, "episode_stale_waypoint_skip_delta", 0)
                ) + max(0, self.current_wp_idx - old_idx)

    def _astar_base_command(self) -> dict:
        """
        Convert A*'s desired ground track into an air-velocity command using
        only predictable wind. RL then corrects the unknown residual wind.
        """
        teacher = self._teacher_reference()
        ground_rad = math.radians(teacher["heading_deg"])
        desired_ground_xy = (
            np.array([math.cos(ground_rad), math.sin(ground_rad)], dtype=float)
            * teacher["speed_mps"]
        )
        predictable_wind = self._predictable_wind(
            float(self.current_pos[0]),
            float(self.current_pos[1]),
            float(self.current_pos[2]),
            float(self.current_time),
        )
        required_air_xy = desired_ground_xy - predictable_wind
        required_air_speed = float(np.linalg.norm(required_air_xy))
        if required_air_speed < 1e-6:
            air_heading = teacher["heading_deg"]
        else:
            air_heading = math.degrees(math.atan2(required_air_xy[1], required_air_xy[0]))
        return {
            "heading_deg": self._wrap_angle_deg(air_heading),
            "airspeed_mps": float(
                np.clip(required_air_speed, self.config.rl_speed_min, self.config.rl_speed_max)
            ),
            "agl_m": teacher["agl_m"],
            "desired_ground_velocity_xy": desired_ground_xy,
        }

    def _world_vector_to_body(self, vec_xy: np.ndarray) -> np.ndarray:
        heading_rad = math.radians(float(self.current_heading))
        forward = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)
        left = np.array([-math.sin(heading_rad), math.cos(heading_rad)], dtype=float)
        return np.array([float(np.dot(vec_xy, forward)), float(np.dot(vec_xy, left))], dtype=float)

    def _circle_oracle_sample_points(self) -> list[np.ndarray]:
        origin = np.asarray(self.current_pos[:2], dtype=float)
        radius_m = float(self.config.v25_radar_radius_m)
        sample_count = int(self.config.v25_circle_oracle_samples)
        heading_rad = math.radians(float(self.current_heading))

        sample_points = [origin]
        front_angles_deg = (
            -75.0, -60.0, -45.0, -32.0, -22.0, -12.0, -6.0,
            0.0,
            6.0, 12.0, 22.0, 32.0, 45.0, 60.0, 75.0,
        )
        front_radii = (0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.92, 1.00)
        for radius_scale in front_radii:
            for rel_angle_deg in front_angles_deg:
                if len(sample_points) >= sample_count:
                    break
                angle = heading_rad + math.radians(rel_angle_deg)
                sample_points.append(
                    origin + radius_scale * radius_m * np.array([math.cos(angle), math.sin(angle)], dtype=float)
                )
            if len(sample_points) >= sample_count:
                break

        rings = (0.25, 0.50, 0.75, 1.00)
        per_ring = max(8, int(math.ceil(max(sample_count - len(sample_points), 1) / len(rings))))
        for ring in rings:
            for idx in range(per_ring):
                if len(sample_points) >= sample_count:
                    break
                angle = 2.0 * math.pi * idx / per_ring
                sample_points.append(origin + ring * radius_m * np.array([math.cos(angle), math.sin(angle)]))
            if len(sample_points) >= sample_count:
                break

        return sample_points

    def _circle_oracle_features(self) -> list[float]:
        if self.disruptions is None:
            return [0.0] * self.CIRCLE_ORACLE_FEATURES

        origin = np.asarray(self.current_pos[:2], dtype=float)
        radius_m = float(self.config.v25_radar_radius_m)
        heading_rad = math.radians(float(self.current_heading))
        forward = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)

        sample_points = self._circle_oracle_sample_points()

        dangers = []
        winds = []
        offsets = []
        forward_dangers = []
        max_wind_mag = 0.0
        max_wind_vec = np.zeros(2, dtype=float)
        nearest_danger_dist = radius_m
        nearest_danger_vec = np.zeros(2, dtype=float)
        weighted_danger_vec = np.zeros(2, dtype=float)
        weighted_danger_sum = 0.0

        for point in sample_points:
            offset = np.asarray(point - origin, dtype=float)
            dist = float(np.linalg.norm(offset))
            travel_dir = offset / dist if dist > 1e-6 else forward
            wind, _ = self.disruptions.disturbance_at(float(point[0]), float(point[1]), self.current_time, travel_dir)
            danger = float(self.disruptions.risk_bonus_at(float(point[0]), float(point[1]), self.current_time))
            storm_danger = float(self.disruptions.destructive_storm.danger_at(float(point[0]), float(point[1]), self.current_time))
            danger = max(danger, storm_danger)

            dangers.append(danger)
            winds.append(wind)
            offsets.append(offset)
            if dist > 1e-6 and np.dot(offset / dist, forward) > math.cos(math.radians(45.0)):
                forward_dangers.append(danger)

            wind_mag = float(np.linalg.norm(wind))
            if wind_mag > max_wind_mag:
                max_wind_mag = wind_mag
                max_wind_vec = wind
            if danger > 1e-6:
                if dist < nearest_danger_dist:
                    nearest_danger_dist = dist
                    nearest_danger_vec = offset
                weighted_danger_vec += offset * danger
                weighted_danger_sum += danger

        mean_wind = np.mean(np.asarray(winds, dtype=float), axis=0) if winds else np.zeros(2, dtype=float)
        mean_wind_body = self._world_vector_to_body(mean_wind) / max(float(self.config.max_wind_speed), 1e-6)
        max_wind_body = self._world_vector_to_body(max_wind_vec) / max(float(self.config.max_wind_speed), 1e-6)

        if np.linalg.norm(nearest_danger_vec) > 1e-6:
            nearest_dir_body = self._world_vector_to_body(nearest_danger_vec / np.linalg.norm(nearest_danger_vec))
        else:
            nearest_dir_body = np.zeros(2, dtype=float)
        if weighted_danger_sum > 1e-6 and np.linalg.norm(weighted_danger_vec) > 1e-6:
            centroid_vec = weighted_danger_vec / weighted_danger_sum
            centroid_dir_body = self._world_vector_to_body(centroid_vec / np.linalg.norm(centroid_vec))
        else:
            centroid_dir_body = np.zeros(2, dtype=float)

        max_danger = float(max(dangers)) if dangers else 0.0
        mean_danger = float(np.mean(dangers)) if dangers else 0.0
        nearest_closeness = 1.0 - min(nearest_danger_dist / max(radius_m, 1e-6), 1.0)
        forward_danger = float(max(forward_dangers)) if forward_dangers else 0.0

        return [
            float(np.clip(max_danger, 0.0, 1.0)),
            float(np.clip(mean_danger, 0.0, 1.0)),
            float(np.clip(nearest_closeness, 0.0, 1.0)),
            float(np.clip(nearest_dir_body[0], -1.0, 1.0)),
            float(np.clip(nearest_dir_body[1], -1.0, 1.0)),
            float(np.clip(centroid_dir_body[0], -1.0, 1.0)),
            float(np.clip(centroid_dir_body[1], -1.0, 1.0)),
            float(np.clip(mean_wind_body[0], -2.0, 2.0)),
            float(np.clip(mean_wind_body[1], -2.0, 2.0)),
            float(np.clip(max_wind_body[0], -2.0, 2.0)),
            float(np.clip(max_wind_body[1], -2.0, 2.0)),
            float(np.clip(forward_danger, 0.0, 1.0)),
        ]

    def _risk_membrane_summary(self) -> dict[str, float]:
        if (
            self.disruptions is None
            or not bool(getattr(self.config, "v25_risk_membrane_enabled", True))
            or getattr(self.config, "v25_sensor_mode", "sector_radar") != "circle_oracle"
        ):
            return {
                "risk_membrane_wall_ahead": 0.0,
                "risk_membrane_no_escape_gap": 0.0,
                "risk_membrane_front_blocked_width_deg": 0.0,
                "risk_membrane_best_gap_angle_deg": 0.0,
                "risk_membrane_best_gap_width_deg": 0.0,
                "risk_membrane_best_gap_side": 0.0,
                "risk_membrane_max_extended_risk": 0.0,
            }

        origin = np.asarray(self.current_pos[:2], dtype=float)
        radius_m = float(self.config.v25_radar_radius_m)
        heading_rad = math.radians(float(self.current_heading))
        angle_min = float(self.config.v25_risk_membrane_angle_min_deg)
        angle_max = float(self.config.v25_risk_membrane_angle_max_deg)
        angle_step = float(self.config.v25_risk_membrane_angle_step_deg)
        angle_bins = np.arange(angle_min, angle_max + 0.5 * angle_step, angle_step, dtype=float)
        radial_bins = np.linspace(
            radius_m / max(int(self.config.v25_risk_membrane_radial_bins), 1),
            radius_m,
            int(self.config.v25_risk_membrane_radial_bins),
            dtype=float,
        )
        if len(angle_bins) == 0 or len(radial_bins) == 0:
            return {
                "risk_membrane_wall_ahead": 0.0,
                "risk_membrane_no_escape_gap": 0.0,
                "risk_membrane_front_blocked_width_deg": 0.0,
                "risk_membrane_best_gap_angle_deg": 0.0,
                "risk_membrane_best_gap_width_deg": 0.0,
                "risk_membrane_best_gap_side": 0.0,
                "risk_membrane_max_extended_risk": 0.0,
            }

        observations: list[tuple[float, float, float]] = []
        for point in self._circle_oracle_sample_points():
            offset = np.asarray(point - origin, dtype=float)
            dist = float(np.linalg.norm(offset))
            if dist < 1e-6:
                continue
            bearing = self._wrap_angle_deg(math.degrees(math.atan2(float(offset[1]), float(offset[0]))) - math.degrees(heading_rad))
            if bearing < angle_min - angle_step or bearing > angle_max + angle_step:
                continue
            travel_dir = offset / dist
            danger = float(self.disruptions.risk_bonus_at(float(point[0]), float(point[1]), self.current_time))
            storm_danger = float(
                self.disruptions.destructive_storm.danger_at(float(point[0]), float(point[1]), self.current_time)
            )
            danger = float(np.clip(max(danger, storm_danger), 0.0, 1.0))
            if danger > 1e-6:
                observations.append((dist, bearing, danger))

        extended = np.zeros((len(radial_bins), len(angle_bins)), dtype=float)
        lambda_r = max(float(self.config.v25_risk_membrane_lambda_r_m), 1e-6)
        lambda_theta = max(float(self.config.v25_risk_membrane_lambda_theta_deg), 1e-6)
        for obs_dist, obs_bearing, danger in observations:
            for r_idx, r_center in enumerate(radial_bins):
                radial_decay = math.exp(-abs(float(r_center) - obs_dist) / lambda_r)
                for a_idx, angle_center in enumerate(angle_bins):
                    angular_decay = math.exp(-abs(float(angle_center) - obs_bearing) / lambda_theta)
                    extended[r_idx, a_idx] = max(extended[r_idx, a_idx], danger * radial_decay * angular_decay)

        angular_risk = np.max(extended, axis=0) if len(extended) else np.zeros(len(angle_bins), dtype=float)
        block_threshold = float(self.config.v25_risk_membrane_block_threshold)
        blocked = angular_risk >= block_threshold
        front_window = float(self.config.v25_risk_membrane_front_window_deg)
        front_mask = np.abs(angle_bins) <= front_window
        front_blocked_width = float(np.sum(blocked & front_mask) * angle_step)
        wall_ahead = front_blocked_width >= float(self.config.v25_risk_membrane_wall_width_deg)

        gaps: list[tuple[float, float, float]] = []
        start_idx: Optional[int] = None
        for idx, is_blocked in enumerate(blocked):
            if not is_blocked and start_idx is None:
                start_idx = idx
            if (is_blocked or idx == len(blocked) - 1) and start_idx is not None:
                end_idx = idx - 1 if is_blocked else idx
                width = float((end_idx - start_idx + 1) * angle_step)
                center = float(0.5 * (angle_bins[start_idx] + angle_bins[end_idx]))
                if width >= float(self.config.v25_risk_membrane_min_gap_width_deg):
                    gaps.append((center, width, float(abs(center))))
                start_idx = None

        if gaps:
            best_gap = min(gaps, key=lambda item: (item[2] - 0.02 * item[1], item[2]))
            best_gap_angle = float(best_gap[0])
            best_gap_width = float(best_gap[1])
            no_escape_gap = 0.0
        else:
            best_gap_angle = 0.0
            best_gap_width = 0.0
            no_escape_gap = 1.0 if wall_ahead else 0.0

        return {
            "risk_membrane_wall_ahead": float(bool(wall_ahead)),
            "risk_membrane_no_escape_gap": float(no_escape_gap),
            "risk_membrane_front_blocked_width_deg": front_blocked_width,
            "risk_membrane_best_gap_angle_deg": best_gap_angle,
            "risk_membrane_best_gap_width_deg": best_gap_width,
            "risk_membrane_best_gap_side": float(np.sign(best_gap_angle)),
            "risk_membrane_max_extended_risk": float(np.max(extended)) if extended.size else 0.0,
        }

    def _sector_radar_features(self) -> list[float]:
        if self.disruptions is None:
            return [0.0] * (int(self.config.v25_radar_sectors) * self.RADAR_FEATURES_PER_SECTOR)
        radar_values = []
        for sector_idx in range(int(self.config.v25_radar_sectors)):
            wind_mag, wind_dir_sin, danger, danger_closeness = self.disruptions.radar_sector(
                origin_xy=np.asarray(self.current_pos[:2], dtype=float),
                heading_deg=float(self.current_heading),
                sector_idx=sector_idx,
                sector_count=int(self.config.v25_radar_sectors),
                radius_m=float(self.config.v25_radar_radius_m),
                t_s=float(self.current_time),
            )
            noisy_wind = wind_mag + float(self.sensor_rng.normal(0.0, self.config.v25_sensor_wind_noise_std_mps))
            noisy_danger = danger + float(self.sensor_rng.normal(0.0, self.config.v25_sensor_risk_noise_std))
            radar_values.extend(
                [
                    float(np.clip(noisy_wind / self.config.max_wind_speed, 0.0, 2.0)),
                    float(np.clip(wind_dir_sin, -1.0, 1.0)),
                    float(np.clip(noisy_danger, 0.0, 1.0)),
                    float(np.clip(danger_closeness, 0.0, 1.0)),
                ]
            )
        return radar_values

    def _empty_local_hazard_summary(self) -> dict[str, float]:
        return {
            "hazard_need": 0.0,
            "base_hazard_need": 0.0,
            "max_danger": 0.0,
            "mean_danger": 0.0,
            "forward_danger": 0.0,
            "nearest_closeness": 0.0,
            "nearest_forward_alignment": 0.0,
            "trend_need": 0.0,
            "delta_max_danger": 0.0,
            "delta_forward_danger": 0.0,
            "delta_nearest_closeness": 0.0,
            "gradual_warning": 0.0,
            "risk_membrane_wall_ahead": 0.0,
            "risk_membrane_no_escape_gap": 0.0,
            "risk_membrane_front_blocked_width_deg": 0.0,
            "risk_membrane_best_gap_angle_deg": 0.0,
            "risk_membrane_best_gap_width_deg": 0.0,
            "risk_membrane_best_gap_side": 0.0,
            "risk_membrane_max_extended_risk": 0.0,
        }

    def _local_hazard_history_features(self, current: dict[str, float]) -> dict[str, float]:
        if not self.local_hazard_history:
            return {
                "trend_need": 0.0,
                "delta_max_danger": 0.0,
                "delta_forward_danger": 0.0,
                "delta_nearest_closeness": 0.0,
            }

        oldest = self.local_hazard_history[0]
        delta_max = float(current["max_danger"] - oldest["max_danger"])
        delta_forward = float(current["forward_danger"] - oldest["forward_danger"])
        delta_near = float(current["nearest_closeness"] - oldest["nearest_closeness"])
        positive_signal = max(0.0, delta_forward) + 0.75 * max(0.0, delta_max) + 0.50 * max(0.0, delta_near)
        threshold = float(max(self.config.v25_local_hazard_trend_threshold, 1e-6))
        trend_need = float(np.clip(positive_signal / (4.0 * threshold), 0.0, 1.0))
        return {
            "trend_need": trend_need,
            "delta_max_danger": delta_max,
            "delta_forward_danger": delta_forward,
            "delta_nearest_closeness": delta_near,
        }

    def _remember_local_hazard(self, summary: dict[str, float]) -> None:
        self.local_hazard_history.append(
            {
                "max_danger": float(summary.get("max_danger", 0.0)),
                "forward_danger": float(summary.get("forward_danger", 0.0)),
                "nearest_closeness": float(summary.get("nearest_closeness", 0.0)),
            }
        )

    def _local_hazard_gradual_warning(self, summary: dict[str, float]) -> float:
        if float(summary.get("base_hazard_need", summary.get("hazard_need", 0.0))) >= float(
            self.config.v25_expert_activation_hazard
        ):
            return 0.0
        if max(float(summary.get("max_danger", 0.0)), float(summary.get("forward_danger", 0.0))) >= float(
            self.config.v25_expert_hard_risk_threshold
        ):
            return 0.0
        if float(summary.get("trend_need", 0.0)) < float(self.config.v25_expert_trend_warning_need):
            return 0.0
        if float(summary.get("delta_forward_danger", 0.0)) < float(self.config.v25_expert_trend_forward_delta):
            return 0.0
        return 1.0

    def _local_hazard_summary(self) -> dict[str, float]:
        if self.disruptions is None:
            return self._empty_local_hazard_summary()

        if getattr(self.config, "v25_sensor_mode", "sector_radar") == "circle_oracle":
            features = self._circle_oracle_features()
            max_danger = float(features[0])
            mean_danger = float(features[1])
            nearest_closeness = float(features[2])
            nearest_forward_alignment = max(0.0, float(features[3]))
            centroid_forward_alignment = max(0.0, float(features[5]))
            forward_danger = float(features[11])
            directional_danger = max(
                1.20 * forward_danger,
                max_danger * nearest_closeness * nearest_forward_alignment,
                mean_danger * centroid_forward_alignment,
            )
            base_hazard_need = max(directional_danger, 0.25 * max_danger * nearest_closeness)
            summary = {
                "hazard_need": float(np.clip(base_hazard_need, 0.0, 1.0)),
                "base_hazard_need": float(np.clip(base_hazard_need, 0.0, 1.0)),
                "max_danger": float(np.clip(max_danger, 0.0, 1.0)),
                "mean_danger": float(np.clip(mean_danger, 0.0, 1.0)),
                "forward_danger": float(np.clip(forward_danger, 0.0, 1.0)),
                "nearest_closeness": float(np.clip(nearest_closeness, 0.0, 1.0)),
                "nearest_forward_alignment": float(np.clip(nearest_forward_alignment, 0.0, 1.0)),
            }
            history = self._local_hazard_history_features(summary)
            summary.update(history)
            membrane = self._risk_membrane_summary()
            summary.update(membrane)
            if membrane["risk_membrane_wall_ahead"] > 0.0:
                summary["hazard_need"] = float(
                    max(summary["hazard_need"], self.config.v25_expert_activation_hazard)
                )
            summary["gradual_warning"] = self._local_hazard_gradual_warning(summary)
            return summary

        sector_values = self._sector_radar_features()
        max_danger = 0.0
        forward_danger = 0.0
        nearest_closeness = 0.0
        sector_count = int(self.config.v25_radar_sectors)
        sector_width = 360.0 / max(sector_count, 1)
        for sector_idx in range(sector_count):
            base = sector_idx * self.RADAR_FEATURES_PER_SECTOR
            danger = float(sector_values[base + 2])
            closeness = float(sector_values[base + 3])
            center_rel = -180.0 + sector_width * (sector_idx + 0.5)
            forward_alignment = max(0.0, math.cos(math.radians(center_rel)))
            max_danger = max(max_danger, danger)
            nearest_closeness = max(nearest_closeness, closeness)
            forward_danger = max(forward_danger, danger * forward_alignment)
        base_hazard_need = max(1.20 * forward_danger, 0.25 * max_danger * nearest_closeness)
        summary = {
            "hazard_need": float(np.clip(base_hazard_need, 0.0, 1.0)),
            "base_hazard_need": float(np.clip(base_hazard_need, 0.0, 1.0)),
            "max_danger": float(np.clip(max_danger, 0.0, 1.0)),
            "mean_danger": 0.0,
            "forward_danger": float(np.clip(forward_danger, 0.0, 1.0)),
            "nearest_closeness": float(np.clip(nearest_closeness, 0.0, 1.0)),
            "nearest_forward_alignment": 0.0,
        }
        history = self._local_hazard_history_features(summary)
        summary.update(history)
        membrane = self._risk_membrane_summary()
        summary.update(membrane)
        if membrane["risk_membrane_wall_ahead"] > 0.0:
            summary["hazard_need"] = float(max(summary["hazard_need"], self.config.v25_expert_activation_hazard))
        summary["gradual_warning"] = self._local_hazard_gradual_warning(summary)
        return summary

    def _sensor_features(self) -> np.ndarray:
        measured_residual = self._measured_residual_wind()
        residual_delta = measured_residual - self.last_measured_residual_wind

        base_command = self._astar_base_command()
        expected_ground_xy = base_command["desired_ground_velocity_xy"]
        tracking_velocity_error = self.last_ground_velocity_xy - expected_ground_xy

        if getattr(self.config, "v25_sensor_mode", "sector_radar") == "circle_oracle":
            local_view_values = self._circle_oracle_features()
        else:
            local_view_values = self._sector_radar_features()

        return np.asarray(
            [
                measured_residual[0] / self.config.max_wind_speed,
                measured_residual[1] / self.config.max_wind_speed,
                residual_delta[0] / self.config.max_wind_speed,
                residual_delta[1] / self.config.max_wind_speed,
                tracking_velocity_error[0] / self.config.rl_speed_max,
                tracking_velocity_error[1] / self.config.rl_speed_max,
                *local_view_values,
            ],
            dtype=np.float32,
        )

    def _tracking_velocity_error(self, base_command: dict) -> np.ndarray:
        return self.last_ground_velocity_xy - base_command["desired_ground_velocity_xy"]

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        return np.concatenate([base_obs, self._sensor_features()]).astype(np.float32)

    def _simulate_true_transition(
        self,
        commanded_heading_deg: float,
        commanded_airspeed_mps: float,
        commanded_agl_m: float,
        random_layer_time_s: float | None = None,
    ) -> dict:
        travel_dir = self.last_ground_velocity_xy
        if np.linalg.norm(travel_dir) < 1e-6:
            rad = math.radians(commanded_heading_deg)
            travel_dir = np.array([math.cos(rad), math.sin(rad)], dtype=float)
        true_wind, severity = self._true_wind(
            float(self.current_pos[0]),
            float(self.current_pos[1]),
            float(self.current_pos[2]),
            float(self.current_time),
            travel_dir,
            disturbance_time_s=random_layer_time_s,
        )
        command_rad = math.radians(commanded_heading_deg)
        commanded_air_xy = (
            np.array([math.cos(command_rad), math.sin(command_rad)], dtype=float)
            * commanded_airspeed_mps
        )
        predicted_xy = self.current_pos[:2] + (commanded_air_xy + true_wind) * self.dt
        predicted_x, predicted_y = self._clamp_position(float(predicted_xy[0]), float(predicted_xy[1]))
        target_ground_alt = self.estimator.get_altitude(predicted_x, predicted_y)
        commanded_altitude = target_ground_alt + commanded_agl_m
        transition = self.dynamics.advance(
            VehicleStateV25(
                position_xyz=np.asarray(self.current_pos, dtype=float),
                heading_deg=float(self.current_heading),
                airspeed_mps=float(self.current_airspeed),
            ),
            commanded_heading_deg=commanded_heading_deg,
            commanded_airspeed_mps=commanded_airspeed_mps,
            commanded_altitude_m=commanded_altitude,
            true_wind_xy=true_wind,
            dt=self.dt,
        )
        new_pos = np.asarray(transition["position_xyz"], dtype=float)
        new_pos[0], new_pos[1] = self._clamp_position(float(new_pos[0]), float(new_pos[1]))
        transition["position_xyz"] = new_pos
        transition["true_wind_xy"] = true_wind
        transition["disturbance_severity"] = float(severity)
        transition["power_w"] = float(
            self.physics.estimate_power_from_vectors(
                np.asarray(transition["ground_velocity_xyz"], dtype=float),
                np.array([true_wind[0], true_wind[1], 0.0], dtype=float),
            )
        )
        base_risk, _ = self.estimator.get_risk(
            float(new_pos[0]),
            float(new_pos[1]),
            float(new_pos[2]),
            max(float(np.linalg.norm(transition["ground_velocity_xyz"])), 1.0),
            self.current_time + self.dt,
        )
        transition["p_crash"] = float(
            min(
                1.0,
                base_risk
                + (
                    self.disruptions.risk_bonus_at(
                        float(new_pos[0]),
                        float(new_pos[1]),
                        self.current_time + self.dt if random_layer_time_s is None else float(random_layer_time_s),
                    )
                    if self.disruptions
                    else 0.0
                ),
            )
        )
        transition["destructive_core_hit"] = bool(
            self.disruptions.core_hit(
                float(new_pos[0]),
                float(new_pos[1]),
                self.current_time + self.dt if random_layer_time_s is None else float(random_layer_time_s),
            )
            if self.disruptions
            else False
        )
        return transition

    def _probe_random_layer_segment(
        self,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        random_layer_time_s: float | None = None,
        samples: int | None = None,
    ) -> dict:
        """
        Probe the currently observed random layer along a motion segment.

        This is a bounded local safety probe, not a future oracle: callers may
        pass the observation time used for the random layer. The expert rollout
        uses the current snapshot time so moving stochastic hazards are frozen
        during candidate scoring.
        """
        disruptions = getattr(self, "disruptions", None)
        if disruptions is None:
            return {"max_risk_bonus": 0.0, "destructive_core_hit": False}

        sample_count = int(samples if samples is not None else self.config.v25_segment_probe_samples)
        sample_count = max(1, sample_count)
        probe_time = float(self.current_time if random_layer_time_s is None else random_layer_time_s)
        start = np.asarray(start_xyz, dtype=float)
        end = np.asarray(end_xyz, dtype=float)

        max_risk_bonus = 0.0
        core_hit = False
        for idx in range(sample_count + 1):
            alpha = idx / sample_count
            point = (1.0 - alpha) * start + alpha * end
            x = float(point[0])
            y = float(point[1])
            max_risk_bonus = max(max_risk_bonus, float(disruptions.risk_bonus_at(x, y, probe_time)))
            core_hit = core_hit or bool(disruptions.core_hit(x, y, probe_time))
        return {"max_risk_bonus": float(max_risk_bonus), "destructive_core_hit": bool(core_hit)}

    def _transition_is_safe_for_apas(self, transition: dict) -> bool:
        new_pos = np.asarray(transition["position_xyz"], dtype=float)
        if self.estimator.map.is_collision(
            float(new_pos[0]),
            float(new_pos[1]),
            float(new_pos[2]),
            nfz_list_km=self._current_nfz_list(),
        ):
            return False
        if float(transition["power_w"]) > self.config.max_power * self.config.rl_overload_power_ratio:
            return False
        if bool(transition.get("destructive_core_hit", False)):
            return False
        return float(transition["p_crash"]) <= self.config.rl_terminate_risk_threshold

    def _transition_segment_is_safe_for_apas(self, transition: dict) -> tuple[bool, dict]:
        if not bool(getattr(self.config, "v25_apas_segment_check_enabled", True)):
            return True, {"max_risk_bonus": 0.0, "destructive_core_hit": False}

        segment_probe = self._probe_random_layer_segment(
            np.asarray(self.current_pos, dtype=float),
            np.asarray(transition["position_xyz"], dtype=float),
            random_layer_time_s=float(self.current_time),
            samples=int(self.config.v25_apas_segment_probe_samples),
        )
        segment_core_hit = bool(segment_probe["destructive_core_hit"])
        segment_high_risk = (
            float(segment_probe["max_risk_bonus"]) > float(self.config.v25_apas_segment_risk_threshold)
        )
        return not (segment_core_hit or segment_high_risk), segment_probe

    def _simulate_apas_true_transition(
        self,
        desired_heading_deg: float,
        desired_airspeed_mps: float,
        desired_agl_m: float,
    ) -> tuple[Optional[dict], dict]:
        """
        Run the original APAS candidate search through v2.5 true dynamics.

        The first candidate is the RL command. Later candidates progressively
        turn aside, slow down, and request more clearance.
        """
        candidate_index = 0
        segment_rejections = 0
        endpoint_rejections = 0
        least_bad: tuple[tuple[float, float, float, int], Optional[dict], dict] | None = None
        for agl_increment in self.config.v25_apas_agl_increments_m:
            test_agl = float(
                np.clip(
                    desired_agl_m + agl_increment,
                    self.min_clearance_agl,
                    self.max_clearance_agl,
                )
            )
            for heading_offset in self.config.v25_apas_heading_offsets_deg:
                test_heading = self._wrap_angle_deg(desired_heading_deg + heading_offset)
                test_speed = float(desired_airspeed_mps)
                while test_speed >= self.config.rl_speed_min - 1e-9:
                    transition = self._simulate_true_transition(test_heading, test_speed, test_agl)
                    segment_safe, segment_probe = self._transition_segment_is_safe_for_apas(transition)
                    endpoint_safe = self._transition_is_safe_for_apas(transition)
                    if not segment_safe:
                        segment_rejections += 1
                    if not endpoint_safe:
                        endpoint_rejections += 1

                    candidate_score = (
                        1.0 if bool(segment_probe["destructive_core_hit"]) else 0.0,
                        float(segment_probe["max_risk_bonus"]),
                        float(transition["p_crash"]),
                        candidate_index,
                    )
                    candidate_info = {
                        "apas_intervened": candidate_index > 0,
                        "apas_heading_offset_deg": float(heading_offset),
                        "apas_speed_reduction_mps": float(max(0.0, desired_airspeed_mps - test_speed)),
                        "apas_agl_increment_m": float(test_agl - desired_agl_m),
                        "apas_candidate_index": candidate_index,
                        "apas_segment_rejections": int(segment_rejections),
                        "apas_endpoint_rejections": int(endpoint_rejections),
                        "apas_no_valid_candidate": False,
                        "apas_segment_max_risk_bonus": float(segment_probe["max_risk_bonus"]),
                        "apas_segment_core_hit": bool(segment_probe["destructive_core_hit"]),
                    }
                    if least_bad is None or candidate_score < least_bad[0]:
                        least_bad = (candidate_score, transition, candidate_info)

                    if segment_safe and endpoint_safe:
                        return transition, {
                            **candidate_info,
                        }
                    candidate_index += 1
                    test_speed -= self.config.v25_apas_speed_decrement_mps

        if least_bad is not None:
            _, fallback_transition, fallback_info = least_bad
            fallback_info = {
                **fallback_info,
                "apas_intervened": True,
                "apas_no_valid_candidate": True,
                "apas_segment_rejections": int(segment_rejections),
                "apas_endpoint_rejections": int(endpoint_rejections),
            }
            return fallback_transition, fallback_info

        return None, {
            "apas_intervened": True,
            "apas_heading_offset_deg": 0.0,
            "apas_speed_reduction_mps": 0.0,
            "apas_agl_increment_m": 0.0,
            "apas_candidate_index": candidate_index,
            "apas_segment_rejections": int(segment_rejections),
            "apas_endpoint_rejections": int(endpoint_rejections),
            "apas_no_valid_candidate": True,
            "apas_segment_max_risk_bonus": 0.0,
            "apas_segment_core_hit": False,
        }

    def local_avoidance_expert_action(self) -> np.ndarray:
        """
        Return a local MPC-style expert action in the same normalized
        residual space as the policy.

        The expert now has a small explicit mode layer: normal/recovering use
        the A* reference, cautious observes soft risk without taking over,
        avoiding searches ordinary residual candidates, and emergency switches
        to conservative candidates only when ordinary candidates cannot satisfy
        hard safety constraints.
        """
        base_command = self._astar_base_command()
        measured_residual = self._measured_residual_wind()
        tracking_error = self._tracking_velocity_error(base_command)
        path_error_before = float(self._estimate_local_path_error(self.current_pos[:2]))
        local_hazard = self._local_hazard_summary()
        self.last_expert_active = True

        rejoin_action = self._safe_rejoin_action(
            base_command=base_command,
            local_hazard=local_hazard,
            path_error_m=path_error_before,
        )
        if rejoin_action is not None:
            intervention_need = compute_intervention_need(
                float(np.linalg.norm(measured_residual)),
                float(np.linalg.norm(tracking_error)),
                path_error_before,
                self.config,
                local_hazard_need=float(local_hazard["hazard_need"]),
            )
            residual_gate = max(compute_residual_gate(intervention_need, self.config), 1e-6)
            self.last_expert_mode = "recovering"
            self.episode_expert_rejoin_actions += 1
            return np.clip(rejoin_action / residual_gate, -1.0, 1.0).astype(np.float32)

        gradual_warning = bool(float(local_hazard.get("gradual_warning", 0.0)) > 0.0)
        if local_hazard["hazard_need"] < self.config.v25_expert_activation_hazard and not gradual_warning:
            self.last_expert_mode = (
                "recovering"
                if path_error_before > float(self.config.v25_expert_recovery_path_error_m)
                else "normal"
            )
            return np.zeros(3, dtype=np.float32)

        decision_hazard_need = max(
            float(local_hazard["hazard_need"]),
            float(self.config.v25_expert_activation_hazard) if gradual_warning else 0.0,
        )
        intervention_need = compute_intervention_need(
            float(np.linalg.norm(measured_residual)),
            float(np.linalg.norm(tracking_error)),
            path_error_before,
            self.config,
            local_hazard_need=decision_hazard_need,
        )
        residual_gate = max(compute_residual_gate(intervention_need, self.config), 1e-6)

        membrane_decision = self._risk_membrane_action(local_hazard)
        if membrane_decision is not None:
            membrane_action, membrane_mode = membrane_decision
            membrane_eval = self._evaluate_expert_rollout(membrane_action)
            if membrane_mode == "pre_emergency_slow" or int(membrane_eval["hard_violation_count"]) == 0:
                self.last_expert_mode = membrane_mode
                raw_action = membrane_action / residual_gate
                return np.clip(raw_action, -1.0, 1.0).astype(np.float32)

        best_action, mode = self._select_expert_action(gradual_warning=gradual_warning)
        self.last_expert_mode = mode

        raw_action = best_action / residual_gate
        return np.clip(raw_action, -1.0, 1.0).astype(np.float32)

    def _risk_membrane_action(self, local_hazard: dict[str, float]) -> Optional[tuple[np.ndarray, str]]:
        if float(local_hazard.get("risk_membrane_wall_ahead", 0.0)) <= 0.0:
            return None

        if float(local_hazard.get("risk_membrane_no_escape_gap", 0.0)) > 0.0:
            return (
                np.array([0.0, float(self.config.v25_expert_pre_emergency_speed_action), 0.0], dtype=float),
                "pre_emergency_slow",
            )

        gap_angle = float(local_hazard.get("risk_membrane_best_gap_angle_deg", 0.0))
        if abs(gap_angle) < 1e-6:
            return None

        heading_action = float(
            np.clip(
                gap_angle / max(float(self.config.rl_heading_delta_max_deg), 1e-6),
                -float(self.config.v25_expert_band_avoid_heading_limit),
                float(self.config.v25_expert_band_avoid_heading_limit),
            )
        )
        return (
            np.array([heading_action, float(self.config.v25_expert_band_avoid_speed_action), 0.0], dtype=float),
            "band_avoidance",
        )

    def _nearest_path_index(self, pos_xy: np.ndarray) -> int:
        if not getattr(self, "global_astar_path", None):
            return 0
        pos = np.asarray(pos_xy, dtype=float)
        distances = [
            float(np.linalg.norm(pos - np.asarray(point[:2], dtype=float)))
            for point in self.global_astar_path
        ]
        return int(np.argmin(distances))

    def _rejoin_target_point(self, pos_xy: np.ndarray) -> np.ndarray:
        if not getattr(self, "global_astar_path", None):
            return np.asarray(self.goal_pos, dtype=float)
        nearest_idx = self._nearest_path_index(pos_xy)
        current_idx = int(max(0, getattr(self, "current_wp_idx", 0)))
        anchor_idx = max(nearest_idx, current_idx)
        target_idx = min(
            anchor_idx + int(self.config.v25_expert_rejoin_lookahead_wps),
            len(self.global_astar_path) - 1,
        )
        return self._path_point(target_idx)

    def _should_rejoin(self, local_hazard: dict[str, float], path_error_m: float) -> bool:
        if not getattr(self, "global_astar_path", None):
            return False

        hard_danger = (
            float(local_hazard["max_danger"]) >= float(self.config.v25_expert_hard_risk_threshold)
            or float(local_hazard["hazard_need"]) >= float(self.config.v25_expert_hard_risk_threshold)
        )
        if hard_danger:
            return False

        risk_low = float(local_hazard["hazard_need"]) <= float(self.config.v25_expert_rejoin_risk_low_threshold)
        path_drift = path_error_m > float(self.config.v25_expert_recovery_path_error_m)
        avoiding_too_long = (
            int(getattr(self, "consecutive_avoiding_steps", 0))
            >= int(self.config.v25_expert_rejoin_max_avoiding_steps)
        )
        cautious_too_long = bool(getattr(self.config, "v25_expert_rejoin_from_cautious", False)) and (
            int(getattr(self, "consecutive_cautious_steps", 0))
            >= int(self.config.v25_expert_rejoin_max_cautious_steps)
        )
        progress_stalled = bool(getattr(self.config, "v25_expert_rejoin_from_low_progress", False)) and (
            int(getattr(self, "consecutive_low_progress_steps", 0))
            >= int(self.config.v25_expert_rejoin_max_low_progress_steps)
        )

        # Drift recovery should wait for genuinely low local risk. Long avoidance,
        # cautious loitering, and stalled progress can be experimented with, but
        # are disabled by default because they over-trigger in ambiguous cases.
        return bool((path_drift and risk_low) or avoiding_too_long or cautious_too_long or progress_stalled)

    def _rejoin_action(self, base_command: dict) -> np.ndarray:
        target = self._rejoin_target_point(self.current_pos[:2])
        to_target = np.asarray(target[:2], dtype=float) - np.asarray(self.current_pos[:2], dtype=float)
        if float(np.linalg.norm(to_target)) < 1e-6:
            heading_action = 0.0
        else:
            desired_heading_deg = math.degrees(math.atan2(float(to_target[1]), float(to_target[0])))
            heading_error_deg = self._wrap_angle_deg(desired_heading_deg - float(base_command["heading_deg"]))
            heading_action = float(heading_error_deg / max(float(self.config.rl_heading_delta_max_deg), 1e-6))
        heading_action = float(
            np.clip(
                heading_action,
                -float(self.config.v25_expert_rejoin_heading_action_limit),
                float(self.config.v25_expert_rejoin_heading_action_limit),
            )
        )
        return np.array(
            [heading_action, float(self.config.v25_expert_rejoin_speed_action), 0.0],
            dtype=float,
        )

    def _safe_rejoin_action(
        self,
        base_command: dict,
        local_hazard: dict[str, float],
        path_error_m: float,
    ) -> Optional[np.ndarray]:
        if not self._should_rejoin(local_hazard, path_error_m):
            return None
        self.episode_expert_rejoin_attempts = int(getattr(self, "episode_expert_rejoin_attempts", 0)) + 1
        action = self._rejoin_action(base_command)
        evaluation = self._evaluate_expert_rollout(action)
        if int(evaluation["hard_violation_count"]) == 0:
            return action
        self.episode_expert_rejoin_rejected = int(getattr(self, "episode_expert_rejoin_rejected", 0)) + 1
        return None

    def _update_replan_memory(self, progress_m: float, apas_info: dict) -> None:
        if progress_m < float(self.config.v25_expert_rejoin_min_progress_m):
            self.consecutive_replan_low_progress_steps += 1
        else:
            self.consecutive_replan_low_progress_steps = 0

        if bool(apas_info.get("apas_no_valid_candidate", False)):
            self.apas_no_valid_linger_steps = int(self.config.v25_replan_no_valid_linger_steps)
        elif self.apas_no_valid_linger_steps > 0:
            self.apas_no_valid_linger_steps -= 1

        if self.replan_cooldown_steps_remaining > 0:
            self.replan_cooldown_steps_remaining -= 1

    def _replan_trigger_reason(self, local_hazard: dict[str, float], path_error_m: float) -> Optional[str]:
        if not bool(getattr(self.config, "v25_replan_enabled", True)):
            return None
        if int(getattr(self.config, "v25_replan_max_per_episode", 0)) <= 0:
            return None
        if self.episode_replans >= int(self.config.v25_replan_max_per_episode):
            return None
        if self.replan_cooldown_steps_remaining > 0:
            return None
        if self.current_step < int(self.config.v25_replan_min_step):
            return None
        if not getattr(self, "global_astar_path", None) or len(self.global_astar_path) < 2:
            return None

        hazard_need = float(local_hazard.get("hazard_need", 0.0))
        max_danger = float(local_hazard.get("max_danger", 0.0))
        hard_threshold = float(self.config.v25_replan_hard_risk_threshold)
        if max(hazard_need, max_danger) >= hard_threshold:
            return None

        risk_low = hazard_need <= float(self.config.v25_replan_risk_low_threshold)
        path_drift = path_error_m >= float(self.config.v25_replan_path_error_m)
        if path_drift and risk_low:
            return "path_drift"

        if (
            self.apas_no_valid_linger_steps > 0
            and self.consecutive_replan_low_progress_steps >= int(self.config.v25_replan_no_valid_low_progress_steps)
        ):
            return "apas_no_valid_aftershock"

        if self.consecutive_replan_low_progress_steps >= int(self.config.v25_replan_low_progress_steps):
            return "low_progress"

        return None

    def _splice_replanned_path(self, prefix_path: list[tuple[float, float, float]], target_idx: int) -> list:
        suffix = list(self.global_astar_path[min(target_idx + 1, len(self.global_astar_path)):])
        if not suffix:
            return list(prefix_path)
        return list(prefix_path) + suffix

    def _try_replan_to_rejoin(self, reason: str) -> bool:
        old_path = list(self.global_astar_path)
        if len(old_path) < 2:
            return False

        nearest_idx = self._nearest_path_index(self.current_pos[:2])
        anchor_idx = max(nearest_idx, int(max(0, getattr(self, "current_wp_idx", 0))))
        target_idx = min(
            anchor_idx + int(self.config.v25_replan_rejoin_lookahead_wps),
            len(old_path) - 1,
        )
        current_xy = (float(self.current_pos[0]), float(self.current_pos[1]))
        target_xy = (float(old_path[target_idx][0]), float(old_path[target_idx][1]))
        min_points = int(self.config.v25_replan_min_path_points)

        rejoin_path = self.planner.search(current_xy, target_xy, start_time_s=float(self.current_time))
        if rejoin_path is not None and len(rejoin_path) >= min_points:
            self.global_astar_path = self._splice_replanned_path(rejoin_path, target_idx)
            self.current_wp_idx = 1 if len(self.global_astar_path) > 1 else 0
            self._refresh_wp_index(self.current_pos[:2])
            self.episode_replan_to_rejoin_successes += 1
            self.last_replan_event = f"{reason}:to_rejoin"
            return True

        goal_xy = (float(self.goal_pos[0]), float(self.goal_pos[1]))
        goal_path = self.planner.search(current_xy, goal_xy, start_time_s=float(self.current_time))
        if goal_path is not None and len(goal_path) >= min_points:
            self.global_astar_path = list(goal_path)
            self.current_wp_idx = 1 if len(self.global_astar_path) > 1 else 0
            self._refresh_wp_index(self.current_pos[:2])
            self.episode_replan_to_goal_successes += 1
            self.last_replan_event = f"{reason}:to_goal"
            return True

        self.global_astar_path = old_path
        self.last_replan_event = f"{reason}:failed"
        return False

    def _maybe_replan_reference_path(
        self,
        local_hazard: dict[str, float],
        path_error_m: float,
    ) -> tuple[bool, str]:
        reason = self._replan_trigger_reason(local_hazard, path_error_m)
        if reason is None:
            self.last_replan_event = "none"
            return False, "none"

        self.episode_replans += 1
        if reason == "path_drift":
            self.episode_replan_path_drift_triggers += 1
        elif reason == "apas_no_valid_aftershock":
            self.episode_replan_no_valid_triggers += 1
        elif reason == "low_progress":
            self.episode_replan_low_progress_triggers += 1

        success = self._try_replan_to_rejoin(reason)
        if success:
            self.episode_replan_successes += 1
            self.consecutive_replan_low_progress_steps = 0
            self.apas_no_valid_linger_steps = 0
        else:
            self.episode_replan_failures += 1
        self.replan_cooldown_steps_remaining = int(self.config.v25_replan_cooldown_steps)
        return bool(success), reason

    def _expert_candidate_actions(self, emergency: bool = False, mild: bool = False) -> list[np.ndarray]:
        if emergency:
            headings = self.config.v25_expert_emergency_heading_actions
            speeds = self.config.v25_expert_emergency_speed_actions
            agls = self.config.v25_expert_emergency_agl_actions
        elif mild:
            headings = self.config.v25_expert_mild_heading_actions
            speeds = self.config.v25_expert_mild_speed_actions
            agls = self.config.v25_expert_mild_agl_actions
        else:
            headings = self.config.v25_expert_heading_actions
            speeds = self.config.v25_expert_speed_actions
            agls = self.config.v25_expert_agl_actions
        return [
            np.array([heading_action, speed_action, agl_action], dtype=float)
            for heading_action in headings
            for speed_action in speeds
            for agl_action in agls
        ]

    def _select_expert_action(self, gradual_warning: bool = False) -> tuple[np.ndarray, str]:
        zero_action = np.zeros(3, dtype=float)
        zero_eval = self._evaluate_expert_rollout(zero_action)
        candidate_mode = "cautious_trend" if gradual_warning else "avoiding"
        improvement_threshold = (
            float(self.config.v25_expert_trend_risk_improvement_threshold)
            if gradual_warning
            else float(self.config.v25_expert_risk_improvement_threshold)
        )
        normal_evaluations = [
            self._evaluate_expert_rollout(action)
            for action in self._expert_candidate_actions(emergency=False, mild=gradual_warning)
        ]
        safe_normal = [entry for entry in normal_evaluations if int(entry["hard_violation_count"]) == 0]
        if safe_normal:
            best = min(safe_normal, key=lambda entry: float(entry["score"]))
            if int(zero_eval["hard_violation_count"]) == 0:
                risk_improvement = float(zero_eval["max_risk"]) - float(best["max_risk"])
                if risk_improvement < improvement_threshold:
                    return zero_action, "cautious_trend" if gradual_warning else "cautious"
            return np.asarray(best["action"], dtype=float), candidate_mode

        emergency_evaluations = [
            self._evaluate_expert_rollout(action)
            for action in self._expert_candidate_actions(emergency=True)
        ]
        safe_emergency = [entry for entry in emergency_evaluations if int(entry["hard_violation_count"]) == 0]
        pool = safe_emergency if safe_emergency else emergency_evaluations
        best = min(
            pool,
            key=lambda entry: (
                int(entry["hard_violation_count"]),
                float(entry["max_risk"]),
                float(entry["score"]),
            ),
        )
        return np.asarray(best["action"], dtype=float), "emergency"

    def _score_expert_rollout(self, first_action: np.ndarray) -> float:
        return float(self._evaluate_expert_rollout(first_action)["score"])

    def _evaluate_expert_rollout(self, first_action: np.ndarray) -> dict[str, object]:
        saved_state = (
            np.asarray(self.current_pos, dtype=float).copy(),
            float(self.current_heading),
            float(self.current_airspeed),
            np.asarray(self.last_ground_velocity_xy, dtype=float).copy(),
            float(self.current_time),
            int(self.current_wp_idx),
        )
        speed_residual_range = 0.5 * (self.config.rl_speed_max - self.config.rl_speed_min)
        total_score = 0.0
        initial_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - np.asarray(self.current_pos[:2], dtype=float)))
        previous_goal_dist = initial_goal_dist
        final_goal_dist = initial_goal_dist
        final_path_error = float(self._estimate_local_path_error(self.current_pos[:2]))
        hard_violation_count = 0
        max_risk = 0.0
        observed_random_layer_time = float(self.current_time)
        suppress_waypoint_metrics = bool(getattr(self, "_suppress_waypoint_skip_metrics", False))

        try:
            self._suppress_waypoint_skip_metrics = True
            for rollout_step in range(int(self.config.v25_expert_rollout_horizon)):
                decay = float(self.config.v25_expert_rollout_decay) ** rollout_step
                action = np.asarray(first_action, dtype=float) * decay
                base_command = self._astar_base_command()
                commanded_heading = self._wrap_angle_deg(
                    base_command["heading_deg"] + action[0] * self.config.rl_heading_delta_max_deg
                )
                commanded_airspeed = float(
                    np.clip(
                        base_command["airspeed_mps"] + action[1] * speed_residual_range,
                        self.config.rl_speed_min,
                        self.config.rl_speed_max,
                    )
                )
                commanded_agl = float(
                    np.clip(
                        base_command["agl_m"] + action[2] * self.config.rl_agl_delta_max_m,
                        self.min_clearance_agl,
                        self.max_clearance_agl,
                    )
                )
                segment_start = np.asarray(self.current_pos, dtype=float).copy()
                transition = self._simulate_true_transition(
                    commanded_heading,
                    commanded_airspeed,
                    commanded_agl,
                    random_layer_time_s=observed_random_layer_time,
                )
                new_pos = np.asarray(transition["position_xyz"], dtype=float)
                segment_probe = self._probe_random_layer_segment(
                    segment_start,
                    new_pos,
                    random_layer_time_s=observed_random_layer_time,
                )
                new_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - new_pos[:2]))
                progress = previous_goal_dist - new_goal_dist
                progress_shortfall = max(0.0, self.config.v25_reward_min_progress_m_true - progress)
                path_error = float(self._estimate_local_path_error(new_pos[:2]))
                final_goal_dist = new_goal_dist
                final_path_error = path_error
                segment_risk_bonus = float(segment_probe["max_risk_bonus"])
                p_crash = max(float(transition["p_crash"]), segment_risk_bonus)
                max_risk = max(max_risk, p_crash)
                power = float(transition["power_w"])
                core_hit = bool(transition.get("destructive_core_hit", False)) or bool(
                    segment_probe["destructive_core_hit"]
                )
                collision = self.estimator.map.is_collision(
                    float(new_pos[0]),
                    float(new_pos[1]),
                    float(new_pos[2]),
                    nfz_list_km=self._current_nfz_list(),
                )
                overload = power > self.config.max_power * self.config.rl_overload_power_ratio

                step_weight = 1.0 + 0.20 * rollout_step
                total_score += step_weight * (
                    self.config.v25_expert_risk_gain * (p_crash ** 2)
                    + self.config.v25_expert_path_error_gain * path_error
                    + self.config.v25_expert_power_gain * max(0.0, power - self.config.base_power)
                    + self.config.v25_expert_action_gain * float(np.linalg.norm(action))
                    + self.config.v25_expert_progress_gain * progress_shortfall
                    - self.config.v25_expert_progress_gain * progress
                )
                if core_hit or p_crash > float(self.config.v25_expert_hard_risk_threshold):
                    hard_violation_count += 1
                    total_score += self.config.v25_expert_core_penalty * step_weight
                if collision or overload:
                    hard_violation_count += 1
                    total_score += self.config.v25_expert_core_penalty * step_weight

                self.current_pos = new_pos
                self.current_heading = float(transition["heading_deg"])
                self.current_airspeed = float(transition["airspeed_mps"])
                self.last_ground_velocity_xy = np.asarray(transition["ground_velocity_xyz"][:2], dtype=float)
                self.current_time += self.dt
                self._refresh_wp_index(self.current_pos[:2])
                previous_goal_dist = new_goal_dist
            terminal_goal_shortfall = max(0.0, final_goal_dist - initial_goal_dist)
            total_score += (
                self.config.v25_expert_final_path_error_gain * final_path_error
                + self.config.v25_expert_final_progress_gain * terminal_goal_shortfall
            )
            if hard_violation_count:
                total_score += self.config.v25_expert_hard_constraint_penalty * hard_violation_count
        finally:
            (
                self.current_pos,
                self.current_heading,
                self.current_airspeed,
                self.last_ground_velocity_xy,
                self.current_time,
                self.current_wp_idx,
            ) = saved_state
            self._suppress_waypoint_skip_metrics = suppress_waypoint_metrics

        return {
            "action": np.asarray(first_action, dtype=float).copy(),
            "score": float(total_score),
            "hard_violation_count": int(hard_violation_count),
            "max_risk": float(max_risk),
            "final_path_error": float(final_path_error),
            "final_goal_dist": float(final_goal_dist),
        }

    def _apply_do_no_harm_gate(self, raw_action: np.ndarray, local_hazard: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Suppress harmful upper-layer residuals using only already-observed history.

        APAS rejects immediately unsafe motions. This gate is different: it
        detects an upper-layer residual that has recently failed to make mission
        progress while risk is not improving and APAS is already working hard.
        """
        action = np.asarray(raw_action, dtype=float).copy()
        info: dict[str, Any] = {
            "do_no_harm_active": False,
            "do_no_harm_reason": "none",
            "do_no_harm_recent_progress_sum": 0.0,
            "do_no_harm_recent_risk_delta": 0.0,
            "do_no_harm_recent_apas_interventions": 0,
            "do_no_harm_recent_segment_rejections": 0,
            "do_no_harm_recent_no_valid_candidates": 0,
        }
        if not bool(getattr(self.config, "v25_do_no_harm_gate_enabled", True)):
            self.last_do_no_harm_active = False
            self.last_do_no_harm_reason = "disabled"
            info["do_no_harm_reason"] = "disabled"
            return action, info

        upper_layer_active = float(np.linalg.norm(action)) >= float(self.config.v25_do_no_harm_min_raw_action_norm)
        if self.do_no_harm_cooldown_steps_remaining > 0:
            self.do_no_harm_cooldown_steps_remaining -= 1
            if upper_layer_active:
                self.last_do_no_harm_active = True
                self.last_do_no_harm_reason = "cooldown"
                self.episode_do_no_harm_suppressed_steps += 1
                self.episode_do_no_harm_cooldown_steps += 1
                info.update(do_no_harm_active=True, do_no_harm_reason="cooldown")
                return np.zeros_like(action), info
            self.last_do_no_harm_active = False
            self.last_do_no_harm_reason = "cooldown_idle"
            info["do_no_harm_reason"] = "cooldown_idle"
            return action, info

        window = int(max(1, self.config.v25_do_no_harm_window_steps))
        if not upper_layer_active or len(self.do_no_harm_history) < window:
            self.last_do_no_harm_active = False
            self.last_do_no_harm_reason = "inactive" if not upper_layer_active else "insufficient_history"
            info["do_no_harm_reason"] = self.last_do_no_harm_reason
            return action, info

        recent = list(self.do_no_harm_history)[-window:]
        progress_sum = float(sum(float(item["progress_m"]) for item in recent))
        risk_delta = float(float(recent[-1]["p_crash"]) - float(recent[0]["p_crash"]))
        apas_interventions = int(sum(int(item["apas_intervened"]) for item in recent))
        segment_rejections = int(sum(int(item["apas_segment_rejections"]) for item in recent))
        no_valid_candidates = int(sum(int(item["apas_no_valid_candidate"]) for item in recent))
        info.update(
            do_no_harm_recent_progress_sum=progress_sum,
            do_no_harm_recent_risk_delta=risk_delta,
            do_no_harm_recent_apas_interventions=apas_interventions,
            do_no_harm_recent_segment_rejections=segment_rejections,
            do_no_harm_recent_no_valid_candidates=no_valid_candidates,
        )

        bad_progress = progress_sum <= float(self.config.v25_do_no_harm_min_progress_sum_m)
        risk_not_improving = risk_delta >= -float(self.config.v25_do_no_harm_risk_drop_eps)
        apas_struggling = (
            apas_interventions >= int(self.config.v25_do_no_harm_min_apas_interventions)
            or segment_rejections >= int(self.config.v25_do_no_harm_min_segment_rejections)
            or no_valid_candidates >= int(self.config.v25_do_no_harm_min_no_valid_candidates)
        )
        slow_into_risk_wall = (
            risk_not_improving
            and progress_sum <= float(self.config.v25_do_no_harm_slow_trap_max_progress_sum_m)
            and float(action[1]) <= float(self.config.v25_do_no_harm_slow_trap_speed_action)
            and (
                float(local_hazard.get("risk_membrane_no_escape_gap", 0.0)) >= 0.5
                or float(local_hazard.get("risk_membrane_wall_ahead", 0.0)) >= 0.5
                or float(local_hazard.get("forward_danger", 0.0)) >= float(self.config.v25_expert_hard_risk_threshold)
            )
        )
        if bad_progress and risk_not_improving and apas_struggling:
            self.do_no_harm_cooldown_steps_remaining = int(self.config.v25_do_no_harm_cooldown_steps)
            self.episode_do_no_harm_events += 1
            self.episode_do_no_harm_suppressed_steps += 1
            self.last_do_no_harm_active = True
            self.last_do_no_harm_reason = "bad_progress_risk_not_improving"
            info.update(do_no_harm_active=True, do_no_harm_reason="bad_progress_risk_not_improving")
            return np.zeros_like(action), info
        if slow_into_risk_wall:
            self.do_no_harm_cooldown_steps_remaining = int(self.config.v25_do_no_harm_cooldown_steps)
            self.episode_do_no_harm_events += 1
            self.episode_do_no_harm_suppressed_steps += 1
            self.last_do_no_harm_active = True
            self.last_do_no_harm_reason = "slow_into_risk_wall"
            info.update(do_no_harm_active=True, do_no_harm_reason="slow_into_risk_wall")
            return np.zeros_like(action), info

        self.last_do_no_harm_active = False
        self.last_do_no_harm_reason = "clear"
        info["do_no_harm_reason"] = "clear"
        return action, info

    def step(self, action: np.ndarray):
        self.current_step += 1
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        base_command = self._astar_base_command()
        measured_residual_before = self._measured_residual_wind()
        tracking_error_before = self._tracking_velocity_error(base_command)
        path_error_before = float(self._estimate_local_path_error(self.current_pos[:2]))
        local_hazard = self._local_hazard_summary()
        intervention_need = compute_intervention_need(
            float(np.linalg.norm(measured_residual_before)),
            float(np.linalg.norm(tracking_error_before)),
            path_error_before,
            self.config,
            local_hazard_need=local_hazard["hazard_need"],
        )
        residual_gate = compute_residual_gate(intervention_need, self.config)
        raw_action = action.copy()
        action, do_no_harm_info = self._apply_do_no_harm_gate(raw_action, local_hazard)
        action = action * residual_gate
        residual_control_cost, residual_magnitude, action_delta = compute_residual_control_cost(
            action,
            self.previous_action,
            intervention_need,
            self.config,
        )

        # The policy is a true residual controller around the A* reference.
        commanded_heading = self._wrap_angle_deg(
            base_command["heading_deg"] + action[0] * self.config.rl_heading_delta_max_deg
        )
        speed_residual_range = 0.5 * (self.config.rl_speed_max - self.config.rl_speed_min)
        commanded_airspeed = float(
            np.clip(
                base_command["airspeed_mps"] + action[1] * speed_residual_range,
                self.config.rl_speed_min,
                self.config.rl_speed_max,
            )
        )
        commanded_agl = float(
            np.clip(
                base_command["agl_m"] + action[2] * self.config.rl_agl_delta_max_m,
                self.min_clearance_agl,
                self.max_clearance_agl,
            )
        )

        old_pos = np.asarray(self.current_pos, dtype=float).copy()
        old_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - old_pos[:2]))
        apas_info = {
            "apas_intervened": False,
            "apas_heading_offset_deg": 0.0,
            "apas_speed_reduction_mps": 0.0,
            "apas_agl_increment_m": 0.0,
            "apas_candidate_index": 0,
            "apas_segment_rejections": 0,
            "apas_endpoint_rejections": 0,
            "apas_no_valid_candidate": False,
            "apas_segment_max_risk_bonus": 0.0,
            "apas_segment_core_hit": False,
        }
        if self.config.rl_enable_apas:
            transition, apas_info = self._simulate_apas_true_transition(
                commanded_heading,
                commanded_airspeed,
                commanded_agl,
            )
            if transition is None:
                transition = self._simulate_true_transition(
                    base_command["heading_deg"],
                    base_command["airspeed_mps"],
                    base_command["agl_m"],
                )
                apas_info["apas_fallback_to_astar"] = True
            else:
                apas_info["apas_fallback_to_astar"] = False
        else:
            transition = self._simulate_true_transition(commanded_heading, commanded_airspeed, commanded_agl)
            apas_info["apas_fallback_to_astar"] = False
        new_pos = np.asarray(transition["position_xyz"], dtype=float)

        self.current_pos = new_pos
        self.current_heading = float(transition["heading_deg"])
        self.current_airspeed = float(transition["airspeed_mps"])
        self.last_ground_velocity_xy = np.asarray(transition["ground_velocity_xyz"][:2], dtype=float)
        self.current_time += self.dt
        self.energy_remaining -= float(transition["power_w"]) * self.dt
        self.last_disturbance_severity = float(transition["disturbance_severity"])
        self._refresh_wp_index(self.current_pos[:2])

        new_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - self.current_pos[:2]))
        progress = old_goal_dist - new_goal_dist
        path_error = float(self._estimate_local_path_error(self.current_pos[:2]))
        power = float(transition["power_w"])
        p_crash = float(transition["p_crash"])
        progress_shortfall_ratio = float(
            np.clip(
                (self.config.v25_reward_min_progress_m_true - progress)
                / max(self.config.v25_reward_min_progress_m_true, 1e-6),
                0.0,
                1.0,
            )
        )
        unproductive_residual_cost = (
            self.config.v25_reward_unproductive_residual_gain_true
            * residual_magnitude
            * progress_shortfall_ratio
            * (1.0 - intervention_need)
        )
        forward_progress_reward = self.config.v25_reward_forward_progress_gain_true * max(0.0, progress)
        speed_residual_cost = (
            self.config.v25_reward_unproductive_speed_gain_true
            * abs(float(action[1]))
            * (0.35 + 0.65 * (1.0 - intervention_need))
            * (0.50 + 0.50 * progress_shortfall_ratio)
        )
        apas_intervention_cost = (
            self.config.v25_reward_apas_intervention_penalty_true
            if apas_info["apas_intervened"]
            else 0.0
        )
        speed_norm = float(
            np.clip(
                (commanded_airspeed - self.config.rl_speed_min)
                / max(self.config.rl_speed_max - self.config.rl_speed_min, 1e-6),
                0.0,
                1.0,
            )
        )
        local_hazard_cost = (
            self.config.v25_reward_local_hazard_gain_true
            * local_hazard["hazard_need"]
            * (0.45 + 0.55 * speed_norm)
        )

        reward = (
            self.config.v25_reward_progress_gain_true * progress
            + forward_progress_reward
            - self.config.v25_reward_path_error_gain_true * path_error
            - self.config.v25_reward_energy_gain_true * max(0.0, power - self.config.base_power)
            - self.config.v25_reward_risk_gain_true * (p_crash ** 2)
            - residual_control_cost
            - unproductive_residual_cost
            - speed_residual_cost
            - apas_intervention_cost
            - local_hazard_cost
        )

        terminated = False
        truncated = False
        info: Dict[str, Any] = {"is_success": False}
        if np.linalg.norm(self.goal_pos - self.current_pos) < self.config.goal_tolerance_3d_m:
            reward += self.config.v25_goal_reward_true
            terminated = True
            info.update(is_success=True, terminated_reason="goal_reached")
        elif bool(transition.get("destructive_core_hit", False)):
            reward -= self.config.rl_storm_penalty
            terminated = True
            info["terminated_reason"] = "destructive_storm_core"
        elif self.estimator.map.is_collision(
            float(new_pos[0]), float(new_pos[1]), float(new_pos[2]), nfz_list_km=self._current_nfz_list()
        ):
            reward -= self.config.v25_collision_penalty_true
            terminated = True
            info["terminated_reason"] = "terrain_or_nfz"
        elif p_crash > self.config.rl_terminate_risk_threshold:
            reward -= self.config.rl_storm_penalty
            terminated = True
            info["terminated_reason"] = "storm_risk_too_high"
        elif power > self.config.max_power * self.config.rl_overload_power_ratio:
            reward -= self.config.rl_storm_penalty
            terminated = True
            info["terminated_reason"] = "overload"
        elif self.energy_remaining <= 0.0:
            reward -= self.config.rl_battery_penalty
            truncated = True
            info["terminated_reason"] = "battery_depleted"
        elif self.current_step >= self.max_steps:
            reward -= self.config.v25_timeout_penalty_true
            truncated = True
            info["terminated_reason"] = "timeout"

        severity = self.last_disturbance_severity
        self.episode_disturbance_max = max(self.episode_disturbance_max, severity)
        self.episode_disturbance_sum += severity
        self.episode_disturbance_steps += int(severity > 1e-9)
        self.episode_residual_action_sum += residual_magnitude
        self.episode_residual_heading_abs_sum += abs(float(action[0]))
        self.episode_residual_speed_abs_sum += abs(float(action[1]))
        self.episode_residual_agl_abs_sum += abs(float(action[2]))
        self.episode_action_delta_sum += action_delta
        self.episode_intervention_need_sum += intervention_need
        self.episode_unneeded_residual_sum += residual_magnitude * (1.0 - intervention_need)
        self.episode_needed_residual_sum += residual_magnitude * intervention_need
        self.episode_apas_interventions += int(apas_info["apas_intervened"])
        self.episode_apas_segment_rejections += int(apas_info.get("apas_segment_rejections", 0))
        self.episode_apas_no_valid_candidates += int(bool(apas_info.get("apas_no_valid_candidate", False)))
        self.episode_destructive_core_hits += int(bool(transition.get("destructive_core_hit", False)))
        self.episode_unproductive_residual_cost_sum += unproductive_residual_cost
        self.episode_progress_shortfall_sum += progress_shortfall_ratio
        self.episode_apas_intervention_cost_sum += apas_intervention_cost
        self.episode_speed_residual_cost_sum += speed_residual_cost
        self.episode_residual_gate_sum += residual_gate
        self.episode_local_hazard_need_sum += local_hazard["hazard_need"]
        self.episode_local_hazard_cost_sum += local_hazard_cost
        self.episode_local_hazard_trend_need_sum += float(local_hazard.get("trend_need", 0.0))
        self.episode_local_hazard_forward_delta_sum += float(local_hazard.get("delta_forward_danger", 0.0))
        self.episode_local_hazard_positive_trend_steps += int(float(local_hazard.get("trend_need", 0.0)) > 1e-6)
        self.do_no_harm_history.append(
            {
                "progress_m": float(progress),
                "p_crash": float(p_crash),
                "apas_intervened": int(bool(apas_info.get("apas_intervened", False))),
                "apas_segment_rejections": int(apas_info.get("apas_segment_rejections", 0)),
                "apas_no_valid_candidate": int(bool(apas_info.get("apas_no_valid_candidate", False))),
            }
        )
        expert_mode_this_step = self.last_expert_mode if self.last_expert_active else "inactive"
        self.episode_expert_normal_steps += int(expert_mode_this_step == "normal")
        self.episode_expert_cautious_steps += int(expert_mode_this_step == "cautious")
        self.episode_expert_cautious_trend_steps += int(expert_mode_this_step == "cautious_trend")
        self.episode_expert_avoiding_steps += int(expert_mode_this_step == "avoiding")
        self.episode_expert_emergency_steps += int(expert_mode_this_step == "emergency")
        self.episode_expert_band_avoidance_steps += int(expert_mode_this_step == "band_avoidance")
        self.episode_expert_pre_emergency_slow_steps += int(expert_mode_this_step == "pre_emergency_slow")
        self.episode_expert_recovering_steps += int(expert_mode_this_step == "recovering")
        expert_mode_burden = 0.0
        if expert_mode_this_step in ("cautious", "cautious_trend"):
            expert_mode_burden = float(self.config.v25_eval_expert_cautious_burden)
        elif expert_mode_this_step in ("avoiding", "band_avoidance"):
            expert_mode_burden = float(self.config.v25_eval_expert_avoiding_burden)
        elif expert_mode_this_step in ("emergency", "pre_emergency_slow"):
            expert_mode_burden = float(self.config.v25_eval_expert_emergency_burden)
        elif expert_mode_this_step == "recovering":
            expert_mode_burden = float(self.config.v25_eval_expert_recovering_burden)

        residual_maneuver_extra_energy_j = (
            float(self.config.v25_eval_maneuver_heading_energy_j) * abs(float(action[0])) ** 2
            + float(self.config.v25_eval_maneuver_speed_energy_j) * abs(float(action[1])) ** 2
            + float(self.config.v25_eval_maneuver_agl_energy_j) * abs(float(action[2])) ** 2
            + float(self.config.v25_eval_maneuver_action_delta_energy_j) * (action_delta ** 2)
        )
        apas_maneuver_extra_energy_j = 0.0
        if bool(apas_info.get("apas_intervened", False)):
            apas_maneuver_extra_energy_j = (
                float(self.config.v25_eval_apas_fixed_energy_j)
                + float(self.config.v25_eval_apas_heading_energy_j_per_deg)
                * abs(float(apas_info.get("apas_heading_offset_deg", 0.0)))
                + float(self.config.v25_eval_apas_speed_reduction_energy_j_per_mps2)
                * (max(0.0, float(apas_info.get("apas_speed_reduction_mps", 0.0))) ** 2)
                + float(self.config.v25_eval_apas_agl_energy_j_per_m)
                * max(0.0, float(apas_info.get("apas_agl_increment_m", 0.0)))
            )
        maneuver_extra_energy_j = float(residual_maneuver_extra_energy_j + apas_maneuver_extra_energy_j)
        safety_intervention_burden = (
            expert_mode_burden
            + float(self.config.v25_eval_apas_intervention_burden)
            * float(bool(apas_info.get("apas_intervened", False)))
            + float(self.config.v25_eval_apas_segment_rejection_burden)
            * float(apas_info.get("apas_segment_rejections", 0))
            + float(self.config.v25_eval_apas_no_valid_burden)
            * float(bool(apas_info.get("apas_no_valid_candidate", False)))
        )
        adjusted_energy_step_j = (
            power * self.dt
            + maneuver_extra_energy_j
            + safety_intervention_burden * float(self.config.v25_eval_burden_energy_equivalent_j)
        )
        self.episode_eval_maneuver_extra_energy_j += maneuver_extra_energy_j
        self.episode_eval_safety_intervention_burden += safety_intervention_burden
        self.episode_eval_adjusted_energy_j += adjusted_energy_step_j
        if expert_mode_this_step in ("avoiding", "emergency", "band_avoidance", "pre_emergency_slow"):
            self.consecutive_avoiding_steps += 1
        else:
            self.consecutive_avoiding_steps = 0
        if expert_mode_this_step in ("cautious", "cautious_trend"):
            self.consecutive_cautious_steps += 1
        else:
            self.consecutive_cautious_steps = 0
        if expert_mode_this_step == "recovering":
            self.consecutive_recovering_steps += 1
        else:
            self.consecutive_recovering_steps = 0
        if (
            expert_mode_this_step in ("cautious", "cautious_trend", "avoiding", "band_avoidance", "recovering")
            and progress < float(self.config.v25_expert_rejoin_min_progress_m)
        ):
            self.consecutive_low_progress_steps += 1
        else:
            self.consecutive_low_progress_steps = 0
        self._update_replan_memory(progress, apas_info)
        replan_success, replan_reason = (False, "none")
        if not terminated and not truncated:
            replan_success, replan_reason = self._maybe_replan_reference_path(local_hazard, path_error)
        self.last_expert_active = False
        self.previous_action = action.copy()
        self._remember_local_hazard(local_hazard)

        measured_residual = self._measured_residual_wind()
        measured_risk = float(
            np.clip(
                p_crash + self.sensor_rng.normal(0.0, self.config.v25_sensor_risk_noise_std),
                0.0,
                1.0,
            )
        )
        self.last_measured_residual_wind = measured_residual

        info.update(
            {
                "power_w": power,
                "p_crash": p_crash,
                "measured_risk": measured_risk,
                "goal_dist_m": new_goal_dist,
                "path_error_m": path_error,
                "energy_remaining_j": self.energy_remaining,
                "disturbance_severity": severity,
                "measured_residual_wind_x": float(measured_residual[0]),
                "measured_residual_wind_y": float(measured_residual[1]),
                "ground_speed_mps": float(np.linalg.norm(self.last_ground_velocity_xy)),
                "airspeed_mps": self.current_airspeed,
                "base_heading_deg": float(base_command["heading_deg"]),
                "base_airspeed_mps": float(base_command["airspeed_mps"]),
                "base_agl_m": float(base_command["agl_m"]),
                "commanded_heading_deg": float(commanded_heading),
                "commanded_airspeed_mps": float(commanded_airspeed),
                "commanded_agl_m": float(commanded_agl),
                "residual_action_magnitude": residual_magnitude,
                "residual_gate": residual_gate,
                "raw_residual_heading_action": float(raw_action[0]),
                "raw_residual_speed_action": float(raw_action[1]),
                "raw_residual_agl_action": float(raw_action[2]),
                "residual_heading_action": float(action[0]),
                "residual_speed_action": float(action[1]),
                "residual_agl_action": float(action[2]),
                "residual_action_delta": action_delta,
                "do_no_harm_active": bool(do_no_harm_info.get("do_no_harm_active", False)),
                "do_no_harm_reason": str(do_no_harm_info.get("do_no_harm_reason", "none")),
                "do_no_harm_recent_progress_sum": float(
                    do_no_harm_info.get("do_no_harm_recent_progress_sum", 0.0)
                ),
                "do_no_harm_recent_risk_delta": float(do_no_harm_info.get("do_no_harm_recent_risk_delta", 0.0)),
                "do_no_harm_recent_apas_interventions": int(
                    do_no_harm_info.get("do_no_harm_recent_apas_interventions", 0)
                ),
                "do_no_harm_recent_segment_rejections": int(
                    do_no_harm_info.get("do_no_harm_recent_segment_rejections", 0)
                ),
                "do_no_harm_recent_no_valid_candidates": int(
                    do_no_harm_info.get("do_no_harm_recent_no_valid_candidates", 0)
                ),
                "do_no_harm_cooldown_steps_remaining": int(self.do_no_harm_cooldown_steps_remaining),
                "residual_control_cost": residual_control_cost,
                "unproductive_residual_cost": float(unproductive_residual_cost),
                "forward_progress_reward": float(forward_progress_reward),
                "speed_residual_cost": float(speed_residual_cost),
                "local_hazard_cost": float(local_hazard_cost),
                "maneuver_extra_energy_j": float(maneuver_extra_energy_j),
                "safety_intervention_burden": float(safety_intervention_burden),
                "adjusted_energy_step_j": float(adjusted_energy_step_j),
                "local_hazard_need": float(local_hazard["hazard_need"]),
                "local_hazard_base_need": float(local_hazard.get("base_hazard_need", local_hazard["hazard_need"])),
                "local_hazard_gradual_warning": float(local_hazard.get("gradual_warning", 0.0)),
                "local_hazard_trend_need": float(local_hazard.get("trend_need", 0.0)),
                "local_hazard_delta_max_danger": float(local_hazard.get("delta_max_danger", 0.0)),
                "local_hazard_delta_forward_danger": float(local_hazard.get("delta_forward_danger", 0.0)),
                "local_hazard_delta_nearest_closeness": float(local_hazard.get("delta_nearest_closeness", 0.0)),
                "local_hazard_max_danger": float(local_hazard["max_danger"]),
                "local_hazard_forward_danger": float(local_hazard["forward_danger"]),
                "local_hazard_nearest_closeness": float(local_hazard["nearest_closeness"]),
                "risk_membrane_wall_ahead": float(local_hazard.get("risk_membrane_wall_ahead", 0.0)),
                "risk_membrane_no_escape_gap": float(local_hazard.get("risk_membrane_no_escape_gap", 0.0)),
                "risk_membrane_front_blocked_width_deg": float(
                    local_hazard.get("risk_membrane_front_blocked_width_deg", 0.0)
                ),
                "risk_membrane_best_gap_angle_deg": float(
                    local_hazard.get("risk_membrane_best_gap_angle_deg", 0.0)
                ),
                "risk_membrane_best_gap_width_deg": float(
                    local_hazard.get("risk_membrane_best_gap_width_deg", 0.0)
                ),
                "risk_membrane_max_extended_risk": float(
                    local_hazard.get("risk_membrane_max_extended_risk", 0.0)
                ),
                "progress_shortfall_ratio": progress_shortfall_ratio,
                "apas_intervention_cost": float(apas_intervention_cost),
                "intervention_need": intervention_need,
                "expert_mode": expert_mode_this_step,
                "episode_disturbance_max": self.episode_disturbance_max,
                "episode_disturbance_mean": self.episode_disturbance_sum / max(self.current_step, 1),
                "episode_disturbance_steps": self.episode_disturbance_steps,
                "episode_residual_action_sum": self.episode_residual_action_sum,
                "episode_residual_heading_abs_sum": self.episode_residual_heading_abs_sum,
                "episode_residual_speed_abs_sum": self.episode_residual_speed_abs_sum,
                "episode_residual_agl_abs_sum": self.episode_residual_agl_abs_sum,
                "episode_action_delta_sum": self.episode_action_delta_sum,
                "episode_intervention_need_mean": self.episode_intervention_need_sum / max(self.current_step, 1),
                "episode_unneeded_residual_sum": self.episode_unneeded_residual_sum,
                "episode_needed_residual_sum": self.episode_needed_residual_sum,
                "episode_apas_interventions": self.episode_apas_interventions,
                "episode_apas_segment_rejections": self.episode_apas_segment_rejections,
                "episode_apas_no_valid_candidates": self.episode_apas_no_valid_candidates,
                "episode_stale_waypoint_skips": self.episode_stale_waypoint_skips,
                "episode_stale_waypoint_skip_delta": self.episode_stale_waypoint_skip_delta,
                "episode_destructive_core_hits": self.episode_destructive_core_hits,
                "episode_unproductive_residual_cost_sum": self.episode_unproductive_residual_cost_sum,
                "episode_progress_shortfall_mean": self.episode_progress_shortfall_sum / max(self.current_step, 1),
                "episode_apas_intervention_cost_sum": self.episode_apas_intervention_cost_sum,
                "episode_speed_residual_cost_sum": self.episode_speed_residual_cost_sum,
                "episode_residual_gate_mean": self.episode_residual_gate_sum / max(self.current_step, 1),
                "episode_local_hazard_need_mean": self.episode_local_hazard_need_sum / max(self.current_step, 1),
                "episode_local_hazard_cost_sum": self.episode_local_hazard_cost_sum,
                "episode_local_hazard_trend_need_mean": (
                    self.episode_local_hazard_trend_need_sum / max(self.current_step, 1)
                ),
                "episode_local_hazard_forward_delta_mean": (
                    self.episode_local_hazard_forward_delta_sum / max(self.current_step, 1)
                ),
                "episode_local_hazard_positive_trend_steps": self.episode_local_hazard_positive_trend_steps,
                "episode_eval_maneuver_extra_energy_j": self.episode_eval_maneuver_extra_energy_j,
                "episode_eval_safety_intervention_burden": self.episode_eval_safety_intervention_burden,
                "episode_eval_adjusted_energy_j": self.episode_eval_adjusted_energy_j,
                "episode_expert_normal_steps": self.episode_expert_normal_steps,
                "episode_expert_cautious_steps": self.episode_expert_cautious_steps,
                "episode_expert_cautious_trend_steps": self.episode_expert_cautious_trend_steps,
                "episode_expert_avoiding_steps": self.episode_expert_avoiding_steps,
                "episode_expert_emergency_steps": self.episode_expert_emergency_steps,
                "episode_expert_band_avoidance_steps": self.episode_expert_band_avoidance_steps,
                "episode_expert_pre_emergency_slow_steps": self.episode_expert_pre_emergency_slow_steps,
                "episode_expert_recovering_steps": self.episode_expert_recovering_steps,
                "episode_expert_rejoin_actions": self.episode_expert_rejoin_actions,
                "episode_expert_rejoin_attempts": self.episode_expert_rejoin_attempts,
                "episode_expert_rejoin_rejected": self.episode_expert_rejoin_rejected,
                "episode_do_no_harm_events": self.episode_do_no_harm_events,
                "episode_do_no_harm_suppressed_steps": self.episode_do_no_harm_suppressed_steps,
                "episode_do_no_harm_cooldown_steps": self.episode_do_no_harm_cooldown_steps,
                "replan_triggered": bool(replan_reason != "none"),
                "replan_success": bool(replan_success),
                "replan_reason": str(replan_reason),
                "last_replan_event": str(self.last_replan_event),
                "episode_replans": self.episode_replans,
                "episode_replan_successes": self.episode_replan_successes,
                "episode_replan_failures": self.episode_replan_failures,
                "episode_replan_to_rejoin_successes": self.episode_replan_to_rejoin_successes,
                "episode_replan_to_goal_successes": self.episode_replan_to_goal_successes,
                "episode_replan_path_drift_triggers": self.episode_replan_path_drift_triggers,
                "episode_replan_low_progress_triggers": self.episode_replan_low_progress_triggers,
                "episode_replan_no_valid_triggers": self.episode_replan_no_valid_triggers,
                "replan_cooldown_steps_remaining": self.replan_cooldown_steps_remaining,
                "consecutive_replan_low_progress_steps": self.consecutive_replan_low_progress_steps,
                "apas_no_valid_linger_steps": self.apas_no_valid_linger_steps,
                "consecutive_avoiding_steps": self.consecutive_avoiding_steps,
                "consecutive_recovering_steps": self.consecutive_recovering_steps,
                "consecutive_cautious_steps": self.consecutive_cautious_steps,
                "consecutive_low_progress_steps": self.consecutive_low_progress_steps,
                "destructive_core_hit": bool(transition.get("destructive_core_hit", False)),
                **apas_info,
            }
        )
        self.telemetry_time_s.append(self.current_time)
        self.telemetry_power_w.append(power)
        self.telemetry_risk.append(p_crash)
        self.telemetry_max_p_crash = max(self.telemetry_max_p_crash, p_crash)
        self.prev_goal_dist = new_goal_dist
        return self._get_obs(), float(reward), terminated, truncated, info
