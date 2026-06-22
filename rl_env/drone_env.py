import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import copy
from typing import Tuple, Dict, Any, Optional

from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner


class GuidedDroneEnv(gym.Env):
    """
    4D A* 全局引导 + RL 局部实时修正环境。
    支持多环境并行，配置深度解耦。
    """
    metadata = {"render_modes": []}

    def __init__(self, config: SimulationConfig):
        super().__init__()
        
        # 🌟 核心：深拷贝，切断与外部全局 config 的引用联系
        self.config = copy.deepcopy(config)

        self.map_manager = MapManager(self.config)
        self.wind_model = WindModelFactory.create(
            self.config.wind_model_type,
            self.config,
            bounds=self.map_manager.get_bounds(),
        )
        self.estimator = StateEstimator(self.map_manager, self.wind_model, self.config)
        self.physics = PhysicsEngine(self.config)
        self.planner = AStarPlanner(self.config, self.estimator, self.physics)

        # 🌟 动作空间严格归一化至 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # 🌟 观测空间：12个基础 + 3个4D时差(差分)特征 + 16个雷达扫描 = 31维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31,),
            dtype=np.float32,
        )

        self.current_pos = np.zeros(3, dtype=np.float64)
        self.current_heading = 0.0
        self.current_time = 0.0
        self.energy_remaining = float(self.config.battery_capacity_j)
        self.current_step = 0
        
        if self.config.curriculum_stage >= 4:
            self.max_steps = self.config.rl_max_steps_stage4
        else:
            self.max_steps = self.config.rl_max_steps
        self.dt = self.config.rl_dt

        self.global_astar_path = []
        self.current_wp_idx = 0
        self.goal_pos = np.zeros(3, dtype=np.float64)
        self.prev_goal_dist = 0.0
        self.goal_dist_50_steps_ago = 0.0

        self.teacher_target_speed = float(self.config.drone_speed)
        self.min_clearance_agl = self.config.rl_min_clearance_agl
        self.max_clearance_agl = min(self.config.rl_max_clearance_agl, self.config.max_ceiling - 20.0)

        self.relative_scan_angles_deg = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]
        self.scan_dist = self.config.rl_scan_distance_m

        # episode 级状态
        self.episode_wind_seed = int(self.config.wind_seed)
        self.episode_nfz_list_km = list(self.config.nfz_list_km)
        self.telemetry_time_s = []
        self.telemetry_power_w = []
        self.telemetry_risk = []
        self.telemetry_max_p_crash = 0.0

    def _wrap_angle_deg(self, angle_deg: float) -> float:
        return (angle_deg + 180.0) % 360.0 - 180.0

    def _clamp_position(self, x: float, y: float) -> Tuple[float, float]:
        min_x, max_x, min_y, max_y = self.estimator.get_bounds()
        return float(np.clip(x, min_x, max_x)), float(np.clip(y, min_y, max_y))

    def _current_nfz_list(self):
        return self.episode_nfz_list_km

    def _is_safe_location(self, x: float, y: float, time_s: float = 0.0) -> bool:
        if self.map_manager.is_in_nfz(x, y, nfz_list_km=self._current_nfz_list()):
            return False

        z = self.estimator.get_altitude(x, y) + self.config.takeoff_altitude_agl
        p_crash, _ = self.estimator.get_risk(
            x, y, z,
            v_ground=self.config.drone_speed,
            t_s=time_s
        )
        return p_crash <= self.config.rl_safe_spawn_risk_threshold

    def _current_agl(self) -> float:
        ground_alt = self.estimator.get_altitude(self.current_pos[0], self.current_pos[1])
        return max(0.0, self.current_pos[2] - ground_alt)

    def _path_point(self, idx: int) -> np.ndarray:
        idx = max(0, min(idx, len(self.global_astar_path) - 1))
        return np.array(self.global_astar_path[idx], dtype=np.float64)

    def _refresh_wp_index(self, pos_xy: np.ndarray) -> None:
        while self.current_wp_idx < len(self.global_astar_path) - 1:
            wp = self._path_point(self.current_wp_idx)
            if np.linalg.norm(pos_xy - wp[:2]) < self.config.rl_waypoint_refresh_radius_m:
                self.current_wp_idx += 1
            else:
                break

    def _estimate_local_path_error(self, pos_xy: np.ndarray) -> float:
        local_indices = range(max(0, self.current_wp_idx - 2), min(len(self.global_astar_path), self.current_wp_idx + 3))
        if not self.global_astar_path:
            return 0.0
        return min(np.linalg.norm(pos_xy - self._path_point(i)[:2]) for i in local_indices)

    def _simulate_nominal_transition(
        self,
        heading_deg: float,
        speed_mps: float,
        desired_agl: float,
    ) -> Dict[str, Any]:
        rad = math.radians(heading_deg)
        dx = speed_mps * math.cos(rad) * self.dt
        dy = speed_mps * math.sin(rad) * self.dt
        new_x, new_y = self._clamp_position(self.current_pos[0] + dx, self.current_pos[1] + dy)

        terrain_alt = self.estimator.get_altitude(new_x, new_y)
        new_z = terrain_alt + desired_agl
        vz = (new_z - self.current_pos[2]) / self.dt

        wind_2d = self.estimator.get_wind(new_x, new_y, new_z, t_s=self.current_time + self.dt)
        wind_3d = np.array([wind_2d[0], wind_2d[1], 0.0], dtype=np.float64)
        ground_velocity = np.array(
            [
                (new_x - self.current_pos[0]) / self.dt,
                (new_y - self.current_pos[1]) / self.dt,
                vz,
            ],
            dtype=np.float64,
        )
        power = float(self.physics.estimate_power_from_vectors(ground_velocity, wind_3d))
        v_ground = float(np.linalg.norm(ground_velocity))
        p_crash, _ = self.estimator.get_risk(
            new_x, new_y, new_z, max(v_ground, 1.0), self.current_time + self.dt
        )
        return {
            "heading_deg": heading_deg,
            "speed_mps": float(speed_mps),
            "new_x": float(new_x),
            "new_y": float(new_y),
            "new_z": float(new_z),
            "power_w": float(power),
            "p_crash": float(p_crash),
            "v_ground": float(v_ground),
        }

    def _simulate_apas_transition(
        self,
        desired_heading_deg: float,
        target_speed_mps: float,
        desired_agl: float,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        power_limit = self.config.max_power * self.config.rl_overload_power_ratio
        fatal_reason = "overload"
        heading_candidates = [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 90.0, -90.0]
        emergency_min_speed = 1.0

        for h_off in heading_candidates:
            heading_deg = self._wrap_angle_deg(desired_heading_deg + h_off)
            test_speed = float(target_speed_mps)

            while test_speed >= emergency_min_speed:
                rad = math.radians(heading_deg)
                dx = test_speed * math.cos(rad) * self.dt
                dy = test_speed * math.sin(rad) * self.dt
                new_x, new_y = self._clamp_position(self.current_pos[0] + dx, self.current_pos[1] + dy)

                terrain_alt_new = self.estimator.get_altitude(new_x, new_y)
                theoretical_z = terrain_alt_new + desired_agl
                vz_req = (theoretical_z - self.current_pos[2]) / self.dt

                if vz_req > 8.0:
                    fatal_reason = "terrain_or_nfz"
                    test_speed -= 2.0
                    continue

                if vz_req < -12.0:
                    vz_req = -12.0

                actual_new_z = self.current_pos[2] + vz_req * self.dt
                if self.estimator.map.is_collision(new_x, new_y, actual_new_z, nfz_list_km=self._current_nfz_list()):
                    fatal_reason = "terrain_or_nfz"
                    break

                wind_2d = self.estimator.get_wind(new_x, new_y, actual_new_z, t_s=self.current_time + self.dt)
                wind_3d = np.array([wind_2d[0], wind_2d[1], 0.0], dtype=np.float64)
                ground_velocity = np.array(
                    [
                        (new_x - self.current_pos[0]) / self.dt,
                        (new_y - self.current_pos[1]) / self.dt,
                        vz_req,
                    ],
                    dtype=np.float64,
                )
                power = float(self.physics.estimate_power_from_vectors(ground_velocity, wind_3d))
                if power <= power_limit:
                    v_ground = float(np.linalg.norm(ground_velocity))
                    p_crash, _ = self.estimator.get_risk(
                        new_x, new_y, actual_new_z, max(v_ground, 1.0), self.current_time + self.dt
                    )
                    return (
                        {
                            "heading_deg": float(heading_deg),
                            "speed_mps": float(test_speed),
                            "new_x": float(new_x),
                            "new_y": float(new_y),
                            "new_z": float(actual_new_z),
                            "power_w": float(power),
                            "p_crash": float(p_crash),
                            "v_ground": float(v_ground),
                        },
                        fatal_reason,
                    )

                fatal_reason = "overload"
                test_speed -= 2.0

        return None, fatal_reason

    def _path_total_length(self, path):
        if path is None or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            total += float(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])))
        return total

    def _teacher_reference(self) -> Dict[str, float]:
        if len(self.global_astar_path) < 2:
            goal_vec = self.goal_pos[:2] - self.current_pos[:2]
            desired_heading = math.degrees(math.atan2(goal_vec[1], goal_vec[0]))
            desired_agl = self.config.takeoff_altitude_agl
            return {
                "heading_deg": desired_heading,
                "speed_mps": self.teacher_target_speed,
                "agl_m": desired_agl,
            }

        self._refresh_wp_index(self.current_pos[:2])
        current_ref = self._path_point(self.current_wp_idx)
        lookahead_idx = min(self.current_wp_idx + 2, len(self.global_astar_path) - 1)
        lookahead_ref = self._path_point(lookahead_idx)

        ref_vec = lookahead_ref[:2] - self.current_pos[:2]
        if np.linalg.norm(ref_vec) < 1e-6:
            ref_vec = current_ref[:2] - self.current_pos[:2]
        desired_heading = math.degrees(math.atan2(ref_vec[1], ref_vec[0]))

        seg_vec = lookahead_ref - current_ref
        seg_dist = np.linalg.norm(seg_vec)
        if seg_dist > 1e-6:
            seg_time = seg_dist / max(self.config.drone_speed, 1.0)
            desired_speed = float(np.clip(seg_dist / max(seg_time, 1e-6), self.config.rl_speed_min, self.config.rl_speed_max))
        else:
            desired_speed = self.teacher_target_speed

        terrain_alt = self.estimator.get_altitude(self.current_pos[0], self.current_pos[1])
        desired_agl = float(np.clip(current_ref[2] - terrain_alt, self.min_clearance_agl, self.max_clearance_agl))

        return {
            "heading_deg": desired_heading,
            "speed_mps": desired_speed,
            "agl_m": desired_agl,
        }

    def _apply_episode_randomization(self, rng, options):
        # 1) 风暴种子
        if options is None:
            self.episode_wind_seed = int(rng.integers(0, 100000))
        else:
            self.episode_wind_seed = int(self.config.wind_seed)
        
        # 🌟 关键：同步给私有 config，让 A* 自动吃对风场
        self.config.wind_seed = self.episode_wind_seed

        if hasattr(self.estimator.wind, "storm_manager"):
            # 🌟 适配无状态风场：每次环境重置，用新的 seed 重新生成长达两小时的风暴队列
            self.estimator.wind.storm_manager.config.wind_seed = self.episode_wind_seed
            self.estimator.wind.storm_manager._pregenerate_storms()

        # 2) NFZ
        if options is not None and "start_xy" in options and "goal_xy" in options:
            self.episode_nfz_list_km = list(self.config.nfz_list_km)
            return

        min_x, max_x, min_y, max_y = self.estimator.get_bounds()

        if self.config.curriculum_stage == 1:
            nfz_low, nfz_high = self.config.rl_nfz_count_stage1_min, self.config.rl_nfz_count_stage1_max
        elif self.config.curriculum_stage == 2:
            nfz_low, nfz_high = self.config.rl_nfz_count_stage2_min, self.config.rl_nfz_count_stage2_max
        elif self.config.curriculum_stage == 3:
            nfz_low, nfz_high = self.config.rl_nfz_count_stage3_min, self.config.rl_nfz_count_stage3_max
        else:
            nfz_low, nfz_high = self.config.rl_nfz_count_stage4_min, self.config.rl_nfz_count_stage4_max

        self.episode_nfz_list_km = []
        for _ in range(int(rng.integers(nfz_low, nfz_high + 1))):
            cx = float(rng.uniform(
                min_x / 1000.0 + self.config.rl_nfz_spawn_margin_m / 1000.0,
                max_x / 1000.0 - self.config.rl_nfz_spawn_margin_m / 1000.0,
            ))
            cy = float(rng.uniform(
                min_y / 1000.0 + self.config.rl_nfz_spawn_margin_m / 1000.0,
                max_y / 1000.0 - self.config.rl_nfz_spawn_margin_m / 1000.0,
            ))
            r = float(rng.uniform(self.config.rl_nfz_radius_min_km, self.config.rl_nfz_radius_max_km))
            self.episode_nfz_list_km.append((cx, cy, r))
            
        # 🌟 关键：同步给私有 config，让 A* 自动避开随机生成的 NFZ
        self.config.nfz_list_km = self.episode_nfz_list_km

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        min_x, max_x, min_y, max_y = self.estimator.get_bounds()

        for attempt_idx in range(self.config.rl_reset_max_attempts):
            self._apply_episode_randomization(rng, options)

            chosen_start_xy = None
            chosen_goal_xy = None
            chosen_start_z = None
            self.global_astar_path = []

            # ==========================================================
            # A. 评估模式
            # ==========================================================
            if options is not None and "start_xy" in options and "goal_xy" in options:
                chosen_start_xy = options["start_xy"]
                chosen_goal_xy = options["goal_xy"]
                chosen_start_z = self.estimator.get_altitude(*chosen_start_xy) + self.config.takeoff_altitude_agl

                self.global_astar_path = self.planner.search(
                    chosen_start_xy, chosen_goal_xy, start_time_s=0.0
                )

                if not self.global_astar_path:
                    self.global_astar_path = [
                        (chosen_start_xy[0], chosen_start_xy[1], chosen_start_z),
                        (chosen_goal_xy[0], chosen_goal_xy[1], self.estimator.get_altitude(*chosen_goal_xy) + self.config.takeoff_altitude_agl),
                    ]
                break

            # ==========================================================
            # B. 训练模式
            # ==========================================================
            if self.config.curriculum_stage == 1:
                goal_min, goal_max = self.config.rl_goal_min_stage1_m, self.config.rl_goal_max_stage1_m
                max_teacher_len = self.config.rl_teacher_len_stage1_max
            elif self.config.curriculum_stage == 2:
                goal_min, goal_max = self.config.rl_goal_min_stage2_m, self.config.rl_goal_max_stage2_m
                max_teacher_len = self.config.rl_teacher_len_stage2_max
            elif self.config.curriculum_stage == 3:
                goal_min, goal_max = self.config.rl_goal_min_stage3_m, self.config.rl_goal_max_stage3_m
                max_teacher_len = self.config.rl_teacher_len_stage3_max
            else:
                goal_min, goal_max = self.config.rl_goal_min_stage4_m, self.config.rl_goal_max_stage4_m
                max_teacher_len = self.config.rl_teacher_len_stage4_max

            valid_plan = False

            for _ in range(self.config.rl_reset_outer_trials):
                start_xy = None
                goal_xy = None

                for _ in range(self.config.rl_reset_inner_trials):
                    start_x = float(rng.uniform(min_x + self.config.rl_spawn_margin_m, max_x - self.config.rl_spawn_margin_m))
                    start_y = float(rng.uniform(min_y + self.config.rl_spawn_margin_m, max_y - self.config.rl_spawn_margin_m))
                    if self._is_safe_location(start_x, start_y, time_s=0.0):
                        start_xy = (start_x, start_y)
                        break

                if start_xy is None:
                    continue

                for _ in range(self.config.rl_reset_inner_trials):
                    target_dist = float(rng.uniform(goal_min, goal_max))
                    angle = float(rng.uniform(0.0, 2 * math.pi))
                    goal_x = float(np.clip(
                        start_xy[0] + target_dist * math.cos(angle),
                        min_x + self.config.rl_goal_margin_m,
                        max_x - self.config.rl_goal_margin_m,
                    ))
                    goal_y = float(np.clip(
                        start_xy[1] + target_dist * math.sin(angle),
                        min_y + self.config.rl_goal_margin_m,
                        max_y - self.config.rl_goal_margin_m,
                    ))
                    if self._is_safe_location(goal_x, goal_y, time_s=0.0):
                        goal_xy = (goal_x, goal_y)
                        break

                if goal_xy is None:
                    continue

                start_z = self.estimator.get_altitude(*start_xy) + self.config.takeoff_altitude_agl

                self.current_pos = np.array([start_xy[0], start_xy[1], start_z], dtype=np.float64)
                self.current_time = 0.0
                self.energy_remaining = float(self.config.battery_capacity_j)
                self.current_step = 0

                self.global_astar_path = self.planner.search(start_xy, goal_xy, start_time_s=0.0)

                if not self.global_astar_path:
                    continue

                if len(self.global_astar_path) > max_teacher_len:
                    continue

                path_total_len = self._path_total_length(self.global_astar_path)

                if self.config.curriculum_stage == 1 and path_total_len > self.config.rl_stage1_path_len_max_m:
                    continue

                if self.config.curriculum_stage == 2 and path_total_len > self.config.rl_stage2_path_len_max_m:
                    continue
                
                if self.config.curriculum_stage >= 4 and path_total_len > self.config.rl_stage4_path_len_max_m:
                    continue

                if 10 <= len(self.global_astar_path) <= max_teacher_len:
                    valid_plan = True
                    chosen_start_xy = start_xy
                    chosen_goal_xy = goal_xy
                    chosen_start_z = start_z
                    break

            if valid_plan:
                break

        if chosen_start_xy is None or chosen_goal_xy is None or chosen_start_z is None:
            raise RuntimeError(
                f"reset failed after {self.config.rl_reset_max_attempts} attempts: "
                "unable to sample a valid start/goal pair and teacher path."
            )

        self.current_pos = np.array([chosen_start_xy[0], chosen_start_xy[1], chosen_start_z], dtype=np.float64)
        self.goal_pos = np.array(
            [chosen_goal_xy[0], chosen_goal_xy[1], self.estimator.get_altitude(*chosen_goal_xy) + self.config.takeoff_altitude_agl],
            dtype=np.float64,
        )
        self.current_heading = math.degrees(math.atan2(chosen_goal_xy[1] - chosen_start_xy[1], chosen_goal_xy[0] - chosen_start_xy[0]))
        self.current_wp_idx = 1 if len(self.global_astar_path) > 1 else 0
        self.current_time = 0.0
        self.energy_remaining = float(self.config.battery_capacity_j)
        self.current_step = 0
        self.prev_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - self.current_pos[:2]))
        self.goal_dist_50_steps_ago = self.prev_goal_dist
        self.telemetry_time_s = []
        self.telemetry_power_w = []
        self.telemetry_risk = []
        self.telemetry_max_p_crash = 0.0

        return self._get_obs(), {
            "is_success": False,
            "teacher_path_len": len(self.global_astar_path),
            "curriculum_stage": self.config.curriculum_stage,
            "episode_wind_seed": self.episode_wind_seed,
            "episode_nfz_count": len(self.episode_nfz_list_km),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        speed_min = self.config.rl_speed_min
        speed_max = self.config.rl_speed_max
        speed_mid = 0.5 * (speed_min + speed_max)
        speed_half_range = 0.5 * (speed_max - speed_min)

        delta_heading = float(action[0] * self.config.rl_heading_delta_max_deg)
        target_speed = float(speed_mid + action[1] * speed_half_range)
        delta_agl = float(action[2] * self.config.rl_agl_delta_max_m)

        info: Dict[str, Any] = {"is_success": False}
        terminated = False
        truncated = False
        reward = 0.0

        teacher = self._teacher_reference()
        target_speed = float(np.clip(target_speed, speed_min, speed_max))
        desired_heading = self._wrap_angle_deg(self.current_heading + delta_heading)

        current_ground_alt = self.estimator.get_altitude(self.current_pos[0], self.current_pos[1])
        current_agl = max(self.min_clearance_agl, self.current_pos[2] - current_ground_alt)
        desired_agl = float(np.clip(current_agl + delta_agl, self.min_clearance_agl, self.max_clearance_agl))

        apas_intervened = False
        if self.config.rl_enable_apas:
            transition, fatal_reason = self._simulate_apas_transition(desired_heading, target_speed, desired_agl)
            if transition is None:
                terminated = True
                info["terminated_reason"] = fatal_reason
                reward -= self.config.rl_collision_penalty if fatal_reason == "terrain_or_nfz" else self.config.rl_storm_penalty
                p_now, _ = self.estimator.get_risk(
                    self.current_pos[0],
                    self.current_pos[1],
                    self.current_pos[2],
                    max(self.config.drone_speed, 1.0),
                    self.current_time,
                )
                info.update(
                    {
                        "power_w": 0.0,
                        "p_crash": float(p_now),
                        "goal_dist_m": float(np.linalg.norm(self.goal_pos[:2] - self.current_pos[:2])),
                        "path_error_m": self._estimate_local_path_error(self.current_pos[:2]),
                        "energy_remaining_j": self.energy_remaining,
                        "heading_err_norm": abs(self._wrap_angle_deg(self.current_heading - teacher["heading_deg"])) / 180.0,
                        "speed_err_norm": abs(target_speed - teacher["speed_mps"]) / max(speed_max - speed_min, 1e-6),
                        "agl_err_norm": abs(desired_agl - teacher["agl_m"]) / max(self.max_clearance_agl - self.min_clearance_agl, 1.0),
                        "apas_intervened": True,
                    }
                )
                self.telemetry_time_s.append(self.current_time)
                self.telemetry_power_w.append(0.0)
                self.telemetry_risk.append(float(p_now))
                self.telemetry_max_p_crash = max(self.telemetry_max_p_crash, float(p_now))
                return self._get_obs(), float(reward), terminated, truncated, info
        else:
            transition = self._simulate_nominal_transition(desired_heading, target_speed, desired_agl)

        new_x = transition["new_x"]
        new_y = transition["new_y"]
        new_z = transition["new_z"]
        power = transition["power_w"]
        p_crash = transition["p_crash"]
        applied_speed = transition["speed_mps"]
        applied_heading = transition["heading_deg"]
        apas_intervened = self.config.rl_enable_apas and (
            abs(self._wrap_angle_deg(applied_heading - desired_heading)) > 1e-6
            or abs(applied_speed - target_speed) > 1e-6
        )
        new_pos = np.array([new_x, new_y, new_z], dtype=np.float64)

        energy_used = power * self.dt
        self.energy_remaining -= energy_used
        self.current_time += self.dt
        self.current_heading = applied_heading

        old_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - self.current_pos[:2]))
        new_goal_dist = float(np.linalg.norm(self.goal_pos[:2] - new_pos[:2]))

        target_wp = self._path_point(self.current_wp_idx)
        old_wp_dist = float(np.linalg.norm(target_wp[:2] - self.current_pos[:2]))
        new_wp_dist = float(np.linalg.norm(target_wp[:2] - new_pos[:2]))

        path_error = self._estimate_local_path_error(new_pos[:2])

        # Reward Shaping
        reward = 0.0
        reward += 0.004 * (old_goal_dist - new_goal_dist)
        reward += 0.002 * (old_wp_dist - new_wp_dist)
        reward -= 0.0005 * path_error
        reward -= 5.0 * (p_crash ** 2)
        reward -= 0.0001 * max(0.0, power - self.config.base_power)

        heading_err_norm = abs(self._wrap_angle_deg(self.current_heading - teacher["heading_deg"])) / 180.0
        speed_err_norm = abs(applied_speed - teacher["speed_mps"]) / max(speed_max - speed_min, 1e-6)
        agl_err_norm = abs(desired_agl - teacher["agl_m"]) / max(self.max_clearance_agl - self.min_clearance_agl, 1.0)

        reward -= 0.001 * heading_err_norm
        reward -= 0.002 * speed_err_norm
        reward -= 0.001 * agl_err_norm
        reward -= 0.002

        self.current_pos = new_pos
        self._refresh_wp_index(self.current_pos[:2])

        # 稀疏/终端奖励触发
        if new_wp_dist < self.config.rl_waypoint_reach_radius_m and self.current_wp_idx < len(self.global_astar_path) - 1:
            reward += self.config.rl_waypoint_reward

        dist_to_goal_3d = float(np.linalg.norm(self.goal_pos - self.current_pos))

        if dist_to_goal_3d < self.config.goal_tolerance_3d_m:
            reward += self.config.rl_goal_reward
            terminated = True
            info["is_success"] = True
            info["terminated_reason"] = "goal_reached"

        if not terminated and self.estimator.map.is_collision(new_x, new_y, new_z, nfz_list_km=self._current_nfz_list()):
            reward -= self.config.rl_collision_penalty
            terminated = True
            info["terminated_reason"] = "terrain_or_nfz"

        if not terminated and p_crash > self.config.rl_terminate_risk_threshold:
            reward -= self.config.rl_storm_penalty
            terminated = True
            info["terminated_reason"] = "storm_risk_too_high"

        if not terminated and power > self.config.max_power * self.config.rl_overload_power_ratio:
            reward -= self.config.rl_storm_penalty
            terminated = True
            info["terminated_reason"] = "overload"

        if not terminated and self.energy_remaining <= 0.0:
            reward -= self.config.rl_battery_penalty
            truncated = True
            info["terminated_reason"] = "battery_depleted"

        if (
            not terminated
            and self.current_step % self.config.rl_progress_check_interval == 0
            and self.current_step > 0
        ):
            progress = self.goal_dist_50_steps_ago - new_goal_dist

            if self.config.curriculum_stage == 1:
                required_progress = self.config.rl_required_progress_stage1_m
            elif self.config.curriculum_stage == 2:
                required_progress = self.config.rl_required_progress_stage2_m
            elif self.config.curriculum_stage == 3:
                required_progress = self.config.rl_required_progress_stage3_m
            else:
                required_progress = self.config.rl_required_progress_stage4_m

            if progress < required_progress:
                reward -= self.config.rl_no_progress_penalty
                truncated = True
                info["terminated_reason"] = "no_progress"

            self.goal_dist_50_steps_ago = new_goal_dist

        if not terminated and self.current_step >= self.max_steps:
            reward -= self.config.rl_timeout_penalty
            truncated = True
            info["terminated_reason"] = "timeout"

        info.update({
            "power_w": power,
            "p_crash": p_crash,
            "goal_dist_m": new_goal_dist,
            "path_error_m": path_error,
            "energy_remaining_j": self.energy_remaining,
            "heading_err_norm": heading_err_norm,
            "speed_err_norm": speed_err_norm,
            "agl_err_norm": agl_err_norm,
            "apas_intervened": apas_intervened,
        })
        self.telemetry_time_s.append(self.current_time)
        self.telemetry_power_w.append(power)
        self.telemetry_risk.append(p_crash)
        self.telemetry_max_p_crash = max(self.telemetry_max_p_crash, float(p_crash))
        self.prev_goal_dist = new_goal_dist

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        teacher = self._teacher_reference()
        target_wp = self._path_point(self.current_wp_idx)

        vec_to_goal = self.goal_pos[:2] - self.current_pos[:2]
        dist_to_goal = float(np.linalg.norm(vec_to_goal))
        angle_to_goal = math.degrees(math.atan2(vec_to_goal[1], vec_to_goal[0]))
        goal_angle_diff = self._wrap_angle_deg(angle_to_goal - self.current_heading)

        vec_to_wp = target_wp[:2] - self.current_pos[:2]
        dist_to_wp = float(np.linalg.norm(vec_to_wp))
        angle_to_wp = math.degrees(math.atan2(vec_to_wp[1], vec_to_wp[0]))
        wp_angle_diff = self._wrap_angle_deg(angle_to_wp - self.current_heading)

        energy_ratio = max(0.0, self.energy_remaining / max(self.config.battery_capacity_j, 1.0))
        current_agl = self._current_agl()
        
        current_wind = self.estimator.get_wind(self.current_pos[0], self.current_pos[1], self.current_pos[2], self.current_time)
        current_speed = self.config.drone_speed
        current_risk, _ = self.estimator.get_risk(self.current_pos[0], self.current_pos[1], self.current_pos[2], current_speed, self.current_time)

        future_time = self.current_time + 5.0
        future_wind = self.estimator.get_wind(self.current_pos[0], self.current_pos[1], self.current_pos[2], future_time)
        future_risk, _ = self.estimator.get_risk(self.current_pos[0], self.current_pos[1], self.current_pos[2], current_speed, future_time)
        
        wind_diff_u = future_wind[0] - current_wind[0]
        wind_diff_v = future_wind[1] - current_wind[1]
        risk_diff = future_risk - current_risk

        expert_heading_diff = self._wrap_angle_deg(teacher["heading_deg"] - self.current_heading)

        obs = [
            dist_to_goal / 5000.0,
            goal_angle_diff / 180.0,
            dist_to_wp / 1000.0,
            wp_angle_diff / 180.0,
            energy_ratio,
            np.clip(current_agl / self.max_clearance_agl, 0.0, 2.0),
            float(np.clip(current_wind[0] / self.config.max_wind_speed, -1.5, 1.5)),
            float(np.clip(current_wind[1] / self.config.max_wind_speed, -1.5, 1.5)),
            float(np.clip(current_risk, 0.0, 1.0)),
            expert_heading_diff / 180.0,
            teacher["speed_mps"] / max(self.config.rl_speed_max, 1.0),
            teacher["agl_m"] / self.max_clearance_agl,
            float(np.clip(wind_diff_u / 5.0, -1.0, 1.0)),
            float(np.clip(wind_diff_v / 5.0, -1.0, 1.0)),
            float(np.clip(risk_diff / 0.5, -1.0, 1.0)),
        ]

        for relative_angle_deg in self.relative_scan_angles_deg:
            rad = math.radians(self.current_heading + relative_angle_deg)
            scan_x = self.current_pos[0] + self.scan_dist * math.cos(rad)
            scan_y = self.current_pos[1] + self.scan_dist * math.sin(rad)

            min_x, max_x, min_y, max_y = self.estimator.get_bounds()
            if scan_x < min_x or scan_x > max_x or scan_y < min_y or scan_y > max_y:
                obs.extend([0.0, 1.0])
                continue

            scan_z = self.estimator.get_altitude(scan_x, scan_y) + current_agl
            wind = self.estimator.get_wind(scan_x, scan_y, scan_z, self.current_time)
            p_crash, _ = self.estimator.get_risk(scan_x, scan_y, scan_z, self.config.drone_speed, self.current_time)
            obs.extend([
                float(np.clip(np.linalg.norm(wind) / self.config.max_wind_speed, 0.0, 2.0)),
                float(np.clip(p_crash, 0.0, 1.0)),
            ])

        obs_array = np.asarray(obs, dtype=np.float32)
        if self.config.obs_ablation_mode == "no_future":
            obs_array[12:15] = 0.0
        elif self.config.obs_ablation_mode == "no_radar":
            obs_array[15:31] = 0.0
        return obs_array
