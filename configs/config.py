# -*- coding: utf-8 -*-
"""
无人机路径规划仿真配置模块 (黄金参数版)

此模块定义了仿真系统的所有配置参数，使用dataclass实现类型安全的配置管理。
专为大创项目4D TKE极值风险规避演示调优，所有参数都经过精心调校。

主要配置类别：
- 地图与环境参数：定义仿真世界的物理属性
- 起点/终点参数：任务起始位置设置
- 风场参数：环境风力模型配置
- 风暴参数：动态风险源配置
- 无人机参数：飞行器物理特性
- 电池参数：能源管理系统
- 规划参数：路径规划算法设置
- 仿真参数：时间和计算控制
- 可视化参数：输出和显示设置

作者：项目团队
版本：1.0.0
更新日期：2026-04-03
"""
from dataclasses import dataclass, field
from typing import Tuple, List
import logging
import os  # 用于动态获取项目根目录路径

logger = logging.getLogger(__name__)

# 动态获取项目根目录（确保在任何工作目录下都能找到资源文件）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class SimulationConfig:
    """
    无人机路径规划仿真系统配置类

    此类使用Python dataclass实现类型安全的配置管理，包含仿真系统的所有参数。
    参数分为多个类别，便于管理和调优。

    属性分类：

    1. 地图与环境参数 (map_path, map_size_km, etc.)
       - 定义仿真世界的地理和物理属性
       - 包括地形数据、高度范围、时间等

    2. 起点/终点参数 (start_offset_*, goal_offset_*)
       - 控制任务的起始和目标位置
       - 支持绝对偏移和相对比例设置

    3. 风场参数 (env_wind_*, k_slope, max_wind_speed)
       - 基础环境风力模型参数
       - 影响无人机运动和能耗计算

    4. 风暴参数 (enable_storms, storm_*, wind_seed)
       - 动态风险源配置
       - 实现4D时空风险规避的核心

    5. TKE极值风险参数 (drone_response_*, fatal_crash_*, tke_*)
       - 湍动能极值概率风险模型
       - 项目核心创新点

    6. 禁飞区参数 (enable_nfz, nfz_list_km)
       - 静态禁飞区域定义
       - 增强安全约束

    7. 无人机物理参数 (drone_speed, drone_mass, etc.)
       - 飞行器物理特性
       - 影响运动学和动力学计算

    8. 电池参数 (battery_*, power_*)
       - 能源管理系统配置
       - 限制任务持续时间

    9. 规划参数 (grid_resolution, safety_margin, etc.)
       - A*路径规划算法设置
       - 影响路径质量和计算效率

    10. 仿真参数 (dt, max_time, etc.)
        - 时间离散化和计算控制
        - 影响仿真精度和性能

    11. 可视化参数 (enable_*, save_*)
        - 输出和显示设置
        - 控制图表生成和保存

    注意：
    - 所有长度单位为米(m)，时间单位为秒(s)
    - 角度单位为度(°)或弧度(rad)，具体见参数名
    - 默认参数经过调优，适合演示和测试
    """
    # =========================
    # 🗺️ 地图与环境参数
    # =========================
    map_path: str = os.path.join(PROJECT_ROOT, 'Bernese_Oberland_46.6241_8.0413.png')  # 使用绝对路径，确保在任何工作目录下都能找到地图
    map_size_km: float = 17.28  # 地图实际尺寸，单位km
    min_alt: float = 563.0  # 最低海拔高度，单位m
    max_alt: float = 3985.4  # 最高海拔高度，单位m
    target_size: Tuple[int, int] = (300, 300)  # 地图缩放目标尺寸，像素
    time_of_day: str = 'Night'  # 时间段，影响可视化主题 
    
    # =========================
    # 🎯 起点/终点参数
    # =========================
    start_offset_x: float = 1000.0  # 起点相对于 min_x 的偏移 (米)
    start_offset_y: float = 1000.0  # 起点相对于 min_y 的偏移 (米)
    goal_offset_factor_x: float = 0.8  # 终点相对位置 X = min_x + (max_x - min_x) * 因子
    goal_offset_factor_y: float = 0.8  # 终点相对位置 Y = min_y + (max_y - min_y) * 因子
    
    # =========================
    # 💨 基础风场参数
    # =========================
    env_wind_u: float = -3.0
    env_wind_v: float = 5.0
    k_slope: float = 15.0       
    max_wind_speed: float = 30.0 

    # =========================
    # ⛈️ 动态移动风暴 (大创杀手锏)
    # =========================
    enable_storms: bool = True   # 🌟 必须开启，展示动态避障
    storm_count: int = 3
    storm_radius_range_m: Tuple[float, float] = (500.0, 1500.0) # 调大了风暴，视觉更震撼
    storm_max_speed_mps: float = 15.0
    storm_movement_speed_mps: float = 5.0 # 让风暴跑快点，动画更好看
    storm_lifetime_s: float = 2000.0
    storm_strength_scale: float = 1.5
    wind_seed: int = 2024

    # =========================
    # ⚠️ TKE 极值概率风险模型 (核心创新)
    # =========================
    drone_response_freq_N0: float = 3.0
    drone_robustness_K: float = 200.0
    # 🌟 稍微调低致命惩罚，允许无人机在绝境中进行“高风险突防”，防止无路可走
    fatal_crash_penalty_j: float = 80000.0 

    tke_shear_coeff: float = 0.2   
    tke_wake_coeff: float = 0.002  
    tke_slope_coeff: float = 0.05  

    # =========================
    # 🚫 静态禁飞区 (NFZ)
    # =========================
    enable_nfz: bool = True  # 🌟 重新开启禁飞区
    nfz_list_km: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (2.0, 2.0, 1.5),    
        (-3.0, 1.0, 1.2)    
    ])

    # =========================
    # 🚁 无人机物理参数 (黄金滑翔机配置)
    # =========================
    drone_speed: float = 10.0       # 稍微降速，让风速(15-30m/s)对它的影响变得极度致命
    drone_mass: float = 9.0      
    air_density: float = 1.225      
    frontal_area: float = 0.25    
    
    # 🌟【关键修改 1】：时间税降为极低。让它不再为了“赶时间”走直线，愿意花时间去绕远路找顺风！
    base_power: float = 800.0       
    # 🌟【关键修改 2】：风阻系数拉爆！现在只要敢迎风飞，耗电量会呈指数级爆炸！
    drag_coeff: float = 0.15      
    # 🌟【关键修改 3】：封印马力上限。遇到大逆风，它直接算出来 power > max_power，算法会强迫它变道！
    max_power: float = 4000.0      

    min_speed: float = 3.0          
    speed_decrement: float = 2.0    
    
    # =========================
    # 🧠 4D 规划器参数
    # =========================
    k_wind: float = 1.0             
    risk_factor: float = 20.0       
    max_steps: int = 100000   

    noise_level: float = 0.0
    z_step: float = 50.0       
    max_ceiling: float = 5000.0 
    
    # 🌟 恢复垂直权重：体现爬山的艰难
    z_weight: float = 1.2   

    # 🚀 救命参数：加权 A* (Weighted A*) 核心因子
    # 从 1.0 提高到 3.5。强迫 A* 算法变得贪婪，像导弹一样直奔终点，极大提升长距离搜索速度！
    heuristic_safety_factor: float = 2

    gravity: float = 9.81
    takeoff_altitude_agl: float = 50.0

    # =========================
    # 🔋 动态任务与电池设置
    # =========================
    mission_update_interval_s: float = 150.0
    max_replans: int = 100
    max_mission_time_s: float = 3600.0
    cruise_speed_mps: float = 15.0

    battery_capacity_j: float = 2500000.0 # 稍微增加点电池，应对长途跋涉
    reserve_energy_ratio: float = 0.15

    # =========================
    # 🎯 目标检查与时变风场
    # =========================
    goal_check_mode: str = "3d_distance"
    goal_tolerance_3d_m: float = 80.0 # 放宽到达判定，接触即算成功
    goal_tolerance_xy_m: float = 40.0
    goal_tolerance_z_m: float = 20.0

    wind_model_type: str = "slope"
    wind_time_model_type: str = "smooth_periodic"
    wind_time_scale_s: float = 300.0
    wind_direction_variation_deg: float = 20.0
    wind_speed_variation_ratio: float = 0.25

    # =========================
    # 🌪️ 执行期随机扰动
    # =========================
    enable_random_gusts: bool = False
    gust_trigger_prob: float = 0.05
    gust_duration_s: float = 10.0
    gust_min_speed_mps: float = 6.0
    gust_max_speed_mps: float = 12.0
    gust_obs_noise_std: float = 0.0
    enable_single_agent_gusts: bool = False

    # =========================
    # 🛡️ Support 主动护盾模式
    # =========================
    enable_support_shield_mode: bool = True
    support_shield_master_radius_m: float = 1200.0
    support_shield_offset_m: float = 400.0
    swarm_nfz_inflation_m: float = 80.0

    obs_ablation_mode: str = "full"
    planner_time_mode: str = "4d"
    swarm_topology_mode: str = "full"
    collect_ablation_telemetry: bool = False
    disable_periodic_replan: bool = False
    frozen_reference_time_s: float = 0.0
    warning_distance_default_m: float = -1.0
    ablation_eval_episodes: int = 50

    planner_verbose: bool = False
    curriculum_stage: int = 1

     # =========================
    # 🤖 RL 强化学习专属参数
    # =========================
    rl_max_steps: int = 600
    rl_dt: float = 2.0

    rl_speed_min: float = 5.0
    rl_speed_max: float = 20.0
    rl_heading_delta_max_deg: float = 25.0
    rl_agl_delta_max_m: float = 15.0

    rl_min_clearance_agl: float = 35.0
    rl_max_clearance_agl: float = 250.0

    rl_waypoint_reach_radius_m: float = 80.0
    rl_waypoint_refresh_radius_m: float = 85.0
    rl_scan_distance_m: float = 220.0

    rl_goal_min_stage1_m: float = 1200.0
    rl_goal_max_stage1_m: float = 2200.0
    rl_goal_min_stage2_m: float = 2400.0
    rl_goal_max_stage2_m: float = 3400.0
    rl_goal_min_stage3_m: float = 3000.0
    rl_goal_max_stage3_m: float = 5000.0
    rl_goal_min_stage4_m: float = 5000.0
    rl_goal_max_stage4_m: float = 8500.0

    rl_nfz_count_stage1_min: int = 0
    rl_nfz_count_stage1_max: int = 2
    rl_nfz_count_stage2_min: int = 1
    rl_nfz_count_stage2_max: int = 3
    rl_nfz_count_stage3_min: int = 1
    rl_nfz_count_stage3_max: int = 4
    rl_nfz_count_stage4_min: int = 2
    rl_nfz_count_stage4_max: int = 6

    rl_teacher_len_stage1_max: int = 35
    rl_teacher_len_stage2_max: int = 60
    rl_teacher_len_stage3_max: int = 200
    rl_teacher_len_stage4_max: int = 320

    rl_stage1_path_len_max_m: float = 2600.0
    rl_stage2_path_len_max_m: float = 4200.0
    rl_stage4_path_len_max_m: float = 11000.0
    rl_max_steps_stage4: int = 900

    rl_spawn_margin_m: float = 1000.0
    rl_goal_margin_m: float = 500.0
    rl_nfz_spawn_margin_m: float = 2000.0
    rl_nfz_radius_min_km: float = 0.8
    rl_nfz_radius_max_km: float = 2.0

    rl_safe_spawn_risk_threshold: float = 0.12

    rl_reset_outer_trials: int = 30
    rl_reset_inner_trials: int = 100
    rl_reset_max_attempts: int = 8

    rl_waypoint_reward: float = 0.5
    rl_goal_reward: float = 25.0
    rl_collision_penalty: float = 15.0
    rl_storm_penalty: float = 15.0
    rl_battery_penalty: float = 8.03
    rl_timeout_penalty: float = 5.0
    rl_no_progress_penalty: float = 3.0

    rl_required_progress_stage1_m: float = 15.0
    rl_required_progress_stage2_m: float = 35.0
    rl_required_progress_stage3_m: float = 70.0
    rl_required_progress_stage4_m: float = 105.0

    rl_progress_check_interval: int = 75
    rl_overload_power_ratio: float = 1.05
    rl_terminate_risk_threshold: float = 0.55
    rl_enable_apas: bool = False
    v25_apas_use_disruption: bool = True
    v25_apas_segment_check_enabled: bool = True
    v25_apas_segment_probe_samples: int = 32
    v25_apas_segment_risk_threshold: float = 0.55
    v25_apas_heading_offsets_deg: Tuple[float, ...] = (
        0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 90.0, -90.0
    )
    v25_apas_speed_decrement_mps: float = 2.0
    v25_apas_agl_increments_m: Tuple[float, ...] = (0.0, 30.0, 60.0)
    v25_expert_heading_actions: Tuple[float, ...] = (-0.65, -0.35, 0.0, 0.35, 0.65)
    v25_expert_speed_actions: Tuple[float, ...] = (-0.25, 0.0, 0.35)
    v25_expert_agl_actions: Tuple[float, ...] = (0.0, 0.45)
    v25_expert_rollout_horizon: int = 3
    v25_expert_rollout_decay: float = 0.72
    v25_expert_activation_hazard: float = 0.18
    v25_expert_risk_gain: float = 12.0
    v25_expert_core_penalty: float = 90.0
    v25_expert_hard_constraint_penalty: float = 1000000.0
    v25_expert_progress_gain: float = 0.085
    v25_expert_path_error_gain: float = 0.0035
    v25_expert_final_path_error_gain: float = 0.008
    v25_expert_final_progress_gain: float = 0.10
    v25_expert_power_gain: float = 0.00015
    v25_expert_action_gain: float = 0.060
    v25_segment_probe_samples: int = 16
    v25_expert_hard_risk_threshold: float = 0.55
    v25_expert_risk_improvement_threshold: float = 0.08
    v25_expert_recovery_path_error_m: float = 650.0
    v25_local_hazard_history_steps: int = 5
    v25_local_hazard_trend_threshold: float = 0.02
    v25_expert_trend_warning_need: float = 0.10
    v25_expert_trend_forward_delta: float = 0.015
    v25_expert_trend_risk_improvement_threshold: float = 0.03
    v25_expert_mild_heading_actions: Tuple[float, ...] = (-0.35, 0.0, 0.35)
    v25_expert_mild_speed_actions: Tuple[float, ...] = (-0.25, 0.0)
    v25_expert_mild_agl_actions: Tuple[float, ...] = (0.0, 0.25)
    v25_expert_rejoin_lookahead_wps: int = 4
    v25_expert_rejoin_risk_low_threshold: float = 0.14
    v25_expert_rejoin_max_avoiding_steps: int = 30
    v25_expert_rejoin_from_cautious: bool = False
    v25_expert_rejoin_from_low_progress: bool = False
    v25_expert_rejoin_max_cautious_steps: int = 45
    v25_expert_rejoin_max_low_progress_steps: int = 18
    v25_expert_rejoin_min_progress_m: float = 3.0
    v25_expert_rejoin_heading_action_limit: float = 0.35
    v25_expert_rejoin_speed_action: float = 0.15
    v25_expert_emergency_heading_actions: Tuple[float, ...] = (-1.0, -0.75, 0.75, 1.0)
    v25_expert_emergency_speed_actions: Tuple[float, ...] = (-1.0, -0.75, -0.50)
    v25_expert_emergency_agl_actions: Tuple[float, ...] = (0.0, 1.0)
    v25_risk_membrane_enabled: bool = True
    v25_risk_membrane_angle_min_deg: float = -90.0
    v25_risk_membrane_angle_max_deg: float = 90.0
    v25_risk_membrane_angle_step_deg: float = 15.0
    v25_risk_membrane_radial_bins: int = 4
    v25_risk_membrane_lambda_theta_deg: float = 18.0
    v25_risk_membrane_lambda_r_m: float = 260.0
    v25_risk_membrane_block_threshold: float = 0.48
    v25_risk_membrane_wall_width_deg: float = 45.0
    v25_risk_membrane_min_gap_width_deg: float = 30.0
    v25_risk_membrane_front_window_deg: float = 45.0
    v25_expert_band_avoid_heading_limit: float = 0.90
    v25_expert_band_avoid_speed_action: float = -0.35
    v25_expert_pre_emergency_speed_action: float = -0.60
    v25_stale_waypoint_skip_enabled: bool = True
    v25_stale_waypoint_min_advance: int = 2
    v25_stale_waypoint_corridor_m: float = 180.0
    v25_replan_enabled: bool = False
    v25_replan_max_per_episode: int = 3
    v25_replan_cooldown_steps: int = 45
    v25_replan_min_step: int = 25
    v25_replan_path_error_m: float = 900.0
    v25_replan_risk_low_threshold: float = 0.18
    v25_replan_hard_risk_threshold: float = 0.55
    v25_replan_low_progress_steps: int = 60
    v25_replan_no_valid_linger_steps: int = 18
    v25_replan_no_valid_low_progress_steps: int = 10
    v25_replan_rejoin_lookahead_wps: int = 8
    v25_replan_min_path_points: int = 2
    # v2.5 disruptive-reward tuning: bias RL toward low energy and low time while keeping risk control.
    v25_reward_power_saving_gain: float = 0.00014
    v25_reward_risk_gain: float = 0.65
    v25_reward_progress_gain: float = 0.0035
    v25_reward_excess_power_penalty_gain: float = 0.00010
    v25_reward_slow_progress_penalty_gain: float = 0.0030
    v25_reward_step_time_penalty_gain: float = 0.0010
    v25_detour_penalty_threshold_m: float = 900.0
    v25_detour_penalty_gain: float = 0.0022

    # v2.5 disruption scaling: strengthen disturbances moderately without becoming unrealistic.
    v25_disruption_storm_strength_scale: float = 1.12
    v25_disruption_pulse_strength_scale: float = 1.10
    v25_disruption_storm_radius_scale: float = 1.05
    v25_disruption_pulse_radius_scale: float = 1.06
    v25_disruption_risk_bonus_scale: float = 0.30

    # v2.5 true-world execution and sensor model.
    v25_max_turn_rate_deg_s: float = 18.0
    v25_max_accel_mps2: float = 2.5
    v25_max_climb_rate_mps: float = 8.0
    v25_max_descent_rate_mps: float = 7.0
    v25_sensor_wind_noise_std_mps: float = 0.35
    v25_sensor_risk_noise_std: float = 0.01
    v25_sensor_scan_distance_m: float = 650.0
    v25_sensor_mode: str = "circle_oracle"
    v25_radar_radius_m: float = 1400.0
    v25_radar_sectors: int = 8
    v25_circle_oracle_samples: int = 96
    v25_disruption_stress_level: str = "normal"
    v25_local_wind_region_count_min: int = 2
    v25_local_wind_region_count_max: int = 4
    v25_local_wind_radius_range_m: Tuple[float, float] = (600.0, 1200.0)
    v25_local_wind_peak_range_mps: Tuple[float, float] = (4.0, 9.0)
    v25_local_wind_risk_bonus_max: float = 0.08
    v25_destructive_storm_radius_range_m: Tuple[float, float] = (450.0, 900.0)
    v25_destructive_storm_probability: float = 0.50
    v25_destructive_storm_core_ratio_range: Tuple[float, float] = (0.35, 0.50)
    v25_destructive_storm_halo_ratio_range: Tuple[float, float] = (0.68, 0.86)
    v25_destructive_storm_halo_danger_floor: float = 0.45
    v25_destructive_storm_wind_range_mps: Tuple[float, float] = (5.0, 11.0)
    v25_destructive_storm_outer_risk_range: Tuple[float, float] = (0.05, 0.20)
    v25_destructive_storm_core_risk_range: Tuple[float, float] = (0.60, 1.00)
    v25_destructive_storm_core_crash: bool = True
    v25_reward_progress_gain_true: float = 0.016
    v25_reward_forward_progress_gain_true: float = 0.006
    v25_reward_path_error_gain_true: float = 0.0004
    v25_reward_energy_gain_true: float = 0.00004
    v25_reward_risk_gain_true: float = 2.5
    v25_reward_residual_gain_true: float = 0.010
    v25_reward_calm_residual_gain_true: float = 0.080
    v25_reward_action_smoothness_gain_true: float = 0.025
    v25_reward_unproductive_residual_gain_true: float = 0.12
    v25_reward_unproductive_speed_gain_true: float = 0.035
    v25_reward_min_progress_m_true: float = 6.0
    v25_reward_apas_intervention_penalty_true: float = 0.08
    v25_reward_local_hazard_gain_true: float = 0.55
    v25_eval_maneuver_heading_energy_j: float = 180.0
    v25_eval_maneuver_speed_energy_j: float = 260.0
    v25_eval_maneuver_agl_energy_j: float = 220.0
    v25_eval_maneuver_action_delta_energy_j: float = 120.0
    v25_eval_apas_fixed_energy_j: float = 1200.0
    v25_eval_apas_heading_energy_j_per_deg: float = 18.0
    v25_eval_apas_speed_reduction_energy_j_per_mps2: float = 140.0
    v25_eval_apas_agl_energy_j_per_m: float = 12.0
    v25_eval_expert_cautious_burden: float = 0.2
    v25_eval_expert_avoiding_burden: float = 1.0
    v25_eval_expert_emergency_burden: float = 5.0
    v25_eval_expert_recovering_burden: float = 0.5
    v25_eval_apas_intervention_burden: float = 10.0
    v25_eval_apas_segment_rejection_burden: float = 0.2
    v25_eval_apas_no_valid_burden: float = 20.0
    v25_eval_burden_energy_equivalent_j: float = 800.0
    v25_goal_reward_true: float = 35.0
    v25_residual_gate_min_scale: float = 0.28
    v25_residual_gate_power: float = 1.1
    v25_intervention_wind_scale_mps: float = 6.0
    v25_intervention_tracking_scale_mps: float = 10.0
    v25_intervention_path_scale_m: float = 600.0
    v25_intervention_hazard_gain: float = 0.35
    v25_collision_penalty_true: float = 45.0
    v25_timeout_penalty_true: float = 25.0

    def __post_init__(self):
        """验证配置参数的合法性，防止非法值导致运行时错误"""
        # 速度验证
        if self.drone_speed <= 0 or self.drone_speed > 60:
            raise ValueError(f"drone_speed must be in (0, 60], got {self.drone_speed}")
        if self.min_speed < 0 or self.min_speed >= self.drone_speed:
            raise ValueError(f"min_speed must be in [0, drone_speed), got {self.min_speed}")
        
        # 电池/能量验证
        if self.battery_capacity_j <= 0:
            raise ValueError(f"battery_capacity_j must > 0, got {self.battery_capacity_j}")
        if not (0 < self.reserve_energy_ratio < 1):
            raise ValueError(f"reserve_energy_ratio must be in (0, 1), got {self.reserve_energy_ratio}")
        
        # 物理参数验证
        if self.drone_mass <= 0:
            raise ValueError(f"drone_mass must > 0, got {self.drone_mass}")
        if self.max_power <= 0:
            raise ValueError(f"max_power must > 0, got {self.max_power}")
        if self.drag_coeff < 0:
            raise ValueError(f"drag_coeff must >= 0, got {self.drag_coeff}")
        
        # 风暴参数验证
        if self.storm_count < 0:
            raise ValueError(f"storm_count cannot be negative, got {self.storm_count}")
        if len(self.storm_radius_range_m) != 2 or self.storm_radius_range_m[0] > self.storm_radius_range_m[1]:
            raise ValueError(f"storm_radius_range_m invalid: {self.storm_radius_range_m}")
        if self.storm_lifetime_s <= 0:
            raise ValueError(f"storm_lifetime_s must > 0, got {self.storm_lifetime_s}")

        # 扰动参数验证
        if not (0.0 <= self.gust_trigger_prob <= 1.0):
            raise ValueError(f"gust_trigger_prob must be in [0, 1], got {self.gust_trigger_prob}")
        if self.gust_duration_s < 0:
            raise ValueError(f"gust_duration_s must be >= 0, got {self.gust_duration_s}")
        if self.gust_min_speed_mps < 0 or self.gust_max_speed_mps < self.gust_min_speed_mps:
            raise ValueError(
                f"gust speed range invalid: min={self.gust_min_speed_mps}, max={self.gust_max_speed_mps}"
            )
        if self.gust_obs_noise_std < 0:
            raise ValueError(f"gust_obs_noise_std must be >= 0, got {self.gust_obs_noise_std}")
        if self.swarm_nfz_inflation_m < 0:
            raise ValueError(f"swarm_nfz_inflation_m must be >= 0, got {self.swarm_nfz_inflation_m}")
        if self.obs_ablation_mode not in {"full", "no_future", "no_radar"}:
            raise ValueError(f"invalid obs_ablation_mode: {self.obs_ablation_mode}")
        if self.planner_time_mode not in {"4d", "frozen_3d"}:
            raise ValueError(f"invalid planner_time_mode: {self.planner_time_mode}")
        if self.swarm_topology_mode not in {"full", "no_relay"}:
            raise ValueError(f"invalid swarm_topology_mode: {self.swarm_topology_mode}")
        if self.ablation_eval_episodes <= 0:
            raise ValueError(f"ablation_eval_episodes must be > 0, got {self.ablation_eval_episodes}")
        
        # 风险模型参数验证  
        if self.fatal_crash_penalty_j < 0:
            raise ValueError(f"fatal_crash_penalty_j cannot be negative, got {self.fatal_crash_penalty_j}")
        if self.risk_factor <= 0:
            raise ValueError(f"risk_factor must > 0, got {self.risk_factor}")
        
        # 地图参数验证
        if self.map_size_km <= 0:
            raise ValueError(f"map_size_km must > 0, got {self.map_size_km}")
        if self.min_alt >= self.max_alt:
            raise ValueError(f"min_alt must < max_alt, got {self.min_alt} vs {self.max_alt}")
        if self.max_ceiling <= 0:
            raise ValueError(f"max_ceiling must > 0, got {self.max_ceiling}")
        
        # 目标容差验证
        if self.goal_tolerance_3d_m < 0 or self.goal_tolerance_xy_m < 0 or self.goal_tolerance_z_m < 0:
            raise ValueError(f"goal tolerances cannot be negative")

        # v2.5 reward/disruption tuning validation
        if self.v25_reward_power_saving_gain < 0:
            raise ValueError(f"v25_reward_power_saving_gain must be >= 0, got {self.v25_reward_power_saving_gain}")
        if self.v25_reward_risk_gain < 0:
            raise ValueError(f"v25_reward_risk_gain must be >= 0, got {self.v25_reward_risk_gain}")
        if self.v25_reward_progress_gain < 0:
            raise ValueError(f"v25_reward_progress_gain must be >= 0, got {self.v25_reward_progress_gain}")
        if self.v25_reward_excess_power_penalty_gain < 0:
            raise ValueError(
                "v25_reward_excess_power_penalty_gain must be >= 0, "
                f"got {self.v25_reward_excess_power_penalty_gain}"
            )
        if self.v25_reward_slow_progress_penalty_gain < 0:
            raise ValueError(
                "v25_reward_slow_progress_penalty_gain must be >= 0, "
                f"got {self.v25_reward_slow_progress_penalty_gain}"
            )
        if self.v25_reward_step_time_penalty_gain < 0:
            raise ValueError(
                f"v25_reward_step_time_penalty_gain must be >= 0, got {self.v25_reward_step_time_penalty_gain}"
            )
        if self.v25_detour_penalty_threshold_m < 0:
            raise ValueError(
                f"v25_detour_penalty_threshold_m must be >= 0, got {self.v25_detour_penalty_threshold_m}"
            )
        if self.v25_detour_penalty_gain < 0:
            raise ValueError(f"v25_detour_penalty_gain must be >= 0, got {self.v25_detour_penalty_gain}")
        if self.v25_disruption_storm_strength_scale <= 0:
            raise ValueError(
                "v25_disruption_storm_strength_scale must be > 0, "
                f"got {self.v25_disruption_storm_strength_scale}"
            )
        if self.v25_disruption_pulse_strength_scale <= 0:
            raise ValueError(
                "v25_disruption_pulse_strength_scale must be > 0, "
                f"got {self.v25_disruption_pulse_strength_scale}"
            )
        if self.v25_disruption_storm_radius_scale <= 0:
            raise ValueError(
                f"v25_disruption_storm_radius_scale must be > 0, got {self.v25_disruption_storm_radius_scale}"
            )
        if self.v25_disruption_pulse_radius_scale <= 0:
            raise ValueError(
                f"v25_disruption_pulse_radius_scale must be > 0, got {self.v25_disruption_pulse_radius_scale}"
            )
        if self.v25_disruption_risk_bonus_scale < 0:
            raise ValueError(
                f"v25_disruption_risk_bonus_scale must be >= 0, got {self.v25_disruption_risk_bonus_scale}"
            )
        if self.v25_max_turn_rate_deg_s <= 0 or self.v25_max_accel_mps2 <= 0:
            raise ValueError("v25 turn-rate and acceleration limits must be > 0")
        if self.v25_max_climb_rate_mps <= 0 or self.v25_max_descent_rate_mps <= 0:
            raise ValueError("v25 vertical-rate limits must be > 0")
        if self.v25_sensor_wind_noise_std_mps < 0 or self.v25_sensor_risk_noise_std < 0:
            raise ValueError("v25 sensor noise values must be >= 0")
        if self.v25_sensor_scan_distance_m <= 0:
            raise ValueError("v25_sensor_scan_distance_m must be > 0")
        if self.v25_sensor_mode not in {"sector_radar", "circle_oracle"}:
            raise ValueError("v25_sensor_mode must be one of: sector_radar, circle_oracle")
        if self.v25_radar_radius_m <= 0 or self.v25_radar_sectors <= 0:
            raise ValueError("v25 radar radius and sectors must be > 0")
        if self.v25_circle_oracle_samples < 8:
            raise ValueError("v25_circle_oracle_samples must be >= 8")
        if self.v25_disruption_stress_level not in {"normal", "hard", "extreme", "fragile"}:
            raise ValueError("v25_disruption_stress_level must be one of: normal, hard, extreme, fragile")
        if self.v25_local_wind_region_count_min < 0:
            raise ValueError("v25 local wind region count cannot be negative")
        if self.v25_local_wind_region_count_max < self.v25_local_wind_region_count_min:
            raise ValueError("v25 local wind region count range is invalid")
        if not (0.0 <= self.v25_destructive_storm_probability <= 1.0):
            raise ValueError("v25_destructive_storm_probability must be in [0, 1]")
        if not (0.0 <= self.v25_destructive_storm_halo_danger_floor <= 1.0):
            raise ValueError("v25_destructive_storm_halo_danger_floor must be in [0, 1]")
        if min(
            self.v25_intervention_wind_scale_mps,
            self.v25_intervention_tracking_scale_mps,
            self.v25_intervention_path_scale_m,
        ) <= 0:
            raise ValueError("v25 intervention scales must be > 0")
        if min(
            self.v25_reward_residual_gain_true,
            self.v25_reward_forward_progress_gain_true,
            self.v25_reward_calm_residual_gain_true,
            self.v25_reward_action_smoothness_gain_true,
            self.v25_reward_unproductive_residual_gain_true,
            self.v25_reward_unproductive_speed_gain_true,
            self.v25_reward_min_progress_m_true,
            self.v25_reward_apas_intervention_penalty_true,
            self.v25_reward_local_hazard_gain_true,
            self.v25_goal_reward_true,
            self.v25_residual_gate_min_scale,
            self.v25_residual_gate_power,
            self.v25_intervention_hazard_gain,
            self.v25_collision_penalty_true,
            self.v25_timeout_penalty_true,
        ) < 0:
            raise ValueError("v25 residual reward and terminal penalties must be >= 0")
        if self.v25_intervention_hazard_gain > 1.0:
            raise ValueError("v25_intervention_hazard_gain must be in [0, 1]")
        if self.v25_residual_gate_min_scale > 1.0:
            raise ValueError("v25_residual_gate_min_scale must be in [0, 1]")
        if self.v25_apas_speed_decrement_mps <= 0:
            raise ValueError("v25_apas_speed_decrement_mps must be > 0")
        if self.v25_apas_segment_probe_samples < 1:
            raise ValueError("v25_apas_segment_probe_samples must be >= 1")
        if not (0.0 <= self.v25_apas_segment_risk_threshold <= 1.0):
            raise ValueError("v25_apas_segment_risk_threshold must be in [0, 1]")
        if not self.v25_apas_heading_offsets_deg or not self.v25_apas_agl_increments_m:
            raise ValueError("v25 APAS candidate lists must not be empty")
        if not self.v25_expert_heading_actions or not self.v25_expert_speed_actions or not self.v25_expert_agl_actions:
            raise ValueError("v25 expert candidate lists must not be empty")
        if self.v25_expert_rollout_horizon < 1:
            raise ValueError("v25_expert_rollout_horizon must be >= 1")
        if not (0.0 <= self.v25_expert_rollout_decay <= 1.0):
            raise ValueError("v25_expert_rollout_decay must be in [0, 1]")
        if self.v25_segment_probe_samples < 1:
            raise ValueError("v25_segment_probe_samples must be >= 1")
        if not (0.0 <= self.v25_expert_hard_risk_threshold <= 1.0):
            raise ValueError("v25_expert_hard_risk_threshold must be in [0, 1]")
        if not (0.0 <= self.v25_expert_risk_improvement_threshold <= 1.0):
            raise ValueError("v25_expert_risk_improvement_threshold must be in [0, 1]")
        if self.v25_expert_recovery_path_error_m < 0:
            raise ValueError("v25_expert_recovery_path_error_m must be >= 0")
        if self.v25_local_hazard_history_steps < 1:
            raise ValueError("v25_local_hazard_history_steps must be >= 1")
        if self.v25_local_hazard_trend_threshold < 0:
            raise ValueError("v25_local_hazard_trend_threshold must be >= 0")
        if not (0.0 <= self.v25_expert_trend_warning_need <= 1.0):
            raise ValueError("v25_expert_trend_warning_need must be in [0, 1]")
        if self.v25_expert_trend_forward_delta < 0:
            raise ValueError("v25_expert_trend_forward_delta must be >= 0")
        if not (0.0 <= self.v25_expert_trend_risk_improvement_threshold <= 1.0):
            raise ValueError("v25_expert_trend_risk_improvement_threshold must be in [0, 1]")
        if not self.v25_expert_mild_heading_actions or not self.v25_expert_mild_speed_actions or not self.v25_expert_mild_agl_actions:
            raise ValueError("v25 expert mild candidate lists must not be empty")
        if self.v25_expert_rejoin_lookahead_wps < 1:
            raise ValueError("v25_expert_rejoin_lookahead_wps must be >= 1")
        if not (0.0 <= self.v25_expert_rejoin_risk_low_threshold <= 1.0):
            raise ValueError("v25_expert_rejoin_risk_low_threshold must be in [0, 1]")
        if self.v25_expert_rejoin_max_avoiding_steps < 1:
            raise ValueError("v25_expert_rejoin_max_avoiding_steps must be >= 1")
        if self.v25_expert_rejoin_max_cautious_steps < 1:
            raise ValueError("v25_expert_rejoin_max_cautious_steps must be >= 1")
        if self.v25_expert_rejoin_max_low_progress_steps < 1:
            raise ValueError("v25_expert_rejoin_max_low_progress_steps must be >= 1")
        if self.v25_expert_rejoin_min_progress_m < 0:
            raise ValueError("v25_expert_rejoin_min_progress_m must be >= 0")
        if not (0.0 <= self.v25_expert_rejoin_heading_action_limit <= 1.0):
            raise ValueError("v25_expert_rejoin_heading_action_limit must be in [0, 1]")
        if not (-1.0 <= self.v25_expert_rejoin_speed_action <= 1.0):
            raise ValueError("v25_expert_rejoin_speed_action must be in [-1, 1]")
        if (
            not self.v25_expert_emergency_heading_actions
            or not self.v25_expert_emergency_speed_actions
            or not self.v25_expert_emergency_agl_actions
        ):
            raise ValueError("v25 expert emergency candidate lists must not be empty")
        if self.v25_risk_membrane_angle_step_deg <= 0:
            raise ValueError("v25_risk_membrane_angle_step_deg must be > 0")
        if self.v25_risk_membrane_angle_max_deg <= self.v25_risk_membrane_angle_min_deg:
            raise ValueError("v25 risk membrane angle range is invalid")
        if self.v25_risk_membrane_radial_bins < 1:
            raise ValueError("v25_risk_membrane_radial_bins must be >= 1")
        if self.v25_risk_membrane_lambda_theta_deg <= 0 or self.v25_risk_membrane_lambda_r_m <= 0:
            raise ValueError("v25 risk membrane decay constants must be > 0")
        if not (0.0 <= self.v25_risk_membrane_block_threshold <= 1.0):
            raise ValueError("v25_risk_membrane_block_threshold must be in [0, 1]")
        if self.v25_risk_membrane_wall_width_deg < 0 or self.v25_risk_membrane_min_gap_width_deg < 0:
            raise ValueError("v25 risk membrane width thresholds must be >= 0")
        if not (0.0 <= self.v25_expert_band_avoid_heading_limit <= 1.0):
            raise ValueError("v25_expert_band_avoid_heading_limit must be in [0, 1]")
        if not (-1.0 <= self.v25_expert_band_avoid_speed_action <= 1.0):
            raise ValueError("v25_expert_band_avoid_speed_action must be in [-1, 1]")
        if not (-1.0 <= self.v25_expert_pre_emergency_speed_action <= 1.0):
            raise ValueError("v25_expert_pre_emergency_speed_action must be in [-1, 1]")
        if self.v25_stale_waypoint_min_advance < 1:
            raise ValueError("v25_stale_waypoint_min_advance must be >= 1")
        if self.v25_stale_waypoint_corridor_m < 0:
            raise ValueError("v25_stale_waypoint_corridor_m must be >= 0")
        if self.v25_replan_max_per_episode < 0:
            raise ValueError("v25_replan_max_per_episode must be >= 0")
        if self.v25_replan_cooldown_steps < 0:
            raise ValueError("v25_replan_cooldown_steps must be >= 0")
        if self.v25_replan_min_step < 0:
            raise ValueError("v25_replan_min_step must be >= 0")
        if self.v25_replan_path_error_m < 0:
            raise ValueError("v25_replan_path_error_m must be >= 0")
        if not (0.0 <= self.v25_replan_risk_low_threshold <= 1.0):
            raise ValueError("v25_replan_risk_low_threshold must be in [0, 1]")
        if not (0.0 <= self.v25_replan_hard_risk_threshold <= 1.0):
            raise ValueError("v25_replan_hard_risk_threshold must be in [0, 1]")
        if self.v25_replan_low_progress_steps < 1:
            raise ValueError("v25_replan_low_progress_steps must be >= 1")
        if self.v25_replan_no_valid_linger_steps < 0:
            raise ValueError("v25_replan_no_valid_linger_steps must be >= 0")
        if self.v25_replan_no_valid_low_progress_steps < 1:
            raise ValueError("v25_replan_no_valid_low_progress_steps must be >= 1")
        if self.v25_replan_rejoin_lookahead_wps < 1:
            raise ValueError("v25_replan_rejoin_lookahead_wps must be >= 1")
        if self.v25_replan_min_path_points < 2:
            raise ValueError("v25_replan_min_path_points must be >= 2")
        if min(
            self.v25_expert_risk_gain,
            self.v25_expert_core_penalty,
            self.v25_expert_hard_constraint_penalty,
            self.v25_expert_progress_gain,
            self.v25_expert_path_error_gain,
            self.v25_expert_final_path_error_gain,
            self.v25_expert_final_progress_gain,
            self.v25_expert_power_gain,
            self.v25_expert_action_gain,
            self.v25_expert_activation_hazard,
        ) < 0:
            raise ValueError("v25 expert scoring gains must be >= 0")
        if min(
            self.v25_eval_maneuver_heading_energy_j,
            self.v25_eval_maneuver_speed_energy_j,
            self.v25_eval_maneuver_agl_energy_j,
            self.v25_eval_maneuver_action_delta_energy_j,
            self.v25_eval_apas_fixed_energy_j,
            self.v25_eval_apas_heading_energy_j_per_deg,
            self.v25_eval_apas_speed_reduction_energy_j_per_mps2,
            self.v25_eval_apas_agl_energy_j_per_m,
            self.v25_eval_expert_cautious_burden,
            self.v25_eval_expert_avoiding_burden,
            self.v25_eval_expert_emergency_burden,
            self.v25_eval_expert_recovering_burden,
            self.v25_eval_apas_intervention_burden,
            self.v25_eval_apas_segment_rejection_burden,
            self.v25_eval_apas_no_valid_burden,
            self.v25_eval_burden_energy_equivalent_j,
        ) < 0:
            raise ValueError("v25 evaluation cost weights must be >= 0")
        
        logger.debug(f"✓ Config validated: drone_speed={self.drone_speed}m/s, battery={self.battery_capacity_j/1e6:.1f}MJ")

   
