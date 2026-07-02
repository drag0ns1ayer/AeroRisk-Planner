"""
无人机路径规划器模块

此模块实现了基于A*算法的三维路径规划器，考虑能量消耗和风险成本。
支持静态规划和动态重新规划功能。

主要组件：
- Node: 搜索节点类
- AStarPlanner: A*规划器主类
"""

import heapq
import math
import numpy as np
from typing import List, Tuple, Optional
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from core.physics import PhysicsEngine


class Node:
    """
    4D 时空搜索节点
    包含空间坐标 (x,y,z) 与 预计到达时间 (time_s)
    """
    def __init__(self, x: float, y: float, z: float, g: float = 0.0, h: float = 0.0, time_s: float = 0.0, parent=None):
        self.x = x
        self.y = y
        self.z = z  # 绝对海拔（m）
        self.g = g  # 从起点到当前节点的实际累积代价 (J)
        self.h = h  # 到终点的启发式预估代价 (J)
        self.f = g + h
        self.time_s = time_s  # 🌟 4D 核心：记录预计到达该节点时的累积时间(秒)
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def get_pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class AStarPlanner:
    """
    4D 极值风险感知 A* 路径规划器 (4D Risk-Aware A* Planner)
    内部坐标计算均采用国际标准单位制 (m, s, J, W)
    """

    def __init__(self, config: SimulationConfig, estimator: StateEstimator, physics: PhysicsEngine):
        self.config = config
        self.estimator = estimator
        self.physics = physics

        # 水平步长（m）: 使用 MapManager 的分辨率（m/pixel）
        self.step_size = float(self.estimator.get_resolution())
        # 垂直步长（m）
        self.z_step = float(self.config.z_step)

        # 地图边界（x, y 单位：米）
        self.min_x, self.max_x, self.min_y, self.max_y = self.estimator.get_bounds()

    def heuristic(self, node_pos: Tuple[float, float, float], goal_pos: Tuple[float, float, float]) -> float:
        """
        返回启发值（单位：焦耳 或 米）
        🌟 包含不对称垂直权重：爬升耗电极大，下降相对轻松
        """
        dx = node_pos[0] - goal_pos[0]
        dy = node_pos[1] - goal_pos[1]
        dz = goal_pos[2] - node_pos[2]  # 目标Z - 当前Z

        dist_xy_m = math.hypot(dx, dy)
        
        # 智能垂直权重分配
        if dz > 0:
            # 目标比我高（需要爬升），施加配置中的高惩罚权重 (如 2.0)
            weighted_dz = dz * self.config.z_weight
        else:
            # 目标比我低（需要下降），垂直代价接近真实物理距离 (权重 1.0)
            weighted_dz = abs(dz) * 1.0

        weighted_dist_m = math.sqrt(dist_xy_m ** 2 + weighted_dz ** 2)

        # 如果不考虑风场，退化为纯几何最短路
        if self.config.k_wind == 0:
            return weighted_dist_m

        # 估算能量: 加权距离 * 理想平飞每米能耗（J/m） * 安全系数
        return weighted_dist_m * self.physics.energy_per_meter * self.config.heuristic_safety_factor

    # 替换 core/planner.py 中的 calculate_cost 第一部分
    def calculate_cost(self, current_node: Node, next_x: float, next_y: float, next_z: float) -> Tuple[float, float]:
        """
        计算期望代价与飞行时间：能耗代价 + 基于 TKE 概率的风险代价
        返回: (总期望代价 J, 该路段耗时 s)
        """
        
        # --- 1. 禁飞区 (NFZ) 与 地形碰撞检测 ---
        # 🌟 给 A* 规划加上 150 米的安全膨胀缓冲，防止它贴边走导致 RL 切弯坠机
        if self.estimator.map.is_collision(next_x, next_y, next_z, inflation_m=150.0):
            return float('inf'), 0.0

        dist_xy_m = math.hypot(next_x - current_node.x, next_y - current_node.y)
        dist_z_m = next_z - current_node.z
        total_dist_m = math.sqrt(dist_xy_m ** 2 + dist_z_m ** 2)

        if self.config.k_wind == 0:
            # 传统无风模式：代价等于物理距离，耗时靠设定的巡航速度算
            return total_dist_m, (total_dist_m / self.config.drone_speed)

        # --- 2. 🌟 4D 时间推演引擎 ---
        v_total = float(self.config.drone_speed)
        segment_time_s = total_dist_m / v_total
        
        # 预测未来：计算无人机真正到达前方网格时的“未来时间”
        if self.config.planner_time_mode == "4d":
            arrival_time_s = current_node.time_s + segment_time_s
        elif self.config.planner_time_mode == "frozen_3d":
            arrival_time_s = self.config.frozen_reference_time_s
        else:
            raise ValueError(f"Unsupported planner_time_mode: {self.config.planner_time_mode}")

        # 计算三维地速向量
        v_z = dist_z_m / segment_time_s
        v_xy = dist_xy_m / segment_time_s
        move_vec_xy = np.array([next_x - current_node.x, next_y - current_node.y], dtype=float)
        if np.linalg.norm(move_vec_xy) > 0:
            move_vec_xy = move_vec_xy / np.linalg.norm(move_vec_xy)
        v_ground_xy = move_vec_xy * v_xy
        v_ground_xyz = np.array([v_ground_xy[0], v_ground_xy[1], v_z])

        # 获取离地高度 AGL
        agl = next_z - self.estimator.get_altitude(next_x, next_y)
        if agl < 0: 
            return float('inf'), 0.0
        
        # --- 3. 物理耗能计算 (基于未来的风场) ---
        # 🌟 把预测的未来时间 (arrival_time_s) 传给气象模型！
        wind_vec_2d = self.estimator.get_wind(next_x, next_y, next_z, t_s=arrival_time_s)
        wind_vec_3d = np.array([wind_vec_2d[0], wind_vec_2d[1], 0.0])

        total_power = self.physics.estimate_power_from_vectors(v_ground_xyz, wind_vec_3d)
        if total_power > self.config.max_power:
            # 超出电机最大功率（逆风太强爬升不动），视为不可行
            return float('inf'), 0.0

        energy_joules = total_power * segment_time_s

        # --- 4. 极值概率风险模型 (Risk-Sensitive MDP) ---
        v_ground_mag = np.linalg.norm(v_ground_xyz)
        
        # 🌟 把预测的未来时间传给极值风险模型，评估未来那个时刻当地的坠机概率！
        p_crash, tke = self.estimator.get_risk(next_x, next_y, next_z, v_ground_mag, t_s=arrival_time_s)
        
        # 风险代价 = 发生致命事故的惩罚 * 发生概率
        risk_cost = self.config.fatal_crash_penalty_j * p_crash * self.config.k_wind

        # 总期望代价
        expected_total_cost = energy_joules + risk_cost

        # --- 可视化 AI 内部决策过程 ---
        # 仅在遇到高风险(P_crash>1%) 时偶尔打印，避免日志刷屏
        if p_crash > 0.01 and np.random.rand() < 0.02:
            print(f"\n[AI 4D预测] 预计在 T={arrival_time_s:.1f}s 抵达高危空域！坐标:({next_x/1000:.1f}km, {next_y/1000:.1f}km, 海拔{next_z:.0f}m)")
            print(f"微气象分析 : TKE = {tke:.2f} m^2/s^2 (受地形/尾流/切变综合影响)")
            print(f"极值风险模型 : 坠机概率 P_crash = {p_crash*100:.2f}%")
            print(f"MDP 代价转化 : 正常能耗 {energy_joules:.0f} J, 附加致命惩罚 {risk_cost:.0f} J")
            if risk_cost > energy_joules * 2:
                print(f"决策结论: 期望风险代价过高，4D A* 将主动变道绕行！")

        return expected_total_cost, segment_time_s

    def search(
        self, 
        start_pos: Tuple[float, float], 
        goal_pos: Tuple[float, float], 
        start_time_s: float = 0.0
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        执行 4D A* 搜索。
        start_time_s: 任务当前的绝对时间（用于动态重规划时接续风暴状态）
        """
        # 初始化起点/终点绝对海拔（m）
        start_z = self.estimator.get_altitude(start_pos[0], start_pos[1]) + self.config.takeoff_altitude_agl
        goal_z = self.estimator.get_altitude(goal_pos[0], goal_pos[1]) + self.config.takeoff_altitude_agl

        # 🌟 实例化起点，并打上初始时间戳
        start_node = Node(start_pos[0], start_pos[1], start_z, g=0.0, h=0.0, time_s=start_time_s)
        start_node.h = self.heuristic(start_node.get_pos(), (goal_pos[0], goal_pos[1], goal_z))

        open_list: List[Node] = []
        closed_set = set()
        heapq.heappush(open_list, start_node)

        steps = 0
        arrival_dist_xy = self.step_size
        arrival_dist_z = self.z_step

        # Keep planner logs ASCII-safe for Windows GBK consoles and test runners.
        print(
            f"4D search start | takeoff_time: T={start_time_s:.1f}s\n"
            f"   start: X={start_pos[0]:.1f}m, Y={start_pos[1]:.1f}m, Z={start_z:.1f}m\n"
            f"   goal: X={goal_pos[0]:.1f}m, Y={goal_pos[1]:.1f}m, Z={goal_z:.1f}m"
        )

        while open_list and steps < self.config.max_steps:
            steps += 1
            current_node = heapq.heappop(open_list)

            # 到达判定（米）
            d_xy = math.hypot(current_node.x - goal_pos[0], current_node.y - goal_pos[1])
            d_z = abs(current_node.z - goal_z)
            if d_xy < arrival_dist_xy and d_z < arrival_dist_z * 2:
                print(
                    f"4D search success | steps: {steps}, "
                    f"expected_cost: {current_node.g:.2f} J, "
                    f"eta: {current_node.time_s:.1f} s"
                )
                return self._reconstruct_path(current_node)

            # 三维网格索引，用于闭集判断（防止走回头路）
            grid_idx = (
                int(math.floor((current_node.x - self.min_x) / self.step_size)),
                int(math.floor((current_node.y - self.min_y) / self.step_size)),
                int(math.floor(current_node.z / self.z_step)),
            )
            if grid_idx in closed_set:
                continue
            closed_set.add(grid_idx)

            # 26 邻域扩展 (全空间 3D 探测)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        next_x = current_node.x + dx * self.step_size
                        next_y = current_node.y + dy * self.step_size
                        next_z = current_node.z + dz * self.z_step

                        # 地图边界硬核检查
                        if not (self.min_x <= next_x <= self.max_x and self.min_y <= next_y <= self.max_y):
                            continue

                        # 高度上下限检查
                        terrain_alt = self.estimator.get_altitude(next_x, next_y)
                        if next_z > terrain_alt + self.config.max_ceiling:
                            continue
                        if next_z < terrain_alt + 10.0:  # 必须离地 10m 以上，防止擦地撞击
                            continue

                        # 🌟 调用 4D 代价函数，同时接收期望代价与分段耗时
                        move_cost, segment_time_s = self.calculate_cost(current_node, next_x, next_y, next_z)
                        
                        if move_cost == float('inf'):
                            continue

                        # 🌟 状态更新：累加代价、累加时间、计算新启发值
                        new_g = current_node.g + move_cost
                        new_time_s = current_node.time_s + segment_time_s 
                        new_h = self.heuristic((next_x, next_y, next_z), (goal_pos[0], goal_pos[1], goal_z))
                        
                        # 压入优先队列
                        heapq.heappush(open_list, Node(
                            x=next_x, y=next_y, z=next_z, 
                            g=new_g, h=new_h, 
                            time_s=new_time_s, 
                            parent=current_node
                        ))

        print("4D search failed: step budget exhausted; no safe path found.")
        return None

    def _reconstruct_path(self, node: Node) -> List[Tuple[float, float, float]]:
        """回溯节点链条，生成 3D 航点坐标列表"""
        path: List[Tuple[float, float, float]] = []
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]  # 反转列表，确保从起点到终点
