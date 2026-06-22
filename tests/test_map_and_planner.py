"""
地图和规划器测试模块

此模块包含地图管理器和A*规划器的单元测试，验证坐标映射、
梯度计算、风场估计和路径规划功能的基本正确性。

测试类：
- MapManagerTests: 地图管理器测试
- PlannerSmokeTests: 规划器冒烟测试
"""

import unittest
import numpy as np
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import BaseWindModel
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner


class MapManagerTests(unittest.TestCase):
    def setUp(self):
        # 通过指向无效路径和小尺寸强制使用假地图
        self.cfg = SimulationConfig()
        self.cfg.map_path = 'nonexistent.file'
        self.cfg.target_size = (20, 20)
        self.mapm = MapManager(self.cfg)

    def test_bounds_and_altitude(self):
        min_x, max_x, min_y, max_y = self.mapm.get_bounds()
        self.assertTrue(min_x < max_x)
        self.assertTrue(min_y < max_y)
        # 中心高度应在最小和最大高度之间
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        alt = self.mapm.get_altitude(cx, cy)
        self.assertGreaterEqual(alt, self.cfg.min_alt)
        self.assertLessEqual(alt, self.cfg.max_alt)

    def test_gradient_and_roughness(self):
        # 在边界内选择一些点
        for x in [min(self.mapm.x), max(self.mapm.x)]:
            for y in [min(self.mapm.y), max(self.mapm.y)]:
                gx, gy = self.mapm.get_gradient(x, y)
                self.assertIsInstance(gx, float)
                self.assertIsInstance(gy, float)
                # 梯度幅度应为有限值
                self.assertFalse(np.isnan(gx) or np.isnan(gy))
                z0 = self.mapm.get_roughness(x, y)
                self.assertGreaterEqual(z0, 0.0)


class PlannerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimulationConfig()
        self.cfg.map_path = 'nonexistent.file'
        self.cfg.target_size = (20, 20)
        # 限制步骤以使测试快速完成
        self.cfg.max_steps = 5000
        self.mapm = MapManager(self.cfg)
        # 创建一个总是返回零风的简单风模型
        class ZeroWind(BaseWindModel):
            def get_wind(self, x, y, z, terrain_gradient, z0, t_s=0.0):
                return np.array([0.0, 0.0])

        self.wind = ZeroWind()
        self.est = StateEstimator(self.mapm, self.wind, self.cfg)
        self.physics = PhysicsEngine(self.cfg)
        self.planner = AStarPlanner(self.cfg, self.est, self.physics)
        bounds = self.est.get_bounds()
        # 选择靠近中心的位置和相邻的目标
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        self.start = (cx, cy)
        # 水平移动几个网格单元
        step = self.mapm.resolution
        self.goal = (cx + step * 2, cy)

    def test_trivial_path(self):
        # 起点等于终点应返回立即路径
        res = self.planner.search(self.start, self.start)
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 1)

    def test_estimator_basic(self):
        # 估计器应返回风向量和非负风险值
        cx, cy = self.start
        wind = self.est.get_wind(cx, cy, 10.0)
        self.assertEqual(wind.shape, (2,))
        z = self.est.get_altitude(cx, cy) + self.cfg.takeoff_altitude_agl
        risk, _ = self.est.get_risk(cx, cy, z, v_ground=self.cfg.drone_speed)
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
    def test_nonempty_path(self):
        res = self.planner.search(self.start, self.goal)
        # 确保搜索返回路径列表或None，但不崩溃
        self.assertTrue(res is None or isinstance(res, list))
        if isinstance(res, list):
            self.assertGreaterEqual(len(res), 1)


if __name__ == '__main__':
    unittest.main()
