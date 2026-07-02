# --- START OF FILE drone_ui.py ---

import os
import sys
import time
import io
import json
from pathlib import Path

# 确保项目根目录在环境变量中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import cv2
import streamlit as st
import streamlit.components.v1 as components  # 🌟 新增：用于加载本地 3D HTML
try:
    from PIL import Image
except ImportError:
    Image = None

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from simulation.mission_executor import MissionExecutor
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.animation_builder import MissionAnimator
from utils.visualizer_core import Visualizer
from v25.disruptions import build_disruption_layer_v25
from v30.experiments.run_task_map_demo import build_real_astar_segment_executor
from v30.mission_map import ChargingStation, InspectionPoint, MissionMap
from v30.segment_executor import V25GuidedSegmentExecutor
from v30.task_executor import SimpleTaskExecutor
from v30.visualization import (
    generate_wind_trajectory_gif,
    plot_mission_elevation_profile,
    plot_mission_terrain_map,
    plot_wind_map,
)

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

RESULTS_ROOT = Path(project_root) / "results"
DEFAULT_MODEL_PATH = Path(project_root) / "v25" / "artifacts" / "models" / "ppo_v25_expert_bc_s200_ft50k_best" / "best_model.zip"


@st.cache_resource(show_spinner=False)
def load_rl_model_cached(model_path: str):
    if not PPO:
        return None
    model_file = Path(model_path)
    if not model_file.exists():
        return None
    return PPO.load(str(model_file), device="cpu")


def create_map_preview(map_manager: MapManager, start_xy, goal_xy, nfz_list_km=None):
    """创建 Plotly 交互式地图 2D 预览"""
    step = max(1, map_manager.size_x // 100)
    fig = go.Figure(
        data=go.Contour(
            z=map_manager.dem[::step, ::step],
            x=map_manager.x[::step],
            y=map_manager.y[::step],
            colorscale="Earth",
            contours_coloring="heatmap",
            colorbar=dict(title="海拔 (m)")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[start_xy[0]],
            y=[start_xy[1]],
            mode="markers+text",
            text=["起点"],
            textposition="top center",
            marker=dict(size=16, color="yellow", symbol="star", line=dict(width=2, color="black")),
            name="起点",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[goal_xy[0]],
            y=[goal_xy[1]],
            mode="markers+text",
            text=["终点"],
            textposition="top center",
            marker=dict(size=16, color="red", symbol="x", line=dict(width=2, color="black")),
            name="终点",
        )
    )
    _add_nfz_overlay(fig, nfz_list_km or [])
    x_min, x_max = float(map_manager.x[0]), float(map_manager.x[-1])
    y_min, y_max = float(map_manager.y[0]), float(map_manager.y[-1])
    fig.update_layout(
        height=500, 
        margin=dict(l=20, r=20, t=40, b=20), 
        title="当前配置任务地图预览",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max], scaleanchor="x", scaleratio=1), # 强制正方形比例
    )
    return fig


def _add_nfz_overlay(fig, nfz_list_km):
    """在 Plotly 图上叠加 NFZ 圆形边界（单位 km -> m）"""
    if not nfz_list_km:
        return

    for idx, (cx_km, cy_km, r_km) in enumerate(nfz_list_km):
        cx_m = float(cx_km) * 1000.0
        cy_m = float(cy_km) * 1000.0
        r_m = max(float(r_km), 0.0) * 1000.0
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=cx_m - r_m,
            x1=cx_m + r_m,
            y0=cy_m - r_m,
            y1=cy_m + r_m,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.08)",
        )
        fig.add_annotation(
            x=cx_m,
            y=cy_m,
            text=f"NFZ {idx + 1}",
            showarrow=False,
            font=dict(color="red", size=11),
        )


def create_terrain_preview(map_manager: MapManager, nfz_list_km=None):
    """生成纯地形预览图（不含起终点）"""
    step = max(1, map_manager.size_x // 100)
    fig = go.Figure(
        data=go.Contour(
            z=map_manager.dem[::step, ::step],
            x=map_manager.x[::step],
            y=map_manager.y[::step],
            colorscale="Earth",
            contours_coloring="heatmap",
            colorbar=dict(title="海拔 (m)"),
        )
    )
    _add_nfz_overlay(fig, nfz_list_km or [])
    x_min, x_max = float(map_manager.x[0]), float(map_manager.x[-1])
    y_min, y_max = float(map_manager.y[0]), float(map_manager.y[-1])
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=40, b=20),
        title="地形图预览",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max], scaleanchor="x", scaleratio=1),
    )
    return fig


def create_terrain_wind_preview(map_manager: MapManager, estimator: StateEstimator, config: SimulationConfig, nfz_list_km=None):
    """生成地形风场图（风速热力 + 稀疏风向箭头）"""
    sample_step = max(1, map_manager.size_x // 70)
    x_sub = map_manager.x[::sample_step]
    y_sub = map_manager.y[::sample_step]
    X, Y = np.meshgrid(x_sub, y_sub)

    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = float(X[i, j])
            y = float(Y[i, j])
            z = float(map_manager.get_altitude(x, y) + config.takeoff_altitude_agl)
            wind = estimator.get_wind(x, y, z, t_s=0.0)
            U[i, j] = float(wind[0])
            V[i, j] = float(wind[1])

    wind_speed = np.sqrt(U**2 + V**2)
    fig = go.Figure(
        data=go.Contour(
            z=wind_speed,
            x=x_sub,
            y=y_sub,
            colorscale="Turbo",
            contours_coloring="heatmap",
            colorbar=dict(title="风速 (m/s)"),
        )
    )

    # 风向箭头：适度增密 + 加长（固定长度，不按风速放大，避免拉爆坐标轴）
    x_min, x_max = float(map_manager.x[0]), float(map_manager.x[-1])
    y_min, y_max = float(map_manager.y[0]), float(map_manager.y[-1])
    arrow_stride = max(1, X.shape[0] // 14)
    arrow_length = max((x_max - x_min) / 24.0, map_manager.resolution * 6.0)
    min_speed_to_draw = 0.05
    xq = []
    yq = []
    uq = []
    vq = []
    for i in range(0, X.shape[0], arrow_stride):
        for j in range(0, X.shape[1], arrow_stride):
            x0 = float(X[i, j])
            y0 = float(Y[i, j])
            u = float(U[i, j])
            v = float(V[i, j])
            speed = float(np.hypot(u, v))
            if speed < min_speed_to_draw:
                continue
            ux = u / speed
            uy = v / speed
            xq.append(x0)
            yq.append(y0)
            uq.append(ux * arrow_length)
            vq.append(uy * arrow_length)

    if xq:
        quiver_fig = ff.create_quiver(
            xq,
            yq,
            uq,
            vq,
            scale=1.0,
            arrow_scale=0.32,
            angle=np.pi / 8.5,
            line=dict(color="white", width=1.15),
            name="wind_vectors",
        )
        for tr in quiver_fig.data:
            tr.opacity = 0.86
            tr.hoverinfo = "skip"
            tr.showlegend = False
            fig.add_trace(tr)

    _add_nfz_overlay(fig, nfz_list_km or [])
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=40, b=20),
        title="地形风场图 (t=0)",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max], scaleanchor="x", scaleratio=1),
    )
    return fig


def build_environment_config(env_state: dict) -> SimulationConfig:
    """仅根据地形/地图/风场参数构建环境配置"""
    config = SimulationConfig()
    config.enable_storms = True

    config.map_path = env_state["map_path"]
    config.map_size_km = env_state["map_size_km"]
    config.min_alt = env_state["min_alt"]
    config.max_alt = env_state["max_alt"]
    config.target_size = (env_state["map_resolution"], env_state["map_resolution"])

    config.time_of_day = env_state["time_of_day"]
    config.env_wind_u = env_state["env_wind_u"]
    config.env_wind_v = env_state["env_wind_v"]
    config.enable_nfz = env_state["enable_nfz"]
    config.nfz_list_km = [tuple(zone) for zone in env_state.get("nfz_list_km", config.nfz_list_km)]
    config.wind_seed = env_state["wind_seed"]
    config.storm_count = env_state["storm_count"]
    return config


def build_environment_preview(env_state: dict):
    """构建环境预览对象：config, map_manager, estimator"""
    config = build_environment_config(env_state)
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    return config, map_manager, estimator


def load_image_preview_for_streamlit(image_path: str):
    """读取任意位深的图片并转换为 Streamlit 可稳定显示的 uint8 预览图。"""
    img = None
    try:
        raw = np.fromfile(image_path, dtype=np.uint8)
        if raw.size > 0:
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype != np.uint8:
        arr = img.astype(np.float32)
        min_v = float(np.nanmin(arr))
        max_v = float(np.nanmax(arr))
        if max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v)
            arr = arr * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        img = np.clip(arr, 0, 255).astype(np.uint8)

    return img


def _to_gray_u8(arr: np.ndarray) -> np.ndarray:
    """将任意位深/通道图片转为灰度 uint8。"""
    img = arr
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.astype(np.float32).mean(axis=2)
    elif img.ndim != 2:
        img = np.squeeze(img)
        if img.ndim != 2:
            raise ValueError(f"不支持的图片维度: {img.shape}")

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        min_v = float(np.nanmin(img))
        max_v = float(np.nanmax(img))
        if max_v > min_v:
            img = (img - min_v) / (max_v - min_v)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def save_uploaded_map_as_stable_png(uploaded_file, out_path: str) -> tuple[bool, str]:
    """
    将上传图片标准化并保存为稳定可读的灰度 PNG。
    这样 MapManager 总是读取同一种格式，避免因源文件位深/编码差异回退虚拟地形。
    """
    try:
        data = uploaded_file.getvalue()
        if not data:
            return False, "上传文件为空。"

        img = None
        try:
            raw = np.frombuffer(data, dtype=np.uint8)
            if raw.size > 0:
                img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None

        if img is None and Image is not None:
            try:
                with Image.open(io.BytesIO(data)) as pil_img:
                    img = np.array(pil_img)
            except Exception:
                img = None

        if img is None:
            return False, "图片解码失败（cv2/PIL 均无法读取）。"

        gray_u8 = _to_gray_u8(img)
        ok, encoded = cv2.imencode(".png", gray_u8)
        if not ok:
            return False, "PNG 编码失败。"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(encoded.tobytes())
        return True, ""
    except Exception as exc:
        return False, str(exc)


def save_3d_interactive_html(map_manager: MapManager, mission_result, save_path: Path):
    """🌟 核心新增：生成并在本地保存真 3D 可交互地形与航迹图 (HTML格式)"""
    path_xyz = np.array(mission_result.actual_flown_path_xyz)
    if len(path_xyz) < 2:
        return

    # 降采样地形以保证浏览器流畅度 (网格控制在 60x60 左右)
    step = max(1, map_manager.size_x // 60)
    X, Y = np.meshgrid(map_manager.x[::step], map_manager.y[::step])
    Z = map_manager.dem[::step, ::step]

    fig = go.Figure()

    # 1. 绘制 3D 地形曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Earth', opacity=0.85, showscale=False, name='地形'
    ))

    # 2. 绘制 3D 飞行轨迹
    fig.add_trace(go.Scatter3d(
        x=path_xyz[:, 0], y=path_xyz[:, 1], z=path_xyz[:, 2],
        mode='lines', line=dict(color='#00FF00', width=6), name='实际飞行 3D 航迹'
    ))

    # 3. 绘制起终点
    fig.add_trace(go.Scatter3d(
        x=[path_xyz[0, 0]], y=[path_xyz[0, 1]], z=[path_xyz[0, 2]],
        mode='markers', marker=dict(size=8, color='yellow', line=dict(width=2, color='black')), name='起点'
    ))
    fig.add_trace(go.Scatter3d(
        x=[path_xyz[-1, 0]], y=[path_xyz[-1, 1]], z=[path_xyz[-1, 2]],
        mode='markers', marker=dict(size=8, color='red', line=dict(width=2, color='black')), name='终点/坠机点'
    ))

    # 4. 配置场景比例
    fig.update_layout(
        title="真 3D 航迹数字孪生 (鼠标左键拖拽旋转、滚轮缩放)",
        scene=dict(
            xaxis_title='X 坐标 (m)',
            yaxis_title='Y 坐标 (m)',
            zaxis_title='海拔高度 (m)',
            aspectmode='data'  # 强制 3D 比例对应真实物理尺寸
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # 写入到 HTML 文件
    fig.write_html(str(save_path), include_plotlyjs="cdn")


def apply_ui_config(ui_state: dict, disturbance_enabled: bool = False) -> SimulationConfig:
    """将 UI 的状态全面注入到底层 config 中"""
    config = SimulationConfig()
    config.curriculum_stage = 3
    config.enable_storms = True

    # 1. 地图与环境
    config.map_path = ui_state["map_path"]
    config.map_size_km = ui_state["map_size_km"]
    config.min_alt = ui_state["min_alt"]
    config.max_alt = ui_state["max_alt"]
    config.target_size = (ui_state["map_resolution"], ui_state["map_resolution"])
    config.time_of_day = ui_state["time_of_day"]
    config.env_wind_u = ui_state["env_wind_u"]
    config.env_wind_v = ui_state["env_wind_v"]
    config.enable_nfz = ui_state["enable_nfz"]
    config.nfz_list_km = [tuple(zone) for zone in ui_state.get("nfz_list_km", config.nfz_list_km)]

    # 2. 风暴与阵风
    config.wind_seed = ui_state["wind_seed"]
    config.storm_count = ui_state["storm_count"]
    config.enable_random_gusts = disturbance_enabled
    config.gust_trigger_prob = ui_state["gust_trigger_prob"]
    config.gust_duration_s = ui_state["gust_duration_s"]
    config.gust_min_speed_mps = ui_state["gust_min_speed_mps"]
    config.gust_max_speed_mps = ui_state["gust_max_speed_mps"]

    # 3. 物理与 A*
    config.cruise_speed_mps = ui_state["cruise_speed_mps"]
    config.drone_speed = ui_state["cruise_speed_mps"]
    config.max_power = ui_state["max_power"]
    config.heuristic_safety_factor = ui_state["heuristic_safety_factor"]
    config.max_replans = ui_state["max_replans"]

    # 4. 集群与护盾
    config.enable_support_shield_mode = ui_state["enable_support_shield_mode"]
    config.support_shield_master_radius_m = ui_state["support_shield_master_radius_m"]
    config.support_shield_offset_m = ui_state["support_shield_offset_m"]

    # 5. RL 参数
    config.gust_obs_noise_std = ui_state["gust_obs_noise_std"]

    return config


def create_v30_mission_preview(map_manager: MapManager, mission_map: MissionMap, nfz_list_km=None):
    fig = create_terrain_preview(map_manager, nfz_list_km=nfz_list_km)
    fig.add_trace(go.Scatter(
        x=[mission_map.start_xy[0]],
        y=[mission_map.start_xy[1]],
        mode="markers+text",
        text=["Start"],
        textposition="top center",
        marker=dict(size=16, color="yellow", symbol="star", line=dict(width=2, color="black")),
        name="Start",
    ))
    if mission_map.home_xy is not None:
        fig.add_trace(go.Scatter(
            x=[mission_map.home_xy[0]],
            y=[mission_map.home_xy[1]],
            mode="markers+text",
            text=["Home"],
            textposition="bottom center",
            marker=dict(size=13, color="white", symbol="hexagon", line=dict(width=2, color="black")),
            name="Home",
        ))
    if mission_map.inspection_points:
        fig.add_trace(go.Scatter(
            x=[p.xy[0] for p in mission_map.inspection_points],
            y=[p.xy[1] for p in mission_map.inspection_points],
            mode="markers+text",
            text=[p.id for p in mission_map.inspection_points],
            textposition="top center",
            marker=dict(size=12, color="orange", symbol="circle", line=dict(width=1, color="black")),
            name="Inspection",
        ))
    if mission_map.charging_stations:
        fig.add_trace(go.Scatter(
            x=[c.xy[0] for c in mission_map.charging_stations],
            y=[c.xy[1] for c in mission_map.charging_stations],
            mode="markers+text",
            text=[c.id for c in mission_map.charging_stations],
            textposition="bottom center",
            marker=dict(size=13, color="dodgerblue", symbol="cross", line=dict(width=1, color="black")),
            name="Charging",
        ))
    fig.update_layout(title="V3.0 地图巡检任务标定预览")
    return fig


def _float_cell(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return float(default)
    if isinstance(value, str) and not value.strip():
        return float(default)
    return float(value)


def _optional_float_cell(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def build_v30_mission_map_from_ui(start_xy, home_xy, inspections_df, chargers_df) -> MissionMap:
    inspection_points = []
    for _, row in inspections_df.iterrows():
        if not bool(row.get("enabled", True)):
            continue
        point_id = str(row.get("id", "")).strip()
        if not point_id:
            continue
        inspection_points.append(InspectionPoint(
            id=point_id,
            xy=(_float_cell(row.get("x_m", 0.0)), _float_cell(row.get("y_m", 0.0))),
            priority=_float_cell(row.get("priority", 1.0), 1.0),
            service_time_s=_float_cell(row.get("service_time_s", 30.0), 30.0),
            risk_value=_float_cell(row.get("risk_value", 0.0), 0.0),
            deadline_s=_optional_float_cell(row.get("deadline_s", None)),
        ))

    charging_stations = []
    for _, row in chargers_df.iterrows():
        if not bool(row.get("available", True)):
            continue
        station_id = str(row.get("id", "")).strip()
        if not station_id:
            continue
        charging_stations.append(ChargingStation(
            id=station_id,
            xy=(_float_cell(row.get("x_m", 0.0)), _float_cell(row.get("y_m", 0.0))),
            charge_rate_j_per_s=_float_cell(row.get("charge_rate_j_per_s", 4000.0), 4000.0),
            docking_time_s=_float_cell(row.get("docking_time_s", 25.0), 25.0),
            target_soc=_float_cell(row.get("target_soc", 0.95), 0.95),
            available=True,
        ))

    return MissionMap(
        name="ui_v30_inspection_mission",
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        home_xy=(float(home_xy[0]), float(home_xy[1])),
        inspection_points=inspection_points,
        charging_stations=charging_stations,
    )


def _v30_route_end_xy(mission_map: MissionMap):
    if mission_map.inspection_points:
        return mission_map.inspection_points[-1].xy
    if mission_map.home_xy is not None:
        return mission_map.home_xy
    return mission_map.start_xy


def run_v30_inspection_case(
    ui_state: dict,
    mission_map: MissionMap,
    control_mode: str,
    rl_model,
    output_dir: Path,
):
    config = apply_ui_config(ui_state, disturbance_enabled=False)
    config.max_replans = int(ui_state.get("v30_max_replans", config.max_replans))
    config.max_mission_time_s = float(ui_state.get("v30_max_mission_time_s", config.max_mission_time_s))
    config.mission_update_interval_s = float(ui_state.get("v30_update_interval_s", config.mission_update_interval_s))
    config.v25_stress_level = str(ui_state.get("v30_stress", "fragile"))
    config.rl_enable_apas = bool(ui_state.get("v30_enable_apas", True))

    base_segment_executor, _ = build_real_astar_segment_executor(config)
    estimator = base_segment_executor.estimator
    segment_executor = base_segment_executor
    if control_mode.startswith("v25_"):
        v25_mode = control_mode.replace("v25_", "")
        model = rl_model if v25_mode == "rl" else None
        if v25_mode == "rl" and model is None:
            raise ValueError("选择 v25_rl 时需要先加载 PPO 模型。")
        segment_executor = V25GuidedSegmentExecutor(
            config=config,
            mode=v25_mode,
            model=model,
            enable_apas=bool(ui_state.get("v30_enable_apas", True)),
            seed=int(config.wind_seed),
        )

    disruptions = build_disruption_layer_v25(
        mission_map.start_xy,
        _v30_route_end_xy(mission_map),
        config=config,
        seed=int(config.wind_seed),
    )
    executor = SimpleTaskExecutor(config, segment_executor=segment_executor)
    result = executor.execute(mission_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    mission_map.save_json(output_dir / "mission_map.json")
    summary = {
        "success": result.success,
        "completed_inspections": result.completed_inspections,
        "charging_visits": result.charging_visits,
        "returned_home": result.returned_home,
        "final_position_xy": result.final_position_xy,
        "total_time_s": result.total_time_s,
        "total_energy_used_j": result.total_energy_used_j,
        "remaining_energy_j": result.remaining_energy_j,
        "failure_reason": result.failure_reason,
        "path_points": len(result.actual_path_xyz),
        "control_mode": control_mode,
        "apas_enabled": bool(ui_state.get("v30_enable_apas", True)) if control_mode.startswith("v25_") else False,
        "map_loaded_from_file": bool(getattr(estimator.map, "map_loaded_from_file", False)),
        "map_source_path": str(getattr(estimator.map, "map_source_path", "")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_mission_terrain_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "mission_terrain_trajectory.png",
        full_map=True,
    )
    plot_mission_terrain_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "mission_terrain_zoom.png",
        title="V3.0 Task Mission Detail View",
        full_map=False,
    )
    plot_wind_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        output_path=output_dir / "observable_wind_trajectory.png",
        include_random_layer=False,
        include_trajectory=True,
        full_map=True,
        title="Trajectory on Observable / Predictable Wind Field",
    )
    plot_wind_map(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "true_wind_trajectory.png",
        include_random_layer=True,
        include_trajectory=True,
        full_map=True,
        title="Trajectory on Observable + Random Layer Wind Field",
    )
    plot_mission_elevation_profile(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        output_path=output_dir / "mission_elevation_profile.png",
    )
    generate_wind_trajectory_gif(
        estimator=estimator,
        mission_map=mission_map,
        result=result,
        disruptions=disruptions,
        output_path=output_dir / "true_wind_trajectory.gif",
        frames=int(ui_state.get("v30_gif_frames", 20)),
    )
    return summary


def run_mission_case(case_name: str, mode_name: str, disturbance_enabled: bool, ui_state: dict, rl_model, output_dir: Path):
    """运行测评单个案例（支持单机与编队）"""
    config = apply_ui_config(ui_state, disturbance_enabled)
    if mode_name != "rl":
        config.gust_obs_noise_std = 0.0

    start_xy = ui_state["matrix_start"]
    goal_xy = ui_state["matrix_goal"]
    is_swarm = ui_state["fleet_mode"] == "Swarm"

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = Visualizer(config, estimator)
    animator = MissionAnimator(config, estimator)

    start_t = time.time()

    if is_swarm:
        # 四机编队模式
        executor = SwarmMissionExecutor(config, estimator, physics, battery, master_mode=mode_name, rl_model=rl_model)
        mission_result = executor.execute_mission(start_xy, goal_xy)
        
        vis.plot_swarm_execution(mission_result, start_xy, goal_xy, save_dir=str(output_dir))
        vis.plot_swarm_elevation_profile(mission_result, save_dir=str(output_dir))
        animator.generate_swarm_gif(mission_result, start_xy, goal_xy, filename=str(output_dir / f"{case_name}.gif"))
    else:
        # 单机模式
        planner = AStarPlanner(config, estimator, physics)
        executor = MissionExecutor(config, estimator, physics, battery, planner)
        mission_result = executor.execute_mission(start_xy, goal_xy)
        
        vis.plot_single_mission_execution(mission_result, start_xy, goal_xy, method_name=mode_name.upper(), save_dir=str(output_dir))
        vis.plot_elevation_profile(mission_result, save_dir=str(output_dir), method_name=mode_name.upper())
        animator.generate_gif(mission_result, start_xy, goal_xy, filename=str(output_dir / f"{case_name}.gif"), physics_engine=physics)

    elapsed = time.time() - start_t

    # 🌟 无论单机还是多机，都生成一份真 3D 的 HTML 文件保存下来
    save_3d_interactive_html(map_manager, mission_result, output_dir / "3d_trajectory.html")

    return {
        "场景": case_name,
        "控制算法": mode_name.upper(),
        "微观扰动": "开启 (GUST)" if disturbance_enabled else "无 (CLEAN)",
        "任务成功": "✅" if mission_result.success else "❌",
        "终止原因": mission_result.failure_reason or "安全抵达",
        "飞行耗时(s)": round(mission_result.total_mission_time_s, 1),
        "消耗能量(kJ)": round(mission_result.total_energy_used_j / 1000.0, 1),
        "重规划次数": mission_result.total_replans,
        "计算耗时(s)": round(elapsed, 2),
        "output_dir": str(output_dir),
    }


def display_artifacts(output_dir: Path, gif_name: str | None = None):
    """端正美观地渲染生成的文件 (加入 3D 视图内嵌)"""
    # 查找静态图纸
    swarm_static = output_dir / "swarm_static_trajectories.png"
    single_static = output_dir / "analysis_01_terrain_and_paths.png"
    static_png = swarm_static if swarm_static.exists() else single_static

    swarm_prof = output_dir / "swarm_elevation_profile.png"
    single_prof = output_dir / "analysis_03_elevation_profile.png"
    profile_png = swarm_prof if swarm_prof.exists() else single_prof

    gif_path = output_dir / gif_name if gif_name else None
    if gif_path is None:
        gifs = list(output_dir.glob("*.gif"))
        gif_path = gifs[0] if gifs else None

    html_3d_path = output_dir / "3d_trajectory.html"

    # 第一排：居中显示大尺寸 4D 动图
    if gif_path and gif_path.exists():
        col_g1, col_g2, col_g3 = st.columns([1, 4, 1]) # 中间占比大，实现居中
        with col_g2:
            st.image(str(gif_path), caption=f"4D 时空动态飞行追踪 ({gif_path.name})", use_container_width=True)
            
    st.divider()

    # 第二排：左右对齐显示 2D 静态结果
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if static_png.exists():
            st.image(str(static_png), caption="最终静态拓扑/平面轨迹图", use_container_width=True)
    with col_s2:
        if profile_png.exists():
            st.image(str(profile_png), caption="飞行高度与地形时间剖面图", use_container_width=True)

    # 🌟 第三排：满宽展示可交互 3D 渲染图
    if html_3d_path.exists():
        st.divider()
        st.markdown("#### 🌍 真 3D 航迹数字孪生 (Interactive 3D View)")
        st.caption("提示：你可以使用鼠标左键旋转视角，滚轮放大缩小，悬停查看具体的空间坐标与海拔。")
        # 读取本地 HTML 并通过 iframe 嵌入 Streamlit
        with open(html_3d_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600)
            
    st.caption(f"📂 成果文件保存路径: {output_dir}")


def list_result_dirs():
    """获取之前所有的结果文件夹"""
    if not RESULTS_ROOT.exists():
        return []
    return sorted([path for path in RESULTS_ROOT.iterdir() if path.is_dir()], key=lambda p: p.name)


def main():
    st.set_page_config(page_title="无人机风暴环境突防测控中心", page_icon="🛰️", layout="wide")
    st.title("🛰️ 4D 时空极值风险规避与协同控制中心")
    st.caption("支持自定义高程地图导入、单/多机编队切换，以及不同算法抗环境扰动的矩阵测评与 3D 可视化。")

    if "env_confirmed" not in st.session_state:
        st.session_state["env_confirmed"] = False
    if "env_confirmed_state" not in st.session_state:
        st.session_state["env_confirmed_state"] = None
    if "env_confirm_error" not in st.session_state:
        st.session_state["env_confirm_error"] = ""

    # ==========================
    # Step 1: 先确认地形/地图/风场参数
    # ==========================
    st.sidebar.header("Step 1/2 · 地形与风场确认")
    uploaded_map_ok = False
    uploaded_map_error = ""
    uploaded_map = st.sidebar.file_uploader(
        "1. 上传自定义高程图 (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        help="不上传则使用默认的瑞士雪山地图。建议上传正方形图片。",
    )
    default_config = SimulationConfig()
    if uploaded_map is not None:
        temp_map_path = os.path.join(project_root, "temp_uploaded_map.png")
        uploaded_map_ok, uploaded_map_error = save_uploaded_map_as_stable_png(uploaded_map, temp_map_path)
        if uploaded_map_ok:
            map_path = temp_map_path
        else:
            map_path = default_config.map_path
            st.sidebar.error(f"上传图预处理失败，已回退默认地图：{uploaded_map_error}")
    else:
        map_path = default_config.map_path

    map_size_km = st.sidebar.number_input("2. 地图物理边长 (km)", value=17.28, step=1.0, help="定义这张图片在真实世界里代表多宽。")
    col_alt1, col_alt2 = st.sidebar.columns(2)
    min_alt = col_alt1.number_input("最低海拔(m)", value=563.0, step=50.0)
    max_alt = col_alt2.number_input("最高海拔(m)", value=3985.4, step=50.0)
    map_res = st.sidebar.slider("3. 内部解析分辨率 (像素)", 100, 600, 300, 50, help="值越大网格越精细，但 A* 寻路会变慢。")

    time_of_day = st.sidebar.selectbox("昼夜模式 (影响地形风向)", ["Day (白昼)", "Night (夜晚)"], index=1)
    time_val = "Day" if "Day" in time_of_day else "Night"
    col_wind1, col_wind2 = st.sidebar.columns(2)
    env_wind_u = col_wind1.number_input("恒定背景风向 X (m/s)", value=-3.0, step=1.0)
    env_wind_v = col_wind2.number_input("恒定背景风向 Y (m/s)", value=5.0, step=1.0)
    enable_nfz = st.sidebar.checkbox("启用静态禁飞区 (NFZ)", value=True)

    default_nfz = [(float(cx), float(cy), float(r)) for (cx, cy, r) in SimulationConfig().nfz_list_km]
    if "nfz_zone_count" not in st.session_state:
        st.session_state["nfz_zone_count"] = len(default_nfz)
    zone_count = st.sidebar.number_input(
        "禁飞区数量",
        min_value=0,
        max_value=10,
        value=int(st.session_state["nfz_zone_count"]),
        step=1,
    )
    st.session_state["nfz_zone_count"] = int(zone_count)

    nfz_list_km = []
    if enable_nfz:
        with st.sidebar.expander("🚫 禁飞区坐标设置 (单位: km)", expanded=False):
            st.caption("每个禁飞区由中心坐标 (X, Y) 和半径 R 构成。")
            for idx in range(int(zone_count)):
                base = default_nfz[idx] if idx < len(default_nfz) else (0.0, 0.0, 1.0)
                c1, c2, c3 = st.columns(3)
                cx_km = c1.number_input(f"Z{idx+1} X", value=float(base[0]), step=0.1, key=f"nfz_x_{idx}")
                cy_km = c2.number_input(f"Z{idx+1} Y", value=float(base[1]), step=0.1, key=f"nfz_y_{idx}")
                r_km = c3.number_input(f"Z{idx+1} R", min_value=0.0, value=float(base[2]), step=0.1, key=f"nfz_r_{idx}")
                nfz_list_km.append((float(cx_km), float(cy_km), float(r_km)))
    else:
        nfz_list_km = []

    wind_seed = st.sidebar.number_input("随机风暴生成种子", min_value=0, value=37, step=1)
    storm_count = st.sidebar.slider("动态风暴数量", 1, 8, 3)

    candidate_env_state = {
        "map_path": map_path,
        "map_size_km": map_size_km,
        "min_alt": min_alt,
        "max_alt": max_alt,
        "map_resolution": map_res,
        "time_of_day": time_val,
        "env_wind_u": env_wind_u,
        "env_wind_v": env_wind_v,
        "enable_nfz": enable_nfz,
        "nfz_list_km": nfz_list_km,
        "wind_seed": int(wind_seed),
        "storm_count": int(storm_count),
        "custom_map_uploaded": uploaded_map is not None and uploaded_map_ok,
        "uploaded_map_error": uploaded_map_error,
    }

    col_c1, col_c2 = st.sidebar.columns(2)
    if col_c1.button("✅ 确认参数并生成地形图", use_container_width=True):
        try:
            _cfg, _map, _est = build_environment_preview(candidate_env_state)
            if candidate_env_state.get("custom_map_uploaded", False) and not getattr(_map, "map_loaded_from_file", True):
                raise ValueError(
                    "上传地图读取失败，当前结果是回退的虚拟地形。"
                    "请检查图片格式或路径（Windows 中文路径可尝试重传后再确认）。"
                )
            st.session_state["env_confirmed_state"] = candidate_env_state
            st.session_state["env_confirmed"] = True
            st.session_state["env_confirm_error"] = ""
        except Exception as exc:
            st.session_state["env_confirmed"] = False
            st.session_state["env_confirm_error"] = str(exc)
        st.rerun()

    if col_c2.button("🔄 重新编辑", use_container_width=True):
        st.session_state["env_confirmed"] = False
        st.session_state["env_confirmed_state"] = None
        st.session_state["env_confirm_error"] = ""
        st.rerun()

    if st.session_state.get("env_confirm_error"):
        st.sidebar.error(f"参数确认失败：{st.session_state['env_confirm_error']}")
    elif candidate_env_state.get("uploaded_map_error"):
        st.sidebar.warning(f"上传图处理提示：{candidate_env_state['uploaded_map_error']}")

    if not st.session_state.get("env_confirmed") or st.session_state.get("env_confirmed_state") is None:
        st.info("请先在左侧完成 Step 1：设置地形/地图/风场参数并点击“确认参数并生成地形图”。")
        return

    env_state = st.session_state["env_confirmed_state"]
    st.success("✅ Step 1 已完成：地形与风场参数已锁定。你现在可以查看预览图，并进入后续任务参数配置。")

    try:
        env_config, env_map, env_estimator = build_environment_preview(env_state)
    except Exception as exc:
        st.error(f"环境预览构建失败：{exc}")
        return

    map_status = "已从文件成功读取" if getattr(env_map, "map_loaded_from_file", False) else "未读取到文件，使用了虚拟地形"
    st.caption(f"地图加载状态：{map_status}")
    st.caption(f"地图来源路径：{getattr(env_map, 'map_source_path', env_state.get('map_path', ''))}")

    if not getattr(env_map, "map_loaded_from_file", True):
        st.warning(
            "⚠️ 当前显示的是回退虚拟地形（未成功读取地图文件）。"
            "请重新上传图片并点击“确认参数并生成地形图”。"
        )
        if getattr(env_map, "map_load_error", ""):
            st.error(f"读取失败原因：{env_map.map_load_error}")

    st.write("### 🗺️ Step 1 结果预览")
    if env_state.get("custom_map_uploaded", False):
        with st.expander("查看上传原图", expanded=False):
            preview_img = load_image_preview_for_streamlit(env_state["map_path"])
            if preview_img is None:
                st.warning("原图预览失败：无法读取上传图片。")
            else:
                st.image(
                    preview_img,
                    caption="上传的原始高程图（已转换为预览格式）",
                    use_container_width=True,
                    clamp=True,
                    output_format="PNG",
                )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.plotly_chart(create_terrain_preview(env_map, env_state.get("nfz_list_km", [])), use_container_width=True)
    with col_p2:
        st.plotly_chart(
            create_terrain_wind_preview(env_map, env_estimator, env_config, env_state.get("nfz_list_km", [])),
            use_container_width=True,
        )

    # ==========================
    # Step 2: 再配置任务与路径规划参数
    # ==========================
    st.sidebar.divider()
    st.sidebar.header("Step 2/2 · 任务与路径规划参数")

    with st.sidebar.expander("🌪️ 微观阵风扰动设定", expanded=False):
        gust_trigger_prob = st.slider("随机阵风触发概率", 0.0, 0.1, 0.02, 0.01)
        gust_duration_s = st.slider("单次阵风持续时间 (s)", 2.0, 30.0, 8.0, 2.0)
        gust_min_speed_mps = st.number_input("阵风最低风速 (m/s)", value=4.0)
        gust_max_speed_mps = st.number_input("阵风最高风速 (m/s)", value=10.0)

    with st.sidebar.expander("🚁 无人机物理与 A* 参数", expanded=False):
        cruise_speed_mps = st.slider("无人机巡航速度 (m/s)", 5.0, 30.0, 15.0, 1.0)
        max_power = st.number_input("电机最大抗风功率限制 (W)", value=4000.0, step=100.0)
        heuristic_safety_factor = st.slider("A* 启发式贪婪加速因子", 1.0, 5.0, 2.0, 0.5)
        max_replans = st.number_input("遇到死胡同时全局最大重规划次数", value=100, step=10)

    with st.sidebar.expander("🛡️ 异构集群护盾设定", expanded=False):
        enable_support_shield_mode = st.checkbox("启用 Support (支援蜂) 物理抗风护盾", value=True)
        support_shield_master_radius_m = st.slider("威胁风暴进入多少米触发护盾", 800, 2500, 1400, 100)
        support_shield_offset_m = st.slider("支援蜂超前掩护距离 (m)", 200, 1000, 450, 50)

    with st.sidebar.expander("🧠 强化学习 (RL) 模型与传感器", expanded=False):
        model_path = st.text_input("PPO 神经网络权重路径", value=str(DEFAULT_MODEL_PATH))
        gust_obs_noise_std = st.slider("RL 雷达传感器环境噪声标准差", 0.0, 0.1, 0.01, 0.01)
        rl_model = load_rl_model_cached(model_path)
        if rl_model is None:
            st.error("❌ RL 模型未加载或不存在。")
        else:
            st.success("✅ PPO RL 模型已就绪！")

    # ==========================
    # 顶部控制：模式与起终点
    # ==========================
    st.subheader("🛠️ 任务编队与坐标设定")
    map_size_km_locked = float(env_state["map_size_km"])

    col_top1, col_top2 = st.columns([1, 2])
    with col_top1:
        fleet_mode_str = st.radio("选择出击机群规模：", ["四机编队 (FANET Swarm)", "单机模式 (Single Drone)"])
        fleet_mode = "Swarm" if "四机" in fleet_mode_str else "Single"
        if fleet_mode == "Single":
            st.warning("⚠️ 警告：单机模式视野严重受限！失去预警与护盾掩护后，迎面撞上风暴大概率坠机！此外，本系统单机回退使用传统 A* 算法。")

    with col_top2:
        half_m = float((map_size_km_locked * 1000) / 2)

        def clamp(v):
            return max(-half_m, min(v, half_m))

        st.write("请在滑块上拖动或直接点击数字修改（单位：米）：")
        col_coord1, col_coord2 = st.columns(2)
        start_x = col_coord1.slider("起点 X (m)", -half_m, half_m, clamp(-8000.0), 100.0)
        start_y = col_coord2.slider("起点 Y (m)", -half_m, half_m, clamp(-8000.0), 100.0)
        goal_x = col_coord1.slider("终点 X (m)", -half_m, half_m, clamp(6000.0), 100.0)
        goal_y = col_coord2.slider("终点 Y (m)", -half_m, half_m, clamp(7500.0), 100.0)

    ui_state = {
        **env_state,
        "gust_trigger_prob": gust_trigger_prob,
        "gust_duration_s": gust_duration_s,
        "gust_min_speed_mps": gust_min_speed_mps,
        "gust_max_speed_mps": gust_max_speed_mps,
        "cruise_speed_mps": cruise_speed_mps,
        "max_power": max_power,
        "heuristic_safety_factor": heuristic_safety_factor,
        "max_replans": max_replans,
        "enable_support_shield_mode": enable_support_shield_mode,
        "support_shield_master_radius_m": support_shield_master_radius_m,
        "support_shield_offset_m": support_shield_offset_m,
        "gust_obs_noise_std": gust_obs_noise_std,
        "fleet_mode": fleet_mode,
        "matrix_start": (start_x, start_y),
        "matrix_goal": (goal_x, goal_y),
    }

    # ==========================
    # 主体界面 Tabs
    # ==========================
    preview_map = env_map
    tab_matrix, tab_v30, tab_artifacts = st.tabs(["🚀 任务推演大厅", "V3.0 地图巡检模式", "📂 历史成果画廊"])

    with tab_matrix:
        col_m1, col_m2 = st.columns([1.5, 1])
        with col_m1:
            st.plotly_chart(
                create_map_preview(preview_map, (start_x, start_y), (goal_x, goal_y), env_state.get("nfz_list_km", [])),
                use_container_width=True,
            )
        with col_m2:
            st.info("💡 **测试说明**\n\n通过对强化学习(RL)与传统路径规划(A*)分别施加不可见微观阵风，测试抗扰动能力与生存率。单机模式强制仅运行 A*。")

            run_astar = st.checkbox("☑️ 对比项：传统 A* (A* Replanning)", value=True)
            run_rl = st.checkbox("☑️ 对比项：强化学习微观控制 (RL Agent)", value=True, disabled=(fleet_mode == "Single"))

            st.divider()
            run_clean = st.checkbox("☑️ 环境：无随机阵风基准环境 (Clean)", value=True)
            run_gust = st.checkbox("☑️ 环境：注入强微观随机阵风干扰 (Gust)", value=True)

            if st.button("▶️ 一键启动选中推演矩阵", type="primary", use_container_width=True):
                results = []
                cases = []

                if run_astar:
                    if run_clean:
                        cases.append(("astar_clean (无阵风)", "astar", False, None))
                    if run_gust:
                        cases.append(("astar_gust (阵风干扰)", "astar", True, None))

                if run_rl and fleet_mode == "Swarm" and rl_model is not None:
                    if run_clean:
                        cases.append(("rl_clean (无阵风)", "rl", False, rl_model))
                    if run_gust:
                        cases.append(("rl_gust (阵风干扰)", "rl", True, rl_model))

                if not cases:
                    st.warning("⚠️ 请至少组合一种要测试的算法和环境！如果选择 RL 需确保在四机编队模式且模型已加载。")
                else:
                    progress_text = "正在执行 4D 物理仿真与图像/3D渲染，请耐心等待..."
                    my_bar = st.progress(0, text=progress_text)
                    for idx, (case_name, mode_name, disturbance_enabled, model) in enumerate(cases):
                        folder_prefix = "ui_swarm" if fleet_mode == "Swarm" else "ui_single"
                        output_dir = RESULTS_ROOT / f"{folder_prefix}_{case_name.split()[0]}"
                        res = run_mission_case(case_name, mode_name, disturbance_enabled, ui_state, model, output_dir)
                        results.append(res)
                        my_bar.progress((idx + 1) / len(cases), text=f"✅ 已完成: {case_name}")

                    st.session_state["matrix_results"] = results
                    st.success("🎉 所有推演任务全部完成，结果已加载！")

        matrix_results = st.session_state.get("matrix_results", [])
        if matrix_results:
            st.divider()
            st.write("### 📊 推演结果与性能比对")
            summary_df = pd.DataFrame(matrix_results)
            st.dataframe(
                summary_df[["场景", "控制算法", "微观扰动", "任务成功", "终止原因", "飞行耗时(s)", "消耗能量(kJ)", "重规划次数"]],
                use_container_width=True,
            )
            st.write("### 🎞️ 动态复盘与 3D 可视化图集")
            for row in matrix_results:
                with st.expander(f"📍 {row['场景']} | 算法: {row['控制算法']} | 环境: {row['微观扰动']}"):
                    display_artifacts(Path(row["output_dir"]), gif_name=f"{row['场景']}.gif")

    with tab_v30:
        st.write("### V3.0 地图巡检任务")
        st.caption("在已确认的地形/风场环境上标定起点、巡检点和充电点，并选择 A* / Expert / RL 执行栈。")

        half_m_v30 = float((map_size_km_locked * 1000.0) / 2.0)
        showcase_path = Path(project_root) / "v30" / "examples" / "mission_map_showcase.json"
        default_showcase = MissionMap.load_json(showcase_path)

        preset_choice = st.selectbox(
            "任务模板",
            ["展示级 Bernese 山地巡检", "空白自定义"],
            index=0,
            help="展示级模板使用全图坐标；空白自定义适合从当前起终点快速开始。",
        )
        if preset_choice == "展示级 Bernese 山地巡检":
            base_mission = default_showcase
        else:
            base_mission = MissionMap(
                name="ui_blank_inspection",
                start_xy=(float(start_x), float(start_y)),
                home_xy=(float(start_x), float(start_y)),
                inspection_points=[InspectionPoint(id="inspection-1", xy=(float(goal_x), float(goal_y)), priority=1.0)],
                charging_stations=[ChargingStation(id="charge-1", xy=(float(start_x), float(start_y)))],
            )

        col_v30_a, col_v30_b = st.columns([1.15, 1.0])
        with col_v30_a:
            st.write("#### 起点 / 归航点")
            c_start_1, c_start_2 = st.columns(2)
            v30_start_x = c_start_1.number_input("Start X (m)", -half_m_v30, half_m_v30, float(base_mission.start_xy[0]), 100.0)
            v30_start_y = c_start_2.number_input("Start Y (m)", -half_m_v30, half_m_v30, float(base_mission.start_xy[1]), 100.0)
            home_default = base_mission.home_xy or base_mission.start_xy
            c_home_1, c_home_2 = st.columns(2)
            v30_home_x = c_home_1.number_input("Home X (m)", -half_m_v30, half_m_v30, float(home_default[0]), 100.0)
            v30_home_y = c_home_2.number_input("Home Y (m)", -half_m_v30, half_m_v30, float(home_default[1]), 100.0)

        with col_v30_b:
            st.write("#### 执行方案")
            control_label = st.selectbox(
                "控制 / 执行栈",
                [
                    "legacy_astar：传统 A* 分段执行",
                    "v25_astar：A* + APAS + waypoint-skip",
                    "v25_expert：Expert + APAS + waypoint-skip",
                    "v25_rl：PPO RL + APAS + waypoint-skip",
                ],
                index=0,
            )
            control_mode = control_label.split("：", 1)[0]
            v30_enable_apas = st.checkbox("v2.5 模式启用 APAS 安全盾", value=True)
            v30_stress = st.selectbox("随机层压力等级", ["normal", "hard", "extreme", "fragile"], index=3)
            v30_max_replans = st.number_input("每段最大 replan 次数", min_value=1, max_value=200, value=35, step=5)
            v30_max_mission_time_s = st.number_input("最大任务时间 (s)", min_value=300.0, max_value=20000.0, value=3600.0, step=300.0)
            v30_update_interval_s = st.number_input("任务推进步长 / 重规划间隔 (s)", min_value=10.0, max_value=300.0, value=60.0, step=10.0)
            v30_gif_frames = st.slider("GIF 帧数", 6, 80, 20, 2)

        st.write("#### 巡检点标定")
        inspections_default = pd.DataFrame([
            {
                "enabled": True,
                "id": p.id,
                "x_m": float(p.xy[0]),
                "y_m": float(p.xy[1]),
                "priority": float(p.priority),
                "service_time_s": float(p.service_time_s),
                "risk_value": float(p.risk_value),
                "deadline_s": np.nan if p.deadline_s is None else float(p.deadline_s),
            }
            for p in base_mission.inspection_points
        ])
        inspections_df = st.data_editor(
            inspections_default,
            num_rows="dynamic",
            use_container_width=True,
            key=f"v30_inspections_{preset_choice}",
        )

        st.write("#### 充电点标定")
        chargers_default = pd.DataFrame([
            {
                "available": bool(c.available),
                "id": c.id,
                "x_m": float(c.xy[0]),
                "y_m": float(c.xy[1]),
                "charge_rate_j_per_s": float(c.charge_rate_j_per_s),
                "docking_time_s": float(c.docking_time_s),
                "target_soc": float(c.target_soc),
            }
            for c in base_mission.charging_stations
        ])
        chargers_df = st.data_editor(
            chargers_default,
            num_rows="dynamic",
            use_container_width=True,
            key=f"v30_chargers_{preset_choice}",
        )

        mission_map_v30 = build_v30_mission_map_from_ui(
            start_xy=(v30_start_x, v30_start_y),
            home_xy=(v30_home_x, v30_home_y),
            inspections_df=inspections_df,
            chargers_df=chargers_df,
        )

        st.plotly_chart(
            create_v30_mission_preview(env_map, mission_map_v30, env_state.get("nfz_list_km", [])),
            use_container_width=True,
        )

        if st.button("运行 V3.0 地图巡检任务", type="primary", use_container_width=True):
            v30_ui_state = {
                **ui_state,
                "v30_max_replans": int(v30_max_replans),
                "v30_max_mission_time_s": float(v30_max_mission_time_s),
                "v30_update_interval_s": float(v30_update_interval_s),
                "v30_enable_apas": bool(v30_enable_apas),
                "v30_stress": str(v30_stress),
                "v30_gif_frames": int(v30_gif_frames),
            }
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = RESULTS_ROOT / f"ui_v30_inspection_{control_mode}_{timestamp}"
            try:
                with st.spinner("正在执行 V3.0 地图巡检并渲染结果图..."):
                    summary = run_v30_inspection_case(
                        ui_state=v30_ui_state,
                        mission_map=mission_map_v30,
                        control_mode=control_mode,
                        rl_model=rl_model,
                        output_dir=output_dir,
                    )
                st.session_state["v30_last_result"] = {"summary": summary, "output_dir": str(output_dir)}
                st.success("V3.0 地图巡检任务完成")
            except Exception as exc:
                st.error(f"V3.0 地图巡检任务失败：{exc}")

        v30_last = st.session_state.get("v30_last_result")
        if v30_last:
            out_dir = Path(v30_last["output_dir"])
            summary = v30_last["summary"]
            st.write("### V3.0 结果摘要")
            st.json(summary)
            for image_name in [
                "mission_terrain_trajectory.png",
                "mission_terrain_zoom.png",
                "true_wind_trajectory.png",
                "observable_wind_trajectory.png",
                "mission_elevation_profile.png",
            ]:
                image_path = out_dir / image_name
                if image_path.exists():
                    st.image(str(image_path), caption=image_name, use_container_width=True)
            gif_path = out_dir / "true_wind_trajectory.gif"
            if gif_path.exists():
                st.image(str(gif_path), caption="true_wind_trajectory.gif", use_container_width=True)
            st.caption(f"输出目录：{out_dir}")

    with tab_artifacts:
        result_dirs = list_result_dirs()
        if not result_dirs:
            st.info("目前尚未生成任何实验数据。")
        else:
            selected_dir = st.selectbox("选择要回放的历史实验归档：", result_dirs, format_func=lambda p: p.name)
            if selected_dir:
                gifs = list(selected_dir.glob("*.gif"))
                gif_name = gifs[0].name if gifs else None
                display_artifacts(selected_dir, gif_name=gif_name)


if __name__ == "__main__":
    main()
