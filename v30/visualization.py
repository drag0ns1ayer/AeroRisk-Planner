from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from core.estimator import StateEstimator
from v25.disruptions import DisruptionLayerV25
from v30.mission_map import MissionMap
from v30.task_executor import TaskExecutionResult


WindSampler = Callable[[float, float, float], np.ndarray]


def _local_bounds(mission_map: MissionMap, result: TaskExecutionResult | None = None, padding_m: float = 350.0):
    xs = [mission_map.start_xy[0]]
    ys = [mission_map.start_xy[1]]
    if mission_map.home_xy is not None:
        xs.append(mission_map.home_xy[0])
        ys.append(mission_map.home_xy[1])
    for point in mission_map.inspection_points:
        xs.append(point.xy[0])
        ys.append(point.xy[1])
    for station in mission_map.charging_stations:
        xs.append(station.xy[0])
        ys.append(station.xy[1])
    if result is not None:
        for x, y, _ in result.actual_path_xyz:
            xs.append(x)
            ys.append(y)

    return min(xs) - padding_m, max(xs) + padding_m, min(ys) - padding_m, max(ys) + padding_m


def _predictable_wind_sampler(estimator: StateEstimator) -> WindSampler:
    def sample(x: float, y: float, t_s: float) -> np.ndarray:
        z = estimator.get_altitude(x, y) + 50.0
        wind = estimator.get_wind(x, y, z=z, t_s=t_s)
        return np.asarray(wind, dtype=float)

    return sample


def _combined_wind_sampler(estimator: StateEstimator, disruptions: DisruptionLayerV25 | None) -> WindSampler:
    predictable = _predictable_wind_sampler(estimator)

    def sample(x: float, y: float, t_s: float) -> np.ndarray:
        base = predictable(x, y, t_s)
        if disruptions is None:
            return base
        random_wind, _ = disruptions.disturbance_at(x, y, t_s, np.array([1.0, 0.0], dtype=float))
        return base + np.asarray(random_wind, dtype=float)

    return sample


def _sample_wind_grid(bounds, wind_sampler: WindSampler, t_s: float, grid_size: int = 24):
    min_x, max_x, min_y, max_y = bounds
    xs = np.linspace(min_x, max_x, grid_size)
    ys = np.linspace(min_y, max_y, grid_size)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    S = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            wind = wind_sampler(float(X[i, j]), float(Y[i, j]), t_s)
            U[i, j] = float(wind[0])
            V[i, j] = float(wind[1])
            S[i, j] = float(np.linalg.norm(wind))
    return X, Y, U, V, S


def _draw_mission_annotations(ax, mission_map: MissionMap):
    ax.scatter(mission_map.start_xy[0] / 1000.0, mission_map.start_xy[1] / 1000.0, c="gold", s=130, marker="*", edgecolors="black", label="Start", zorder=5)
    if mission_map.home_xy is not None:
        ax.scatter(mission_map.home_xy[0] / 1000.0, mission_map.home_xy[1] / 1000.0, c="white", s=85, marker="H", edgecolors="black", label="Home", zorder=5)
    for idx, point in enumerate(mission_map.inspection_points):
        label = "Inspection" if idx == 0 else None
        ax.scatter(point.xy[0] / 1000.0, point.xy[1] / 1000.0, c="tab:orange", s=75, marker="o", edgecolors="black", label=label, zorder=5)
        ax.text(point.xy[0] / 1000.0, point.xy[1] / 1000.0, f" {point.id}", fontsize=8, zorder=6)
    for idx, station in enumerate(mission_map.charging_stations):
        label = "Charging" if idx == 0 else None
        ax.scatter(station.xy[0] / 1000.0, station.xy[1] / 1000.0, c="tab:blue", s=80, marker="P", edgecolors="black", label=label, zorder=5)


def _draw_random_hazards(ax, disruptions: DisruptionLayerV25 | None, t_s: float):
    if disruptions is None:
        return
    storm = disruptions.destructive_storm
    if storm.lifetime_s > 0.0:
        center = storm.center_at(t_s)
        halo = plt.Circle((center[0] / 1000.0, center[1] / 1000.0), storm.halo_radius_m / 1000.0, color="purple", alpha=0.12, label="Random storm halo")
        core = plt.Circle((center[0] / 1000.0, center[1] / 1000.0), storm.core_radius_m / 1000.0, color="crimson", alpha=0.25, label="Destructive core")
        ax.add_patch(halo)
        ax.add_patch(core)
    for idx, region in enumerate(disruptions.local_wind_regions):
        label = "Random wind region" if idx == 0 else None
        circle = plt.Circle((region.center_xy[0] / 1000.0, region.center_xy[1] / 1000.0), region.radius_m / 1000.0, color="cyan", alpha=0.10, label=label)
        ax.add_patch(circle)


def plot_wind_map(
    *,
    estimator: StateEstimator,
    mission_map: MissionMap,
    output_path: str | Path,
    result: TaskExecutionResult | None = None,
    disruptions: DisruptionLayerV25 | None = None,
    include_random_layer: bool = False,
    include_trajectory: bool = False,
    title: str = "Wind Field",
    t_s: float = 0.0,
) -> None:
    bounds = _local_bounds(mission_map, result)
    sampler = _combined_wind_sampler(estimator, disruptions) if include_random_layer else _predictable_wind_sampler(estimator)
    X, Y, U, V, S = _sample_wind_grid(bounds, sampler, t_s=t_s)

    fig, ax = plt.subplots(figsize=(10, 8))
    speed = ax.contourf(X / 1000.0, Y / 1000.0, S, levels=20, cmap="viridis", alpha=0.82)
    fig.colorbar(speed, ax=ax, label="Wind speed (m/s)")
    ax.quiver(X / 1000.0, Y / 1000.0, U, V, color="white", alpha=0.75, scale=260)

    if include_random_layer:
        _draw_random_hazards(ax, disruptions, t_s)
    _draw_mission_annotations(ax, mission_map)
    if include_trajectory and result is not None and len(result.actual_path_xyz) >= 2:
        arr = np.asarray(result.actual_path_xyz, dtype=float)
        ax.plot(arr[:, 0] / 1000.0, arr[:, 1] / 1000.0, color="lime", linewidth=2.8, label="Executed trajectory", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def generate_wind_trajectory_gif(
    *,
    estimator: StateEstimator,
    mission_map: MissionMap,
    result: TaskExecutionResult,
    output_path: str | Path,
    disruptions: DisruptionLayerV25 | None = None,
    include_random_layer: bool = True,
    frames: int = 60,
) -> None:
    if len(result.actual_path_xyz) < 2:
        return

    bounds = _local_bounds(mission_map, result)
    sampler = _combined_wind_sampler(estimator, disruptions) if include_random_layer else _predictable_wind_sampler(estimator)
    path = np.asarray(result.actual_path_xyz, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y, U, V, S = _sample_wind_grid(bounds, sampler, t_s=0.0, grid_size=20)
    speed = ax.contourf(X / 1000.0, Y / 1000.0, S, levels=20, cmap="viridis", alpha=0.82)
    fig.colorbar(speed, ax=ax, label="Wind speed (m/s)")
    quiver = ax.quiver(X / 1000.0, Y / 1000.0, U, V, color="white", alpha=0.75, scale=260)
    _draw_mission_annotations(ax, mission_map)
    line, = ax.plot([], [], color="lime", linewidth=2.8, label="Executed trajectory")
    dot, = ax.plot([], [], marker="o", color="white", markeredgecolor="black", markersize=8)
    title = ax.set_title("V3.0 Mission Wind + Trajectory")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    total_time = max(float(result.total_time_s), 1.0)
    frame_count = max(2, int(frames))

    def update(frame_idx: int):
        t_s = total_time * frame_idx / max(frame_count - 1, 1)
        _, _, U_new, V_new, _ = _sample_wind_grid(bounds, sampler, t_s=t_s, grid_size=20)
        quiver.set_UVC(U_new, V_new)

        path_idx = min(len(path) - 1, int(math.ceil((len(path) - 1) * frame_idx / max(frame_count - 1, 1))))
        visible = path[: path_idx + 1]
        line.set_data(visible[:, 0] / 1000.0, visible[:, 1] / 1000.0)
        dot.set_data([visible[-1, 0] / 1000.0], [visible[-1, 1] / 1000.0])
        title.set_text(f"V3.0 Mission Wind + Trajectory | t={t_s:.1f}s")
        return line, dot, quiver, title

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=120, blit=False)
    ani.save(output, writer="pillow", fps=8)
    plt.close(fig)
