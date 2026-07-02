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


def _map_bounds(estimator: StateEstimator):
    return tuple(float(v) for v in estimator.get_bounds())


def _path_array(result: TaskExecutionResult | None) -> np.ndarray | None:
    if result is None or len(result.actual_path_xyz) < 2:
        return None
    path = np.asarray(result.actual_path_xyz, dtype=float)
    valid = np.isfinite(path).all(axis=1)
    path = path[valid]
    if len(path) < 2:
        return None
    return path


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


def _draw_terrain_background(ax, estimator: StateEstimator, bounds):
    terrain = estimator.map
    min_x, max_x, min_y, max_y = bounds
    X = np.asarray(terrain.X, dtype=float)
    Y = np.asarray(terrain.Y, dtype=float)
    dem = np.asarray(terrain.dem, dtype=float)
    mask = (X >= min_x) & (X <= max_x) & (Y >= min_y) & (Y <= max_y)
    if not np.any(mask):
        return

    terrain_layer = ax.contourf(
        X / 1000.0,
        Y / 1000.0,
        dem,
        levels=42,
        cmap="gist_earth",
        alpha=0.55,
        zorder=0,
    )
    ax.contour(
        X / 1000.0,
        Y / 1000.0,
        dem,
        levels=12,
        colors="black",
        linewidths=0.25,
        alpha=0.20,
        zorder=1,
    )
    ax.set_xlim(min_x / 1000.0, max_x / 1000.0)
    ax.set_ylim(min_y / 1000.0, max_y / 1000.0)
    return terrain_layer


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


def _draw_static_constraints(ax, estimator: StateEstimator):
    if not getattr(estimator.config, "enable_nfz", False):
        return
    zones = getattr(estimator.config, "nfz_list_km", [])
    for idx, (cx_km, cy_km, radius_km) in enumerate(zones):
        label = "Predictable NFZ" if idx == 0 else None
        nfz = plt.Circle(
            (float(cx_km), float(cy_km)),
            float(radius_km),
            facecolor="red",
            edgecolor="darkred",
            linewidth=1.8,
            alpha=0.18,
            hatch="///",
            label=label,
            zorder=4,
        )
        ax.add_patch(nfz)
        ax.text(
            float(cx_km),
            float(cy_km),
            "NFZ",
            color="darkred",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )


def _draw_forecast_storms(ax, estimator: StateEstimator, t_s: float, duration_s: float = 300.0):
    storm_manager = getattr(getattr(estimator, "wind", None), "storm_manager", None)
    if storm_manager is None or not getattr(estimator.config, "enable_storms", False):
        return
    active_storms = storm_manager.get_active_storms(float(t_s))
    for idx, storm in enumerate(active_storms):
        center = storm.center_at(float(t_s))
        cx_km = float(center[0]) / 1000.0
        cy_km = float(center[1]) / 1000.0
        vx_km = float(storm.velocity_xy[0]) / 1000.0
        vy_km = float(storm.velocity_xy[1]) / 1000.0
        label = "Forecast moving storm" if idx == 0 else None
        circle = plt.Circle(
            (cx_km, cy_km),
            float(storm.radius_m) / 1000.0,
            facecolor="royalblue",
            edgecolor="midnightblue",
            linewidth=1.5,
            alpha=0.16,
            label=label,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.annotate(
            "",
            xy=(cx_km + vx_km * duration_s, cy_km + vy_km * duration_s),
            xytext=(cx_km, cy_km),
            arrowprops=dict(arrowstyle="->", color="midnightblue", linestyle="--", linewidth=1.2, alpha=0.85),
            zorder=6,
        )
        ax.text(
            cx_km,
            cy_km,
            "Forecast\nstorm",
            color="midnightblue",
            fontsize=8,
            ha="center",
            va="center",
            zorder=7,
        )


def _draw_executed_path(ax, result: TaskExecutionResult | None, *, color: str = "lime", label: str = "Executed trajectory") -> None:
    path = _path_array(result)
    if path is None:
        return
    ax.plot(path[:, 0] / 1000.0, path[:, 1] / 1000.0, color=color, linewidth=3.0, label=label, zorder=6)
    ax.scatter(path[-1, 0] / 1000.0, path[-1, 1] / 1000.0, color=color, edgecolors="black", s=45, zorder=7)


def _draw_random_hazards(ax, disruptions: DisruptionLayerV25 | None, t_s: float):
    if disruptions is None:
        return
    storm = disruptions.destructive_storm
    if storm.lifetime_s > 0.0:
        center = storm.center_at(t_s)
        halo = plt.Circle(
            (center[0] / 1000.0, center[1] / 1000.0),
            storm.halo_radius_m / 1000.0,
            facecolor="mediumpurple",
            edgecolor="indigo",
            linewidth=1.6,
            alpha=0.18,
            label="Random storm halo",
            zorder=4,
        )
        core = plt.Circle(
            (center[0] / 1000.0, center[1] / 1000.0),
            storm.core_radius_m / 1000.0,
            facecolor="crimson",
            edgecolor="darkred",
            linewidth=1.8,
            alpha=0.34,
            label="Destructive storm core",
            zorder=5,
        )
        ax.add_patch(halo)
        ax.add_patch(core)
        ax.text(
            center[0] / 1000.0,
            center[1] / 1000.0,
            "Random\nstorm",
            color="indigo",
            fontsize=8,
            ha="center",
            va="center",
            zorder=7,
        )
    for idx, region in enumerate(disruptions.local_wind_regions):
        label = "Random wind region" if idx == 0 else None
        circle = plt.Circle(
            (region.center_xy[0] / 1000.0, region.center_xy[1] / 1000.0),
            region.radius_m / 1000.0,
            facecolor="cyan",
            edgecolor="deepskyblue",
            linewidth=1.2,
            alpha=0.18,
            label=label,
            zorder=4,
        )
        ax.add_patch(circle)


def plot_mission_terrain_map(
    *,
    estimator: StateEstimator,
    mission_map: MissionMap,
    output_path: str | Path,
    result: TaskExecutionResult | None = None,
    disruptions: DisruptionLayerV25 | None = None,
    title: str = "V3.0 Task Mission on Bernese Terrain",
    full_map: bool = True,
    t_s: float = 0.0,
) -> None:
    bounds = _map_bounds(estimator) if full_map else _local_bounds(mission_map, result)
    fig, ax = plt.subplots(figsize=(11, 9))
    terrain = ax.contourf(
        estimator.map.X / 1000.0,
        estimator.map.Y / 1000.0,
        estimator.map.dem,
        levels=58,
        cmap="gist_earth",
        alpha=0.86,
        zorder=0,
    )
    fig.colorbar(terrain, ax=ax, pad=0.02, label="Altitude (m)")
    ax.contour(
        estimator.map.X / 1000.0,
        estimator.map.Y / 1000.0,
        estimator.map.dem,
        levels=18,
        colors="black",
        linewidths=0.22,
        alpha=0.22,
        zorder=1,
    )

    _draw_static_constraints(ax, estimator)
    _draw_forecast_storms(ax, estimator, t_s)
    _draw_random_hazards(ax, disruptions, t_s)
    _draw_mission_annotations(ax, mission_map)
    _draw_executed_path(ax, result)

    min_x, max_x, min_y, max_y = bounds
    ax.set_xlim(min_x / 1000.0, max_x / 1000.0)
    ax.set_ylim(min_y / 1000.0, max_y / 1000.0)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


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
    full_map: bool = False,
) -> None:
    bounds = _map_bounds(estimator) if full_map else _local_bounds(mission_map, result)
    sampler = _combined_wind_sampler(estimator, disruptions) if include_random_layer else _predictable_wind_sampler(estimator)
    X, Y, U, V, S = _sample_wind_grid(bounds, sampler, t_s=t_s, grid_size=34 if full_map else 24)

    fig, ax = plt.subplots(figsize=(10, 8))
    if not full_map:
        _draw_terrain_background(ax, estimator, bounds)
    speed = ax.contourf(X / 1000.0, Y / 1000.0, S, levels=24, cmap="turbo", alpha=0.78, zorder=2)
    fig.colorbar(speed, ax=ax, label="Wind speed (m/s)")
    ax.streamplot(
        X / 1000.0,
        Y / 1000.0,
        U,
        V,
        density=1.05,
        color="white",
        linewidth=0.85,
        arrowsize=1.1,
        zorder=3,
    )

    if include_random_layer:
        _draw_random_hazards(ax, disruptions, t_s)
    _draw_static_constraints(ax, estimator)
    _draw_forecast_storms(ax, estimator, t_s)
    _draw_mission_annotations(ax, mission_map)
    if include_trajectory:
        _draw_executed_path(ax, result)

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


def plot_mission_elevation_profile(
    *,
    estimator: StateEstimator,
    mission_map: MissionMap,
    result: TaskExecutionResult,
    output_path: str | Path,
    title: str = "V3.0 Task Mission Elevation Profile",
) -> None:
    path = _path_array(result)
    fig, ax = plt.subplots(figsize=(12, 6))
    if path is None:
        ax.text(0.5, 0.5, "No trajectory data available", ha="center", va="center", transform=ax.transAxes)
    else:
        dx = np.diff(path[:, 0])
        dy = np.diff(path[:, 1])
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
        total_dist = max(float(dist[-1]), 1.0)
        total_time = max(float(result.total_time_s), 1.0)
        times = dist / total_dist * total_time
        terrain_z = np.asarray([estimator.map.get_altitude(float(x), float(y)) for x, y in path[:, :2]], dtype=float)
        flight_z = path[:, 2].copy()
        invalid_z = flight_z <= 0.0
        flight_z[invalid_z] = terrain_z[invalid_z] + float(estimator.config.takeoff_altitude_agl)

        ax.fill_between(times, 0.0, terrain_z, color="gray", alpha=0.35, label="Terrain under route")
        ax.plot(times, flight_z, color="lime", linewidth=3.0, label="Executed altitude")
        ax.plot(
            times,
            terrain_z + float(estimator.config.takeoff_altitude_agl),
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            label=f"+{estimator.config.takeoff_altitude_agl:.0f}m AGL clearance",
        )
        for event in result.events:
            if event.kind in {"inspection_done", "charging_done"}:
                ax.axvline(float(event.time_s), color="black", alpha=0.18, linewidth=0.8)
                ax.text(
                    float(event.time_s),
                    max(flight_z) + 40.0,
                    str(event.target_id),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    alpha=0.75,
                )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Absolute altitude (m)")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
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
    _draw_terrain_background(ax, estimator, bounds)
    speed = ax.contourf(X / 1000.0, Y / 1000.0, S, levels=20, cmap="viridis", alpha=0.58, zorder=2)
    fig.colorbar(speed, ax=ax, label="Wind speed (m/s)")
    quiver = ax.quiver(X / 1000.0, Y / 1000.0, U, V, color="white", alpha=0.80, scale=260, zorder=3)
    _draw_static_constraints(ax, estimator)
    forecast_storm_patches = []
    forecast_storm_labels = []
    forecast_storm_arrows = []
    storm_manager = getattr(getattr(estimator, "wind", None), "storm_manager", None)
    if storm_manager is not None and getattr(estimator.config, "enable_storms", False):
        storm_count = int(max(0, getattr(estimator.config, "storm_count", 0)))
        for idx in range(storm_count):
            patch = plt.Circle(
                (0.0, 0.0),
                0.0,
                facecolor="royalblue",
                edgecolor="midnightblue",
                linewidth=1.4,
                alpha=0.16,
                label="Forecast moving storm" if idx == 0 else None,
                visible=False,
                zorder=4,
            )
            ax.add_patch(patch)
            label = ax.text(
                0.0,
                0.0,
                "Forecast\nstorm",
                color="midnightblue",
                fontsize=8,
                ha="center",
                va="center",
                visible=False,
                zorder=7,
            )
            arrow = ax.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="midnightblue", linestyle="--", linewidth=1.2, alpha=0.85),
                visible=False,
                zorder=6,
            )
            forecast_storm_patches.append(patch)
            forecast_storm_labels.append(label)
            forecast_storm_arrows.append(arrow)
    random_hazard_patches = []
    random_hazard_labels = []
    if include_random_layer and disruptions is not None:
        storm = disruptions.destructive_storm
        halo_patch = plt.Circle(
            (0.0, 0.0),
            0.0,
            facecolor="mediumpurple",
            edgecolor="indigo",
            linewidth=1.4,
            alpha=0.18,
            label="Random storm halo",
            visible=False,
            zorder=4,
        )
        core_patch = plt.Circle(
            (0.0, 0.0),
            0.0,
            facecolor="crimson",
            edgecolor="darkred",
            linewidth=1.5,
            alpha=0.34,
            label="Destructive storm core",
            visible=False,
            zorder=5,
        )
        ax.add_patch(halo_patch)
        ax.add_patch(core_patch)
        storm_label = ax.text(
            0.0,
            0.0,
            "Random\nstorm",
            color="indigo",
            fontsize=8,
            ha="center",
            va="center",
            visible=False,
            zorder=7,
        )
        random_hazard_patches.extend([halo_patch, core_patch])
        random_hazard_labels.append(storm_label)
        for idx, region in enumerate(disruptions.local_wind_regions):
            patch = plt.Circle(
                (region.center_xy[0] / 1000.0, region.center_xy[1] / 1000.0),
                region.radius_m / 1000.0,
                facecolor="cyan",
                edgecolor="deepskyblue",
                linewidth=1.1,
                alpha=0.16,
                label="Random wind region" if idx == 0 else None,
                zorder=4,
            )
            ax.add_patch(patch)
    if include_random_layer:
        pass
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

        if storm_manager is not None:
            active_storms = storm_manager.get_active_storms(float(t_s))
            for idx, patch in enumerate(forecast_storm_patches):
                if idx < len(active_storms):
                    storm = active_storms[idx]
                    center = storm.center_at(float(t_s))
                    cx_km = float(center[0]) / 1000.0
                    cy_km = float(center[1]) / 1000.0
                    vx_km = float(storm.velocity_xy[0]) / 1000.0
                    vy_km = float(storm.velocity_xy[1]) / 1000.0
                    patch.set_center((cx_km, cy_km))
                    patch.set_radius(float(storm.radius_m) / 1000.0)
                    patch.set_visible(True)
                    forecast_storm_labels[idx].set_position((cx_km, cy_km))
                    forecast_storm_labels[idx].set_visible(True)
                    forecast_storm_arrows[idx].xy = (cx_km + vx_km * 300.0, cy_km + vy_km * 300.0)
                    forecast_storm_arrows[idx].set_position((cx_km, cy_km))
                    forecast_storm_arrows[idx].set_visible(True)
                else:
                    patch.set_visible(False)
                    forecast_storm_labels[idx].set_visible(False)
                    forecast_storm_arrows[idx].set_visible(False)

        if include_random_layer and disruptions is not None and len(random_hazard_patches) >= 2:
            storm = disruptions.destructive_storm
            active = bool(storm.lifetime_s > 0.0 and storm.is_active(float(t_s)))
            if active:
                center = storm.center_at(float(t_s))
                cx_km = float(center[0]) / 1000.0
                cy_km = float(center[1]) / 1000.0
                random_hazard_patches[0].set_center((cx_km, cy_km))
                random_hazard_patches[0].set_radius(float(storm.halo_radius_m) / 1000.0)
                random_hazard_patches[0].set_visible(True)
                random_hazard_patches[1].set_center((cx_km, cy_km))
                random_hazard_patches[1].set_radius(float(storm.core_radius_m) / 1000.0)
                random_hazard_patches[1].set_visible(True)
                if random_hazard_labels:
                    random_hazard_labels[0].set_position((cx_km, cy_km))
                    random_hazard_labels[0].set_visible(True)
            else:
                random_hazard_patches[0].set_visible(False)
                random_hazard_patches[1].set_visible(False)
                if random_hazard_labels:
                    random_hazard_labels[0].set_visible(False)

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
