from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def world_vector_to_body(vec_xy: np.ndarray, heading_deg: float) -> np.ndarray:
    heading_rad = math.radians(float(heading_deg))
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)
    left = np.array([-math.sin(heading_rad), math.cos(heading_rad)], dtype=float)
    vec = np.asarray(vec_xy, dtype=float)
    return np.array([float(np.dot(vec, forward)), float(np.dot(vec, left))], dtype=float)


def circle_oracle_sample_points(
    *,
    origin_xy: np.ndarray,
    heading_deg: float,
    radius_m: float,
    sample_count: int,
) -> list[np.ndarray]:
    origin = np.asarray(origin_xy, dtype=float)
    heading_rad = math.radians(float(heading_deg))
    count = int(sample_count)
    if count <= 0:
        return []

    sample_points = [origin]
    front_angles_deg = (
        -75.0, -60.0, -45.0, -32.0, -22.0, -12.0, -6.0,
        0.0,
        6.0, 12.0, 22.0, 32.0, 45.0, 60.0, 75.0,
    )
    front_radii = (0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.92, 1.00)
    for radius_scale in front_radii:
        for rel_angle_deg in front_angles_deg:
            if len(sample_points) >= count:
                break
            angle = heading_rad + math.radians(rel_angle_deg)
            sample_points.append(
                origin + radius_scale * float(radius_m) * np.array([math.cos(angle), math.sin(angle)], dtype=float)
            )
        if len(sample_points) >= count:
            break

    rings = (0.25, 0.50, 0.75, 1.00)
    per_ring = max(8, int(math.ceil(max(count - len(sample_points), 1) / len(rings))))
    for ring in rings:
        for idx in range(per_ring):
            if len(sample_points) >= count:
                break
            angle = 2.0 * math.pi * idx / per_ring
            sample_points.append(origin + ring * float(radius_m) * np.array([math.cos(angle), math.sin(angle)]))
        if len(sample_points) >= count:
            break

    return sample_points


def circle_oracle_features(
    *,
    origin_xy: np.ndarray,
    heading_deg: float,
    current_time_s: float,
    sample_points: Sequence[np.ndarray],
    radius_m: float,
    max_wind_speed: float,
    disturbance_at: Callable[[float, float, float, np.ndarray], tuple[np.ndarray, float]],
    risk_bonus_at: Callable[[float, float, float], float],
    storm_danger_at: Callable[[float, float, float], float],
) -> list[float]:
    origin = np.asarray(origin_xy, dtype=float)
    heading_rad = math.radians(float(heading_deg))
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)

    dangers = []
    winds = []
    forward_dangers = []
    max_wind_mag = 0.0
    max_wind_vec = np.zeros(2, dtype=float)
    nearest_danger_dist = float(radius_m)
    nearest_danger_vec = np.zeros(2, dtype=float)
    weighted_danger_vec = np.zeros(2, dtype=float)
    weighted_danger_sum = 0.0

    for point in sample_points:
        point_arr = np.asarray(point, dtype=float)
        offset = np.asarray(point_arr - origin, dtype=float)
        dist = float(np.linalg.norm(offset))
        travel_dir = offset / dist if dist > 1e-6 else forward
        wind, _ = disturbance_at(float(point_arr[0]), float(point_arr[1]), float(current_time_s), travel_dir)
        danger = float(risk_bonus_at(float(point_arr[0]), float(point_arr[1]), float(current_time_s)))
        storm_danger = float(storm_danger_at(float(point_arr[0]), float(point_arr[1]), float(current_time_s)))
        danger = max(danger, storm_danger)

        dangers.append(danger)
        winds.append(np.asarray(wind, dtype=float))
        if dist > 1e-6 and np.dot(offset / dist, forward) > math.cos(math.radians(45.0)):
            forward_dangers.append(danger)

        wind_mag = float(np.linalg.norm(wind))
        if wind_mag > max_wind_mag:
            max_wind_mag = wind_mag
            max_wind_vec = np.asarray(wind, dtype=float)
        if danger > 1e-6:
            if dist < nearest_danger_dist:
                nearest_danger_dist = dist
                nearest_danger_vec = offset
            weighted_danger_vec += offset * danger
            weighted_danger_sum += danger

    mean_wind = np.mean(np.asarray(winds, dtype=float), axis=0) if winds else np.zeros(2, dtype=float)
    wind_scale = max(float(max_wind_speed), 1e-6)
    mean_wind_body = world_vector_to_body(mean_wind, heading_deg) / wind_scale
    max_wind_body = world_vector_to_body(max_wind_vec, heading_deg) / wind_scale

    if np.linalg.norm(nearest_danger_vec) > 1e-6:
        nearest_dir_body = world_vector_to_body(nearest_danger_vec / np.linalg.norm(nearest_danger_vec), heading_deg)
    else:
        nearest_dir_body = np.zeros(2, dtype=float)
    if weighted_danger_sum > 1e-6 and np.linalg.norm(weighted_danger_vec) > 1e-6:
        centroid_vec = weighted_danger_vec / weighted_danger_sum
        centroid_dir_body = world_vector_to_body(centroid_vec / np.linalg.norm(centroid_vec), heading_deg)
    else:
        centroid_dir_body = np.zeros(2, dtype=float)

    max_danger = float(max(dangers)) if dangers else 0.0
    mean_danger = float(np.mean(dangers)) if dangers else 0.0
    nearest_closeness = 1.0 - min(nearest_danger_dist / max(float(radius_m), 1e-6), 1.0)
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


def sector_radar_features(
    *,
    origin_xy: np.ndarray,
    heading_deg: float,
    current_time_s: float,
    sector_count: int,
    radius_m: float,
    max_wind_speed: float,
    wind_noise_std_mps: float,
    risk_noise_std: float,
    sensor_rng: np.random.Generator,
    radar_sector: Callable[..., tuple[float, float, float, float]],
) -> list[float]:
    radar_values = []
    for sector_idx in range(int(sector_count)):
        wind_mag, wind_dir_sin, danger, danger_closeness = radar_sector(
            origin_xy=np.asarray(origin_xy, dtype=float),
            heading_deg=float(heading_deg),
            sector_idx=sector_idx,
            sector_count=int(sector_count),
            radius_m=float(radius_m),
            t_s=float(current_time_s),
        )
        noisy_wind = wind_mag + float(sensor_rng.normal(0.0, float(wind_noise_std_mps)))
        noisy_danger = danger + float(sensor_rng.normal(0.0, float(risk_noise_std)))
        radar_values.extend(
            [
                float(np.clip(noisy_wind / max(float(max_wind_speed), 1e-6), 0.0, 2.0)),
                float(np.clip(wind_dir_sin, -1.0, 1.0)),
                float(np.clip(noisy_danger, 0.0, 1.0)),
                float(np.clip(danger_closeness, 0.0, 1.0)),
            ]
        )
    return radar_values


def compose_sensor_features(
    *,
    measured_residual_wind: np.ndarray,
    last_measured_residual_wind: np.ndarray,
    tracking_velocity_error: np.ndarray,
    local_view_values: Sequence[float],
    max_wind_speed: float,
    rl_speed_max: float,
) -> np.ndarray:
    measured = np.asarray(measured_residual_wind, dtype=float)
    residual_delta = measured - np.asarray(last_measured_residual_wind, dtype=float)
    tracking_error = np.asarray(tracking_velocity_error, dtype=float)
    wind_scale = max(float(max_wind_speed), 1e-6)
    speed_scale = max(float(rl_speed_max), 1e-6)

    return np.asarray(
        [
            measured[0] / wind_scale,
            measured[1] / wind_scale,
            residual_delta[0] / wind_scale,
            residual_delta[1] / wind_scale,
            tracking_error[0] / speed_scale,
            tracking_error[1] / speed_scale,
            *local_view_values,
        ],
        dtype=np.float32,
    )
