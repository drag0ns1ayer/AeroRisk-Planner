from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from configs.config import SimulationConfig

Point2D = Tuple[float, float]


def _rotate(vec_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([c * vec_xy[0] - s * vec_xy[1], s * vec_xy[0] + c * vec_xy[1]], dtype=float)


def _unit(vec_xy: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec_xy))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float) if fallback is None else np.asarray(fallback, dtype=float)
    return np.asarray(vec_xy, dtype=float) / norm


@dataclass
class LocalWindRegion:
    center_xy: np.ndarray
    direction_xy: np.ndarray
    radius_m: float
    start_time_s: float
    end_time_s: float
    peak_speed_mps: float
    risk_bonus_max: float

    def is_active(self, t_s: float) -> bool:
        return self.start_time_s <= t_s <= self.end_time_s

    def _weight(self, x: float, y: float, t_s: float) -> float:
        if not self.is_active(t_s):
            return 0.0
        dist = float(np.linalg.norm(np.array([x, y], dtype=float) - self.center_xy))
        if dist > self.radius_m:
            return 0.0
        spatial = max(0.0, 1.0 - (dist / self.radius_m) ** 2)
        phase = (t_s - self.start_time_s) / max(self.end_time_s - self.start_time_s, 1e-6)
        temporal = math.sin(math.pi * min(max(phase, 0.0), 1.0))
        return float(spatial * temporal)

    def wind_at(self, x: float, y: float, t_s: float, travel_dir_xy: np.ndarray | None = None) -> np.ndarray:
        return _unit(self.direction_xy) * (self.peak_speed_mps * self._weight(x, y, t_s))

    def severity(self, x: float, y: float, t_s: float) -> float:
        return self._weight(x, y, t_s)

    def risk_bonus(self, x: float, y: float, t_s: float) -> float:
        return self.risk_bonus_max * self.severity(x, y, t_s)


@dataclass
class DestructiveStormDisturbance:
    center_xy_t0: np.ndarray
    pre_velocity_xy: np.ndarray
    post_velocity_xy: np.ndarray
    trigger_time_s: float
    early_appearance_s: float
    lifetime_s: float
    radius_m: float
    halo_radius_m: float
    core_radius_m: float
    halo_danger_floor: float
    strength_mps: float
    outer_risk_bonus: float
    core_risk_bonus: float
    core_crash: bool

    def _start_time(self) -> float:
        return self.trigger_time_s - self.early_appearance_s

    def is_active(self, t_s: float) -> bool:
        start = self._start_time()
        return start <= t_s <= (start + self.lifetime_s)

    def center_at(self, t_s: float):
        if not self.is_active(t_s):
            return None
        start = self._start_time()
        if t_s <= self.trigger_time_s:
            return self.center_xy_t0 + self.pre_velocity_xy * (t_s - start)
        pivot = self.center_xy_t0 + self.pre_velocity_xy * (self.trigger_time_s - start)
        return pivot + self.post_velocity_xy * (t_s - self.trigger_time_s)

    def _distance(self, x: float, y: float, t_s: float) -> float | None:
        center = self.center_at(t_s)
        if center is None:
            return None
        return float(np.linalg.norm(np.array([x, y], dtype=float) - center))

    def severity(self, x: float, y: float, t_s: float) -> float:
        dist = self._distance(x, y, t_s)
        if dist is None or dist > self.radius_m:
            return 0.0
        return float(max(0.0, 1.0 - (dist / self.radius_m) ** 2))

    def danger_at(self, x: float, y: float, t_s: float) -> float:
        dist = self._distance(x, y, t_s)
        if dist is None or dist > self.radius_m:
            return 0.0
        if dist <= self.core_radius_m:
            return 1.0
        halo_radius = min(max(self.halo_radius_m, self.core_radius_m), self.radius_m)
        halo_floor = float(np.clip(self.halo_danger_floor, 0.0, 1.0))
        if dist <= halo_radius:
            shell = (dist - self.core_radius_m) / max(halo_radius - self.core_radius_m, 1e-6)
            return float(halo_floor + (1.0 - halo_floor) * max(0.0, 1.0 - shell))
        outer_shell = (dist - halo_radius) / max(self.radius_m - halo_radius, 1e-6)
        return float(halo_floor * max(0.0, 1.0 - outer_shell))

    def core_hit(self, x: float, y: float, t_s: float) -> bool:
        dist = self._distance(x, y, t_s)
        return bool(self.core_crash and dist is not None and dist <= self.core_radius_m)

    def wind_at(self, x: float, y: float, t_s: float) -> np.ndarray:
        center = self.center_at(t_s)
        if center is None:
            return np.zeros(2, dtype=float)
        dist = float(np.linalg.norm(np.array([x, y], dtype=float) - center))
        if dist > self.radius_m * 1.5:
            return np.zeros(2, dtype=float)
        sigma = max(self.radius_m / 2.6, 1.0)
        spatial_weight = math.exp(-0.5 * (dist / sigma) ** 2)
        velocity = self.pre_velocity_xy if t_s <= self.trigger_time_s else self.post_velocity_xy
        return _unit(velocity) * (self.strength_mps * spatial_weight)

    def risk_bonus(self, x: float, y: float, t_s: float) -> float:
        danger = self.danger_at(x, y, t_s)
        if danger <= 0.0:
            return 0.0
        return float(self.outer_risk_bonus * danger + (self.core_risk_bonus - self.outer_risk_bonus) * (danger ** 2))


class DisruptionLayerV25:
    def __init__(
        self,
        destructive_storm: DestructiveStormDisturbance,
        local_wind_regions: list[LocalWindRegion],
    ) -> None:
        self.destructive_storm = destructive_storm
        self.local_wind_regions = local_wind_regions
        # Compatibility aliases for older diagnostics.
        self.storm_mutation = destructive_storm
        self.headwind_pulse = local_wind_regions[0]

    def disturbance_at(
        self,
        x: float,
        y: float,
        t_s: float,
        travel_dir_xy: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        wind = self.destructive_storm.wind_at(x, y, t_s)
        wind_severity = 0.7 * self.destructive_storm.severity(x, y, t_s)
        danger = self.destructive_storm.danger_at(x, y, t_s)
        for region in self.local_wind_regions:
            wind += region.wind_at(x, y, t_s, travel_dir_xy)
            wind_severity += 0.35 * region.severity(x, y, t_s)
        severity = min(1.0, max(wind_severity, danger))
        return wind, float(severity)

    def risk_bonus_at(self, x: float, y: float, t_s: float) -> float:
        risk = self.destructive_storm.risk_bonus(x, y, t_s)
        for region in self.local_wind_regions:
            risk += region.risk_bonus(x, y, t_s)
        return float(min(1.0, risk))

    def risk_bonus(self, severity: float) -> float:
        return float(0.30 * severity)

    def core_hit(self, x: float, y: float, t_s: float) -> bool:
        return self.destructive_storm.core_hit(x, y, t_s)

    def radar_sector(
        self,
        origin_xy: np.ndarray,
        heading_deg: float,
        sector_idx: int,
        sector_count: int,
        radius_m: float,
        t_s: float,
    ) -> tuple[float, float, float, float]:
        sector_width = 360.0 / sector_count
        center_rel = -180.0 + sector_width * (sector_idx + 0.5)
        angles = [center_rel - 0.35 * sector_width, center_rel, center_rel + 0.35 * sector_width]
        radii = [0.35 * radius_m, 0.70 * radius_m, radius_m]
        max_wind = 0.0
        wind_vec_at_max = np.zeros(2, dtype=float)
        max_danger = 0.0
        nearest_danger = radius_m
        for rel_angle in angles:
            rad = math.radians(heading_deg + rel_angle)
            direction = np.array([math.cos(rad), math.sin(rad)], dtype=float)
            for radius in radii:
                p = origin_xy + direction * radius
                wind, _ = self.disturbance_at(float(p[0]), float(p[1]), t_s, direction)
                wind_mag = float(np.linalg.norm(wind))
                danger = self.destructive_storm.danger_at(float(p[0]), float(p[1]), t_s)
                if wind_mag > max_wind:
                    max_wind = wind_mag
                    wind_vec_at_max = wind
                if danger > max_danger:
                    max_danger = danger
                if danger > 1e-6:
                    nearest_danger = min(nearest_danger, radius)
        wind_dir = math.atan2(wind_vec_at_max[1], wind_vec_at_max[0]) - math.radians(heading_deg)
        return (
            max_wind,
            math.sin(wind_dir),
            max_danger,
            1.0 - min(nearest_danger / max(radius_m, 1e-6), 1.0),
        )


def build_disruption_layer_v25(
    start_xy: Point2D,
    goal_xy: Point2D,
    nominal_speed_mps: float = 12.0,
    config: SimulationConfig | None = None,
    seed: int | None = None,
) -> DisruptionLayerV25:
    cfg = config or SimulationConfig()
    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    route_vec = goal - start
    route_len = float(np.linalg.norm(route_vec))
    route_dir = _unit(route_vec)
    perp = np.array([-route_dir[1], route_dir[0]], dtype=float)
    nominal_time_s = route_len / max(nominal_speed_mps, 1.0)
    rng = np.random.default_rng(seed)
    stress_level = getattr(cfg, "v25_disruption_stress_level", "normal")
    if stress_level == "hard":
        storm_probability = max(float(cfg.v25_destructive_storm_probability), 0.75)
        storm_strength_scale = 1.25
        storm_radius_scale = 1.12
        core_ratio_scale = 1.12
        local_count_min = max(int(cfg.v25_local_wind_region_count_min), 4)
        local_count_max = max(int(cfg.v25_local_wind_region_count_max), 6)
        local_strength_scale = 1.35
        local_radius_scale = 1.12
        local_risk_scale = 1.25
    elif stress_level == "extreme":
        storm_probability = max(float(cfg.v25_destructive_storm_probability), 0.90)
        storm_strength_scale = 1.45
        storm_radius_scale = 1.25
        core_ratio_scale = 1.25
        local_count_min = max(int(cfg.v25_local_wind_region_count_min), 5)
        local_count_max = max(int(cfg.v25_local_wind_region_count_max), 8)
        local_strength_scale = 1.65
        local_radius_scale = 1.25
        local_risk_scale = 1.60
    elif stress_level == "fragile":
        storm_probability = max(float(cfg.v25_destructive_storm_probability), 0.85)
        storm_strength_scale = 1.30
        storm_radius_scale = 1.18
        core_ratio_scale = 1.15
        local_count_min = max(int(cfg.v25_local_wind_region_count_min), 4)
        local_count_max = max(int(cfg.v25_local_wind_region_count_max), 6)
        local_strength_scale = 1.55
        local_radius_scale = 1.10
        local_risk_scale = 1.35
    else:
        storm_probability = float(cfg.v25_destructive_storm_probability)
        storm_strength_scale = 1.0
        storm_radius_scale = 1.0
        core_ratio_scale = 1.0
        local_count_min = int(cfg.v25_local_wind_region_count_min)
        local_count_max = int(cfg.v25_local_wind_region_count_max)
        local_strength_scale = 1.0
        local_radius_scale = 1.0
        local_risk_scale = 1.0

    if rng.random() <= storm_probability:
        storm_trigger_s = float(np.clip(rng.uniform(0.22, 0.48) * nominal_time_s, 45.0, 480.0))
        storm_early_s = float(np.clip(rng.uniform(0.05, 0.16) * nominal_time_s, 15.0, 150.0))
        storm_lifetime_s = float(np.clip(rng.uniform(0.9, 1.6) * nominal_time_s, 300.0, 1500.0))
        storm_radius = (
            rng.uniform(*cfg.v25_destructive_storm_radius_range_m)
            * cfg.v25_disruption_storm_radius_scale
            * storm_radius_scale
        )
        core_ratio = min(0.85, rng.uniform(*cfg.v25_destructive_storm_core_ratio_range) * core_ratio_scale)
        halo_ratio = min(
            0.96,
            max(
                core_ratio + 0.08,
                rng.uniform(*cfg.v25_destructive_storm_halo_ratio_range),
            ),
        )
        if stress_level == "fragile":
            storm_spawn = (
                start
                + route_dir * rng.uniform(0.34, 0.66) * route_len
                + perp * rng.uniform(-0.10, 0.10) * max(route_len, 1000.0)
            )
        else:
            storm_spawn = (
                start
                + route_dir * rng.uniform(0.25, 0.72) * route_len
                + perp * rng.uniform(-0.45, 0.45) * max(route_len, 1600.0)
            )
        pre_v = _rotate(route_dir, rng.uniform(-40.0, 40.0)) * rng.uniform(1.5, 4.0)
        post_v = _rotate(pre_v, rng.uniform(45.0, 135.0) * rng.choice([-1.0, 1.0])) * rng.uniform(0.9, 1.6)
        destructive_storm = DestructiveStormDisturbance(
            center_xy_t0=storm_spawn,
            pre_velocity_xy=pre_v,
            post_velocity_xy=post_v,
            trigger_time_s=storm_trigger_s,
            early_appearance_s=storm_early_s,
            lifetime_s=storm_lifetime_s,
            radius_m=storm_radius,
            halo_radius_m=storm_radius * halo_ratio,
            core_radius_m=storm_radius * core_ratio,
            halo_danger_floor=float(cfg.v25_destructive_storm_halo_danger_floor),
            strength_mps=(
                rng.uniform(*cfg.v25_destructive_storm_wind_range_mps)
                * cfg.v25_disruption_storm_strength_scale
                * storm_strength_scale
            ),
            outer_risk_bonus=rng.uniform(*cfg.v25_destructive_storm_outer_risk_range),
            core_risk_bonus=rng.uniform(*cfg.v25_destructive_storm_core_risk_range),
            core_crash=bool(cfg.v25_destructive_storm_core_crash),
        )
    else:
        destructive_storm = DestructiveStormDisturbance(
            center_xy_t0=start.copy(),
            pre_velocity_xy=np.zeros(2, dtype=float),
            post_velocity_xy=np.zeros(2, dtype=float),
            trigger_time_s=-1.0,
            early_appearance_s=0.0,
            lifetime_s=0.0,
            radius_m=1.0,
            halo_radius_m=0.0,
            core_radius_m=0.0,
            halo_danger_floor=0.0,
            strength_mps=0.0,
            outer_risk_bonus=0.0,
            core_risk_bonus=0.0,
            core_crash=False,
        )

    local_count = int(rng.integers(local_count_min, local_count_max + 1))
    local_regions: list[LocalWindRegion] = []
    for _ in range(local_count):
        if stress_level == "fragile":
            center = (
                start
                + route_dir * rng.uniform(0.18, 0.88) * route_len
                + perp * rng.uniform(-0.08, 0.08) * max(route_len, 900.0)
            )
            direction = _rotate(-route_dir, rng.uniform(-28.0, 28.0))
        else:
            center = (
                start
                + route_dir * rng.uniform(0.18, 0.88) * route_len
                + perp * rng.uniform(-0.22, 0.22) * max(route_len, 1000.0)
            )
            direction = _rotate(-route_dir, rng.uniform(-65.0, 65.0))
        start_t = float(np.clip(rng.uniform(0.12, 0.70) * nominal_time_s, 30.0, 700.0))
        end_t = float(np.clip(start_t + rng.uniform(0.35, 0.95) * nominal_time_s, start_t + 180.0, 1400.0))
        local_regions.append(
            LocalWindRegion(
                center_xy=center,
                direction_xy=direction,
                radius_m=(
                    rng.uniform(*cfg.v25_local_wind_radius_range_m)
                    * cfg.v25_disruption_pulse_radius_scale
                    * local_radius_scale
                ),
                start_time_s=start_t,
                end_time_s=end_t,
                peak_speed_mps=(
                    rng.uniform(*cfg.v25_local_wind_peak_range_mps)
                    * cfg.v25_disruption_pulse_strength_scale
                    * local_strength_scale
                ),
                risk_bonus_max=cfg.v25_local_wind_risk_bonus_max * local_risk_scale,
            )
        )

    if not local_regions:
        local_regions.append(
            LocalWindRegion(start.copy(), -route_dir, 1.0, 0.0, 0.0, 0.0, 0.0)
        )
    return DisruptionLayerV25(destructive_storm=destructive_storm, local_wind_regions=local_regions)
