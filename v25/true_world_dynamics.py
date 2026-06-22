from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from configs.config import SimulationConfig


def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


@dataclass
class VehicleStateV25:
    position_xyz: np.ndarray
    heading_deg: float
    airspeed_mps: float


class TrueWorldDynamicsV25:
    """Constrained point-mass dynamics used by every v2.5 controller."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def advance(
        self,
        state: VehicleStateV25,
        commanded_heading_deg: float,
        commanded_airspeed_mps: float,
        commanded_altitude_m: float,
        true_wind_xy: np.ndarray,
        dt: float,
    ) -> dict:
        max_turn = self.config.v25_max_turn_rate_deg_s * dt
        heading_error = wrap_angle_deg(commanded_heading_deg - state.heading_deg)
        heading = wrap_angle_deg(state.heading_deg + float(np.clip(heading_error, -max_turn, max_turn)))

        max_dv = self.config.v25_max_accel_mps2 * dt
        airspeed = float(
            np.clip(
                state.airspeed_mps
                + np.clip(commanded_airspeed_mps - state.airspeed_mps, -max_dv, max_dv),
                self.config.rl_speed_min,
                self.config.rl_speed_max,
            )
        )

        rad = math.radians(heading)
        air_velocity_xy = np.array([math.cos(rad), math.sin(rad)], dtype=float) * airspeed
        ground_velocity_xy = air_velocity_xy + np.asarray(true_wind_xy, dtype=float)

        dz_command = commanded_altitude_m - float(state.position_xyz[2])
        vertical_speed = float(
            np.clip(
                dz_command / max(dt, 1e-6),
                -self.config.v25_max_descent_rate_mps,
                self.config.v25_max_climb_rate_mps,
            )
        )
        ground_velocity_xyz = np.array(
            [ground_velocity_xy[0], ground_velocity_xy[1], vertical_speed],
            dtype=float,
        )
        new_position = np.asarray(state.position_xyz, dtype=float) + ground_velocity_xyz * dt

        return {
            "position_xyz": new_position,
            "heading_deg": heading,
            "airspeed_mps": airspeed,
            "air_velocity_xy": air_velocity_xy,
            "ground_velocity_xyz": ground_velocity_xyz,
        }
