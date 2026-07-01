from __future__ import annotations

import numpy as np

from configs.config import SimulationConfig


def expert_candidate_actions(
    config: SimulationConfig,
    *,
    emergency: bool = False,
    mild: bool = False,
) -> list[np.ndarray]:
    if emergency:
        headings = config.v25_expert_emergency_heading_actions
        speeds = config.v25_expert_emergency_speed_actions
        agls = config.v25_expert_emergency_agl_actions
    elif mild:
        headings = config.v25_expert_mild_heading_actions
        speeds = config.v25_expert_mild_speed_actions
        agls = config.v25_expert_mild_agl_actions
    else:
        headings = config.v25_expert_heading_actions
        speeds = config.v25_expert_speed_actions
        agls = config.v25_expert_agl_actions
    return [
        np.array([heading_action, speed_action, agl_action], dtype=float)
        for heading_action in headings
        for speed_action in speeds
        for agl_action in agls
    ]
