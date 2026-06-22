from __future__ import annotations

import rl_env.drone_env
from configs.config import SimulationConfig
from rl_env.drone_env import GuidedDroneEnv


def main() -> None:
    print("loaded file:", rl_env.drone_env.__file__)

    config = SimulationConfig()
    env = GuidedDroneEnv(config)

    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)

    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("info:", info)

    for i in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={i}, reward={reward:.3f}, "
            f"terminated={terminated}, truncated={truncated}, info={info}"
        )
        if terminated or truncated:
            print("episode ended, reset")
            obs, info = env.reset()
            print("new info:", info)


if __name__ == "__main__":
    main()
