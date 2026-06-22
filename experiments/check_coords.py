from __future__ import annotations

from configs.config import SimulationConfig
from environment.map_manager import MapManager


def main() -> None:
    config = SimulationConfig()
    map_manager = MapManager(config)

    print("bounds:", map_manager.get_bounds())
    print("center:", (0.0, 0.0))

    try:
        print("alt(center):", map_manager.get_altitude(0.0, 0.0))
    except Exception as exc:
        print("alt(center) error:", exc)


if __name__ == "__main__":
    main()
