from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v30.experiments.run_task_map_demo import build_demo_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a v3.0 mission map JSON template.")
    parser.add_argument("--output", default="v30/examples/mission_map_template.generated.json")
    args = parser.parse_args()

    mission_map = build_demo_map(start_xy=(0.0, 0.0), distance_scale=1.0)
    mission_map.name = "v30_template_generated"
    mission_map.save_json(args.output)
    print(f"saved_template: {args.output}")


if __name__ == "__main__":
    main()
