#!/usr/bin/env python3
"""
Unified launcher for legacy planning, swarm matrix experiments, showcase demos,
ablation experiments, and UI.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
ABLATION_EXPERIMENTS = ["control", "observation", "topology", "shield", "planner_time"]

# Unified UI endpoint used by both launcher and desktop shell.
UI_HOST = "localhost"
UI_PORT = 8502
UI_URL = f"http://{UI_HOST}:{UI_PORT}"


# Hint for packagers: include the Streamlit UI module dependency graph.
# This branch is never executed at runtime.
if False:  # pragma: no cover
    import ui.drone_ui  # noqa: F401


def is_frozen() -> bool:
    """Return True when running from a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def resolve_ui_script() -> Path:
    """Resolve Streamlit UI script for both source and frozen modes."""
    candidates = [
        PROJECT_ROOT / "ui" / "drone_ui.py",
        Path.cwd() / "ui" / "drone_ui.py",
    ]
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        candidates.insert(0, Path(getattr(sys, "_MEIPASS")) / "ui" / "drone_ui.py")

    for path in candidates:
        if path.exists():
            return path.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Cannot locate ui/drone_ui.py.\nSearched:\n{searched}")


def run_command(cmd: list[str], description: str) -> bool:
    """Run one command and return success status."""
    print(f"[RUN] {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
        return False
    except Exception as exc:
        print(f"[ERROR] Failed to execute command: {exc}")
        return False


def run_ui() -> bool:
    """
    Launch Streamlit UI.

    - Source mode: shell out to `python -m streamlit run ...` (keeps existing behavior).
    - Frozen mode: run Streamlit bootstrap in-process (sys.executable is not a Python interpreter).
    """
    ui_script = resolve_ui_script()

    if not is_frozen():
        return run_command(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ui_script),
                "--server.address",
                UI_HOST,
                "--server.port",
                str(UI_PORT),
            ],
            f"Launch swarm command center UI ({UI_URL})",
        )

    print(f"[RUN] Launch swarm command center UI (frozen mode) at {UI_URL}")
    print(f"Script: {ui_script}")
    print("-" * 60)
    try:
        from streamlit.web import bootstrap

        flag_options = {
            "server.headless": True,
            "server.address": UI_HOST,
            "server.port": UI_PORT,
            "browser.gatherUsageStats": False,
            "browser.serverAddress": UI_HOST,
            "server.fileWatcherType": "none",
        }
        bootstrap.run(str(ui_script), False, [], flag_options)
        return True
    except KeyboardInterrupt:
        print("\n[WARN] UI interrupted by user")
        return False
    except Exception as exc:
        print(f"[ERROR] Failed to launch UI in frozen mode: {exc}")
        return False


def parse_parallel_ablation_args(extra_args: Sequence[str]):
    parent_output_root = PROJECT_ROOT / "results"
    selected_experiments = list(ABLATION_EXPERIMENTS)
    forwarded_args: list[str] = []

    idx = 0
    while idx < len(extra_args):
        token = extra_args[idx]
        if token == "--output-root" and idx + 1 < len(extra_args):
            parent_output_root = Path(extra_args[idx + 1])
            idx += 2
            continue
        if token == "--exp":
            idx += 2 if idx + 1 < len(extra_args) else 1
            continue
        if token == "--parallel-exps":
            idx += 1
            selected_experiments = []
            while idx < len(extra_args) and not str(extra_args[idx]).startswith("--"):
                parts = [part.strip() for part in str(extra_args[idx]).split(",") if part.strip()]
                selected_experiments.extend(parts)
                idx += 1
            if not selected_experiments:
                selected_experiments = list(ABLATION_EXPERIMENTS)
            continue
        forwarded_args.append(str(token))
        idx += 1

    invalid = [exp for exp in selected_experiments if exp not in ABLATION_EXPERIMENTS]
    if invalid:
        raise ValueError(
            f"Unsupported experiments for --parallel-exps: {', '.join(invalid)}. "
            f"Valid options: {', '.join(ABLATION_EXPERIMENTS)}"
        )
    return parent_output_root, selected_experiments, forwarded_args


def run_parallel_ablation(extra_args: Sequence[str]) -> bool:
    parent_output_root, selected_experiments, forwarded_args = parse_parallel_ablation_args(extra_args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = parent_output_root / f"ablation_parallel_{timestamp}"
    logs_root = bundle_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    print("[RUN] Launch parallel ablation bundle")
    print(f"Bundle root: {bundle_root}")
    print(f"Experiments: {', '.join(selected_experiments)}")
    print("-" * 60)

    processes: list[dict[str, object]] = []
    try:
        for experiment in selected_experiments:
            experiment_output_root = bundle_root / experiment
            log_path = logs_root / f"{experiment}.log"
            log_handle = open(log_path, "w", encoding="utf-8")
            cmd = [
                sys.executable,
                "analysis/run_ablation_suite.py",
                "--exp",
                experiment,
                "--output-root",
                str(experiment_output_root),
                *forwarded_args,
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(
                {
                    "experiment": experiment,
                    "process": proc,
                    "log_path": log_path,
                    "log_handle": log_handle,
                }
            )
            print(f"[LAUNCH] {experiment} | pid={proc.pid} | log={log_path}")

        all_success = True
        for item in processes:
            proc = item["process"]  # type: ignore[assignment]
            return_code = proc.wait()  # type: ignore[attr-defined]
            item["log_handle"].close()  # type: ignore[union-attr]
            if return_code == 0:
                print(f"[OK] {item['experiment']} finished | log={item['log_path']}")
            else:
                print(f"[ERROR] {item['experiment']} failed (code={return_code}) | log={item['log_path']}")
                all_success = False
        print(f"[DONE] Parallel ablation bundle complete: {bundle_root}")
        return all_success
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user, terminating child jobs...")
        for item in processes:
            proc = item["process"]  # type: ignore[assignment]
            if proc.poll() is None:  # type: ignore[attr-defined]
                proc.terminate()  # type: ignore[attr-defined]
        for item in processes:
            item["log_handle"].close()  # type: ignore[union-attr]
        return False


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Drone planning project launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py ui
  python launcher.py desktop
  python launcher.py swarm
  python launcher.py astar25
  python launcher.py rl25
  python launcher.py showcase
  python launcher.py ablation --exp all
  python launcher.py ablation --exp control --control-seeds 10
  python launcher.py ablation_parallel --parallel-exps control topology shield
  python launcher.py sim
  python launcher.py rl --timesteps 300000 --envs 4
  python launcher.py test
  python launcher.py install
        """.strip(),
    )
    parser.add_argument(
        "mode",
        choices=["sim", "rl", "rl25", "ui", "desktop", "swarm", "astar25", "showcase", "ablation", "ablation_parallel", "test", "install"],
        help="launch target",
    )
    parser.add_argument("--timesteps", type=int, default=500000, help="RL training timesteps")
    parser.add_argument("--envs", type=int, default=4, help="parallel RL environments")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="additional arguments forwarded to the selected target script",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launcher entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    success = False
    if args.mode == "install":
        success = run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            "Install project dependencies",
        )
    elif args.mode == "sim":
        success = run_command([sys.executable, "main.py"], "Run legacy single-drone simulation")
    elif args.mode == "rl":
        success = run_command(
            [sys.executable, "rl_training/train_ppo.py"],
            f"Run RL training (timesteps={args.timesteps}, envs={args.envs})",
        )
    elif args.mode == "rl25":
        success = run_command(
            [sys.executable, "v25/train_rl_disruptive.py", *args.extra],
            "Run RL v2.5 (A* teacher + disruptive execution)",
        )
    elif args.mode == "ui":
        success = run_ui()
    elif args.mode == "desktop":
        success = run_command(
            [sys.executable, "desktop_launcher.py"],
            "Launch desktop UI shell (pywebview + Streamlit)",
        )
    elif args.mode == "swarm":
        success = run_command(
            [sys.executable, "experiments/test_swarm_standalone.py"],
            "Run four-case swarm matrix (A*/RL x clean/gust)",
        )
    elif args.mode == "astar25":
        success = run_command(
            [sys.executable, "v25/experiments/single_astar_disruptive.py", *args.extra],
            "Run single-drone A* with disruptive execution disturbances",
        )
    elif args.mode == "showcase":
        success = run_command(
            [sys.executable, "analysis/showcase_support_flank_screen.py"],
            "Run Support flank-screen showcase",
        )
    elif args.mode == "ablation":
        cmd = [sys.executable, "analysis/run_ablation_suite.py", *args.extra]
        success = run_command(
            cmd,
            "Run ablation suite",
        )
    elif args.mode == "ablation_parallel":
        success = run_parallel_ablation(args.extra)
    elif args.mode == "test":
        success = run_command([sys.executable, "-m", "pytest", "tests/", "-v"], "Run pytest suite")

    print("\n" + "=" * 60)
    if success:
        print("[OK] Task finished successfully")
        return 0

    print("[ERROR] Task failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
