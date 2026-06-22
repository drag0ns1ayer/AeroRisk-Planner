from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from v25.experiments.diagnose_local_risk_history import _run_method


def _parse_seeds(values: Iterable[str]) -> List[int]:
    seeds: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    return seeds


def _classify_trace(
    rows: list[dict],
    terminated_reason: str,
    first_danger_threshold: float,
    full_danger_threshold: float,
    gradual_lead_steps: int,
    sudden_lead_steps: int,
    trend_threshold: float,
    min_trend_steps: int,
) -> dict:
    first_danger_row = None
    first_full_row = None
    peak_row = None
    positive_trend_steps = 0

    for row in rows:
        danger_signal = max(float(row["max_danger"]), float(row["forward_danger"]))
        if peak_row is None or danger_signal > max(float(peak_row["max_danger"]), float(peak_row["forward_danger"])):
            peak_row = row
        if first_danger_row is None and danger_signal >= first_danger_threshold:
            first_danger_row = row
        if first_full_row is None and danger_signal >= full_danger_threshold:
            first_full_row = row
        if (
            float(row["delta_forward_danger"]) >= trend_threshold
            or float(row["delta_max_danger"]) >= trend_threshold
            or float(row["delta_nearest_closeness"]) >= trend_threshold
        ):
            positive_trend_steps += 1

    if first_danger_row is None:
        if terminated_reason == "destructive_storm_core":
            hazard_class = "sensor_blind_core"
        elif terminated_reason == "storm_risk_too_high":
            hazard_class = "sensor_blind_risk"
        else:
            hazard_class = "no_danger"
        lead_steps = None
        lead_time_s = None
    elif first_full_row is None:
        hazard_class = "ambiguous"
        lead_steps = None
        lead_time_s = None
    else:
        lead_steps = int(first_full_row["step"]) - int(first_danger_row["step"])
        lead_time_s = float(first_full_row["time_s"]) - float(first_danger_row["time_s"])
        if lead_steps <= sudden_lead_steps:
            hazard_class = "sudden"
        elif lead_steps >= gradual_lead_steps and positive_trend_steps >= min_trend_steps:
            hazard_class = "gradual"
        else:
            hazard_class = "ambiguous"

    peak_danger = 0.0
    peak_step = None
    if peak_row is not None:
        peak_danger = max(float(peak_row["max_danger"]), float(peak_row["forward_danger"]))
        peak_step = int(peak_row["step"])

    return {
        "hazard_class": hazard_class,
        "first_danger_step": int(first_danger_row["step"]) if first_danger_row is not None else "",
        "first_danger_time_s": float(first_danger_row["time_s"]) if first_danger_row is not None else "",
        "first_full_step": int(first_full_row["step"]) if first_full_row is not None else "",
        "first_full_time_s": float(first_full_row["time_s"]) if first_full_row is not None else "",
        "lead_steps": lead_steps if lead_steps is not None else "",
        "lead_time_s": lead_time_s if lead_time_s is not None else "",
        "positive_trend_steps": int(positive_trend_steps),
        "peak_danger": float(peak_danger),
        "peak_step": peak_step if peak_step is not None else "",
    }


def _summarize(classified_rows: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in classified_rows:
        by_method[str(row["method"])].append(row)

    method_summaries = {}
    for method, rows in sorted(by_method.items()):
        n = max(len(rows), 1)
        class_counts = Counter(str(row["hazard_class"]) for row in rows)
        reason_counts = Counter(str(row["terminated_reason"]) for row in rows)
        class_success = {}
        for hazard_class in sorted(class_counts):
            sub = [row for row in rows if row["hazard_class"] == hazard_class]
            class_success[hazard_class] = {
                "episodes": len(sub),
                "success_rate": sum(1 for row in sub if bool(row["success"])) / max(len(sub), 1),
                "terminated_reasons": dict(Counter(str(row["terminated_reason"]) for row in sub)),
            }
        method_summaries[method] = {
            "episodes": len(rows),
            "success_rate": sum(1 for row in rows if bool(row["success"])) / n,
            "class_counts": dict(class_counts),
            "class_rates": {key: value / n for key, value in class_counts.items()},
            "terminated_reasons": dict(reason_counts),
            "by_class": class_success,
        }
    return method_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify v2.5 local hazards into gradual/sudden/ambiguous/no-danger cases."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed used when --seeds is omitted.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of consecutive seeds to classify.")
    parser.add_argument("--seeds", nargs="*", default=None, help="Explicit seeds or comma-separated seed lists.")
    parser.add_argument("--methods", nargs="+", choices=("astar", "expert"), default=["astar"])
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--planner-max-steps", type=int, default=15000)
    parser.add_argument("--task-sampling-trials", type=int, default=180)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--first-danger-threshold", type=float, default=0.05)
    parser.add_argument("--full-danger-threshold", type=float, default=0.95)
    parser.add_argument("--gradual-lead-steps", type=int, default=8)
    parser.add_argument("--sudden-lead-steps", type=int, default=1)
    parser.add_argument("--trend-threshold", type=float, default=0.02)
    parser.add_argument("--min-trend-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "hazard_type_classification")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seeds:
        seeds = _parse_seeds(args.seeds)
    else:
        seeds = [int(args.seed + idx) for idx in range(int(args.episodes))]

    config = SimulationConfig()
    config.curriculum_stage = int(args.curriculum_stage)
    config.max_steps = int(args.planner_max_steps)
    config.v25_disruption_stress_level = str(args.stress)
    config.v25_sensor_mode = "circle_oracle"
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"

    classified_rows = []
    for seed in seeds:
        for method in args.methods:
            print(f"[classify] seed={seed} method={method}", flush=True)
            trace_rows, run_summary = _run_method(
                config=config,
                seed=int(seed),
                method=str(method),
                max_steps=int(args.max_episode_steps),
                sampling_trials=int(args.task_sampling_trials),
            )
            classification = _classify_trace(
                trace_rows,
                terminated_reason=str(run_summary["terminated_reason"]),
                first_danger_threshold=float(args.first_danger_threshold),
                full_danger_threshold=float(args.full_danger_threshold),
                gradual_lead_steps=int(args.gradual_lead_steps),
                sudden_lead_steps=int(args.sudden_lead_steps),
                trend_threshold=float(args.trend_threshold),
                min_trend_steps=int(args.min_trend_steps),
            )
            row = {
                "seed": int(seed),
                "method": str(method),
                "success": bool(run_summary["success"]),
                "terminated_reason": str(run_summary["terminated_reason"]),
                "steps": int(run_summary["steps"]),
                **classification,
            }
            classified_rows.append(row)
            print(
                f"  class={row['hazard_class']} success={row['success']} "
                f"reason={row['terminated_reason']} lead_steps={row['lead_steps']}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"hazard_type_classification_{timestamp}.csv"
    summary_path = output_dir / f"hazard_type_summary_{timestamp}.json"

    if classified_rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(classified_rows[0].keys()))
            writer.writeheader()
            writer.writerows(classified_rows)

    payload = {
        "meta": {
            "seeds": seeds,
            "methods": list(args.methods),
            "curriculum_stage": int(args.curriculum_stage),
            "stress": str(args.stress),
            "first_danger_threshold": float(args.first_danger_threshold),
            "full_danger_threshold": float(args.full_danger_threshold),
            "gradual_lead_steps": int(args.gradual_lead_steps),
            "sudden_lead_steps": int(args.sudden_lead_steps),
            "trend_threshold": float(args.trend_threshold),
            "min_trend_steps": int(args.min_trend_steps),
        },
        "summary": _summarize(classified_rows),
        "classification_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Hazard Type Summary ===")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"classification_csv: {csv_path}")
    print(f"summary:            {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
