from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _safe_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no", ""}:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(row: dict, key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    return value in {"1", "true", "yes"}


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _rows_for(trace_rows: Iterable[dict], seed: int, method: str) -> list[dict]:
    rows = [
        row
        for row in trace_rows
        if _safe_int(row, "seed") == int(seed) and str(row.get("method", "")) == method
    ]
    rows.sort(key=lambda row: _safe_int(row, "step"))
    return rows


def _window(rows: list[dict], n: int) -> list[dict]:
    return rows[-min(len(rows), int(n)) :] if rows else []


def _mean(rows: list[dict], key: str, default: float = 0.0) -> float:
    values = [_safe_float(row, key, default) for row in rows]
    return float(mean(values)) if values else 0.0


def _max(rows: list[dict], key: str, default: float = 0.0) -> float:
    values = [_safe_float(row, key, default) for row in rows]
    return float(max(values)) if values else 0.0


def _sum(rows: list[dict], key: str, default: float = 0.0) -> float:
    return float(sum(_safe_float(row, key, default) for row in rows))


def _delta(rows: list[dict], key: str, default: float = 0.0) -> float:
    if len(rows) < 2:
        return 0.0
    return _safe_float(rows[-1], key, default) - _safe_float(rows[0], key, default)


def _classify_timeout(rows: list[dict]) -> str:
    last = _window(rows, 50)
    tail = _window(rows, 20)
    if not last:
        return "timeout_untraced"

    hazard_max = max(
        _max(last, "p_crash"),
        _max(last, "local_hazard_max_danger"),
        _max(last, "obs_max_danger"),
    )
    apas_steps = _sum(last, "apas_intervened")
    mean_speed = _mean(last, "ground_speed_mps")
    goal_progress = _safe_float(last[0], "post_goal_dist_m") - _safe_float(last[-1], "post_goal_dist_m")
    wp_delta = _safe_int(last[-1], "post_wp_idx") - _safe_int(last[0], "pre_wp_idx")
    nearest_delta = _safe_int(last[-1], "post_nearest_path_idx") - _safe_int(last[0], "pre_nearest_path_idx")
    dist_to_wp_delta = _safe_float(last[0], "post_dist_to_wp_m") - _safe_float(last[-1], "post_dist_to_wp_m")
    final_goal = _safe_float(last[-1], "post_goal_dist_m")
    final_path_error = _safe_float(last[-1], "post_path_error_m")

    if final_goal <= 700.0:
        return "timeout_near_goal_threshold_or_finish_logic"
    if hazard_max <= 0.05 and apas_steps <= 0.5 and mean_speed < 7.0:
        if goal_progress < 150.0 and wp_delta <= 0 and nearest_delta <= 0:
            return "timeout_low_risk_slow_waypoint_stuck"
        return "timeout_low_risk_slow_cruise"
    if wp_delta <= 0 and nearest_delta <= 0 and dist_to_wp_delta <= 50.0:
        return "timeout_waypoint_stuck"
    if final_path_error <= 120.0 and final_goal > 1500.0:
        return "timeout_path_too_long_or_speed_budget"
    if _mean(tail, "progress_m") <= 1.0:
        return "timeout_low_progress_or_loitering"
    return "timeout_mixed"


def _classify_core(rows: list[dict]) -> str:
    tail = _window(rows, 10)
    if not tail:
        return "core_untraced"
    visible_danger = max(_max(tail, "local_hazard_max_danger"), _max(tail, "obs_max_danger"))
    no_valid = _sum(tail, "apas_no_valid_candidate")
    if visible_danger < 0.15 and no_valid > 0:
        return "core_sensor_blind_one_step_entry"
    if visible_danger >= 0.15 and no_valid > 0:
        return "core_warning_seen_but_no_valid_escape"
    return "core_warning_seen_but_not_avoided"


def _classify_storm(rows: list[dict]) -> str:
    tail = _window(rows, 20)
    if not tail:
        return "storm_untraced"
    visible_danger = max(_max(tail, "local_hazard_max_danger"), _max(tail, "obs_max_danger"))
    no_valid = _sum(tail, "apas_no_valid_candidate")
    segment_rejections = _sum(tail, "apas_segment_rejections")
    final_path_error = _safe_float(tail[-1], "post_path_error_m")
    if visible_danger >= 0.8 and no_valid > 0:
        return "storm_visible_risk_wall_no_valid_escape"
    if visible_danger < 0.2 and no_valid > 0:
        return "storm_late_or_blind_risk_entry"
    if segment_rejections > 0:
        return "storm_segment_rejection_funnel"
    if final_path_error > 150.0:
        return "storm_high_risk_after_large_deviation"
    return "storm_accumulated_risk"


def _classify_terrain(rows: list[dict]) -> str:
    tail = _window(rows, 20)
    if not tail:
        return "terrain_untraced"
    final_path_error = _safe_float(tail[-1], "post_path_error_m")
    no_valid = _sum(tail, "apas_no_valid_candidate")
    visible_danger = max(_max(tail, "local_hazard_max_danger"), _max(tail, "obs_max_danger"))
    if final_path_error >= 200.0 and no_valid > 0:
        return "terrain_escape_pushed_out_of_corridor"
    if visible_danger >= 0.2 and no_valid > 0:
        return "terrain_risk_avoidance_collision"
    return "terrain_or_nfz_direct_collision"


def _classify_failure(reason: str, rows: list[dict]) -> str:
    if reason == "timeout":
        return _classify_timeout(rows)
    if reason == "destructive_storm_core":
        return _classify_core(rows)
    if reason == "storm_risk_too_high":
        return _classify_storm(rows)
    if reason == "terrain_or_nfz":
        return _classify_terrain(rows)
    return f"{reason}_unclassified"


def _episode_metrics(seed: int, method: str, reason: str, rows: list[dict]) -> dict:
    last = _window(rows, 50)
    tail = _window(rows, 20)
    final = rows[-1] if rows else {}
    return {
        "seed": int(seed),
        "method": method,
        "terminated_reason": reason,
        "failure_class": _classify_failure(reason, rows),
        "trace_rows": len(rows),
        "final_step": _safe_int(final, "step"),
        "final_goal_dist_m": _safe_float(final, "post_goal_dist_m"),
        "final_path_error_m": _safe_float(final, "post_path_error_m"),
        "last50_goal_progress_m": (
            _safe_float(last[0], "post_goal_dist_m") - _safe_float(last[-1], "post_goal_dist_m")
            if len(last) >= 2
            else 0.0
        ),
        "last50_mean_progress_m": _mean(last, "progress_m"),
        "last50_mean_ground_speed_mps": _mean(last, "ground_speed_mps"),
        "last50_max_p_crash": _max(last, "p_crash"),
        "last50_max_obs_danger": _max(last, "obs_max_danger"),
        "last50_max_local_danger": _max(last, "local_hazard_max_danger"),
        "last50_apas_steps": _sum(last, "apas_intervened"),
        "last50_no_valid_steps": _sum(last, "apas_no_valid_candidate"),
        "last50_segment_rejections": _sum(last, "apas_segment_rejections"),
        "last50_wp_idx_delta": _safe_int(last[-1], "post_wp_idx") - _safe_int(last[0], "pre_wp_idx")
        if len(last) >= 2
        else 0,
        "last50_nearest_path_idx_delta": (
            _safe_int(last[-1], "post_nearest_path_idx") - _safe_int(last[0], "pre_nearest_path_idx")
            if len(last) >= 2 and "post_nearest_path_idx" in last[-1]
            else 0
        ),
        "last20_mean_commanded_speed_mps": _mean(tail, "commanded_airspeed_mps"),
        "last20_mean_base_speed_mps": _mean(tail, "base_airspeed_mps"),
        "final_expert_mode": str(final.get("executed_expert_mode", "")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify failure modes from diagnose_failure_traces CSV output.")
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "failure_trace_analysis")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = _load_csv(args.episodes_csv)
    traces = _load_csv(args.trace_csv)

    failed = [
        row
        for row in episodes
        if not _safe_bool(row, "success") and _safe_bool(row, "trace_included")
    ]
    analysis_rows = []
    for row in failed:
        seed = _safe_int(row, "seed")
        method = str(row.get("method", ""))
        reason = str(row.get("terminated_reason", "unknown"))
        rows = _rows_for(traces, seed, method)
        analysis_rows.append(_episode_metrics(seed, method, reason, rows))

    by_method = defaultdict(list)
    for row in analysis_rows:
        by_method[row["method"]].append(row)

    summary = {}
    for method, rows in sorted(by_method.items()):
        summary[method] = {
            "failures": len(rows),
            "by_reason": dict(Counter(row["terminated_reason"] for row in rows)),
            "by_failure_class": dict(Counter(row["failure_class"] for row in rows)),
            "timeout_classes": dict(
                Counter(row["failure_class"] for row in rows if row["terminated_reason"] == "timeout")
            ),
            "storm_classes": dict(
                Counter(row["failure_class"] for row in rows if row["terminated_reason"] == "storm_risk_too_high")
            ),
            "core_classes": dict(
                Counter(row["failure_class"] for row in rows if row["terminated_reason"] == "destructive_storm_core")
            ),
            "terrain_classes": dict(
                Counter(row["failure_class"] for row in rows if row["terminated_reason"] == "terrain_or_nfz")
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.trace_csv).stem.replace("failure_trace_steps_", "")
    row_path = args.output_dir / f"failure_trace_analysis_{stem}.csv"
    summary_path = args.output_dir / f"failure_trace_analysis_summary_{stem}.json"

    if analysis_rows:
        with row_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(analysis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(analysis_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Failure Trace Analysis ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"analysis_csv: {row_path}")
    print(f"summary:      {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
