"""Aggregate dynamic rescheduling stability results into one scenario CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


SEED_RE = re.compile(r"_seed(\d+)$")
METRICS = ("start_shift", "completion_shift")
STATISTICS = ("mean", "max", "min", "q25", "q75", "std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260812\動態穩定性",
        help="Directory containing the baseline, 多單, and 急單 folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to dynamic_stability_summary_mean_std.csv in input-root.",
    )
    return parser.parse_args()


def as_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def collect_seed_directories(scenario_dir: Path) -> Dict[int, Path]:
    candidates: Dict[int, List[Path]] = {}
    for path in scenario_dir.iterdir():
        if not path.is_dir():
            continue
        match = SEED_RE.search(path.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if 1 <= seed <= 10:
            candidates.setdefault(seed, []).append(path)

    selected: Dict[int, Path] = {}
    for seed, paths in sorted(candidates.items()):
        # If duplicate runs exist, use the latest timestamped folder and make
        # the choice visible instead of counting the same seed twice.
        selected[seed] = max(paths, key=lambda path: path.name)
        if len(paths) > 1:
            names = ", ".join(path.name for path in sorted(paths))
            print(f"[WARN] {scenario_dir.name} seed={seed:03d} has duplicates: {names}")
            print(f"       using: {selected[seed].name}")
    return selected


def event_statistic(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {name: 0.0 for name in STATISTICS}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "max": float(array.max()),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "std": float(array.std(ddof=0)),
    }


def format_mean_std(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return "0.00 (0.00)"
    return f"{array.mean():.2f} ({array.std(ddof=0):.2f})"


def event_values(
    summary_row: Mapping[str, str],
    operation_rows: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    start_shifts = []
    completion_shifts = []
    for row in operation_rows:
        start_shift = as_float(row.get("start_shift"))
        end_before = as_float(row.get("end_before"))
        end_after = as_float(row.get("end_after"))
        if start_shift is not None:
            start_shifts.append(abs(start_shift))
        if end_before is not None and end_after is not None:
            completion_shifts.append(abs(end_after - end_before))

    total_jobs = as_float(summary_row.get("release_jobs"))
    total_ops = as_float(summary_row.get("release_operations"))
    old_jobs = as_float(summary_row.get("old_future_jobs"))
    old_ops = as_float(summary_row.get("old_future_ops"))
    machine_changes = as_float(summary_row.get("machine_changes"))
    machine_change_rate = as_float(summary_row.get("machine_change_rate"))

    return {
        "total_jobs": 0.0 if total_jobs is None else total_jobs,
        "total_ops": 0.0 if total_ops is None else total_ops,
        "old_jobs": 0.0 if old_jobs is None else old_jobs,
        "old_ops": 0.0 if old_ops is None else old_ops,
        "start_shift": event_statistic(start_shifts),
        # These are already event-level values in the reschedule summary.
        "machine_changes": 0.0 if machine_changes is None else machine_changes,
        "machine_change_rate": (
            0.0 if machine_change_rate is None else machine_change_rate
        ),
        "completion_shift": event_statistic(completion_shifts),
    }


def summarize_scenario(scenario_dir: Path) -> Dict[str, str] | None:
    seed_dirs = collect_seed_directories(scenario_dir)
    missing = [seed for seed in range(1, 11) if seed not in seed_dirs]
    if missing:
        print(f"[WARN] {scenario_dir.name} missing seeds: {missing}")

    samples: Dict[str, List[float]] = {
        "total_jobs": [],
        "total_ops": [],
        "old_jobs": [],
        "old_ops": [],
    }
    for metric in METRICS:
        for statistic in STATISTICS:
            samples[f"{metric}_{statistic}"] = []
    samples["machine_changes"] = []
    samples["machine_change_rate"] = []

    for seed, run_dir in sorted(seed_dirs.items()):
        summary_path = run_dir / "dynamic_reschedule_summary.csv"
        operation_path = run_dir / "dynamic_operation_drift.csv"
        if not summary_path.is_file() or not operation_path.is_file():
            print(f"[WARN] incomplete run skipped: {run_dir}")
            continue

        summary_rows = read_csv(summary_path)
        operation_rows = read_csv(operation_path)
        operations_by_event: Dict[int, List[Mapping[str, str]]] = {}
        for row in operation_rows:
            event_id = int(float(row["event_id"]))
            operations_by_event.setdefault(event_id, []).append(row)

        for summary_row in summary_rows:
            event_id = int(float(summary_row["event_id"]))
            values = event_values(
                summary_row,
                operations_by_event.get(event_id, []),
            )
            for field in ("total_jobs", "total_ops", "old_jobs", "old_ops"):
                samples[field].append(float(values[field]))
            samples["machine_changes"].append(float(values["machine_changes"]))
            samples["machine_change_rate"].append(
                float(values["machine_change_rate"])
            )
            for metric in METRICS:
                metric_values = values[metric]
                for statistic in STATISTICS:
                    samples[f"{metric}_{statistic}"].append(
                        float(metric_values[statistic])
                    )

    if not samples["total_jobs"]:
        return None

    result: Dict[str, str] = {"scenario": scenario_dir.name}
    for field, values in samples.items():
        if field in ("machine_changes", "machine_change_rate"):
            continue
        result[field] = format_mean_std(values)

    # Machine changes is an event-level count, not an operation-level 0/1 flag.
    # Therefore its distribution is computed directly over rescheduling events.
    machine_change_stats = event_statistic(samples["machine_changes"])
    for statistic in STATISTICS:
        result[f"machine_changes_{statistic}"] = f"{machine_change_stats[statistic]:.2f}"
    result["machine_change_rate_mean"] = format_mean_std(
        samples["machine_change_rate"]
    )
    return result


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    rows = []
    for scenario_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        summary = summarize_scenario(scenario_dir)
        if summary is not None:
            rows.append(summary)

    if not rows:
        raise RuntimeError("No valid scenario data found")

    output_path = Path(args.output) if args.output else input_root / "dynamic_stability_summary_mean_std.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "total_jobs", "total_ops", "old_jobs", "old_ops"]
    fieldnames.extend(
        f"{metric}_{statistic}"
        for metric in METRICS
        for statistic in STATISTICS
    )
    fieldnames.extend(
        f"machine_changes_{statistic}" for statistic in STATISTICS
    )
    fieldnames.append("machine_change_rate_mean")
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] wrote {output_path}")


if __name__ == "__main__":
    main()
