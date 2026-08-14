"""Measure how each dynamic release changes the existing future schedule.

Only operations planned strictly after the release time are compared. This
matches the requested stability definition and excludes history, running
operations, and operations starting exactly at the release time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT,
    "yaml_config",
    "eval_baseline_seed1_greedy_cadence1_1run.yml",
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "analysis_results",
    "dynamic_reschedule_stability",
)

OpKey = Tuple[int, int]

SUMMARY_COLUMNS = [
    "scenario",
    "strategy",
    "seed",
    "event_id",
    "sim_time",
    "buffer_jobs_before",
    "wip_jobs_before",
    "release_jobs",
    "release_operations",
    "old_future_jobs",
    "old_future_ops",
    "after_future_jobs",
    "after_future_ops",
    "new_future_jobs",
    "new_future_ops",
    "common_old_ops",
    "missing_old_ops",
    "start_shift_mean",
    "start_shift_p50",
    "start_shift_p90",
    "start_shift_max",
    "start_shift_signed_mean",
    "machine_changes",
    "machine_change_rate",
    "sequence_pairs",
    "sequence_flips",
    "sequence_flip_rate",
    "job_sequence_pairs",
    "job_sequence_flips",
    "job_sequence_flip_rate",
    "position_shift_mean",
    "position_shift_p90",
    "position_shift_max",
    "sub_makespan",
    "sub_tardiness",
]

OPERATION_COLUMNS = [
    "scenario",
    "strategy",
    "seed",
    "event_id",
    "sim_time",
    "job_id",
    "operation_id",
    "matched_after",
    "machine_before",
    "machine_after",
    "start_before",
    "start_after",
    "start_shift",
    "abs_start_shift",
    "end_before",
    "end_after",
    "machine_changed",
    "sequence_position_before",
    "sequence_position_after",
    "sequence_position_shift",
]

MACHINE_COLUMNS = [
    "scenario",
    "strategy",
    "seed",
    "event_id",
    "sim_time",
    "machine",
    "old_ops_before",
    "old_ops_after",
    "new_ops_after",
    "same_machine_common_ops",
    "comparable_pairs",
    "flipped_pairs",
    "flip_rate",
    "job_comparable_pairs",
    "job_flipped_pairs",
    "job_flip_rate",
    "position_shift_mean",
    "position_shift_p90",
    "position_shift_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze old-operation drift at real dynamic release events."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Evaluation YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Override event/sample seed.")
    parser.add_argument(
        "--max_events",
        type=int,
        default=None,
        help="Shorter event horizon for a smoke test.",
    )
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--keep_action_modes",
        action="store_true",
        help="Keep action modes from YAML instead of forcing greedy inference.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unknown arguments: {' '.join(unknown)}")
    return args


LOCAL_ARGS = parse_args()
CONFIG_PATH = os.path.abspath(LOCAL_ARGS.config)
if not os.path.isfile(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# params.py parses process arguments at import time.
sys.argv = [sys.argv[0], "--config", CONFIG_PATH]

from hrl_main import run_event_driven_until_nevents  # noqa: E402
from params import configs  # noqa: E402


def op_key(row: Mapping[str, object]) -> OpKey:
    return int(row["job"]), int(row["op"])


def future_rows(rows: Iterable[Mapping[str, object]], sim_time: float) -> List[dict]:
    return [
        dict(row)
        for row in rows
        if float(row["start"]) > float(sim_time)
    ]


def row_map(rows: Iterable[Mapping[str, object]]) -> Dict[OpKey, dict]:
    return {op_key(row): dict(row) for row in rows}


def sequence_by_machine(rows: Iterable[Mapping[str, object]]) -> Dict[int, List[OpKey]]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["machine"]), []).append(dict(row))
    return {
        machine: [
            op_key(row)
            for row in sorted(
                machine_rows,
                key=lambda item: (
                    float(item["start"]),
                    float(item["end"]),
                    int(item["job"]),
                    int(item["op"]),
                ),
            )
        ]
        for machine, machine_rows in grouped.items()
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean_value(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def max_value(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=float)))


def write_csv(path: str, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class DynamicStabilityCollector:
    def __init__(self, *, scenario: str, strategy: str, seed: int):
        self.scenario = scenario
        self.strategy = strategy
        self.seed = int(seed)
        self.summary_rows: List[dict] = []
        self.operation_rows: List[dict] = []
        self.machine_rows: List[dict] = []

    def __call__(self, payload: Mapping[str, object]) -> None:
        event_id = int(payload["event_id"])
        sim_time = float(payload["sim_time"])
        before = future_rows(payload["before_rows"], sim_time)
        after = future_rows(payload["after_rows"], sim_time)
        before_map = row_map(before)
        after_map = row_map(after)

        old_keys = set(before_map)
        after_keys = set(after_map)
        common_keys = old_keys & after_keys
        released_buffer_jobs = {
            int(job_id) for job_id in payload["buffer_job_ids_before"]
        }
        new_keys = {
            key for key in after_keys if key[0] in released_buffer_jobs
        }
        missing_keys = old_keys - after_keys

        before_sequences = sequence_by_machine(before)
        after_sequences = sequence_by_machine(after)
        before_positions = {
            key: position
            for sequence in before_sequences.values()
            for position, key in enumerate(sequence)
        }
        after_positions = {
            key: position
            for sequence in after_sequences.values()
            for position, key in enumerate(sequence)
        }

        abs_start_shifts: List[float] = []
        signed_start_shifts: List[float] = []
        position_shifts: List[float] = []
        machine_changes = 0

        for key in sorted(old_keys):
            row_before = before_map[key]
            row_after = after_map.get(key)
            matched = row_after is not None
            if matched:
                start_shift = float(row_after["start"]) - float(row_before["start"])
                abs_start_shift = abs(start_shift)
                machine_changed = int(row_after["machine"]) != int(row_before["machine"])
                sequence_shift = (
                    int(after_positions[key]) - int(before_positions[key])
                    if not machine_changed
                    else ""
                )
                signed_start_shifts.append(start_shift)
                abs_start_shifts.append(abs_start_shift)
                machine_changes += int(machine_changed)
                if sequence_shift != "":
                    position_shifts.append(abs(float(sequence_shift)))
            else:
                start_shift = ""
                abs_start_shift = ""
                machine_changed = ""
                sequence_shift = ""

            self.operation_rows.append(
                {
                    "scenario": self.scenario,
                    "strategy": self.strategy,
                    "seed": self.seed,
                    "event_id": event_id,
                    "sim_time": sim_time,
                    "job_id": key[0],
                    "operation_id": key[1],
                    "matched_after": int(matched),
                    "machine_before": int(row_before["machine"]),
                    "machine_after": int(row_after["machine"]) if matched else "",
                    "start_before": float(row_before["start"]),
                    "start_after": float(row_after["start"]) if matched else "",
                    "start_shift": start_shift,
                    "abs_start_shift": abs_start_shift,
                    "end_before": float(row_before["end"]),
                    "end_after": float(row_after["end"]) if matched else "",
                    "machine_changed": int(machine_changed) if matched else "",
                    "sequence_position_before": int(before_positions[key]),
                    "sequence_position_after": int(after_positions[key]) if matched else "",
                    "sequence_position_shift": sequence_shift,
                }
            )

        total_pairs = 0
        total_flips = 0
        total_job_pairs = 0
        total_job_flips = 0
        all_machines = sorted(set(before_sequences) | set(after_sequences))
        for machine in all_machines:
            before_sequence = before_sequences.get(machine, [])
            after_sequence = after_sequences.get(machine, [])
            before_position = {key: idx for idx, key in enumerate(before_sequence)}
            after_position = {key: idx for idx, key in enumerate(after_sequence)}
            comparable = [
                key
                for key in before_sequence
                if key in common_keys
                and int(after_map[key]["machine"]) == machine
            ]
            comparable_pairs = 0
            flipped_pairs = 0
            job_comparable_pairs = 0
            job_flipped_pairs = 0
            for first, second in itertools.combinations(comparable, 2):
                comparable_pairs += 1
                before_delta = before_position[first] - before_position[second]
                after_delta = after_position[first] - after_position[second]
                is_flipped = before_delta * after_delta < 0
                if is_flipped:
                    flipped_pairs += 1
                if first[0] != second[0]:
                    job_comparable_pairs += 1
                    if is_flipped:
                        job_flipped_pairs += 1

            machine_position_shifts = [
                abs(float(after_position[key] - before_position[key]))
                for key in comparable
            ]
            total_pairs += comparable_pairs
            total_flips += flipped_pairs
            total_job_pairs += job_comparable_pairs
            total_job_flips += job_flipped_pairs
            self.machine_rows.append(
                {
                    "scenario": self.scenario,
                    "strategy": self.strategy,
                    "seed": self.seed,
                    "event_id": event_id,
                    "sim_time": sim_time,
                    "machine": machine,
                    "old_ops_before": len(before_sequence),
                    "old_ops_after": sum(key in common_keys for key in after_sequence),
                    "new_ops_after": sum(key in new_keys for key in after_sequence),
                    "same_machine_common_ops": len(comparable),
                    "comparable_pairs": comparable_pairs,
                    "flipped_pairs": flipped_pairs,
                    "flip_rate": (
                        float(flipped_pairs / comparable_pairs)
                        if comparable_pairs
                        else 0.0
                    ),
                    "job_comparable_pairs": job_comparable_pairs,
                    "job_flipped_pairs": job_flipped_pairs,
                    "job_flip_rate": (
                        float(job_flipped_pairs / job_comparable_pairs)
                        if job_comparable_pairs
                        else 0.0
                    ),
                    "position_shift_mean": mean_value(machine_position_shifts),
                    "position_shift_p90": percentile(machine_position_shifts, 90),
                    "position_shift_max": max_value(machine_position_shifts),
                }
            )

        old_jobs = {key[0] for key in old_keys}
        after_jobs = {key[0] for key in after_keys}
        new_jobs = {key[0] for key in new_keys}
        common_count = len(common_keys)
        self.summary_rows.append(
            {
                "scenario": self.scenario,
                "strategy": self.strategy,
                "seed": self.seed,
                "event_id": event_id,
                "sim_time": sim_time,
                "buffer_jobs_before": len(payload["buffer_job_ids_before"]),
                "wip_jobs_before": len(old_jobs),
                "release_jobs": int(payload["release_jobs_count"]),
                "release_operations": int(payload["release_operations_count"]),
                "old_future_jobs": len(old_jobs),
                "old_future_ops": len(old_keys),
                "after_future_jobs": len(after_jobs),
                "after_future_ops": len(after_keys),
                "new_future_jobs": len(new_jobs),
                "new_future_ops": len(new_keys),
                "common_old_ops": common_count,
                "missing_old_ops": len(missing_keys),
                "start_shift_mean": mean_value(abs_start_shifts),
                "start_shift_p50": percentile(abs_start_shifts, 50),
                "start_shift_p90": percentile(abs_start_shifts, 90),
                "start_shift_max": max_value(abs_start_shifts),
                "start_shift_signed_mean": mean_value(signed_start_shifts),
                "machine_changes": machine_changes,
                "machine_change_rate": (
                    float(machine_changes / common_count) if common_count else 0.0
                ),
                "sequence_pairs": total_pairs,
                "sequence_flips": total_flips,
                "sequence_flip_rate": (
                    float(total_flips / total_pairs) if total_pairs else 0.0
                ),
                "job_sequence_pairs": total_job_pairs,
                "job_sequence_flips": total_job_flips,
                "job_sequence_flip_rate": (
                    float(total_job_flips / total_job_pairs)
                    if total_job_pairs
                    else 0.0
                ),
                "position_shift_mean": mean_value(position_shifts),
                "position_shift_p90": percentile(position_shifts, 90),
                "position_shift_max": max_value(position_shifts),
                "sub_makespan": float(payload["sub_makespan"]),
                "sub_tardiness": float(payload["sub_tardiness"]),
            }
        )
        print(
            f"[STABILITY] event={event_id:03d} t={sim_time:.1f} "
            f"old_ops={len(old_keys)} common={common_count} "
            f"|dStart|={mean_value(abs_start_shifts):.2f} "
            f"mch={machine_changes}/{common_count} "
            f"flip={total_flips}/{total_pairs}"
        )


def main() -> None:
    seed = int(
        getattr(configs, "event_seed", 1)
        if LOCAL_ARGS.seed is None
        else LOCAL_ARGS.seed
    )
    max_events = int(
        getattr(configs, "event_horizon", 160)
        if LOCAL_ARGS.max_events is None
        else LOCAL_ARGS.max_events
    )
    if max_events < 1:
        raise ValueError("--max_events must be >= 1")

    configs.disable_main_baseline = True
    configs.fast_mode = True
    configs.main_sample_runs = 1
    if not LOCAL_ARGS.keep_action_modes:
        configs.hl_eval_action_selection = "greedy"
        configs.ll_eval_action_selection = "greedy"

    scenario = str(getattr(configs, "hl_env_scenario", "baseline"))
    strategy = str(getattr(configs, "hl_gate_policy", "ppo"))
    collector = DynamicStabilityCollector(
        scenario=scenario,
        strategy=strategy,
        seed=seed,
    )

    print(
        f"[START] scenario={scenario} strategy={strategy} seed={seed} "
        f"events={max_events} config={CONFIG_PATH}"
    )
    final_mk, stats = run_event_driven_until_nevents(
        max_events=max_events,
        interarrival_mean=float(getattr(configs, "interarrival_mean", 25.0)),
        burst_K=int(getattr(configs, "burst_size", 1)),
        write_outputs=False,
        seed_override=seed,
        sample_seed_override=seed,
        reschedule_observer=collector,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{scenario}_{strategy}_seed{seed:03d}"
    output_dir = os.path.abspath(os.path.join(LOCAL_ARGS.output_root, run_name))
    summary_path = os.path.join(output_dir, "dynamic_reschedule_summary.csv")
    operation_path = os.path.join(output_dir, "dynamic_operation_drift.csv")
    machine_path = os.path.join(output_dir, "dynamic_machine_sequence_drift.csv")
    write_csv(summary_path, SUMMARY_COLUMNS, collector.summary_rows)
    write_csv(operation_path, OPERATION_COLUMNS, collector.operation_rows)
    write_csv(machine_path, MACHINE_COLUMNS, collector.machine_rows)

    print(
        f"[DONE] releases={len(collector.summary_rows)} "
        f"final_mk={final_mk:.3f} final_td={float(stats['total_tardiness']):.3f}"
    )
    print(f"[CSV] {summary_path}")
    print(f"[CSV] {operation_path}")
    print(f"[CSV] {machine_path}")


if __name__ == "__main__":
    main()
