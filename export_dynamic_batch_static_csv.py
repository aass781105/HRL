from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RunInput:
    seed: int
    init_jobs: List[JobSpec]
    events: List[Tuple[int, float, List[JobSpec]]]
    n_machines: int
    source: str


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance_json", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--cadence", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=None)
    parser.add_argument("--event_horizon", type=int, default=None)
    parser.add_argument("--interarrival_mean", type=float, default=None)
    parser.add_argument("--burst_k", type=int, default=None)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def load_payload(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_unique_job_ids(payload: Dict):
    job_ids = [int(job.get("job_id", -1)) for job in payload.get("jobs", [])]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Dynamic instance JSON has duplicate job_id values.")


def infer_n_machines_from_payload(payload: Dict) -> int:
    meta_n_m = int(payload.get("meta", {}).get("n_machines", 0))
    if meta_n_m > 0:
        return meta_n_m

    max_m = -1
    for job in payload.get("jobs", []):
        for op in job.get("operations", []):
            for m_key in op.get("machine_times", {}).keys():
                max_m = max(max_m, int(m_key))
    if max_m < 0:
        raise ValueError("Failed to infer n_machines from payload.")
    return max_m + 1


def build_job_spec(job_payload: Dict, n_m: int) -> JobSpec:
    from global_env import JobSpec, OperationSpec

    operations: List[OperationSpec] = []
    ops_payload = sorted(job_payload.get("operations", []), key=lambda x: int(x.get("op_id", 0)))
    if not ops_payload:
        raise ValueError(f"Job {job_payload.get('job_id')} has no operations.")

    total_proc_time = float(job_payload.get("total_proc_time", 0.0))
    min_total_proc_time = float(job_payload.get("min_total_proc_time", 0.0))

    for op in ops_payload:
        row = [0.0] * int(n_m)
        machine_times = op.get("machine_times", {})
        if not machine_times:
            raise ValueError(f"Job {job_payload.get('job_id')} op {op.get('op_id')} has no feasible machine.")
        for m_key, pt in machine_times.items():
            m_idx = int(m_key)
            if m_idx >= int(n_m):
                raise ValueError(f"Machine index {m_idx} out of range for n_m={n_m}.")
            row[m_idx] = float(pt)
        valid = [float(v) for v in row if float(v) > 0]
        avg_pt = float(np.mean(valid)) if valid else 0.0
        operations.append(OperationSpec(time_row=row, avg_proc_time=avg_pt))

    meta = {
        "t_arrive": float(job_payload.get("arrive_time", 0.0)),
        "due_date": float(job_payload.get("due_date", 0.0)),
        "total_proc_time": total_proc_time,
        "min_total_proc_time": min_total_proc_time,
        "total_ops": int(job_payload.get("total_ops", len(operations))),
        "op_offset": 0,
    }
    return JobSpec(job_id=int(job_payload.get("job_id", 0)), operations=operations, meta=meta)


def load_instance_as_run_input(path: str) -> RunInput:
    from global_env import JobSpec, OperationSpec
    from params import configs

    payload = load_payload(path)
    validate_unique_job_ids(payload)
    n_m = infer_n_machines_from_payload(payload)

    init_jobs = [build_job_spec(job, n_m) for job in sorted(payload.get("init_jobs", []), key=lambda x: int(x.get("job_id", 0)))]
    events = []
    for event in sorted(payload.get("events", []), key=lambda x: int(x.get("event_id", 0))):
        jobs = [build_job_spec(job, n_m) for job in sorted(event.get("jobs", []), key=lambda x: int(x.get("job_id", 0)))]
        events.append((int(event.get("event_id", 0)), float(event.get("time", 0.0)), jobs))

    seed = int(payload.get("meta", {}).get("seed", getattr(configs, "event_seed", 42)))
    return RunInput(seed=seed, init_jobs=init_jobs, events=events, n_machines=int(n_m), source=path)


def generate_run_input(seed: int, event_horizon: int, interarrival_mean: float, burst_k: int) -> RunInput:
    from dynamic_job_stream import generate_dynamic_job_stream
    from params import configs

    stream = generate_dynamic_job_stream(
        configs,
        max_events=int(event_horizon),
        interarrival_mean=float(interarrival_mean),
        burst_k=int(burst_k),
        seed=int(seed),
    )
    init_jobs = list(stream["init_jobs"])
    events = [(int(ev.event_id), float(ev.time), list(ev.jobs)) for ev in stream["events"]]
    n_m = int(getattr(configs, "n_m", 0))
    if n_m <= 0:
        raise ValueError("configs.n_m must be set before generating dynamic runs.")
    return RunInput(seed=int(seed), init_jobs=init_jobs, events=events, n_machines=n_m, source=f"generated_seed_{int(seed)}")


def record_batch_rows(
    *,
    run_id: int,
    run_input: RunInput,
    orch: GlobalTimelineOrchestrator,
    batch_label: str,
    batch_manifest: Dict,
    output_rows: List[Dict],
):
    batch_jobs = batch_manifest.get("jobs", [])
    batch_job_count = int(batch_manifest.get("n_jobs", len(batch_jobs)))
    batch_time_abs = float(batch_manifest.get("batch_time_abs", 0.0))

    for job in batch_jobs:
        output_rows.append(
            {
                "run_id": int(run_id),
                "batch_id": str(batch_label),
                "batch_job_count": batch_job_count,
                "job_id": int(job["job_id"]),
                "job_op_count": int(job["remaining_ops"]),
                "due_date_rel": float(job["due_date_rel"]),
            }
        )


def run_one_environment(run_id: int, run_input: RunInput, cadence: int, output_rows: List[Dict]):
    from global_env import GlobalTimelineOrchestrator

    orch = GlobalTimelineOrchestrator(int(run_input.n_machines), job_generator=None, t0=0.0)
    orch.reset(t0=0.0)

    if run_input.init_jobs:
        orch.buffer.extend(run_input.init_jobs)
        init_result = orch.event_release_and_reschedule(0.0, event_id=0)
        if init_result.get("event") == "batch_finalized" and orch.last_batch_manifest:
            record_batch_rows(
                run_id=run_id,
                run_input=run_input,
                orch=orch,
                batch_label="init",
                batch_manifest=orch.last_batch_manifest,
                output_rows=output_rows,
            )

    arrival_count = 0
    for event_id, t_event, jobs in run_input.events:
        if jobs:
            orch.buffer.extend(jobs)
        arrival_count += 1
        should_release = bool(arrival_count % cadence == 0 or arrival_count >= len(run_input.events))
        if should_release:
            result = orch.event_release_and_reschedule(float(t_event), event_id=int(event_id))
            if result.get("event") == "batch_finalized" and orch.last_batch_manifest:
                record_batch_rows(
                    run_id=run_id,
                    run_input=run_input,
                    orch=orch,
                    batch_label=f"event_{event_id}",
                    batch_manifest=orch.last_batch_manifest,
                    output_rows=output_rows,
                )
        else:
            orch.tick_without_release(float(t_event))

    while len(orch.buffer) > 0:
        t_flush = float(orch.t)
        result = orch.event_release_and_reschedule(t_flush, event_id=10_000_000 + run_id)
        if result.get("event") == "batch_finalized" and orch.last_batch_manifest:
            record_batch_rows(
                run_id=run_id,
                run_input=run_input,
                orch=orch,
                batch_label="flush",
                batch_manifest=orch.last_batch_manifest,
                output_rows=output_rows,
            )
        else:
            break


def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "run_id",
        "batch_id",
        "job_id",
        "due_date_rel",
        "job_op_count",
        "batch_job_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": int(row["run_id"]),
                    "batch_id": row["batch_id"],
                    "job_id": int(row["job_id"]),
                    "due_date_rel": f"{float(row['due_date_rel']):.6f}",
                    "job_op_count": int(row["job_op_count"]),
                    "batch_job_count": int(row["batch_job_count"]),
                }
            )


def main():
    args = parse_args()
    num_runs = max(1, int(args.num_runs))

    run_inputs: List[RunInput] = []
    instance_json_path = str(args.instance_json or "").strip()
    if instance_json_path:
        from params import configs

        cadence = max(1, int(args.cadence if args.cadence is not None else getattr(configs, "gate_cadence", 1)))
        run_inputs.append(load_instance_as_run_input(instance_json_path))
        if num_runs > 1:
            for run_idx in range(1, num_runs):
                cloned = run_inputs[0]
                run_inputs.append(
                    RunInput(
                        seed=int(cloned.seed) + run_idx,
                        init_jobs=cloned.init_jobs,
                        events=cloned.events,
                        n_machines=cloned.n_machines,
                        source=f"{cloned.source}#replica_{run_idx + 1}",
                    )
                )
    else:
        from params import configs

        cadence = max(1, int(args.cadence if args.cadence is not None else getattr(configs, "gate_cadence", 1)))
        seed_base = int(args.seed_base if args.seed_base is not None else getattr(configs, "event_seed", 42))
        event_horizon = int(args.event_horizon if args.event_horizon is not None else getattr(configs, "event_horizon", 0))
        interarrival_mean = float(
            args.interarrival_mean if args.interarrival_mean is not None else getattr(configs, "interarrival_mean", 100.0)
        )
        burst_k = int(args.burst_k if args.burst_k is not None else getattr(configs, "burst_size", 1))
        configs.n_m = int(getattr(configs, "n_m", 0) or 0)
        if int(configs.n_m) <= 0:
            raise ValueError("configs.n_m must be set when generating environments without --instance_json.")
        instance_json_path = ""
        for run_idx in range(num_runs):
            seed = int(seed_base) + run_idx
            run_inputs.append(
                generate_run_input(
                    seed=seed,
                    event_horizon=int(event_horizon),
                    interarrival_mean=float(interarrival_mean),
                    burst_k=int(burst_k),
                )
            )

    output_rows: List[Dict] = []
    for run_id, run_input in enumerate(run_inputs, start=1):
        run_one_environment(run_id, run_input, cadence, output_rows)

    output_csv = str(args.output_csv or "").strip()
    if not output_csv:
        output_csv = os.path.join(
            "batch_static_exports",
            f"batch_static_jobs_runs{num_runs}_cad{cadence}.csv",
        )
    write_csv(output_csv, output_rows)

    print(f"Saved CSV: {output_csv}")
    print(f"Runs: {len(run_inputs)} | Rows: {len(output_rows)} | Cadence: {cadence}")


if __name__ == "__main__":
    main()
