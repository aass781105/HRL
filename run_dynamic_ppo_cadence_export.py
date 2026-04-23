import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

from global_env import GlobalTimelineOrchestrator, JobSpec, OperationSpec
from params import configs


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance_json", type=str, default=str(getattr(configs, "instance_json", "") or ""))
    parser.add_argument("--output_dir", type=str, default=os.path.join("evaluation_results", "dynamic_ppo_cadence_export"))
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--cadence", type=int, default=int(getattr(configs, "gate_cadence", 1)))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def resolve_instance_json_path(cli_path: str) -> str:
    cli_path = str(cli_path or "").strip()
    if cli_path:
        if not os.path.exists(cli_path):
            raise FileNotFoundError(f"Configured instance_json not found: {cli_path}")
        return cli_path

    candidates = []
    for pattern in (
        os.path.join("evaluation_results", "dynamic_instance*.json"),
        os.path.join("or_tools_solutions", "dynamic", "*_instance.json"),
    ):
        candidates.extend(glob.glob(pattern))

    if not candidates:
        raise ValueError(
            "No dynamic instance JSON found. "
            "Export one first or pass --instance_json explicitly."
        )

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_payload(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_unique_job_ids(payload: Dict):
    job_ids = [int(job.get("job_id", -1)) for job in payload.get("jobs", [])]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(
            "Dynamic instance JSON has duplicate job_id values. "
            "Please re-export the instance before running this script."
        )


def infer_n_machines(payload: Dict) -> int:
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


def ensure_model_path() -> str:
    model_path = str(getattr(configs, "ppo_model_path", "") or "").strip()
    if not model_path:
        raise ValueError("Missing configs.ppo_model_path for low-level PPO scheduler.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model path not found: {model_path}")
    return model_path


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_schedule_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "Batch_ID",
        "Batch_Time",
        "Job",
        "Op",
        "Machine",
        "Start",
        "End",
        "Duration",
        "Arrive_Time",
        "Due_Date",
        "Is_Last_Op",
        "Tardiness",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Batch_ID": int(row["Batch_ID"]),
                    "Batch_Time": f"{float(row['Batch_Time']):.10f}",
                    "Job": int(row["Job"]),
                    "Op": int(row["Op"]),
                    "Machine": int(row["Machine"]),
                    "Start": f"{float(row['Start']):.10f}",
                    "End": f"{float(row['End']):.10f}",
                    "Duration": f"{float(row['Duration']):.10f}",
                    "Arrive_Time": f"{float(row['Arrive_Time']):.10f}",
                    "Due_Date": f"{float(row['Due_Date']):.10f}",
                    "Is_Last_Op": int(row["Is_Last_Op"]),
                    "Tardiness": f"{float(row['Tardiness']):.10f}",
                }
            )


def main():
    args = parse_args()
    instance_json_path = resolve_instance_json_path(args.instance_json)
    print(f"[INFO] Using instance_json: {instance_json_path}")
    payload = load_payload(instance_json_path)
    validate_unique_job_ids(payload)

    cadence = max(1, int(args.cadence))
    jobs_payload = sorted(payload.get("jobs", []), key=lambda x: int(x.get("job_id", 0)))
    init_jobs_payload = sorted(payload.get("init_jobs", []), key=lambda x: int(x.get("job_id", 0)))
    events_payload = sorted(payload.get("events", []), key=lambda x: int(x.get("event_id", 0)))

    n_m = infer_n_machines(payload)
    configs.n_m = int(n_m)
    configs.scheduler_type = "PPO"
    ensure_model_path()

    batch_jobs_due_dates = {int(job.get("job_id", 0)): float(job.get("due_date", 0.0)) for job in jobs_payload}

    orch = GlobalTimelineOrchestrator(int(n_m), job_generator=None, t0=0.0)

    schedule_rows: List[Dict] = []
    manifest_rows: List[Dict] = []
    batch_summary_rows: List[Dict] = []
    release_count = 0

    def record_release(result: Dict, batch_label: str):
        nonlocal release_count
        manifest = getattr(orch, "last_batch_manifest", None)
        if not manifest:
            return
        release_count += 1
        manifest = dict(manifest)
        manifest["batch_label"] = batch_label
        manifest["num_rows"] = int(len(result.get("rows", [])))
        manifest_rows.append(manifest)
        batch_summary_rows.append(
            {
                "batch_label": batch_label,
                "event_id": manifest.get("event_id"),
                "batch_time_abs": float(manifest.get("batch_time_abs", 0.0)),
                "n_jobs": int(manifest.get("n_jobs", 0)),
                "n_ops": int(manifest.get("n_ops", 0)),
                "job_ids": [int(job["job_id"]) for job in manifest.get("jobs", [])],
                "due_dates_abs": [float(job["due_date_abs"]) for job in manifest.get("jobs", [])],
                "due_dates_rel": [float(job["due_date_rel"]) for job in manifest.get("jobs", [])],
            }
        )

        for row in result.get("rows", []):
            job_id = int(row["job"])
            op_id = int(row["op"])
            due_date = float(batch_jobs_due_dates.get(job_id, 0.0))
            job_len = int(next((job["total_ops"] for job in manifest.get("jobs", []) if int(job["job_id"]) == job_id), op_id + 1))
            is_last_op = int(op_id == job_len - 1)
            tardiness = max(0.0, float(row["end"]) - due_date) if is_last_op else 0.0
            schedule_rows.append(
                {
                    "Batch_ID": int(release_count),
                    "Batch_Time": float(manifest.get("batch_time_abs", 0.0)),
                    "Job": job_id,
                    "Op": op_id,
                    "Machine": int(row["machine"]),
                    "Start": float(row["start"]),
                    "End": float(row["end"]),
                    "Duration": float(row["duration"]),
                    "Arrive_Time": float(next((job["t_arrive_abs"] for job in manifest.get("jobs", []) if int(job["job_id"]) == job_id), 0.0)),
                    "Due_Date": due_date,
                    "Is_Last_Op": is_last_op,
                    "Tardiness": tardiness,
                }
            )

    init_jobs = [build_job_spec(job, n_m) for job in init_jobs_payload]
    if init_jobs:
        orch.buffer.extend(init_jobs)
        init_result = orch.event_release_and_reschedule(0.0, event_id=0)
        record_release(init_result, "init")

    arrival_count = 0
    for event_idx, event in enumerate(events_payload, start=1):
        t_now = float(event.get("time", 0.0))
        new_jobs = [build_job_spec(job, n_m) for job in event.get("jobs", [])]
        if new_jobs:
            orch.buffer.extend(new_jobs)
        arrival_count += 1
        is_last_event = bool(event_idx >= len(events_payload))
        should_release = bool(is_last_event or (arrival_count % cadence == 0))

        if should_release:
            result = orch.event_release_and_reschedule(t_now, event_id=int(event.get("event_id", event_idx)))
            if result.get("event") == "batch_finalized":
                record_release(result, f"event_{event_idx}")
        else:
            orch.tick_without_release(t_now)

    while len(orch.buffer) > 0:
        flush_time = float(orch.t)
        result = orch.event_release_and_reschedule(flush_time, event_id=int(len(events_payload) + 1))
        if result.get("event") == "batch_finalized":
            record_release(result, "flush")
        else:
            break

    final_stats = orch.get_final_kpi_stats(batch_jobs_due_dates)
    makespan = float(final_stats["makespan"])
    total_tardiness = float(final_stats["tardiness"])
    objective_value = 0.5 * makespan + 0.5 * total_tardiness

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_name = args.name or f"ppo_cadence_{cadence:02d}_{os.path.splitext(os.path.basename(instance_json_path))[0]}"
    manifest_jsonl = os.path.join(output_dir, f"{base_name}_batch_manifests.jsonl")
    summary_csv = os.path.join(output_dir, f"{base_name}_batch_summary.csv")
    schedule_csv = os.path.join(output_dir, f"{base_name}_schedule.csv")
    summary_json = os.path.join(output_dir, f"{base_name}_summary.json")

    write_jsonl(manifest_jsonl, manifest_rows)
    write_schedule_csv(schedule_csv, schedule_rows)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_label",
                "event_id",
                "batch_time_abs",
                "n_jobs",
                "n_ops",
                "job_ids",
                "due_dates_abs",
                "due_dates_rel",
            ],
        )
        writer.writeheader()
        for row in batch_summary_rows:
            writer.writerow(
                {
                    "batch_label": row["batch_label"],
                    "event_id": "" if row["event_id"] is None else int(row["event_id"]),
                    "batch_time_abs": f"{float(row['batch_time_abs']):.10f}",
                    "n_jobs": int(row["n_jobs"]),
                    "n_ops": int(row["n_ops"]),
                    "job_ids": json.dumps(row["job_ids"], ensure_ascii=False),
                    "due_dates_abs": json.dumps(row["due_dates_abs"], ensure_ascii=False),
                    "due_dates_rel": json.dumps(row["due_dates_rel"], ensure_ascii=False),
                }
            )

    summary = {
        "instance_json": instance_json_path,
        "solver": "low_level_ppo_cadence_export",
        "ppo_model_path": str(getattr(configs, "ppo_model_path", "")),
        "cadence": int(cadence),
        "num_jobs_total": int(len(jobs_payload)),
        "num_events": int(len(events_payload)),
        "num_init_jobs": int(len(init_jobs_payload)),
        "num_batches_recorded": int(len(manifest_rows)),
        "makespan": makespan,
        "total_tardiness": total_tardiness,
        "objective": "0.5*MK + 0.5*TD",
        "objective_value": objective_value,
        "manifest_jsonl": manifest_jsonl,
        "summary_csv": summary_csv,
        "schedule_csv": schedule_csv,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Instance JSON: {instance_json_path}")
    print(f"Summary: {summary_json}")
    print(f"Manifests: {manifest_jsonl}")
    print(f"Schedule: {schedule_csv}")
    print(
        f"Cadence={cadence} | Batches={len(manifest_rows)} | "
        f"MK={makespan:.4f} | TD={total_tardiness:.4f}"
    )


if __name__ == "__main__":
    main()
