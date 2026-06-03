import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from gantt import plot_global_gantt
from global_env import GlobalTimelineOrchestrator, JobSpec, OperationSpec
from params import configs


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance_json", type=str, default=str(getattr(configs, "instance_json", "") or ""))
    parser.add_argument("--output_dir", type=str, default=str(getattr(configs, "plot_global_dir", "plots/global")))
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


def get_payload_arrive_time(payload: Dict, job_id: int) -> float:
    for job in payload.get("jobs", []) or []:
        if int(job.get("job_id", -1)) == int(job_id):
            return float(job.get("arrive_time", job.get("t_arrive_abs", 0.0)))
    return 0.0


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


def _mean_std(values: List[float]):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def make_unique_output_dir(base_dir: str, env_name: str) -> str:
    base_path = os.path.join(base_dir, env_name)
    if not os.path.exists(base_path):
        return base_path
    return f"{base_path}_{time.strftime('%Y%m%d_%H%M%S')}"


def run_once(
    args,
    payload: Dict,
    instance_json_path: str,
    *,
    run_idx: int,
    sample_seed: int,
    output_dir: str,
    write_outputs: bool,
) -> Dict:
    cadence = max(1, int(args.cadence))
    jobs_payload = sorted(payload.get("jobs", []), key=lambda x: int(x.get("job_id", 0)))
    init_jobs_payload = sorted(payload.get("init_jobs", []), key=lambda x: int(x.get("job_id", 0)))
    events_payload = sorted(payload.get("events", []), key=lambda x: int(x.get("event_id", 0)))

    n_m = infer_n_machines(payload)
    configs.n_m = int(n_m)
    configs.scheduler_type = "PPO"
    ensure_model_path()
    configs._active_eval_env_seed = int(getattr(configs, "event_seed", 42))
    configs._active_eval_sample_seed = int(sample_seed)
    torch.manual_seed(int(sample_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sample_seed))

    batch_jobs_due_dates = {int(job.get("job_id", 0)): float(job.get("due_date", 0.0)) for job in jobs_payload}

    orch = GlobalTimelineOrchestrator(int(n_m), job_generator=None, t0=0.0)

    schedule_rows: List[Dict] = []
    manifest_rows: List[Dict] = []
    batch_summary_rows: List[Dict] = []
    release_log_rows: List[Dict] = []
    previous_release_job_ids = set()
    release_count = 0
    plot_seq = 0

    def compute_rows_kpis(rows, due_dates: Dict[int, float]) -> Dict[str, float]:
        job_finish: Dict[int, float] = {}
        max_end = 0.0
        for row in rows:
            jid = int(row["job"])
            end = float(row["end"])
            job_finish[jid] = max(job_finish.get(jid, 0.0), end)
            max_end = max(max_end, end)
        total_td = sum(
            max(0.0, float(job_finish.get(int(job_id), 0.0)) - float(due))
            for job_id, due in due_dates.items()
            if int(job_id) in job_finish
        )
        return {
            "makespan": float(max_end),
            "total_tardiness": float(total_td),
            "objective_value": float(0.5 * max_end + 0.5 * total_td),
        }

    def compute_global_kpis() -> Dict[str, float]:
        rows = list(getattr(orch, "_global_rows", [])) + list(getattr(orch, "_last_full_rows", []))
        info = compute_rows_kpis(rows, batch_jobs_due_dates)
        if getattr(orch, "machine_free_time", None) is not None and len(orch.machine_free_time) > 0:
            info["makespan"] = max(info["makespan"], float(np.max(orch.machine_free_time)))
            info["objective_value"] = 0.5 * info["makespan"] + 0.5 * info["total_tardiness"]
        return info

    def record_release(result: Dict, batch_label: str):
        nonlocal release_count, plot_seq, previous_release_job_ids
        manifest = getattr(orch, "last_batch_manifest", None)
        if not manifest:
            return
        release_count += 1
        manifest = dict(manifest)
        manifest["batch_label"] = batch_label
        manifest["num_rows"] = int(len(result.get("rows", [])))
        manifest_rows.append(manifest)
        rows = result.get("rows", [])
        current_job_ids = {int(row["job"]) for row in rows}
        repeated_job_ids = sorted(current_job_ids & previous_release_job_ids)
        sub_info = compute_rows_kpis(rows, batch_jobs_due_dates)
        global_info = compute_global_kpis()
        release_log_rows.append(
            {
                "Event_ID": "" if manifest.get("event_id") is None else int(manifest.get("event_id")),
                "Release_Type": str(batch_label).upper(),
                "Release_Time": float(manifest.get("batch_time_abs", 0.0)),
                "Objective_0p5MK_0p5TD": float(sub_info["objective_value"]),
                "Makespan": float(sub_info["makespan"]),
                "Total_Tardiness": float(sub_info["total_tardiness"]),
                "Global_Objective_0p5MK_0p5TD": float(global_info["objective_value"]),
                "Global_Makespan": float(global_info["makespan"]),
                "Global_Total_Tardiness": float(global_info["total_tardiness"]),
                "Num_Committed_Jobs": int(len(getattr(orch, "_committed_jobs", []))),
                "Num_Rows": int(len(rows)),
                "Subproblem_Job_Count": int(len(current_job_ids)),
                "Repeated_Job_Count": int(len(repeated_job_ids)),
                "Repeated_Job_IDs": ";".join(str(job_id) for job_id in repeated_job_ids),
            }
        )
        previous_release_job_ids = current_job_ids
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
                    "Arrive_Time": float(next((job.get("arrive_time", job.get("t_arrive_abs", 0.0)) for job in manifest.get("jobs", []) if int(job["job_id"]) == job_id), 0.0)),
                    "Due_Date": due_date,
                    "Is_Last_Op": is_last_op,
                    "Tardiness": tardiness,
                }
            )
        if write_outputs:
            os.makedirs(output_dir, exist_ok=True)
            t_abs = float(manifest.get("batch_time_abs", 0.0))
            label = "_INIT" if str(batch_label).lower() == "init" else ""
            details_path = os.path.join(output_dir, f"details_r{release_count:03d}_t{int(t_abs):05d}{label}.csv")
            unique_rows = {}

            def row_status(row):
                return "History" if float(row["start"]) < t_abs else "NewPlan"

            def should_replace(existing, new_row):
                if existing is None:
                    return True
                _, existing_status = existing
                new_status = row_status(new_row)
                if existing_status == "History" and new_status == "NewPlan":
                    return False
                if existing_status == "NewPlan" and new_status == "History":
                    return True
                return True

            for row in list(getattr(orch, "_global_rows", [])):
                key = (int(row["job"]), int(row["op"]))
                if should_replace(unique_rows.get(key), row):
                    unique_rows[key] = (dict(row), row_status(row))
            for row in list(getattr(orch, "_last_full_rows", [])):
                key = (int(row["job"]), int(row["op"]))
                if should_replace(unique_rows.get(key), row):
                    unique_rows[key] = (dict(row), row_status(row))

            job_max_op = {}
            for jid, op_id in unique_rows.keys():
                job_max_op[jid] = max(job_max_op.get(jid, -1), op_id)

            detail_rows = []
            for (jid, op_id), (row, status) in sorted(unique_rows.items()):
                due_date = float(batch_jobs_due_dates.get(jid, 0.0))
                is_last_op = bool(op_id == job_max_op[jid])
                detail_rows.append(
                    {
                        "job": jid,
                        "op": op_id,
                        "machine": int(row["machine"]),
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "duration": float(row["duration"]),
                        "arrive_time": get_payload_arrive_time(payload, jid),
                        "status": status,
                        "due_date": due_date,
                        "tardiness": max(0.0, float(row["end"]) - due_date) if is_last_op else 0.0,
                    }
                )
            with open(details_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["job", "op", "machine", "start", "end", "duration", "arrive_time", "status", "due_date", "tardiness"],
                )
                writer.writeheader()
                writer.writerows(detail_rows)

            global_plot_rows = []
            for grow in list(getattr(orch, "_global_rows", [])) + list(getattr(orch, "_last_full_rows", [])):
                gid = int(grow["job"])
                global_plot_rows.append(
                    {
                        "job": gid,
                        "op": int(grow["op"]),
                        "machine": int(grow["machine"]),
                        "start": float(grow["start"]),
                        "end": float(grow["end"]),
                        "duration": float(grow["duration"]),
                        "phase": "history" if float(grow["start"]) < t_abs else "newplan",
                        "due_date": float(batch_jobs_due_dates.get(gid, 0.0)),
                    }
                )
            gantt_name = f"global_r{plot_seq:03d}_t0.png" if plot_seq == 0 else f"global_r{plot_seq:03d}_t{int(t_abs):05d}.png"
            plot_global_gantt(
                global_plot_rows,
                os.path.join(output_dir, gantt_name),
                t_now=t_abs,
                title=f"PPO Replay cad{cadence} {batch_label}",
            )
            plot_seq += 1

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

    stem = os.path.splitext(os.path.basename(instance_json_path))[0]
    run_prefix = f"cad{cadence}"
    base_name = f"{run_prefix}_run{run_idx + 1:02d}_s{sample_seed}"
    manifest_jsonl = os.path.join(output_dir, f"{base_name}_batch_manifests.jsonl")
    summary_csv = os.path.join(output_dir, f"{base_name}_batch_summary.csv")
    schedule_csv = os.path.join(output_dir, f"{base_name}_schedule.csv")
    release_log_csv = os.path.join(output_dir, f"{run_prefix}_ppo_release_log.csv")
    gantt_png = os.path.join(output_dir, "global_r000_t0.png")
    summary_json = os.path.join(output_dir, f"{base_name}_summary.json")

    if write_outputs:
        os.makedirs(output_dir, exist_ok=True)
        write_jsonl(manifest_jsonl, manifest_rows)
        write_schedule_csv(schedule_csv, schedule_rows)
        with open(release_log_csv, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "Event_ID",
                "Release_Type",
                "Release_Time",
                "Objective_0p5MK_0p5TD",
                "Makespan",
                "Total_Tardiness",
                "Global_Objective_0p5MK_0p5TD",
                "Global_Makespan",
                "Global_Total_Tardiness",
                "Num_Committed_Jobs",
                "Num_Rows",
                "Subproblem_Job_Count",
                "Repeated_Job_Count",
                "Repeated_Job_IDs",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in release_log_rows:
                writer.writerow(
                    {
                        "Event_ID": row["Event_ID"],
                        "Release_Type": row["Release_Type"],
                        "Release_Time": f"{float(row['Release_Time']):.4f}",
                        "Objective_0p5MK_0p5TD": f"{float(row['Objective_0p5MK_0p5TD']):.4f}",
                        "Makespan": f"{float(row['Makespan']):.4f}",
                        "Total_Tardiness": f"{float(row['Total_Tardiness']):.4f}",
                        "Global_Objective_0p5MK_0p5TD": f"{float(row['Global_Objective_0p5MK_0p5TD']):.4f}",
                        "Global_Makespan": f"{float(row['Global_Makespan']):.4f}",
                        "Global_Total_Tardiness": f"{float(row['Global_Total_Tardiness']):.4f}",
                        "Num_Committed_Jobs": int(row["Num_Committed_Jobs"]),
                        "Num_Rows": int(row["Num_Rows"]),
                        "Subproblem_Job_Count": int(row["Subproblem_Job_Count"]),
                        "Repeated_Job_Count": int(row["Repeated_Job_Count"]),
                        "Repeated_Job_IDs": str(row["Repeated_Job_IDs"]),
                    }
                )
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
        "ppo_model_name": os.path.splitext(os.path.basename(str(getattr(configs, "ppo_model_path", ""))))[0],
        "run": int(run_idx + 1),
        "sample_seed": int(sample_seed),
        "eval_action_selection": str(getattr(configs, "eval_action_selection", "greedy")),
        "cadence": int(cadence),
        "num_jobs_total": int(len(jobs_payload)),
        "num_events": int(len(events_payload)),
        "num_init_jobs": int(len(init_jobs_payload)),
        "num_batches_recorded": int(len(manifest_rows)),
        "makespan": makespan,
        "total_tardiness": total_tardiness,
        "objective": "0.5*MK + 0.5*TD",
        "objective_value": objective_value,
        "details_written": bool(write_outputs),
        "manifest_jsonl": manifest_jsonl if write_outputs else "",
        "summary_json": summary_json if write_outputs else "",
        "summary_csv": summary_csv if write_outputs else "",
        "schedule_csv": schedule_csv if write_outputs else "",
        "release_log_csv": release_log_csv if write_outputs else "",
        "gantt_png": gantt_png if write_outputs else "",
    }
    if write_outputs:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Instance JSON: {instance_json_path}")
    if write_outputs:
        print(f"Summary: {summary_json}")
        print(f"Manifests: {manifest_jsonl}")
        print(f"Schedule: {schedule_csv}")
        print(f"Release log: {release_log_csv}")
        print(f"Gantt: {gantt_png}")
    print(
        f"Run {run_idx + 1:02d} | sample_seed={sample_seed} | Cadence={cadence} | Batches={len(manifest_rows)} | "
        f"MK={makespan:.4f} | TD={total_tardiness:.4f}"
    )
    return summary


def main():
    args = parse_args()
    instance_json_path = resolve_instance_json_path(args.instance_json)
    print(f"[INFO] Using instance_json: {instance_json_path}")
    payload = load_payload(instance_json_path)
    validate_unique_job_ids(payload)

    eval_runs = int(getattr(configs, "main_sample_runs", 1))
    if eval_runs <= 0:
        eval_runs = 1

    cadence = max(1, int(args.cadence))
    stem = os.path.splitext(os.path.basename(instance_json_path))[0]
    run_name = args.name or f"ppo_cadence_{cadence:02d}_{stem}"
    env_output_name = args.name or stem
    output_dir = make_unique_output_dir(args.output_dir, env_output_name)
    os.makedirs(output_dir, exist_ok=True)

    base_seed = int(getattr(configs, "event_seed", 42))
    run_rows = []
    for run_idx in range(eval_runs):
        sample_seed = base_seed + run_idx
        write_outputs = bool(run_idx == eval_runs - 1)
        summary = run_once(
            args,
            payload,
            instance_json_path,
            run_idx=run_idx,
            sample_seed=sample_seed,
            output_dir=output_dir,
            write_outputs=write_outputs,
        )
        run_rows.append(summary)

    mk_mean, mk_std = _mean_std([float(row["makespan"]) for row in run_rows])
    td_mean, td_std = _mean_std([float(row["total_tardiness"]) for row in run_rows])
    obj_mean, obj_std = _mean_std([float(row["objective_value"]) for row in run_rows])
    rel_mean, rel_std = _mean_std([float(row["num_batches_recorded"]) for row in run_rows])

    sample_csv_path = os.path.join(output_dir, "sample_runs_summary.csv")
    with open(sample_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "run",
            "ppo_model_name",
            "ppo_model_path",
            "instance_json",
            "sample_seed",
            "eval_action_selection",
            "cadence",
            "makespan",
            "total_tardiness",
            "obj",
            "release_count",
            "summary_json",
            "schedule_csv",
            "batch_summary_csv",
            "manifest_jsonl",
            "release_log_csv",
            "gantt_png",
            "details_written",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(
                {
                    "run": int(row["run"]),
                    "ppo_model_name": row["ppo_model_name"],
                    "ppo_model_path": row["ppo_model_path"],
                    "instance_json": row["instance_json"],
                    "sample_seed": int(row["sample_seed"]),
                    "eval_action_selection": row["eval_action_selection"],
                    "cadence": int(row["cadence"]),
                    "makespan": f"{float(row['makespan']):.6f}",
                    "total_tardiness": f"{float(row['total_tardiness']):.6f}",
                    "obj": f"{float(row['objective_value']):.6f}",
                    "release_count": int(row["num_batches_recorded"]),
                    "summary_json": row["summary_json"],
                    "schedule_csv": row["schedule_csv"],
                    "batch_summary_csv": row["summary_csv"],
                    "manifest_jsonl": row["manifest_jsonl"],
                    "release_log_csv": row["release_log_csv"],
                    "gantt_png": row["gantt_png"],
                    "details_written": int(bool(row["details_written"])),
                }
            )
        common = {
            "run": "MEAN",
            "ppo_model_name": run_rows[0]["ppo_model_name"] if run_rows else "",
            "ppo_model_path": run_rows[0]["ppo_model_path"] if run_rows else "",
            "instance_json": instance_json_path,
            "sample_seed": "",
            "eval_action_selection": str(getattr(configs, "eval_action_selection", "greedy")),
            "cadence": int(cadence),
            "summary_json": "",
            "schedule_csv": "",
            "batch_summary_csv": "",
            "manifest_jsonl": "",
            "release_log_csv": "",
            "gantt_png": "",
            "details_written": "",
        }
        writer.writerow({
            **common,
            "makespan": f"{mk_mean:.6f}",
            "total_tardiness": f"{td_mean:.6f}",
            "obj": f"{obj_mean:.6f}",
            "release_count": f"{rel_mean:.6f}",
        })
        writer.writerow({
            **common,
            "run": "STD",
            "makespan": f"{mk_std:.6f}",
            "total_tardiness": f"{td_std:.6f}",
            "obj": f"{obj_std:.6f}",
            "release_count": f"{rel_std:.6f}",
        })

    aggregate_json = os.path.join(output_dir, "aggregate_summary.json")
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "instance_json": instance_json_path,
                "solver": "low_level_ppo_cadence_export",
                "ppo_model_path": str(getattr(configs, "ppo_model_path", "")),
                "ppo_model_name": os.path.splitext(os.path.basename(str(getattr(configs, "ppo_model_path", ""))))[0],
                "eval_runs": int(eval_runs),
                "eval_action_selection": str(getattr(configs, "eval_action_selection", "greedy")),
                "cadence": int(cadence),
                "makespan_mean": mk_mean,
                "makespan_std": mk_std,
                "total_tardiness_mean": td_mean,
                "total_tardiness_std": td_std,
                "objective_mean": obj_mean,
                "objective_std": obj_std,
                "release_count_mean": rel_mean,
                "release_count_std": rel_std,
                "sample_runs_summary_csv": sample_csv_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Sample run CSV: {sample_csv_path}")
    print(f"Aggregate summary: {aggregate_json}")
    print(
        f"Sample runs={eval_runs} | MK mean/std={mk_mean:.3f}/{mk_std:.3f} | "
        f"TD mean/std={td_mean:.3f}/{td_std:.3f} | Obj mean/std={obj_mean:.3f}/{obj_std:.3f}"
    )


if __name__ == "__main__":
    main()
