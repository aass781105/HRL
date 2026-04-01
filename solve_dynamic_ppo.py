import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

from common_utils import greedy_select_action, sample_action
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from model.PPO import PPO_initialize
from ortools_gantt import plot_ortools_gantt_with_due_dates
from params import configs


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=os.path.join("evaluation_results", "dynamic_ppo_runs"))
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--sample", type=str, default="")
    parser.add_argument("--action_selection", type=str, default="", choices=["", "greedy", "sample"])
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def resolve_instance_json_path(cli_path: str) -> str:
    cli_path = str(cli_path or "").strip()
    if cli_path:
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
            "Please re-export the instance before running solve_dynamic_ppo.py."
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


def build_env_inputs(payload: Dict):
    jobs = sorted(payload.get("jobs", []), key=lambda x: int(x.get("job_id", 0)))
    if not jobs:
        raise ValueError("Dynamic instance JSON has no jobs.")

    n_j = len(jobs)
    n_m = infer_n_machines(payload)
    job_length = np.zeros(n_j, dtype=np.int32)
    op_rows: List[List[float]] = []
    due_dates_abs = np.zeros(n_j, dtype=np.float64)
    release_times = np.zeros(n_j, dtype=np.float64)

    for j_idx, job in enumerate(jobs):
        operations = sorted(job.get("operations", []), key=lambda x: int(x.get("op_id", 0)))
        if not operations:
            raise ValueError(f"Job {job.get('job_id')} has no operations.")
        job_length[j_idx] = len(operations)
        due_dates_abs[j_idx] = float(job.get("due_date", 0.0))
        release_times[j_idx] = float(job.get("arrive_time", 0.0))

        for op in operations:
            row = np.zeros(n_m, dtype=np.float64)
            machine_times = op.get("machine_times", {})
            if not machine_times:
                raise ValueError(f"Job {job.get('job_id')} op {op.get('op_id')} has no feasible machine.")
            for m_key, pt in machine_times.items():
                m_idx = int(m_key)
                if m_idx >= n_m:
                    raise ValueError(f"Machine index {m_idx} out of range for n_m={n_m}.")
                row[m_idx] = float(pt)
            op_rows.append(row.tolist())

    op_pt = np.asarray(op_rows, dtype=np.float64)
    return jobs, job_length, op_pt, due_dates_abs, release_times, n_j, n_m


def write_detail_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["Job"]), int(r["Op"]))):
            writer.writerow(
                {
                    "Job": int(row["Job"]),
                    "Op": int(row["Op"]),
                    "Machine": int(row["Machine"]),
                    "Start": f"{float(row['Start']):.4f}",
                    "End": f"{float(row['End']):.4f}",
                    "Duration": f"{float(row['Duration']):.4f}",
                    "Arrive_Time": f"{float(row['Arrive_Time']):.4f}",
                    "Due_Date": f"{float(row['Due_Date']):.4f}",
                    "Is_Last_Op": int(row["Is_Last_Op"]),
                    "Tardiness": f"{float(row['Tardiness']):.4f}",
                }
            )


def write_machine_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["Machine"]), float(r["Start"]), int(r["Job"]), int(r["Op"]))):
            writer.writerow(
                {
                    "Job": int(row["Job"]),
                    "Op": int(row["Op"]),
                    "Machine": int(row["Machine"]),
                    "Start": f"{float(row['Start']):.4f}",
                    "End": f"{float(row['End']):.4f}",
                    "Duration": f"{float(row['Duration']):.4f}",
                    "Arrive_Time": f"{float(row['Arrive_Time']):.4f}",
                    "Due_Date": f"{float(row['Due_Date']):.4f}",
                    "Is_Last_Op": int(row["Is_Last_Op"]),
                    "Tardiness": f"{float(row['Tardiness']):.4f}",
                }
            )


def main():
    args = parse_args()
    instance_json_path = resolve_instance_json_path(args.instance_json)
    payload = load_payload(instance_json_path)
    validate_unique_job_ids(payload)

    _, jl, pt, due_dates_abs, release_times, n_j, n_m = build_env_inputs(payload)
    configs.n_m = int(n_m)

    device = torch.device(getattr(configs, "device", "cpu"))
    ppo = PPO_initialize()
    model_path = str(getattr(configs, "ppo_model_path", "") or "").strip()
    if not model_path:
        raise ValueError("Missing configs.ppo_model_path for low-level PPO scheduler.")
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ppo.policy.to(device)
    ppo.policy.eval()

    env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
    pt_scale = (float(configs.low) + float(configs.high)) / 2.0
    due_dates_ppo = due_dates_abs / pt_scale

    state = env.set_initial_data(
        job_length_list=[jl],
        op_pt_list=[pt],
        due_date_list=[due_dates_ppo],
        normalize_due_date=False,
        true_due_date_list=[due_dates_abs],
        release_time_list=[release_times],
    )

    if args.action_selection:
        use_sample = args.action_selection == "sample"
    else:
        use_sample = str(args.sample or getattr(configs, "ppo_sample", False)).lower() in ("1", "true", "t", "yes", "y")
    schedule_rows: List[Dict] = []
    done = False
    while not done:
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )

        if use_sample:
            action, _ = sample_action(pi)
            action_np = action.cpu().numpy()
        else:
            action = greedy_select_action(pi)
            action_np = action.cpu().numpy()

        state, _, done, info = env.step(action_np)
        det = info["scheduled_op_details"]
        job_id = int(det["job_id"])
        op_id = int(det["op_id_in_job"])
        due_date = float(due_dates_abs[job_id])
        is_last_op = int(op_id == jl[job_id] - 1)
        tardiness = max(0.0, float(det["end_time"]) - due_date) if is_last_op else 0.0
        schedule_rows.append(
            {
                "Job": job_id,
                "Op": op_id,
                "Machine": int(det["machine_id"]),
                "Start": float(det["start_time"]),
                "End": float(det["end_time"]),
                "Duration": float(det["proc_time"]),
                "Arrive_Time": float(release_times[job_id]),
                "Due_Date": due_date,
                "Is_Last_Op": is_last_op,
                "Tardiness": tardiness,
            }
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_name = args.name or f"ppo_{os.path.splitext(os.path.basename(instance_json_path))[0]}"
    detail_csv = os.path.join(output_dir, f"{base_name}_detail.csv")
    detail_machine_csv = os.path.join(output_dir, f"{base_name}_detail_by_machine.csv")
    gantt_png = os.path.join(output_dir, f"{base_name}_gantt.png")
    summary_json = os.path.join(output_dir, f"{base_name}_summary.json")

    write_detail_csv(detail_csv, schedule_rows)
    write_machine_csv(detail_machine_csv, schedule_rows)
    plot_ortools_gantt_with_due_dates(schedule_rows, gantt_png, title=f"Single PPO Dynamic Schedule\nMK={float(env.current_makespan[0]):.2f}, TD={float(env.accumulated_tardiness[0]):.2f}")

    makespan = float(env.current_makespan[0])
    total_tardiness = float(env.accumulated_tardiness[0])
    objective_value = 0.5 * makespan + 0.5 * total_tardiness

    summary = {
        "instance_json": instance_json_path,
        "solver": "single_low_level_ppo",
        "ppo_model_path": model_path,
        "action_selection": "sample" if use_sample else "greedy",
        "num_jobs": int(n_j),
        "num_operations": int(sum(jl)),
        "n_machines": int(n_m),
        "objective": "0.5*MK + 0.5*TD",
        "objective_value": objective_value,
        "makespan": makespan,
        "total_tardiness": total_tardiness,
        "detail_csv": detail_csv,
        "detail_by_machine_csv": detail_machine_csv,
        "gantt_png": gantt_png,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Instance JSON: {instance_json_path}")
    print(f"Summary: {summary_json}")
    print(
        f"Obj={summary['objective_value']:.4f} | "
        f"MK={summary['makespan']:.4f} | "
        f"TD={summary['total_tardiness']:.4f}"
    )


if __name__ == "__main__":
    main()
