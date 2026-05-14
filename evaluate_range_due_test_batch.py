import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from common_utils import sample_action
from data_utils import text_to_matrix
from model.PPO import PPO_initialize
from ortools_gantt import plot_ortools_gantt_with_due_dates
from params import configs


BASE_DIR = "or_instances_uniform_test_30_50"


def parse_scale(scale_name: str):
    match = re.match(r"^(\d+)x(\d+)_(.+)$", str(scale_name))
    if not match:
        return None, None, ""
    return int(match.group(1)), int(match.group(2)), match.group(3)


def load_due_dates_json(due_data):
    due_dates = due_data["due_dates"]
    if isinstance(due_dates, dict):
        return np.asarray([due_dates[str(i)] for i in sorted(map(int, due_dates.keys()))], dtype=np.float64)
    return np.asarray(due_dates, dtype=np.float64)


def extract_schedule_rows(env, env_idx=0):
    rows = []
    env_idx = int(env_idx)
    job_first = np.asarray(env.job_first_op_id[env_idx], dtype=int)
    job_last = np.asarray(env.job_last_op_id[env_idx], dtype=int)
    job_completion = np.asarray(env.true_candidate_free_time[env_idx], dtype=np.float64)
    due_dates = np.asarray(env.true_due_date[env_idx], dtype=np.float64)

    for machine in range(int(env.number_of_machines)):
        q_len = int(env.mch_queue_len[env_idx, machine])
        for pos in range(q_len):
            op_id = int(env.mch_queue[env_idx, machine, pos])
            if op_id < 0:
                continue
            job_candidates = np.where((job_first <= op_id) & (op_id <= job_last))[0]
            if job_candidates.size == 0:
                continue
            job = int(job_candidates[0])
            local_op = int(op_id - job_first[job])
            end = float(env.true_op_ct[env_idx, op_id])
            duration = float(env.true_op_pt[env_idx, op_id, machine])
            start = end - duration
            completion = float(job_completion[job])
            due_date = float(due_dates[job])
            rows.append({
                "Job": job,
                "Op": local_op,
                "Global_Op": op_id,
                "Machine": int(machine),
                "Start": start,
                "End": end,
                "Duration": duration,
                "Due_Date": due_date,
                "Job_Completion": completion,
                "Job_Tardiness": max(0.0, completion - due_date),
            })

    return sorted(rows, key=lambda r: (int(r["Machine"]), float(r["Start"]), int(r["Job"]), int(r["Op"])))


def run_sample_episode(ppo, jl, pt, due_dates_abs, n_j, n_m, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(
        job_length_list=[jl],
        op_pt_list=[pt],
        due_date_list=[due_dates_abs],
        true_due_date_list=[due_dates_abs],
    )

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
            action, _ = sample_action(pi)
        state, _, done, _ = env.step(action.cpu().numpy())

    makespan = float(env.current_makespan[0])
    total_tardiness = float(env.accumulated_tardiness[0])
    obj = 0.5 * makespan + 0.5 * total_tardiness
    return makespan, total_tardiness, obj


def run_sample_episodes_batched(ppo, jl, pt, due_dates_abs, n_j, n_m, num_runs, seed=None, return_schedules=False):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(
        job_length_list=[jl.copy() for _ in range(int(num_runs))],
        op_pt_list=[pt.copy() for _ in range(int(num_runs))],
        due_date_list=[due_dates_abs.copy() for _ in range(int(num_runs))],
        true_due_date_list=[due_dates_abs.copy() for _ in range(int(num_runs))],
    )

    while True:
        with torch.no_grad():
            batch_idx = ~torch.from_numpy(env.done_flag).to(state.fea_j_tensor.device)
            if not batch_idx.any():
                break
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor[batch_idx],
                op_mask=state.op_mask_tensor[batch_idx],
                candidate=state.candidate_tensor[batch_idx],
                fea_m=state.fea_m_tensor[batch_idx],
                mch_mask=state.mch_mask_tensor[batch_idx],
                comp_idx=state.comp_idx_tensor[batch_idx],
                dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                fea_pairs=state.fea_pairs_tensor[batch_idx],
            )
            action, _ = sample_action(pi)
        state, _, done, _ = env.step(action.cpu().numpy())
        if done.all():
            break

    makespans = np.asarray(env.current_makespan, dtype=np.float64)
    tardiness = np.asarray(env.accumulated_tardiness, dtype=np.float64)
    objs = 0.5 * makespans + 0.5 * tardiness
    if return_schedules:
        schedule_rows = [extract_schedule_rows(env, env_idx=i) for i in range(int(num_runs))]
        return makespans.tolist(), tardiness.tolist(), objs.tolist(), schedule_rows
    return makespans.tolist(), tardiness.tolist(), objs.tolist()


def evaluate_range_due_test(base_dir=BASE_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)

    ppo = PPO_initialize()
    if not os.path.exists(configs.ppo_model_path):
        raise FileNotFoundError(f"PPO model not found: {configs.ppo_model_path}")
    ppo.policy.load_state_dict(torch.load(configs.ppo_model_path, map_location=device, weights_only=True))
    ppo.policy.to(device)
    ppo.policy.eval()
    print(f"Loaded PPO model from {configs.ppo_model_path}")

    eval_runs = int(getattr(configs, "eval_runs_per_instance", 10))
    eval_runs = max(1, eval_runs)
    seed_base = int(getattr(configs, "eval_seed", 42))
    print(f"Range-due static evaluation | base_dir={base_dir} | sample runs={eval_runs}")

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Test directory not found: {base_dir}")

    detail_rows = []
    run_rows = []
    scale_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )

    for scale in scale_dirs:
        n_j_from_name, n_m_from_name, due_setting_from_name = parse_scale(scale)
        curr_path = os.path.join(base_dir, scale)
        fjs_files = sorted(f for f in os.listdir(curr_path) if f.endswith(".fjs"))
        print(f"\nEvaluating {scale} ({len(fjs_files)} instances)...")

        for fjs_name in tqdm(fjs_files):
            base_name = fjs_name[:-4]
            fjs_path = os.path.join(curr_path, fjs_name)
            json_path = os.path.join(curr_path, f"{base_name}.json")
            if not os.path.exists(json_path):
                print(f"Missing JSON, skipped: {json_path}")
                continue

            with open(fjs_path, "r", encoding="utf-8") as f:
                jl, pt = text_to_matrix(f.readlines())
            with open(json_path, "r", encoding="utf-8") as f:
                due_data = json.load(f)

            due_dates_abs = load_due_dates_json(due_data)
            n_j = int(jl.shape[0])
            n_m = int(pt.shape[1])
            due_setting = str(due_data.get("due_setting", due_setting_from_name))
            instance_seed = due_data.get("instance_seed", "")
            due_seed = due_data.get("due_seed", "")
            range_low = due_data.get("range_low", "")
            range_high = due_data.get("range_high", "")

            mk_runs = []
            td_runs = []
            obj_runs = []
            instance_idx = len(detail_rows)
            batch_seed = seed_base + instance_idx * 1000
            mk_runs, td_runs, obj_runs, schedule_runs = run_sample_episodes_batched(
                ppo,
                jl,
                pt,
                due_dates_abs,
                n_j,
                n_m,
                eval_runs,
                seed=batch_seed,
                return_schedules=True,
            )
            best_run_idx = int(np.argmin(obj_runs))
            for run_idx, (mk, td, obj) in enumerate(zip(mk_runs, td_runs, obj_runs)):
                run_seed = batch_seed + run_idx
                run_rows.append({
                    "scale": scale,
                    "n_j": n_j,
                    "n_m": n_m,
                    "due_setting": due_setting,
                    "instance": base_name,
                    "run": run_idx + 1,
                    "sample_seed": run_seed,
                    "makespan": mk,
                    "total_tardiness": td,
                    "obj": obj,
                })

            detail_rows.append({
                "scale": scale,
                "n_j": n_j if n_j_from_name is None else n_j_from_name,
                "n_m": n_m if n_m_from_name is None else n_m_from_name,
                "due_setting": due_setting,
                "instance": base_name,
                "runs": eval_runs,
                "instance_seed": instance_seed,
                "due_seed": due_seed,
                "range_low": range_low,
                "range_high": range_high,
                "makespan_mean": float(np.mean(mk_runs)),
                "makespan_std": float(np.std(mk_runs, ddof=0)),
                "total_tardiness_mean": float(np.mean(td_runs)),
                "total_tardiness_std": float(np.std(td_runs, ddof=0)),
                "obj_mean": float(np.mean(obj_runs)),
                "obj_std": float(np.std(obj_runs, ddof=0)),
                "best_run": best_run_idx + 1,
                "best_makespan": float(mk_runs[best_run_idx]),
                "best_total_tardiness": float(td_runs[best_run_idx]),
                "best_obj": float(obj_runs[best_run_idx]),
                "_best_schedule_rows": schedule_runs[best_run_idx],
            })

    model_name = os.path.basename(configs.ppo_model_path).replace(".pth", "")
    prefix = f"range_due_test_{model_name}_sample{eval_runs}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join("evaluation_results", f"{prefix}_{timestamp}"))
    os.makedirs(output_dir, exist_ok=True)

    detail_df = pd.DataFrame(detail_rows)
    schedule_payloads = []
    if not detail_df.empty and "_best_schedule_rows" in detail_df.columns:
        for _, row in detail_df.iterrows():
            schedule_payloads.append({
                "scale": row["scale"],
                "due_setting": row["due_setting"],
                "instance": row["instance"],
                "best_run": int(row["best_run"]),
                "rows": row["_best_schedule_rows"],
            })
        detail_df = detail_df.drop(columns=["_best_schedule_rows"])
    run_df = pd.DataFrame(run_rows)
    if detail_df.empty:
        raise RuntimeError(f"No instances evaluated under {base_dir}")

    detail_df["ppo_model"] = configs.ppo_model_path
    run_df["ppo_model"] = configs.ppo_model_path

    per_scale_summary = detail_df.groupby(["scale", "n_j", "n_m", "due_setting"], as_index=False).agg({
        "makespan_mean": "mean",
        "makespan_std": "mean",
        "total_tardiness_mean": "mean",
        "total_tardiness_std": "mean",
        "obj_mean": "mean",
        "obj_std": "mean",
    })
    due_summary = detail_df.groupby(["due_setting"], as_index=False).agg({
        "makespan_mean": "mean",
        "makespan_std": "mean",
        "total_tardiness_mean": "mean",
        "total_tardiness_std": "mean",
        "obj_mean": "mean",
        "obj_std": "mean",
    })
    size_summary = detail_df.groupby(["n_j", "n_m"], as_index=False).agg({
        "makespan_mean": "mean",
        "makespan_std": "mean",
        "total_tardiness_mean": "mean",
        "total_tardiness_std": "mean",
        "obj_mean": "mean",
        "obj_std": "mean",
    })

    # Keep filenames short because Windows can fail with FileNotFoundError when
    # the full path exceeds the legacy MAX_PATH limit.
    file_prefix = time.strftime("%H%M%S")
    detail_csv = os.path.join(output_dir, f"{file_prefix}_details.csv")
    runs_csv = os.path.join(output_dir, f"{file_prefix}_runs.csv")
    scale_summary_csv = os.path.join(output_dir, f"{file_prefix}_summary_by_scale.csv")
    due_summary_csv = os.path.join(output_dir, f"{file_prefix}_summary_by_due.csv")
    size_summary_csv = os.path.join(output_dir, f"{file_prefix}_summary_by_size.csv")

    def safe_to_csv(df, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_csv(path, index=False)

    safe_to_csv(detail_df, detail_csv)
    safe_to_csv(run_df, runs_csv)
    safe_to_csv(per_scale_summary, scale_summary_csv)
    safe_to_csv(due_summary, due_summary_csv)
    safe_to_csv(size_summary, size_summary_csv)

    schedule_dir = os.path.join(output_dir, "schedule_details")
    gantt_dir = os.path.join(output_dir, "gantt")
    os.makedirs(schedule_dir, exist_ok=True)
    os.makedirs(gantt_dir, exist_ok=True)
    for payload in schedule_payloads:
        safe_name = f"{payload['scale']}_{payload['instance']}_best_run{payload['best_run']:02d}"
        schedule_csv = os.path.join(schedule_dir, f"{safe_name}_schedule.csv")
        gantt_png = os.path.join(gantt_dir, f"{safe_name}_gantt.png")
        schedule_df = pd.DataFrame(payload["rows"])
        safe_to_csv(schedule_df, schedule_csv)
        plot_ortools_gantt_with_due_dates(
            payload["rows"],
            gantt_png,
            title=f"PPO {payload['scale']} {payload['instance']} best run {payload['best_run']} ({payload['due_setting']})",
        )

    print("\n--- Summary by scale ---")
    print(per_scale_summary)
    print(f"\nRun-level results: {runs_csv}")
    print(f"Instance details: {detail_csv}")
    print(f"Summary by scale: {scale_summary_csv}")
    print(f"Summary by due: {due_summary_csv}")
    print(f"Summary by size: {size_summary_csv}")
    print(f"Best-run schedule details: {schedule_dir}")
    print(f"Best-run gantt charts: {gantt_dir}")


if __name__ == "__main__":
    evaluate_range_due_test()
