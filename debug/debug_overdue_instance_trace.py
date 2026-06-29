import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SETTINGS = {
    "config": "",
    "n_j": 50,
    "n_m": 5,
    "seed": 42,
    "sample_seed": 42,
    "overdue_jobs": 2,
    "action_mode": "greedy",  # "greedy" or "sample"
    "model_path": os.path.join("trained_network", "SD2", "ll_u1030_esttd_odprog.pth"),
    "output_dir": os.path.join("debug", "overdue_instance_trace"),
    "device": "",  # Empty means cuda if available, else cpu.
    "low": 1,
    "high": 99,
    "op_per_job": 5,
    "ll_due_range_scale": 0.7,
    "ll_range3_overdue_factor_low": -1.5,
    "ll_range3_overdue_factor_high": -0.7,
    "ll_overdue_progress_coef": 0.5,
    "ll_td_mode": "chosen_est_tardiness_delta",
    "ll_mk_coef": 10.0,
    "ll_td_coef": 1.0,
    "enable_gap_insertion": False,
}

sys.argv = [sys.argv[0]]
if SETTINGS["config"]:
    sys.argv.extend(["--config", SETTINGS["config"]])

from data_utils import SD2_instance_generator, generate_due_dates
from ll_fjsp_env import LLFJSPEnv
from model.ll_ppo import ll_ppo_initialize
from ortools_gantt import plot_ortools_gantt_with_due_dates
from params import configs

sys.argv = [sys.argv[0]]


class Settings:
    def __init__(self, values):
        self.__dict__.update(values)


def extract_schedule_rows(env, env_idx=0):
    rows = []
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
            due_date = float(due_dates[job])
            completion = float(job_completion[job])
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


def build_overdue_instance(seed, n_j, n_m, overdue_jobs):
    rng = np.random.default_rng(int(seed))
    old_n_j, old_n_m = int(configs.n_j), int(configs.n_m)
    configs.n_j = int(n_j)
    configs.n_m = int(n_m)
    try:
        job_length, op_pt, _ = SD2_instance_generator(configs, rng=rng)
        due_dates = generate_due_dates(job_length, op_pt, due_date_mode="range3_mixed", rng=rng)
    finally:
        configs.n_j = old_n_j
        configs.n_m = old_n_m

    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    due_range = float(getattr(configs, "ll_due_range_scale", 0.7)) * int(n_j) * mean_pt
    overdue_count = max(1, min(int(overdue_jobs), int(n_j)))
    overdue_job_ids = rng.choice(np.arange(int(n_j)), size=overdue_count, replace=False)
    factors = rng.uniform(
        min(float(getattr(configs, "ll_range3_overdue_factor_low", -1.5)), float(getattr(configs, "ll_range3_overdue_factor_high", -0.7))),
        max(float(getattr(configs, "ll_range3_overdue_factor_low", -1.5)), float(getattr(configs, "ll_range3_overdue_factor_high", -0.7))),
        size=overdue_count,
    )
    for jid, factor in zip(overdue_job_ids, factors):
        due_dates[int(jid)] = float(factor) * due_range

    return job_length, op_pt, due_dates, {int(j): float(f) for j, f in zip(overdue_job_ids, factors)}


def candidate_trace_rows(env, pi_np, step_idx):
    rows = []
    env_idx = 0
    n_m = int(env.number_of_machines)
    probs = pi_np.reshape(int(env.number_of_jobs), n_m)
    candidate_ops = np.asarray(env.candidate[env_idx], dtype=int)
    due_dates = np.asarray(env.true_due_date[env_idx], dtype=np.float64)
    current_time = float(env.next_schedule_time[env_idx])

    flat_probs = probs.reshape(-1)
    order = np.argsort(-flat_probs)
    rank_by_action = np.empty_like(order)
    rank_by_action[order] = np.arange(1, len(order) + 1)

    for job in range(int(env.number_of_jobs)):
        op_id = int(candidate_ops[job])
        if bool(env.mask[env_idx, job]):
            continue
        remain_work = float(env.true_job_remain_work[env_idx][job])
        due = float(due_dates[job])
        slack = due - current_time - remain_work
        overdue_weight = float(np.log1p(max(0.0, -due) / max(float(env.mean_op_pt), 1e-8)))
        for machine in range(n_m):
            action_id = job * n_m + machine
            if bool(env.dynamic_pair_mask[env_idx, job, machine]):
                continue
            pt = float(env.true_op_pt[env_idx, op_id, machine])
            rows.append({
                "step": int(step_idx),
                "job": int(job),
                "op_global": int(op_id),
                "op_in_job": int(op_id - env.job_first_op_id[env_idx, job]),
                "machine": int(machine),
                "prob": float(probs[job, machine]),
                "rank": int(rank_by_action[action_id]),
                "pt": pt,
                "due": due,
                "current_time": current_time,
                "remain_work": remain_work,
                "slack": float(slack),
                "is_due_negative": int(due < 0.0),
                "overdue_weight": overdue_weight,
                "pair_est_lateness_state": float(env.fea_pairs[env_idx, job, machine, -1]),
            })
    return rows


def run_trace(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configs.device = str(device)
    configs.n_j = int(args.n_j)
    configs.n_m = int(args.n_m)
    configs.low = int(args.low)
    configs.high = int(args.high)
    configs.op_per_job = int(args.op_per_job)
    configs.fea_j_input_dim = 20
    configs.fea_m_input_dim = 9
    configs.fea_pair_input_dim = 8
    configs.ll_due_range_scale = float(args.ll_due_range_scale)
    configs.ll_range3_overdue_factor_low = float(args.ll_range3_overdue_factor_low)
    configs.ll_range3_overdue_factor_high = float(args.ll_range3_overdue_factor_high)
    configs.ll_overdue_progress_coef = float(args.ll_overdue_progress_coef)
    configs.ll_td_mode = str(args.ll_td_mode)
    configs.ll_mk_coef = float(args.ll_mk_coef)
    configs.ll_td_coef = float(args.ll_td_coef)
    configs.enable_gap_insertion = bool(args.enable_gap_insertion)

    job_length, op_pt, due_dates, injected = build_overdue_instance(
        seed=args.seed,
        n_j=args.n_j,
        n_m=args.n_m,
        overdue_jobs=args.overdue_jobs,
    )

    ppo = ll_ppo_initialize()
    model_path = args.model_path or str(getattr(configs, "ll_ppo_model_path", ""))
    loaded_model = False
    if model_path and os.path.exists(model_path):
        ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        loaded_model = True
    ppo.policy.to(device)
    ppo.policy.eval()

    torch.manual_seed(int(args.sample_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.sample_seed))

    env = LLFJSPEnv(n_j=args.n_j, n_m=args.n_m)
    state = env.set_initial_data(
        job_length_list=[job_length],
        op_pt_list=[op_pt],
        due_date_list=[due_dates],
        true_due_date_list=[due_dates],
    )

    decision_rows = []
    candidate_rows = []
    step_idx = 0
    done = False
    while not bool(done):
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
        pi_np = pi.detach().cpu().numpy()[0]
        candidate_rows.extend(candidate_trace_rows(env, pi_np, step_idx))

        if args.action_mode == "sample":
            action_t = torch.distributions.Categorical(probs=pi).sample().view(1, 1)
        else:
            action_t = torch.argmax(pi, dim=1).view(1, 1)
        action = int(action_t.item())
        chosen_job = action // int(args.n_m)
        chosen_machine = action % int(args.n_m)
        chosen_op = int(env.candidate[0, chosen_job])
        due = float(env.true_due_date[0, chosen_job])
        current_time = float(env.next_schedule_time[0])
        remain_work = float(env.true_job_remain_work[0][chosen_job])
        slack = due - current_time - remain_work
        overdue_weight = float(np.log1p(max(0.0, -due) / max(float(env.mean_op_pt), 1e-8)))
        chosen_prob = float(pi_np[action])
        sorted_probs = np.sort(pi_np)[::-1]
        chosen_rank = int(np.where(np.argsort(-pi_np) == action)[0][0] + 1)

        state, reward, done_arr, info = env.step(action_t.cpu().numpy())
        done = bool(done_arr[0])
        detail = info.get("scheduled_op_details", {})
        decision_rows.append({
            "step": int(step_idx),
            "action_mode": args.action_mode,
            "chosen_job": int(chosen_job),
            "chosen_op_global": int(chosen_op),
            "chosen_op_in_job": int(chosen_op - env.job_first_op_id[0, chosen_job]),
            "chosen_machine": int(chosen_machine),
            "chosen_prob": chosen_prob,
            "chosen_rank": chosen_rank,
            "top1_prob": float(sorted_probs[0]) if sorted_probs.size else 0.0,
            "due": due,
            "current_time_before": current_time,
            "remain_work_before": remain_work,
            "slack_before": float(slack),
            "is_due_negative": int(due < 0.0),
            "overdue_weight": overdue_weight,
            "reward": float(reward[0]),
            "reward_mk_step": float(np.asarray(info.get("reward_mk_step", [0.0]))[0]),
            "reward_td_step": float(np.asarray(info.get("reward_td_step", [0.0]))[0]),
            "reward_od_step": float(np.asarray(info.get("reward_od_step", [0.0]))[0]),
            "start_time": float(detail.get("start_time", np.nan)),
            "end_time": float(detail.get("end_time", np.nan)),
            "proc_time": float(detail.get("proc_time", np.nan)),
        })
        step_idx += 1

    schedule_rows = extract_schedule_rows(env, env_idx=0)
    stem = f"overdue_trace_j{args.n_j}_m{args.n_m}_seed{args.seed}_{args.action_mode}"
    decision_csv = os.path.join(args.output_dir, f"{stem}_decisions.csv")
    candidates_csv = os.path.join(args.output_dir, f"{stem}_candidates.csv")
    schedule_csv = os.path.join(args.output_dir, f"{stem}_schedule.csv")
    gantt_png = os.path.join(args.output_dir, f"{stem}_gantt.png")
    summary_csv = os.path.join(args.output_dir, f"{stem}_summary.csv")

    pd.DataFrame(decision_rows).to_csv(decision_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(schedule_rows).to_csv(schedule_csv, index=False, encoding="utf-8-sig")
    plot_ortools_gantt_with_due_dates(schedule_rows, gantt_png, title=f"PPO overdue trace {args.action_mode}")

    summary = {
        "model_path": model_path,
        "loaded_model": loaded_model,
        "n_j": int(args.n_j),
        "n_m": int(args.n_m),
        "seed": int(args.seed),
        "sample_seed": int(args.sample_seed),
        "action_mode": args.action_mode,
        "makespan": float(env.current_makespan[0]),
        "total_tardiness": float(env.accumulated_tardiness[0]),
        "objective": float(0.5 * env.current_makespan[0] + 0.5 * env.accumulated_tardiness[0]),
        "injected_overdue_jobs": ";".join(f"{job}:{factor:.4f}" for job, factor in sorted(injected.items())),
        "decision_csv": decision_csv,
        "candidates_csv": candidates_csv,
        "schedule_csv": schedule_csv,
        "gantt_png": gantt_png,
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Loaded model: {loaded_model} | {model_path}")
    print(f"Summary: {summary_csv}")
    print(f"Decisions: {decision_csv}")
    print(f"Candidates: {candidates_csv}")
    print(f"Schedule: {schedule_csv}")
    print(f"Gantt: {gantt_png}")


if __name__ == "__main__":
    run_trace(Settings(SETTINGS))
