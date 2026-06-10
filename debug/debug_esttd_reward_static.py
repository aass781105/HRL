import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common_utils import greedy_select_action, sample_action
from data_utils import text_to_matrix
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from model.PPO import PPO_initialize
from params import configs


DEFAULT_BASE_DIR = "or_instances_uniform_test_30_50_due_scaled"


def load_due_dates_json(due_data):
    due_dates = due_data["due_dates"]
    if isinstance(due_dates, dict):
        return np.asarray([due_dates[str(i)] for i in sorted(map(int, due_dates.keys()))], dtype=np.float64)
    return np.asarray(due_dates, dtype=np.float64)


def find_default_instance():
    for root, _, files in os.walk(DEFAULT_BASE_DIR):
        for name in sorted(files):
            if not name.endswith(".fjs"):
                continue
            fjs_path = os.path.join(root, name)
            json_path = os.path.splitext(fjs_path)[0] + ".json"
            if os.path.exists(json_path):
                return fjs_path, json_path
    raise FileNotFoundError(f"No .fjs/.json pair found under {DEFAULT_BASE_DIR}")


def machine_est_completion_summary(env, env_idx, job_idx, op_idx):
    values = []
    chosen_value = None
    for m in range(env.number_of_machines):
        if env.reverse_process_relation[env_idx, op_idx, m]:
            continue
        earliest = max(
            float(env.true_candidate_free_time[env_idx, job_idx]),
            float(env.true_release_time[env_idx, job_idx]),
        )
        pt = float(env.true_op_pt[env_idx, op_idx, m])
        if env.enable_gap_insertion:
            start = env._find_earliest_gap(env.true_machine_calendars[env_idx][m], earliest, pt)
        else:
            start = max(earliest, float(env.true_mch_free_time[env_idx, m]))
        completion = start + pt
        values.append((m, completion))
    if not values:
        return np.nan, np.nan, np.nan, np.nan
    completions = np.asarray([v for _, v in values], dtype=np.float64)
    return float(np.mean(completions)), float(np.min(completions)), float(np.max(completions)), values


def estimate_current_job_td(env, env_idx, job_idx):
    if env.mask[env_idx, job_idx]:
        curr = max(0.0, float(env.true_candidate_free_time[env_idx, job_idx] - env.true_due_date[env_idx, job_idx]))
        return curr, float(env.true_candidate_free_time[env_idx, job_idx]), 1.0

    op_idx = int(env.candidate[env_idx, job_idx])
    first_op = int(env.job_first_op_id[env_idx, job_idx])
    total_work = max(float(env.true_job_total_work[env_idx, job_idx]), 1e-6)
    prefix_work = float(np.sum(env.true_op_mean_pt[env_idx, first_op:op_idx + 1]))
    accuracy_rate = min(max(prefix_work / total_work, 0.0), 1.0)
    c_est = float(env._estimate_op_completion_mean(env_idx, job_idx, op_idx, true_time=True))
    est_td = max(0.0, accuracy_rate * (c_est - float(env.true_due_date[env_idx, job_idx])))
    return est_td, c_est, accuracy_rate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)

    fjs_path = str(getattr(configs, "debug_instance_path", "") or "").strip()
    due_path = str(getattr(configs, "debug_due_path", "") or "").strip()
    if fjs_path:
        if not due_path:
            due_path = os.path.splitext(fjs_path)[0] + ".json"
    else:
        fjs_path, due_path = find_default_instance()

    if not os.path.exists(fjs_path):
        raise FileNotFoundError(f"Instance not found: {fjs_path}")
    if not os.path.exists(due_path):
        raise FileNotFoundError(f"Due JSON not found: {due_path}")
    if not os.path.exists(configs.ppo_model_path):
        raise FileNotFoundError(f"PPO model not found: {configs.ppo_model_path}")

    with open(fjs_path, "r", encoding="utf-8") as f:
        jl, pt = text_to_matrix(f.readlines())
    with open(due_path, "r", encoding="utf-8") as f:
        due_data = json.load(f)
    due_dates = load_due_dates_json(due_data)

    n_j = int(jl.shape[0])
    n_m = int(pt.shape[1])
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(configs.ppo_model_path, map_location=device, weights_only=True))
    ppo.policy.to(device)
    ppo.policy.eval()

    env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(
        job_length_list=[jl],
        op_pt_list=[pt],
        due_date_list=[due_dates],
        true_due_date_list=[due_dates],
    )

    action_mode = str(getattr(configs, "debug_action_mode", "") or getattr(configs, "eval_action_selection", "sample")).lower()
    action_mode = "greedy" if action_mode == "greedy" else "sample"

    rows = []
    step = 0
    while not env.done_flag.all():
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
            if action_mode == "greedy":
                action = greedy_select_action(pi)
            else:
                action, _ = sample_action(pi)

        action_id = int(action.cpu().numpy().reshape(-1)[0])
        job_id = int(action_id // n_m)
        machine_id = int(action_id % n_m)
        op_global_id = int(env.candidate[0, job_id])
        op_id_in_job = int(op_global_id - env.job_first_op_id[0, job_id])
        is_last_op = bool(op_global_id == env.job_last_op_id[0, job_id])
        proc_time = float(env.true_op_pt[0, op_global_id, machine_id])
        due_date = float(env.true_due_date[0, job_id])
        prev_est_td = float(env.chosen_est_tardiness_baseline[0, job_id])

        chosen_est_mean, min_est_completion, max_est_completion, machine_values = machine_est_completion_summary(
            env, 0, job_id, op_global_id
        )
        chosen_machine_est_completion = np.nan
        if isinstance(machine_values, list):
            for m, value in machine_values:
                if int(m) == machine_id:
                    chosen_machine_est_completion = float(value)
                    break

        state, reward, done, info = env.step(action.cpu().numpy())
        curr_est_td, est_completion_mean, accuracy_rate = estimate_current_job_td(env, 0, job_id)
        est_td_increase = max(0.0, curr_est_td - prev_est_td)
        td_reward_raw = -est_td_increase
        td_reward_scaled = td_reward_raw / max(float(env.mean_op_pt), 1e-6)
        actual_terminal_td = max(0.0, float(env.true_candidate_free_time[0, job_id] - due_date)) if is_last_op else 0.0
        mk_reward_step = float(np.asarray(info.get("reward_mk_step", [0.0]))[0])
        td_reward_step = float(np.asarray(info.get("reward_td_step", [0.0]))[0])
        total_reward = float(np.asarray(reward)[0])
        td_abs_share = abs(td_reward_step) / (abs(mk_reward_step) + abs(td_reward_step) + 1e-12)
        detail = info.get("scheduled_op_details_all", [None])[0] or info.get("scheduled_op_details", {})

        rows.append({
            "step": step + 1,
            "job_id": job_id,
            "op_id_in_job": op_id_in_job,
            "machine_id": machine_id,
            "is_last_op": int(is_last_op),
            "start_time": float(detail.get("start_time", np.nan)),
            "end_time": float(detail.get("end_time", np.nan)),
            "proc_time": proc_time,
            "due_date": due_date,
            "est_completion_mean": est_completion_mean,
            "accuracy_rate": accuracy_rate,
            "est_td_prev": prev_est_td,
            "est_td_curr": curr_est_td,
            "est_td_increase": est_td_increase,
            "td_reward_raw": td_reward_raw,
            "td_reward_scaled": td_reward_scaled,
            "actual_terminal_td": actual_terminal_td,
            "mk_reward_step": mk_reward_step,
            "td_reward_step": td_reward_step,
            "total_reward": total_reward,
            "td_abs_share": td_abs_share,
            "chosen_machine_est_completion": chosen_machine_est_completion,
            "min_machine_est_completion": min_est_completion,
            "max_machine_est_completion": max_est_completion,
        })
        step += 1

    step_df = pd.DataFrame(rows)
    td_abs = step_df["td_reward_step"].abs()
    last_td_abs = td_abs[step_df["is_last_op"] == 1].sum()
    nonlast_td_abs = td_abs[step_df["is_last_op"] == 0].sum()
    summary = pd.DataFrame([{
        "instance_path": fjs_path,
        "due_path": due_path,
        "model_path": configs.ppo_model_path,
        "ll_td_mode": getattr(configs, "ll_td_mode", ""),
        "enable_gap_insertion": bool(getattr(configs, "enable_gap_insertion", False)),
        "action_mode": action_mode,
        "makespan": float(env.current_makespan[0]),
        "total_tardiness": float(env.accumulated_tardiness[0]),
        "sum_total_reward": float(step_df["total_reward"].sum()),
        "sum_mk_reward": float(step_df["mk_reward_step"].sum()),
        "sum_td_reward": float(step_df["td_reward_step"].sum()),
        "td_nonzero_steps": int((td_abs > 1e-12).sum()),
        "td_nonzero_ratio": float((td_abs > 1e-12).mean()),
        "last_op_td_reward_sum_abs": float(last_td_abs),
        "nonlast_op_td_reward_sum_abs": float(nonlast_td_abs),
        "last_op_td_share": float(last_td_abs / (last_td_abs + nonlast_td_abs + 1e-12)),
        "mean_td_abs_share": float(step_df["td_abs_share"].mean()),
        "max_est_td_increase": float(step_df["est_td_increase"].max()),
    }])

    out_dir = os.path.abspath(str(getattr(configs, "debug_output_dir", "debug/esttd_reward_runs")))
    os.makedirs(out_dir, exist_ok=True)
    model_name = os.path.basename(configs.ppo_model_path).replace(".pth", "")
    instance_name = os.path.splitext(os.path.basename(fjs_path))[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    step_csv = os.path.join(out_dir, f"steps_{model_name}_{instance_name}_{stamp}.csv")
    summary_csv = os.path.join(out_dir, f"summary_{model_name}_{instance_name}_{stamp}.csv")
    step_df.to_csv(step_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"Debug EstTD reward finished.")
    print(f"Steps CSV: {step_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
