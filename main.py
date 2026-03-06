import os
import time
import copy
import csv
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List

from params import configs
from common_utils import *
from global_env import (
    GlobalTimelineOrchestrator,
    EventBurstGenerator,
    split_matrix_to_jobs,
)
from data_utils import SD2_instance_generator, generate_due_dates

# Plotting
from gantt import plot_global_gantt

import torch
import torch.nn as nn
from model.ddqn_model import QNet, calculate_ddqn_state, log_scale_reward

# -----------------------------------------------------------------------------

def fixed_k_sampler(K: int):
    def _fn(rng: np.random.Generator) -> int:
        return int(K)
    return _fn

def run_event_driven_until_nevents(*, max_events: int, interarrival_mean: float, burst_K: int = 1, plot_global_dir: Optional[str] = None):
    seed = int(getattr(configs, "event_seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    base_cfg = copy.deepcopy(configs); rng = np.random.default_rng(seed)
    gen = EventBurstGenerator(SD2_instance_generator, base_cfg, configs.n_m, interarrival_mean, fixed_k_sampler(int(burst_K)), rng)
    orch = GlobalTimelineOrchestrator(configs.n_m, gen, t0=0.0)
    
    gate_policy = str(getattr(configs, "gate_policy", "ddqn")).lower()
    is_ddqn = (gate_policy == "ddqn")
    shadow_orch = None

    all_job_due_dates: Dict[int, float] = {}
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    reward_scale = mean_pt
    stability_scale, buffer_penalty_coef, release_penalty_coef = float(getattr(configs, "stability_scale", 0.0)), float(getattr(configs, "buffer_penalty_coef", 0.05)), float(getattr(configs, "release_penalty_coef", 0.5))

    ddqn_model = None; ddqn_device = torch.device(getattr(configs, "device", "cpu"))
    if gate_policy == "ddqn":
        ddqn_model = QNet(obs_dim=18, hidden=128, n_actions=2).to(ddqn_device)
        try:
            ddqn_model.load_state_dict(torch.load(getattr(configs, "ddqn_model_path", ""), map_location=ddqn_device)); ddqn_model.eval()
            print(f"[DDQN] Loaded weights.")
        except:
            print(f"[WARN] Fallback."); gate_policy = "cadence"

    csv_dir = plot_global_dir or "plots/global"; os.makedirs(csv_dir, exist_ok=True)
    suffix = f"DDQN_{getattr(configs, 'ddqn_name', 'default')}" if gate_policy=="ddqn" else f"Cadence_{getattr(configs, 'gate_cadence', 1)}"
    csv_file = open(os.path.join(csv_dir, f"log_{suffix}.csv"), "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    
    csv_headers = ["Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str", "Buffer_Count", "Avg_Load", "Min_Load", "Max_Load", "Load_Std", "Weighted_Idle", "Unweighted_Idle", "Buf_NegSlack_Ratio", "Buf_Min_Slack", "Buf_Avg_Slack", "Buf_Slack_Std", "WIP_Job_Count", "WIP_Tardy_Ratio", "WIP_Min_Slack", "WIP_Avg_Slack", "WIP_Slack_Std", "WIP_Planned_TD", "WIP_Theoretical_TD", "Batch_Jobs", "Batch_Ops", "Actual_TD", "Shadow_TD", "History_Tardiness", "Reward_Total", "Reward_Idle", "Reward_Stab", "Reward_Buffer", "Reward_Release", "Reward_Flush", "Final_Makespan", "Final_Tardiness", "Release_Count"]
    csv_writer.writerow(csv_headers)

    release_count, t_prev_reward, last_act, last_advantage_td, plot_seq = 0, 0.0, None, 0.0, 0
    total_cumulative_reward = 0.0; pending_row_base = None

    def get_raw_state_info(orchestrator, t_now):
        b_slacks, b_neg = [], 0
        for j in orchestrator.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0)); due = all_job_due_dates[j.job_id]; s = due - t_now - mw
            b_slacks.append(s); 
            if t_now + mw > due: b_neg += 1
        b_stats = (b_neg/len(orchestrator.buffer), min(b_slacks), sum(b_slacks)/len(b_slacks), np.std(b_slacks)) if orchestrator.buffer else (0.0, 0.0, 0.0, 0.0)
        wip = orchestrator.get_wip_stats(t_now); mft = np.asarray(orchestrator.machine_free_time, dtype=float); rem = np.maximum(0.0, mft - t_now); mx_l = np.max(rem)
        w_idle = orchestrator.compute_weighted_idle(t_now, float(mx_l)) if mx_l>0 else 0.0
        u_idle = orchestrator.compute_unweighted_idle(t_now, float(mx_l)) if mx_l>0 else 0.0
        return [len(orchestrator.buffer), np.mean(rem), np.min(rem), mx_l, np.std(rem), w_idle, u_idle, b_stats[0], b_stats[1], b_stats[2], b_stats[3], wip["wip_count"], wip["wip_tardy_ratio"], wip["wip_min_slack"], wip["wip_avg_slack"], wip["wip_slack_std"], wip["planned_td"], wip["theoretical_td"]]

    if int(getattr(configs, "init_jobs", 0)) > 0:
        jl, pt, _ = SD2_instance_generator(copy.deepcopy(configs), rng=rng)
        dd_rel = generate_due_dates(jl, pt, tightness=getattr(configs, "due_date_tightness", 1.2), due_date_mode='k', rng=rng)
        init_jobs = split_matrix_to_jobs(jl, pt, base_job_id=0, t_arrive=0.0, due_dates=0.0 + dd_rel)
        for j in init_jobs: all_job_due_dates[j.job_id] = j.meta["due_date"]
        orch.buffer.extend(init_jobs); gen.bump_next_id(max((j.job_id for j in init_jobs)) + 1); orch.event_release_and_reschedule(0.0); release_count += 1
        if is_ddqn: shadow_orch = orch.clone()
        raw_s0 = get_raw_state_info(orch, 0.0); last_act = 1
        pending_row_base = [0, "0.00", "0.00", 1, "RELEASE"] + raw_s0 + [len(set(r['job'] for r in orch._last_full_rows)), len(orch._last_full_rows), "0.00", "0.00", "0.00"]
        plot_seq += 1; plot_global_gantt([dict(r, phase="history" if r["start"] < 0.0 else "newplan") for r in (orch._global_rows + orch._last_full_rows)], os.path.join(csv_dir, f"global_r{plot_seq:03d}_t0.png"), t_now=0.0, title="Initial")

    stats = {"arrive": 0}; t_now, t_prev = 0.0, 0.0; t_next = gen.sample_next_time(t_now)
    while stats["arrive"] < int(max_events):
        t_now = float(t_next); inter_arrival = t_now - t_prev
        if last_act is not None and pending_row_base:
            # [REWARD CALCULATION: ONLY FOR DDQN ANALYSIS]
            if is_ddqn:
                metrics = orch.compute_interval_metrics(t_prev_reward, t_now); scale = reward_scale
                r_idle = -(metrics["total_idle"] * float(configs.idle_penalty_coef)) / scale
                r_stab = -float(last_act) * stability_scale; dt = t_now - t_prev_reward
                r_buf = -(len(orch.buffer) * dt * buffer_penalty_coef) / scale
                r_rel_treated = -(last_advantage_td * release_penalty_coef) / scale
                if r_rel_treated > 0: r_rel_treated *= 0.5
                step_reward = r_idle + r_stab + r_buf + r_rel_treated
            else:
                r_idle = r_stab = r_buf = r_rel_treated = step_reward = 0.0

            total_cumulative_reward += step_reward
            csv_writer.writerow(pending_row_base + [f"{step_reward:.4f}", f"{r_idle:.4f}", f"{r_stab:.4f}", f"{r_buf:.4f}", f"{r_rel_treated:.4f}", "0.0000", release_count])
            t_prev_reward = t_now

        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            for j in new_jobs: all_job_due_dates[j.job_id] = j.meta["due_date"]
            orch.buffer.extend(new_jobs)
            if is_ddqn: shadow_orch.buffer.extend(copy.deepcopy(new_jobs))
            stats["arrive"] += 1
        
        # [SHADOW ALWAYS RELEASES: ONLY FOR DDQN]
        if is_ddqn: shadow_orch.event_release_and_reschedule(t_now)
        
        raw_s = get_raw_state_info(orch, t_now); b_dict = {"tardiness_ratio": raw_s[7], "min_slack": raw_s[8], "avg_slack": raw_s[9], "slack_std": raw_s[10]}; w_dict = {"wip_count": raw_s[11], "wip_tardy_ratio": raw_s[12], "wip_min_slack": raw_s[13], "wip_avg_slack": raw_s[14], "wip_slack_std": raw_s[15], "planned_td": raw_s[16], "theoretical_td": raw_s[17]}
        
        if is_ddqn:
            obs = calculate_ddqn_state(len(orch.buffer), orch.machine_free_time, t_now, configs.n_m, 0, reward_scale, raw_s[5], raw_s[6], b_dict, w_dict)
            act = ddqn_model(torch.from_numpy(obs).float().unsqueeze(0).to(ddqn_device)).argmax(dim=1).item()
        else:
            act = 1 if (stats["arrive"] % configs.gate_cadence == 0) else 0

        old_td_est = orch.get_total_tardiness_estimate()
        shadow_td_est = shadow_orch.get_total_tardiness_estimate() if is_ddqn else 0.0
        pending_row_base = [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, "RELEASE" if act==1 else "HOLD"] + raw_s + [0, 0, f"{old_td_est:.2f}", f"{shadow_td_est:.2f}", "0.00"]
        
        if act == 1:
            orch.event_release_and_reschedule(t_now); release_count += 1
            actual_td_after = orch.get_total_tardiness_estimate()
            if is_ddqn:
                last_advantage_td = actual_td_after - shadow_td_est
                shadow_orch = orch.clone()
            else:
                last_advantage_td = 0.0
            
            # Indices: 23:Batch_Jobs, 24:Batch_Ops, 25:Actual_TD, 26:Shadow_TD
            pending_row_base[23], pending_row_base[24], pending_row_base[25], pending_row_base[26] = len(set(r['job'] for r in orch._last_full_rows)), len(orch._last_full_rows), f"{actual_td_after:.2f}", f"{shadow_td_est:.2f}"
            plot_seq += 1; plot_global_gantt([dict(r, phase="history" if r["start"] < t_now else "newplan") for r in (orch._global_rows + orch._last_full_rows)], os.path.join(csv_dir, f"global_r{plot_seq:03d}_t{int(t_now):05d}.png"), t_now=t_now, title=f"Event #{stats['arrive']}")
        else:
            orch.tick_without_release(t_now)
            actual_td_now = orch.get_total_tardiness_estimate()
            last_advantage_td = 0.0
            pending_row_base[25], pending_row_base[26] = f"{actual_td_now:.2f}", f"{shadow_td_est:.2f}"
        
        last_act = act; t_prev = t_now; t_next = gen.sample_next_time(t_now)

    while len(orch.buffer) > 0:
        orch.event_release_and_reschedule(t_now); release_count += 1; plot_seq += 1; plot_global_gantt([dict(r, phase="history" if r["start"] < t_now else "newplan") for r in (orch._global_rows + orch._last_full_rows)], os.path.join(csv_dir, f"global_r{plot_seq:03d}_FLUSH.png"), t_now=t_now, title="FLUSH")

    final_stats = orch.get_final_kpi_stats(all_job_due_dates); total_td, final_mk = final_stats["tardiness"], final_stats["makespan"]; r_flush = (-final_mk / reward_scale) * float(configs.flush_penalty_coef); total_cumulative_reward += r_flush
    summary_line = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""]*18 + ["", "", "", "", f"{total_td:.2f}", f"{total_cumulative_reward:.4f}", "", "", "", "", f"{r_flush:.4f}", f"{final_mk:.2f}", f"{total_td:.2f}", release_count]
    csv_writer.writerow(summary_line); csv_file.close(); return final_mk, {"release_count": release_count, "total_tardiness": total_td}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))
    print("-" * 20 + " Dynamic FJSP HRL Evaluation " + "-" * 20)
    
    # [FIXED] Dynamic output naming based on actual policy
    if configs.gate_policy == "ddqn":
        output_name = f"DDQN_{getattr(configs, 'ddqn_name', 'default')}"
    else:
        output_name = f"Cadence_{getattr(configs, 'gate_cadence', 1)}"
        
    plot_dir = os.path.join("plots/global", output_name)
    
    mk, stats = run_event_driven_until_nevents(
        max_events=int(configs.event_horizon), 
        interarrival_mean=configs.interarrival_mean, 
        burst_K=configs.burst_size, 
        plot_global_dir=plot_dir
    )
    print(f"\nMakespan: {mk:.3f}, Tardiness: {stats['total_tardiness']:.3f}, Releases: {stats['release_count']}")

if __name__ == "__main__": main()
