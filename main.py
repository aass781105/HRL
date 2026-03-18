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
from plot_utils import plot_simulation_summary_stats

import torch
import torch.nn as nn
from model.ddqn_model import QNet, calculate_ddqn_state, log_scale_reward

# -----------------------------------------------------------------------------

def fixed_k_sampler(K: int):
    def _fn(rng: np.random.Generator) -> int:
        return int(K)
    return _fn

def run_event_driven_until_nevents(*, max_events: int, interarrival_mean: float, burst_K: int = 1, plot_global_dir: Optional[str] = None):
    # [FAST MODE] Skip heavy I/O tasks if enabled
    FAST_MODE = getattr(configs, "fast_mode", True)
    all_sim_job_stats = [] # Store {due_date, slack}

    seed = int(getattr(configs, "event_seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    base_cfg = copy.deepcopy(configs); rng = np.random.default_rng(seed)
    gen = EventBurstGenerator(SD2_instance_generator, base_cfg, configs.n_m, interarrival_mean, fixed_k_sampler(int(burst_K)), rng)
    orch = GlobalTimelineOrchestrator(configs.n_m, gen, t0=0.0)
    
    gate_policy = str(getattr(configs, "gate_policy", "ddqn")).lower()
    is_ddqn = (gate_policy == "ddqn")

    all_job_due_dates: Dict[int, float] = {}
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    reward_scale = mean_pt
    stability_scale = float(getattr(configs, "stability_scale", 0.0))
    buffer_penalty_coef = float(getattr(configs, "buffer_penalty_coef", 0.0))
    shaping_reward_coef = float(getattr(configs, "shaping_reward_coef", getattr(configs, "release_penalty_coef", 0.1)))
    terminal_reward_coef = float(getattr(configs, "terminal_reward_coef", getattr(configs, "release_penalty_coef", 0.1)))

    def compress_rel_tail(raw_value: float, threshold: float = 2.0, tail_scale: float = 1.0) -> float:
        abs_value = abs(float(raw_value))
        if abs_value <= threshold:
            return float(raw_value)
        tail = np.log1p((abs_value - threshold) / tail_scale)
        return float(np.sign(raw_value) * (threshold + tail))

    def collect_subproblem_stats(jobs, t_now):
        """Records dynamic due date and slack for jobs in the current PPO subproblem."""
        for j in jobs:
            due_abs = float(j.meta.get("due_date", 0.0))
            ready_abs = float(j.meta.get("ready_at", t_now))
            # Calculate remaining work based on currently pending operations
            rem_work = 0.0
            for op in j.operations:
                v = np.array(op.time_row)
                rem_work += np.mean(v[v > 0]) if v[v > 0].size else 0.0

            # Record relative values as seen by PPO (before scaling)
            all_sim_job_stats.append({
                'due_date': due_abs - t_now,
                'slack': due_abs - ready_abs - rem_work
            })

    ddqn_model = None; ddqn_device = torch.device(getattr(configs, "device", "cpu"))
    if gate_policy == "ddqn":
        ddqn_model = QNet(
            obs_dim=18,
            n_actions=2,
            hidden=configs.ddqn_hidden_dim,
            num_layers=configs.ddqn_num_layers,
            dueling=bool(getattr(configs, "ddqn_dueling", True)),
        ).to(ddqn_device)
        try:
            ddqn_model.load_state_dict(torch.load(getattr(configs, "ddqn_model_path", ""), map_location=ddqn_device, weights_only=True)); ddqn_model.eval()
            print(f"[DDQN] Loaded weights.")
        except:
            print(f"[WARN] Fallback."); gate_policy = "cadence"; is_ddqn = False

    csv_dir = plot_global_dir or "plots/global"; os.makedirs(csv_dir, exist_ok=True)
    suffix = f"DDQN_{getattr(configs, 'ddqn_name', 'default')}" if gate_policy=="ddqn" else f"Cadence_{getattr(configs, 'gate_cadence', 1)}"
    raw_csv_file = open(os.path.join(csv_dir, f"log_{suffix}_raw_state.csv"), "w", newline="", encoding="utf-8")
    raw_csv_writer = csv.writer(raw_csv_file)
    obs_csv_file = open(os.path.join(csv_dir, f"log_{suffix}_agent_state.csv"), "w", newline="", encoding="utf-8")
    obs_csv_writer = csv.writer(obs_csv_file)

    raw_headers = [
        "Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str",
        "Raw_Buffer_Count", "Raw_Avg_Load", "Raw_Min_Load", "Raw_Max_Load", "Raw_Load_Std",
        "Raw_Weighted_Idle", "Raw_Unweighted_Idle",
        "Raw_Buf_NegSlack_Ratio", "Raw_Buf_Min_Slack", "Raw_Buf_Avg_Slack", "Raw_Buf_Slack_Std",
        "Raw_WIP_Job_Count", "Raw_WIP_Tardy_Ratio", "Raw_WIP_Min_Slack", "Raw_WIP_Avg_Slack", "Raw_WIP_Slack_Std",
        "Raw_WIP_Planned_TD", "Raw_WIP_Total_Rem_Work",
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
    ]
    obs_headers = ["Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str"] + [f"Obs_{i}" for i in range(18)] + [
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
    ]
    raw_csv_writer.writerow(raw_headers)
    obs_csv_writer.writerow(obs_headers)

    release_count, plot_seq = 0, 0
    total_cumulative_reward = 0.0
    cached_projected_td = None

    def clone_orchestrator(src):
        dst = GlobalTimelineOrchestrator.__new__(GlobalTimelineOrchestrator)
        dst.M = int(src.M)
        dst.t = float(src.t)
        dst.generator = src.generator
        dst.select_from_buffer = src.select_from_buffer
        dst.buffer = copy.deepcopy(src.buffer)
        dst.machine_free_time = np.asarray(src.machine_free_time, dtype=float).copy()
        dst._global_rows = [dict(r) for r in src._global_rows]
        dst._global_row_keys = set(src._global_row_keys)
        dst._last_full_rows = [dict(r) for r in src._last_full_rows]
        dst._last_jobs_snapshot = copy.deepcopy(src._last_jobs_snapshot)
        dst._job_history_finishes = dict(src._job_history_finishes)
        dst._release_count = int(src._release_count)
        dst.method = src.method
        dst._ppo = src._ppo
        return dst

    def estimate_release_now_td(orchestrator, t_event):
        actual_td = float(orchestrator.get_total_tardiness_estimate(all_job_due_dates))
        if len(orchestrator.buffer) <= 0:
            return actual_td
        shadow = clone_orchestrator(orchestrator)
        shadow.event_release_and_reschedule(float(t_event))
        return float(shadow.get_total_tardiness_estimate(all_job_due_dates))

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
        return [len(orchestrator.buffer), np.mean(rem), np.min(rem), mx_l, np.std(rem), w_idle, u_idle, b_stats[0], b_stats[1], b_stats[2], b_stats[3], wip["wip_count"], wip["wip_tardy_ratio"], wip["wip_min_slack"], wip["wip_avg_slack"], wip["wip_slack_std"], wip["planned_td"], wip["total_rem_work"]]

    def save_details(orch, seq, t, label=""):
        # Skip if Fast Mode is on
        if FAST_MODE: return
        unique_rows = {}

        def row_status(row):
            return "History" if float(row["start"]) < float(t) else "NewPlan"

        def should_replace(existing, new_row):
            if existing is None:
                return True
            existing_row, existing_status = existing
            new_status = row_status(new_row)
            # Once an op has started, always show it as History.
            if existing_status == "History" and new_status == "NewPlan":
                return False
            if existing_status == "NewPlan" and new_status == "History":
                return True
            # For the same status, prefer the latest row we saw.
            return True

        for r in orch._global_rows:
            key = (int(r["job"]), int(r["op"]))
            if should_replace(unique_rows.get(key), r):
                unique_rows[key] = (dict(r), row_status(r))
        for r in orch._last_full_rows:
            key = (int(r["job"]), int(r["op"]))
            if should_replace(unique_rows.get(key), r):
                unique_rows[key] = (dict(r), row_status(r))
        job_max_op = {}
        for (jid, opid) in unique_rows.keys(): job_max_op[jid] = max(job_max_op.get(jid, -1), opid)
        full_data_rows = []
        for (jid, opid), (row, status) in sorted(unique_rows.items()):
            dd = all_job_due_dates.get(jid, 0.0); is_last = (opid == job_max_op[jid])
            row.update({"status": status, "due_date": f"{dd:.2f}", "tardiness": f"{max(0.0, float(row['end']) - dd):.2f}" if is_last else "0.00"})
            full_data_rows.append(row)
        fname = f"details_r{seq:03d}_t{int(t):05d}{label}.csv"; df = pd.DataFrame(full_data_rows); df.to_csv(os.path.join(csv_dir, fname), index=False)
        csv_sum_td = df["tardiness"].astype(float).sum(); print(f"  [CSV Export] {fname} | Unique Jobs: {len(job_max_op)} | Total TD: {csv_sum_td:.2f}")

    if int(getattr(configs, "init_jobs", 0)) > 0:
        jl, pt, _ = SD2_instance_generator(copy.deepcopy(configs), rng=rng)
        dd_rel = generate_due_dates(jl, pt, tightness=getattr(configs, "due_date_tightness", 1.2), due_date_mode='k', rng=rng)
        init_jobs = split_matrix_to_jobs(jl, pt, base_job_id=0, t_arrive=0.0, due_dates=0.0 + dd_rel)
        for j in init_jobs: all_job_due_dates[j.job_id] = j.meta["due_date"]
        orch.buffer.extend(init_jobs); gen.bump_next_id(max((j.job_id for j in init_jobs)) + 1); orch.event_release_and_reschedule(0.0); release_count += 1
        collect_subproblem_stats(orch._committed_jobs, 0.0) # [STATS: INITIAL SUBPROBLEM]
        raw_s0 = get_raw_state_info(orch, 0.0)
        if not FAST_MODE:
            save_details(orch, plot_seq+1, 0.0, "_INIT")
            plot_global_gantt([dict(r, phase="history" if r["start"] < 0.0 else "newplan") for r in (orch._global_rows + orch._last_full_rows)], os.path.join(csv_dir, f"global_r{plot_seq:03d}_t0.png"), t_now=0.0, title="Initial")
        plot_seq += 1

    stats = {"arrive": 0}
    t_now, t_prev = 0.0, 0.0
    t_next = gen.sample_next_time(t_now)
    while stats["arrive"] < int(max_events):
        t_now = float(t_next)
        inter_arrival = t_now - t_prev
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            for j in new_jobs: all_job_due_dates[j.job_id] = j.meta["due_date"]
            orch.buffer.extend(new_jobs)
        stats["arrive"] += 1
        
        raw_s = get_raw_state_info(orch, t_now)
        b_dict = {"tardiness_ratio": raw_s[7], "min_slack": raw_s[8], "avg_slack": raw_s[9], "slack_std": raw_s[10]}
        w_dict = {
            "wip_count": raw_s[11], 
            "wip_tardy_ratio": raw_s[12], 
            "wip_min_slack": raw_s[13], 
            "wip_avg_slack": raw_s[14], 
            "wip_slack_std": raw_s[15], 
            "planned_td": raw_s[16], 
            "total_rem_work": raw_s[17]
        }
        
        obs = calculate_ddqn_state(
            len(orch.buffer),
            orch.machine_free_time,
            t_now,
            configs.n_m,
            0,
            reward_scale,
            raw_s[5],
            raw_s[6],
            b_dict,
            w_dict
        )
        if is_ddqn:
            act = ddqn_model(torch.from_numpy(obs).float().unsqueeze(0).to(ddqn_device)).argmax(dim=1).item()
        else:
            act = 1 if (stats["arrive"] % configs.gate_cadence == 0) else 0

        use_shaping = abs(shaping_reward_coef) > 1e-12
        if use_shaping:
            phi_before = float(cached_projected_td) if cached_projected_td is not None else estimate_release_now_td(orch, t_now)
        else:
            phi_before = 0.0
            cached_projected_td = None
        actual_td_logged = 0.0
        if act == 1:
            orch.event_release_and_reschedule(t_now)
            release_count += 1
            collect_subproblem_stats(orch._committed_jobs, t_now) # [STATS: DYNAMIC SUBPROBLEM]
            actual_td_after = orch.get_total_tardiness_estimate(all_job_due_dates)
            actual_td_logged = float(actual_td_after)
            if not FAST_MODE:
                save_details(orch, plot_seq+1, t_now)
                plot_global_gantt([dict(r, phase="history" if r["start"] < t_now else "newplan") for r in (orch._global_rows + orch._last_full_rows)], os.path.join(csv_dir, f"global_r{plot_seq:03d}_t{int(t_now):05d}.png"), t_now=t_now, title=f"Event #{stats['arrive']}")
                plot_seq += 1
        else:
            orch.tick_without_release(t_now)
            actual_td_now = orch.get_total_tardiness_estimate(all_job_due_dates)
            actual_td_logged = float(actual_td_now)

        t_next_future = float(gen.sample_next_time(t_now))
        scale = reward_scale
        r_stab = -float(act) * stability_scale
        total_neg_slack = 0.0
        for job_state in orch.buffer:
            due_abs = float(job_state.meta.get("due_date", t_now))
            rem_work = 0.0
            for op in job_state.operations:
                v = np.array(op.time_row)
                rem_work += float(np.mean(v[v > 0])) if v[v > 0].size else 0.0
            slack = due_abs - t_now - rem_work
            if slack < 0.0:
                total_neg_slack += float(-slack)
            elif slack < 100.0:
                total_neg_slack += float((100.0 - slack) * 0.5)
        r_buf = -(total_neg_slack * buffer_penalty_coef) / scale
        r_shape = 0.0
        r_terminal = 0.0
        r_flush = 0.0
        final_mk = ""
        final_td = ""
        phi_before_csv = f"{phi_before:.2f}" if use_shaping else ""
        phi_after_csv = ""
        agent_final_td_csv = ""

        done = bool(stats["arrive"] >= int(max_events))
        if done:
            while len(orch.buffer) > 0:
                orch.event_release_and_reschedule(t_next_future)
                release_count += 1
                collect_subproblem_stats(orch._committed_jobs, t_next_future)
                if not FAST_MODE:
                    save_details(orch, plot_seq+1, t_next_future, "_FLUSH")
                    plot_global_gantt([dict(r, phase="history" if r["start"] < t_next_future else "newplan") for r in (orch._global_rows + orch._last_full_rows)], os.path.join(csv_dir, f"global_r{plot_seq:03d}_FLUSH.png"), t_now=t_next_future, title="FLUSH")
                plot_seq += 1

            final_stats = orch.get_final_kpi_stats(all_job_due_dates)
            total_td = float(final_stats["tardiness"])
            final_mk_val = float(final_stats["makespan"])
            phi_after = float(total_td) if use_shaping else 0.0
            terminal_scale = max(scale * float(max_events), scale)
            r_terminal = float((-(total_td) / terminal_scale) * terminal_reward_coef)
            cached_projected_td = None

            mk_norm = max(scale * float(max_events), scale)
            mk_ratio = (final_mk_val / mk_norm)
            r_flush_raw = -(((mk_ratio + 1.0) ** 2) - 1.0) * float(configs.flush_penalty_coef)
            r_flush = float(r_flush_raw)
            final_mk = f"{final_mk_val:.2f}"
            final_td = f"{total_td:.2f}"
            phi_after_csv = f"{phi_after:.2f}" if use_shaping else ""
            agent_final_td_csv = f"{total_td:.2f}"
        else:
            if use_shaping:
                phi_after = float(estimate_release_now_td(orch, t_next_future))
                cached_projected_td = phi_after
                phi_after_csv = f"{phi_after:.2f}"
            else:
                phi_after = 0.0
                cached_projected_td = None

        r_shape = float((-(phi_after - phi_before) / scale) * shaping_reward_coef) if use_shaping else 0.0
        step_reward = r_stab + r_buf + r_shape + r_terminal + r_flush
        total_cumulative_reward += step_reward
        action_str = "RELEASE" if act == 1 else "HOLD"
        common_tail = [
            f"{actual_td_logged:.2f}",
            f"{step_reward:.4f}",
            f"{r_stab:.4f}",
            f"{r_buf:.4f}",
            f"{r_shape:.4f}",
            f"{r_terminal:.4f}",
            f"{r_flush:.4f}",
            phi_before_csv,
            phi_after_csv,
            agent_final_td_csv,
            final_mk,
            final_td,
            release_count,
        ]
        raw_csv_writer.writerow(
            [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, action_str] +
            [f"{float(x):.6f}" for x in raw_s] +
            common_tail
        )
        obs_csv_writer.writerow(
            [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, action_str] +
            [f"{float(x):.6f}" for x in obs] +
            common_tail
        )

        t_prev = t_now
        t_next = t_next_future

    final_stats = orch.get_final_kpi_stats(all_job_due_dates)
    total_td, final_mk = float(final_stats["tardiness"]), float(final_stats["makespan"])
    
    # [DISABLED] Skip summary boxplots
    # plot_simulation_summary_stats(all_sim_job_stats, csv_dir)

    raw_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 18 + [
        f"{total_td:.2f}", f"{total_cumulative_reward:.4f}", "", "", "", "", "",
        "", "", f"{final_mk:.2f}", f"{total_td:.2f}", release_count
    ]
    obs_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 18 + [
        f"{total_td:.2f}", f"{total_cumulative_reward:.4f}", "", "", "", "", "",
        "", "", f"{final_mk:.2f}", f"{total_td:.2f}", release_count
    ]
    raw_csv_writer.writerow(raw_summary)
    obs_csv_writer.writerow(obs_summary)
    raw_csv_file.close()
    obs_csv_file.close()
    return final_mk, {"release_count": release_count, "total_tardiness": total_td}

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
