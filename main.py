import os
import time
import copy
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List

from params import configs
from common_utils import *
from global_env import GlobalTimelineOrchestrator

# Plotting
from gantt import plot_global_gantt
from plot_utils import plot_simulation_summary_stats

import torch
from model.gate_state import calculate_gate_state
from model.ppo_gate_model import PPOGateNet
from dynamic_job_stream import create_dynamic_world, register_initial_jobs, sample_initial_jobs

# -----------------------------------------------------------------------------

def run_event_driven_until_nevents(*, max_events: int, interarrival_mean: float, burst_K: int = 1, plot_global_dir: Optional[str] = None):
    # [FAST MODE] Skip heavy I/O tasks if enabled
    FAST_MODE = getattr(configs, "fast_mode", True)
    all_sim_job_stats = [] # Store {due_date, slack}

    seed = int(getattr(configs, "event_seed", 42))
    rng, gen, orch = create_dynamic_world(
        configs,
        interarrival_mean=float(interarrival_mean),
        burst_k=int(burst_K),
        seed=seed,
    )
    
    gate_policy = str(getattr(configs, "gate_policy", "ppo")).lower()
    is_ppo = (gate_policy == "ppo")

    all_job_due_dates: Dict[int, float] = {}
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    reward_scale = mean_pt
    stability_scale = float(getattr(configs, "stability_scale", 0.0))
    buffer_penalty_coef = float(getattr(configs, "buffer_penalty_coef", 0.0))
    shaping_reward_coef = float(getattr(configs, "shaping_reward_coef", 0.0))
    td_reward_coef = float(getattr(configs, "td_reward_coef", 0.1))

    def resolve_td_signal_source() -> str:
        explicit = str(getattr(configs, "td_signal_source", "")).strip().lower()
        if explicit:
            return explicit
        if abs(shaping_reward_coef) > 1e-12:
            return "baseline_gap_release_interval"
        if abs(td_reward_coef) > 1e-12:
            return "baseline_gap_final"
        return "none"

    def resolve_td_credit_mode() -> str:
        explicit = str(getattr(configs, "td_credit_mode", "")).strip().lower()
        if explicit:
            return explicit
        if abs(shaping_reward_coef) > 1e-12:
            return "redistribute" if bool(getattr(configs, "release_reward_redistribute", False)) else "step_only"
        if abs(td_reward_coef) > 1e-12:
            return "terminal_only"
        return "step_only"

    def resolve_stability_mode() -> str:
        explicit = str(getattr(configs, "stability_mode_v2", "")).strip().lower()
        if explicit:
            return explicit
        legacy = str(getattr(configs, "stability_mode", "immediate_all")).strip().lower()
        if abs(stability_scale) <= 1e-12:
            return "off"
        if legacy == "immediate_all":
            return "immediate_all"
        if bool(getattr(configs, "stability_terminal_only", False)):
            return "free_threshold_terminal"
        return "free_threshold_distributed"

    def resolve_td_step_coef() -> float:
        if abs(shaping_reward_coef) > 1e-12:
            return shaping_reward_coef
        return td_reward_coef

    def resolve_td_terminal_coef() -> float:
        if abs(td_reward_coef) > 1e-12:
            return td_reward_coef
        return shaping_reward_coef

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

    ppo_gate_model = None
    gate_device = torch.device(getattr(configs, "device", "cpu"))
    if gate_policy == "ppo":
        ppo_gate_model = PPOGateNet(
            obs_dim=21,
            n_actions=2,
            hidden=int(getattr(configs, "ppo_gate_hidden_dim", 256)),
            num_layers=int(getattr(configs, "ppo_gate_num_layers", 3)),
            separate_trunks=bool(getattr(configs, "ppo_gate_separate_trunks", False)),
            actor_hidden=int(getattr(configs, "ppo_gate_actor_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
            actor_num_layers=int(getattr(configs, "ppo_gate_actor_num_layers", getattr(configs, "ppo_gate_num_layers", 3))),
            critic_hidden=int(getattr(configs, "ppo_gate_critic_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
            critic_num_layers=int(getattr(configs, "ppo_gate_critic_num_layers", getattr(configs, "ppo_gate_num_layers", 3))),
            value_hidden=int(getattr(configs, "ppo_gate_value_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
            value_num_layers=int(getattr(configs, "ppo_gate_value_num_layers", 1)),
        ).to(gate_device)
        try:
            ppo_gate_model.load_state_dict(torch.load(getattr(configs, "ppo_gate_model_path", ""), map_location=gate_device, weights_only=True))
            ppo_gate_model.eval()
            print(f"[PPO-GATE] Loaded weights.")
        except:
            print(f"[WARN] Fallback."); gate_policy = "cadence"; is_ppo = False

    if gate_policy == "ppo":
        suffix = f"PPO_{getattr(configs, 'ppo_gate_name', 'default')}"
    elif gate_policy == "slack_threshold":
        thr = float(getattr(configs, "buffer_slack_release_threshold", 0.0))
        suffix = f"SlackThr_{thr:g}"
    else:
        suffix = f"Cadence_{getattr(configs, 'gate_cadence', 1)}"
    base_plot_dir = plot_global_dir or "plots/global"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if gate_policy == "ppo":
        safe_name = str(getattr(configs, "ppo_gate_name", "run"))
    elif gate_policy == "slack_threshold":
        safe_name = f"slk{float(getattr(configs, 'buffer_slack_release_threshold', 0.0)):g}"
    else:
        safe_name = f"cad{int(getattr(configs, 'gate_cadence', 1))}"
    override_name = str(getattr(configs, "plot_run_name", "")).strip()
    if override_name:
        safe_name = override_name
    safe_name = safe_name.replace(" ", "_")[:48]
    run_dir_name = f"{timestamp}_{safe_name}"
    csv_dir = os.path.join(base_plot_dir, run_dir_name)
    os.makedirs(csv_dir, exist_ok=True)
    raw_csv_file = open(os.path.join(csv_dir, "raw_state.csv"), "w", newline="", encoding="utf-8")
    raw_csv_writer = csv.writer(raw_csv_file)
    obs_csv_file = open(os.path.join(csv_dir, "agent_state.csv"), "w", newline="", encoding="utf-8")
    obs_csv_writer = csv.writer(obs_csv_file)
    step1_csv_file = open(os.path.join(csv_dir, "step1_td_check.csv"), "w", newline="", encoding="utf-8")
    step1_csv_writer = csv.writer(step1_csv_file)

    raw_headers = [
        "Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str",
        "Raw_Buffer_Count", "Raw_Avg_Load", "Raw_Min_Load", "Raw_Max_Load", "Raw_Load_Std",
        "Raw_Weighted_Idle", "Raw_Unweighted_Idle",
        "Raw_Buffer_NegSlack_Ratio", "Raw_Buf_Min_Slack", "Raw_Buf_Avg_Slack", "Raw_Buf_Slack_Std", "Raw_Buf_Slack_Q25",
        "Raw_WIP_Job_Count", "Raw_WIP_Tardy_Ratio", "Raw_WIP_Min_Slack", "Raw_WIP_Avg_Slack", "Raw_WIP_Slack_Std",
        "Raw_WIP_Planned_TD", "Raw_WIP_Total_Rem_Work",
        "Baseline_Step_TD", "Baseline_Prev_Release_Event_ID", "Baseline_Prev_Release_TD", "Baseline_TD_Delta",
        "Agent_Prev_Release_Event_ID", "Agent_Prev_Release_TD", "Agent_TD_Delta",
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD", "TD_Gap_vs_Baseline_Cadence",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
    ]
    obs_headers = [
        "Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str",
        "Log_Buffer_Count", "Norm_Avg_Load", "Norm_Min_Load", "Norm_Load_Range", "Norm_Load_Std",
        "Norm_Weighted_Idle", "Norm_Unweighted_Idle",
        "Buffer_NegSlack_Ratio", "Norm_Buf_Min_Slack", "Norm_Buf_Avg_Slack", "Norm_Buf_Slack_Std", "Norm_Buf_Slack_Q25",
        "WIP_Job_Count", "WIP_Tardy_Ratio", "Norm_WIP_Min_Slack", "Norm_WIP_Avg_Slack", "Norm_WIP_Slack_Std",
        "Clipped_Planned_TD_Ratio", "Avg_WIP_Slack_Per_Job",
        "Scaled_Inter_Arrival", "Log_Steps_Since_Last_Release",
    ] + [
        "Baseline_Step_TD", "Baseline_Prev_Release_Event_ID", "Baseline_Prev_Release_TD", "Baseline_TD_Delta",
        "Agent_Prev_Release_Event_ID", "Agent_Prev_Release_TD", "Agent_TD_Delta",
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD", "TD_Gap_vs_Baseline_Cadence",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
    ]
    obs_csv_order = [0, 1, 2, 15, 10, 3, 17, 4, 5, 6, 12, 14, 9, 7, 8, 13, 11, 16, 18, 19, 20]
    raw_csv_writer.writerow(raw_headers)
    obs_csv_writer.writerow(obs_headers)
    step1_csv_writer.writerow(["Event_ID", "Agent_Step_TD", "Baseline_Step_TD"])

    release_count, plot_seq = 0, 0
    total_cumulative_reward = 0.0
    baseline_cadence = 1
    stability_mode = resolve_stability_mode()
    stability_free_releases = max(0, int(getattr(configs, "stability_free_releases", 0)))
    td_signal_source = resolve_td_signal_source()
    td_credit_mode = resolve_td_credit_mode()
    td_step_coef = resolve_td_step_coef()
    td_terminal_coef = resolve_td_terminal_coef()
    pending_raw_rows = []
    pending_obs_rows = []
    release_row_indices = []
    steps_since_last_release = 0

    def compute_total_stability_penalty(agent_release_count: int) -> float:
        if abs(stability_scale) <= 1e-12:
            return 0.0
        if stability_mode == "off":
            return 0.0
        if stability_mode in ("immediate_all", "immediate_all_terminal"):
            return float(-stability_scale * max(0, int(agent_release_count)))
        excess_releases = max(0, int(agent_release_count) - stability_free_releases)
        return float(-stability_scale * (excess_releases * (excess_releases + 1) / 2.0))

    def advance_sim_to_next_arrival(local_gen, local_orch, local_all_due, next_time: float):
        t_event = float(next_time)
        new_jobs = local_gen.generate_burst(t_event)
        if new_jobs:
            for j in new_jobs:
                local_all_due[j.job_id] = j.meta["due_date"]
            local_orch.buffer.extend(new_jobs)
        return t_event, float(local_gen.sample_next_time(t_event))

    def build_simulation():
        local_rng, local_gen, local_orch = create_dynamic_world(
            configs,
            interarrival_mean=float(interarrival_mean),
            burst_k=int(burst_K),
            seed=seed,
        )
        local_all_due: Dict[int, float] = {}
        local_release_count = 0
        local_t_now = 0.0
        local_init_jobs = sample_initial_jobs(configs, rng=local_rng, base_job_id=0, t_arrive=0.0)
        if local_init_jobs:
            local_release_count += register_initial_jobs(local_orch, local_gen, local_init_jobs, local_all_due, t0=0.0)
        local_t_next = float(local_gen.sample_next_time(local_t_now))
        local_t_now, local_t_next = advance_sim_to_next_arrival(local_gen, local_orch, local_all_due, local_t_next)
        return local_rng, local_gen, local_orch, local_all_due, local_t_now, local_t_next, local_release_count

    def run_cadence_baseline(cadence=None):
        cadence = baseline_cadence if cadence is None else cadence
        cadence = max(1, int(cadence))
        _, base_gen, base_orch, base_all_due, base_t_now, base_t_next, base_release_count = build_simulation()
        base_events = 1
        base_event_td = []
        while True:
            if base_events % cadence == 0:
                base_orch.event_release_and_reschedule(float(base_t_now))
                base_release_count += 1
            else:
                base_orch.tick_without_release(float(base_t_now))
            base_event_td.append(float(base_orch.get_total_tardiness_estimate(base_all_due)))
            if base_events >= int(max_events):
                break
            base_t_now, base_t_next = advance_sim_to_next_arrival(base_gen, base_orch, base_all_due, base_t_next)
            base_events += 1
        while len(base_orch.buffer) > 0:
            base_orch.event_release_and_reschedule(base_t_next)
            base_release_count += 1
        base_final = base_orch.get_final_kpi_stats(base_all_due)
        return float(base_final["tardiness"]), float(base_final["makespan"]), int(base_release_count), [float(x) for x in base_event_td]

    def get_raw_state_info(orchestrator, t_now):
        b_slacks, b_neg = [], 0
        for j in orchestrator.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0)); due = all_job_due_dates[j.job_id]; s = due - t_now - mw
            b_slacks.append(s); 
            if t_now + mw > due: b_neg += 1
        b_stats = (b_neg/len(orchestrator.buffer), min(b_slacks), sum(b_slacks)/len(b_slacks), np.std(b_slacks), np.percentile(b_slacks, 25)) if orchestrator.buffer else (0.0, 0.0, 0.0, 0.0, 0.0)
        wip = orchestrator.get_wip_stats(t_now); mft = np.asarray(orchestrator.machine_free_time, dtype=float); rem = np.maximum(0.0, mft - t_now); mx_l = np.max(rem)
        w_idle = orchestrator.compute_weighted_idle(t_now, float(mx_l)) if mx_l>0 else 0.0
        u_idle = orchestrator.compute_unweighted_idle(t_now, float(mx_l)) if mx_l>0 else 0.0
        return [len(orchestrator.buffer), np.mean(rem), np.min(rem), mx_l, np.std(rem), w_idle, u_idle, b_stats[0], b_stats[1], b_stats[2], b_stats[3], b_stats[4], wip["wip_count"], wip["wip_tardy_ratio"], wip["wip_min_slack"], wip["wip_avg_slack"], wip["wip_slack_std"], wip["planned_td"], wip["total_rem_work"]]

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

    def build_plot_rows(t_marker: float):
        return [
            dict(
                r,
                phase="history" if float(r["start"]) < float(t_marker) else "newplan",
                due_date=float(all_job_due_dates.get(int(r["job"]), 0.0)),
            )
            for r in (orch._global_rows + orch._last_full_rows)
        ]

    init_jobs = sample_initial_jobs(configs, rng=rng, base_job_id=0, t_arrive=0.0)
    if init_jobs:
        release_count += register_initial_jobs(orch, gen, init_jobs, all_job_due_dates, t0=0.0)
        collect_subproblem_stats(orch._committed_jobs, 0.0) # [STATS: INITIAL SUBPROBLEM]
        raw_s0 = get_raw_state_info(orch, 0.0)
        if not FAST_MODE:
            save_details(orch, plot_seq+1, 0.0, "_INIT")
            plot_global_gantt(build_plot_rows(0.0), os.path.join(csv_dir, f"global_r{plot_seq:03d}_t0.png"), t_now=0.0, title="Initial")
        plot_seq += 1

    baseline_final_td, baseline_final_mk, baseline_release_count, baseline_event_td = run_cadence_baseline()
    last_release_event_idx = 0
    last_release_td = 0.0

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
        b_dict = {"buffer_neg_slack_ratio": raw_s[7], "min_slack": raw_s[8], "avg_slack": raw_s[9], "slack_std": raw_s[10], "slack_q25": raw_s[11]}
        w_dict = {
            "wip_count": raw_s[12], 
            "wip_tardy_ratio": raw_s[13], 
            "wip_min_slack": raw_s[14], 
            "wip_avg_slack": raw_s[15], 
            "wip_slack_std": raw_s[16], 
            "planned_td": raw_s[17], 
            "total_rem_work": raw_s[18]
        }
        
        obs = calculate_gate_state(
            len(orch.buffer),
            orch.machine_free_time,
            t_now,
            configs.n_m,
            0,
            reward_scale,
            raw_s[5],
            raw_s[6],
            b_dict,
            w_dict,
            inter_arrival_scaled=(float(inter_arrival) / float(reward_scale)) if reward_scale > 0 else 0.0,
            steps_since_last_release=steps_since_last_release,
        )
        if is_ppo:
            with torch.no_grad():
                logits, _ = ppo_gate_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
                if str(getattr(configs, "eval_action_selection", "greedy")).lower() == "sample":
                    dist = torch.distributions.Categorical(logits=logits)
                    act = int(dist.sample().item())
                else:
                    act = int(torch.argmax(logits, dim=1).item())
        elif gate_policy == "slack_threshold":
            min_slack = float(raw_s[8])
            act = 1 if min_slack < float(getattr(configs, "buffer_slack_release_threshold", 0.0)) else 0
        else:
            act = 1 if (stats["arrive"] % configs.gate_cadence == 0) else 0

        actual_td_logged = 0.0
        if act == 1:
            orch.event_release_and_reschedule(t_now)
            release_count += 1
            steps_since_last_release = 0
            collect_subproblem_stats(orch._committed_jobs, t_now) # [STATS: DYNAMIC SUBPROBLEM]
            actual_td_after = orch.get_total_tardiness_estimate(all_job_due_dates)
            actual_td_logged = float(actual_td_after)
            if not FAST_MODE:
                save_details(orch, plot_seq+1, t_now)
                plot_global_gantt(build_plot_rows(t_now), os.path.join(csv_dir, f"global_r{plot_seq:03d}_t{int(t_now):05d}.png"), t_now=t_now, title=f"Event #{stats['arrive']}")
                plot_seq += 1
        else:
            orch.tick_without_release(t_now)
            steps_since_last_release += 1
            actual_td_now = orch.get_total_tardiness_estimate(all_job_due_dates)
            actual_td_logged = float(actual_td_now)

        t_next_future = float(gen.sample_next_time(t_now))
        scale = reward_scale
        r_stab = 0.0
        if stability_mode == "immediate_all":
            r_stab = -stability_scale if int(act) == 1 else 0.0
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
        r_td = 0.0
        r_mk = 0.0
        final_mk = ""
        final_td = ""
        phi_before_csv = ""
        phi_after_csv = ""
        agent_final_td_csv = ""
        td_gap_csv = ""
        baseline_step_td = float(baseline_event_td[stats["arrive"] - 1]) if stats["arrive"] - 1 < len(baseline_event_td) else 0.0
        prev_release_event_idx = int(last_release_event_idx)
        prev_agent_release_td = float(last_release_td)
        prev_baseline_td = float(baseline_event_td[prev_release_event_idx - 1]) if prev_release_event_idx > 0 and (prev_release_event_idx - 1) < len(baseline_event_td) else 0.0
        baseline_td_delta_csv = ""
        agent_td_delta_csv = ""

        done = bool(stats["arrive"] >= int(max_events))
        if done:
            while len(orch.buffer) > 0:
                orch.event_release_and_reschedule(t_next_future)
                release_count += 1
                collect_subproblem_stats(orch._committed_jobs, t_next_future)
                if not FAST_MODE:
                    save_details(orch, plot_seq+1, t_next_future, "_FLUSH")
                    plot_global_gantt(build_plot_rows(t_next_future), os.path.join(csv_dir, f"global_r{plot_seq:03d}_FLUSH.png"), t_now=t_next_future, title="FLUSH")
                plot_seq += 1

            final_stats = orch.get_final_kpi_stats(all_job_due_dates)
            total_td = float(final_stats["tardiness"])
            final_mk_val = float(final_stats["makespan"])
            terminal_scale = max(scale, 1e-8)
            td_gap = float(total_td - baseline_final_td)
            if td_credit_mode == "terminal_only":
                if td_signal_source == "baseline_gap_final":
                    r_td = float((-(td_gap) / terminal_scale) * td_terminal_coef)
                elif td_signal_source == "agent_only":
                    r_td = float((-(total_td) / terminal_scale) * td_terminal_coef)

            mk_norm = max(scale * float(max_events), scale)
            mk_ratio = (final_mk_val / mk_norm)
            r_mk_raw = -(((mk_ratio + 1.0) ** 2) - 1.0) * float(configs.mk_reward_coef)
            r_mk = float(r_mk_raw)
            final_mk = f"{final_mk_val:.2f}"
            final_td = f"{total_td:.2f}"
            agent_final_td_csv = f"{total_td:.2f}"
            td_gap_csv = f"{td_gap:.2f}"
        else:
            td_gap = 0.0

        if int(act) == 1 and td_credit_mode in ("step_only", "redistribute") and abs(td_step_coef) > 1e-12:
            agent_td_delta = float(actual_td_logged - last_release_td)
            baseline_td_delta = float(baseline_step_td - prev_baseline_td)
            if td_signal_source == "agent_only":
                td_signal_value = float(agent_td_delta)
            elif td_signal_source == "baseline_gap_release_interval":
                td_signal_value = float(agent_td_delta - baseline_td_delta)
            else:
                td_signal_value = 0.0
            r_shape = float((-(td_signal_value) / scale) * td_step_coef)
            baseline_td_delta_csv = f"{baseline_td_delta:.2f}"
            agent_td_delta_csv = f"{agent_td_delta:.2f}"
            last_release_event_idx = int(stats["arrive"])
            last_release_td = float(actual_td_logged)
        elif done and td_signal_source == "agent_only" and td_credit_mode in ("step_only", "redistribute") and abs(td_step_coef) > 1e-12:
            agent_td_delta = float(total_td - last_release_td)
            r_shape = float((-(agent_td_delta) / scale) * td_step_coef)
            agent_td_delta_csv = f"{agent_td_delta:.2f}"

        step_reward = r_stab + r_buf + r_shape + r_td + r_mk
        action_str = "RELEASE" if act == 1 else "HOLD"
        common_tail = [
            f"{baseline_step_td:.2f}",
            prev_release_event_idx if prev_release_event_idx > 0 else "",
            f"{prev_baseline_td:.2f}" if prev_release_event_idx > 0 else "",
            baseline_td_delta_csv,
            prev_release_event_idx if prev_release_event_idx > 0 else "",
            f"{prev_agent_release_td:.2f}" if prev_release_event_idx > 0 else "",
            agent_td_delta_csv,
            f"{actual_td_logged:.2f}",
            f"{step_reward:.4f}",
            f"{r_stab:.4f}",
            f"{r_buf:.4f}",
            f"{r_shape:.4f}",
            f"{r_td:.4f}",
            f"{r_mk:.4f}",
            phi_before_csv,
            phi_after_csv,
            agent_final_td_csv,
            td_gap_csv,
            final_mk,
            final_td,
            release_count,
        ]
        raw_row = [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, action_str] + [f"{float(x):.6f}" for x in raw_s] + common_tail
        obs_row = [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, action_str] + [f"{float(obs[i]):.6f}" for i in obs_csv_order] + common_tail
        pending_raw_rows.append(raw_row)
        pending_obs_rows.append(obs_row)
        if int(act) == 1:
            release_row_indices.append(len(pending_raw_rows) - 1)
        if int(stats["arrive"]) == 1:
            step1_csv_writer.writerow([1, f"{actual_td_logged:.2f}", f"{baseline_step_td:.2f}"])

        t_prev = t_now
        t_next = t_next_future

    final_stats = orch.get_final_kpi_stats(all_job_due_dates)
    total_td, final_mk = float(final_stats["tardiness"]), float(final_stats["makespan"])
    total_stab_penalty = compute_total_stability_penalty(release_count)
    if stability_mode in ("immediate_all_terminal", "free_threshold_terminal", "free_threshold_distributed") and abs(total_stab_penalty) > 1e-12:
        if stability_mode in ("immediate_all_terminal", "free_threshold_terminal") and len(pending_raw_rows) > 0:
            idx = len(pending_raw_rows) - 1
            for rows in (pending_raw_rows, pending_obs_rows):
                reward_total = float(rows[idx][-13]) + float(total_stab_penalty)
                rows[idx][-13] = f"{reward_total:.4f}"
                rows[idx][-12] = f"{float(total_stab_penalty):.4f}"
        elif stability_mode == "free_threshold_distributed" and release_row_indices:
            add_each = float(total_stab_penalty) / float(len(release_row_indices))
            for idx in release_row_indices:
                for rows in (pending_raw_rows, pending_obs_rows):
                    reward_total = float(rows[idx][-13]) + add_each
                    reward_stab = float(rows[idx][-12]) + add_each
                    rows[idx][-13] = f"{reward_total:.4f}"
                    rows[idx][-12] = f"{reward_stab:.4f}"
    for row in pending_raw_rows:
        total_cumulative_reward += float(row[-13])
        raw_csv_writer.writerow(row)
    for row in pending_obs_rows:
        obs_csv_writer.writerow(row)
    
    # [DISABLED] Skip summary boxplots
    # plot_simulation_summary_stats(all_sim_job_stats, csv_dir)

    raw_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 18 + [
        f"{total_td:.2f}", f"{total_cumulative_reward:.4f}", "", "", "", "", "",
        "", "", "", f"{final_mk:.2f}", f"{total_td:.2f}", release_count
    ]
    obs_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 18 + [
        f"{total_td:.2f}", f"{total_cumulative_reward:.4f}", "", "", "", "", "",
        "", "", "", f"{final_mk:.2f}", f"{total_td:.2f}", release_count
    ]
    raw_csv_writer.writerow(raw_summary)
    obs_csv_writer.writerow(obs_summary)
    raw_csv_file.close()
    obs_csv_file.close()
    step1_csv_file.close()
    return final_mk, {"release_count": release_count, "total_tardiness": total_td}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))
    print("-" * 20 + " Dynamic FJSP HRL Evaluation " + "-" * 20)
    
    # [FIXED] Dynamic output naming based on actual policy
    plot_dir = getattr(configs, "plot_global_dir", "plots/global")
    
    mk, stats = run_event_driven_until_nevents(
        max_events=int(configs.event_horizon), 
        interarrival_mean=configs.interarrival_mean, 
        burst_K=configs.burst_size, 
        plot_global_dir=plot_dir
    )
    print(f"\nMakespan: {mk:.3f}, Tardiness: {stats['total_tardiness']:.3f}, Releases: {stats['release_count']}")

if __name__ == "__main__": main()
