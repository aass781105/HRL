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
from hrl_orchestrator import GlobalTimelineOrchestrator

# Plotting
from gantt import plot_global_gantt
from plot_utils import plot_simulation_summary_stats

import torch
from model.hl_gate_state import HL_GATE_STATE_DIM, calculate_hl_gate_state
from model.hl_ppo_gate_model import HLPPOGateNet
from dynamic_job_stream import create_dynamic_world, register_initial_jobs, sample_initial_jobs

# -----------------------------------------------------------------------------

def _mean_std(values: List[float]):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def run_event_driven_until_nevents(
    *,
    max_events: int,
    interarrival_mean: float,
    burst_K: int = 1,
    plot_global_dir: Optional[str] = None,
    write_outputs: bool = True,
    seed_override: Optional[int] = None,
    sample_seed_override: Optional[int] = None,
    aggregate_prior: Optional[Dict[str, List[float]]] = None,
):
    t_sim_start = time.perf_counter()
    # [FAST MODE] Skip heavy I/O tasks if enabled
    FAST_MODE = getattr(configs, "fast_mode", True) or (not write_outputs)
    all_sim_job_stats = [] # Store {due_date, slack}

    seed = int(getattr(configs, "event_seed", 42) if seed_override is None else seed_override)
    sample_seed = int(seed if sample_seed_override is None else sample_seed_override)
    configs._active_eval_env_seed = seed
    configs._active_eval_sample_seed = sample_seed
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    rng, gen, orch = create_dynamic_world(
        configs,
        interarrival_mean=float(interarrival_mean),
        burst_k=int(burst_K),
        seed=seed,
    )
    # create_dynamic_world seeds torch with env_seed for reproducible instances.
    # Reset torch afterwards so policy sampling can vary independently per run.
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    
    gate_policy = str(getattr(configs, "hl_gate_policy", "ppo")).lower()
    is_ppo = (gate_policy == "ppo")

    all_job_due_dates: Dict[int, float] = {}
    all_job_arrive_times: Dict[int, float] = {}
    all_job_is_urgent: Dict[int, bool] = {}
    all_job_due_date_k: Dict[int, float] = {}
    env_job_info_rows: List[Dict] = []
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    reward_scale = mean_pt
    stability_scale = float(getattr(configs, "hl_stability_scale", 0.0))
    buffer_penalty_coef = float(getattr(configs, "hl_buffer_penalty_coef", 0.0))
    shaping_reward_coef = float(getattr(configs, "hl_shaping_reward_coef", 0.0))
    td_reward_coef = float(getattr(configs, "hl_td_reward_coef", 0.1))

    def resolve_td_signal_source() -> str:
        explicit = str(getattr(configs, "hl_td_signal_source", "")).strip().lower()
        if explicit:
            return explicit
        if abs(shaping_reward_coef) > 1e-12:
            return "baseline_gap_release_interval"
        if abs(td_reward_coef) > 1e-12:
            return "baseline_gap_final"
        return "none"

    def resolve_td_credit_mode() -> str:
        explicit = str(getattr(configs, "hl_td_credit_mode", "")).strip().lower()
        if explicit:
            return explicit
        if abs(shaping_reward_coef) > 1e-12:
            return "redistribute" if bool(getattr(configs, "hl_release_reward_redistribute", False)) else "step_only"
        if abs(td_reward_coef) > 1e-12:
            return "terminal_only"
        return "step_only"

    def resolve_stability_mode() -> str:
        explicit = str(getattr(configs, "hl_stability_mode_v2", "")).strip().lower()
        if explicit:
            return explicit
        legacy = str(getattr(configs, "hl_stability_mode", "immediate_all")).strip().lower()
        if abs(stability_scale) <= 1e-12:
            return "off"
        if legacy == "immediate_all":
            return "immediate_all"
        if bool(getattr(configs, "hl_stability_terminal_only", False)):
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

    def record_env_job_info(job, *, event_id: int, phase: str, inter_arrival: float) -> None:
        arrive_time = float(job.meta.get("t_arrive", 0.0))
        due_date = float(job.meta.get("due_date", 0.0))
        total_pt = float(job.meta.get("total_proc_time", 0.0))
        min_total_pt = float(job.meta.get("min_total_proc_time", 0.0))
        total_ops = int(job.meta.get("total_ops", len(job.operations)))
        avg_pt = total_pt / max(float(total_ops), 1.0)
        k_value = (due_date - arrive_time) / total_pt if total_pt > 1e-12 else 0.0
        row = {
            "event_id": int(event_id),
            "phase": str(phase),
            "job_id": int(job.job_id),
            "inter_arrival": float(inter_arrival),
            "arrive_time": arrive_time,
            "due_date": due_date,
            "relative_due": due_date - arrive_time,
            "k_value": k_value,
            "total_proc_time_mean": total_pt,
            "min_total_proc_time": min_total_pt,
            "avg_op_proc_time": avg_pt,
            "total_ops": total_ops,
        }
        for op_idx, op in enumerate(job.operations):
            if op.time_row is not None:
                pts = [float(x) for x in op.time_row]
            elif op.machine_times is not None:
                pts = [0.0 for _ in range(int(configs.n_m))]
                for m, pt in op.machine_times.items():
                    pts[int(m)] = float(pt)
            else:
                pts = []
            feasible = [pt for pt in pts if pt > 0]
            row[f"op{op_idx}_pt_mean"] = float(np.mean(feasible)) if feasible else 0.0
            row[f"op{op_idx}_pt_row"] = "|".join(f"{pt:.6g}" for pt in pts)
        env_job_info_rows.append(row)

    def write_env_job_info_csv() -> None:
        if not write_outputs or not env_job_info_rows:
            return
        path = os.path.join(csv_dir, f"{csv_prefix}_env_jobs.csv")
        fieldnames = []
        for row in env_job_info_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(env_job_info_rows)
        print(f"[CSV Export] Env jobs: {path}")

    hl_ppo_model = None
    gate_device = torch.device(getattr(configs, "device", "cpu"))
    if gate_policy == "ppo":
        hl_ppo_model = HLPPOGateNet(
            obs_dim=HL_GATE_STATE_DIM,
            n_actions=2,
            hidden=int(getattr(configs, "hl_ppo_hidden_dim", 256)),
            num_layers=int(getattr(configs, "hl_ppo_num_layers", 3)),
            separate_trunks=bool(getattr(configs, "hl_ppo_separate_trunks", False)),
            actor_hidden=int(getattr(configs, "hl_ppo_actor_hidden_dim", getattr(configs, "hl_ppo_hidden_dim", 256))),
            actor_num_layers=int(getattr(configs, "hl_ppo_actor_num_layers", getattr(configs, "hl_ppo_num_layers", 3))),
            critic_hidden=int(getattr(configs, "hl_ppo_critic_hidden_dim", getattr(configs, "hl_ppo_hidden_dim", 256))),
            critic_num_layers=int(getattr(configs, "hl_ppo_critic_num_layers", getattr(configs, "hl_ppo_num_layers", 3))),
            value_hidden=int(getattr(configs, "hl_ppo_value_hidden_dim", getattr(configs, "hl_ppo_hidden_dim", 256))),
            value_num_layers=int(getattr(configs, "hl_ppo_value_num_layers", 1)),
            use_residual=bool(getattr(configs, "hl_ppo_use_residual", False)),
            use_glu=bool(getattr(configs, "hl_ppo_use_glu", False)),
            pre_norm=bool(getattr(configs, "hl_ppo_pre_norm", False)),
        ).to(gate_device)
        try:
            hl_ppo_model.load_state_dict(torch.load(getattr(configs, "hl_ppo_model_path", ""), map_location=gate_device, weights_only=True))
            hl_ppo_model.eval()
            print(f"[PPO-GATE] Loaded weights.")
        except:
            print(f"[WARN] Fallback."); gate_policy = "cadence"; is_ppo = False

    if gate_policy == "ppo":
        suffix = f"PPO_{getattr(configs, 'hl_ppo_name', 'default')}"
    elif gate_policy == "slack_threshold":
        thr = float(getattr(configs, "hl_buffer_slack_release_threshold", 0.0))
        suffix = f"SlackThr_{thr:g}"
    elif gate_policy == "random":
        prob = float(getattr(configs, "hl_random_release_prob", 0.087))
        suffix = f"Random_{prob:g}"
    else:
        suffix = f"Cadence_{getattr(configs, 'hl_gate_cadence', 1)}"
    base_plot_dir = plot_global_dir or "plots/global"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if gate_policy == "ppo":
        safe_name = str(getattr(configs, "hl_ppo_name", "") or "").strip()
        if not safe_name:
            model_path = str(getattr(configs, "hl_ppo_model_path", "") or "")
            if model_path:
                safe_name = os.path.splitext(os.path.basename(model_path))[0]
            else:
                safe_name = "ppo"
    elif gate_policy == "slack_threshold":
        safe_name = f"slk{float(getattr(configs, 'hl_buffer_slack_release_threshold', 0.0)):g}"
    elif gate_policy == "random":
        safe_name = f"rnd{float(getattr(configs, 'hl_random_release_prob', 0.087)):g}"
    else:
        safe_name = f"cad{int(getattr(configs, 'hl_gate_cadence', 1))}"
    override_name = str(getattr(configs, "plot_run_name", "")).strip()
    if override_name:
        safe_name = override_name
    safe_name = safe_name.replace(" ", "_")[:48]
    event_seed = seed
    csv_dir = None
    raw_csv_file = None
    raw_csv_writer = None
    obs_csv_file = None
    obs_csv_writer = None
    release_csv_file = None
    release_csv_writer = None
    if write_outputs:
        run_dir_name = f"{timestamp}_{safe_name}_seed{event_seed:03d}"
        csv_dir = os.path.join(base_plot_dir, run_dir_name)
        os.makedirs(csv_dir, exist_ok=True)
    csv_prefix = safe_name
    if write_outputs:
        raw_csv_file = open(os.path.join(csv_dir, f"{csv_prefix}_raw_state.csv"), "w", newline="", encoding="utf-8")
        raw_csv_writer = csv.writer(raw_csv_file)
        obs_csv_file = open(os.path.join(csv_dir, f"{csv_prefix}_agent_state.csv"), "w", newline="", encoding="utf-8")
        obs_csv_writer = csv.writer(obs_csv_file)
        release_csv_file = open(os.path.join(csv_dir, f"{csv_prefix}_ppo_release_log.csv"), "w", newline="", encoding="utf-8")
        release_csv_writer = csv.writer(release_csv_file)

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
        "Eval_Runs", "Makespan_Mean", "Makespan_Std",
        "Tardiness_Mean", "Tardiness_Std", "Obj_Mean", "Obj_Std",
        "Release_Count_Mean", "Release_Count_Std",
    ]
    obs_headers = [
        "Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str",
        "Log_Buffer_Count", "Norm_Avg_Load", "Norm_Min_Load", "Norm_Load_Range", "Norm_Load_Std",
        "Norm_Weighted_Idle", "Norm_Unweighted_Idle",
        "Buffer_NegSlack_Ratio", "Norm_Buf_Min_Slack", "Norm_Buf_Avg_Slack", "Norm_Buf_Slack_Std", "Norm_Buf_Slack_Q25",
        "WIP_Job_Count", "WIP_Tardy_Ratio", "Norm_WIP_Min_Slack", "Norm_WIP_Avg_Slack", "Norm_WIP_Slack_Std",
        "Clipped_Planned_TD_Ratio", "Avg_WIP_Slack_Per_Job",
        "Scaled_Inter_Arrival", "Log_Steps_Since_Last_Release", "Release_Rate_So_Far",
        "Buffer_Demand_Max_Share", "Buffer_WIP_Load_Overlap", "Is_Last_Step",
        "Baseline_Step_TD", "Baseline_Prev_Release_Event_ID", "Baseline_Prev_Release_TD", "Baseline_TD_Delta",
        "Agent_Prev_Release_Event_ID", "Agent_Prev_Release_TD", "Agent_TD_Delta",
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD", "TD_Gap_vs_Baseline_Cadence",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
        "Eval_Runs", "Makespan_Mean", "Makespan_Std",
        "Tardiness_Mean", "Tardiness_Std", "Obj_Mean", "Obj_Std",
        "Release_Count_Mean", "Release_Count_Std",
    ]
    obs_csv_order = [0, 1, 2, 15, 10, 3, 17, 4, 5, 6, 12, 23, 14, 9, 7, 8, 13, 11, 16, 18, 19, 20, 21, 22, 24]
    if write_outputs:
        raw_csv_writer.writerow(raw_headers)
        obs_csv_writer.writerow(obs_headers)
        release_csv_writer.writerow([
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
            "Solve_Time_Sec",
        ])

    release_count, plot_seq = 0, 0
    gate_release_count = 0
    previous_release_job_ids = set()
    total_cumulative_reward = 0.0
    total_r_stab = 0.0
    total_r_buf = 0.0
    total_r_shape = 0.0
    total_r_td = 0.0
    total_r_mk = 0.0
    baseline_cadence = int(getattr(
        configs,
        "baseline_cadence",
        getattr(configs, "hl_gate_decision_interval", 1),
    ))
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
        local_init_jobs = sample_initial_jobs(local_gen.cfg, rng=local_rng, base_job_id=0, t_arrive=0.0)
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
                base_orch.event_release_and_reschedule(float(base_t_now), event_id=int(base_events))
                base_release_count += 1
            else:
                base_orch.tick_without_release(float(base_t_now))
            base_event_td.append(float(base_orch.get_total_tardiness_estimate(base_all_due)))
            if base_events >= int(max_events):
                break
            base_t_now, base_t_next = advance_sim_to_next_arrival(base_gen, base_orch, base_all_due, base_t_next)
            base_events += 1
        while len(base_orch.buffer) > 0:
            base_orch.event_release_and_reschedule(base_t_next, event_id=int(base_events))
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
            row.update({
                "arrive_time": f"{float(all_job_arrive_times.get(jid, 0.0)):.2f}",
                "status": status,
                "due_date": f"{dd:.2f}",
                "is_urgent": int(bool(all_job_is_urgent.get(jid, False))),
                "due_date_k": f"{float(all_job_due_date_k.get(jid, 0.0)):.4f}",
                "tardiness": f"{max(0.0, float(row['end']) - dd):.2f}" if is_last else "0.00",
            })
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
        rows = list(orch._global_rows) + list(orch._last_full_rows)
        info = compute_rows_kpis(rows, all_job_due_dates)
        if getattr(orch, "machine_free_time", None) is not None and len(orch.machine_free_time) > 0:
            info["makespan"] = max(info["makespan"], float(np.max(orch.machine_free_time)))
            info["objective_value"] = 0.5 * info["makespan"] + 0.5 * info["total_tardiness"]
        return info

    def write_release_log(event_id: int, release_type: str, release_time: float, rows, solve_time: float = 0.0) -> None:
        nonlocal previous_release_job_ids
        if not write_outputs:
            return
        sub_info = compute_rows_kpis(rows, all_job_due_dates)
        global_info = compute_global_kpis()
        current_job_ids = {int(row["job"]) for row in rows}
        repeated_job_ids = sorted(current_job_ids & previous_release_job_ids)
        release_csv_writer.writerow([
            int(event_id),
            str(release_type),
            f"{float(release_time):.4f}",
            f"{sub_info['objective_value']:.4f}",
            f"{sub_info['makespan']:.4f}",
            f"{sub_info['total_tardiness']:.4f}",
            f"{global_info['objective_value']:.4f}",
            f"{global_info['makespan']:.4f}",
            f"{global_info['total_tardiness']:.4f}",
            len(getattr(orch, "_committed_jobs", [])),
            len(rows),
            len(current_job_ids),
            len(repeated_job_ids),
            ";".join(str(job_id) for job_id in repeated_job_ids),
            f"{solve_time:.6f}",
        ])
        previous_release_job_ids = current_job_ids

    init_jobs = sample_initial_jobs(gen.cfg, rng=rng, base_job_id=0, t_arrive=0.0)
    if init_jobs:
        for job in init_jobs:
            all_job_arrive_times[job.job_id] = float(job.meta.get("t_arrive", 0.0))
            all_job_is_urgent[job.job_id] = bool(job.meta.get("is_urgent", False))
            all_job_due_date_k[job.job_id] = float(job.meta.get("due_date_k", 0.0))
            record_env_job_info(job, event_id=0, phase="init", inter_arrival=0.0)
        t_start = time.perf_counter()
        release_count += register_initial_jobs(orch, gen, init_jobs, all_job_due_dates, t0=0.0)
        solve_time = time.perf_counter() - t_start
        write_release_log(0, "INIT", 0.0, getattr(orch, "last_batch_rows", []), solve_time)
        collect_subproblem_stats(orch._committed_jobs, 0.0) # [STATS: INITIAL SUBPROBLEM]
        raw_s0 = get_raw_state_info(orch, 0.0)
        if not FAST_MODE:
            save_details(orch, plot_seq+1, 0.0, "_INIT")
            plot_global_gantt(build_plot_rows(0.0), os.path.join(csv_dir, f"global_r{plot_seq:03d}_t0.png"), t_now=0.0, title="Initial")
        plot_seq += 1

    # Always enable baseline cadence run in hrl_main.py to populate Baseline columns in CSV,
    # unless disable_main_baseline is explicitly set to True.
    baseline_needed = not bool(getattr(configs, "disable_main_baseline", False))
    if baseline_needed:
        baseline_final_td, baseline_final_mk, baseline_release_count, baseline_event_td = run_cadence_baseline()
    else:
        print("[INFO] Skipping main cadence baseline simulation (disable_main_baseline=True).")
        baseline_final_td = 0.0
        baseline_final_mk = 0.0
        baseline_release_count = 0
        baseline_event_td = [0.0 for _ in range(int(max_events))]
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
            for j in new_jobs:
                all_job_due_dates[j.job_id] = j.meta["due_date"]
                all_job_arrive_times[j.job_id] = float(j.meta.get("t_arrive", t_now))
                all_job_is_urgent[j.job_id] = bool(j.meta.get("is_urgent", False))
                all_job_due_date_k[j.job_id] = float(j.meta.get("due_date_k", 0.0))
                record_env_job_info(j, event_id=int(stats["arrive"]), phase="arrival", inter_arrival=float(inter_arrival))
            orch.buffer.extend(new_jobs)
        stats["arrive"] += 1
        is_last_step = bool(stats["arrive"] >= int(max_events))
        K = int(getattr(configs, "hl_gate_decision_interval", 1))
        
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
        
        obs = calculate_hl_gate_state(
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
            release_count_so_far=gate_release_count,
            decision_steps_elapsed=max(0, int(stats["arrive"] - 1) // max(1, int(K))),
            is_last_step=is_last_step,
            buffer_jobs=orch.buffer,
        )
        is_decision_step = (stats["arrive"] % K == 0)
        
        if is_last_step:
            act = 1
        elif not is_decision_step:
            act = 0
        elif is_ppo:
            with torch.no_grad():
                logits, _ = hl_ppo_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
                if str(getattr(configs, "hl_eval_action_selection", "greedy")).lower() == "sample":
                    dist = torch.distributions.Categorical(logits=logits)
                    act = int(dist.sample().item())
                else:
                    act = int(torch.argmax(logits, dim=1).item())
        elif gate_policy == "slack_threshold":
            min_slack = float(raw_s[8])
            act = 1 if min_slack < float(getattr(configs, "hl_buffer_slack_release_threshold", 0.0)) else 0
        elif gate_policy == "random":
            prob = float(getattr(configs, "hl_random_release_prob", 0.087))
            act = 1 if (np.random.rand() < prob) else 0
        else:
            decision_step_idx = stats["arrive"] // K
            act = 1 if (decision_step_idx % configs.hl_gate_cadence == 0) else 0

        actual_td_logged = 0.0
        if act == 1:
            t_start = time.perf_counter()
            release_result = orch.event_release_and_reschedule(t_now, event_id=int(stats["arrive"]))
            solve_time = time.perf_counter() - t_start
            release_count += 1
            gate_release_count += 1
            if release_result.get("event") == "batch_finalized":
                print(
                    f"[Reschedule] Event ID={stats['arrive']} | Jobs={release_result['jobs_count']} | K={release_result['K']} | "
                    f"MK={release_result['sub_makespan']:.2f} | TD={release_result['sub_tardiness']:.2f} | SolveTime={solve_time:.3f}s"
                )
            write_release_log(int(stats["arrive"]), "EVENT", t_now, release_result.get("rows", getattr(orch, "last_batch_rows", [])), solve_time)
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
                t_start = time.perf_counter()
                flush_result = orch.event_release_and_reschedule(t_next_future, event_id=int(max_events) + 1)
                solve_time = time.perf_counter() - t_start
                release_count += 1
                write_release_log(int(max_events) + 1, "FLUSH", t_next_future, flush_result.get("rows", getattr(orch, "last_batch_rows", [])), solve_time)
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
            r_mk_raw = -(((mk_ratio + 1.0) ** 2) - 1.0) * float(configs.hl_mk_reward_coef)
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
            if r_shape > 0.0:
                r_shape = 0.0
            baseline_td_delta_csv = f"{baseline_td_delta:.2f}"
            agent_td_delta_csv = f"{agent_td_delta:.2f}"
            last_release_event_idx = int(stats["arrive"])
            last_release_td = float(actual_td_logged)
        elif done and td_signal_source == "agent_only" and td_credit_mode in ("step_only", "redistribute") and abs(td_step_coef) > 1e-12:
            agent_td_delta = float(total_td - last_release_td)
            r_shape = float((-(agent_td_delta) / scale) * td_step_coef)
            if r_shape > 0.0:
                r_shape = 0.0
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

        t_prev = t_now
        t_next = t_next_future

    final_stats = orch.get_final_kpi_stats(all_job_due_dates)
    total_td, final_mk = float(final_stats["tardiness"]), float(final_stats["makespan"])
    prior_mk = list((aggregate_prior or {}).get("makespan", []))
    prior_td = list((aggregate_prior or {}).get("tardiness", []))
    prior_release = list((aggregate_prior or {}).get("release_count", []))
    all_mk = prior_mk + [final_mk]
    all_td = prior_td + [total_td]
    all_obj = [0.5 * mk + 0.5 * td for mk, td in zip(all_mk, all_td)]
    mk_mean, mk_std = _mean_std(all_mk)
    td_mean, td_std = _mean_std(all_td)
    obj_mean, obj_std = _mean_std(all_obj)
    release_mean, release_std = _mean_std(prior_release + [float(release_count)])
    aggregate_tail = [
        len(prior_mk) + 1,
        f"{mk_mean:.2f}",
        f"{mk_std:.2f}",
        f"{td_mean:.2f}",
        f"{td_std:.2f}",
        f"{obj_mean:.2f}",
        f"{obj_std:.2f}",
        f"{release_mean:.2f}",
        f"{release_std:.2f}",
    ]
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
        total_r_stab += float(row[-12])
        total_r_buf += float(row[-11])
        total_r_shape += float(row[-10])
        total_r_td += float(row[-9])
        total_r_mk += float(row[-8])
        if write_outputs:
            raw_csv_writer.writerow(row + [""] * 9)
    if write_outputs:
        for row in pending_obs_rows:
            obs_csv_writer.writerow(row + [""] * 9)
    
    # [DISABLED] Skip summary boxplots
    # plot_simulation_summary_stats(all_sim_job_stats, csv_dir)

    summary_common_tail = [
        "", "", "", "", "", "", "",
        f"{total_td:.2f}",
        f"{total_cumulative_reward:.4f}",
        f"{total_r_stab:.4f}",
        f"{total_r_buf:.4f}",
        f"{total_r_shape:.4f}",
        f"{total_r_td:.4f}",
        f"{total_r_mk:.4f}",
        "", "",
        f"{total_td:.2f}",
        "",
        f"{final_mk:.2f}",
        f"{total_td:.2f}",
        release_count,
    ]
    raw_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 19 + summary_common_tail + aggregate_tail
    obs_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 22 + summary_common_tail + aggregate_tail
    simulation_elapsed = time.perf_counter() - t_sim_start
    if write_outputs:
        raw_csv_writer.writerow(raw_summary)
        obs_csv_writer.writerow(obs_summary)
        release_csv_writer.writerow([
            "END",
            "SUMMARY",
            f"{t_now:.4f}",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "Total_Simulation_Time_Sec",
            f"{simulation_elapsed:.6f}"
        ])
        write_env_job_info_csv()
        raw_csv_file.close()
        obs_csv_file.close()
        release_csv_file.close()
    return final_mk, {
        "release_count": release_count,
        "total_tardiness": total_td,
        "output_dir": csv_dir,
        "elapsed_time_sec": simulation_elapsed
    }

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))
    print("-" * 20 + " Dynamic FJSP HRL Evaluation " + "-" * 20)
    
    # [FIXED] Dynamic output naming based on actual policy
    plot_dir = getattr(configs, "plot_global_dir", "plots/global")
    main_sample_runs = int(getattr(configs, "main_sample_runs", -1))
    if main_sample_runs > 0:
        eval_runs = main_sample_runs
    else:
        eval_runs = int(getattr(configs, "eval_runs_per_instance", 10))
    base_seed = int(getattr(configs, "event_seed", 42))
    ppo_model_path = str(getattr(configs, "ll_ppo_model_path", "") or "")
    ppo_model_name = os.path.splitext(os.path.basename(ppo_model_path))[0] if ppo_model_path else ""
    previous_hl_action_selection = str(getattr(configs, "hl_eval_action_selection", "greedy"))
    if previous_hl_action_selection != "greedy":
        configs.hl_eval_action_selection = "sample"
    previous_ll_action_selection = str(getattr(configs, "ll_eval_action_selection", "greedy"))
    if previous_ll_action_selection != "greedy":
        configs.ll_eval_action_selection = "sample"

    makespans: List[float] = []
    tardiness_values: List[float] = []
    release_counts: List[float] = []
    elapsed_times: List[float] = []
    run_records = []
    output_dir = None
    stats = None

    best_obj = float('inf')
    best_seed_info = None
    best_run_idx = -1

    try:
        for run_idx in range(eval_runs):
            sample_seed = base_seed + run_idx
            aggregate_prior = {
                "makespan": makespans,
                "tardiness": tardiness_values,
                "release_count": release_counts,
            }
            mk, stats = run_event_driven_until_nevents(
                max_events=int(configs.event_horizon),
                interarrival_mean=configs.interarrival_mean,
                burst_K=configs.burst_size,
                plot_global_dir=plot_dir,
                write_outputs=(eval_runs == 1),  # Run directly with output writing if only 1 run
                seed_override=base_seed,
                sample_seed_override=sample_seed,
                aggregate_prior=aggregate_prior,
            )
            makespans.append(float(mk))
            tardiness_values.append(float(stats["total_tardiness"]))
            release_counts.append(float(stats["release_count"]))
            elapsed_times.append(float(stats["elapsed_time_sec"]))
            obj = 0.5 * float(mk) + 0.5 * float(stats["total_tardiness"])
            run_records.append({
                "run": run_idx + 1,
                "ppo_model_name": ppo_model_name,
                "ppo_model_path": ppo_model_path,
                "env_seed": base_seed,
                "sample_seed": sample_seed,
                "makespan": float(mk),
                "total_tardiness": float(stats["total_tardiness"]),
                "obj": obj,
                "release_count": int(stats["release_count"]),
                "elapsed_time_sec": float(stats["elapsed_time_sec"]),
            })
            if obj < best_obj:
                best_obj = obj
                best_seed_info = (base_seed, sample_seed)
                best_run_idx = run_idx + 1
            print(
                f"Run {run_idx + 1:02d}/{eval_runs} env_seed={base_seed} sample_seed={sample_seed} | "
                f"MK={float(mk):.3f}, TD={float(stats['total_tardiness']):.3f}, "
                f"Releases={int(stats['release_count'])}, "
                f"Elapsed={float(stats['elapsed_time_sec']):.3f}s"
            )

        # Replay the best run with write_outputs=True to output Gantt charts, logs and matrices
        if best_seed_info is not None:
            if eval_runs > 1:
                print("=" * 60)
                print(f"[BEST RUN REPLAY] Re-running Best Run #{best_run_idx} (Seed={best_seed_info[1]}, Obj={best_obj:.3f}) with output writing enabled...")
                print("=" * 60)
                
                _, best_stats = run_event_driven_until_nevents(
                    max_events=int(configs.event_horizon),
                    interarrival_mean=configs.interarrival_mean,
                    burst_K=configs.burst_size,
                    plot_global_dir=plot_dir,
                    write_outputs=True,  # Enable output writing for the best run
                    seed_override=best_seed_info[0],
                    sample_seed_override=best_seed_info[1],
                    aggregate_prior=None,
                )
                if best_stats.get("output_dir"):
                    output_dir = best_stats["output_dir"]
            else:
                # If only 1 run, we already wrote the outputs in the first trial
                if stats is not None and stats.get("output_dir"):
                    output_dir = stats["output_dir"]
    finally:
        configs.hl_eval_action_selection = previous_hl_action_selection
        configs.ll_eval_action_selection = previous_ll_action_selection

    mk_mean, mk_std = _mean_std(makespans)
    td_mean, td_std = _mean_std(tardiness_values)
    obj_values = [0.5 * mk + 0.5 * td for mk, td in zip(makespans, tardiness_values)]
    obj_mean, obj_std = _mean_std(obj_values)
    release_mean, release_std = _mean_std(release_counts)
    elapsed_mean, elapsed_std = _mean_std(elapsed_times)
    elapsed_total = sum(elapsed_times)
    if output_dir is None:
        output_dir = plot_dir
        os.makedirs(output_dir, exist_ok=True)
    sample_csv_path = os.path.join(output_dir, "sample_runs_summary.csv")
    with open(sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run", "ppo_model_name", "ppo_model_path", "env_seed", "sample_seed",
                "makespan", "total_tardiness", "obj", "release_count", "elapsed_time_sec"
            ],
        )
        writer.writeheader()
        for row in run_records:
            writer.writerow({
                "run": row["run"],
                "ppo_model_name": row["ppo_model_name"],
                "ppo_model_path": row["ppo_model_path"],
                "env_seed": row["env_seed"],
                "sample_seed": row["sample_seed"],
                "makespan": f"{row['makespan']:.6f}",
                "total_tardiness": f"{row['total_tardiness']:.6f}",
                "obj": f"{row['obj']:.6f}",
                "release_count": row["release_count"],
                "elapsed_time_sec": f"{row['elapsed_time_sec']:.6f}",
            })
        writer.writerow({
            "run": "mean",
            "ppo_model_name": ppo_model_name,
            "ppo_model_path": ppo_model_path,
            "env_seed": base_seed,
            "sample_seed": "",
            "makespan": f"{mk_mean:.6f}",
            "total_tardiness": f"{td_mean:.6f}",
            "obj": f"{obj_mean:.6f}",
            "release_count": f"{release_mean:.6f}",
            "elapsed_time_sec": f"{elapsed_mean:.6f}",
        })
        writer.writerow({
            "run": "std",
            "ppo_model_name": ppo_model_name,
            "ppo_model_path": ppo_model_path,
            "env_seed": base_seed,
            "sample_seed": "",
            "makespan": f"{mk_std:.6f}",
            "total_tardiness": f"{td_std:.6f}",
            "obj": f"{obj_std:.6f}",
            "release_count": f"{release_std:.6f}",
            "elapsed_time_sec": f"{elapsed_std:.6f}",
        })
        writer.writerow({
            "run": "total",
            "ppo_model_name": ppo_model_name,
            "ppo_model_path": ppo_model_path,
            "env_seed": base_seed,
            "sample_seed": "",
            "makespan": "",
            "total_tardiness": "",
            "obj": "",
            "release_count": f"{sum(release_counts):.6f}",
            "elapsed_time_sec": f"{elapsed_total:.6f}",
        })
    print(
        f"\nSample x{eval_runs} | "
        f"MK mean/std: {mk_mean:.3f}/{mk_std:.3f}, "
        f"TD mean/std: {td_mean:.3f}/{td_std:.3f}, "
        f"Obj mean/std: {obj_mean:.3f}/{obj_std:.3f}, "
        f"Releases mean/std: {release_mean:.3f}/{release_std:.3f}, "
        f"Elapsed total: {elapsed_total:.3f}s (mean/std: {elapsed_mean:.3f}/{elapsed_std:.3f}s)"
    )
    print(f"Sample run CSV: {sample_csv_path}")

if __name__ == "__main__": main()
