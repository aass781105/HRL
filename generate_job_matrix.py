import os
import time
import copy
import csv
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from params import configs
from common_utils import *
from global_env import (
    GlobalTimelineOrchestrator,
    EventBurstGenerator,
    split_matrix_to_jobs,
)
from data_utils import SD2_instance_generator, generate_due_dates
from dynamic_job_stream import register_initial_jobs, sample_initial_jobs

import torch
from model.gate_state import calculate_gate_state
from model.ppo_gate_model import PPOGateNet

def run_matrix_tracing_simulation():
    seed = int(getattr(configs, "event_seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # [FORCE CONSISTENCY]
    setattr(configs, "eval_action_selection", "greedy")
    
    base_cfg = copy.deepcopy(configs); rng = np.random.default_rng(seed)
    gen = EventBurstGenerator(SD2_instance_generator, base_cfg, configs.n_m, configs.interarrival_mean, lambda _r: int(configs.burst_size), rng)
    orch = GlobalTimelineOrchestrator(configs.n_m, gen, lambda b, o: list(range(len(b))), t0=0.0)

    gate_policy = str(getattr(configs, "gate_policy", "ppo")).lower()
    ppo_gate_model = None
    gate_device = torch.device(getattr(configs, "device", "cpu"))
    if gate_policy == "ppo":
        ppo_gate_model = PPOGateNet(
            obs_dim=22,
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
        ppo_gate_model.load_state_dict(torch.load(configs.ppo_gate_model_path, map_location=gate_device, weights_only=True))
        ppo_gate_model.eval()

    job_matrix_data = {}
    all_job_due_dates = {}
    
    def get_buffer_stats_for_obs(orchestrator, t_now):
        if not orchestrator.buffer: return {"buffer_neg_slack_ratio": 0.0, "min_slack": 0.0, "avg_slack": 0.0, "slack_std": 0.0, "slack_q25": 0.0}
        slacks, neg_slack_count = [], 0
        for j in orchestrator.buffer:
            min_work = float(j.meta.get("min_total_proc_time", 0.0))
            due = all_job_due_dates[j.job_id]; s = due - t_now - min_work
            slacks.append(s)
            if t_now + min_work > due: neg_slack_count += 1
        return {"buffer_neg_slack_ratio": neg_slack_count / len(orchestrator.buffer), "min_slack": min(slacks), "avg_slack": sum(slacks) / len(slacks), "slack_std": float(np.std(slacks)), "slack_q25": float(np.percentile(slacks, 25))}

    def record_snapshot(orchestrator, t_now, step_label, is_terminal=False):
        job_rows = {}
        for r in orchestrator._last_full_rows: job_rows.setdefault(r["job"], []).append(r)
        for j in orchestrator.buffer:
            jid = j.job_id; slack = float(j.meta.get("due_date", 0.0)) - t_now - float(j.meta.get("total_proc_time", 0.0))
            job_matrix_data.setdefault(jid, {})[step_label] = f"S:{slack:.1f} Buffer"
        if orchestrator._last_jobs_snapshot:
            for js in orchestrator._last_jobs_snapshot:
                jid = js.job_id; rows = job_rows.get(jid, []); started_ops = [r for r in rows if r["start"] < t_now]; op_idx = len(started_ops); rem_work = 0.0
                if op_idx < len(js.operations):
                    for op in js.operations[op_idx:]:
                        if op.time_row:
                            rv = np.array(op.time_row); rem_work += np.min(rv[rv > 0]) if rv[rv > 0].size > 0 else 0.0
                slack = float(js.meta.get("due_date", 0.0)) - t_now - rem_work
                if is_terminal: status = "Done" if op_idx >= len(js.operations) else f"Op{op_idx}"
                else: status = f"Op{op_idx}" if op_idx < len(js.operations) else "Finished"
                job_matrix_data.setdefault(jid, {})[step_label] = status if status in ["Done", "Finished"] else f"S:{slack:.1f} {status}"

    # --- 模擬開始 ---
    INIT_J = int(getattr(configs, "init_jobs", 0))
    if INIT_J > 0:
        init_jobs = sample_initial_jobs(configs, rng=rng, base_job_id=0, t_arrive=0.0)
        register_initial_jobs(orch, gen, init_jobs, all_job_due_dates, t0=0.0)
        record_snapshot(orch, 0.0, "Step_0")

    t_now, t_prev = 0.0, 0.0; t_next = gen.sample_next_time(t_now); max_events = int(configs.event_horizon)
    steps_since_last_release = 0
    for i in tqdm(range(1, max_events + 1), desc="Tracing"):
        t_now = float(t_next); new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            for j in new_jobs: all_job_due_dates[j.job_id] = j.meta["due_date"]
            orch.buffer.extend(new_jobs)
        record_snapshot(orch, t_now, f"Step_{i}")
        if gate_policy == "ppo":
            mft_abs = np.asarray(orch.machine_free_time, dtype=float); rem = np.maximum(0.0, mft_abs - t_now); max_load = np.max(rem); mean_pt = (float(configs.low) + float(configs.high)) / 2.0
            b_dict = get_buffer_stats_for_obs(orch, t_now); w_dict = orch.get_wip_stats(t_now)
            inter_arrival_scaled = float((t_now - t_prev) / mean_pt) if mean_pt > 0 else 0.0
            obs = calculate_gate_state(
                len(orch.buffer),
                orch.machine_free_time,
                t_now,
                configs.n_m,
                0,
                mean_pt,
                orch.compute_weighted_idle(t_now, float(max_load)) if max_load > 0 else 0.0,
                orch.compute_unweighted_idle(t_now, float(max_load)) if max_load > 0 else 0.0,
                b_dict,
                w_dict,
                inter_arrival_scaled=inter_arrival_scaled,
                steps_since_last_release=steps_since_last_release,
                is_last_step=bool(i >= max_events),
            )
            with torch.no_grad():
                logits, _ = ppo_gate_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
                if str(getattr(configs, "eval_action_selection", "greedy")).lower() == "sample":
                    act = int(torch.distributions.Categorical(logits=logits).sample().item())
                else:
                    act = int(torch.argmax(logits, dim=1).item())
            if float(obs[-1]) >= 0.5:
                act = 1
        else:
            act = 1 if (i >= max_events or (i % configs.gate_cadence == 0)) else 0
        if act == 1:
            orch.event_release_and_reschedule(t_now)
            steps_since_last_release = 0
        else:
            orch.tick_without_release(t_now)
            steps_since_last_release += 1
        t_prev = t_now
        t_next = gen.sample_next_time(t_now)

    record_snapshot(orch, t_now, "Pre_Flush")
    while len(orch.buffer) > 0: orch.event_release_and_reschedule(t_now)
    final_mk = float(np.max(orch.machine_free_time)); record_snapshot(orch, final_mk, "Final_Flush", is_terminal=True)
    job_finishes = {}
    for r in orch._global_rows: jid = int(r["job"]); job_finishes[jid] = max(job_finishes.get(jid, 0.0), float(r["end"]))
    for r in orch._last_full_rows: jid = int(r["job"]); job_finishes[jid] = max(job_finishes.get(jid, 0.0), float(r["end"]))
    df = pd.DataFrame.from_dict(job_matrix_data, orient='index'); df['Raw_Due_Date'] = [all_job_due_dates.get(jid, 0.0) for jid in df.index]; df['Raw_Finish_Time'] = [job_finishes.get(jid, 0.0) for jid in df.index]; df['Final_TD'] = [max(0.0, job_finishes.get(jid, 0.0) - all_job_due_dates[jid]) if jid in all_job_due_dates else 0.0 for jid in df.index]
    df.sort_index(inplace=True); suffix = f"PPO_{getattr(configs, 'ppo_gate_name', 'default')}" if gate_policy=="ppo" else f"Cadence_{getattr(configs, 'gate_cadence', 1)}"
    out_path = f"matrix_tracing_{suffix}.csv"; df.to_csv(out_path); print(f"Matrix saved to: {out_path}")

if __name__ == "__main__": run_matrix_tracing_simulation()
