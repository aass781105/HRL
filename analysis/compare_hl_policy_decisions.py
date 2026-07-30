import os
import sys
import csv
import time
import numpy as np
import pandas as pd

# 1. Manually parse arguments at the very top to avoid conflicts with params.py
config_path = ""
seeds = list(range(1, 11))
pre_window = 5
post_window = 10

if "--config" in sys.argv:
    idx = sys.argv.index("--config")
    config_path = sys.argv[idx + 1]
else:
    print("Error: --config <path> is required!")
    sys.exit(1)

if "--seeds" in sys.argv:
    idx = sys.argv.index("--seeds")
    temp_seeds = []
    for val in sys.argv[idx + 1:]:
        if val.startswith("--"):
            break
        try:
            temp_seeds.append(int(val))
        except ValueError:
            break
    if temp_seeds:
        seeds = temp_seeds

if "--pre_window" in sys.argv:
    idx = sys.argv.index("--pre_window")
    pre_window = int(sys.argv[idx + 1])

if "--post_window" in sys.argv:
    idx = sys.argv.index("--post_window")
    post_window = int(sys.argv[idx + 1])

# Clean sys.argv to contain ONLY --config before importing other project files
sys.argv = ["", "--config", config_path]

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project environment and tools
import torch
from params import configs
from hl_gate_env import HLGateEnv
from model.hl_gate_state import HL_GATE_STATE_DIM, HL_LL_BUFFER_EMBED_DIM, get_hl_gate_state_dim
from model.hl_ppo_gate_model import HLPPOGateNet
from common_utils import resolve_high_level_weight_path
from model.hl_gate_state import _buffer_machine_demand_stats

def calculate_tardiness_components(env):
    """
    Computes components of tardiness:
    1. Confirmed TD (Finished jobs that are committed in history)
    2. WIP TD before reschedule (Active WIP jobs)
    """
    t_now = env.t_now
    all_due = env.all_job_due_dates
    
    # 1. Finished job IDs: started but no longer active
    active_jids = {js.job_id for js in env.orch._last_jobs_snapshot}
    finished_jids = set(env.orch._job_history_finishes.keys()) - active_jids
    
    confirmed_td = 0.0
    for jid in finished_jids:
        due = all_due.get(jid, 0.0)
        fin_time = env.orch._job_history_finishes.get(jid, 0.0)
        confirmed_td += max(0.0, fin_time - due)
        
    # 2. WIP job IDs: active and already started
    wip_jids = active_jids & set(env.orch._job_history_finishes.keys())
    
    # Group current plan by job
    by_j_plan = {}
    for r in env.orch._last_full_rows:
        by_j_plan.setdefault(int(r["job"]), []).append(r)
        
    wip_td_before = 0.0
    for jid in wip_jids:
        due = all_due.get(jid, 0.0)
        j_rows = by_j_plan.get(jid, [])
        if j_rows:
            p_finish = max(float(r["end"]) for r in j_rows)
            wip_td_before += max(0.0, p_finish - due)
            
    return confirmed_td, wip_td_before

def run_simulation_episode(env, seed, policy_type, ppo_model=None, gate_device=None):
    """
    Runs a single simulation episode under a specific policy and seed,
    collecting decision trace logs.
    """
    # Force policy parameter
    configs.hl_gate_policy = "ppo" if policy_type == "ppo" else "slack_threshold"
    if policy_type == "slack_threshold":
        configs.hl_buffer_slack_release_threshold = 0.0
        
    obs, info = env.reset(seed=seed, options={"needs_baseline": False, "count_episode": False})
    done = False
    
    trace_rows = []
    
    while not done:
        event_id = env.events_done
        t_now = env.t_now
        
        # Calculate WIP and Buffer stats before action
        buf_stats = env._get_buffer_stats(t_now)
        wip_stats = env.orch.get_wip_stats(t_now)
        
        # Completed & WIP tardiness before reschedule
        confirmed_td_before, wip_td_before = calculate_tardiness_components(env)
        
        # Machine load pressure
        rem = np.maximum(0.0, env.orch.machine_free_time - t_now)
        if rem.size > 0:
            machine_load_std = float(np.std(rem))
            machine_load_span = float(np.max(rem) - np.min(rem))
            avg_machine_load = float(rem.mean())
        else:
            machine_load_std = 0.0
            machine_load_span = 0.0
            avg_machine_load = 0.0
            
        # Overlap and demand statistics
        demand_max_share, load_overlap, demand_entropy = _buffer_machine_demand_stats(
            env.orch.buffer, env.orch.machine_free_time, t_now, env.M
        )
        
        # Decide action and extract PPO release probability
        ppo_release_prob = ""
        if policy_type == "ppo" and ppo_model is not None:
            with torch.no_grad():
                logits, _ = ppo_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
                probs = torch.softmax(logits, dim=1).squeeze(0)
                ppo_release_prob = float(probs[1].item())
                
                # Greedy selection
                action = int(torch.argmax(logits, dim=1).item())
        else:
            # Heuristic slack0 selection
            min_slack = buf_stats["min_slack"]
            action = 1 if (len(env.orch.buffer) > 0 and min_slack < 0.0) else 0
            
        # Record pre-decision stats
        td_actual = float(env.orch.get_total_tardiness_estimate(env.all_job_due_dates))
        mk_estimate = float(env.orch.machine_free_time.max())
        
        # Step the environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        # Record post-decision stats
        wip_stats_after = env.orch.get_wip_stats(t_now) # t_now is before step, but we query after scheduler has run
        rescheduled_td_after = float(wip_stats_after["planned_td"])
        
        trace_rows.append({
            "Seed": seed,
            "Policy": policy_type,
            "Event_ID": event_id,
            "Time": t_now,
            "Action": action,
            "PPO_Release_Prob": ppo_release_prob,
            "Tardiness_Actual": td_actual,
            "Makespan_Estimate": mk_estimate,
            "Confirmed_TD_Before": confirmed_td_before,
            "WIP_TD_Before": wip_td_before,
            "Rescheduled_TD_After": rescheduled_td_after,
            "WIP_Count_Before": int(wip_stats["wip_count"]),
            "WIP_Min_Slack_Before": float(wip_stats["wip_min_slack"]),
            "WIP_Total_Rem_Work_Before": float(wip_stats["total_rem_work"]),
            "WIP_Tardy_Ratio_Before": float(wip_stats["wip_tardy_ratio"]),
            "Machine_Load_Std_Before": machine_load_std,
            "Machine_Load_Span_Before": machine_load_span,
            "Avg_Machine_Load_Before": avg_machine_load,
            "Buffer_Count_Before": len(env.orch.buffer),
            "Buffer_Min_Slack_Before": float(buf_stats["min_slack"]),
            "Buffer_Neg_Slack_Sum_Before": float(buf_stats["neg_slack_sum"]),
            "Buffer_Total_Work_Before": float(buf_stats["total_work"]),
            "Buffer_WIP_Load_Overlap_Before": load_overlap,
            "Demand_Entropy_Before": demand_entropy,
            "Demand_Max_Share_Before": demand_max_share
        })
        
    return trace_rows

def main():
    global config_path, seeds, pre_window, post_window
    
    # Output directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, "analysis_results", "policy_compare", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Comparing PPO vs Slack0 policies on seeds {seeds}...")
    print(f"Results will be saved to: {out_dir}")
    
    # Initialize Environment
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    # Load PPO Model
    gate_device = torch.device(getattr(configs, "device", "cpu"))
    ppo_model = HLPPOGateNet(
        obs_dim=get_hl_gate_state_dim(configs),
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
        manual_obs_dim=get_hl_gate_state_dim(configs),
        ll_embed_raw_dim=int(getattr(configs, "hl_ll_buffer_embedding_dim", HL_LL_BUFFER_EMBED_DIM)) if bool(getattr(configs, "hl_use_ll_buffer_embedding", False)) else 0,
        ll_embed_proj_dim=int(getattr(configs, "hl_ll_buffer_projection_dim", 16)),
        initial_release_prob=float(getattr(configs, "hl_initial_release_prob", -1.0)),
    ).to(gate_device)
    
    hl_path = resolve_high_level_weight_path(getattr(configs, "hl_ppo_model_path", ""))
    ppo_model.load_state_dict(torch.load(hl_path, map_location=gate_device, weights_only=True))
    ppo_model.eval()
    print(f"Loaded high-level PPO model: {hl_path}")
    
    ppo_decision_trace = []
    slack_decision_trace = []
    
    summary_rows = []
    
    ppo_divergence_windows = []
    slack_divergence_windows = []
    
    ppo_release_aligned = []
    slack_release_aligned = []
    
    for seed in seeds:
        print(f"\n---> Running Seed {seed}...")
        # 1. Run PPO
        ppo_trace = run_simulation_episode(env, seed, "ppo", ppo_model, gate_device)
        # 2. Run Heuristic Slack0
        slack_trace = run_simulation_episode(env, seed, "slack0")
        
        ppo_decision_trace.extend(ppo_trace)
        slack_decision_trace.extend(slack_trace)
        
        # Calculate summary metrics
        final_ppo = ppo_trace[-1]
        final_slack = slack_trace[-1]
        summary_rows.append({
            "Seed": seed,
            "PPO_Makespan": final_ppo["Makespan_Estimate"],
            "PPO_Tardiness": final_ppo["Tardiness_Actual"],
            "PPO_Releases": final_ppo["Event_ID"] + 1,
            "Slack_Makespan": final_slack["Makespan_Estimate"],
            "Slack_Tardiness": final_slack["Tardiness_Actual"],
        })
        
        # 3. Find first divergence point
        div_idx = -1
        for idx in range(min(len(ppo_trace), len(slack_trace))):
            if ppo_trace[idx]["Action"] != slack_trace[idx]["Action"]:
                div_idx = idx
                break
                
        if div_idx != -1:
            div_step = ppo_trace[div_idx]["Event_ID"]
            div_type = "A" if ppo_trace[div_idx]["Action"] == 1 else "B"
            print(f"  First divergence found at Event_ID {div_step} (Type {div_type}: PPO={ppo_trace[div_idx]['Action']}, Slack0={slack_trace[div_idx]['Action']})")
            
            # Extract window for PPO and Slack0
            # relative steps from -pre_window to +post_window
            for rel in range(-pre_window, post_window + 1):
                target_idx = div_idx + rel
                # PPO Window
                if 0 <= target_idx < len(ppo_trace):
                    row = ppo_trace[target_idx].copy()
                    row["Divergence_ID"] = f"seed{seed}_ev{div_step}"
                    row["Divergence_Type"] = div_type
                    row["Relative_Step"] = rel
                    ppo_divergence_windows.append(row)
                # Slack0 Window
                if 0 <= target_idx < len(slack_trace):
                    row = slack_trace[target_idx].copy()
                    row["Divergence_ID"] = f"seed{seed}_ev{div_step}"
                    row["Divergence_Type"] = div_type
                    row["Relative_Step"] = rel
                    slack_divergence_windows.append(row)
        else:
            print("  No divergence found on this seed! Both policies matched actions 100%.")
            
        # 4. Release Aligned Extraction
        ppo_releases = [r for r in ppo_trace if r["Action"] == 1]
        slack_releases = [r for r in slack_trace if r["Action"] == 1]
        
        for rel_idx in range(len(ppo_releases)):
            row = ppo_releases[rel_idx].copy()
            row["Release_Index"] = rel_idx + 1
            ppo_release_aligned.append(row)
            
        for rel_idx in range(len(slack_releases)):
            row = slack_releases[rel_idx].copy()
            row["Release_Index"] = rel_idx + 1
            slack_release_aligned.append(row)
                
    # Save CSVs (separated)
    pd.DataFrame(ppo_decision_trace).to_csv(os.path.join(out_dir, "decision_trace_ppo.csv"), index=False)
    pd.DataFrame(slack_decision_trace).to_csv(os.path.join(out_dir, "decision_trace_slack0.csv"), index=False)
    
    pd.DataFrame(ppo_divergence_windows).to_csv(os.path.join(out_dir, "divergence_windows_ppo.csv"), index=False)
    pd.DataFrame(slack_divergence_windows).to_csv(os.path.join(out_dir, "divergence_windows_slack0.csv"), index=False)
    
    pd.DataFrame(ppo_release_aligned).to_csv(os.path.join(out_dir, "release_aligned_ppo.csv"), index=False)
    pd.DataFrame(slack_release_aligned).to_csv(os.path.join(out_dir, "release_aligned_slack0.csv"), index=False)
    
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary_by_seed.csv"), index=False)
    
    print(f"\nSuccess! Separated CSV files saved successfully at: {out_dir}")

if __name__ == "__main__":
    main()
