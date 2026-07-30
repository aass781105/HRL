import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import torch

# 1. Manually parse arguments at the very top to avoid conflicts with params.py
config_path = ""
if "--config" in sys.argv:
    idx = sys.argv.index("--config")
    config_path = sys.argv[idx + 1]
else:
    print("Error: --config <path> is required!")
    sys.exit(1)

# Clean sys.argv to contain ONLY --config before importing other project files
sys.argv = ["", "--config", config_path]

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project environment and tools
from params import configs
from hl_gate_env import HLGateEnv
from model.hl_gate_state import HL_GATE_STATE_DIM, HL_LL_BUFFER_EMBED_DIM, get_hl_gate_state_dim
from model.hl_ppo_gate_model import HLPPOGateNet
from common_utils import resolve_high_level_weight_path

def get_buffer_slack_stats(env):
    t_now = env.t_now
    all_due = env.all_job_due_dates
    
    buf_slacks = []
    buf_urgent = 0
    buf_normal = 0
    for j in env.orch.buffer:
        mw = float(j.meta.get("total_proc_time", 0.0))
        if mw <= 0.0:
            mw = float(sum(float(getattr(op, "avg_proc_time", 0.0)) for op in getattr(j, "operations", []) or []))
        due = all_due.get(j.job_id, 0.0)
        s = due - t_now - mw
        buf_slacks.append(s)
        if j.meta.get("is_urgent", False):
            buf_urgent += 1
        else:
            buf_normal += 1
            
    if buf_slacks:
        buf_min = min(buf_slacks)
        buf_avg = np.mean(buf_slacks)
        buf_q25 = np.percentile(buf_slacks, 25)
    else:
        buf_min, buf_avg, buf_q25 = 0.0, 0.0, 0.0
        
    return {
        "buf_min": buf_min, "buf_avg": buf_avg, "buf_q25": buf_q25,
        "buf_urgent": buf_urgent, "buf_normal": buf_normal
    }

def get_machine_stats(env):
    t_now = env.t_now
    mft = env.orch.machine_free_time
    rem_load = np.maximum(0.0, mft - t_now)
    avg_load = np.mean(rem_load)
    load_std = np.std(mft)
    load_span = np.max(mft) - np.min(mft)
    return avg_load, load_std, load_span

def get_ppo_release_probability(env, ppo_model, gate_device):
    obs = env._observe()
    with torch.no_grad():
        logits, _ = ppo_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
        probs = torch.softmax(logits, dim=1)
        prob_release = float(probs[0, 1].item())
    return prob_release

def get_ppo_action(obs, ppo_model, gate_device):
    with torch.no_grad():
        logits, _ = ppo_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
        action = int(torch.argmax(logits, dim=1).item())
    return action

def get_slack0_action(env):
    buf_stats = env._get_buffer_stats(env.t_now)
    min_slack = buf_stats["min_slack"]
    action = 1 if (len(env.orch.buffer) > 0 and min_slack < 0.0) else 0
    return action

def main():
    # Load analysis settings from config
    seed = int(getattr(configs, "branching_seed", 2))
    pre_policy = str(getattr(configs, "branching_pre_policy", "cadence")).lower()
    pre_cadence = int(getattr(configs, "branching_pre_cadence", 5))
    switch_event = int(getattr(configs, "branching_switch_event", 40))
    event_horizon = int(configs.event_horizon)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, "analysis_results", "branching_compare", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting Branching Simulation on Seed {seed}...")
    print(f"Base Path: Event 0 to {switch_event} using Cadence {pre_cadence}")
    
    # Initialize Environment
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=event_horizon,
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
    
    # -------------------------------------------------------------------------
    # STEP 1: Find Divergence Event A and Subsequent Release Event B
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Probing for Divergence Event A & B ---")
    
    # Get PPO path decisions
    configs.hl_gate_policy = "ppo"
    obs, info = env.reset(seed=seed, options={"needs_baseline": False, "count_episode": False})
    
    # 1. Run base path to switch_event
    ppo_actions = []
    while env.events_done < switch_event:
        decision_idx = env.events_done
        action = 1 if (decision_idx % pre_cadence == 0) else 0
        ppo_actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        
    # 2. Run PPO policy onwards
    while env.events_done < event_horizon:
        action = get_ppo_action(obs, ppo_model, gate_device)
        ppo_actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
            
    # Get Slack0 path decisions
    configs.hl_gate_policy = "slack_threshold"
    configs.hl_buffer_slack_release_threshold = 0.0
    obs, info = env.reset(seed=seed, options={"needs_baseline": False, "count_episode": False})
    
    # 1. Run base path to switch_event
    slack_actions = []
    while env.events_done < switch_event:
        decision_idx = env.events_done
        action = 1 if (decision_idx % pre_cadence == 0) else 0
        slack_actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        
    # 2. Run Slack0 policy onwards
    while env.events_done < event_horizon:
        action = get_slack0_action(env)
        slack_actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
            
    # Find first divergence point after switch_event
    div_event_A = -1
    for ev in range(switch_event, min(len(ppo_actions), len(slack_actions))):
        if ppo_actions[ev] != slack_actions[ev]:
            div_event_A = ev
            break
            
    if div_event_A == -1:
        print("Error: No divergence point found after switch_event! Both policies matched actions 100%.")
        return
        
    # Determine who is early (A) and who is late (B)
    early_policy = ""
    late_policy = ""
    late_release_event_B = -1
    
    if slack_actions[div_event_A] == 1 and ppo_actions[div_event_A] == 0:
        early_policy = "slack0"
        late_policy = "ppo"
        # Find subsequent PPO release event B (B > A)
        for ev in range(div_event_A + 1, len(ppo_actions)):
            if ppo_actions[ev] == 1:
                late_release_event_B = ev
                break
    elif slack_actions[div_event_A] == 0 and ppo_actions[div_event_A] == 1:
        early_policy = "ppo"
        late_policy = "slack0"
        # Find subsequent Slack0 release event B (B > A)
        for ev in range(div_event_A + 1, len(slack_actions)):
            if slack_actions[ev] == 1:
                late_release_event_B = ev
                break
    else:
        print("Error: Divergence found, but both actions are identical?!")
        return
        
    if late_release_event_B == -1:
        print(f"Error: Late policy ({late_policy}) never released after divergence Event {div_event_A}!")
        return
        
    print(f"Divergence detected at Event A = {div_event_A}")
    print(f"  Early Policy: {early_policy} (Action=1)")
    print(f"  Late Policy:  {late_policy} (Action=0)")
    print(f"Next Late Policy release detected at Event B = {late_release_event_B}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Execute the Four Target Runs (Vertical Table Structure)
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Running 4 Target Branching Trajectories ---")
    
    # Get base actions up to A-1 (since they matched)
    base_actions_up_to_A_minus_1 = ppo_actions[:div_event_A] if early_policy == "ppo" else slack_actions[:div_event_A]
    
    # S0: RELEASE at A, HOLD from A+1 to B-1, RELEASE at B
    s0_seq = base_actions_up_to_A_minus_1 + [1] + [0] * (late_release_event_B - div_event_A - 1) + [1]
    
    # S1: RELEASE from A to B (cadence 1)
    s1_seq = base_actions_up_to_A_minus_1 + [1] * (late_release_event_B - div_event_A + 1)
    
    # Late Policy: HOLD from A to B-1, RELEASE at B
    late_seq = base_actions_up_to_A_minus_1 + [0] * (late_release_event_B - div_event_A) + [1]
    
    # Early Policy: RELEASE at A
    early_seq = ppo_actions[:late_release_event_B + 1] if early_policy == "ppo" else slack_actions[:late_release_event_B + 1]
    
    rows_data = []
    
    # Helper to capture release stats
    def capture_release_row_data(branch_name, target_event, action_sequence):
        env.reset(seed=seed, options={"needs_baseline": False, "count_episode": False})
        # Run up to target event
        for ev in range(target_event):
            env.step(action_sequence[ev])
            
        # Get Before State
        release_time = env.t_now
        ppo_release_prob = get_ppo_release_probability(env, ppo_model, gate_device)
        buf_stats = get_buffer_slack_stats(env)
        buffer_size_before = len(env.orch.buffer)
        
        # 1. Map current plan rows by job
        job_rows = {}
        for r in env.orch._last_full_rows:
            job_rows.setdefault(int(r["job"]), []).append(r)
            
        # 2. Identify WIP jobs before release (released and max(end) > release_time)
        all_due = env.all_job_due_dates
        wip_jids = set()
        conf_td = 0.0
        wip_td_before = 0.0
        wip_slacks_bf = []
        
        for js in env.orch._last_jobs_snapshot:
            jid = int(js.job_id)
            rows = job_rows.get(jid, [])
            if rows:
                p_finish = max(float(r["end"]) for r in rows)
                due = all_due.get(jid, 0.0)
                if p_finish <= release_time:
                    conf_td += max(0.0, p_finish - due)
                else:
                    wip_jids.add(jid)
                    wip_td_before += max(0.0, p_finish - due)
                    rem_w = sum(float(op.avg_proc_time) for op in js.operations)
                    wip_slacks_bf.append(due - (release_time + rem_w))
                    
        wip_before = len(wip_jids)
        total_td_before = conf_td + wip_td_before
        makespan_before = float(env.orch.machine_free_time.max())
        m_avg_before, m_std_before, m_span_before = get_machine_stats(env)
        
        if wip_slacks_bf:
            w_min_bf = min(wip_slacks_bf)
            w_avg_bf = np.mean(wip_slacks_bf)
            w_q25_bf = np.percentile(wip_slacks_bf, 25)
        else:
            w_min_bf, w_avg_bf, w_q25_bf = 0.0, 0.0, 0.0
            
        # Step the release
        env.step(1)
        
        # Get After State
        new_job_rows = {}
        for r in env.orch._last_full_rows:
            new_job_rows.setdefault(int(r["job"]), []).append(r)
            
        # Calculate WIP count after release (any job released and not completed at env.t_now)
        wip_after = 0
        wip_slacks_af = []
        for js in env.orch._last_jobs_snapshot:
            jid = int(js.job_id)
            rows = new_job_rows.get(jid, [])
            if rows:
                p_finish = max(float(r["end"]) for r in rows)
                due = all_due.get(jid, 0.0)
                if p_finish > env.t_now:
                    wip_after += 1
                    rem_w = sum(float(op.avg_proc_time) for op in js.operations)
                    wip_slacks_af.append(due - (env.t_now + rem_w))
                    
        # Calculate new tardiness for the EXACT SAME wip_jids set (disruption analysis)
        wip_td_after = 0.0
        for jid in wip_jids:
            rows = new_job_rows.get(jid, [])
            if rows:
                p_finish = max(float(r["end"]) for r in rows)
                due = all_due.get(jid, 0.0)
                wip_td_after += max(0.0, p_finish - due)
                
        makespan_after = float(env.orch.machine_free_time.max())
        resched_td_after = float(env.orch.get_total_tardiness_estimate(env.all_job_due_dates))
        total_td_after = resched_td_after
        m_avg_after, m_std_after, m_span_after = get_machine_stats(env)
        
        if wip_slacks_af:
            w_min_af = min(wip_slacks_af)
            w_avg_af = np.mean(wip_slacks_af)
            w_q25_af = np.percentile(wip_slacks_af, 25)
        else:
            w_min_af, w_avg_af, w_q25_af = 0.0, 0.0, 0.0
            
        return {
            "Seed": seed,
            "Branch": branch_name,
            "Event": target_event,
            "Time": f"{release_time:.4f}",
            "PPO_Prob": f"{ppo_release_prob:.4f}",
            "Buf_Size": buffer_size_before,
            "Buf_Slack_Min": f"{buf_stats['buf_min']:.2f}",
            "Buf_Slack_Avg": f"{buf_stats['buf_avg']:.2f}",
            "Buf_Slack_Q25": f"{buf_stats['buf_q25']:.2f}",
            "Buf_Urgent": buf_stats["buf_urgent"],
            "Buf_Normal": buf_stats["buf_normal"],
            
            "WIP_Bf": wip_before,
            "WIP_Af": wip_after,
            "MK_Bf": f"{makespan_before:.2f}",
            "MK_Af": f"{makespan_after:.2f}",
            
            "Conf_TD": f"{conf_td:.2f}",
            "WIP_TD_Bf": f"{wip_td_before:.2f}",
            "WIP_TD_Af": f"{wip_td_after:.2f}",
            "Total_TD_Bf": f"{total_td_before:.2f}",
            "Total_TD_Af": f"{total_td_after:.2f}",
            
            "WIP_Slack_Min_Bf": f"{w_min_bf:.2f}",
            "WIP_Slack_Min_Af": f"{w_min_af:.2f}",
            "WIP_Slack_Avg_Bf": f"{w_avg_bf:.2f}",
            "WIP_Slack_Avg_Af": f"{w_avg_af:.2f}",
            "WIP_Slack_Q25_Bf": f"{w_q25_bf:.2f}",
            "WIP_Slack_Q25_Af": f"{w_q25_af:.2f}",
            
            "Mch_Load_Avg_Bf": f"{m_avg_before:.2f}",
            "Mch_Load_Avg_Af": f"{m_avg_after:.2f}",
            "Mch_Load_Std_Bf": f"{m_std_before:.2f}",
            "Mch_Load_Std_Af": f"{m_std_after:.2f}",
            "Mch_Load_Span_Bf": f"{m_span_before:.2f}",
            "Mch_Load_Span_Af": f"{m_span_after:.2f}"
        }

    # 1. Late Policy at Event B
    print(f"Running Late Policy ({late_policy}) to capture Event B...")
    rows_data.append(capture_release_row_data(f"Late_Policy ({late_policy})", late_release_event_B, late_seq))
    
    # 2. Early Policy at Event A
    print(f"Running Early Policy ({early_policy}) to capture Event A...")
    rows_data.append(capture_release_row_data(f"Early_Policy ({early_policy})", div_event_A, early_seq))
    
    # 3. S0 Branch at Event B
    print("Running Early_Release_Then_Hold Branch to capture Event B...")
    rows_data.append(capture_release_row_data("Early_Release_Then_Hold", late_release_event_B, s0_seq))
    
    # 4. S1 Branch at Event B
    print("Running Early_Release_Always Branch to capture Event B...")
    rows_data.append(capture_release_row_data("Early_Release_Always", late_release_event_B, s1_seq))
    
    # -------------------------------------------------------------------------
    # STEP 3: Write Output CSV
    # -------------------------------------------------------------------------
    out_csv = os.path.join(out_dir, "release_timing_bridge.csv")
    headers = [
        "Seed", "Branch", "Event", "Time", "PPO_Prob",
        "Buf_Size", "Buf_Slack_Min", "Buf_Slack_Avg", "Buf_Slack_Q25",
        "Buf_Urgent", "Buf_Normal",
        "WIP_Bf", "WIP_Af", "MK_Bf", "MK_Af",
        "Conf_TD", "WIP_TD_Bf", "WIP_TD_Af", "Total_TD_Bf", "Total_TD_Af",
        "WIP_Slack_Min_Bf", "WIP_Slack_Min_Af", "WIP_Slack_Avg_Bf", "WIP_Slack_Avg_Af", "WIP_Slack_Q25_Bf", "WIP_Slack_Q25_Af",
        "Mch_Load_Avg_Bf", "Mch_Load_Avg_Af", "Mch_Load_Std_Bf", "Mch_Load_Std_Af", "Mch_Load_Span_Bf", "Mch_Load_Span_Af"
    ]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows_data:
            writer.writerow(r)
            
    print(f"\nSuccess! Branching analysis completed. Result saved to: {out_csv}")

if __name__ == "__main__":
    main()
