import os
import sys
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.argv = ["", "--config", "yaml_config/eval_baseline_seed1_greedy_cadence1_1run.yml"]

from params import configs
from hl_gate_env import HLGateEnv

def calculate_tardiness_components_fixed(env):
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

def main():
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    print("Running PPO on Seed 1 step-by-step with STRICT confirmed TD logic...")
    configs.hl_gate_policy = "ppo"
    
    import torch
    from model.hl_gate_state import get_hl_gate_state_dim, HL_LL_BUFFER_EMBED_DIM
    from model.hl_ppo_gate_model import HLPPOGateNet
    from common_utils import resolve_high_level_weight_path
    
    gate_device = torch.device("cpu")
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
    
    obs, info = env.reset(seed=1, options={"needs_baseline": False})
    done = False
    
    confirmed_history = []
    
    while not done:
        t_now = env.t_now
        confirmed_td, wip_td = calculate_tardiness_components_fixed(env)
        confirmed_history.append((env.events_done, confirmed_td))
        
        with torch.no_grad():
            logits, _ = ppo_model(torch.from_numpy(obs).float().unsqueeze(0).to(gate_device))
            action = int(torch.argmax(logits, dim=1).item())
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    print("\nConfirmed_TD_Before at decision steps:")
    monotonic = True
    for idx in range(len(confirmed_history)):
        ev, val = confirmed_history[idx]
        if idx > 0 and val < confirmed_history[idx-1][1]:
            print(f"  Event {ev:3d}: {val:8.2f}  <--- [DECREASED]")
            monotonic = False
        else:
            print(f"  Event {ev:3d}: {val:8.2f}")
            
    if monotonic:
        print("\nSUCCESS: Confirmed_TD_Before is now strictly monotonically increasing!")
    else:
        print("\nFAILED: Decreases still found.")

if __name__ == "__main__":
    main()
