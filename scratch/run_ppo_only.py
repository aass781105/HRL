import os
import sys
import numpy as np

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set CLI args dynamically to load configs
from params import configs

def run_ppo_simulation():
    import yaml
    from params import parser
    
    config_path = os.path.join(project_root, "yaml_config", "eval_baseline_seed4_greedy_cadence1_1run.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
        
    file_cfg["scheduler_type"] = "PPO"
    configs.__dict__.update(file_cfg)
    parser.set_defaults(**file_cfg)
    cfg = parser.parse_args(args=["--config", config_path])
    
    setattr(configs, "device", "cpu")
    
    from hl_gate_env import HLGateEnv
    
    print("Initializing HLGateEnv with PPO...", flush=True)
    env = HLGateEnv(
        n_machines=cfg.n_m,
        heuristic="PPO",
        interarrival_mean=cfg.interarrival_mean,
        burst_K=cfg.burst_size,
        event_horizon=int(cfg.event_horizon),
        init_jobs=int(getattr(cfg, "init_jobs", 0)),
    )
    
    print("Resetting environment (loading PPO weights)...", flush=True)
    env.reset(seed=4)
    print("Environment reset successful!\n", flush=True)
    
    print(f"{'Event_ID':<8} | {'Time':<10} | {'WIP Jobs':<10} | {'Buf Jobs':<10} | {'Est Tardiness':<15}", flush=True)
    print("-" * 65, flush=True)
    
    done = False
    while not done:
        # Step the environment (Action 1 = Release in cad 1)
        _, _, done, _, info = env.step(1)
        
        # Calculate current WIP and Buffer status
        # WIP jobs are jobs currently in the orchestrator schedule that are not finished
        finished_jobs = env.orch._job_history_finishes
        total_released = env.release_count
        # Approximate active jobs in floor
        wip_count = len(env.orch._job_history_finishes) - sum(1 for fid, fend in env.orch._job_history_finishes.items() if fend <= env.t_now)
        buf_count = len(env.orch.buffer)
        
        est_td = env.orch.get_total_tardiness_estimate(env.all_job_due_dates)
        
        print(f"Event {env.events_done:<3} | {env.t_now:<10.2f} | {wip_count:<10} | {buf_count:<10} | {est_td:<15.4f}", flush=True)
        
    final_stats = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    print("\n--- Final Simulation KPIs ---", flush=True)
    print(f"Final Makespan:  {final_stats['makespan']:.4f}", flush=True)
    print(f"Final Tardiness: {final_stats['tardiness']:.4f}", flush=True)

def main():
    try:
        run_ppo_simulation()
    except Exception as e:
        print("\n[ERROR] Simulation failed:", e, flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
