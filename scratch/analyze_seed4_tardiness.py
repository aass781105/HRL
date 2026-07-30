import os
import sys
import numpy as np

# Append project root
project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set CLI args dynamically to load configs
from params import configs

# Define a custom list class to intercept appends/extends
class PatchedList(list):
    def __init__(self, callback, initial_list=None):
        if initial_list:
            super().__init__(initial_list)
        else:
            super().__init__()
        self.callback = callback

    def extend(self, iterable):
        self.callback(list(iterable))
        super().extend(iterable)

    def append(self, item):
        self.callback([item])
        super().append(item)

# Globally patch GlobalTimelineOrchestrator to capture all incoming jobs
from hrl_orchestrator import GlobalTimelineOrchestrator
original_init = GlobalTimelineOrchestrator.__init__
collected_jobs = []

def patched_init(self_orch, *args, **kwargs):
    original_init(self_orch, *args, **kwargs)
    self_orch.buffer = PatchedList(collected_jobs.extend, self_orch.buffer)

GlobalTimelineOrchestrator.__init__ = patched_init

def run_seed_simulation_and_get_stats(config_name):
    global collected_jobs
    collected_jobs = [] # Clear
    
    from params import parser
    import yaml
    
    config_path = os.path.join(project_root, "yaml_config", config_name)
    with open(config_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
    
    parser.set_defaults(**file_cfg)
    cfg = parser.parse_args(args=["--config", config_path])
    
    from hl_gate_env import HLGateEnv
    
    env = HLGateEnv(
        n_machines=cfg.n_m,
        heuristic=cfg.scheduler_type,
        interarrival_mean=cfg.interarrival_mean,
        burst_K=cfg.burst_size,
        event_horizon=int(cfg.event_horizon),
        init_jobs=int(getattr(cfg, "init_jobs", 0)),
    )
    
    env.reset(seed=cfg.event_seed)
    
    # We will step using action 1 (release) to simulate cad 1 baseline
    done = False
    while not done:
        _, _, done, _, _ = env.step(1) # Action 1 = Release
        
    unique_jobs = {}
    for job in collected_jobs:
        unique_jobs[job.job_id] = job
    all_jobs = list(unique_jobs.values())
    
    total_jobs = len(all_jobs)
    urgent_jobs = [j for j in all_jobs if j.meta.get("is_urgent", False)]
    urgent_ratio = len(urgent_jobs) / total_jobs if total_jobs > 0 else 0
    
    # Processing times
    all_pts = []
    for j in all_jobs:
        for op in j.operations:
            v = np.array(op.time_row)
            valid_pts = v[v > 0]
            if valid_pts.size > 0:
                all_pts.extend(list(valid_pts))
    mean_pt = np.mean(all_pts) if all_pts else 0
    
    # Due dates
    due_ks = [j.meta.get("due_date_k", 0.0) for j in all_jobs]
    mean_k = np.mean(due_ks) if due_ks else 0
    min_k = np.min(due_ks) if due_ks else 0
    
    # Arrival spacing (check for burst tightness)
    arrival_times = sorted([j.meta.get("t_arrive", 0.0) for j in all_jobs])
    arrival_intervals = np.diff(arrival_times) if len(arrival_times) > 1 else [0]
    min_interval = np.min(arrival_intervals) if len(arrival_intervals) > 0 else 0
    mean_interval = np.mean(arrival_intervals) if len(arrival_intervals) > 0 else 0
    # Count intervals that are very tight (< 50 minutes)
    tight_arrivals = sum(1 for i in arrival_intervals if i < 50)
    
    # Final tardiness from env
    final_stats = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    
    return {
        "Tardiness": final_stats["tardiness"],
        "Makespan": final_stats["makespan"],
        "Urgent Jobs": len(urgent_jobs),
        "Urgent Ratio": urgent_ratio,
        "Mean PT": mean_pt,
        "Mean k": mean_k,
        "Min k": min_k,
        "Mean Interval": mean_interval,
        "Min Interval": min_interval,
        "Tight Intervals Count (<50)": tight_arrivals
    }

def main():
    s4 = run_seed_simulation_and_get_stats("eval_baseline_seed4_greedy_cadence1_1run.yml")
    s10 = run_seed_simulation_and_get_stats("eval_baseline_seed10_greedy_cadence1_1run.yml")
    
    print("\n--- Seed 4 vs Seed 10 Diagnostic ---")
    print(f"{'Metric':<30} | {'Seed 4 (High TD)':<18} | {'Seed 10 (Low TD)':<18}")
    print("-" * 75)
    for key in s4.keys():
        val4 = s4[key]
        val10 = s10[key]
        if isinstance(val4, float):
            print(f"{key:<30} | {val4:<18.4f} | {val10:<18.4f}")
        else:
            print(f"{key:<30} | {val4:<18} | {val10:<18}")

if __name__ == "__main__":
    main()
