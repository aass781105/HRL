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
    # Wrap the buffer in our PatchedList
    self_orch.buffer = PatchedList(collected_jobs.extend, self_orch.buffer)

GlobalTimelineOrchestrator.__init__ = patched_init

def collect_seed_physics(config_name):
    global collected_jobs
    collected_jobs = [] # Clear for new run
    
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
    
    # Trigger simulation build and initial jobs loading
    env.reset(seed=cfg.event_seed)
    
    # Step through the entire event horizon to trigger all arrivals
    done = False
    while not done:
        _, _, done, _, _ = env.step(0)
        
    # Deduplicate captured jobs by job_id (just in case they are re-added during rescheduling)
    unique_jobs = {}
    for job in collected_jobs:
        unique_jobs[job.job_id] = job
        
    all_jobs = list(unique_jobs.values())
    
    total_jobs = len(all_jobs)
    total_ops = sum(len(j.operations) for j in all_jobs)
    
    urgent_jobs = [j for j in all_jobs if j.meta.get("is_urgent", False)]
    urgent_ratio = len(urgent_jobs) / total_jobs if total_jobs > 0 else 0
    
    # Collect Processing Times (PT) from operations
    all_pts = []
    for j in all_jobs:
        for op in j.operations:
            v = np.array(op.time_row)
            # Gather valid processing times (greater than 0)
            valid_pts = v[v > 0]
            if valid_pts.size > 0:
                all_pts.extend(list(valid_pts))
                
    mean_pt = np.mean(all_pts) if all_pts else 0
    
    # Collect due date factors (k)
    due_ks = [j.meta.get("due_date_k", 0.0) for j in all_jobs]
    mean_k = np.mean(due_ks) if due_ks else 0
    min_k = np.min(due_ks) if due_ks else 0
    
    # Analyze arrival time intervals
    arrival_times = [j.meta.get("t_arrive", 0.0) for j in all_jobs]
    arrival_intervals = np.diff(sorted(arrival_times)) if len(arrival_times) > 1 else [0]
    mean_arrival_interval = np.mean(arrival_intervals)
    
    return {
        "Total Arrived Jobs": total_jobs,
        "Total Operations Count": total_ops,
        "Urgent Jobs Ratio": urgent_ratio,
        "Average Operation PT": mean_pt,
        "Average Due Date k": mean_k,
        "Minimum Due Date k": min_k,
        "Mean Arrival Interval": mean_arrival_interval
    }

def main():
    s1_stats = collect_seed_physics("eval_baseline_seed1_greedy_cadence1_1run.yml")
    s10_stats = collect_seed_physics("eval_baseline_seed10_greedy_cadence1_1run.yml")
    
    print("\n--- Physical Comparison Result ---")
    print(f"{'Metric':<30} | {'Seed 1':<12} | {'Seed 10':<12}")
    print("-" * 60)
    for key in s1_stats.keys():
        val1 = s1_stats[key]
        val2 = s10_stats[key]
        if isinstance(val1, float):
            print(f"{key:<30} | {val1:<12.4f} | {val2:<12.4f}")
        else:
            print(f"{key:<30} | {val1:<12} | {val2:<12}")

if __name__ == "__main__":
    main()
