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

def analyze_seed_mch_compatibility(config_name):
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
    
    done = False
    while not done:
        _, _, done, _, _ = env.step(0)
        
    unique_jobs = {}
    for job in collected_jobs:
        unique_jobs[job.job_id] = job
    all_jobs = list(unique_jobs.values())
    
    # Analyze compatibility and processing times for each machine (0 to 4)
    m_ops_count = {m: 0 for m in range(cfg.n_m)}
    m_total_pt = {m: 0.0 for m in range(cfg.n_m)}
    
    for j in all_jobs:
        for op in j.operations:
            for m, pt_val in enumerate(op.time_row):
                if pt_val > 0:
                    m_ops_count[m] += 1
                    m_total_pt[m] += float(pt_val)
                    
    return m_ops_count, m_total_pt

def main():
    # 5 machines (n_m = 5)
    s1_ops, s1_pts = analyze_seed_mch_compatibility("eval_baseline_seed1_greedy_cadence1_1run.yml")
    s10_ops, s10_pts = analyze_seed_mch_compatibility("eval_baseline_seed10_greedy_cadence1_1run.yml")
    
    print("\n--- Machine Loading Compatibility Analysis ---")
    print(f"{'Machine':<8} | {'Seed 1 Ops':<12} | {'Seed 10 Ops':<12} | {'Seed 1 Total PT':<15} | {'Seed 10 Total PT':<15}")
    print("-" * 75)
    for m in range(5):
        print(f"Mch {m:<4} | {s1_ops[m]:<12} | {s10_ops[m]:<12} | {s1_pts[m]:<15.2f} | {s10_pts[m]:<15.2f}")
        
    # Standard deviation of machine total PTs to check imbalance
    s1_pt_list = list(s1_pts.values())
    s10_pt_list = list(s10_pts.values())
    print("\nImbalance Metrics:")
    print(f"Seed 1 Total PT Std:   {np.std(s1_pt_list):.2f} (Max-Min Span: {np.max(s1_pt_list) - np.min(s1_pt_list):.2f})")
    print(f"Seed 10 Total PT Std:  {np.std(s10_pt_list):.2f} (Max-Min Span: {np.max(s10_pt_list) - np.min(s10_pt_list):.2f})")

if __name__ == "__main__":
    main()
