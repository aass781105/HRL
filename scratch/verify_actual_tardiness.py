import os
import sys
import numpy as np

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set CLI args dynamically to load configs
from params import configs
from hrl_orchestrator import GlobalTimelineOrchestrator

# Define custom list patch to intercept all jobs correctly
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

original_init = GlobalTimelineOrchestrator.__init__
collected_jobs = []
def patched_init(self_orch, *args, **kwargs):
    original_init(self_orch, *args, **kwargs)
    self_orch.buffer = PatchedList(collected_jobs.extend, self_orch.buffer)
GlobalTimelineOrchestrator.__init__ = patched_init

def run_actual_seed_tardiness(config_name):
    global collected_jobs
    collected_jobs = [] # Clear
    
    import yaml
    from params import parser
    
    config_path = os.path.join(project_root, "yaml_config", config_name)
    with open(config_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
        
    # CRITICAL: We must update the global configs object so hl_gate_env.py reads correct parameters!
    configs.__dict__.update(file_cfg)
    
    # Also update parser defaults
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
    
    # Run simulation with standard step logic to match real evaluation
    done = False
    while not done:
        # In cadence policy, step actions are checked by policy.
        # But we can just use env's standard run baseline logic or step(1) since cadence = 1
        _, _, done, _, _ = env.step(1)
        
    final_stats = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    return final_stats["tardiness"], final_stats["makespan"]

def main():
    print("\n--- Correct Simulation Results under YAML configs ---")
    print(f"{'Seed':<8} | {'Tardiness':<15} | {'Makespan':<15}")
    print("-" * 45)
    for i in [1, 4, 10]:
        td, mk = run_actual_seed_tardiness(f"eval_baseline_seed{i}_greedy_cadence1_1run.yml")
        print(f"Seed {i:<3} | {td:<15.4f} | {mk:<15.4f}")

if __name__ == "__main__":
    main()
