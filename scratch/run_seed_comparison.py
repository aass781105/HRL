import os
import sys
import numpy as np

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set CLI args dynamically to load configs
from params import configs
from hrl_orchestrator import GlobalTimelineOrchestrator

# Custom list wrapper to capture jobs
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

def run_simulation(scheduler_type):
    global collected_jobs
    collected_jobs = []
    
    import yaml
    from params import parser
    
    config_path = os.path.join(project_root, "yaml_config", "eval_baseline_seed4_greedy_cadence1_1run.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
        
    # Set scheduler type
    file_cfg["scheduler_type"] = scheduler_type
    
    configs.__dict__.update(file_cfg)
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
    
    env.reset(seed=4)
    
    done = False
    while not done:
        # Cad 1 release step
        _, _, done, _, _ = env.step(1)
        
    final_stats = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    return final_stats["tardiness"], final_stats["makespan"]

def main():
    print("Simulating seed 4 under currently configured YAML file parameters...")
    try:
        # 1. Run using OR-Tools (heuristic)
        ort_td, ort_mk = run_simulation("OR-Tools")
        print(f"Scheduler: OR-Tools | Tardiness: {ort_td:.4f} | Makespan: {ort_mk:.4f}")
    except Exception as e:
        print("OR-Tools Run Failed:", e)
        
    try:
        # 2. Run using PPO (heuristic)
        ppo_td, ppo_mk = run_simulation("PPO")
        print(f"Scheduler: PPO      | Tardiness: {ppo_td:.4f} | Makespan: {ppo_mk:.4f}")
    except Exception as e:
        print("PPO Run Failed:", e)

if __name__ == "__main__":
    main()
