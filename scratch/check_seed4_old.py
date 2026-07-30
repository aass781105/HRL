import os
import sys
import numpy as np

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define custom list patch
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

from hrl_orchestrator import GlobalTimelineOrchestrator
original_init = GlobalTimelineOrchestrator.__init__
collected_jobs = []
def patched_init(self_orch, *args, **kwargs):
    original_init(self_orch, *args, **kwargs)
    self_orch.buffer = PatchedList(collected_jobs.extend, self_orch.buffer)
GlobalTimelineOrchestrator.__init__ = patched_init

def main():
    global collected_jobs
    from params import parser
    import yaml
    
    # 1. Simulate seed 4 under OLD configuration (before alignment)
    collected_jobs = []
    parser.set_defaults(
        init_jobs=30,
        hl_burst_due_date_scale_alpha=0.0,
        interarrival_uniform_low=40,
        interarrival_uniform_high=85,
        event_seed=4,
        fast_mode=True
    )
    cfg = parser.parse_args(args=[])
    
    from hl_gate_env import HLGateEnv
    env = HLGateEnv(
        n_machines=cfg.n_m,
        heuristic=cfg.scheduler_type,
        interarrival_mean=cfg.interarrival_mean,
        burst_K=cfg.burst_size,
        event_horizon=int(cfg.event_horizon),
        init_jobs=30,
    )
    env.reset(seed=4)
    done = False
    while not done:
        env.step(1)
    old_td = env.orch.get_final_kpi_stats(env.all_job_due_dates)["tardiness"]
    
    print(f"Seed 4 Old Config Tardiness: {old_td:.4f}")

if __name__ == "__main__":
    main()
