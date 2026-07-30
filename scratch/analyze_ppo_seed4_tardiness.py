import os
import sys
import numpy as np
import torch

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set CLI args dynamically to load configs
from params import configs
from hrl_orchestrator import GlobalTimelineOrchestrator

# Custom list wrapper to capture all JobSpecs
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

def main():
    import yaml
    from params import parser
    
    config_path = os.path.join(project_root, "yaml_config", "eval_baseline_seed4_greedy_cadence1_1run.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
        
    file_cfg["scheduler_type"] = "PPO"
    configs.__dict__.update(file_cfg)
    parser.set_defaults(**file_cfg)
    cfg = parser.parse_args(args=["--config", config_path])
    
    # CRITICAL: Match model device to CUDA to resolve the mat1 device mismatch error!
    setattr(configs, "device", "cuda")
    
    from hl_gate_env import HLGateEnv
    
    env = HLGateEnv(
        n_machines=cfg.n_m,
        heuristic="PPO",
        interarrival_mean=cfg.interarrival_mean,
        burst_K=cfg.burst_size,
        event_horizon=int(cfg.event_horizon),
        init_jobs=int(getattr(cfg, "init_jobs", 0)),
    )
    
    env.reset(seed=4)
    
    # Simulate to completion (cad 1 = release all at each step)
    done = False
    while not done:
        env.step(1)
        
    # Deduplicate captured jobs
    unique_jobs = {}
    for job in collected_jobs:
        unique_jobs[job.job_id] = job
    all_jobs = list(unique_jobs.values())
    
    finishes = env.orch._job_history_finishes
    
    job_tardiness = []
    for job in all_jobs:
        jid = job.job_id
        due = env.all_job_due_dates.get(jid, 0.0)
        finish = finishes.get(jid, 0.0)
        td = max(0.0, finish - due)
        job_tardiness.append({
            "job_id": jid,
            "t_arrive": job.meta.get("t_arrive", 0.0),
            "due_date": due,
            "finish_time": finish,
            "tardiness": td,
            "is_urgent": job.meta.get("is_urgent", False),
            "due_k": job.meta.get("due_date_k", 0.0)
        })
        
    job_tardiness.sort(key=lambda x: x["tardiness"], reverse=True)
    
    print("\n================ Seed 4 PPO Simulation Diagnostic ================")
    print(f"Total Tardiness: {sum(x['tardiness'] for x in job_tardiness):.4f}")
    print(f"Total Jobs: {len(job_tardiness)} (Urgent Jobs: {sum(1 for x in job_tardiness if x['is_urgent'])})")
    
    print("\nTop 15 Most Delayed Jobs:")
    print(f"{'Job_ID':<8} | {'Type':<8} | {'Arrive':<10} | {'Due Date':<10} | {'Finish':<10} | {'Tardiness':<12} | {'k Factor':<8}")
    print("-" * 75)
    for jt in job_tardiness[:15]:
        jtype = "URGENT" if jt["is_urgent"] else "NORMAL"
        print(f"{jt['job_id']:<8} | {jtype:<8} | {jt['t_arrive']:<10.2f} | {jt['due_date']:<10.2f} | {jt['finish_time']:<10.2f} | {jt['tardiness']:<12.4f} | {jt['due_k']:<8.4f}")
        
    mch_durations = {m: 0.0 for m in range(cfg.n_m)}
    for r in env.orch._global_rows:
        m = int(r["machine"])
        mch_durations[m] += float(r["duration"])
        
    print("\nMachine Total Occupancy Durations:")
    for m in range(cfg.n_m):
        print(f"Machine {m}: {mch_durations[m]:.2f} mins")

if __name__ == "__main__":
    main()
