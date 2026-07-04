import sys
import os
import numpy as np

# Append the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]
from params import configs
from dynamic_job_stream import generate_dynamic_job_stream

def test_metrics(seed):
    stream = generate_dynamic_job_stream(
        configs,
        max_events=int(configs.event_horizon),
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        seed=seed
    )
    
    all_jobs = stream["all_jobs"]
    M = configs.n_m  # 5
    
    # 1. Compatibility
    compatibilities = []
    # 2. Machine expected workload
    machine_expected_workload = np.zeros(M)
    
    for job in all_jobs:
        for op in job.operations:
            # op.time_row is a list of processing times of length M
            # compat machines are those with pt > 0
            compat_mchs = [m_idx for m_idx, pt in enumerate(op.time_row) if pt > 0]
            compat_count = len(compat_mchs)
            compatibilities.append(compat_count)
            
            # Expected workload on each compatible machine: average_pt / compat_count
            avg_pt = op.avg_proc_time
            if compat_count > 0:
                share = avg_pt / compat_count
                for m_idx in compat_mchs:
                    machine_expected_workload[m_idx] += share
                    
    mean_compat = np.mean(compatibilities)
    mch_load_std = np.std(machine_expected_workload)
    
    print(f"Seed {seed} | Mean Compat: {mean_compat:.3f} | Machine Load Std: {mch_load_std:.2f} | Machine Workloads: {machine_expected_workload}")

print("Testing Seed 200042 (Easy)...")
test_metrics(200042)
print("Testing Seed 200046 (Very Hard)...")
test_metrics(200046)
