import sys
import os
import numpy as np

# Find the project root directory (parent of scratch/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]

from params import configs
configs.hl_td_signal_source = "baseline_gap_final"
configs.baseline_cadence = 1

from hl_gate_env import HLGateEnv

def analyze_seed(seed):
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    # We will step through the environment using Cadence 1 to collect job completion stats
    obs, _ = env.reset(seed=seed)
    done = False
    
    # Manually run the Cadence 1 policy to completion
    while not done:
        act = 1  # Release and schedule at every step (equivalent to Cadence 1)
        obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        
    # Get stats from the orchestrator
    kpi = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    
    # We can inspect env.orch._global_rows (scheduled operations) and env.all_job_due_dates
    # Let's count how many jobs are urgent and normal
    # The job details can be reconstructed from env.orch._last_jobs_snapshot or env.orch._global_rows
    # Wait, the simplest way is to collect the actual finished jobs:
    finishes = env.orch._job_history_finishes
    due_dates = env.all_job_due_dates
    
    # We want to analyze:
    # 1. Total jobs
    # 2. Number of tardy jobs
    # 3. Average processing time per job (we can get this from job specs if we intercept them, or estimate from finishes)
    # Since we have the due dates and finishes, let's collect stats:
    job_ids = sorted(list(due_dates.keys()))
    total_jobs = len(job_ids)
    
    tardy_count = 0
    tardy_details = []
    total_tardiness = 0.0
    
    for jid in job_ids:
        due = due_dates[jid]
        finish = finishes.get(jid, 0.0)
        tardy = max(0.0, finish - due)
        total_tardiness += tardy
        if tardy > 0:
            tardy_count += 1
            tardy_details.append(tardy)
            
    avg_tardiness_of_tardy = sum(tardy_details) / len(tardy_details) if tardy_details else 0.0
    max_tardiness = max(tardy_details) if tardy_details else 0.0
    
    return {
        "makespan": kpi["makespan"],
        "tardiness_sum": total_tardiness,
        "total_jobs": total_jobs,
        "tardy_count": tardy_count,
        "avg_tardy": avg_tardiness_of_tardy,
        "max_tardy": max_tardiness,
        "tardy_ratio": tardy_count / total_jobs if total_jobs > 0 else 0.0
    }

def main():
    print("Analyzing Seed 200042 (Easy/Normal Seed)...")
    stats_42 = analyze_seed(200042)
    
    print("Analyzing Seed 200046 (Very Hard Seed)...")
    stats_46 = analyze_seed(200046)
    
    print("\n================ Diagnostic Comparison ================")
    print(f"| Metric | Seed 200042 | Seed 200046 | Difference |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| Total Jobs | {stats_42['total_jobs']} | {stats_46['total_jobs']} | {stats_46['total_jobs'] - stats_42['total_jobs']} |")
    print(f"| Makespan | {stats_42['makespan']:.2f} | {stats_46['makespan']:.2f} | {stats_46['makespan'] - stats_42['makespan']:.2f} |")
    print(f"| Total Tardiness | {stats_42['tardiness_sum']:.2f} | {stats_46['tardiness_sum']:.2f} | {stats_46['tardiness_sum'] - stats_42['tardiness_sum']:.2f} |")
    print(f"| Tardy Jobs Count | {stats_42['tardy_count']} | {stats_46['tardy_count']} | {stats_46['tardy_count'] - stats_42['tardy_count']} |")
    print(f"| Tardy Ratio | {stats_42['tardy_ratio']*100:.1f}% | {stats_46['tardy_ratio']*100:.1f}% | {(stats_46['tardy_ratio'] - stats_42['tardy_ratio'])*100:+.1f}% |")
    print(f"| Avg Tardy (Tardy only) | {stats_42['avg_tardy']:.2f} | {stats_46['avg_tardy']:.2f} | {stats_46['avg_tardy'] - stats_42['avg_tardy']:.2f} |")
    print(f"| Max Tardy | {stats_42['max_tardy']:.2f} | {stats_46['max_tardy']:.2f} | {stats_46['max_tardy'] - stats_42['max_tardy']:.2f} |")

if __name__ == "__main__":
    main()
