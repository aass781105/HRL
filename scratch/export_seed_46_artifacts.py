import sys
import os
import csv

# Append the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]

from params import configs
configs.hl_td_signal_source = "baseline_gap_final"
configs.baseline_cadence = 1

from hl_gate_env import HLGateEnv
from gantt import plot_global_gantt

def run_and_export(seed, name):
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    obs, _ = env.reset(seed=seed)
    done = False
    
    release_log = []
    step_idx = 0
    
    while not done:
        t_now = env.t_now
        # Jobs currently in buffer before step (release decision)
        buffer_jobs = [j.job_id for j in env.orch.buffer]
        
        # Under Cadence 1, we release at every decision point
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        
        # Jobs released in this step
        released_jobs = list(buffer_jobs)
        
        # Collect current WIP (jobs active on floor, not finished yet)
        all_jobs_snapshot = env.orch._last_jobs_snapshot or []
        finished_jobs = env.orch._job_history_finishes or {}
        wip_jobs = [j.job_id for j in all_jobs_snapshot if j.job_id not in finished_jobs]
        
        release_log.append({
            "Step": step_idx,
            "Time": f"{t_now:.2f}",
            "Buffer_Size_Before_Release": len(buffer_jobs),
            "Buffer_Job_IDs": str(buffer_jobs),
            "Released_Job_IDs": str(released_jobs),
            "WIP_Count_After_Release": len(wip_jobs),
            "WIP_Job_IDs": str(wip_jobs)
        })
        step_idx += 1
        
    # 1. Export Release Log to CSV
    csv_path = os.path.join(os.path.dirname(__file__), f"release_log_{name}.csv")
    headers = [
        "Step", "Time", "Buffer_Size_Before_Release", 
        "Buffer_Job_IDs", "Released_Job_IDs", "WIP_Count_After_Release", "WIP_Job_IDs"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in release_log:
            writer.writerow(r)
    print(f"Exported Release Log: {csv_path}")
    
    # 2. Export Gantt Chart to PNG
    gantt_path = os.path.join(os.path.dirname(__file__), f"gantt_{name}.png")
    plot_global_gantt(
        env.orch._global_rows, 
        gantt_path, 
        t_now=env.t_now, 
        title=f"Gantt Chart - {name} (Cadence 1)"
    )
    print(f"Exported Gantt Chart: {gantt_path}")

def main():
    scratch_dir = os.path.dirname(__file__)
    os.makedirs(scratch_dir, exist_ok=True)
    
    print("Running and exporting Seed 200046 (Very Hard)...")
    run_and_export(200046, "seed_200046")
    
    print("\nRunning and exporting Seed 200042 (Easy)...")
    run_and_export(200042, "seed_200042")
    
    print("\nAll exports completed successfully!")

if __name__ == "__main__":
    main()
