import sys
import os
import numpy as np

# Append the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]

from params import configs
configs.hl_td_signal_source = "baseline_gap_final"
configs.baseline_cadence = 1

from hl_gate_env import HLGateEnv

def analyze_seed_properties(env, seed):
    obs, _ = env.reset(seed=seed)
    
    # We will track the release time of each job
    release_times = {}
    
    # Initial jobs are released at t = 0.0
    for job in env.orch.buffer:
        release_times[job.job_id] = 0.0
        
    all_generated_jobs = list(env.orch.buffer)
    
    done = False
    while not done:
        t_now = env.t_now
        # Any jobs currently in buffer are about to be released at this step (t_now)
        for job in env.orch.buffer:
            if job.job_id not in release_times:
                release_times[job.job_id] = t_now
                
        # Reschedule/Release
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        
        # Track all unique generated jobs
        for job in (env.orch._last_jobs_snapshot or []):
            if job.job_id not in [j.job_id for j in all_generated_jobs]:
                all_generated_jobs.append(job)
                
    due_dates = env.all_job_due_dates
    finishes = env.orch._job_history_finishes
    
    total_avg_pt = 0.0
    urgent_count = 0
    subproblem_slacks = []
    arrival_times = []
    
    for job in all_generated_jobs:
        jid = job.job_id
        t_arr = job.meta.get("t_arrive", 0.0)
        due = due_dates.get(jid, 0.0)
        is_urg = job.meta.get("is_urgent", False)
        
        arrival_times.append(t_arr)
        
        job_avg_pt = sum(op.avg_proc_time for op in job.operations)
        job_min_pt = sum(min([x for x in op.time_row if x > 0]) if op.time_row else op.avg_proc_time for op in job.operations)
        
        total_avg_pt += job_avg_pt
        
        # Relative Slack at the moment of dynamic release to subproblem:
        # Due_Date - t_release - Job_Min_PT
        t_rel = release_times.get(jid, t_arr)  # Fallback to arrival time if never released
        slack_sub = due - t_rel - job_min_pt
        subproblem_slacks.append(slack_sub)
        
        if is_urg:
            urgent_count += 1
            
    # Tardiness and Makespan
    total_tardiness = 0.0
    for jid in due_dates.keys():
        due = due_dates[jid]
        finish = finishes.get(jid, 0.0)
        tardy = max(0.0, finish - due)
        total_tardiness += tardy
            
    kpi = env.orch.get_final_kpi_stats(due_dates)
    makespan = kpi["makespan"]
    
    arrival_times = sorted(arrival_times)
    simulation_duration = max(arrival_times) - min(arrival_times)
    num_machines = configs.n_m
    total_machine_capacity = num_machines * simulation_duration
    load_ratio = total_avg_pt / total_machine_capacity if total_machine_capacity > 0 else 0.0
    
    return {
        "seed": seed,
        "makespan": makespan,
        "tardiness": total_tardiness,
        "load_ratio": load_ratio * 100,
        "urgent_ratio": (urgent_count / len(all_generated_jobs) * 100) if all_generated_jobs else 0.0,
        "mean_slack": np.mean(subproblem_slacks) if subproblem_slacks else 0.0,
        "std_slack": np.std(subproblem_slacks) if subproblem_slacks else 0.0,
        "duration": simulation_duration
    }

def main():
    # Instantiate the env ONCE to avoid multiple slow PyTorch/CUDA initializations
    print("Initializing environment and loading weights...")
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    # Evaluate 14 seeds total
    seeds = [
        # Comparison Seeds
        200042, 200046,
        # I7
        49, 100049, 200049, 300049,
        # I8
        50, 100050, 200050, 300050,
        # I9
        51, 100051, 200051, 300051
    ]
    
    results = []
    for seed in seeds:
        print(f"Analyzing Seed {seed}...", flush=True)
        res = analyze_seed_properties(env, seed)
        results.append(res)
        
    # Sort results by Tardiness from lowest to highest
    results = sorted(results, key=lambda x: x["tardiness"])
    
    print("\n================ Batch Structural Correlation Analysis ================")
    print(f"| Seed | Tardiness (Cad 1) | Makespan | Load Ratio | Urgent Job Ratio | Mean Slack (Subproblem) | Std Slack (Subproblem) |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['seed']} | {r['tardiness']:.2f} | {r['makespan']:.2f} | {r['load_ratio']:.1f}% | {r['urgent_ratio']:.1f}% | {r['mean_slack']:.2f} | {r['std_slack']:.2f} |")

if __name__ == "__main__":
    main()
