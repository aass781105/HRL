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

def analyze_global_properties(seed):
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    # We will step through the environment and intercept ALL generated jobs
    env.reset(seed=seed)
    
    # We can collect jobs by hooking into the event generation loop
    # Let's run a manual simulation loop
    done = False
    all_generated_jobs = []
    
    # Initial jobs
    initial_jobs = list(env.orch.buffer)
    all_generated_jobs.extend(initial_jobs)
    
    while not done:
        # Before step, we check the buffer and any new arrivals
        # To get new arrivals at this step, we can run step and see what arrived
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        
        # After reschedule, we can check env.orch._last_jobs_snapshot
        # To make sure we capture all jobs, let's keep a set of all unique job specs
        for job in (env.orch._last_jobs_snapshot or []):
            if job.job_id not in [j.job_id for j in all_generated_jobs]:
                all_generated_jobs.append(job)
                
    # Now we have the complete list of all jobs that entered the system
    due_dates = env.all_job_due_dates
    
    total_avg_pt = 0.0
    total_min_pt = 0.0
    urgent_count = 0
    normal_count = 0
    urgent_ks = []
    normal_ks = []
    initial_slacks = []
    arrival_times = []
    
    for job in all_generated_jobs:
        jid = job.job_id
        t_arr = job.meta.get("t_arrive", 0.0)
        due = due_dates.get(jid, 0.0)
        is_urg = job.meta.get("is_urgent", False)
        k_val = job.meta.get("due_date_k", 1.0)
        
        arrival_times.append(t_arr)
        
        job_avg_pt = sum(op.avg_proc_time for op in job.operations)
        job_min_pt = sum(min([x for x in op.time_row if x > 0]) if op.time_row else op.avg_proc_time for op in job.operations)
        
        total_avg_pt += job_avg_pt
        total_min_pt += job_min_pt
        
        initial_slack = due - t_arr - job_min_pt
        initial_slacks.append(initial_slack)
        
        if is_urg:
            urgent_count += 1
            urgent_ks.append(k_val)
        else:
            normal_count += 1
            normal_ks.append(k_val)
            
    arrival_times = sorted(arrival_times)
    intervals = np.diff(arrival_times)
    
    simulation_duration = max(arrival_times) - min(arrival_times)
    num_machines = configs.n_m
    total_machine_capacity = num_machines * simulation_duration
    load_ratio = total_avg_pt / total_machine_capacity if total_machine_capacity > 0 else 0.0
    
    return {
        "total_jobs": len(all_generated_jobs),
        "total_ops": len(all_generated_jobs) * 5,
        "total_avg_pt": total_avg_pt,
        "urgent_count": urgent_count,
        "normal_count": normal_count,
        "urgent_ratio": urgent_count / len(all_generated_jobs) if all_generated_jobs else 0.0,
        "mean_urgent_k": np.mean(urgent_ks) if urgent_ks else 0.0,
        "mean_normal_k": np.mean(normal_ks) if normal_ks else 0.0,
        "mean_interval": np.mean(intervals) if len(intervals) > 0 else 0.0,
        "load_ratio": load_ratio,
        "mean_initial_slack": np.mean(initial_slacks) if initial_slacks else 0.0,
        "min_initial_slack": np.min(initial_slacks) if initial_slacks else 0.0,
        "simulation_duration": simulation_duration
    }

def main():
    print("Extracting global features from Seed 200042...")
    s42 = analyze_global_properties(200042)
    
    print("Extracting global features from Seed 200046...")
    s46 = analyze_global_properties(200046)
    
    print("\n================ GLOBAL Structural Bottleneck Analysis ================")
    print(f"| Metric | Seed 200042 (Easy) | Seed 200046 (Very Hard) | Comparison |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| Total Jobs (Global) | {s42['total_jobs']} | {s46['total_jobs']} | - |")
    print(f"| Total Workload (Avg PT) | {s42['total_avg_pt']:.1f} | {s46['total_avg_pt']:.1f} | {s46['total_avg_pt'] - s42['total_avg_pt']:+.1f} |")
    print(f"| Simulation Duration | {s42['simulation_duration']:.1f} | {s46['simulation_duration']:.1f} | {s46['simulation_duration'] - s42['simulation_duration']:+.1f} |")
    print(f"| **Load Ratio (Workload/Capacity)** | **{s42['load_ratio']*100:.1f}%** | **{s46['load_ratio']*100:.1f}%** | **{(s46['load_ratio'] - s42['load_ratio'])*100:+.1f}%** |")
    print(f"| **Global Urgent Job Ratio** | **{s42['urgent_ratio']*100:.1f}%** (Count: {s42['urgent_count']}) | **{s46['urgent_ratio']*100:.1f}%** (Count: {s46['urgent_count']}) | **{(s46['urgent_ratio'] - s42['urgent_ratio'])*100:+.1f}%** |")
    print(f"| Mean Urgent k (Due date) | {s42['mean_urgent_k']:.3f} | {s46['mean_urgent_k']:.3f} | {s46['mean_urgent_k'] - s42['mean_urgent_k']:+.3f} |")
    print(f"| Mean Normal k (Due date) | {s42['mean_normal_k']:.3f} | {s46['mean_normal_k']:.3f} | {s46['mean_normal_k'] - s42['mean_normal_k']:+.3f} |")
    print(f"| Mean Inter-arrival Interval | {s42['mean_interval']:.2f} | {s46['mean_interval']:.2f} | {s46['mean_interval'] - s42['mean_interval']:+.2f} |")
    print(f"| **Mean Initial Slack** | **{s42['mean_initial_slack']:.2f}** | **{s46['mean_initial_slack']:.2f}** | **{s46['mean_initial_slack'] - s42['mean_initial_slack']:+.2f}** |")
    print(f"| **Min Initial Slack** | **{s42['min_initial_slack']:.2f}** | **{s46['min_initial_slack']:.2f}** | **{s46['min_initial_slack'] - s42['min_initial_slack']:+.2f}** |")

if __name__ == "__main__":
    main()
