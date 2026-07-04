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

def analyze_structural_properties(seed):
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    # We only need to reset the env to generate the jobs, then we can inspect the generated specs
    env.reset(seed=seed)
    
    # Collect all jobs generated
    # The orchestrator stores them in buffer during simulation, but we can reconstruct them
    # from env.orch._last_jobs_snapshot at the end, or we can run the simulation and extract them
    done = False
    all_jobs = []
    
    # We step through the simulation to let all jobs arrive
    while not done:
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        
    # The orchestrator's _last_jobs_snapshot contains all JobSpec objects
    all_jobs = env.orch._last_jobs_snapshot
    due_dates = env.all_job_due_dates
    
    # 1. Workload calculation
    total_ops = 0
    total_avg_pt = 0.0
    total_min_pt = 0.0
    
    urgent_count = 0
    normal_count = 0
    urgent_ks = []
    normal_ks = []
    
    arrival_times = []
    initial_slacks = []
    
    for job in all_jobs:
        jid = job.job_id
        t_arr = job.meta.get("t_arrive", 0.0)
        due = due_dates.get(jid, 0.0)
        is_urg = job.meta.get("is_urgent", False)
        k_val = job.meta.get("due_date_k", 1.0)
        
        arrival_times.append(t_arr)
        
        # Op processing times
        job_avg_pt = 0.0
        job_min_pt = 0.0
        for op in job.operations:
            job_avg_pt += op.avg_proc_time
            # Get min proc time from non-zero elements in time_row
            valid_pt = [x for x in op.time_row if x > 0] if op.time_row else [op.avg_proc_time]
            job_min_pt += min(valid_pt) if valid_pt else 0.0
            total_ops += 1
            
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
            
    # Arrival intervals
    arrival_times = sorted(arrival_times)
    intervals = np.diff(arrival_times)
    
    mean_interval = np.mean(intervals) if len(intervals) > 0 else 0.0
    std_interval = np.std(intervals) if len(intervals) > 0 else 0.0
    min_interval = np.min(intervals) if len(intervals) > 0 else 0.0
    max_interval = np.max(intervals) if len(intervals) > 0 else 0.0
    
    # Capacity calculation
    simulation_duration = max(arrival_times) - min(arrival_times)
    num_machines = configs.n_m
    total_machine_capacity = num_machines * simulation_duration
    
    # Load ratio: workload / capacity
    load_ratio = total_avg_pt / total_machine_capacity if total_machine_capacity > 0 else 0.0
    
    return {
        "total_jobs": len(all_jobs),
        "total_ops": total_ops,
        "total_avg_pt": total_avg_pt,
        "total_min_pt": total_min_pt,
        "avg_job_pt": total_avg_pt / len(all_jobs) if all_jobs else 0.0,
        "urgent_ratio": urgent_count / len(all_jobs) if all_jobs else 0.0,
        "mean_urgent_k": np.mean(urgent_ks) if urgent_ks else 0.0,
        "mean_normal_k": np.mean(normal_ks) if normal_ks else 0.0,
        "mean_interval": mean_interval,
        "std_interval": std_interval,
        "min_interval": min_interval,
        "max_interval": max_interval,
        "load_ratio": load_ratio,
        "mean_initial_slack": np.mean(initial_slacks) if initial_slacks else 0.0,
        "min_initial_slack": np.min(initial_slacks) if initial_slacks else 0.0,
        "simulation_duration": simulation_duration
    }

def main():
    print("Extracting features from Seed 200042...")
    s42 = analyze_structural_properties(200042)
    
    print("Extracting features from Seed 200046...")
    s46 = analyze_structural_properties(200046)
    
    print("\n================ Structural Bottleneck Analysis ================")
    print(f"| Metric | Seed 200042 (Easy) | Seed 200046 (Very Hard) | Comparison |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| Total Jobs | {s42['total_jobs']} | {s46['total_jobs']} | - |")
    print(f"| Total Operations | {s42['total_ops']} | {s46['total_ops']} | - |")
    print(f"| Total Avg PT (Workload) | {s42['total_avg_pt']:.1f} | {s46['total_avg_pt']:.1f} | {s46['total_avg_pt'] - s42['total_avg_pt']:+.1f} (+{ (s46['total_avg_pt'] - s42['total_avg_pt'])/s42['total_avg_pt']*100:.1f}%) |")
    print(f"| Avg Job PT | {s42['avg_job_pt']:.2f} | {s46['avg_job_pt']:.2f} | {s46['avg_job_pt'] - s42['avg_job_pt']:+.2f} |")
    print(f"| Simulation Duration | {s42['simulation_duration']:.1f} | {s46['simulation_duration']:.1f} | {s46['simulation_duration'] - s42['simulation_duration']:+.1f} |")
    print(f"| **Load Ratio (Workload/Capacity)** | **{s42['load_ratio']*100:.1f}%** | **{s46['load_ratio']*100:.1f}%** | **{s46['load_ratio'] - s42['load_ratio']:+.1f}%** |")
    print(f"| Urgent Job Ratio | {s42['urgent_ratio']*100:.1f}% | {s46['urgent_ratio']*100:.1f}% | {s46['urgent_ratio'] - s42['urgent_ratio']:+.1f}% |")
    print(f"| Mean Urgent k (Due date) | {s42['mean_urgent_k']:.3f} | {s46['mean_urgent_k']:.3f} | {s46['mean_urgent_k'] - s42['mean_urgent_k']:+.3f} |")
    print(f"| Mean Normal k (Due date) | {s42['mean_normal_k']:.3f} | {s46['mean_normal_k']:.3f} | {s46['mean_normal_k'] - s42['mean_normal_k']:+.3f} |")
    print(f"| Mean Inter-arrival Interval | {s42['mean_interval']:.2f} | {s46['mean_interval']:.2f} | {s46['mean_interval'] - s42['mean_interval']:+.2f} |")
    print(f"| Std Inter-arrival Interval | {s42['std_interval']:.2f} | {s46['std_interval']:.2f} | {s46['std_interval'] - s42['std_interval']:+.2f} |")
    print(f"| Min Inter-arrival Interval | {s42['min_interval']:.2f} | {s46['min_interval']:.2f} | {s46['min_interval'] - s42['min_interval']:+.2f} |")
    print(f"| **Mean Initial Slack** | **{s42['mean_initial_slack']:.2f}** | **{s46['mean_initial_slack']:.2f}** | **{s46['mean_initial_slack'] - s42['mean_initial_slack']:+.2f}** |")
    print(f"| **Min Initial Slack** | **{s42['min_initial_slack']:.2f}** | **{s46['min_initial_slack']:.2f}** | **{s46['min_initial_slack'] - s42['min_initial_slack']:+.2f}** |")

if __name__ == "__main__":
    main()
