import sys
import os
import numpy as np

# Append the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load configs
sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]
from params import configs
from dynamic_job_stream import generate_dynamic_job_stream

# Pre-calculated Cadence 1 Tardiness and Makespan from our previous runs
PRE_CALCULATED_STATS = {
    # Comparison Seeds
    200042: {"tardiness": 0.00, "makespan": 4881.00},
    200046: {"tardiness": 28481.90, "makespan": 6923.81},
    # I7
    200049: {"tardiness": 138.41, "makespan": 6413.93},
    100049: {"tardiness": 6111.23, "makespan": 6568.58},
    300049: {"tardiness": 1537.74, "makespan": 6515.21},
    49: {"tardiness": 11839.93, "makespan": 6435.31},
    # I8
    300050: {"tardiness": 859.50, "makespan": 6900.00},
    100050: {"tardiness": 1061.53, "makespan": 6519.45},
    50: {"tardiness": 2349.40, "makespan": 6271.99},
    200050: {"tardiness": 5903.56, "makespan": 6629.49},
    # I9
    100051: {"tardiness": 1406.40, "makespan": 6475.89},
    200051: {"tardiness": 1474.93, "makespan": 6581.93},
    300051: {"tardiness": 2218.33, "makespan": 6677.91},
    51: {"tardiness": 6091.16, "makespan": 6555.72}
}

def analyze_seed_properties_instantly(seed):
    # Generate the dynamic stream instantly
    stream = generate_dynamic_job_stream(
        configs,
        max_events=int(configs.event_horizon),
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        seed=seed
    )
    
    all_jobs = stream["all_jobs"]
    due_dates = stream["all_job_due_dates"]
    events = stream["events"]
    init_jobs = stream["init_jobs"]
    
    # Map event ID -> arrival time
    event_times = {0: 0.0}
    for event in events:
        event_times[event.event_id] = event.time
        
    # Map Job ID -> arrival time & event ID
    job_arrival_time = {}
    job_event_id = {}
    
    for job in init_jobs:
        job_arrival_time[job.job_id] = 0.0
        job_event_id[job.job_id] = 0
        
    for event in events:
        for job in event.jobs:
            job_arrival_time[job.job_id] = event.time
            job_event_id[job.job_id] = event.event_id
            
    # Calculate release times under Cadence 1
    # Decision interval = 2 events.
    # Jobs arriving at event ID are released at release_event_id = event_id if event_id is even, else event_id + 1
    release_times = {}
    for job in all_jobs:
        event_id = job_event_id[job.job_id]
        release_event_id = event_id if event_id % 2 == 0 else event_id + 1
        release_event_id = min(release_event_id, len(events))
        release_times[job.job_id] = event_times[release_event_id]
        
    # Calculate structural metrics
    total_avg_pt = 0.0
    urgent_count = 0
    subproblem_slacks = []
    
    for job in all_jobs:
        jid = job.job_id
        due = due_dates[jid]
        is_urg = job.meta.get("is_urgent", False)
        
        job_avg_pt = sum(op.avg_proc_time for op in job.operations)
        job_min_pt = sum(min([x for x in op.time_row if x > 0]) if op.time_row else op.avg_proc_time for op in job.operations)
        
        total_avg_pt += job_avg_pt
        
        # Slack relative to dynamic release: Due_Date - t_release - Min_PT
        t_rel = release_times[jid]
        slack_sub = due - t_rel - job_min_pt
        subproblem_slacks.append(slack_sub)
        
        if is_urg:
            urgent_count += 1
            
    max_time = max(event_times.values())
    min_time = min(event_times.values())
    simulation_duration = max_time - min_time
    num_machines = configs.n_m
    total_machine_capacity = num_machines * simulation_duration
    load_ratio = total_avg_pt / total_machine_capacity if total_machine_capacity > 0 else 0.0
    
    pre_stats = PRE_CALCULATED_STATS.get(seed, {"tardiness": 0.0, "makespan": 0.0})
    
    return {
        "seed": seed,
        "makespan": pre_stats["makespan"],
        "tardiness": pre_stats["tardiness"],
        "load_ratio": load_ratio * 100,
        "urgent_ratio": (urgent_count / len(all_jobs) * 100) if all_jobs else 0.0,
        "mean_slack": np.mean(subproblem_slacks) if subproblem_slacks else 0.0,
        "std_slack": np.std(subproblem_slacks) if subproblem_slacks else 0.0,
    }

def main():
    seeds = [
        200042, 200046,
        49, 100049, 200049, 300049,
        50, 100050, 200050, 300050,
        51, 100051, 200051, 300051
    ]
    
    results = []
    for seed in seeds:
        res = analyze_seed_properties_instantly(seed)
        results.append(res)
        
    # Sort by Tardiness
    results = sorted(results, key=lambda x: x["tardiness"])
    
    print("\n================ Batch Structural Correlation Analysis ================")
    print(f"| Seed | Tardiness (Cad 1) | Makespan | Load Ratio | Urgent Job Ratio | Mean Slack (Subproblem) | Std Slack (Subproblem) |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['seed']} | {r['tardiness']:.2f} | {r['makespan']:.2f} | {r['load_ratio']:.1f}% | {r['urgent_ratio']:.1f}% | {r['mean_slack']:.2f} | {r['std_slack']:.2f} |")

if __name__ == "__main__":
    main()
