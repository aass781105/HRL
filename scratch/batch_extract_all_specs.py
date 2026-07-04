import sys
import os
import csv
import numpy as np

# Append the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load config parameters
sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]
from params import configs
from dynamic_job_stream import generate_dynamic_job_stream

def analyze_seed_properties_instantly(seed):
    # Generate the dynamic stream instantly without scheduling
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
    M = configs.n_m
    
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
    urgent_slacks = []
    normal_slacks = []
    
    compatibilities = []
    machine_expected_workload = np.zeros(M)
    
    # Gather arrival times of all jobs to calculate inter-arrival statistics
    all_arrival_times = []
    for job in all_jobs:
        jid = job.job_id
        t_arr = job_arrival_time[jid]
        all_arrival_times.append(t_arr)
        
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
            urgent_slacks.append(slack_sub)
        else:
            normal_slacks.append(slack_sub)
            
        # Compatibility and expected workload distribution
        for op in job.operations:
            compat_mchs = [m_idx for m_idx, pt in enumerate(op.time_row) if pt > 0]
            compat_count = len(compat_mchs)
            compatibilities.append(compat_count)
            
            if compat_count > 0:
                share = op.avg_proc_time / compat_count
                for m_idx in compat_mchs:
                    machine_expected_workload[m_idx] += share
            
    # Inter-arrival stats
    all_arrival_times = sorted(all_arrival_times)
    intervals = np.diff(all_arrival_times)
    interarrival_mean = np.mean(intervals) if len(intervals) > 0 else 0.0
    interarrival_std = np.std(intervals) if len(intervals) > 0 else 0.0
            
    max_time = max(event_times.values())
    min_time = min(event_times.values())
    simulation_duration = max_time - min_time
    total_machine_capacity = M * simulation_duration
    load_ratio = total_avg_pt / total_machine_capacity if total_machine_capacity > 0 else 0.0
    
    mean_compat = np.mean(compatibilities) if compatibilities else 0.0
    mch_load_std = np.std(machine_expected_workload)
    
    return {
        "load_ratio": load_ratio * 100,
        "urgent_count": urgent_count,
        "interarrival_mean": interarrival_mean,
        "interarrival_std": interarrival_std,
        "mean_slack": np.mean(subproblem_slacks) if subproblem_slacks else 0.0,
        "std_slack": np.std(subproblem_slacks) if subproblem_slacks else 0.0,
        "urgent_mean_slack": np.mean(urgent_slacks) if urgent_slacks else 0.0,
        "urgent_std_slack": np.std(urgent_slacks) if urgent_slacks else 0.0,
        "normal_mean_slack": np.mean(normal_slacks) if normal_slacks else 0.0,
        "normal_std_slack": np.std(normal_slacks) if normal_slacks else 0.0,
        "mean_compat": mean_compat,
        "mch_load_std": mch_load_std
    }

def main():
    # Load Cadence 1 baseline results
    csv_in_path = os.path.join(project_root, "scratch", "eval_baseline_i0_i9_comparison.csv")
    csv_out_path = os.path.join(project_root, "scratch", "structural_bottleneck_analysis_all_seeds.csv")
    
    print(f"Reading Cadence 1 results from: {csv_in_path}")
    
    pre_stats = []
    with open(csv_in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre_stats.append({
                "Group": row["Group"],
                "Env": row["Env"],
                "Seed": int(row["Seed"]),
                "Makespan": float(row["Makespan_Cadence1"]),
                "Tardiness": float(row["Tardiness_Cadence1"])
            })
            
    # Process all 40 seeds
    combined_results = []
    for item in pre_stats:
        seed = item["Seed"]
        print(f"Extracting structural specs for Seed {seed}...", flush=True)
        specs = analyze_seed_properties_instantly(seed)
        
        combined_results.append({
            "Group": item["Group"],
            "Env": item["Env"],
            "Seed": seed,
            "Makespan_Cadence1": item["Makespan"],
            "Tardiness_Cadence1": item["Tardiness"],
            "Load_Ratio": specs["load_ratio"],
            "Urgent_Count": specs["urgent_count"],
            "Interarrival_Mean": specs["interarrival_mean"],
            "Interarrival_Std": specs["interarrival_std"],
            "Mean_Slack": specs["mean_slack"],
            "Std_Slack": specs["std_slack"],
            "Urgent_Mean_Slack": specs["urgent_mean_slack"],
            "Urgent_Std_Slack": specs["urgent_std_slack"],
            "Normal_Mean_Slack": specs["normal_mean_slack"],
            "Normal_Std_Slack": specs["normal_std_slack"],
            "Mean_Compatibility": specs["mean_compat"],
            "Machine_Load_Std": specs["mch_load_std"]
        })
        
    # Write to output CSV (keeping existing columns and appending new ones at the end)
    headers = [
        "Group", "Env", "Seed", "Makespan_Cadence1", "Tardiness_Cadence1",
        "Load_Ratio", "Urgent_Count", "Interarrival_Mean", "Interarrival_Std", 
        "Mean_Slack", "Std_Slack", "Urgent_Mean_Slack", "Urgent_Std_Slack", 
        "Normal_Mean_Slack", "Normal_Std_Slack", "Mean_Compatibility", "Machine_Load_Std"
    ]
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in combined_results:
            writer.writerow(r)
            
    print(f"Exported combined structural analysis to: {csv_out_path}")
    
    # Sort results by Tardiness from lowest to highest for report formatting
    sorted_results = sorted(combined_results, key=lambda x: x["Tardiness_Cadence1"])
    
    print("\n================ GLOBAL BATCH STRUCTURAL ANALYSIS (Sorted by Tardiness) ================")
    print(f"| Group | Env | Seed | Tardiness (Cad 1) | Makespan | Load Ratio | Urgent Job Count | Inter-arrival Avg | Inter-arrival Std | Mean Slack (Sub) | Std Slack (Sub) | Urgent Mean Slack | Urgent Std Slack | Normal Mean Slack | Normal Std Slack | Mean Compat | Machine Load Std |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in sorted_results:
        print(f"| {r['Group']} | {r['Env']} | {r['Seed']} | {r['Tardiness_Cadence1']:.2f} | {r['Makespan_Cadence1']:.2f} | {r['Load_Ratio']:.1f}% | {r['Urgent_Count']} | {r['Interarrival_Mean']:.2f} | {r['Interarrival_Std']:.2f} | {r['Mean_Slack']:.2f} | {r['Std_Slack']:.2f} | {r['Urgent_Mean_Slack']:.2f} | {r['Urgent_Std_Slack']:.2f} | {r['Normal_Mean_Slack']:.2f} | {r['Normal_Std_Slack']:.2f} | {r['Mean_Compatibility']:.3f} | {r['Machine_Load_Std']:.2f} |")

if __name__ == "__main__":
    main()
