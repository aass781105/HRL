import os
import sys
import csv
import numpy as np

# Find project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force the config path
config_file_path = os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")
sys.argv = ["", "--config", config_file_path]

from params import configs
from dynamic_job_stream import generate_dynamic_job_stream

def analyze_seed_properties_precise(seed, scenario_name):
    # Temporarily set config parameters
    configs.hl_env_scenario = scenario_name
    configs.event_seed = seed
    
    # Generate the dynamic stream
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
    release_times = {}
    for job in all_jobs:
        event_id = job_event_id[job.job_id]
        release_event_id = event_id if event_id % 2 == 0 else event_id + 1
        release_event_id = min(release_event_id, len(events))
        release_times[job.job_id] = event_times[release_event_id]
        
    # Calculate structural metrics
    urgent_count = 0
    subproblem_slacks = []
    urgent_slacks = []
    normal_slacks = []
    
    compatibilities = []
    machine_expected_workload = np.zeros(M)
    
    for job in all_jobs:
        jid = job.job_id
        due = due_dates[jid]
        is_urg = job.meta.get("is_urgent", False)
        
        job_min_pt = sum(min([x for x in op.time_row if x > 0]) if op.time_row else op.avg_proc_time for op in job.operations)
        
        # Slack relative to dynamic release: Due_Date - t_release - Min_PT
        t_rel = release_times[jid]
        slack_sub = due - t_rel - job_min_pt
        subproblem_slacks.append(slack_sub)
        
        if is_urg:
            urgent_count += 1
            urgent_slacks.append(slack_sub)
        else:
            normal_slacks.append(slack_sub)
            
        # Workload & compatibility
        for op in job.operations:
            compat_mchs = [m_idx for m_idx, pt in enumerate(op.time_row) if pt > 0]
            compat_count = len(compat_mchs)
            compatibilities.append(compat_count)
            
            if compat_count > 0:
                for m_idx in compat_mchs:
                    pt_on_m = op.time_row[m_idx]
                    machine_expected_workload[m_idx] += pt_on_m / compat_count
            
    mean_compat = np.mean(compatibilities) if compatibilities else 0.0
    mch_load_std = np.std(machine_expected_workload)
    
    return {
        "urgent_count": urgent_count,
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
    output_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\換題差異"
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "structural_bottleneck_analysis_all_seeds.csv")
    
    combined_results = []
    
    envs = {
        "baseline": "baseline",
        "bottleneck": "bottleneck_order"
    }
    
    for display_name, scenario_name in envs.items():
        for seed in range(1, 11):
            print(f"Analyzing static specs: Scenario={display_name}, Seed={seed}...", flush=True)
            
            # Extract structural specs statically (no scheduling baseline run needed!)
            specs = analyze_seed_properties_precise(seed, scenario_name)
            
            combined_results.append({
                "Group": f"seed{seed}",
                "Env": display_name,
                "Seed": seed,
                "Makespan_Cadence1": "",      # Leave blank to bypass scheduling
                "Tardiness_Cadence1": "",     # Leave blank to bypass scheduling
                "Urgent_Count": specs["urgent_count"],
                "Mean_Slack": int(round(specs["mean_slack"])),
                "Std_Slack": int(round(specs["std_slack"])),
                "Urgent_Mean_Slack": int(round(specs["urgent_mean_slack"])),
                "Urgent_Std_Slack": int(round(specs["urgent_std_slack"])),
                "Normal_Mean_Slack": int(round(specs["normal_mean_slack"])),
                "Normal_Std_Slack": int(round(specs["normal_std_slack"])),
                "Machine_Load_Std": int(round(specs["mch_load_std"])),
                "Mean_Compatibility": round(specs["mean_compat"], 3)
            })
            
    # Write to CSV
    headers = [
        "Group", "Env", "Seed", "Makespan_Cadence1", "Tardiness_Cadence1",
        "Urgent_Count", "Mean_Slack", "Std_Slack", "Urgent_Mean_Slack", 
        "Urgent_Std_Slack", "Normal_Mean_Slack", "Normal_Std_Slack", 
        "Machine_Load_Std", "Mean_Compatibility"
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in combined_results:
            formatted_row = {}
            for k in headers:
                v = r[k]
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)
            
    print(f"Successfully generated static structural analysis CSV at: {output_csv_path}")

if __name__ == "__main__":
    main()
