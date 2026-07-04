import sys
import os
import csv
import shutil
import numpy as np

# 1. Find project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. Find the config file dynamically in the entire project root, bypassing crazy folder name encodings
config_file_path = None
for root, dirs, files in os.walk(project_root):
    # Skip large directories to keep it fast
    dirs[:] = [d for d in dirs if d not in ('.git', 'trained_network', 'plots', 'ppo_ckpt')]
    if "eval_baseline_seed1_greedy_cadence1_1run.yml" in files:
        config_file_path = os.path.join(root, "eval_baseline_seed1_greedy_cadence1_1run.yml")
        break

if not config_file_path:
    raise FileNotFoundError("Could not find eval_baseline_seed1_greedy_cadence1_1run.yml in the project workspace.")

# 3. Copy it to scratch with a clean, ASCII name
scratch_dir = os.path.join(project_root, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
temp_config_path = os.path.join(scratch_dir, "temp_config_no_chinese.yml")
shutil.copyfile(config_file_path, temp_config_path)

# 4. Set sys.argv using the temporary config path and import params
sys.argv = ["", "--config", temp_config_path]

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from params import configs
from dynamic_job_stream import generate_dynamic_job_stream

def analyze_seed_properties_precise(seed):
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
            
        # Mathematically precise workload & compatibility
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
    csv_path = os.path.join(project_root, "scratch", "structural_bottleneck_analysis_all_seeds.csv")
    
    print(f"Reading user CSV from: {csv_path}")
    
    pre_stats = []
    # Seed mapping to ensure 200ep naming matches the correct Seed index:
    # 42, 100042, 200042, 300042 -> ep1~20
    # 43, 100043, 200043, 300043 -> ep21~40
    # ...
    # 51, 100051, 200051, 300051 -> ep181~200
    def seed_to_group(s):
        base_seed = s % 1000
        idx = base_seed - 42
        return f"ep{idx * 20 + 1}~{(idx + 1) * 20}"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row["Seed"])
            pre_stats.append({
                "Group": seed_to_group(seed),
                "Env": row["Env"],
                "Seed": seed,
                "Makespan_Cadence1": row["Makespan_Cadence1"],
                "Tardiness_Cadence1": row["Tardiness_Cadence1"]
            })
            
    # Process all 40 seeds
    combined_results = []
    for item in pre_stats:
        seed = item["Seed"]
        print(f"Processing Seed {seed}...", flush=True)
        specs = analyze_seed_properties_precise(seed)
        
        combined_results.append({
            "Group": item["Group"], # Mapped Group (ep1~20, ep21~40, etc.)
            "Env": item["Env"],
            "Seed": seed,
            "Makespan_Cadence1": int(float(item["Makespan_Cadence1"])),
            "Tardiness_Cadence1": int(float(item["Tardiness_Cadence1"])),
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
        
    # Write back to the CSV with the exact same columns and appending the new one at the end
    headers = [
        "Group", "Env", "Seed", "Makespan_Cadence1", "Tardiness_Cadence1",
        "Urgent_Count", "Mean_Slack", "Std_Slack", "Urgent_Mean_Slack", 
        "Urgent_Std_Slack", "Normal_Mean_Slack", "Normal_Std_Slack", 
        "Machine_Load_Std", "Mean_Compatibility"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in combined_results:
            writer.writerow(r)
            
    # Clean up temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
            
    print(f"Successfully updated CSV with ep group names: {csv_path}")
    
    # Sort by Tardiness for printing
    sorted_results = sorted(combined_results, key=lambda x: x["Tardiness_Cadence1"])
    
    print("\n================ UPDATED STRUCTURAL ANALYSIS (Sorted by Tardiness) ================")
    print(f"| Group | Env | Seed | Makespan_Cadence1 | Tardiness_Cadence1 | Urgent_Count | Mean_Slack | Std_Slack | Urgent_Mean_Slack | Urgent_Std_Slack | Normal_Mean_Slack | Normal_Std_Slack | Machine_Load_Std | Mean_Compatibility |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in sorted_results:
        print(f"| {r['Group']} | {r['Env']} | {r['Seed']} | {r['Makespan_Cadence1']} | {r['Tardiness_Cadence1']} | {r['Urgent_Count']} | {r['Mean_Slack']} | {r['Std_Slack']} | {r['Urgent_Mean_Slack']} | {r['Urgent_Std_Slack']} | {r['Normal_Mean_Slack']} | {r['Normal_Std_Slack']} | {r['Machine_Load_Std']} | {r['Mean_Compatibility']:.3f} |")

if __name__ == "__main__":
    main()
