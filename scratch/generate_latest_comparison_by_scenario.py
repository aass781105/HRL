import os
import re
import csv
import math
import argparse

global_plots_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\plots\global"

def get_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def get_std(lst):
    if len(lst) <= 1:
        return 0.0
    mean_val = get_mean(lst)
    variance = sum((x - mean_val) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)

def read_mean_metrics(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["run"].strip() == "mean":
                return {
                    "makespan": float(row["makespan"]),
                    "tardiness": float(row["total_tardiness"]),
                    "obj": float(row["obj"]),
                    "release_count": float(row["release_count"]),
                    "elapsed_time": float(row["elapsed_time_sec"])
                }
    return None

def classify_folder(folder_path):
    jobs_csv = os.path.join(folder_path, "odprog_env_jobs.csv")
    if not os.path.exists(jobs_csv):
        return "unknown"
    
    arrive_times = []
    try:
        with open(jobs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                arrive_times.append(float(row["arrive_time"]))
    except Exception:
        return "error"
        
    n_init = sum(1 for t in arrive_times if t == 0.0)
    
    t_counts = {}
    for t in arrive_times:
        if t > 0:
            t_counts[t] = t_counts.get(t, 0) + 1
            
    n_bursts = sum(1 for t, count in t_counts.items() if count > 1)
    
    if n_init == 50:
        return "baseline"
    elif n_init == 30:
        if n_bursts > 0:
            return "burst_cluster"
        else:
            return "bottleneck_order"
    return "other"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=["baseline", "bottleneck_order", "burst_cluster"])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not os.path.exists(global_plots_dir):
        print(f"Directory not found: {global_plots_dir}")
        return

    folders = os.listdir(global_plots_dir)
    matched_folders = []
    
    for f in folders:
        path = os.path.join(global_plots_dir, f)
        if not os.path.isdir(path):
            continue
            
        m = re.search(r"^(2026\d{4}_\d{6})_.*_seed0*(\d+)$", f)
        if m:
            timestamp = m.group(1)
            seed_num = int(m.group(2))
            
            # Check classification
            scenario_type = classify_folder(path)
            if scenario_type != args.scenario:
                continue
                
            summary_path = os.path.join(path, "sample_runs_summary.csv")
            if os.path.exists(summary_path) and os.path.getsize(summary_path) > 100:
                matched_folders.append({
                    "folder": f,
                    "timestamp": timestamp,
                    "seed": seed_num
                })
                
    # Group by seed and find latest
    latest_runs = {}
    for entry in matched_folders:
        s = entry["seed"]
        if 1 <= s <= 10:
            if s not in latest_runs or entry["timestamp"] > latest_runs[s]["timestamp"]:
                latest_runs[s] = entry
                
    print(f"Scenario: {args.scenario}")
    print(f"Found latest folders for each seed:")
    for s in sorted(latest_runs.keys()):
        print(f"  Seed {s} -> {latest_runs[s]['folder']}")
        
    results = {}
    for s in range(1, 11):
        if s in latest_runs:
            csv_path = os.path.join(global_plots_dir, latest_runs[s]["folder"], "sample_runs_summary.csv")
            results[s] = read_mean_metrics(csv_path)
            results[s]["folder"] = latest_runs[s]["folder"]
        else:
            results[s] = None
            
    headers = ["Seed", "Folder_Name", "Makespan", "Tardiness", "Objective", "Releases", "Elapsed_Time_Sec"]
    
    rows_output = []
    col_vals = {h: [] for h in ["Makespan", "Tardiness", "Objective", "Releases", "Elapsed_Time_Sec"]}
    
    for s in range(1, 11):
        row = {"Seed": f"seed{s}"}
        res = results[s]
        if res:
            row["Folder_Name"] = res["folder"]
            row["Makespan"] = res["makespan"]
            row["Tardiness"] = res["tardiness"]
            row["Objective"] = res["obj"]
            row["Releases"] = res["release_count"]
            row["Elapsed_Time_Sec"] = res["elapsed_time"]
            
            col_vals["Makespan"].append(res["makespan"])
            col_vals["Tardiness"].append(res["tardiness"])
            col_vals["Objective"].append(res["obj"])
            col_vals["Releases"].append(res["release_count"])
            col_vals["Elapsed_Time_Sec"].append(res["elapsed_time"])
        else:
            row["Folder_Name"] = "N/A"
            row["Makespan"] = ""
            row["Tardiness"] = ""
            row["Objective"] = ""
            row["Releases"] = ""
            row["Elapsed_Time_Sec"] = ""
        rows_output.append(row)
        
    # Add Mean row
    mean_row = {"Seed": "mean", "Folder_Name": ""}
    for h in col_vals:
        mean_row[h] = get_mean(col_vals[h]) if col_vals[h] else ""
    rows_output.append(mean_row)
    
    # Add Std row
    std_row = {"Seed": "std", "Folder_Name": ""}
    for h in col_vals:
        std_row[h] = get_std(col_vals[h]) if len(col_vals[h]) > 1 else ""
    rows_output.append(std_row)
    
    # Write to CSV
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows_output:
            writer.writerow(r)
            
    print(f"Successfully generated comparison CSV at: {args.output}\n")

if __name__ == "__main__":
    main()
