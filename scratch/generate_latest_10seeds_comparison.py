import os
import re
import csv
import math

global_plots_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\plots\global"
output_csv_path = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\eval_baseline_10seeds_latest_comparison.csv"

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

def main():
    if not os.path.exists(global_plots_dir):
        print(f"Directory not found: {global_plots_dir}")
        return
        
    folders = os.listdir(global_plots_dir)
    matched_folders = []
    
    # We want to match folders with timestamp and seed number, e.g. YYYYMMDD_HHMMSS_*_seed00X or _seedX
    # Pattern: 2026062*_*seed*
    # We extract the seed number and the timestamp
    for f in folders:
        if not os.path.isdir(os.path.join(global_plots_dir, f)):
            continue
            
        # Match pattern: 2026062* (specifically checking for recent runs)
        # Check if "seed" is in the folder name
        m = re.search(r"^(2026062\d_\d{6})_.*_seed0*(\d+)$", f)
        if m:
            timestamp = m.group(1)
            seed_num = int(m.group(2))
            
            # Make sure it contains sample_runs_summary.csv and it has non-zero size
            summary_path = os.path.join(global_plots_dir, f, "sample_runs_summary.csv")
            if os.path.exists(summary_path) and os.path.getsize(summary_path) > 100:
                matched_folders.append({
                    "folder": f,
                    "timestamp": timestamp,
                    "seed": seed_num
                })
                
    # Group by seed (1 to 10) and find the latest timestamp
    latest_runs = {} # seed_num -> entry
    for entry in matched_folders:
        s = entry["seed"]
        if 1 <= s <= 10:
            if s not in latest_runs or entry["timestamp"] > latest_runs[s]["timestamp"]:
                latest_runs[s] = entry
                
    print("Found latest folders for each seed:")
    for s in sorted(latest_runs.keys()):
        print(f"  Seed {s} -> {latest_runs[s]['folder']}")
        
    # Read metrics
    results = {} # seed -> metrics
    for s in range(1, 11):
        if s in latest_runs:
            csv_path = os.path.join(global_plots_dir, latest_runs[s]["folder"], "sample_runs_summary.csv")
            results[s] = read_mean_metrics(csv_path)
            results[s]["folder"] = latest_runs[s]["folder"]
        else:
            results[s] = None
            
    # Headers
    headers = ["Seed", "Folder_Name", "Makespan", "Tardiness", "Objective", "Releases", "Elapsed_Time_Sec"]
    
    rows_output = []
    # Collect values for mean/std
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
            row["Folder_Name"] = "MISSING"
            row["Makespan"] = ""
            row["Tardiness"] = ""
            row["Objective"] = ""
            row["Releases"] = ""
            row["Elapsed_Time_Sec"] = ""
        rows_output.append(row)
        
    # Mean row
    mean_row = {"Seed": "mean", "Folder_Name": "-"}
    for h in col_vals.keys():
        mean_row[h] = get_mean(col_vals[h]) if col_vals[h] else ""
    rows_output.append(mean_row)
    
    # Std row
    std_row = {"Seed": "std", "Folder_Name": "-"}
    for h in col_vals.keys():
        std_row[h] = get_std(col_vals[h]) if col_vals[h] else ""
    rows_output.append(std_row)
    
    # Write to output CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows_output:
            formatted_row = {}
            for k, v in r.items():
                if k in ("Seed", "Folder_Name"):
                    formatted_row[k] = v
                elif isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)
            
    print(f"\nSuccessfully generated comparison CSV at: {output_csv_path}")

if __name__ == "__main__":
    main()
