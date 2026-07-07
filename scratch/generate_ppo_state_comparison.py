import os
import re
import csv
import math
import sys

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
    if len(sys.argv) > 1:
        target_dir = sys.argv[1].strip('"').strip("'")
    else:
        target_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPO瓶頸"
        
    output_csv_path = os.path.join(target_dir, "eval_baseline_10seeds_latest_comparison.csv")
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return
        
    folders = os.listdir(target_dir)
    matched_folders = []
    
    for f in folders:
        folder_path = os.path.join(target_dir, f)
        if not os.path.isdir(folder_path):
            continue
            
        m = re.search(r"^(2026\d{4}_\d{6})_.*_seed0*(\d+)$", f)
        if m:
            timestamp = m.group(1)
            seed_num = int(m.group(2))
            
            summary_path = os.path.join(folder_path, "sample_runs_summary.csv")
            if os.path.exists(summary_path) and os.path.getsize(summary_path) > 100:
                matched_folders.append({
                    "folder": f,
                    "timestamp": timestamp,
                    "seed": seed_num,
                    "path": summary_path
                })
                
    # Group by seed and get latest
    latest_runs = {}
    for entry in matched_folders:
        s = entry["seed"]
        if 1 <= s <= 10:
            if s not in latest_runs or entry["timestamp"] > latest_runs[s]["timestamp"]:
                latest_runs[s] = entry
                
    print(f"Scanning target dir: {target_dir}")
    print("Found seed folders:")
    for s in sorted(latest_runs.keys()):
        print(f"  Seed {s} -> {latest_runs[s]['folder']}")
        
    results = {}
    for s in range(1, 11):
        if s in latest_runs:
            results[s] = read_mean_metrics(latest_runs[s]["path"])
            results[s]["folder"] = latest_runs[s]["folder"]
        else:
            results[s] = None
            
    headers = ["Seed", "Folder_Name", "Makespan", "Tardiness", "Objective", "Releases", "Tardiness / Releases", "Elapsed_Time_Sec"]
    
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
            
            t_int = int(round(res["tardiness"]))
            r_int = int(round(res["release_count"]))
            row["Tardiness / Releases"] = f"{t_int} / {r_int}"
            
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
            row["Tardiness / Releases"] = ""
            row["Elapsed_Time_Sec"] = ""
        rows_output.append(row)
        
    # Mean row
    mean_row = {"Seed": "mean", "Folder_Name": "-"}
    for h in col_vals.keys():
        mean_row[h] = get_mean(col_vals[h]) if col_vals[h] else ""
    if col_vals["Tardiness"] and col_vals["Releases"]:
        m_t = int(round(get_mean(col_vals["Tardiness"])))
        m_r = get_mean(col_vals["Releases"])
        mean_row["Tardiness / Releases"] = f"{m_t} / {m_r:.1f}"
    else:
        mean_row["Tardiness / Releases"] = ""
    rows_output.append(mean_row)
    
    # Std row
    std_row = {"Seed": "std", "Folder_Name": "-"}
    for h in col_vals.keys():
        std_row[h] = get_std(col_vals[h]) if col_vals[h] else ""
    if col_vals["Tardiness"] and col_vals["Releases"]:
        s_t = int(round(get_std(col_vals["Tardiness"])))
        s_r = get_std(col_vals["Releases"])
        std_row["Tardiness / Releases"] = f"{s_t} / {s_r:.4f}"
    else:
        std_row["Tardiness / Releases"] = ""
    rows_output.append(std_row)
    
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows_output:
            formatted_row = {}
            for k, v in r.items():
                if k in ("Seed", "Folder_Name", "Tardiness / Releases"):
                    formatted_row[k] = v
                elif isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)
            
    print(f"Generated comparison CSV at: {output_csv_path}")

if __name__ == "__main__":
    main()
