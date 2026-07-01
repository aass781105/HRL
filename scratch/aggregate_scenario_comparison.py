import os
import re
import csv
import math

scenarios = {
    "多單": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\多單",
    "瓶頸": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\瓶頸",
    "普通": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\普通"
}

def get_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def get_std(lst):
    if len(lst) <= 1:
        return 0.0
    mean_val = get_mean(lst)
    variance = sum((x - mean_val) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)

def read_summary_metrics(summary_path):
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
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
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
    return None

def normalize_policy_name(name):
    clean = name.strip().lower().replace(" ", "")
    if clean in ("cad1", "cad 1"):
        return "cad1"
    elif clean in ("cad5", "cad 5"):
        return "cad5"
    elif clean in ("slack0", "slack 0"):
        return "slack0"
    elif clean == "ppo":
        return "PPO"
    return name

def main():
    policy_order = ["cad1", "cad5", "slack0", "PPO"]
    metrics_list = [
        ("Makespan", "makespan"),
        ("Tardiness", "tardiness"),
        ("Objective", "obj"),
        ("Releases", "release_count"),
        ("Elapsed_Time_Sec", "elapsed_time")
    ]
    
    for sc_name, sc_path in scenarios.items():
        print(f"\n=================== Processing Scenario: {sc_name} ===================")
        if not os.path.exists(sc_path):
            print(f"Scenario path not found: {sc_path}")
            continue
            
        # Find all directories in the scenario folder
        subdirs = [d for d in os.listdir(sc_path) if os.path.isdir(os.path.join(sc_path, d))]
        
        policy_data = {} # policy_name -> { metric_name -> [val_seed1, val_seed2, ... val_seed10] }
        
        for subdir in subdirs:
            pol_name = normalize_policy_name(subdir)
            if pol_name not in policy_order:
                # Skip non-policy directories like generated comparisons
                continue
                
            subdir_path = os.path.join(sc_path, subdir)
            seed_folders = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
            
            # Map folders by seed number
            seed_runs = {} # seed_num (1..10) -> folder_name
            for sf in seed_folders:
                m = re.search(r"seed0*(\d+)$", sf)
                if m:
                    seed_num = int(m.group(1))
                    if 1 <= seed_num <= 10:
                        seed_runs[seed_num] = sf
            
            # Extract metrics for seeds 1..10
            metrics_history = {m_name: [None]*10 for m_name, _ in metrics_list}
            for s in range(1, 11):
                if s in seed_runs:
                    summary_path = os.path.join(subdir_path, seed_runs[s], "sample_runs_summary.csv")
                    metrics = read_summary_metrics(summary_path)
                    if metrics:
                        for m_name, key in metrics_list:
                            metrics_history[m_name][s-1] = metrics[key]
                            
            policy_data[pol_name] = metrics_history
            print(f"  Policy: {pol_name} -> found runs for seeds: {sorted(list(seed_runs.keys()))}")
            
        # Build comparison table
        output_csv = os.path.join(sc_path, "eval_baseline_10seeds_latest_comparison.csv")
        headers = ["Policy", "Metric", "seed1", "seed2", "seed3", "seed4", "seed5", "seed6", "seed7", "seed8", "seed9", "seed10", "Mean", "Std"]
        
        rows = []
        for pol in policy_order:
            if pol not in policy_data:
                print(f"  Warning: Policy '{pol}' data not found in scenario '{sc_name}' subdirectories.")
                continue
                
            for m_name, _ in metrics_list:
                row = {
                    "Policy": pol,
                    "Metric": m_name
                }
                vals = policy_data[pol][m_name]
                # Filter out None values
                valid_vals = [v for v in vals if v is not None]
                
                for s in range(1, 11):
                    val = vals[s-1]
                    row[f"seed{s}"] = f"{val:.4f}" if val is not None else "N/A"
                    
                row["Mean"] = f"{get_mean(valid_vals):.4f}" if valid_vals else "N/A"
                row["Std"] = f"{get_std(valid_vals):.4f}" if len(valid_vals) > 1 else "N/A"
                rows.append(row)
                
        # Write comparison table to CSV
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
                
        print(f"Successfully generated scenario comparison table at: {output_csv}")

if __name__ == "__main__":
    main()
