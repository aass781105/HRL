import os
import csv

global_plots_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\plots\global"

def classify_folder(folder_path):
    jobs_csv = os.path.join(folder_path, "odprog_env_jobs.csv")
    if not os.path.exists(jobs_csv):
        return "unknown"
    
    # Read arrive times
    arrive_times = []
    try:
        with open(jobs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                arrive_times.append(float(row["arrive_time"]))
    except Exception as e:
        return f"error: {str(e)}"
        
    n_init = sum(1 for t in arrive_times if t == 0.0)
    
    # Check bursts for t > 0
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
    else:
        return f"other (init={n_init}, bursts={n_bursts})"

def main():
    folders = sorted(os.listdir(global_plots_dir))
    for f in folders:
        path = os.path.join(global_plots_dir, f)
        if os.path.isdir(path) and f.startswith("2026"):
            scenario = classify_folder(path)
            print(f"{f} -> {scenario}")

if __name__ == "__main__":
    main()
