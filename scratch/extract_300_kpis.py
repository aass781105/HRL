import os
import pandas as pd

def extract_metrics(folder_path):
    summary_file = os.path.join(folder_path, "sample_runs_summary.csv")
    if not os.path.exists(summary_file):
        return None, None, None
    try:
        df = pd.read_csv(summary_file)
        mean_row = df[df["run"] == "mean"]
        if mean_row.empty:
            mean_row = df.iloc[0:1]
        mk = float(mean_row["makespan"].values[0])
        td = float(mean_row["total_tardiness"].values[0])
        rel = float(mean_row["release_count"].values[0])
        return mk, td, rel
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return None, None, None

def main():
    project_root = os.getcwd()
    # project_root is: C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN
    # We go up two levels to reach: C:\Users\123\Desktop\李信翰\碩一
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    
    # We will search for all subdirectories in PPObaseline300 and slack0瓶頸300
    folders = {
        "PPObaseline300": os.path.join(meeting_dir, "PPObaseline300"),
        "slack0瓶頸300": os.path.join(meeting_dir, "slack0瓶頸300")
    }
    
    for name, path in folders.items():
        print(f"\n--- Results for {name} ---")
        if not os.path.exists(path):
            print(f"Directory {path} does not exist!")
            continue
            
        subdirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        print("Seed | Makespan | Tardiness | Release Count")
        print("-" * 50)
        
        for sd in subdirs:
            # Extract seed from folder name (e.g., ..._seed001 -> Seed 1)
            try:
                parts = sd.split("_")
                seed_str = parts[-1].replace("seed", "")
                seed = int(seed_str)
            except:
                continue
                
            mk, td, rel = extract_metrics(os.path.join(path, sd))
            if td is not None:
                print(f"{seed:4d} | {mk:8.2f} | {td:9.2f} | {rel:12.1f}")
            else:
                print(f"{sd} | Data missing")

if __name__ == "__main__":
    main()
