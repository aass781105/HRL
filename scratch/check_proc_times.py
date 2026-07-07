import os
import pandas as pd
import numpy as np

def extract_tardiness(folder_path):
    summary_file = os.path.join(folder_path, "sample_runs_summary.csv")
    if not os.path.exists(summary_file):
        return None
    try:
        df = pd.read_csv(summary_file)
        mean_row = df[df["run"] == "mean"]
        if mean_row.empty:
            mean_row = df.iloc[0:1]
        return float(mean_row["total_tardiness"].values[0])
    except:
        return None

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    
    slack_path = os.path.join(meeting_dir, "slack0瓶頸300")
    
    if not os.path.exists(slack_path):
        print(f"Path does not exist: {slack_path}")
        return
        
    results = []
    
    for sd in sorted(os.listdir(slack_path)):
        if not os.path.isdir(os.path.join(slack_path, sd)): continue
        try:
            seed = int(sd.split("_")[-1].replace("seed", ""))
        except:
            continue
            
        td = extract_tardiness(os.path.join(slack_path, sd))
        
        # Read the job processing times
        env_jobs_file = os.path.join(slack_path, sd, "odprog_env_jobs.csv")
        if os.path.exists(env_jobs_file):
            df_jobs = pd.read_csv(env_jobs_file)
            avg_proc_time = df_jobs["total_proc_time_mean"].mean()
            avg_min_proc_time = df_jobs["min_total_proc_time"].mean()
            avg_k = df_jobs["k_value"].mean()
            results.append({
                "Seed": seed,
                "Tardiness": td,
                "Avg_Proc_Time": avg_proc_time,
                "Avg_Min_Proc_Time": avg_min_proc_time,
                "Avg_K_Value": avg_k
            })
            
    df = pd.DataFrame(results)
    print("================== PROCESS TIME VS TARDINESS (slack0瓶頸300) ==================")
    print(df.to_string(index=False))
    print("\nCorrelation with Tardiness:")
    print(df.corr()["Tardiness"])

if __name__ == "__main__":
    main()
