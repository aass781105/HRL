import os
import csv
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
    except:
        return None, None, None

def extract_proc_times(folder_path):
    env_jobs_file = os.path.join(folder_path, "odprog_env_jobs.csv")
    if not os.path.exists(env_jobs_file):
        return None, None, None
    try:
        df = pd.read_csv(env_jobs_file)
        avg_pt = float(df["total_proc_time_mean"].mean())
        avg_min_pt = float(df["min_total_proc_time"].mean())
        avg_k = float(df["k_value"].mean())
        return avg_pt, avg_min_pt, avg_k
    except:
        return None, None, None

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    output_csv = os.path.join(meeting_dir, "換題差異", "seed_performance_structural_correlation_300.csv")
    
    # Load structural analysis
    struct_csv = os.path.join(meeting_dir, "換題差異", "structural_bottleneck_analysis_all_seeds.csv")
    if not os.path.exists(struct_csv):
        print(f"Error: {struct_csv} not found")
        return
        
    df_struct = pd.read_csv(struct_csv)
    
    # We will build data for:
    # 1. PPObaseline300 (PPO policy in baseline env)
    # 2. slack0瓶頸300 (slack_threshold policy in bottleneck env)
    ppo_path = os.path.join(meeting_dir, "PPObaseline300")
    slack_path = os.path.join(meeting_dir, "slack0瓶頸300")
    
    rows = []
    
    # Process PPO Baseline
    if os.path.exists(ppo_path):
        for sd in sorted(os.listdir(ppo_path)):
            if not os.path.isdir(os.path.join(ppo_path, sd)): continue
            try:
                seed = int(sd.split("_")[-1].replace("seed", ""))
            except:
                continue
                
            mk, td, rel = extract_metrics(os.path.join(ppo_path, sd))
            avg_pt, avg_min_pt, avg_k = extract_proc_times(os.path.join(ppo_path, sd))
            
            # Find matching struct row
            struct_match = df_struct[(df_struct["Env"] == "baseline") & (df_struct["Seed"] == seed)]
            if not struct_match.empty and td is not None:
                s_row = struct_match.iloc[0]
                rows.append({
                    "Seed": seed,
                    "Env": "baseline",
                    "Policy": "ppo",
                    "Makespan": mk,
                    "Tardiness": td,
                    "Release_Count": rel,
                    "Urgent_Count": int(s_row["Urgent_Count"]),
                    "Mean_Slack": float(s_row["Mean_Slack"]),
                    "Std_Slack": float(s_row["Std_Slack"]),
                    "Urgent_Mean_Slack": float(s_row["Urgent_Mean_Slack"]),
                    "Urgent_Std_Slack": float(s_row["Urgent_Std_Slack"]),
                    "Normal_Mean_Slack": float(s_row["Normal_Mean_Slack"]),
                    "Normal_Std_Slack": float(s_row["Normal_Std_Slack"]),
                    "Machine_Load_Std": float(s_row["Machine_Load_Std"]),
                    "Mean_Compatibility": float(s_row["Mean_Compatibility"]),
                    "Avg_Proc_Time": avg_pt,
                    "Avg_Min_Proc_Time": avg_min_pt,
                    "Avg_K_Value": avg_k
                })
                
    # Process Slack0 Bottleneck
    if os.path.exists(slack_path):
        for sd in sorted(os.listdir(slack_path)):
            if not os.path.isdir(os.path.join(slack_path, sd)): continue
            try:
                seed = int(sd.split("_")[-1].replace("seed", ""))
            except:
                continue
                
            mk, td, rel = extract_metrics(os.path.join(slack_path, sd))
            avg_pt, avg_min_pt, avg_k = extract_proc_times(os.path.join(slack_path, sd))
            
            # Find matching struct row
            struct_match = df_struct[(df_struct["Env"] == "bottleneck") & (df_struct["Seed"] == seed)]
            if not struct_match.empty and td is not None:
                s_row = struct_match.iloc[0]
                rows.append({
                    "Seed": seed,
                    "Env": "bottleneck",
                    "Policy": "slack0",
                    "Makespan": mk,
                    "Tardiness": td,
                    "Release_Count": rel,
                    "Urgent_Count": int(s_row["Urgent_Count"]),
                    "Mean_Slack": float(s_row["Mean_Slack"]),
                    "Std_Slack": float(s_row["Std_Slack"]),
                    "Urgent_Mean_Slack": float(s_row["Urgent_Mean_Slack"]),
                    "Urgent_Std_Slack": float(s_row["Urgent_Std_Slack"]),
                    "Normal_Mean_Slack": float(s_row["Normal_Mean_Slack"]),
                    "Normal_Std_Slack": float(s_row["Normal_Std_Slack"]),
                    "Machine_Load_Std": float(s_row["Machine_Load_Std"]),
                    "Mean_Compatibility": float(s_row["Mean_Compatibility"]),
                    "Avg_Proc_Time": avg_pt,
                    "Avg_Min_Proc_Time": avg_min_pt,
                    "Avg_K_Value": avg_k
                })
                
    if not rows:
        print("No rows generated! Check directory names or data files.")
        return
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Successfully generated correlation data CSV with {len(df_out)} rows at: {output_csv}")

if __name__ == "__main__":
    main()
