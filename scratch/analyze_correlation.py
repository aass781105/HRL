import os
import pandas as pd
import numpy as np

def extract_metrics(folder_path):
    summary_file = os.path.join(folder_path, "sample_runs_summary.csv")
    if not os.path.exists(summary_file):
        return None, None
    try:
        df = pd.read_csv(summary_file)
        mean_row = df[df["run"] == "mean"]
        if mean_row.empty:
            mean_row = df.iloc[0:1]
        mk = float(mean_row["makespan"].values[0])
        td = float(mean_row["total_tardiness"].values[0])
        return mk, td
    except:
        return None, None

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    
    # Load structural analysis
    struct_csv = os.path.join(meeting_dir, "換題差異", "structural_bottleneck_analysis_all_seeds.csv")
    if not os.path.exists(struct_csv):
        print(f"Structural CSV not found: {struct_csv}")
        return
        
    df_struct = pd.read_csv(struct_csv)
    
    # Target folders
    ppo_path = os.path.join(meeting_dir, "PPObaseline300")
    slack_path = os.path.join(meeting_dir, "slack0瓶頸300")
    
    # We will build two datasets
    ppo_rows = []
    slack_rows = []
    
    # 1. PPO Baseline
    if os.path.exists(ppo_path):
        for sd in sorted(os.listdir(ppo_path)):
            if not os.path.isdir(os.path.join(ppo_path, sd)): continue
            try:
                seed = int(sd.split("_")[-1].replace("seed", ""))
            except:
                continue
            mk, td = extract_metrics(os.path.join(ppo_path, sd))
            if td is not None:
                ppo_rows.append({"Seed": seed, "Makespan": mk, "Tardiness": td})
                
    # 2. Slack0 Bottleneck
    if os.path.exists(slack_path):
        for sd in sorted(os.listdir(slack_path)):
            if not os.path.isdir(os.path.join(slack_path, sd)): continue
            try:
                seed = int(sd.split("_")[-1].replace("seed", ""))
            except:
                continue
            mk, td = extract_metrics(os.path.join(slack_path, sd))
            if td is not None:
                slack_rows.append({"Seed": seed, "Makespan": mk, "Tardiness": td})
                
    df_ppo_kpi = pd.DataFrame(ppo_rows)
    df_slack_kpi = pd.DataFrame(slack_rows)
    
    # Merge for PPO Baseline
    df_ppo_struct = df_struct[df_struct["Env"] == "baseline"]
    df_ppo = pd.merge(df_ppo_kpi, df_ppo_struct, on="Seed")
    
    # Merge for Slack Bottleneck
    df_slack_struct = df_struct[df_struct["Env"] == "bottleneck"]
    df_slack = pd.merge(df_slack_kpi, df_slack_struct, on="Seed")
    
    print("================================== PPO BASELINE (300 Horizon) CORRELATION ==================================")
    print(df_ppo[["Seed", "Tardiness", "Urgent_Count", "Mean_Slack", "Machine_Load_Std", "Mean_Compatibility"]].to_string(index=False))
    print("\nCorrelation matrix with Tardiness (PPO Baseline):")
    corr_ppo = df_ppo[["Tardiness", "Urgent_Count", "Mean_Slack", "Machine_Load_Std", "Mean_Compatibility"]].corr()["Tardiness"]
    print(corr_ppo)
    
    print("\n================================== SLACK0 BOTTLENECK (300 Horizon) CORRELATION =================================")
    print(df_slack[["Seed", "Tardiness", "Urgent_Count", "Mean_Slack", "Machine_Load_Std", "Mean_Compatibility"]].to_string(index=False))
    print("\nCorrelation matrix with Tardiness (Slack0 Bottleneck):")
    corr_slack = df_slack[["Tardiness", "Urgent_Count", "Mean_Slack", "Machine_Load_Std", "Mean_Compatibility"]].corr()["Tardiness"]
    print(corr_slack)

if __name__ == "__main__":
    main()
