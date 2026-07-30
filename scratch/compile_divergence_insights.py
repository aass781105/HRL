import os
import pandas as pd
import numpy as np

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root)) # 碩一
    # Find the latest compare folder inside analysis_results/policy_compare/
    compare_base = os.path.join(project_root, "analysis_results", "policy_compare")
    if not os.path.exists(compare_base):
        print("Compare base folder not found!")
        return
        
    subdirs = sorted([d for d in os.listdir(compare_base) if os.path.isdir(os.path.join(compare_base, d))])
    if not subdirs:
        print("No run folder found!")
        return
        
    latest_run = os.path.join(compare_base, subdirs[-1])
    print(f"Analyzing latest run data from: {latest_run}")
    
    div_csv = os.path.join(latest_run, "divergence_windows.csv")
    if not os.path.exists(div_csv):
        print(f"Error: {div_csv} not found!")
        return
        
    df = pd.read_csv(div_csv)
    
    print("\n" + "=" * 40 + " TYPE B ANALYSIS (PPO Hold, Slack0 Release) " + "=" * 40)
    df_b = df[df["Divergence_Type"] == "B"]
    if not df_b.empty:
        # At Relative_Step = 0, state features are identical between PPO and Slack0
        df_b0 = df_b[df_b["Relative_Step"] == 0]
        # Since it's identical, we can just group by Seed and take one policy's values
        df_b0_unique = df_b0[df_b0["Policy"] == "ppo"]
        
        avg_wip_0 = df_b0_unique["WIP_Count_Before"].mean()
        avg_buf_0 = df_b0_unique["Buffer_Count_Before"].mean()
        avg_min_slack_0 = df_b0_unique["Buffer_Min_Slack_Before"].mean()
        avg_neg_slack_sum_0 = df_b0_unique["Buffer_Neg_Slack_Sum_Before"].mean()
        avg_overlap_0 = df_b0_unique["Buffer_WIP_Load_Overlap_Before"].mean()
        avg_load_std_0 = df_b0_unique["Machine_Load_Std_Before"].mean()
        
        print(f"At Divergence Point (Relative Step = 0):")
        print(f"  - Average WIP count on floor: {avg_wip_0:.1f} jobs")
        print(f"  - Average Buffer queue length: {avg_buf_0:.1f} jobs")
        print(f"  - Average Buffer Min Slack: {avg_min_slack_0:.2f} mins (Slack0 is triggered because it is < 0)")
        print(f"  - Average Buffer Neg Slack Sum: {avg_neg_slack_sum_0:.2f} mins")
        print(f"  - Average Buffer-WIP Load Overlap: {avg_overlap_0:.3f} (Conflict index)")
        print(f"  - Average Machine Load Std: {avg_load_std_0:.2f}")
        
        # Now check post-decision trajectory at Relative_Step = 1 (1 step after)
        df_b1_ppo = df_b[(df_b["Relative_Step"] == 1) & (df_b["Policy"] == "ppo")]
        df_b1_slack = df_b[(df_b["Relative_Step"] == 1) & (df_b["Policy"] == "slack0")]
        
        print(f"\nAfter 1 decision step (Relative Step = 1):")
        print(f"  - PPO (chose HOLD):")
        print(f"    * Average WIP count on floor: {df_b1_ppo['WIP_Count_Before'].mean():.1f} jobs")
        print(f"    * Average Makespan estimate: {df_b1_ppo['Makespan_Estimate'].mean():.2f} mins")
        print(f"    * Average Actual TD: {df_b1_ppo['Tardiness_Actual'].mean():.2f} mins")
        
        print(f"  - Slack0 (chose RELEASE):")
        print(f"    * Average WIP count on floor: {df_b1_slack['WIP_Count_Before'].mean():.1f} jobs")
        print(f"    * Average Makespan estimate: {df_b1_slack['Makespan_Estimate'].mean():.2f} mins")
        print(f"    * Average Actual TD: {df_b1_slack['Tardiness_Actual'].mean():.2f} mins")
        
        # Check post-decision trajectory at Relative_Step = 5 (5 steps after)
        df_b5_ppo = df_b[(df_b["Relative_Step"] == 5) & (df_b["Policy"] == "ppo")]
        df_b5_slack = df_b[(df_b["Relative_Step"] == 5) & (df_b["Policy"] == "slack0")]
        
        print(f"\nAfter 5 decision steps (Relative Step = 5):")
        print(f"  - PPO (chose HOLD):")
        print(f"    * Average WIP count on floor: {df_b5_ppo['WIP_Count_Before'].mean():.1f} jobs")
        print(f"    * Average Makespan estimate: {df_b5_ppo['Makespan_Estimate'].mean():.2f} mins")
        print(f"    * Average Actual TD: {df_b5_ppo['Tardiness_Actual'].mean():.2f} mins")
        
        print(f"  - Slack0 (chose RELEASE):")
        print(f"    * Average WIP count on floor: {df_b5_slack['WIP_Count_Before'].mean():.1f} jobs")
        print(f"    * Average Makespan estimate: {df_b5_slack['Makespan_Estimate'].mean():.2f} mins")
        print(f"    * Average Actual TD: {df_b5_slack['Tardiness_Actual'].mean():.2f} mins")
        
    print("\n" + "=" * 40 + " TYPE A ANALYSIS (PPO Release, Slack0 Hold) " + "=" * 40)
    df_a = df[df["Divergence_Type"] == "A"]
    if not df_a.empty:
        df_a0 = df_a[df_a["Relative_Step"] == 0]
        df_a0_unique = df_a0[df_a0["Policy"] == "ppo"]
        
        avg_wip_0 = df_a0_unique["WIP_Count_Before"].mean()
        avg_buf_0 = df_a0_unique["Buffer_Count_Before"].mean()
        avg_min_slack_0 = df_a0_unique["Buffer_Min_Slack_Before"].mean()
        avg_neg_slack_sum_0 = df_a0_unique["Buffer_Neg_Slack_Sum_Before"].mean()
        avg_overlap_0 = df_a0_unique["Buffer_WIP_Load_Overlap_Before"].mean()
        avg_load_std_0 = df_a0_unique["Machine_Load_Std_Before"].mean()
        
        print(f"At Divergence Point (Relative Step = 0):")
        print(f"  - Average WIP count on floor: {avg_wip_0:.1f} jobs")
        print(f"  - Average Buffer queue length: {avg_buf_0:.1f} jobs")
        print(f"  - Average Buffer Min Slack: {avg_min_slack_0:.2f} mins (Slack0 is NOT triggered because it is >= 0)")
        print(f"  - Average Buffer Neg Slack Sum: {avg_neg_slack_sum_0:.2f} mins")
        print(f"  - Average Buffer-WIP Load Overlap: {avg_overlap_0:.3f}")
        print(f"  - Average Machine Load Std: {avg_load_std_0:.2f}")
        
        # Check post-decision trajectory at Relative_Step = 5 (5 steps after)
        df_a5_ppo = df_a[(df_a["Relative_Step"] == 5) & (df_a["Policy"] == "ppo")]
        df_a5_slack = df_a[(df_a["Relative_Step"] == 5) & (df_a["Policy"] == "slack0")]
        
        print(f"\nAfter 5 decision steps (Relative Step = 5):")
        print(f"  - PPO (chose RELEASE):")
        print(f"    * Average WIP count on floor: {df_a5_ppo['WIP_Count_Before'].mean():.1f} jobs")
        print(f"    * Average Makespan estimate: {df_a5_ppo['Makespan_Estimate'].mean():.2f} mins")
        print(f"    * Average Actual TD: {df_a5_ppo['Tardiness_Actual'].mean():.2f} mins")
        
        print(f"  - Slack0 (chose HOLD):")
        print(f"    * Average WIP count on floor: {df_a5_slack['WIP_Count_Before'].mean():.1f} jobs")
        print(f"    * Average Makespan estimate: {df_a5_slack['Makespan_Estimate'].mean():.2f} mins")
        print(f"    * Average Actual TD: {df_a5_slack['Tardiness_Actual'].mean():.2f} mins")
    else:
        print("No Type A divergence windows found in this data!")

if __name__ == "__main__":
    main()
