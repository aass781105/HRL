import csv
import os

def read_mean_row(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Seed"] == "mean":
                return row
    return None

def main():
    paths = {
        "PPObaseline": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPObaseline\ppo_all_releases_detail.csv",
        "slack0baseline": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\slack0baseline\slack0_all_releases_detail.csv",
        "PPO瓶頸": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPO瓶頸\ppo_all_releases_detail.csv",
        "slack0瓶頸": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\slack0瓶頸\slack0_all_releases_detail.csv",
    }
    
    print("Scenario | Buffer_Jobs | WIP_Jobs | WIP_Tardy_Ratio | Min_Slack | Avg_Slack | Min_Load | Avg_Load | Subproblem_TD | Total_TD | Overlap")
    print("-" * 140)
    for name, p in paths.items():
        row = read_mean_row(p)
        if row:
            print(f"{name:14s} | {row['Raw_Buffer_Count']:11s} | {row['Raw_WIP_Job_Count']:8s} | {float(row['Raw_WIP_Tardy_Ratio'])*100:13.2f}% | {row['Raw_Buf_Min_Slack']:9s} | {row['Raw_Buf_Avg_Slack']:9s} | {row['Raw_Min_Load']:8s} | {row['Raw_Avg_Load']:8s} | {row['Rescheduled_Subproblem_TD']:13s} | {row['PPO_Scheduled_TD']:8s} | {row['Buffer_WIP_Load_Overlap']:7s}")
            
if __name__ == "__main__":
    main()
