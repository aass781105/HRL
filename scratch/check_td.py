import csv
import os

def main():
    target_file = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPO瓶頸\ppo_all_releases_detail.csv"
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return
        
    print(f"Reading file: {target_file}")
    non_zero_baseline = 0
    non_zero_ppo = 0
    total = 0
    with open(target_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Seed"] in ("mean", "std"):
                continue
            total += 1
            ppo_td = float(row["PPO_Scheduled_TD"])
            base_td = float(row["Baseline_Scheduled_TD"])
            if ppo_td > 0:
                non_zero_ppo += 1
            if base_td > 0:
                non_zero_baseline += 1
                
    print(f"Total release events: {total}")
    print(f"Non-zero PPO Scheduled TD count: {non_zero_ppo}")
    print(f"Non-zero Baseline Scheduled TD count: {non_zero_baseline}")

if __name__ == "__main__":
    main()
