import csv
import os
import re

def main():
    target_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPO瓶頸"
    if not os.path.exists(target_dir):
        print(f"Dir not found: {target_dir}")
        return
        
    non_zero_base = 0
    total_valid = 0
    for f in os.listdir(target_dir):
        folder_path = os.path.join(target_dir, f)
        if not os.path.isdir(folder_path):
            continue
        m = re.search(r"^(2026\d{4}_\d{6})_.*_seed0*(\d+)$", f)
        if not m:
            continue
            
        raw_path = os.path.join(folder_path, "odprog_raw_state.csv")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if not row.get("Event_ID") or row["Event_ID"].strip() == "" or row["Event_ID"] == "END":
                        continue
                    total_valid += 1
                    try:
                        base_val = float(row["Baseline_Step_TD"]) if row["Baseline_Step_TD"].strip() else 0.0
                    except (ValueError, KeyError, TypeError):
                        base_val = 0.0
                    if base_val > 0:
                        non_zero_base += 1
                        
    print(f"Across all seeds in PPO瓶頸:")
    print(f"  Total valid decision points: {total_valid}")
    print(f"  Non-zero Baseline_Step_TD count: {non_zero_base}")

if __name__ == "__main__":
    main()
