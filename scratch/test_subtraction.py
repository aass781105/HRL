import csv
import os

def main():
    target_file = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPO瓶頸\20260706_150950_odprog_seed001\odprog_raw_state.csv"
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return
        
    print(f"Event_ID | Time | Actual_TD (Total) | Raw_WIP_Planned_TD (WIP) | Completed_TD (Diff)")
    print("-" * 75)
    with open(target_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Event_ID") or row["Event_ID"].strip() == "" or row["Event_ID"] == "END":
                continue
            act = int(float(row["Action"]))
            if act == 1:
                eid = int(row["Event_ID"])
                t_now = float(row["Time"])
                actual_td = float(row["Actual_TD"]) if row["Actual_TD"].strip() else 0.0
                wip_td = float(row["Raw_WIP_Planned_TD"]) if row["Raw_WIP_Planned_TD"].strip() else 0.0
                completed_td = max(0.0, actual_td - wip_td)
                print(f"{eid:8d} | {t_now:8.2f} | {actual_td:17.2f} | {wip_td:24.2f} | {completed_td:18.2f}")

if __name__ == "__main__":
    main()
