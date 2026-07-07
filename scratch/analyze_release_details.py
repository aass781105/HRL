import os
import re
import csv
import math
import sys

def get_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def get_std(lst):
    if len(lst) <= 1:
        return 0.0
    mean_val = get_mean(lst)
    variance = sum((x - mean_val) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)

def safe_float(val):
    if val is None:
        return 0.0
    val_str = str(val).strip()
    if not val_str or val_str == "":
        return 0.0
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def main():
    if len(sys.argv) > 1:
        target_dir = sys.argv[1].strip('"').strip("'")
    else:
        print("Usage: python analyze_release_details.py <target_directory> [output_filename]")
        return
        
    output_name = sys.argv[2].strip('"').strip("'") if len(sys.argv) > 2 else "ppo_all_releases_detail.csv"
    output_csv_path = os.path.join(target_dir, output_name)
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return
        
    folders = os.listdir(target_dir)
    matched_folders = []
    
    for f in folders:
        folder_path = os.path.join(target_dir, f)
        if not os.path.isdir(folder_path):
            continue
            
        m = re.search(r"^(2026\d{4}_\d{6})_.*_seed0*(\d+)$", f)
        if m:
            timestamp = m.group(1)
            seed_num = int(m.group(2))
            
            raw_path = os.path.join(folder_path, "odprog_raw_state.csv")
            agent_path = os.path.join(folder_path, "odprog_agent_state.csv")
            log_path = os.path.join(folder_path, "odprog_ppo_release_log.csv")
            
            if os.path.exists(raw_path) and os.path.exists(agent_path) and os.path.exists(log_path):
                matched_folders.append({
                    "folder": f,
                    "timestamp": timestamp,
                    "seed": seed_num,
                    "raw_path": raw_path,
                    "agent_path": agent_path,
                    "log_path": log_path
                })
                
    # Group by seed and get latest
    latest_runs = {}
    for entry in matched_folders:
        s = entry["seed"]
        if 1 <= s <= 10:
            if s not in latest_runs or entry["timestamp"] > latest_runs[s]["timestamp"]:
                latest_runs[s] = entry
                
    print(f"Scanning target dir: {target_dir}")
    print(f"Output filename: {output_name}")
    
    headers = [
        "Seed", "Event_ID", "Time", 
        "Raw_Buffer_Count", "Raw_WIP_Job_Count", "Raw_WIP_Tardy_Ratio",
        "Raw_Buf_Min_Slack", "Raw_Buf_Slack_Q25", "Raw_Buf_Avg_Slack", "Raw_Buf_Slack_Std",
        "Raw_Min_Load", "Raw_Avg_Load", "Raw_Load_Std",
        "Buffer_WIP_Load_Overlap",
        "Rescheduled_Subproblem_TD",
        "PPO_Scheduled_TD", "Baseline_Scheduled_TD"
    ]
    
    stats_cols = [
        "Raw_Buffer_Count", "Raw_WIP_Job_Count", "Raw_WIP_Tardy_Ratio",
        "Raw_Buf_Min_Slack", "Raw_Buf_Slack_Q25", "Raw_Buf_Avg_Slack", "Raw_Buf_Slack_Std",
        "Raw_Min_Load", "Raw_Avg_Load", "Raw_Load_Std",
        "Buffer_WIP_Load_Overlap",
        "Rescheduled_Subproblem_TD",
        "PPO_Scheduled_TD", "Baseline_Scheduled_TD"
    ]
    col_vals = {c: [] for c in stats_cols}
    
    all_release_rows = []
    
    for s in sorted(latest_runs.keys()):
        raw_path = latest_runs[s]["raw_path"]
        agent_path = latest_runs[s]["agent_path"]
        log_path = latest_runs[s]["log_path"]
        
        # Read agent state to get overlap values by Event_ID
        agent_data = {}
        with open(agent_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    eid = int(row["Event_ID"])
                    agent_data[eid] = {
                        "Buffer_WIP_Load_Overlap": safe_float(row.get("Buffer_WIP_Load_Overlap", 0.0))
                    }
                except (ValueError, KeyError, TypeError):
                    continue
                    
        # Read release log to get subproblem tardiness by Event_ID
        log_data = {}
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    eid_str = row["Event_ID"].strip()
                    if eid_str.isdigit():
                        eid = int(eid_str)
                        log_data[eid] = {
                            "Total_Tardiness": safe_float(row.get("Total_Tardiness", 0.0))
                        }
                except (ValueError, KeyError, TypeError):
                    continue
                    
        # Read raw state to get raw features and match by Event_ID
        with open(raw_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    action_val = int(safe_float(row.get("Action", 0.0)))
                except (ValueError, TypeError):
                    continue
                    
                if action_val == 1:
                    eid = int(row["Event_ID"])
                    # Check if we have matching agent data
                    if eid not in agent_data:
                        continue
                        
                    sub_td = log_data.get(eid, {}).get("Total_Tardiness", 0.0)
                    
                    release_entry = {
                        "Seed": f"seed{s}",
                        "Event_ID": eid,
                        "Time": safe_float(row.get("Time", 0.0)),
                        "Raw_Buffer_Count": safe_float(row.get("Raw_Buffer_Count", 0.0)),
                        "Raw_WIP_Job_Count": safe_float(row.get("Raw_WIP_Job_Count", 0.0)),
                        "Raw_WIP_Tardy_Ratio": safe_float(row.get("Raw_WIP_Tardy_Ratio", 0.0)),
                        "Raw_Buf_Min_Slack": safe_float(row.get("Raw_Buf_Min_Slack", 0.0)),
                        "Raw_Buf_Slack_Q25": safe_float(row.get("Raw_Buf_Slack_Q25", 0.0)),
                        "Raw_Buf_Avg_Slack": safe_float(row.get("Raw_Buf_Avg_Slack", 0.0)),
                        "Raw_Buf_Slack_Std": safe_float(row.get("Raw_Buf_Slack_Std", 0.0)),
                        "Raw_Min_Load": safe_float(row.get("Raw_Min_Load", 0.0)),
                        "Raw_Avg_Load": safe_float(row.get("Raw_Avg_Load", 0.0)),
                        "Raw_Load_Std": safe_float(row.get("Raw_Load_Std", 0.0)),
                        "Buffer_WIP_Load_Overlap": agent_data[eid]["Buffer_WIP_Load_Overlap"],
                        "Rescheduled_Subproblem_TD": sub_td,
                        "PPO_Scheduled_TD": safe_float(row.get("Actual_TD", 0.0)),
                        "Baseline_Scheduled_TD": safe_float(row.get("Baseline_Step_TD", 0.0))
                    }
                    
                    for c in stats_cols:
                        col_vals[c].append(release_entry[c])
                        
                    all_release_rows.append(release_entry)
                    
    print(f"Total release events found: {len(all_release_rows)}")
    
    # Compute mean row
    mean_row = {"Seed": "mean", "Event_ID": "-", "Time": "-"}
    for c in stats_cols:
        mean_row[c] = get_mean(col_vals[c]) if col_vals[c] else ""
        
    # Compute std row
    std_row = {"Seed": "std", "Event_ID": "-", "Time": "-"}
    for c in stats_cols:
        std_row[c] = get_std(col_vals[c]) if col_vals[c] else ""
        
    # Write to CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        # Write individual events
        for r in all_release_rows:
            formatted_row = {}
            for k, v in r.items():
                if k in ("Seed", "Event_ID"):
                    formatted_row[k] = v
                elif k == "Time":
                    formatted_row[k] = f"{v:.2f}"
                elif k in ("Raw_Buffer_Count", "Raw_WIP_Job_Count"):
                    formatted_row[k] = str(int(round(v)))
                elif isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)
            
        # Write mean & std
        for r in (mean_row, std_row):
            formatted_row = {}
            for k, v in r.items():
                if k in ("Seed", "Event_ID", "Time"):
                    formatted_row[k] = v
                elif isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)
            
    print(f"Successfully generated release details CSV at: {output_csv_path}")

if __name__ == "__main__":
    main()
