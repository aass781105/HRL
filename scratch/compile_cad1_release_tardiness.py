import os
import re
import pandas as pd

def main():
    base_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\普通\cad 1"
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return
        
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    all_data = []
    
    for d in subdirs:
        # Extract seed number from folder name (e.g. seed001 -> 1)
        match = re.search(r"seed(\d+)", d, re.IGNORECASE)
        if not match:
            continue
        seed_num = int(match.group(1))
        
        log_file = os.path.join(base_dir, d, "odprog_ppo_release_log.csv")
        if os.path.exists(log_file):
            print(f"Processing Seed {seed_num:02d} from folder: {d}")
            df = pd.read_csv(log_file)
            
            # Keep tardiness and identifying columns
            cols_to_keep = ["Event_ID", "Release_Type", "Release_Time", "Total_Tardiness", "Global_Total_Tardiness", "Makespan", "Global_Makespan"]
            cols_present = [c for c in cols_to_keep if c in df.columns]
            
            df_sub = df[cols_present].copy()
            df_sub["Seed"] = seed_num
            all_data.append(df_sub)
            
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        # Put Seed column first
        cols = ["Seed"] + [c for c in df_all.columns if c != "Seed"]
        df_all = df_all[cols]
        # Sort by Seed and Event_ID
        df_all = df_all.sort_values(by=["Seed", "Event_ID"])
        
        out_csv = os.path.join(base_dir, "release_tardiness_summary.csv")
        df_all.to_csv(out_csv, index=False)
        print(f"\nSuccess! All release logs compiled and saved to: {out_csv}")
    else:
        print("No odprog_ppo_release_log.csv files found in subdirectories!")

if __name__ == "__main__":
    main()
