import os
import re
import pandas as pd

def main():
    base_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\普通\cad 1"
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return
        
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dfs = []
    
    for d in subdirs:
        match = re.search(r"seed(\d+)", d, re.IGNORECASE)
        if not match:
            continue
        seed_num = int(match.group(1))
        
        log_file = os.path.join(base_dir, d, "odprog_ppo_release_log.csv")
        if os.path.exists(log_file):
            print(f"Extracting Global_Total_Tardiness for Seed {seed_num:02d}...")
            df = pd.read_csv(log_file)
            
            # Filter out non-numeric summary rows (like Event_ID = 'END')
            df = df[df["Event_ID"] != "END"]
            
            # Keep only Event_ID and Global_Total_Tardiness, renaming the tardiness column to the seed name
            df_sub = df[["Event_ID", "Global_Total_Tardiness"]].copy()
            df_sub = df_sub.rename(columns={"Global_Total_Tardiness": f"Seed_{seed_num}"})
            dfs.append(df_sub)
            
    if dfs:
        # Merge all dataframes on Event_ID using outer join
        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = pd.merge(df_merged, df, on="Event_ID", how="outer")
            
        # Sort columns to make sure Seed_1 to Seed_10 are in sequential order
        seed_cols = sorted([c for c in df_merged.columns if c.startswith("Seed_")], key=lambda x: int(x.split("_")[1]))
        df_merged = df_merged[["Event_ID"] + seed_cols]
        
        # Convert Event_ID to integer to ensure numerical sorting (not alphabetical)
        df_merged["Event_ID"] = df_merged["Event_ID"].astype(int)
        df_merged = df_merged.sort_values(by="Event_ID")
        
        out_csv = os.path.join(base_dir, "release_tardiness_pivot.csv")
        df_merged.to_csv(out_csv, index=False)
        print(f"\nSuccess! Pivot CSV compiled and saved to: {out_csv}")
    else:
        print("No odprog_ppo_release_log.csv files found!")

if __name__ == "__main__":
    main()
