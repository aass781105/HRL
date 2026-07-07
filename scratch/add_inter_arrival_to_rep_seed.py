import os
import re
import pandas as pd
import numpy as np

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    
    rep_csv_path = os.path.join(meeting_dir, "找代表seed.csv")
    if not os.path.exists(rep_csv_path):
        print(f"Error: {rep_csv_path} not found!")
        return
        
    df_rep = pd.read_csv(rep_csv_path)
    
    # We will walk through the meeting directory to find any valid env_jobs CSV files for each seed and environment
    # Let's map (env, seed) -> (avg_inter_arrival, std_inter_arrival)
    inter_arrival_map = {}
    
    print("Scanning directories under meeting_dir to find job logs...")
    for root, dirs, files in os.walk(meeting_dir):
        if "odprog_env_jobs.csv" in files:
            # Detect scenario and seed from parent folder name or path
            # Parent folder name typically contains seed (e.g. ..._seed001)
            folder_name = os.path.basename(root)
            m = re.search(r"seed0*(\d+)", folder_name)
            if m:
                seed = int(m.group(1))
                # Determine environment: baseline or bottleneck
                env = None
                if "baseline" in root.lower() or "ppobaseline" in root.lower():
                    env = "baseline"
                elif "瓶頸" in root.lower() or "bottleneck" in root.lower() or "slack0" in root.lower():
                    env = "bottleneck"
                
                if env and seed:
                    key = (env, seed)
                    if key not in inter_arrival_map:
                        try:
                            df_jobs = pd.read_csv(os.path.join(root, "odprog_env_jobs.csv"))
                            # Skip the first row if its inter_arrival is 0.0 (initial jobs at t=0) or check all
                            # Let's calculate mean and std of inter_arrival column
                            if "inter_arrival" in df_jobs.columns:
                                # Usually inter_arrival for job_id 0 is 0.0, we want to include all or non-zero?
                                # Let's include all arrival intervals that are part of the stream
                                intervals = df_jobs["inter_arrival"].values
                                # Note: the first few jobs arrive at 0.0, which is correct (inter-arrival = 0.0)
                                mean_val = float(np.mean(intervals))
                                std_val = float(np.std(intervals))
                                inter_arrival_map[key] = (mean_val, std_val)
                                print(f"Found log for Env={env}, Seed={seed}: Mean={mean_val:.4f}, Std={std_val:.4f} (Source: {folder_name})")
                        except Exception as e:
                            print(f"Error reading env_jobs in {root}: {e}")

    # Now we map these new metrics to df_rep
    avg_arr_col = []
    std_arr_col = []
    
    for idx, row in df_rep.iterrows():
        seed = int(row["Seed"])
        env = row["Env"]  # 'baseline' or 'bottleneck'
        
        # If the env name is 'bottleneck', match it with our mapped 'bottleneck'
        env_key = "bottleneck" if "bottleneck" in env.lower() else "baseline"
        
        key = (env_key, seed)
        if key in inter_arrival_map:
            avg_arr_col.append(round(inter_arrival_map[key][0], 2))
            std_arr_col.append(round(inter_arrival_map[key][1], 2))
        else:
            avg_arr_col.append(None)
            std_arr_col.append(None)
            print(f"Warning: No job log found for Seed {seed} in {env_key} environment!")

    df_rep["Avg_Inter_Arrival"] = avg_arr_col
    df_rep["Std_Inter_Arrival"] = std_arr_col
    
    # Save back to CSV
    df_rep.to_csv(rep_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSuccessfully updated {rep_csv_path} with arrival intervals columns!")

if __name__ == "__main__":
    main()
