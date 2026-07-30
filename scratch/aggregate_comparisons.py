import os
import sys
import pandas as pd
import numpy as np

# Target directory containing the policy folders
root_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260715\新環境情境"
policies = ["PPO", "cad1", "cad5", "slack0"]

def format_td_releases(td, rel):
    try:
        td_val = int(round(float(td)))
    except:
        td_val = "-"
    
    # Format releases: if float and represents whole number, show as int, else format to float representation
    try:
        rel_float = float(rel)
        if rel_float.is_integer():
            rel_val = str(int(rel_float))
        else:
            rel_val = f"{rel_float:.4f}" if len(str(rel_float).split('.')[1]) > 2 else f"{rel_float:.1f}"
    except:
        rel_val = str(rel)
        
    return f"{td_val} / {rel_val}"

for policy in policies:
    policy_path = os.path.join(root_dir, policy)
    if not os.path.exists(policy_path):
        print(f"Policy folder not found: {policy_path}")
        continue
        
    print(f"\nProcessing policy: {policy}")
    
    # Find subdirectories in this policy folder
    subdirs = [d for d in os.listdir(policy_path) if os.path.isdir(os.path.join(policy_path, d))]
    
    # Build a map from seed index (1 to 10) to the corresponding directory name
    seed_to_dir = {}
    for i in range(1, 11):
        suffix = f"_seed{i:03d}"
        matched = [d for d in subdirs if d.endswith(suffix)]
        if matched:
            seed_to_dir[i] = matched[0]
            
    # Gather data for seeds 1 to 10
    rows = []
    
    makespans = []
    tardinesses = []
    objectives = []
    releases = []
    elapsed_times = []
    
    for i in range(1, 11):
        seed_label = f"seed{i}"
        if i not in seed_to_dir:
            print(f"Warning: Seed {i} folder not found in {policy}!")
            continue
            
        dir_name = seed_to_dir[i]
        csv_filepath = os.path.join(policy_path, dir_name, "sample_runs_summary.csv")
        
        if not os.path.exists(csv_filepath):
            print(f"Warning: CSV not found: {csv_filepath}")
            continue
            
        # Read the CSV
        df_csv = pd.read_csv(csv_filepath)
        # Find the row where 'run' column is 'mean'
        mean_row = df_csv[df_csv['run'] == 'mean']
        if mean_row.empty:
            print(f"Warning: No 'mean' row in {csv_filepath}")
            continue
            
        mean_row = mean_row.iloc[0]
        
        mk = float(mean_row['makespan'])
        td = float(mean_row['total_tardiness'])
        obj = float(mean_row['obj'])
        rel = float(mean_row['release_count'])
        elapsed = float(mean_row['elapsed_time_sec'])
        
        makespans.append(mk)
        tardinesses.append(td)
        objectives.append(obj)
        releases.append(rel)
        elapsed_times.append(elapsed)
        
        td_rel_str = format_td_releases(td, rel)
        
        rows.append({
            "Seed": seed_label,
            "Folder_Name": dir_name,
            "Makespan": mk,
            "Tardiness": td,
            "Objective": obj,
            "Releases": int(rel) if rel.is_integer() else rel,
            "Tardiness / Releases": td_rel_str,
            "Elapsed_Time_Sec": elapsed
        })
        
    if not rows:
        print(f"No data gathered for policy {policy}")
        continue
        
    # Calculate Mean (Population stats, matching reference)
    m_mk = np.mean(makespans)
    m_td = np.mean(tardinesses)
    m_obj = np.mean(objectives)
    m_rel = np.mean(releases)
    m_el = np.mean(elapsed_times)
    
    # Calculate Std (ddof=0 to match reference population std)
    s_mk = np.std(makespans, ddof=0)
    s_td = np.std(tardinesses, ddof=0)
    s_obj = np.std(objectives, ddof=0)
    s_rel = np.std(releases, ddof=0)
    s_el = np.std(elapsed_times, ddof=0)
    
    # Append Mean Row
    rows.append({
        "Seed": "mean",
        "Folder_Name": "-",
        "Makespan": m_mk,
        "Tardiness": m_td,
        "Objective": m_obj,
        "Releases": round(m_rel, 4),
        "Tardiness / Releases": format_td_releases(m_td, m_rel),
        "Elapsed_Time_Sec": m_el
    })
    
    # Append Std Row
    rows.append({
        "Seed": "std",
        "Folder_Name": "-",
        "Makespan": s_mk,
        "Tardiness": s_td,
        "Objective": s_obj,
        "Releases": round(s_rel, 4),
        "Tardiness / Releases": format_td_releases(s_td, s_rel),
        "Elapsed_Time_Sec": s_el
    })
    
    # Create DataFrame and save to CSV
    df_out = pd.DataFrame(rows)
    
    # Format specific columns to match precision of reference file
    # For individual seeds: Makespan, Tardiness, Objective, Elapsed_Time_Sec are floats or ints
    # In reference: Mean/Std row uses 4 decimal places for most columns, and Makespan std uses 3 decimal places.
    # Let's keep the raw float values and format them during printing or just write them as floats, pandas does it cleanly.
    # To match reference layout perfectly, we can format floats to matching decimals
    # Seed 1-10: Makespan (int if whole, else float), Tardiness, Objective, Elapsed_Time_Sec
    # Mean: 4 decimals. Std: Makespan 3 decimals, others 4 decimals.
    
    output_filename = f"{policy}_10seeds_comparison.csv"
    output_filepath = os.path.join(root_dir, output_filename)
    
    df_out.to_csv(output_filepath, index=False)
    print(f"Saved: {output_filepath}")
