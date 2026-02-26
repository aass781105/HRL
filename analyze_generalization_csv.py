
import os
import pandas as pd
import glob
import re
import numpy as np

def analyze_generalization_csv():
    results_dir = 'test_results'
    output_file = 'Generalization_Analysis_Summary.csv'
    
    # 1. Define Groups and their filename patterns
    groups = {
        'S1_Base': r'generalization_curric_s1_baseline_seed\d',
        'S2_Size': r'generalization_curric_s2_size_seed\d',
        'S3_SizeM':r'generalization_curric_s3_size_m_seed\d',
        'S4_Noise':r'generalization_curric_s4_noise_shift_seed\d'
    }
    
    # Order for columns
    group_order = ['S1_Base', 'S2_Size', 'S3_SizeM', 'S4_Noise']

    # 2. Data Collection
    m_values = ['0.2', '0.4', '0.6']
    metrics = ['MS', 'TD', 'OBJ']
    job_sizes = [10, 20, 30, 40, 50, 60]

    # Structure: {M_Val: {Metric: {Group: {JobSize: [values]}}}}
    # New Structure for Detail: {M_Val: {Group: {SeedID: {JobSize: value}}}}
    raw_data = {
        m: {
            met: {g: {s: [] for s in job_sizes} for g in group_order} 
            for met in metrics
        } for m in m_values
    }
    
    # {M_Val: {Group: {Seed: {JobSize: Value}}}}
    detail_data = {
        m: {g: {} for g in group_order} for m in m_values
    }

    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Identify group
        matched_group = None
        for g_name, pattern in groups.items():
            if re.search(pattern, filename):
                matched_group = g_name
                break
        
        if not matched_group:
            continue
            
        # Extract Seed ID (assume format _seedX, default to 1 if not found)
        seed_match = re.search(r'_seed(\d+)', filename)
        seed_id = int(seed_match.group(1)) if seed_match else 1
            
        try:
            df = pd.read_csv(file_path)
            
            for index, row in df.iterrows():
                size_val = row['Job_Size']
                try:
                    size = int(float(size_val))
                except:
                    continue
                
                if size not in job_sizes: continue
                
                for m in m_values:
                    # Collect Aggregated Data
                    for met in metrics:
                        col_name = f'{met}_{m}'
                        if col_name in row:
                            val = row[col_name]
                            raw_data[m][met][matched_group][size].append(val)
                    
                    # Collect Detail Data (OBJ only)
                    obj_col = f'OBJ_{m}'
                    if obj_col in row:
                        if seed_id not in detail_data[m][matched_group]:
                            detail_data[m][matched_group][seed_id] = {}
                        detail_data[m][matched_group][seed_id][size] = row[obj_col]
                            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 3. Generate Tables
    final_dfs = []
    
    # For each M-value (difficulty)
    for m in m_values:
        metric = 'OBJ' 
        
        # ... [Existing MEAN/STD/GAP tables code] ...
        
        # --- Table 1: MEAN ---
        mean_rows = []
        for size in job_sizes:
            row = {'Job_Size': size}
            for g in group_order:
                vals = raw_data[m][metric][g][size]
                row[g] = np.mean(vals) if vals else np.nan
            mean_rows.append(row)
        
        df_mean = pd.DataFrame(mean_rows)
        header_mean = pd.DataFrame([["" for _ in df_mean.columns]], columns=df_mean.columns)
        header_mean.iloc[0, 0] = f"--- [MEAN] Objective (M={m}) ---"
        final_dfs.append(header_mean)
        final_dfs.append(df_mean)
        final_dfs.append(pd.DataFrame([["" for _ in df_mean.columns]], columns=df_mean.columns))

        # --- Table 2: STD ---
        std_rows = []
        for size in job_sizes:
            row = {'Job_Size': size}
            for g in group_order:
                vals = raw_data[m][metric][g][size]
                row[g] = np.std(vals) if vals else np.nan
            std_rows.append(row)
        df_std = pd.DataFrame(std_rows)
        header_std = pd.DataFrame([["" for _ in df_std.columns]], columns=df_std.columns)
        header_std.iloc[0, 0] = f"--- [STD] Objective (M={m}) ---"
        final_dfs.append(header_std)
        final_dfs.append(df_std)
        final_dfs.append(pd.DataFrame([["" for _ in df_std.columns]], columns=df_std.columns))

        # --- Table 3: GAP ---
        gap_rows = []
        baseline_group = 'S1_Base'
        for i, row in df_mean.iterrows():
            gap_row = {'Job_Size': row['Job_Size']}
            base_val = row[baseline_group]
            for g in group_order:
                val = row[g]
                if g == baseline_group:
                    gap_row[g] = "0.00%"
                elif pd.notna(val) and pd.notna(base_val) and base_val != 0:
                    gap = ((val - base_val) / base_val) * 100
                    gap_row[g] = f"{gap:+.2f}%"
                else:
                    gap_row[g] = "N/A"
            gap_rows.append(gap_row)
        df_gap = pd.DataFrame(gap_rows)
        header_gap = pd.DataFrame([["" for _ in df_gap.columns]], columns=df_gap.columns)
        header_gap.iloc[0, 0] = f"--- [GAP] vs S1_Base (M={m}) ---"
        final_dfs.append(header_gap)
        final_dfs.append(df_gap)
        final_dfs.append(pd.DataFrame([["" for _ in df_gap.columns]], columns=df_gap.columns))

        # --- [NEW] Table 4: DETAIL (Individual Seeds) ---
        # Columns: Job_Size, S1_s1, S1_s2..., S2_s1...
        detail_rows = []
        
        # Prepare columns
        detail_cols = ['Job_Size']
        for g in group_order:
            for s in range(1, 6): # Seeds 1-5
                detail_cols.append(f"{g}_s{s}")
        
        for size in job_sizes:
            row = {'Job_Size': size}
            for g in group_order:
                for s in range(1, 6):
                    val = detail_data[m][g].get(s, {}).get(size, np.nan)
                    row[f"{g}_s{s}"] = val
            detail_rows.append(row)
            
        df_detail = pd.DataFrame(detail_rows, columns=detail_cols)
        header_detail = pd.DataFrame([["" for _ in df_detail.columns]], columns=df_detail.columns)
        header_detail.iloc[0, 0] = f"--- [DETAIL] Individual Seeds (M={m}) ---"
        
        final_dfs.append(header_detail)
        final_dfs.append(df_detail)
        final_dfs.append(pd.DataFrame([["" for _ in df_detail.columns]], columns=df_detail.columns)) # Spacer

    # Concatenate all
    full_report = pd.concat(final_dfs, ignore_index=True)
    
    full_report.to_csv(output_file, index=False)
    print(f"Generalization Analysis complete! Results saved to {output_file}")

if __name__ == "__main__":
    analyze_generalization_csv()
