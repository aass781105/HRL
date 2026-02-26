
import os
import pandas as pd
import glob
import re
import numpy as np

def analyze_benchmark_csv():
    results_dir = 'test_results_benchmark'
    output_file = 'Benchmark_Analysis_Summary.csv'
    
    # 1. Define Groups and their filename patterns
    groups = {
        'S1_Base': r'TestResult_curric_s1_baseline_seed\d',
        'S2_Size': r'TestResult_curric_s2_size_seed\d',
        'S3_SizeM':r'TestResult_curric_s3_size_m_seed\d',
        'S4_Noise':r'TestResult_curric_s4_noise_shift_seed\d'
    }
    
    group_order = ['S1_Base', 'S2_Size', 'S3_SizeM', 'S4_Noise']

    # 2. Hurink vdata mapping: Instance_ID (0-59) -> Job Size
    # Pattern: 5 instances per size
    size_names = [
        "10x5", "15x5", "20x5", "10x10", "15x10", "20x10",
        "30x10", "15x15", "20x15", "30x15", "15x20", "20x20"
    ]
    id_to_size = {}
    for i in range(60):
        id_to_size[i] = size_names[i // 5]

    # 3. Data Collection
    raw_data = {
        '0.9': {g: {s: [] for s in size_names} for g in group_order},
        '1.2': {g: {s: [] for s in size_names} for g in group_order},
        '1.5': {g: {s: [] for s in size_names} for g in group_order}
    }
    
    # Detail data structure: {Factor: {Group: {SeedID: {Size: Value}}}}
    detail_data = {
        '0.9': {g: {} for g in group_order},
        '1.2': {g: {} for g in group_order},
        '1.5': {g: {} for g in group_order}
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
            
        # Extract Seed ID
        seed_match = re.search(r'_seed(\d+)', filename)
        seed_id = int(seed_match.group(1)) if seed_match else 1
            
        try:
            df = pd.read_csv(file_path)
            # Remove 'Average' row if exists
            df = df[df['Instance_ID'].apply(lambda x: str(x).isdigit())].copy()
            df['Instance_ID'] = df['Instance_ID'].astype(int)
            
            # Map ID to Size
            df['Size'] = df['Instance_ID'].map(id_to_size)
            
            # Group by Size and take mean (of 5 instances)
            size_summary = df.groupby('Size').mean(numeric_only=True)
            
            # Extract Objectives for each factor
            for factor in ['0.9', '1.2', '1.5']:
                col_name = f'Objective_{factor}'
                if col_name in size_summary.columns:
                    for size in size_names:
                        if size in size_summary.index:
                            val = size_summary.loc[size, col_name]
                            raw_data[factor][matched_group][size].append(val)
                            
                            # Store Detail
                            if seed_id not in detail_data[factor][matched_group]:
                                detail_data[factor][matched_group][seed_id] = {}
                            detail_data[factor][matched_group][seed_id][size] = val
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 4. Aggregate Seeds and Format Output
    final_dfs = []
    
    for factor in ['0.9', '1.2', '1.5']:
        # Create table for this factor
        factor_rows = []
        for size in size_names:
            row = {'Job_Size': size}
            for g_name in group_order:
                vals = raw_data[factor][g_name][size]
                if vals:
                    row[g_name] = np.mean(vals)
                else:
                    row[g_name] = np.nan
            factor_rows.append(row)
        
        df_factor = pd.DataFrame(factor_rows)
        
        header = pd.DataFrame([["" for _ in df_factor.columns]], columns=df_factor.columns)
        header.iloc[0, 0] = f"--- [MEAN] Objective k = {factor} ---"
        
        final_dfs.append(header)
        final_dfs.append(df_factor)
        final_dfs.append(pd.DataFrame([["" for _ in df_factor.columns]], columns=df_factor.columns)) # Empty row spacer

        # [NEW] Generate STD Table for this factor
        std_rows = []
        for size in size_names:
            row = {'Job_Size': size}
            for g_name in group_order:
                vals = raw_data[factor][g_name][size]
                if vals:
                    row[g_name] = np.std(vals)
                else:
                    row[g_name] = np.nan
            std_rows.append(row)
            
        df_std = pd.DataFrame(std_rows)
        std_header = pd.DataFrame([["" for _ in df_std.columns]], columns=df_std.columns)
        std_header.iloc[0, 0] = f"--- [STD] Objective k = {factor} ---"
        
        final_dfs.append(std_header)
        final_dfs.append(df_std)
        final_dfs.append(pd.DataFrame([["" for _ in df_std.columns]], columns=df_std.columns))

        # [NEW] Generate GAP Table for this factor
        baseline_col = 'S1_Base'
        if baseline_col in df_factor.columns:
            gap_rows = []
            for _, row in df_factor.iterrows():
                gap_row = {'Job_Size': row['Job_Size']}
                baseline_val = row[baseline_col]
                
                for col in df_factor.columns:
                    if col == 'Job_Size': continue
                    if col == baseline_col:
                        gap_row[col] = "0.00%" # Baseline is 0% gap
                    else:
                        val = row[col]
                        if pd.notna(val) and pd.notna(baseline_val) and baseline_val != 0:
                            gap = ((val - baseline_val) / baseline_val) * 100
                            gap_row[col] = f"{gap:+.2f}%"
                        else:
                            gap_row[col] = "N/A"
                gap_rows.append(gap_row)
            
            df_gap = pd.DataFrame(gap_rows)
            
            # Add header for GAP section
            gap_header = pd.DataFrame([["" for _ in df_gap.columns]], columns=df_gap.columns)
            gap_header.iloc[0, 0] = f"--- [GAP] vs S1_Base k = {factor} ---"
            
            final_dfs.append(gap_header)
            final_dfs.append(df_gap)
            final_dfs.append(pd.DataFrame([["" for _ in df_gap.columns]], columns=df_gap.columns)) # Empty row spacer

        # --- [NEW] Table 4: DETAIL (Individual Seeds) ---
        detail_rows = []
        
        # Prepare columns: Job_Size, S1_s1, S1_s2...
        detail_cols = ['Job_Size']
        for g in group_order:
            for s in range(1, 6): # Seeds 1-5
                detail_cols.append(f"{g}_s{s}")
        
        for size in size_names:
            row = {'Job_Size': size}
            for g in group_order:
                for s in range(1, 6):
                    val = detail_data[factor][g].get(s, {}).get(size, np.nan)
                    row[f"{g}_s{s}"] = val
            detail_rows.append(row)
            
        df_detail = pd.DataFrame(detail_rows, columns=detail_cols)
        header_detail = pd.DataFrame([["" for _ in df_detail.columns]], columns=df_detail.columns)
        header_detail.iloc[0, 0] = f"--- [DETAIL] Individual Seeds k = {factor} ---"
        
        final_dfs.append(header_detail)
        final_dfs.append(df_detail)
        final_dfs.append(pd.DataFrame([["" for _ in df_detail.columns]], columns=df_detail.columns)) # Spacer

    # Concatenate all sections
    full_report = pd.concat(final_dfs, ignore_index=True)
    
    # Save to CSV
    full_report.to_csv(output_file, index=False)
    print(f"Benchmark Analysis complete! Results saved to {output_file}")

if __name__ == "__main__":
    analyze_benchmark_csv()
