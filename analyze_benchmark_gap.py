
import pandas as pd
import os
import glob
import re
import numpy as np

def analyze_gap_pivot(results_dir="evaluation_results"):
    # 1. 載入 OR-Tools 基準數據
    or_path = os.path.join(results_dir, "or_tools_benchmark_results.csv")
    if not os.path.exists(or_path):
        print(f"Error: Base OR-Tools results not found at {or_path}")
        return
    
    or_df = pd.read_csv(or_path)
    or_df = or_df.rename(columns={'makespan': 'OR_MK', 'total_tardiness': 'OR_TD'})
    # 保留必要欄位供合併
    or_base = or_df[['scale', 'instance', 'OR_MK', 'OR_TD']]

    # 2. 搜尋並載入 PPO 所有結果
    ppo_files = glob.glob(os.path.join(results_dir, "ppo_static_bench_*_details.csv"))
    all_ppo_data = []
    for f in ppo_files:
        match = re.search(r"ppo_static_bench_(same|s2|s3)_seed(\d)_", os.path.basename(f))
        if match:
            strategy = match.group(1).upper()
            df = pd.read_csv(f)
            df['Strategy'] = strategy
            all_ppo_data.append(df)
    
    if not all_ppo_data:
        print("No matching PPO detailed results found.")
        return

    full_ppo_df = pd.concat(all_ppo_data)

    # 3. 按 (Scale, Instance, Strategy) 分組計算 3 個 Seed 的平均值
    # 我們這裡先拿到每個 (Scale, Instance, Strategy) 的平均物理值
    grouped = full_ppo_df.groupby(['scale', 'instance', 'Strategy']).agg({
        'makespan': 'mean',
        'total_tardiness': 'mean'
    }).reset_index()

    # 4. 合併 OR-Tools 基準
    merged = pd.merge(grouped, or_base, on=['scale', 'instance'], how='left')

    # 5. 計算 GAP: (PPO - OR) / OR (以小數呈現)
    def calc_gap(ppo_val, or_val):
        if or_val == 0:
            return 0.0 if ppo_val == 0 else np.nan
        return (ppo_val - or_val) / or_val

    merged['Gap_MK'] = merged.apply(lambda x: calc_gap(x['makespan'], x['OR_MK']), axis=1)
    merged['Gap_TD'] = merged.apply(lambda x: calc_gap(x['total_tardiness'], x['OR_TD']), axis=1)

    # 6. 【關鍵步驟】執行 Pivot (軸轉) 將策略轉為 Col
    # 我們需要轉出 MK Gap 和 TD Gap 兩類欄位
    pivot_mk = merged.pivot(index=['scale', 'instance', 'OR_MK', 'OR_TD'], 
                            columns='Strategy', values='Gap_MK').add_suffix('_Gap_MK')
    pivot_td = merged.pivot(index=['scale', 'instance', 'OR_MK', 'OR_TD'], 
                            columns='Strategy', values='Gap_TD').add_suffix('_Gap_TD')

    # 合併兩類 Gap 欄位
    final_pivot = pd.concat([pivot_mk, pivot_td], axis=1).reset_index()

    # 7. 整理欄位順序 (OR數據放在最前面)
    # 目標順序: Scale, Instance, OR_MK, OR_TD, SAME_Gap_MK, SAME_Gap_TD, S2_Gap_MK, S2_Gap_TD, S3_Gap_MK, S3_Gap_TD
    cols = ['scale', 'instance', 'OR_MK', 'OR_TD']
    # 動態獲取現有的策略 Gap 欄位
    gap_cols = sorted([c for c in final_pivot.columns if '_Gap_' in c])
    # 按照 SAME -> S2 -> S3 的邏輯排序 (如果都有的話)
    def strat_sort_key(name):
        if 'SAME' in name: return 0
        if 'S2' in name: return 1
        if 'S3' in name: return 2
        return 3
    gap_cols.sort(key=strat_sort_key)
    
    final_df = final_pivot[cols + gap_cols]
    
    # 8. 儲存與輸出
    output_file = os.path.join(results_dir, "pivot_gap_analysis.csv")
    final_df.round(4).to_csv(output_file, index=False)
    
    print(f"\nPivot Gap Analysis complete!")
    print(f"File saved to: {output_file}")
    print("\nPreview (First 5 rows):")
    print(final_df.head().to_string())

if __name__ == "__main__":
    analyze_gap_pivot()
