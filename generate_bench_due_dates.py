
import os
import json
import numpy as np
from data_utils import text_to_matrix, generate_due_dates

def generate_due_for_bench(bench_dir="data/BenchData/Hurink_vdata", output_dir="or_instances_la_k"):
    # 1. 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 獲取所有 FJS 檔案
    fjs_files = [f for f in os.listdir(bench_dir) if f.endswith(".fjs")]
    print(f"Found {len(fjs_files)} benchmark files.")

    # 3. 循環生成交期 (K-mode)
    # 我們這裡固定 k = 1.2 作為基準 (您可以調整)
    k_value = 1.2
    
    for f_name in fjs_files:
        f_path = os.path.join(bench_dir, f_name)
        
        # 讀取並轉換為矩陣以計算工時
        with open(f_path, "r") as f:
            content = f.readlines()
        jl, pt = text_to_matrix(content)
        
        # 使用 'k' 模式生成交期
        # 注意：seed 固定為 42 確保基準一致
        due_dates = generate_due_dates(jl, pt, tightness=k_value, due_date_mode='k', seed=42)
        
        # 儲存對應的 JSON
        base_name = f_name.replace(".fjs", "")
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump({"due_dates": due_dates.tolist()}, f, indent=4)
            
        # 同步複製 FJS 到新目錄方便 evaluate_static_batch 讀取
        import shutil
        shutil.copy(f_path, os.path.join(output_dir, f_name))

    print(f"\nBenchmark instances with K-mode due dates (k={k_value}) prepared in {output_dir}/")

if __name__ == "__main__":
    generate_due_for_bench()
