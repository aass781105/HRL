
import os
import json
import numpy as np
import copy
from data_utils import SD2_instance_generator, generate_due_dates, matrix_to_text
from params import configs

def generate_custom_or_batch(base_dir="or_instances_uniform"):
    # 設定目標規模與數量
    target_configs = [
        (10, 5),
        (20, 5),
        (30, 5)
    ]
    n_instances_per_config = 10
    due_mode = "range" # 使用您要求的 range 模式
    
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Starting batch generation (Mode: PURE UNIFORM)")
    print(f"One JSON due-date file per instance.")

    for nj, nm in target_configs:
        config_name = f"{nj}x{nm}"
        output_dir = os.path.join(base_dir, config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing {config_name}...")

        # 更新 configs 物件以符合當前規模
        configs.n_j = nj
        configs.n_m = nm

        for i in range(n_instances_per_config):
            instance_id = f"{i+1:03d}"
            base_filename = f"instance_{config_name}_{instance_id}"
            
            # 1. 生成物理數據 - 100% Uniform
            jl, pt, op_per_mch = SD2_instance_generator(configs, seed=200+i, mode='uniform')
            
            # 2. 生成交期數據 - 使用 'range' 模式
            due_dates = generate_due_dates(jl, pt, due_date_mode=due_mode, seed=200+i)
            
            # 3. 儲存 .fjs 檔案
            fjs_path = os.path.join(output_dir, f"{base_filename}.fjs")
            text_lines = matrix_to_text(jl, pt, op_per_mch)
            with open(fjs_path, "w") as f:
                for line in text_lines:
                    f.write(line + "\n")
            
            # 4. 儲存專屬 .json 交期檔
            json_path = os.path.join(output_dir, f"{base_filename}.json")
            # 存儲格式：{"due_dates": [d1, d2, ...]}
            with open(json_path, "w") as f:
                json.dump({"due_dates": due_dates.tolist()}, f, indent=4)
        
        print(f"  - Generated 10 pairs of (.fjs, .json) files in {output_dir}")

    print(f"\nBatch generation complete!")

if __name__ == "__main__":
    generate_custom_or_batch()
