
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from params import configs
from model.PPO import PPO_initialize
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from data_utils import text_to_matrix, generate_due_dates

def evaluate_vdata():
    # 1. 配置與路徑
    vdata_dir = "data/BenchData/Hurink_vdata"
    model_dir = "trained_network/SD2"
    output_base_dir = "vdata_evaluation_results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    pt_scale = (float(configs.low) + float(configs.high)) / 2.0

    # 搜尋所有目標模型 (SAME, S2, S3 * Seed 1,2,3)
    strategies = ['same', 's2', 's3']
    seeds = [1, 2, 3]
    model_paths = []
    for strat in strategies:
        for seed in seeds:
            m_name = f"{strat}_seed{seed}_1500.pth"
            path = os.path.join(model_dir, m_name)
            if os.path.exists(path):
                model_paths.append(path)

    if not model_paths:
        print(f"No models found in {model_dir}. Please check filenames.")
        return

    fjs_files = [f for f in os.listdir(vdata_dir) if f.endswith(".fjs")]
    print(f"Found {len(model_paths)} models and {len(fjs_files)} vdata instances.")

    # 2. 迴圈測試每個模型
    all_details = []
    
    for m_path in model_paths:
        # Extract Strategy and Seed from filename (e.g., s3_seed2_1500.pth)
        fname = os.path.basename(m_path)
        match = re.search(r"(same|s2|s3)_seed(\d)_", fname)
        if not match: continue
        strategy = match.group(1).upper()
        seed = int(match.group(2))
        
        print(f"\n>>> Evaluating Model: {strategy} Seed {seed}")
        
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
        ppo.policy.to(device)
        ppo.policy.eval()

        for f_name in tqdm(fjs_files, desc="Instances"):
            f_path = os.path.join(vdata_dir, f_name)
            # A. 讀取與計算手動交期 (k=1.2)
            with open(f_path, "r") as f: content = f.readlines()
            jl, pt = text_to_matrix(content)
            n_j, n_m = jl.shape[0], pt.shape[1]
            scale_label = f"{n_j}x{n_m}"
            
            # [CUSTOM DUE DATE LOGIC]
            # Calculate total average work per job
            job_work = np.zeros(n_j)
            op_cursor = 0
            for j in range(n_j):
                for _ in range(jl[j]):
                    row = pt[op_cursor]
                    valid = row[row > 0]
                    if valid.size > 0: job_work[j] += np.mean(valid)
                    op_cursor += 1
            
            # Apply fixed k=1.2
            due_dates_abs = job_work * 1.2
            due_dates_ppo = due_dates_abs / pt_scale

            env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
            # [FIXED] Wrap all list parameters in lists to match multi-env logic
            state = env.set_initial_data(job_length_list=[jl], op_pt_list=[pt], 
                                         due_date_list=[due_dates_ppo], normalize_due_date=False, 
                                         true_due_date_list=[due_dates_abs])

            done = False
            while not done:
                with torch.no_grad():
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                                        fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
                    action = int(pi.argmax(dim=1).item())
                state, _, done, _ = env.step(np.array([action]))

            all_details.append({
                "Strategy": strategy,
                "Seed": seed,
                "Scale": scale_label,
                "Instance": f_name,
                "Makespan": float(env.current_makespan[0]),
                "Total_Tardiness": float(env.accumulated_tardiness[0])
            })

    # 3. 儲存全體明細
    df_full = pd.DataFrame(all_details)
    df_full.to_csv(os.path.join(output_base_dir, "vdata_full_details.csv"), index=False)

    # 4. 生成策略匯總 (平均 3 個 Seed)
    # Group by (Strategy, Scale)
    summary = df_full.groupby(["Strategy", "Scale"]).agg({
        "Makespan": "mean",
        "Total_Tardiness": "mean"
    }).reset_index()

    # 5. 自定義排序
    scale_order = {
        "10x5": 0, "15x5": 1, "20x5": 2,
        "10x10": 3, "15x10": 4, "20x10": 5, "30x10": 6,
        "15x15": 7
    }
    summary["Order"] = summary["Scale"].map(scale_order)
    # Handle scales not in list
    summary["Order"] = summary["Order"].fillna(99)
    summary = summary.sort_values(by=["Strategy", "Order"])
    summary = summary.drop(columns=["Order"])

    summary_path = os.path.join(output_base_dir, "vdata_strategy_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    
    print(f"\nValidation Breakdown by Strategy (Average of 3 Seeds):")
    print(summary.to_string())
    print(f"\nFinal strategy comparison saved to: {summary_path}")

import re # Need re at top level
if __name__ == "__main__":
    evaluate_vdata()

    print("\nAll VData evaluations complete!")

if __name__ == "__main__":
    evaluate_vdata()
