
import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from params import configs
from model.PPO import PPO_initialize
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from data_utils import text_to_matrix

def evaluate_on_stored_instances(base_dir="or_instances_uniform"):
    # 1. 設備與模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    
    ppo = PPO_initialize()
    if os.path.exists(configs.ppo_model_path):
        ppo.policy.load_state_dict(torch.load(configs.ppo_model_path, map_location=device, weights_only=True))
        print(f"Loaded PPO model from {configs.ppo_model_path}")
    ppo.policy.to(device)
    ppo.policy.eval()

    results = []
    pt_scale = (float(configs.low) + float(configs.high)) / 2.0

    # 2. 走訪目錄
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found scales: {subdirs}")

    for scale in subdirs:
        curr_path = os.path.join(base_dir, scale)
        fjs_files = [f for f in os.listdir(curr_path) if f.endswith(".fjs")]
        
        print(f"\nEvaluating {scale} ({len(fjs_files)} instances)...")

        for fjs_name in tqdm(fjs_files):
            base_name = fjs_name.replace(".fjs", "")
            json_name = f"{base_name}.json"
            json_path = os.path.join(curr_path, json_name)
            fjs_path = os.path.join(curr_path, fjs_name)

            if not os.path.exists(json_path):
                continue

            # A. 讀取 FJS 數據
            with open(fjs_path, "r") as f:
                content = f.readlines()
            jl, pt = text_to_matrix(content)

            # B. 讀取 JSON 交期
            with open(json_path, "r") as f:
                due_data = json.load(f)
            due_dates_abs = np.array(due_data["due_dates"])

            # C. 初始化環境並排程
            n_j = jl.shape[0]
            n_m = pt.shape[1]
            env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
            
            # 手動縮放交期以對齊 PPO 特徵
            due_dates_ppo = due_dates_abs / pt_scale
            
            state = env.set_initial_data(
                job_length_list=[jl], 
                op_pt_list=[pt], 
                due_date_list=[due_dates_ppo], 
                normalize_due_date=False, 
                true_due_date_list=[due_dates_abs]
            )

            done = False
            while not done:
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor
                    )
                    action = int(pi.argmax(dim=1).item())
                state, _, done, _ = env.step(np.array([action]))

            # D. 記錄結果
            results.append({
                "scale": scale,
                "instance": base_name,
                "makespan": round(float(env.current_makespan[0]), 2),
                "total_tardiness": round(float(env.accumulated_tardiness[0]), 2)
            })

    # 3. 輸出匯總與 CSV
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    model_filename = os.path.basename(configs.ppo_model_path).replace(".pth", "")
    detail_csv = os.path.join(output_dir, f"ppo_static_bench_{model_filename}_details.csv")
    summary_csv = os.path.join(output_dir, f"ppo_static_bench_{model_filename}_summary.csv")
    
    df = pd.DataFrame(results)
    df["ppo_model"] = configs.ppo_model_path
    df.to_csv(detail_csv, index=False)
    
    # 計算匯總數據
    summary_df = df.groupby("scale").agg({"makespan": "mean", "total_tardiness": "mean"}).reset_index()
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n--- Evaluation Summary (Model: {model_filename}) ---")
    print(summary_df)
    print(f"\nDetailed results saved to: {detail_csv}")
    print(f"Summary results saved to: {summary_csv}")

if __name__ == "__main__":
    evaluate_on_stored_instances()
