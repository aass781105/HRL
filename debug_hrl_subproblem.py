
import os
import time
import json
import random
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from params import configs
from model.PPO import PPO_initialize
from common_utils import greedy_select_action, setup_seed
from data_utils import SD2_instance_generator, generate_due_dates, matrix_to_text
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums

def main():
    # 0. 初始化配置
    output_dir = "debug_final_output"
    states_dir = os.path.join(output_dir, "step_states")
    os.makedirs(states_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    seed = 42
    setup_seed(seed)
    
    # 1. 模擬動態環境參數
    n_j = 10
    n_m = 5
    configs.n_j = n_j
    configs.n_m = n_m
    mean_pt = (configs.low + configs.high) / 2.0
    pt_scale = mean_pt

    # [寫死] 機器可用時間 (MFT) - 模擬先前工序的佔用
    # mft_abs = np.array([0.0, 50.0, 120.0, 30.0, 80.0])
    mft_abs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"--- Unified Debug Script (Dynamic Subproblem) ---")
    print(f"Device: {device} | Seed: {seed}")
    print(f"MFT (Static): {mft_abs}")

    # 2. 生成實例 (使用 Dynamic 邏輯)
    # 100% Uniform 模式
    jl, pt, _ = SD2_instance_generator(configs, seed=seed, mode='uniform')
    # 使用 'range' 模式模擬負值交期
    due_dates_abs = generate_due_dates(jl, pt, due_date_mode='range', seed=seed)
    
    # 3. 初始化環境
    env = FJSPEnvForVariousOpNums(n_j, n_m)
    due_dates_ppo = due_dates_abs / pt_scale
    
    # 初始化資料
    state = env.set_initial_data(
        job_length_list=[jl], 
        op_pt_list=[pt], 
        due_date_list=[due_dates_ppo], 
        normalize_due_date=False, 
        true_due_date_list=[due_dates_abs]
    )

    # 【時間校正】覆寫環境狀態
    env.true_mch_free_time[0] = mft_abs
    env.mch_free_time[0] = mft_abs / pt_scale
    env.true_candidate_free_time[0] = 0.0 # 假設工單都已就緒
    env.candidate_free_time[0] = 0.0
    
    # 重建狀態 (更新 op_ct_lb 等)
    state = env.rebuild_state_from_current()

    # 4. 載入模型
    ppo = PPO_initialize()
    if os.path.exists(configs.ppo_model_path):
        ppo.policy.load_state_dict(torch.load(configs.ppo_model_path, map_location=device, weights_only=True))
        print(f"Loaded PPO model from {configs.ppo_model_path}")
    ppo.policy.to(device)
    ppo.policy.eval()

    # 5. 排程循環
    schedule_details = []
    scheduled_history = {}
    feature_names = ["Scheduled_Flag", "CT_Lower_Bound", "Min_PT", "PT_Span", "Mean_PT", "Waiting_Time", "Remain_Work_Op", "Job_Left_Ops", "Job_Remain_Work", "Avail_Mch_Ratio", "Rem_Due_Time", "Slack", "CR_Log", "Is_Tardy"]
    
    step = 0
    done = False
    print("Starting simulation loop...")

    while not done:
        # A. 捕獲當前狀態
        norm_features = env.fea_j[0]
        raw_features = env.raw_fea_j[0]
        candidates = env.candidate[0]

        # B. 模型決策
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor
            )
            action = int(pi.argmax(dim=1).item())
        
        chosen_job_idx = action // n_m
        chosen_mch_idx = action % n_m
        chosen_op_global_idx = int(env.candidate[0, chosen_job_idx])

        # C. 執行動作
        state, reward, done, info = env.step(np.array([action]))
        det = info["scheduled_op_details"]
        
        # D. 更新歷史與 CSV 記錄 (整合兩者優點)
        jid = det['job_id']
        oid = det['op_id_in_job']
        is_final = (oid == jl[jid] - 1)
        scheduled_history[(jid, oid)] = {
            "machine": det["machine_id"], "start": round(det["start_time"], 2), "end": round(det["end_time"], 2),
            "td": round(max(0, det['end_time'] - due_dates_abs[jid]), 2) if is_final else 0.0,
            "step": step
        }

        def save_state_csv(feat_matrix, suffix):
            rows = []
            for j in range(n_j):
                for o in range(jl[j]):
                    op_g = env.job_first_op_id[0, j] + o
                    is_chosen = (op_g == chosen_op_global_idx)
                    hist = scheduled_history.get((j, o))
                    
                    row = {"job_id": j, "op_id": o, "is_candidate": (op_g == candidates[j]), "is_chosen": is_chosen, "due_date": round(due_dates_abs[j], 2)}
                    if hist:
                        row.update({"mch": hist["machine"], "start": hist["start"], "end": hist["end"], "td": hist["td"], "at_step": hist["step"]})
                    else:
                        row.update({"mch": "", "start": "", "end": "", "td": "", "at_step": ""})
                    
                    for d, name in enumerate(feature_names): row[name] = feat_matrix[op_g, d]
                    rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(states_dir, f"step{step:03d}_{suffix}.csv"), index=False)

        save_state_csv(norm_features, "normalized")
        save_state_csv(raw_features, "raw")

        # 匯總記錄
        schedule_details.append({
            "step": step, "job": jid, "op": oid, "mch": det['machine_id'],
            "start": round(det['start_time'], 2), "end": round(det['end_time'], 2),
            "due": round(due_dates_abs[jid], 2), "td": scheduled_history[(jid, oid)]["td"],
            "reward_mk": round(info['reward_mk'][0], 4), "reward_td": round(info['reward_td'][0], 4)
        })
        step += 1

    # 6. 產出匯總結果
    df_summary = pd.DataFrame(schedule_details)
    df_summary.to_csv(os.path.join(output_dir, "final_schedule_summary.csv"), index=False)
    
    print(f"\n--- Simulation Complete ---")
    print(f"Final Makespan: {env.current_makespan[0]:.2f}")
    print(f"Total Tardiness: {env.accumulated_tardiness[0]:.2f}")
    print(f"Results saved to: {output_dir}/")

    # 7. 繪製甘特圖
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, n_j)]
    for _, r in df_summary.iterrows():
        ax.barh(r['mch'], r['end'] - r['start'], left=r['start'], color=colors[int(r['job'])], edgecolor='black', alpha=0.8)
        ax.text(r['start'] + (r['end'] - r['start'])/2, r['mch'], f"J{int(r['job'])}\nOp{int(r['op'])}", ha='center', va='center', fontsize=7)
    for i, d in enumerate(due_dates_abs):
        ax.axvline(x=d, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.6)
        ax.text(d, -0.5, f"D{i}", color=colors[i], rotation=90, fontsize=8, fontweight='bold')
    ax.set_title(f"Unified Debug (Seed={seed})\nMK={env.current_makespan[0]:.1f}, TD={env.accumulated_tardiness[0]:.1f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_gantt.png"))

if __name__ == '__main__':
    main()
