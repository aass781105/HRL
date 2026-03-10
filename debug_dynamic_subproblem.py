
import numpy as np
import torch
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from model.PPO import PPO_initialize
from params import configs

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed at: {seed}")

def simulate_dynamic_subproblem():
    # 0. 固定種子
    set_seed(42)

    # 1. 配置設備與模型 (支援 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    print(f"Using device: {device}")
    
    n_j = 10
    n_m = 5
    mean_pt = 50.0 # (0+100)/2
    pt_scale = mean_pt
    
    # 初始化 PPO
    ppo = PPO_initialize()
    if os.path.exists(configs.ppo_model_path):
        ppo.policy.load_state_dict(torch.load(configs.ppo_model_path, map_location=device, weights_only=True))
        print(f"Loaded PPO model from {configs.ppo_model_path}")
    ppo.policy.to(device)
    ppo.policy.eval()

    # 2. 生成模擬數據 (符合您的 Uniform 分佈需求)
    # Job Lengths: 1~5 ops
    jl = [np.random.randint(1, 6) for _ in range(n_j)]
    num_total_ops = sum(jl)
    
    # PT Matrix: [TotalOps, M] ~ Uniform(0, 100)
    pt = np.random.uniform(0, 100, (num_total_ops, n_m))
    mask = np.random.rand(*pt.shape) < 0.2
    pt[mask] = 0
    for i in range(num_total_ops):
        if np.all(pt[i] == 0): pt[i, np.random.randint(0, n_m)] = mean_pt

    # Due Dates: 生成原始值後執行移動
    due_range = n_j * mean_pt
    orig_due_dates = np.random.uniform(-due_range, due_range, n_j)
    # [FIXED] 基於生成後的固定值做 (x/2)+200 的移動
    due_dates_abs = (orig_due_dates / 8.0) + 300.0

    # Machine Free Time (MFT): Uniform(0, 100)
    mft_abs = np.random.uniform(0, 100, n_m)

    # 3. 初始化靜態環境
    env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
    due_dates_ppo = due_dates_abs / pt_scale
    
    state = env.set_initial_data(
        job_length_list=[np.array(jl)], 
        op_pt_list=[pt], 
        due_date_list=[np.array(due_dates_ppo)], 
        normalize_due_date=False, 
        true_due_date_list=[np.array(due_dates_abs)]
    )

    # 4. 【時間校正】模擬動態環境轉換
    env.true_mch_free_time[0] = mft_abs
    env.mch_free_time[0] = mft_abs / pt_scale
    env.true_candidate_free_time[0] = 0.0
    env.candidate_free_time[0] = 0.0

    # 重建狀態
    state = env.rebuild_state_from_current()

    # 5. 開始 PPO 排程迴圈
    print("\n--- Starting PPO Scheduling ---")
    schedule_details = []
    done = False
    step = 0
    
    output_dir = "debug_dynamic_output"
    states_dir = os.path.join(output_dir, "step_states")
    os.makedirs(states_dir, exist_ok=True)

    # 特徵名稱清單 (14維)
    feature_names = [
        "Scheduled_Flag", "CT_Lower_Bound", "Min_PT", "PT_Span", "Mean_PT",
        "Waiting_Time", "Remain_Work_Op", "Job_Left_Ops", "Job_Remain_Work",
        "Avail_Mch_Ratio", "Rem_Due_Time", "Slack", "CR_Log", "Is_Tardy"
    ]

    # 儲存已完工工序的歷史資訊
    scheduled_history = {} # Key: (job_id, op_id)

    while not done:
        # 準備特徵數據
        norm_features = env.fea_j[0] 
        raw_features = env.raw_fea_j[0]
        candidates = env.candidate[0]
        
        # 預先計算 PPO 決策
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

        # 執行動作獲取結果
        state, reward, done, info = env.step(np.array([action]))
        det = info["scheduled_op_details"]
        
        # 更新歷史記錄
        jid = det['job_id']
        oid = det['op_id_in_job']
        is_final = (oid == jl[jid] - 1)
        scheduled_history[(jid, oid)] = {
            "machine_id": det["machine_id"],
            "start": round(det["start_time"], 2),
            "end": round(det["end_time"], 2),
            "tardiness": round(max(0, det['end_time'] - due_dates_abs[jid]), 2) if is_final else 0.0,
            "at_step": step # [NEW] 記錄是哪個 step 被排的
        }

        def save_enhanced_state_csv(feat_matrix, suffix):
            rows = []
            for j_idx in range(n_j):
                for o_idx in range(jl[j_idx]):
                    op_g_idx = env.job_first_op_id[0, j_idx] + o_idx
                    is_cand = (op_g_idx == candidates[j_idx])
                    is_chosen = (op_g_idx == chosen_op_global_idx)
                    
                    row = {
                        "job_id": j_idx,
                        "op_id": o_idx,
                        "op_global_idx": op_g_idx,
                        "is_candidate": is_cand,
                        "is_chosen": is_chosen,
                        "due_date": round(due_dates_abs[j_idx], 2),
                    }
                    
                    # 檢查歷史記錄或當前選中項
                    hist_data = scheduled_history.get((j_idx, o_idx))
                    if hist_data:
                        row["machine_id"] = hist_data["machine_id"]
                        row["start"] = hist_data["start"]
                        row["end"] = hist_data["end"]
                        row["step_tardiness"] = hist_data["tardiness"]
                        row["scheduled_at_step"] = hist_data["at_step"] # [NEW]
                    else:
                        row["machine_id"] = ""
                        row["start"] = ""
                        row["end"] = ""
                        row["step_tardiness"] = ""
                        row["scheduled_at_step"] = "" # [NEW]

                    # 加入 14 維特徵
                    for d, name in enumerate(feature_names):
                        row[name] = feat_matrix[op_g_idx, d]
                    rows.append(row)
            
            pd.DataFrame(rows).to_csv(os.path.join(states_dir, f"step{step:03d}_{suffix}.csv"), index=False)

        save_enhanced_state_csv(norm_features, "normalized")
        save_enhanced_state_csv(raw_features, "raw")

        # 收集匯總資訊
        schedule_details.append({
            "step": step,
            "job_id": chosen_job_idx,
            "op_id": det['op_id_in_job'],
            "machine_id": det['machine_id'],
            "start_time": round(det['start_time'], 2),
            "end_time": round(det['end_time'], 2),
            "proc_time": round(det['proc_time'], 2),
            "due_date": round(due_dates_abs[chosen_job_idx], 2),
            "tardiness": round(scheduled_history[(chosen_job_idx, det['op_id_in_job'])]["tardiness"], 2),
            "is_final_op": (det['op_id_in_job'] == jl[chosen_job_idx] - 1)
        })
        
        step += 1

    # 6. 輸出結果與 CSV
    final_mk = env.current_makespan[0]
    total_td = env.accumulated_tardiness[0]
    print(f"\n--- Final Results ---")
    print(f"Makespan (MK): {final_mk:.2f}")
    print(f"Total Tardiness (TD): {total_td:.2f}")

    output_dir = "debug_dynamic_output"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(schedule_details)
    csv_name = os.path.join(output_dir, "schedule_details.csv")
    df.to_csv(csv_name, index=False)
    print(f"Schedule details exported to '{csv_name}'")

    # 7. 繪製甘特圖
    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, n_j)]
        
        for _, r in df.iterrows():
            ax.barh(r['machine_id'], r['end_time'] - r['start_time'], left=r['start_time'], color=colors[int(r['job_id'])], edgecolor='black', alpha=0.8)
            ax.text(r['start_time'] + (r['end_time'] - r['start_time'])/2, r['machine_id'], f"J{int(r['job_id'])}\nOp{int(r['op_id'])}", ha='center', va='center', fontsize=7)
        
        for i, d in enumerate(due_dates_abs):
            ax.axvline(x=d, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.6)
            # [FIXED] 移除 d > 0 的限制，讓負值交期也能標註 D{i}
            ax.text(d, -0.5, f"D{i}", color=colors[i], rotation=90, fontsize=8, fontweight='bold')
        
        ax.set_xlabel("Time (Minutes)")
        ax.set_ylabel("Machine ID")
        ax.set_title(f"Dynamic Subproblem Simulation (Seed=42)\nMK={final_mk:.1f}, TD={total_td:.1f}")
        ax.set_yticks(range(n_m))
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        png_name = os.path.join(output_dir, "gantt_chart.png")
        plt.savefig(png_name)
        print(f"Gantt chart saved as '{png_name}'")
    except Exception as e:
        print(f"Gantt plotting failed: {e}")

if __name__ == "__main__":
    simulate_dynamic_subproblem()
