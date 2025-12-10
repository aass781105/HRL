# evaluate_static.py

import os
import random
import numpy as np
import torch
import csv
import copy
from typing import List, Dict, Any

# --- 硬編碼參數，可在此修改 ---
NUM_JOBS = 50
NUM_MACHINES = 5
RANDOM_SEED = 42  # 用於生成固定的測試問題，確保可複現
# MODEL_PATH = r"cotrain_network\curriculum_train_10x5+mix.pth" # <<< 請務必換成您要評估的模型路徑
MODEL_PATH = r"trained_network\unified\unified_ppo_final_100ep_train_from0.pth"
# MODEL_PATH = r"trained_network\unified\unified_ppo_final_100ep_load_ppo.pth"
# --- 輸出檔案名稱 ---
# name = "normal50"
# name = "unified50"
name = "from0"
OUTPUT_CSV_PATH = f"static_schedule_result_{name}.csv"
GANTT_CHART_PATH = f"plots/static_gantt_chart_{name}.png"

# --- 導入項目 ---
from params import configs # 用於 SD2_instance_generator 的配置
from data_utils import SD2_instance_generator
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from model.PPO import PPO # PPO 模型的載入器 (用於靜態模型)
import common_utils # 貪婪策略選擇動作
import gantt as gantt_chart_maker # 繪製甘特圖



def evaluate():
    # --- Seeding for reproducibility ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 資料生成 ---
    # SD2_instance_generator 需要一個類似 configs 的物件
    class ConfigsStub:
        def __init__(self, n_j, n_m, seed):
            self.n_j = n_j
            self.n_m = n_m
            self.seed = seed
            self.op_pt_type = 'SD' # 假設 SD2 類型
            self.m_max = n_m # For SD2_instance_generator
            self.j_max = n_j # For SD2_instance_generator
            self.low = 1
            self.high = 99
            self.scale = 1 # For SD2_instance_generator

    stub_configs = ConfigsStub(NUM_JOBS, NUM_MACHINES, RANDOM_SEED)
    job_lengths, op_pts, _ = SD2_instance_generator(stub_configs, RANDOM_SEED)
    
    # SD2_instance_generator 返回單一實例的 np.ndarray，但環境 `set_initial_data` 期望批次化的 List[np.ndarray]
    # 因此我們將返回的 ndarray 包裝在一個 list 中，形成一個大小為 1 的批次
    job_lengths_list = [job_lengths]
    op_pts_list = [op_pts]

    # --- 2. 環境初始化 ---
    env = FJSPEnvForVariousOpNums(n_j=NUM_JOBS, n_m=NUM_MACHINES)
    state = env.set_initial_data(job_lengths_list, op_pts_list)

    # --- 3. 模型載入與設定 ---
    # 為了讓 PPO 模型載入時 configs 有值，這裡暫時將 params.configs 傳入
    # 實際應用中，如果模型需要特定 config，應從模型訓練時的 config 載入
    from params import configs as global_configs_for_ppo_model # 導入 params.py 中的 configs
    ppo_model = PPO(global_configs_for_ppo_model) 
    
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤: 模型檔案未找到於 {MODEL_PATH}")
        return

    ppo_model.policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    ppo_model.policy.eval() # 設定為評估模式

    # --- 4. 排程循環與記錄 ---
    schedule_records: List[Dict[str, Any]] = []
    
    done = np.array([False]) # env.done_flag 是 numpy array
    while not done.all(): # 當所有環境都完成時 (對於單一環境，即 done[0] 為 True)
        with torch.no_grad():
            # 將 state 中的 tensor 直接移動到指定設備
            fea_j_tensor = state.fea_j_tensor.float().to(device)
            op_mask_tensor = state.op_mask_tensor.bool().to(device)
            candidate_tensor = state.candidate_tensor.long().to(device)
            fea_m_tensor = state.fea_m_tensor.float().to(device)
            mch_mask_tensor = state.mch_mask_tensor.bool().to(device)
            comp_idx_tensor = state.comp_idx_tensor.bool().to(device)
            dynamic_pair_mask_tensor = state.dynamic_pair_mask_tensor.bool().to(device)
            fea_pairs_tensor = state.fea_pairs_tensor.float().to(device)

            pi, _ = ppo_model.policy(
                fea_j=fea_j_tensor, op_mask=op_mask_tensor,
                candidate=candidate_tensor, fea_m=fea_m_tensor,
                mch_mask=mch_mask_tensor, comp_idx=comp_idx_tensor,
                dynamic_pair_mask=dynamic_pair_mask_tensor, fea_pairs=fea_pairs_tensor
            )
            
            # 使用貪婪策略選擇動作
            # pi.squeeze(0) 因為環境是 batch of 1，模型的輸出 pi 會多一個 batch 維度
            action_ppo = common_utils.greedy_select_action(pi.squeeze(0)) 
        
        # 執行動作，獲取排程細節
        next_state, reward, done, info = env.step(action_ppo.cpu().numpy())
        
        # 將排程細節加入記錄
        if 'scheduled_op_details' in info:
            schedule_records.append(info['scheduled_op_details'])
        
        state = next_state
    
    # --- 5. 結果匯出 ---
    # 獲取最終 Makespan
    final_makespan = float(env.current_makespan[0]) # 對於單一環境

    # 寫入 CSV
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['job_id', 'op_id_in_job', 'op_global_id', 'machine_id', 
                      'start_time', 'end_time', 'proc_time', 'makespan_of_problem']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in schedule_records:
            record_with_makespan = record.copy()
            record_with_makespan['makespan_of_problem'] = final_makespan
            writer.writerow(record_with_makespan)
    print(f"排程結果已寫入 {OUTPUT_CSV_PATH}，最終 Makespan: {final_makespan:.2f}")

    # --- 6. 繪製甘特圖 ---
    # gantt.py 的繪圖函數可能需要特定的數據格式
    # 這裡假設它能直接接受 schedule_records 或類似的 list of dicts
    # 需要將 schedule_records 轉換為 gantt.py 期望的格式
    
    gantt_data = []
    for record in schedule_records:
        gantt_data.append({
            'job': record['job_id'],
            'op': record['op_id_in_job'], # 增加 'op' 鍵
            'machine': record['machine_id'],
            'start': record['start_time'],
            'end': record['end_time']
        })

    # 呼叫 gantt.py 中正確的繪圖函數
    gantt_chart_maker.plot_global_gantt(gantt_data, GANTT_CHART_PATH)
    print(f"甘特圖已儲存為 {GANTT_CHART_PATH}")


if __name__ == "__main__":
    print(f"開始評估靜態 FJSP 問題 (J={NUM_JOBS}, M={NUM_MACHINES})...")
    evaluate()
    print("評估完成。")
