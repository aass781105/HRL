import os
import time
import pandas as pd
import torch
import numpy as np
from params import configs
from model.PPO import PPO_initialize
from common_utils import greedy_select_action, setup_seed
from data_utils import SD2_instance_generator, generate_due_dates
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums

def main():
    # Setup
    # configs.device = 'cpu' # Don't force CPU, let it follow params
    device = torch.device(configs.device)
    seed = 42
    setup_seed(seed)
    
    # Instance Config
    n_j = 20
    n_m = 5
    configs.n_j = n_j
    configs.n_m = n_m
    tightness = 1 # Hard
    
    print(f"Generating Debug Instance: {n_j}x{n_m}, k={tightness}")
    jl, pt, _ = SD2_instance_generator(configs, seed=seed)
    dd = generate_due_dates(jl, pt, tightness=tightness)
    
    # [DEBUG] Verify Total Work vs Due Date
    print("\n[Data Debug]")
    op_cursor = 0
    for j in range(n_j):
        work = 0.0
        for _ in range(jl[j]):
            row = pt[op_cursor]
            valid = row[row > 0]
            if valid.size > 0:
                work += np.mean(valid)
            op_cursor += 1
        
        print(f"Job {j}: Work={work:.2f}, Due={dd[j]:.2f}, Ratio={dd[j]/work:.2f}")
        if j == 0:
            print(f"Job 0 Ops PT:\n{pt[0:jl[0]]}")

    # Initialize Env & Model
    env = FJSPEnvForVariousOpNums(n_j, n_m)
    # Important: Pass true_due_date_list!
    state = env.set_initial_data([jl], [pt], [dd], true_due_date_list=[dd])
    
    # Load Model (Optional: if you want to test a trained model)
    ppo = PPO_initialize()
    # Change this path to your trained model
    model_path = r'trained_network\SD2\alt_alpha1.pth' 
    
    if os.path.exists(model_path):
        try:
            ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}. Using random weights.")
    else:
        print(f"Model not found at {model_path}. Using random weights.")
    ppo.policy.eval()
    
    # Run Episode
    logs = []
    step_cnt = 0
    
    print("Start simulation...")
    while True:
        step_cnt += 1
        with torch.no_grad():
            pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                               op_mask=state.op_mask_tensor,
                               candidate=state.candidate_tensor,
                               fea_m=state.fea_m_tensor,
                               mch_mask=state.mch_mask_tensor,
                               comp_idx=state.comp_idx_tensor,
                               dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                               fea_pairs=state.fea_pairs_tensor)
        
        action = greedy_select_action(pi)
        
        state, reward, done, info = env.step(actions=action.cpu().numpy())
        
        # Extract details
        op_info = info['scheduled_op_details']
        job_due_date = float(env.true_due_date[0, op_info['job_id']])
        
        logs.append({
            'Step': step_cnt,
            'Job': op_info['job_id'],
            'Op_Global': op_info['op_global_id'],
            'Op_Local': op_info['op_id_in_job'],
            'Machine': op_info['machine_id'],
            'Start_Time': float(op_info['start_time']),
            'End_Time': float(op_info['end_time']),
            'Proc_Time': float(op_info['proc_time']),
            'Job_Due_Date': job_due_date,
            'Makespan_So_Far': float(info['current_makespan']),
            'Real_Tardiness_Acc': float(info['raw_accumulated_tardiness']),
            'Dense_Tardiness_Step': float(info['raw_local_tardiness']),
            'Norm_Mk_Reward': float(info['reward_mk'][0]),
            'Norm_Td_Reward': float(info['reward_td'][0]),
            'Total_Reward': float(reward.item() if hasattr(reward, 'item') else reward)
        })
        
        if done.all():
            break
            
    # Save CSV
    log_dir = 'debug_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    df = pd.DataFrame(logs)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(log_dir, f'debug_log_{n_j}x{n_m}_k{tightness}_{timestamp}.csv')
    df.to_csv(out_file, index=False)
    print(f"Debug log saved to {out_file}")
    
    # Summary
    print("\n=== Episode Summary ===")
    print(f"Final Makespan: {df['Makespan_So_Far'].iloc[-1]}")
    print(f"Total Tardiness: {df['Real_Tardiness_Acc'].iloc[-1]}")
    print(f"Total Reward: {df['Total_Reward'].sum()}")

if __name__ == '__main__':
    main()
