import time
import os
import sys
import numpy as np
import torch
import pandas as pd
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed, greedy_select_action
# Use the environment that supports due dates and new features
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums 

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
device = torch.device(configs.device)

ppo = PPO_initialize()

def generate_fixed_due_date(job_length, op_pt, factor):
    """
    Generate due dates based on a fixed tightness factor.
    D_j = Total_Work_j * factor
    """
    n_j = job_length.shape[0]
    job_work = np.zeros(n_j)
    op_idx = 0
    for j in range(n_j):
        for _ in range(job_length[j]):
            pt_row = op_pt[op_idx]
            compat_pt = pt_row[pt_row > 0]
            if len(compat_pt) > 0:
                job_work[j] += np.mean(compat_pt)
            op_idx += 1
            
    return job_work * factor

def test_benchmark_multifactors(data_set, model_path, factors=[0.9, 1.2, 1.5]):
    """
    Test the model on benchmark data with various due date tightness factors.
    Returns a DataFrame with results.
    """
    # Load Model
    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        return pd.DataFrame()
        
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
    ppo.policy.eval()

    job_lengths, op_pts = data_set[0], data_set[1]
    num_instances = len(job_lengths)
    
    results = []

    print(f"Testing on {num_instances} instances with factors: {factors}")

    # Loop over each instance
    for i in tqdm(range(num_instances), desc="Instances"):
        jl = job_lengths[i]
        pt = op_pts[i]
        n_j = jl.shape[0]
        n_m = pt.shape[1]
        
        # Loop over each tightness factor
        for k in factors:
            # Generate deterministic due dates for this factor
            dd = generate_fixed_due_date(jl, pt, factor=k)
            
            # Initialize Environment (Single Env Mode for Testing)
            # Note: FJSPEnvForVariousOpNums is designed for batched envs.
            # We can treat this single instance as a batch of size 1.
            env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
            
            # set_initial_data expects list of arrays
            # Also pass true_due_date_list for correct tardiness calculation
            state = env.set_initial_data([jl], [pt], [dd], true_due_date_list=[dd])
            
            t_start = time.time()
            while True:
                with torch.no_grad():
                    # Batch size is 1, so no need for complex masking usually, 
                    # but code requires it.
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                                       op_mask=state.op_mask_tensor,
                                       candidate=state.candidate_tensor,
                                       fea_m=state.fea_m_tensor,
                                       mch_mask=state.mch_mask_tensor,
                                       comp_idx=state.comp_idx_tensor,
                                       dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                       fea_pairs=state.fea_pairs_tensor)
                
                # Greedy Decoding
                action = greedy_select_action(pi)
                state, _, done, _ = env.step(actions=action.cpu().numpy())
                
                if done.all():
                    break
            
            t_end = time.time()
            
            # Record Results
            # env.current_makespan is an array [ms]
            ms = env.current_makespan[0]
            td = env.accumulated_tardiness[0]
            obj = 0.5 * ms + 0.5 * td
            
            results.append({
                'Instance_ID': i,
                'Tightness_Factor': k,
                'Makespan': ms,
                'Tardiness': td,
                'Objective': obj,
                'Solve_Time': t_end - t_start
            })

    return pd.DataFrame(results)

def main():
    # Configuration for manual run or args
    # Example: Test Hurink_vdata using model 'due_date_ppo'
    
    # 1. Define Model Path
    # Using eval_model_name from configs
    model_name = configs.eval_model_name
    if not model_name.endswith('.pth'):
        model_name += '.pth'
        
    model_dir = f'./trained_network/{configs.data_source}/'
    model_path = os.path.join(model_dir, model_name)
    
    # 2. Define Data Source
    # configs.data_source = 'BenchData'
    # configs.test_data = ['Hurink_vdata'] 
    
    # Let's load specifically Hurink_vdata
    print("Loading Benchmark Data...")
    # Using existing utility but pointing to BenchData
    # Note: data_utils.load_data_from_files scans directory.
    # BenchData path structure: ./data/BenchData/Hurink_vdata/
    from data_utils import load_data_from_files
    data_path = './data/BenchData/Hurink_vdata/'
    data_set = load_data_from_files(data_path)
    
    if not data_set[0]:
        print(f"No data found in {data_path}")
        return

    # 3. Run Test
    factors = [0.9, 1.2, 1.5]
    df_results = test_benchmark_multifactors(data_set, model_path, factors)
    
    if not df_results.empty:
        # 4. Save Results
        output_dir = './test_results_benchmark/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Clean model name for filename
        clean_model_name = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # --- Pivot to Wide Format ---
        # Pivot table: Index=Instance_ID, Columns=Tightness_Factor, Values=[Makespan, Tardiness]
        df_pivot = df_results.pivot(index='Instance_ID', columns='Tightness_Factor', values=['Makespan', 'Tardiness'])
        
        # Flatten MultiIndex columns
        # e.g., ('Makespan', 0.9) -> 'Makespan_0.9'
        df_pivot.columns = [f'{col[0]}_{col[1]}' for col in df_pivot.columns]
        
        # Reorder columns to group by Factor (MS_0.9, TD_0.9, MS_1.2, TD_1.2 ...)
        factors = sorted(df_results['Tightness_Factor'].unique())
        ordered_cols = []
        for f in factors:
            ordered_cols.append(f'Makespan_{f}')
            ordered_cols.append(f'Tardiness_{f}')
            
        df_final = df_pivot[ordered_cols].reset_index()
        
        csv_path = os.path.join(output_dir, f'TestResult_{clean_model_name}_{timestamp}_wide.csv')
        df_final.to_csv(csv_path, index=False)
        
        print("\n" + "="*30)
        print(f"Test Complete! Results saved to: {csv_path}")
        print("Summary by Factor:")
        print(df_results.groupby('Tightness_Factor')[['Makespan', 'Tardiness', 'Objective']].mean())
        print("="*30)
    else:
        print("Test failed or no results generated.")

if __name__ == '__main__':
    main()