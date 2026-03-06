
import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from tqdm import tqdm
from params import configs
from model.PPO import PPO_initialize
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from data_utils import SD2_instance_generator, generate_due_dates
from common_utils import greedy_select_action, setup_seed

def test_generalization(model_path, device_id="0", noise_level=0.3):
    # 1. Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    
    # Load Model
    print(f"Loading model from: {model_path}")
    ppo = PPO_initialize()
    if os.path.exists(model_path):
        ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo.policy.eval()
    else:
        print(f"Model path not found: {model_path}")
        return

    # 2. Test Configuration
    job_sizes = [10, 20, 30, 40, 50, 60]
    m_values = [0.2, 0.4, 0.6] # Updated as requested
    n_m = 5 # Fixed machine count
    num_instances = 50
    base_seed = 9999 # Fixed seed for reproducibility
    
    # Storage for results: Dict[JobSize, Dict[M, Stats]]
    # Stats = {ms: [], td: [], obj: []}
    results = {}

    print(f"Starting Generalization Test (Noise: {noise_level*100}%)...")
    print(f"Sizes: {job_sizes}")
    print(f"M-values: {m_values}")

    for n_j in tqdm(job_sizes, desc="Job Sizes"):
        results[n_j] = {}
        
        # Adjust config for generation
        configs.n_j = n_j
        configs.n_m = n_m
        
        for m_val in m_values:
            # Batch Generation
            batch_jl = []
            batch_pt = []
            batch_dd = []
            
            for i in range(num_instances):
                # Deterministic Seed
                m_idx = int(m_val * 10)
                seed = base_seed + n_j * 1000 + m_idx * 100 + i
                
                # Generate Instance
                # 100% Uniform
                gen_mode = 'uniform'
                
                jl, pt, _ = SD2_instance_generator(configs, seed=seed, mode=gen_mode)
                
                # Generate Common Due Date (Mode M)
                dd = generate_due_dates(jl, pt, tightness=m_val, due_date_mode='M', seed=seed, noise_level=noise_level)
                
                batch_jl.append(jl)
                batch_pt.append(pt)
                batch_dd.append(dd)
            
            # Batch Inference
            env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
            # Pass true_due_date_list as batch_dd for correct reward/tardiness calculation
            state = env.set_initial_data(batch_jl, batch_pt, batch_dd, true_due_date_list=batch_dd)
            
            while True:
                with torch.no_grad():
                    # Check for done envs to mask them out if needed, 
                    # but PPO policy handles full batch. Efficient implementation runs all.
                    # Batch masking is handled inside env.step() logic usually or we just infer all.
                    # For max speed, we infer all but only step valid ones if env supports it.
                    # FJSPEnvForVariousOpNums step() filters incomplete envs internally.
                    
                    # Optimization: Only infer if not all done
                    if env.done_flag.all():
                        break
                        
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                                       op_mask=state.op_mask_tensor,
                                       candidate=state.candidate_tensor,
                                       fea_m=state.fea_m_tensor,
                                       mch_mask=state.mch_mask_tensor,
                                       comp_idx=state.comp_idx_tensor,
                                       dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                       fea_pairs=state.fea_pairs_tensor)
                
                action = greedy_select_action(pi)
                state, _, done, _ = env.step(action.cpu().numpy())
                
                if done.all():
                    break
            
            # Batch Metrics Collection
            ms_list = env.current_makespan.tolist()
            td_list = env.accumulated_tardiness.tolist()
            obj_list = [(0.5 * m + 0.5 * t) for m, t in zip(ms_list, td_list)]
            
            # Store Mean Values
            results[n_j][m_val] = {
                'MS': np.mean(ms_list),
                'TD': np.mean(td_list),
                'OBJ': np.mean(obj_list)
            }

    # 3. Export to CSV (Wide Format)
    # Row: Job Size
    # Columns: MS_1.2, MS_1.0, ..., TD_1.2, ..., OBJ_1.2 ...
    
    rows = []
    for n_j in job_sizes:
        row = {'Job_Size': n_j}
        for m_val in m_values:
            stats = results[n_j][m_val]
            row[f'MS_{m_val}'] = stats['MS']
            row[f'TD_{m_val}'] = stats['TD']
            row[f'OBJ_{m_val}'] = stats['OBJ']
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Reorder columns for clarity
    cols = ['Job_Size']
    for metric in ['MS', 'TD', 'OBJ']:
        for m_val in m_values:
            cols.append(f'{metric}_{m_val}')
            
    df = df[cols]
    
    # Filename
    model_name = os.path.basename(model_path).replace('.pth', '')
    out_path = f'test_results/generalization_{model_name}.csv'
    
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
        
    df.to_csv(out_path, index=False)
    print(f"\nTest Complete. Results saved to: {out_path}")
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Support either direct path or config file
    parser.add_argument('--model_path', type=str, default=None, help='Path to .pth model file')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--device', type=str, default='0', help='CUDA device ID')
    parser.add_argument('--due_date_noise', type=float, default=None, help='Noise level override (if not set, uses YAML or 0.0)')
    args = parser.parse_args()
    
    target_path = args.model_path
    noise_level = args.due_date_noise
    
    # If config is provided, resolve model path from YAML
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f) or {}
            
        if not target_path:
            model_name = yaml_cfg.get('eval_model_name')
            if not model_name:
                print("Error: 'eval_model_name' not found in YAML config.")
                exit(1)
            target_path = f"trained_network/SD2/{model_name}.pth"
            
        # Get noise level from YAML if not overridden
        if noise_level is None:
            noise_level = yaml_cfg.get('due_date_noise', 0.0)
    
    if not target_path:
        print("Error: Please provide either --model_path or --config")
        exit(1)
        
    if noise_level is None:
        noise_level = 0.3 # 30% default noise for testing
        
    test_generalization(target_path, args.device, noise_level)
