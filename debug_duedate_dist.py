import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from params import configs
from data_utils import SD2_instance_generator, generate_due_dates
from tqdm import tqdm

def visualize_due_dates_batch():
    # 1. Setup
    num_instances = 100
    n_j = 20
    n_m = 5
    base_seed_val = 42
    
    # Temporarily update configs
    configs.n_j = n_j
    configs.n_m = n_m
    
    all_data = []

    print(f"Generating {num_instances} Instances ({n_j}x{n_m}) for statistics...")

    for i in tqdm(range(num_instances)):
        seed = base_seed_val + i
        
        # Generate Instance
        # Use 50/50 mix to match training/testing distribution
        gen_mode = 'realistic' if i % 2 == 0 else 'uniform'
        jl, pt, _ = SD2_instance_generator(configs, seed=seed, mode=gen_mode)
        
        # Calculate Job Work for reference
        job_work_list = []
        op_idx = 0
        for j in range(n_j):
            work = 0
            for _ in range(jl[j]):
                pt_row = pt[op_idx]
                compat = pt_row[pt_row > 0]
                if len(compat) > 0:
                    work += np.mean(compat)
                op_idx += 1
            job_work_list.append(work)
        job_work = np.array(job_work_list)
        
        M_baseline = 0.6
        # Generate Due Dates
        # 1. K=1.2 (Individual Baseline)
        dd_k = generate_due_dates(jl, pt, tightness=1.2, due_date_mode='k')
        
        # 2. M=0.6 Pure (Common Baseline)
        dd_m_pure = generate_due_dates(jl, pt, tightness=M_baseline, due_date_mode='M', noise_level=0.0)
        
        # 3. M=0.6 Noise 30% (Random Noise - matches test default)
        dd_m_rand30 = generate_due_dates(jl, pt, tightness=M_baseline, due_date_mode='M', seed=seed, noise_level=0.3)
        
        # 4. M=0.6 Comp 50% (Structural Noise - matches training)
        dd_m_comp50 = generate_due_dates(jl, pt, tightness=M_baseline, due_date_mode='M', seed=seed, noise_level=-0.5)

        # Record each job
        for j in range(n_j):
            all_data.append({
                'Job_Work': job_work[j],
                'K=1.2': dd_k[j],
                'M=0.6_Pure': dd_m_pure[j],
                'M=0.6_Rand30': dd_m_rand30[j],
                'M=0.6_Comp50': dd_m_comp50[j],
                'Slack_K': dd_k[j] - job_work[j],
                'Slack_Pure': dd_m_pure[j] - job_work[j],
                'Slack_Rand30': dd_m_rand30[j] - job_work[j],
                'Slack_Comp50': dd_m_comp50[j] - job_work[j]
            })
    
    df = pd.DataFrame(all_data)
    
    # 2. Display Statistics
    cols_dd = ['K=1.2', 'M=0.6_Pure', 'M=0.6_Rand30', 'M=0.6_Comp50']
    cols_slack = ['Slack_K', 'Slack_Pure', 'Slack_Rand30', 'Slack_Comp50']
    
    pd.set_option('display.float_format', '{:.1f}'.format)
    print("\n--- Due Date Statistics (Total Jobs: {})" .format(len(df)))
    print(df[cols_dd].describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\n--- Slack Statistics (Initial Buffer Time) ---")
    print(df[cols_slack].describe().loc[['mean', 'std', 'min', 'max']])

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Boxplot of Due Dates
    df[cols_dd].boxplot(ax=axes[0])
    axes[0].grid(False)
    axes[0].set_title('Distribution of Due Date Values')
    axes[0].set_ylabel('Absolute Time')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Boxplot of Slacks
    df[cols_slack].boxplot(ax=axes[1])
    axes[1].grid(False)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Distribution of Initial Slack (Due - Work)')
    axes[1].set_ylabel('Buffer Time')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('debug_duedate_dist_batch.png')
    print("\nAnalysis complete. Plot saved to 'debug_duedate_dist_batch.png'")

if __name__ == "__main__":
    visualize_due_dates_batch()