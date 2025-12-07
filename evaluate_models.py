# evaluate_models.py
# This script benchmarks PPO models on a fixed set of statically generated instances.
# It reads configuration from `evaluation.yml` to determine which models to test,
# how many times to run each instance, and what action selection strategy to use.
# The results, including mean and standard deviation of makespan, are printed
# to the console and saved to a CSV file.

import os
import yaml
import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import copy

# --- Core components from the project ---
from params import configs
from model.PPO_dynamic import PPO
from common_utils import sample_action, greedy_select_action
from data_utils import SD2_instance_generator
from global_env import GlobalTimelineOrchestrator, split_matrix_to_jobs
from PPO_orchestrator_adapter import OrchestratorAdapter

def run_ppo_episode(adapter, ppo_policy, action_selection, device):
    """
    Runs a single episode for a given instance using the PPO policy.
    Returns the final makespan.
    """
    t_now = 0.0
    state_ppo = adapter.begin_new_batch(t_now)
    
    if not state_ppo:
        return 0.0  # Return 0 if there are no jobs to schedule

    while True:
        with torch.no_grad():
            pi, _ = ppo_policy(
                fea_j=state_ppo.fea_j_tensor, op_mask=state_ppo.op_mask_tensor,
                candidate=state_ppo.candidate_tensor, fea_m=state_ppo.fea_m_tensor,
                mch_mask=state_ppo.mch_mask_tensor, comp_idx=state_ppo.comp_idx_tensor,
                dynamic_pair_mask=state_ppo.dynamic_pair_mask_tensor, fea_pairs=state_ppo.fea_pairs_tensor
            )
        
        if action_selection == 'stochastic':
            action_ppo, _ = sample_action(pi)
        else:  # 'greedy'
            action_ppo = greedy_select_action(pi)

        state_ppo, _, sub_done, _ = adapter.step_in_batch(action_ppo.cpu())
        
        if sub_done:
            break
            
    adapter.finalize_batch()
    final_makespan = np.max(adapter.o.machine_free_time)
    return final_makespan

def main():
    # --- 1. Load Configuration ---
    config_path = 'evaluation.yml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return
        
    with open(config_path, 'r') as f:
        eval_config = yaml.safe_load(f)

    test_seed = eval_config.get('test_instances_seed', 42)
    runs_per_instance = eval_config.get('runs_per_instance', 10)
    models_to_test = eval_config.get('models_to_test', [])
    
    device = torch.device(configs.device)
    
    print("-" * 25 + " Starting Model Evaluation " + "-" * 25)
    print(f"Test instances seed: {test_seed}")
    print(f"Runs per instance: {runs_per_instance}")
    print(f"Found {len(models_to_test)} models to test.")

    # --- 2. Generate Fixed Test Instances ---
    num_test_instances = 10
    print(f"Generating {num_test_instances} fixed test instances...")
    test_instances = []
    # Use a separate random generator for instance creation to not interfere with other random processes
    instance_rng = np.random.default_rng(test_seed)
    for i in range(num_test_instances):
        # Pass a random seed from our fixed generator to SD2 generator
        instance_seed = instance_rng.integers(low=0, high=1_000_000)
        job_length, op_pt, _ = SD2_instance_generator(configs, seed=instance_seed)
        jobs = split_matrix_to_jobs(job_length, op_pt, t_arrive=0.0)
        test_instances.append(jobs)
    print("Test instances generated successfully.")

    all_results = []

    # --- 3. Main Evaluation Loop ---
    for model_info in models_to_test:
        model_name = model_info['name']
        model_type = model_info['type']
        model_path = model_info['path']
        action_selection = model_info.get('action_selection', 'greedy')
        
        print(f"\n--- Testing Model: {model_name} ---")

        if not os.path.exists(model_path):
            print(f"Warning: Model path not found for '{model_name}'. Skipping.")
            continue
            
        if model_type != "PPO":
            print(f"Warning: Model type '{model_type}' is not supported for this evaluation script. Skipping.")
            continue

        # Load PPO policy
        # We only need the policy for evaluation, not the full PPO agent with optimizer
        ppo_policy = PPO(configs).policy
        ppo_policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_policy.to(device)
        ppo_policy.eval()

        # Loop through each fixed instance
        for i, instance_jobs in enumerate(tqdm(test_instances, desc=f"Instances for {model_name}")):
            instance_makespans = []
            
            # Run the same instance multiple times
            for _ in range(runs_per_instance):
                # We must create a new orchestrator and adapter for each run to ensure a fresh start
                orchestrator = GlobalTimelineOrchestrator(
                    n_machines=configs.n_m,
                    job_generator=None,
                    select_from_buffer=lambda buf, o: list(range(len(buf))),
                    t0=0.0
                )
                # Use deepcopy to avoid modifying the original instance jobs list
                orchestrator.buffer.extend(copy.deepcopy(instance_jobs))
                adapter = OrchestratorAdapter(orchestrator, n_machines=configs.n_m)
                
                makespan = run_ppo_episode(adapter, ppo_policy, action_selection, device)
                instance_makespans.append(makespan)

            # Calculate statistics for the current instance
            avg_makespan = np.mean(instance_makespans)
            std_makespan = np.std(instance_makespans)
            
            all_results.append({
                "model_name": model_name,
                "instance_id": i + 1,
                "avg_makespan": avg_makespan,
                "std_makespan": std_makespan,
                "raw_makespans": instance_makespans
            })

    # --- 4. Output Results ---
    print("\n" + "=" * 25 + " Evaluation Results " + "=" * 25)
    
    # Console Output
    current_model = ""
    for res in all_results:
        if res['model_name'] != current_model:
            current_model = res['model_name']
            print(f"\nModel: {current_model}")
            print("-" * 40)
            print(f"{ 'Instance':<12} | {'Avg Makespan':<15} | {'Std Dev':<15}")
            print("-" * 40)
        print(f"{res['instance_id']:<12} | {res['avg_makespan']:<15.2f} | {res['std_makespan']:<15.2f}")

    # CSV Output
    csv_file = 'evaluation_results.csv'
    print(f"\nSaving detailed results to {csv_file}...")
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model Name", "Instance ID", "Avg Makespan", "Std Dev Makespan"])
            for res in all_results:
                writer.writerow([res['model_name'], res['instance_id'], res['avg_makespan'], res['std_makespan']])
        print("Successfully saved results to CSV.")
    except IOError as e:
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    main()
