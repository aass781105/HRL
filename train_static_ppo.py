# train_static_ppo.py
# This script is designed to train a PPO agent on statically generated FJSP instances.
# It combines the high-level logic of static training with the robust, bug-free
# implementation details (environment, adapter, state/mask creation) from the
# 'train_unified.py' script.

import os
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# --- Core components from the project ---
from params import configs
from model.PPO_dynamic import PPO, Memory
from common_utils import sample_action

# Data generation and environment setup
from data_utils import SD2_instance_generator
from global_env import GlobalTimelineOrchestrator, split_matrix_to_jobs
from PPO_orchestrator_adapter import OrchestratorAdapter

def plot_training_results(updates, makespans, pi_losses, v_losses, save_dir):
    """
    Plots the training progress (makespan, policy loss, value loss) and saves it to a file.
    """
    if not updates:
        print("No data to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Plot Average Makespan
    axs[0].plot(updates, makespans, label='Average Makespan')
    axs[0].set_ylabel('Avg Makespan')
    axs[0].set_title('Training Progress')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Policy Loss
    axs[1].plot(updates, pi_losses, label='Policy Loss (pi_loss)', color='red')
    axs[1].set_ylabel('Policy Loss')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Value Loss
    axs[2].plot(updates, v_losses, label='Value Loss (v_loss)', color='green')
    axs[2].set_xlabel('Update Steps')
    axs[2].set_ylabel('Value Loss')
    axs[2].legend()
    axs[2].grid(True)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")
    plt.close(fig)

def train_static_ppo():
    """
    Main training function for PPO on static FJSP problems.
    """
    # --- 1. Initialization ---
    # Seeding for reproducibility
    seed = int(getattr(configs, "seed_train", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device(configs.device)

    # Hyperparameters
    max_updates = int(getattr(configs, "max_updates", 200))
    reset_env_timestep = int(getattr(configs, "reset_env_timestep", 20))
    num_envs = int(getattr(configs, "num_envs", 10))
    log_interval = int(getattr(configs, "validate_timestep", 10)) # Using validate_timestep for logging
    save_interval = int(getattr(configs, "validate_timestep", 10)) # Using validate_timestep for saving

    # PPO Agent and Memory
    ppo_agent = PPO(configs)
    # The memory accumulates experiences from num_envs instances before update
    ppo_memory = Memory(gamma=configs.gamma, gae_lambda=configs.gae_lambda)

    # Optional: Load pre-trained model
    # ppo_model_path = getattr(configs, "ppo_model_path_static", "trained_network/static_ppo_initial.pth")
    ppo_model_path  = ''
    if os.path.exists(ppo_model_path):
        print(f"Loading pre-trained PPO model from {ppo_model_path}")
        ppo_agent.policy.load_state_dict(torch.load(ppo_model_path, map_location=device))
        ppo_agent.policy_old.load_state_dict(torch.load(ppo_model_path, map_location=device))
    else:
        print(f"Warning: Pre-trained PPO model not found at {ppo_model_path}. Starting from scratch.")

    # Output directory for saving models
    output_dir = "trained_network/static_ppo"
    os.makedirs(output_dir, exist_ok=True)

    print("-" * 25 + " Starting PPO Static Training " + "-" * 25)

    # Data collectors for plotting
    plot_updates = []
    all_avg_makespans = []
    all_pi_losses = []
    all_v_losses = []

    # Stores the batch of instances for the current reset_env_timestep cycle
    instance_batch = [] 

    # --- 2. Main Training Loop (based on max_updates) ---
    for i_update in tqdm(range(1, max_updates + 1), desc="PPO Static Updates"):
        # --- 2a. Resample instances periodically ---
        if (i_update - 1) % reset_env_timestep == 0: # -1 because i_update starts from 1
            instance_batch = []
            for i in range(num_envs):
                # Use a combined seed for reproducibility across updates and instances
                # The seed uses the initial 'seed' + an offset based on the current batch and instance index.
                current_instance_seed = (i_update // reset_env_timestep) * num_envs + i + seed 
                job_length, op_pt, _ = SD2_instance_generator(configs, seed=current_instance_seed)
                jobs = split_matrix_to_jobs(job_length, op_pt, t_arrive=0.0)
                instance_batch.append(jobs)
            tqdm.write(f"Generated a new batch of {num_envs} instances for this cycle.")

        # --- 2b. Run episodes for the entire batch and collect experiences ---
        total_makespan_batch = []
        for instance_idx, jobs_for_instance in enumerate(instance_batch):
            # Setup Environment for this single instance from the batch
            # Note: A new orchestrator/adapter is created for each instance to maintain isolation
            orchestrator = GlobalTimelineOrchestrator(
                n_machines=configs.n_m,
                job_generator=None,  # No dynamic job generation
                select_from_buffer=lambda buf, o: list(range(len(buf))),
                t0=0.0
            )
            orchestrator.buffer.extend(jobs_for_instance)
            adapter = OrchestratorAdapter(orchestrator, n_machines=configs.n_m)

            # Run one PPO episode for this specific instance
            t_now = 0.0
            state_ppo = adapter.begin_new_batch(t_now)
            
            if state_ppo:
                while True:
                    # Store the current state for the update later
                    ppo_memory.push(state_ppo)

                    # Agent selects an action
                    with torch.no_grad():
                        pi, val = ppo_agent.policy_old(
                            fea_j=state_ppo.fea_j_tensor, op_mask=state_ppo.op_mask_tensor,
                            candidate=state_ppo.candidate_tensor, fea_m=state_ppo.fea_m_tensor,
                            mch_mask=state_ppo.mch_mask_tensor, comp_idx=state_ppo.comp_idx_tensor,
                            dynamic_pair_mask=state_ppo.dynamic_pair_mask_tensor, fea_pairs=state_ppo.fea_pairs_tensor
                        )
                    
                    action_ppo, log_prob_ppo = sample_action(pi)
                    
                    # Environment steps forward
                    state_ppo, reward_ppo, sub_done, _ = adapter.step_in_batch(action_ppo.cpu())
                    
                    # Store outcomes in memory
                    ppo_memory.done_seq.append(torch.from_numpy(np.array([sub_done])).to(device))
                    ppo_memory.reward_seq.append(torch.from_numpy(reward_ppo).to(device))
                    ppo_memory.action_seq.append(action_ppo)
                    ppo_memory.log_probs.append(log_prob_ppo)
                    ppo_memory.val_seq.append(val.squeeze(1))
                    
                    if sub_done:
                        break
                
                # Finalize the batch scheduling for this instance
                adapter.finalize_batch()
                total_makespan_batch.append(np.max(orchestrator.machine_free_time))
            else:
                tqdm.write(f"Warning: Instance {instance_idx} in batch had no jobs to schedule.")

        # --- 2c. Update PPO after processing the entire batch of instances ---
        if len(ppo_memory.reward_seq) > 0:
            _, v_loss, p_loss = ppo_agent.update(ppo_memory)
        else:
            p_loss, v_loss = 0, 0
            tqdm.write(f"Warning: No experiences collected for update {i_update}.")
        
        # --- 2d. Logging and Cleanup ---
        # Memory is cleared after update, having accumulated experiences from all num_envs episodes
        ppo_memory.clear_memory()

        avg_makespan = np.mean(total_makespan_batch) if total_makespan_batch else 0.0
        
        # Store data for plotting
        plot_updates.append(i_update)
        all_avg_makespans.append(avg_makespan)
        all_pi_losses.append(p_loss)
        all_v_losses.append(v_loss)

        if i_update % log_interval == 0:
            tqdm.write(f"[Update {i_update:04d}] Avg Makespan: {avg_makespan:.2f} | Pi Loss: {p_loss:.4f} | V Loss: {v_loss:.4f}")

        # --- 2e. Save Model Periodically ---
        if i_update % save_interval == 0:
            save_path = os.path.join(output_dir, f"ppo_static_update_{i_update}.pth")
            torch.save(ppo_agent.policy.state_dict(), save_path)
            tqdm.write(f"[Update {i_update:04d}] Model saved to {save_path}")

    # --- 3. Final Model Save ---
    final_save_path = os.path.join(output_dir, "ppo_static_final.pth")
    torch.save(ppo_agent.policy.state_dict(), final_save_path)
    print(f"\nTraining finished. Final model saved to {final_save_path}")

    # --- 4. Plot Training Results ---
    plot_training_results(
        updates=plot_updates,
        makespans=all_avg_makespans,
        pi_losses=all_pi_losses,
        v_losses=all_v_losses,
        save_dir="plots/train_ppo_static"
    )

if __name__ == "__main__":
    train_static_ppo()
