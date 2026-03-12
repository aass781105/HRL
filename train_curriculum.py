from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import SD2_instance_generator, generate_due_dates, load_data_from_files
from common_utils import strToSuffix, setup_seed
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from copy import deepcopy
import os
import random
import time
import sys
import pandas as pd 
from model.PPO import PPO_initialize
from model.PPO import Memory

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

class Trainer:
    def __init__(self, config):
        # Initial configurations
        self.initial_n_j = 10
        self.fixed_n_m = config.n_m
        
        configs.n_j = self.initial_n_j 
        configs.data_source = 'SD2'
        configs.data_suffix = 'mix'
        
        # [FORCE UNIFIED MODES] 
        configs.due_date_mode = 'range'
        
        self.n_j = configs.n_j
        self.n_m = configs.n_m
        self.config = config

        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs

        if not os.path.exists(f'./trained_network/{self.config.data_source}'):
            os.makedirs(f'./trained_network/{self.config.data_source}')
        if not os.path.exists(f'./train_log/{self.config.data_source}'):
            os.makedirs(f'./train_log/{self.config.data_source}')

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model_name = configs.eval_model_name

        # --- Fixed Validation Suite [10, 20, 30] with 'range' mode and 'uniform' data ---
        self.vali_data_batches = self.generate_fixed_validation_data(mode='uniform')

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def generate_fixed_validation_data(self, mode='uniform'):
        """
        Generates fixed validation instances.
        - Job sizes: [10, 20, 30]
        - Due Date Mode: 'range'
        - Data Mode: 'uniform' (matches dynamic subproblems)
        """
        print("-" * 25 + f"Generating Validation Suite (Mode: {mode} | Range-based Due Dates)" + "-" * 25)
        sizes = [10, 20, 30]
        num_per_size = 50
        total_instances = len(sizes) * num_per_size
        
        vali_batches = []
        old_n_j = configs.n_j
        
        for n_j in sizes:
            configs.n_j = n_j
            size_jl, size_pt, size_dd = [], [], []
            for i in range(num_per_size):
                vali_seed = 1000 + n_j * 100 + i
                # Using pure uniform for dynamic subproblem consistency
                jl, pt, _ = SD2_instance_generator(configs, seed=vali_seed, mode='uniform')
                dd = generate_due_dates(jl, pt, due_date_mode='range', seed=vali_seed)
                size_jl.append(jl); size_pt.append(pt); size_dd.append(dd)
            
            vali_batches.append({'n_j': n_j, 'jl': size_jl, 'pt': size_pt, 'dd': size_dd})
            print(f"Generated Validation Batch for Size {n_j} (range mode)")
            
        configs.n_j = old_n_j
        return vali_batches

    def train(self):
        setup_seed(self.config.seed_train)
        self.log, self.detailed_log, self.validation_log, self.validation_tardiness_log, self.loss_log = [], [], [], [], []
        self.record = float('inf')

        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"Model: {self.model_name} | Data: Pure Uniform | DueMode: range")

        self.train_st = time.time()
        
        def create_stage(n_j, multiplier):
            reset_step = int(5 * n_j)
            return {
                "n_j": n_j, "reset_step": reset_step, "duration": int(reset_step * multiplier),
                "stage_label": f"J{n_j}_Uniform_Range"
            }

        # [REFACTORED] Curriculum Schedule: 10 -> 20 -> 30
        # Increased multiplier from 9 to 25 for more training episodes
        curriculum_schedule = [
            create_stage(10, 5), # Stage 1: 10 Jobs
            create_stage(15, 5), # Stage 2: 20 Jobs
            create_stage(20, 5)  # Stage 3: 30 Jobs
        ]
        
        self.max_updates = sum(stage['duration'] for stage in curriculum_schedule)
        
        current_stage_idx = 0
        current_cfg = curriculum_schedule[0]
        configs.n_j, current_reset_step = current_cfg["n_j"], current_cfg["reset_step"]
        current_stage_end_step = current_cfg["duration"]
        current_stage_duration = current_cfg["duration"]
        stage_start_step = 0
        
        import math
        self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            if i_update >= current_stage_end_step and current_stage_idx < len(curriculum_schedule) - 1:
                current_stage_idx += 1
                current_cfg = curriculum_schedule[current_stage_idx]
                configs.n_j, current_reset_step = current_cfg["n_j"], current_cfg["reset_step"]
                stage_start_step = current_stage_end_step
                current_stage_duration = current_cfg["duration"]
                current_stage_end_step += current_stage_duration
                tqdm.write(f"\nCURRICULUM UPDATE: Stage {current_stage_idx+1}. Job Size={configs.n_j}")
                
                # [NEW] Force immediate environment reset on stage transition
                dataset_job_length, dataset_op_pt, dataset_due_date = self.sample_training_instances(i_update)
                self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_due_date, true_due_date_list=dataset_due_date)
                continue # Skip the normal modulo check below for this step

            if i_update % current_reset_step == 0:
                dataset_job_length, dataset_op_pt, dataset_due_date = self.sample_training_instances(i_update)
                self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_due_date, true_due_date_list=dataset_due_date)
            else:
                state = self.env.reset()

            # Sawtooth LR
            peak_lr = configs.lr * (0.95 ** current_stage_idx)
            steps_in_stage = i_update - stage_start_step
            cycle_progress = min(1.0, steps_in_stage / max(1, current_stage_duration))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_progress))
            current_lr = max(peak_lr * (0.1 + 0.9 * cosine_decay), 5e-5)
            for param_group in self.ppo.optimizer.param_groups: param_group['lr'] = current_lr
            self.ppo.entloss_coef = configs.entloss_coef
            
            ep_rewards = np.zeros(self.num_envs)
            ep_mk_gain, ep_td_penalty = 0.0, 0.0
            all_mk_rewards, all_td_rewards = [], []

            while True:
                self.memory.push(state)
                with torch.no_grad():
                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                                                             fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
                action_envs, action_logprob_envs = sample_action(pi_envs)
                state, reward, done, info = self.env.step(actions=action_envs.cpu().numpy())
                ep_mk_gain += np.mean(info['reward_mk']); ep_td_penalty += np.mean(info['reward_td'])
                all_mk_rewards.extend(info['reward_mk'].flatten()); all_td_rewards.extend(info['reward_td'].flatten())
                ep_rewards += reward
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(torch.from_numpy(reward).to(device))
                self.memory.action_seq.append(action_envs.squeeze(-1))
                self.memory.log_probs.append(action_logprob_envs.squeeze(-1))
                self.memory.val_seq.append(vals_envs.squeeze(1))
                if done.all(): break

            loss, v_loss, p_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mk_std, td_std = np.std(all_mk_rewards), np.std(all_td_rewards)
            self.log.append([i_update, np.mean(ep_rewards)])
            self.detailed_log.append([i_update, np.mean(ep_rewards), ep_mk_gain, mk_std, ep_td_penalty, td_std, np.mean(self.env.current_makespan), np.mean(self.env.accumulated_tardiness)])
            self.loss_log.append([i_update, loss, v_loss, p_loss])

            if (i_update + 1) % self.validate_timestep == 0:
                # Get per-size validation results
                vali_results_per_size = self.validate_envs_with_various_op_nums(self.vali_data_batches)
                
                # Calculate overall mean
                all_td, all_ms = [], []
                breakdown_entry = {'update': i_update + 1}
                for res in vali_results_per_size:
                    nj_key = res['n_j']
                    all_td.extend(res['td_list'])
                    all_ms.extend(res['ms_list'])
                    breakdown_entry[f'ms_{nj_key}j'] = res['ms_mean']
                    breakdown_entry[f'td_{nj_key}j'] = res['td_mean']
                
                overall_td_mean = np.mean(all_td)
                overall_ms_mean = np.mean(all_ms)
                
                # Save best model
                if overall_td_mean < self.record:
                    self.save_model(); self.record = overall_td_mean
                
                # Logs
                self.validation_log.append(overall_ms_mean)
                self.validation_tardiness_log.append(overall_td_mean)
                if not hasattr(self, 'validation_breakdown_log'): self.validation_breakdown_log = []
                self.validation_breakdown_log.append(breakdown_entry)
                
                self.save_validation_log()
                
                # [UPDATED] Clean Console Output with Training Metrics
                avg_reward = np.mean(ep_rewards)
                # Use current loss values from the update step
                tqdm.write(f'Update {i_update+1}/{self.max_updates} | R: {avg_reward:.2f} | Loss: {loss:.4f} | V-Loss: {v_loss:.4f} | Vali MK: {overall_ms_mean:.1f} | Vali TD: {overall_td_mean:.1f} | Best TD: {self.record:.1f}')

        self.train_et = time.time()
        self.save_training_log()

    def save_training_log(self):
        def to_native(obj):
            if isinstance(obj, list): return [to_native(i) for i in obj]
            if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
            if hasattr(obj, 'item'): return obj.item()
            return obj

        log_model_name = f'{self.model_name}_{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(self.config.data_suffix)}'
        log_path_base = f'./train_log/{self.config.data_source}/'

        # Convert and save all logs to .txt
        with open(f'{log_path_base}reward_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.log)))
        with open(f'{log_path_base}detailed_reward_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.detailed_log)))
        with open(f'{log_path_base}valiquality_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.validation_log)))
        with open(f'{log_path_base}valitardiness_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.validation_tardiness_log)))
        with open(f'{log_path_base}loss_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.loss_log)))
        
        # [ALIGNED] Save breakdown log as .txt
        if hasattr(self, 'validation_breakdown_log') and self.validation_breakdown_log:
            with open(f'{log_path_base}valibreakdown_{log_model_name}.txt', 'w') as f: 
                f.write(str(to_native(self.validation_breakdown_log)))

            # Plotting (Directly from memory for performance)
            try:
                import matplotlib.pyplot as plt
                df_breakdown = pd.DataFrame(self.validation_breakdown_log)
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                sizes = [10, 20, 30]
                for i, n_j in enumerate(sizes):
                    ax = axes[i]
                    ax.plot(df_breakdown['update'], df_breakdown[f'td_{n_j}j'], color='red', label='Tardiness')
                    ax.set_title(f'Validation Trend - {n_j} Jobs')
                    ax.set_xlabel('Updates')
                    ax.set_ylabel('Mean Total Tardiness', color='red')
                    ax2 = ax.twinx()
                    ax2.plot(df_breakdown['update'], df_breakdown[f'ms_{n_j}j'], color='blue', linestyle='--', label='Makespan')
                    ax2.set_ylabel('Mean Makespan', color='blue')
                plt.tight_layout()
                plt.savefig(f'{log_path_base}valitrend_{log_model_name}.png')
                plt.close(fig)
            except Exception as e:
                print(f"Plotting failed: {e}")

    def save_validation_log(self):
        def to_native(obj):
            if isinstance(obj, list): return [to_native(i) for i in obj]
            if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
            if hasattr(obj, 'item'): return obj.item()
            return obj

        log_model_name = f'{self.model_name}_{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(self.config.data_suffix)}'
        log_path_base = f'./train_log/{self.config.data_source}/'
        
        with open(f'{log_path_base}valiquality_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.validation_log)))
        with open(f'{log_path_base}valitardiness_{log_model_name}.txt', 'w') as f: f.write(str(to_native(self.validation_tardiness_log)))
        
        # [ALIGNED] Real-time Breakdown .txt
        if hasattr(self, 'validation_breakdown_log') and self.validation_breakdown_log:
            with open(f'{log_path_base}valibreakdown_{log_model_name}.txt', 'w') as f: 
                f.write(str(to_native(self.validation_breakdown_log)))

    def sample_training_instances(self, i_update):
        dataset_JobLength, dataset_OpPT, dataset_DueDate = [], [], []
        # [UPDATED] 100% Uniform for consistency
        for i in range(self.num_envs):
            instance_seed = self.config.seed_train + i_update * self.num_envs + i
            # Force 'uniform' mode
            JobLength, OpPT, _ = SD2_instance_generator(config=self.config, seed=instance_seed, mode='uniform')
            # Force 'range' mode
            DueDate = generate_due_dates(JobLength, OpPT, due_date_mode='range', seed=instance_seed)
            dataset_JobLength.append(JobLength); dataset_OpPT.append(OpPT); dataset_DueDate.append(DueDate)
        return dataset_JobLength, dataset_OpPT, dataset_DueDate

    def validate_envs_with_various_op_nums(self, batches):
        self.ppo.policy.eval()
        results_per_batch = []
        for batch in batches:
            temp_env = FJSPEnvForVariousOpNums(n_j=batch['n_j'], n_m=self.fixed_n_m)
            state = temp_env.set_initial_data(batch['jl'], batch['pt'], batch['dd'], true_due_date_list=batch['dd'], normalize_due_date=False)
            while True:
                with torch.no_grad():
                    batch_idx = ~torch.from_numpy(temp_env.done_flag)
                    if batch_idx.any():
                        pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx], op_mask=state.op_mask_tensor[batch_idx], candidate=state.candidate_tensor[batch_idx],
                                                fea_m=state.fea_m_tensor[batch_idx], mch_mask=state.mch_mask_tensor[batch_idx], comp_idx=state.comp_idx_tensor[batch_idx],
                                                dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx], fea_pairs=state.fea_pairs_tensor[batch_idx])
                        action = greedy_select_action(pi)
                        state, _, done, _ = temp_env.step(action.cpu().numpy())
                    else: break
                if done.all(): break
            
            results_per_batch.append({
                'n_j': batch['n_j'],
                'ms_mean': np.mean(temp_env.current_makespan),
                'td_mean': np.mean(temp_env.accumulated_tardiness),
                'ms_list': temp_env.current_makespan.tolist(),
                'td_list': temp_env.accumulated_tardiness.tolist()
            })
            
        self.ppo.policy.train()
        return results_per_batch

    def save_model(self):
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.config.data_source}/{self.model_name}.pth')

def main():
    setup_seed(configs.seed_train)
    configs.n_j, configs.n_m, configs.data_source, configs.data_suffix = 10, 5, 'SD2', 'mix'
    trainer = Trainer(configs)
    trainer.train()

if __name__ == '__main__': main()
