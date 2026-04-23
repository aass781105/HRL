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
        self.config = config
        self.initial_n_j = int(getattr(config, "n_j", 10))
        self.fixed_n_m = config.n_m
        
        configs.n_j = self.initial_n_j 
        # Training setup: fixed 5 operations per job.
        setattr(configs, "op_per_job", 5)
        setattr(self.config, "op_per_job", 5)
        setattr(configs, "enable_op_mixture", False)
        setattr(self.config, "enable_op_mixture", False)
        configs.data_source = 'SD2'
        configs.data_suffix = 'mix'
        
        self.n_j = configs.n_j
        self.n_m = configs.n_m

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

        # --- Fixed Validation Suite [10, 20, 30] with the configured due-date mode and uniform data ---
        self.vali_data_batches = self.generate_fixed_validation_data(mode='uniform')

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def generate_fixed_validation_data(self, mode='uniform'):
        """
        Generates fixed validation instances.
        - Job sizes: [10, 20, 30]
        - Due Date Mode: configs.val_due_date_mode (fallback to configs.due_date_mode)
        - Data Mode: 'uniform' (matches dynamic subproblems)
        """
        train_due_mode = str(getattr(configs, "due_date_mode", "k"))
        val_due_mode_raw = str(getattr(configs, "val_due_date_mode", "") or "").strip()
        due_mode = val_due_mode_raw if val_due_mode_raw else train_due_mode
        print("-" * 25 + f"Generating Validation Suite (Mode: {mode} | Train Due: {train_due_mode} | Val Due: {due_mode})" + "-" * 25)
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
                dd = generate_due_dates(jl, pt, due_date_mode=due_mode, seed=vali_seed)
                size_jl.append(jl); size_pt.append(pt); size_dd.append(dd)
            
            vali_batches.append({'n_j': n_j, 'jl': size_jl, 'pt': size_pt, 'dd': size_dd})
            print(f"Generated Validation Batch for Size {n_j} ({due_mode} mode)")
            
        configs.n_j = old_n_j
        return vali_batches

    def train(self):
        setup_seed(self.config.seed_train)
        self.log, self.detailed_log, self.validation_log, self.validation_tardiness_log, self.loss_log = [], [], [], [], []
        self.record = float('inf')

        print("-" * 25 + "Training Setting" + "-" * 25)
        train_due_mode = str(getattr(configs, 'due_date_mode', 'k'))
        val_due_mode_raw = str(getattr(configs, 'val_due_date_mode', '') or '').strip()
        val_due_mode = val_due_mode_raw if val_due_mode_raw else train_due_mode
        print(f"Model: {self.model_name} | Data: Pure Uniform | TrainDue: {train_due_mode} | ValDue: {val_due_mode}")

        self.train_st = time.time()
        
        def create_stage(n_j, multiplier):
            reset_step = int(5 * n_j)
            return {
                "n_j": n_j, "reset_step": reset_step, "duration": int(reset_step * multiplier),
                "stage_label": f"J{n_j}_Uniform_{str(getattr(configs, 'due_date_mode', 'k')).upper()}"
            }

        def sample_uniform_job_size(low, high, update_idx):
            rng = random.Random(int(self.config.seed_train) + 100000 + int(update_idx))
            return rng.randint(low, high)

        def resolve_stage_job_size(stage_cfg, update_idx):
            if "n_j_range" in stage_cfg:
                return sample_uniform_job_size(stage_cfg["n_j_range"][0], stage_cfg["n_j_range"][1], update_idx)
            return stage_cfg["n_j"]

        # Supported schedules:
        # - u10_50: sample n_j uniformly from [10, 50] at each reset.
        # - otherwise: fixed n_j from configs.n_j.
        if str(getattr(configs, "schedule_type", "")).lower() == "u10_50":
            curriculum_schedule = [{
                "n_j_range": (10, 50),
                "reset_step": 1,
                "duration": int(getattr(configs, "max_updates", 1000)),
                "stage_label": f"JU10_50_Uniform_{str(getattr(configs, 'due_date_mode', 'k')).upper()}",
            }]
        else:
            fixed_n_j = int(getattr(configs, "n_j", self.initial_n_j))
            curriculum_schedule = [{
                "n_j": fixed_n_j,
                "reset_step": int(5 * fixed_n_j),
                "duration": int(getattr(configs, "max_updates", 1000)),
                "stage_label": f"J{fixed_n_j}_Uniform_{str(getattr(configs, 'due_date_mode', 'k')).upper()}",
            }]
        
        self.max_updates = int(sum(stage['duration'] for stage in curriculum_schedule))
        
        current_stage_idx = 0
        current_cfg = curriculum_schedule[0]
        configs.n_j = resolve_stage_job_size(current_cfg, 0)
        current_reset_step = current_cfg["reset_step"]
        current_stage_end_step = current_cfg["duration"]
        current_stage_duration = current_cfg["duration"]
        stage_start_step = 0
        
        import math
        self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            if i_update >= current_stage_end_step and current_stage_idx < len(curriculum_schedule) - 1:
                current_stage_idx += 1
                current_cfg = curriculum_schedule[current_stage_idx]
                configs.n_j = resolve_stage_job_size(current_cfg, i_update)
                current_reset_step = current_cfg["reset_step"]
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
                configs.n_j = resolve_stage_job_size(current_cfg, i_update)
                if "n_j_range" in current_cfg:
                    tqdm.write(f"Sampled mixed training size n_j={configs.n_j} at update {i_update+1}")
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
            td_mode_rollout = str(getattr(configs, "ll_td_mode", "mean_pt")).strip().lower()
            use_legacy_split_redistribution = (td_mode_rollout == "mean_pt_split_ops_legacy")

            while True:
                self.memory.push(state)
                with torch.no_grad():
                    batch_idx = ~torch.from_numpy(self.env.done_flag).to(state.fea_j_tensor.device)
                    valid_action_counts = (~state.dynamic_pair_mask_tensor.reshape(state.dynamic_pair_mask_tensor.size(0), -1)).sum(dim=1)
                    if (valid_action_counts[batch_idx] == 0).any():
                        bad_local = torch.where(valid_action_counts[batch_idx] == 0)[0]
                        bad_global = torch.where(batch_idx)[0][bad_local].detach().cpu().tolist()
                        raise RuntimeError(
                            "rollout encountered all-masked rows before policy_old: "
                            f"env_idx={bad_global}, "
                            f"done_flag={self.env.done_flag[bad_global].tolist()}, "
                            f"unscheduled_op_nums={self.env.unscheduled_op_nums[bad_global].tolist()}, "
                            f"candidate={self.env.candidate[bad_global].tolist()}, "
                            f"valid_action_counts={valid_action_counts[bad_global].detach().cpu().tolist()}"
                        )
                    pi_valid, vals_valid = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor[batch_idx], op_mask=state.op_mask_tensor[batch_idx], candidate=state.candidate_tensor[batch_idx],
                        fea_m=state.fea_m_tensor[batch_idx], mch_mask=state.mch_mask_tensor[batch_idx], comp_idx=state.comp_idx_tensor[batch_idx],
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx], fea_pairs=state.fea_pairs_tensor[batch_idx]
                    )
                action_valid, action_logprob_valid = sample_action(pi_valid)

                full_actions = torch.zeros((self.num_envs, 1), dtype=action_valid.dtype, device=action_valid.device)
                full_logprobs = torch.zeros((self.num_envs, 1), dtype=action_logprob_valid.dtype, device=action_logprob_valid.device)
                full_vals = torch.zeros((self.num_envs, 1), dtype=vals_valid.dtype, device=vals_valid.device)
                full_actions[batch_idx] = action_valid
                full_logprobs[batch_idx] = action_logprob_valid
                full_vals[batch_idx] = vals_valid

                state, reward, done, info = self.env.step(actions=full_actions.cpu().numpy())
                ep_mk_gain += np.mean(info['reward_mk']); ep_td_penalty += np.mean(info['reward_td'])
                all_mk_rewards.extend(info['reward_mk'].flatten()); all_td_rewards.extend(info['reward_td'].flatten())
                ep_rewards += reward
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(torch.from_numpy(reward).to(device))
                self.memory.action_seq.append(full_actions.squeeze(-1))
                self.memory.log_probs.append(full_logprobs.squeeze(-1))
                self.memory.val_seq.append(full_vals.squeeze(1))
                # Record rollout metadata for TD redistribution mode.
                if use_legacy_split_redistribution:
                    if not hasattr(self, "_rollout_jobs_seq"):
                        self._rollout_jobs_seq = []
                        self._rollout_active_seq = []
                        self._rollout_td_step_seq = []
                    chosen_jobs_np = (full_actions.squeeze(-1).detach().cpu().numpy() // self.env.number_of_machines).astype(np.int32)
                    active_mask_np = batch_idx.detach().cpu().numpy().astype(bool)
                    td_step_np = np.asarray(info.get('reward_td_step', np.zeros(self.num_envs)), dtype=np.float64)
                    self._rollout_jobs_seq.append(chosen_jobs_np)
                    self._rollout_active_seq.append(active_mask_np)
                    self._rollout_td_step_seq.append(td_step_np)
                if done.all(): break

            # Optional TD credit redistribution: spread each job's final TD evenly across all its op decisions.
            if use_legacy_split_redistribution and len(self.memory.reward_seq) > 0:
                T = len(self.memory.reward_seq)
                E = self.num_envs
                rewards_mat = np.stack([r.detach().cpu().numpy() for r in self.memory.reward_seq], axis=0).astype(np.float64)
                jobs_mat = np.stack(self._rollout_jobs_seq, axis=0).astype(np.int32)
                active_mat = np.stack(self._rollout_active_seq, axis=0).astype(bool)
                td_old_mat = np.stack(self._rollout_td_step_seq, axis=0).astype(np.float64)
                td_new_mat = np.zeros_like(td_old_mat)

                for e in range(E):
                    job_to_steps = {}
                    for t in range(T):
                        if not active_mat[t, e]:
                            continue
                        j = int(jobs_mat[t, e])
                        if j not in job_to_steps:
                            job_to_steps[j] = []
                        job_to_steps[j].append(t)
                    for _, step_ids in job_to_steps.items():
                        if not step_ids:
                            continue
                        td_total = float(np.sum(td_old_mat[step_ids, e]))
                        if abs(td_total) <= 1e-12:
                            continue
                        # Geometric weighting with larger weights on later ops:
                        # e.g., for 5 ops -> [0.9^4, 0.9^3, 0.9^2, 0.9^1, 1.0]
                        # while preserving total TD.
                        n_steps = len(step_ids)
                        weights = np.power(0.9, np.arange(n_steps - 1, -1, -1, dtype=np.float64))
                        weights_sum = float(np.sum(weights))
                        if weights_sum <= 1e-12:
                            continue
                        td_new_mat[step_ids, e] = td_total * (weights / weights_sum)

                rewards_mat = rewards_mat - td_old_mat + td_new_mat
                self.memory.reward_seq = [torch.from_numpy(rewards_mat[t].astype(np.float32)).to(device) for t in range(T)]
                ep_rewards = np.sum(rewards_mat, axis=0)
                all_td_rewards = td_new_mat.reshape(-1).tolist()

            # Clear rollout metadata buffers for next update.
            if hasattr(self, "_rollout_jobs_seq"):
                self._rollout_jobs_seq.clear()
                self._rollout_active_seq.clear()
                self._rollout_td_step_seq.clear()

            loss, v_loss, p_loss = self.ppo.update(self.memory)
            v_term_abs = abs(float(v_loss) * float(getattr(configs, "vloss_coef", 1.0)))
            p_term_abs = abs(float(p_loss) * float(getattr(configs, "ploss_coef", 1.0)))
            vp_den = v_term_abs + p_term_abs + 1e-8
            v_share_loss = v_term_abs / vp_den
            p_share_loss = p_term_abs / vp_den
            self.memory.clear_memory()

            mk_std, td_std = np.std(all_mk_rewards), np.std(all_td_rewards)
            mk_mean = float(np.mean(all_mk_rewards)) if all_mk_rewards else 0.0
            td_mean = float(np.mean(all_td_rewards)) if all_td_rewards else 0.0
            mk_abs_sum = float(np.sum(np.abs(all_mk_rewards))) if all_mk_rewards else 0.0
            td_abs_sum = float(np.sum(np.abs(all_td_rewards))) if all_td_rewards else 0.0
            total_abs_sum = mk_abs_sum + td_abs_sum
            if total_abs_sum > 1e-12:
                mk_share = mk_abs_sum / total_abs_sum
                td_share = td_abs_sum / total_abs_sum
            else:
                mk_share = 0.0
                td_share = 0.0
            self.log.append([i_update, np.mean(ep_rewards)])
            self.detailed_log.append([
                i_update,
                np.mean(ep_rewards),
                ep_mk_gain,
                mk_std,
                ep_td_penalty,
                td_std,
                mk_mean,
                td_mean,
                mk_share,
                td_share,
                np.mean(self.env.current_makespan),
                np.mean(self.env.accumulated_tardiness)
            ])
            self.loss_log.append([i_update, loss, v_loss, p_loss, v_share_loss, p_share_loss])

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
                tqdm.write(
                    f'Update {i_update+1}/{self.max_updates} | '
                    f'R: {avg_reward:.2f} | Loss: {loss:.4f} | V-Loss: {v_loss:.4f} | '
                    f'Vshare: {v_share_loss*100:5.1f}% | Pshare: {p_share_loss*100:5.1f}% | '
                    f'MK_r: {mk_mean:.4f} ({mk_share*100:5.1f}%) | '
                    f'TD_r: {td_mean:.4f} ({td_share*100:5.1f}%) | '
                    f'Vali MK: {overall_ms_mean:.1f} | Vali TD: {overall_td_mean:.1f} | Best TD: {self.record:.1f}'
                )
            else:
                avg_reward = np.mean(ep_rewards)
                tqdm.write(
                    f'Update {i_update+1}/{self.max_updates} | '
                    f'R: {avg_reward:.2f} | Loss: {loss:.4f} | V-Loss: {v_loss:.4f} | '
                    f'Vshare: {v_share_loss*100:5.1f}% | Pshare: {p_share_loss*100:5.1f}% | '
                    f'MK_r: {mk_mean:.4f} ({mk_share*100:5.1f}%) | '
                    f'TD_r: {td_mean:.4f} ({td_share*100:5.1f}%)'
                )

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
        due_mode = str(getattr(configs, "due_date_mode", "k"))
        # [UPDATED] 100% Uniform for consistency
        for i in range(self.num_envs):
            instance_seed = self.config.seed_train + i_update * self.num_envs + i
            # Force 'uniform' mode
            JobLength, OpPT, _ = SD2_instance_generator(config=self.config, seed=instance_seed, mode='uniform')
            DueDate = generate_due_dates(JobLength, OpPT, due_date_mode=due_mode, seed=instance_seed)
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
    configs.data_source, configs.data_suffix = 'SD2', 'mix'
    trainer = Trainer(configs)
    trainer.train()

if __name__ == '__main__': main()
