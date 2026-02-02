from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import SD2_instance_generator, generate_due_dates # Imported generate_due_dates
from common_utils import strToSuffix, setup_seed
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums # Always use various for dynamic
from copy import deepcopy
import os
import random
import time
import sys
from model.PPO import PPO_initialize
from model.PPO import Memory

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)



class Trainer:
    def __init__(self, config):

        # Initial n_j for the curriculum
        self.initial_n_j = 10
        self.fixed_n_m = config.n_m # Machine count remains constant
        
        # Update config with initial values for the first instance generation
        configs.n_j = self.initial_n_j 
        configs.data_source = 'SD2' # Ensure SD2 for curriculum training
        configs.data_suffix = 'mix' # Ensure mix for SD2 generator
        
        self.n_j = configs.n_j # Use updated config.n_j
        self.n_m = configs.n_m # Use updated config.n_m

        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source # Will be 'SD2' now
        self.config = config # Store config object

        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs

        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Model name will reflect the initial job size and then dynamically change
        self.data_name = f'{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(config.data_suffix)}' 
        self.model_name = configs.eval_model_name

        # --- New: Dynamic Fixed Validation Suite (150 Instances) ---
        self.vali_data_batches = self.generate_fixed_validation_data()

        # [NEW] In-Distribution Validation Suite (Manually Defined Groups)
        # These 4 groups run throughout the entire training to monitor cross-size performance.
        self.indist_schedule = [
            # {"name": "G1", "n_j": 10, "k": 0.8, "num": 100},
            {"name": "G2", "n_j": 10, "k": 1.2, "num": 100},
            {"name": "G3", "n_j": 20, "k": 1.2, "num": 100},
            # {"name": "G4", "n_j": 20, "k": 2.4, "num": 100},
        ]
        
        self.vali_indist_batches = self.generate_indist_validation_data(self.indist_schedule)
        self.vali_indist_logs = { g['name']: [] for g in self.indist_schedule }
        
        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def generate_indist_validation_data(self, schedule):
        """
        Generates fixed validation batches based on manual schedule.
        """
        print("-" * 25 + "Generating In-Dist Validation Suite" + "-" * 25)
        batches = {}
        old_n_j = configs.n_j
        
        for group in schedule:
            name = group['name']
            n_j = group['n_j']
            k = group['k']
            num = group['num']
            
            configs.n_j = n_j
            jl_list, pt_list, dd_list = [], [], []
            
            for i in range(num):
                # Seed base 2000+
                s = 2000 + n_j * 100 + i
                jl, pt, _ = SD2_instance_generator(configs, seed=s)
                dd = generate_due_dates(jl, pt, tightness=k)
                jl_list.append(jl); pt_list.append(pt); dd_list.append(dd)
            
            batches[name] = {'n_j': n_j, 'jl': jl_list, 'pt': pt_list, 'dd': dd_list, 'k': k}
            print(f"Generated In-Dist Batch {name}: Size {n_j}, k={k}")
            
        configs.n_j = old_n_j
        return batches

    def generate_fixed_validation_data(self):
        """
        Generates fixed validation instances: 
        Sizes: [10, 15, 20] jobs
        K Values: [0.6, 0.8, 1.0, 1.2, 1.5]
        Count: 10 per combination.
        Total: 3 * 5 * 10 = 150.
        Batched by size for efficient inference.
        """
        print("-" * 25 + "Generating Validation Suite" + "-" * 25)
        k_levels = [0.6, 0.8, 1.0, 1.2, 1.5]
        sizes = [10, 15, 20]
        num_per_combo = 10
        total_instances = len(sizes) * len(k_levels) * num_per_combo
        
        print(f"Creating {total_instances} fixed instances (Sizes: {sizes} | K: {k_levels})")
        
        vali_batches = []
        old_n_j = configs.n_j
        
        for n_j in sizes:
            configs.n_j = n_j
            size_jl = []
            size_pt = []
            size_dd = []
            
            for k_idx, k in enumerate(k_levels):
                for i in range(num_per_combo):
                    # [FIX] Use a fixed seed for validation instances, independent of training seed.
                    # This ensures the validation suite is identical across all experiments.
                    vali_seed = 1000 + n_j * 100 + k_idx * 10 + i
                    jl, pt, _ = SD2_instance_generator(configs, seed=vali_seed)
                    dd = generate_due_dates(jl, pt, tightness=k)
                    size_jl.append(jl)
                    size_pt.append(pt)
                    size_dd.append(dd)
            
            # One batch per size (50 instances)
            vali_batches.append({
                'n_j': n_j,
                'jl': size_jl,
                'pt': size_pt,
                'dd': size_dd
            })
        
        configs.n_j = old_n_j # Restore global config
        print(f"Validation Suite Ready: {len(vali_batches)} batches (one per size).")
        return vali_batches

    def train(self):
        """
            train the model following the config, with curriculum learning
        """
        setup_seed(self.config.seed_train)
        self.log = []
        self.detailed_log = [] # [ADDED] [ep, r, mk_mean, mk_std, td_mean, td_std]
        self.validation_log = []
        self.validation_tardiness_log = []
        self.loss_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"Curriculum Training from {self.initial_n_j} up to 20 jobs.")
        print("\n")

        self.train_st = time.time()
        
        # --- Curriculum Learning Initial Setup ---
        fixed_n_m = self.fixed_n_m
        
        # --- Curriculum Learning Initial Setup ---
        fixed_n_m = self.fixed_n_m
        
        # Stage-based Curriculum Schedules
        # Each stage lasts for 'cycle_len' updates
        cycle_len = configs.curriculum_cycle
        
        # Difficulty Distributions (Unified for all stages)
        # 20x1.0 (Very Tight), 60x1.2 (Tight), 20x1.5 (Moderate) = 100 total
        k_s1 = [1]*20 + [1.2]*60 + [1.5]*20
        k_s2 = [0.6] * 20 + [0.8] *20 + [1.0] *20 + [1.2] * 20 + [1.5] * 20
        # [NEW] Helper function for automatic k scaling
        def scale_k_distribution(base_dist, base_n, target_n, power):
            ratio = (target_n / base_n) ** power
            return [round(k * ratio, 2) for k in base_dist]

        # Automatic generation for s2 (power=0.5) and s3 (power=0.8) with Dual Loop & Dynamic Duration
        
        k08 = [0.8] * 100
        k1 = [1.0] *100 
        k12 = [1.2] *100 
        k15 = [1.5] * 100 
        k24 = [2.4] *100
        k20 = [2.0] *100
        k166 = [1.66] *100
        k147 = [1.47] *100
        k170 = [1.7] * 100
        k195 = [1.95] * 100
        k209 = [2.09] *100
        k224 = [2.24] *100
        
        # [NEW] Helper for Dual Loop Schedule Generation with Hybrid Mixing
        def generate_dual_loop_schedule(job_sizes, k_mix_stages, power, duration_multiplier=3):
            """
            job_sizes: list of n_j (e.g., [10, 15, 20])
            k_mix_stages: list of dicts specifying mix, e.g. [{"name": "EasyStart", "mix": {1.5: 0.5, 1.2: 0.5}}]
            power: scaling exponent
            duration_multiplier: how many reset_steps per sub-stage
            """
            schedule = []
            for n_j in job_sizes:
                # Dynamic Reset Step: Scales with job size (10->50, 20->100)
                reset_step = int(50 * (n_j / 10))
                # Dynamic Duration based on multiplier
                duration = reset_step * duration_multiplier
                
                for stage_cfg in k_mix_stages:
                    mix_def = stage_cfg["mix"]
                    
                    # Construct the final 100-element distribution list
                    current_k_dist = []
                    for base_k, weight in mix_def.items():
                        # Scale the base k
                        scaled_k = round(base_k * ((n_j / 10) ** power), 2)
                        # Add to list based on weight (assuming weight * 100 is int)
                        count = int(weight * 100)
                        current_k_dist.extend([scaled_k] * count)
                    
                    # Ensure it's exactly 100 (handle rounding)
                    if len(current_k_dist) < 100:
                        current_k_dist.extend([current_k_dist[-1]] * (100 - len(current_k_dist)))
                    
                    schedule.append({
                        "n_j": n_j,
                        "reset_step": reset_step,
                        "duration": duration,
                        "k_dist": current_k_dist,
                        "stage_label": stage_cfg.get("name", "unnamed")
                    })
            return schedule

        # --- Define your Due Date Stages here ---
        # Proportions must sum to 1.0
        my_k_mix_stages2 = [
            # {"name": "StartMixed", "mix": {1.5: 0.5, 1.2: 0.5}}, # 50% Easy, 50% Medium
            {"name": "MidHard",    "mix": {1.5: 0.2, 1.2: 0.6, 1.0: 0.2}}, # 20% Easy, 60% Medium, 20% Hard
            {"name": "FocusHard",  "mix": {1.2: 0.3, 1.0: 0.5, 0.8: 0.2}}  # 30% Medium, 70% Hard
        ]
        my_k_mix_stages3 = [
            {"name": "StartMixed", "mix": {1.5: 0.5, 1.2: 0.5}}, # 50% Easy, 50% Medium
            {"name": "MidHard",    "mix": {1.5: 0.2, 1.2: 0.6, 1.0: 0.2}}, # 20% Easy, 60% Medium, 20% Hard
            {"name": "FocusHard",  "mix": {1.2: 0.3, 1.0: 0.5, 0.8: 0.2}}  # 30% Medium, 70% Hard
        ]
    
        my_k_mix_stages4 = [
            {"name": "MidHard",    "mix": {1.5: 0.2, 1.2: 0.6, 1.0: 0.2}}, # 20% Easy, 60% Medium, 20% Hard
            {"name": "FocusHard",  "mix": {1.2: 0.3, 1.0: 0.5, 0.8: 0.2}},  # 30% Medium, 70% Hard
            {"name": "FocusHard",  "mix": {1.0: 0.3, 0.8: 0.5, 0.6: 0.2}}  # 30% Medium, 70% Hard
        ]

        if configs.schedule_type == 's1':
            # Classic s1 (Fixed duration based on cycle_len)
            cycle = configs.curriculum_cycle
            curriculum_schedule = [
                {"n_j": 10, "reset_step": 50, "duration": cycle, "k_dist": k_s1},
                {"n_j": 15, "reset_step": 125, "duration": cycle, "k_dist": k_s1},
                {"n_j": 20, "reset_step": 125, "duration": cycle, "k_dist": k_s1},
                {"n_j": 25, "reset_step": 250, "duration": cycle, "k_dist": k_s1},
            ]
        elif configs.schedule_type == 's2_2':
            # Advanced Hybrid Mode: Job [10,15,20,25] x K-Mix (Decaying)
            # Power 0.5 (Sqrt)
            job_sizes = [10, 15, 20]
            curriculum_schedule = generate_dual_loop_schedule(job_sizes, my_k_mix_stages2, power=0.5, duration_multiplier=2)
            
        elif configs.schedule_type == 's2_3':
            # Power 0.8
            job_sizes = [10, 15, 20]
            curriculum_schedule = generate_dual_loop_schedule(job_sizes, my_k_mix_stages3, power=0.5, duration_multiplier=3)

        elif configs.schedule_type == 's2_4':
            # Power 0.8
            job_sizes = [10, 15, 20]
            curriculum_schedule = generate_dual_loop_schedule(job_sizes, my_k_mix_stages4, power=0.5, duration_multiplier=3)

        elif configs.schedule_type.startswith('same'):
            # Legacy single stage mode
            # Extract params if possible or fallback to specific logic
            cycle = 1000
            if configs.schedule_type == 'same_10_5_12':
                 curriculum_schedule = [{"n_j": 10, "reset_step": 100, "duration": cycle, "k_dist": k12}]
            else:
                 # Default fallback for same_*
                 curriculum_schedule = [{"n_j": 10, "reset_step": 50, "duration": cycle, "k_dist": k12}]
        else:
            # Fallback
            curriculum_schedule = [{"n_j": 10, "reset_step": 50, "duration": 250, "k_dist": k12}]
        
        # [NEW] Auto-calculate Max Updates
        total_duration = sum(stage['duration'] for stage in curriculum_schedule)
        self.max_updates = total_duration
        tqdm.write(f"Auto-configured Max Updates: {self.max_updates} based on schedule duration.")

        # Initialize
        current_stage_idx = 0
        current_cfg = curriculum_schedule[0]
        configs.n_j = current_cfg["n_j"]
        current_reset_step = current_cfg["reset_step"]
        current_k_dist = current_cfg["k_dist"]
        current_stage_end_step = current_cfg["duration"] # Cumulative step when this stage ends
        current_stage_duration = current_cfg["duration"] # Duration of current stage for LR calculation
        stage_start_step = 0 # When did this stage start
        
        import math # Ensure math is available
        tqdm.write(f"Starting curriculum training ({configs.schedule_type}) with {configs.n_j} jobs. Reset every {current_reset_step} steps.")
        self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # 1. Check for Curriculum Stage Update (Dynamic Duration Logic)
            if i_update >= current_stage_end_step and current_stage_idx < len(curriculum_schedule) - 1:
                current_stage_idx += 1
                current_cfg = curriculum_schedule[current_stage_idx]
                
                configs.n_j = current_cfg["n_j"]
                current_reset_step = current_cfg["reset_step"]
                current_k_dist = current_cfg["k_dist"]
                
                # Update timing markers
                stage_start_step = current_stage_end_step
                current_stage_duration = current_cfg["duration"]
                current_stage_end_step += current_stage_duration
                
                tqdm.write(f"\nCURRICULUM UPDATE: Stage {current_stage_idx+1}. Job Size={configs.n_j}, K~{current_k_dist[0]:.2f}, Reset={current_reset_step}, Duration={current_stage_duration}")

            # 2. Check for Environment Reset (Modulo Logic)
            if i_update % current_reset_step == 0:
                dataset_job_length, dataset_op_pt, dataset_due_date = self.sample_training_instances(i_update, current_k_dist)
                
                # [ADDED] Reconstruct the k-values used for this batch to pass to environment
                k_list_used = [current_k_dist[i] if i < len(current_k_dist) else 1.2 for i in range(self.num_envs)]
                
                self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_due_date, 
                                                  true_due_date_list=dataset_due_date, tightness=k_list_used)
            else:
                state = self.env.reset()

            # --- [NEW] Sawtooth Learning Rate Scheduler (Adapted for Dynamic Duration) ---
            # Global Trend: Peak LR decays by 0.95 every stage
            peak_lr = configs.lr * (0.95 ** current_stage_idx)
            
            # Intra-Cycle: Cosine Annealing (1.0 -> 0.1) based on stage progress
            # Calculate progress within current stage (0.0 -> 1.0)
            steps_in_stage = i_update - stage_start_step
            # Avoid division by zero if duration is 0 (unlikely)
            cycle_progress = min(1.0, steps_in_stage / max(1, current_stage_duration))
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_progress)) # 1.0 -> 0.0
            # Remap to 1.0 -> 0.1 range
            current_decay = 0.1 + 0.9 * cosine_decay
            
            # Final LR calculation with floor clamp
            current_lr = peak_lr * current_decay
            current_lr = max(current_lr, 5e-5) # Global minimum floor
            
            # Update Optimizer
            for param_group in self.ppo.optimizer.param_groups:
                param_group['lr'] = current_lr
                
            # --- [NEW] Fixed Small Entropy Coefficient ---
            # Set to a value provided by configs (default 0.01) for easier adjustment
            self.ppo.entloss_coef = configs.entloss_coef
            # ----------------------------------------------------------------
            
            # [FIXED] Initialize rewards to zero to maximize signal-to-noise ratio
            # This makes PPO focus on learning marginal gains rather than constant baseline.
            ep_rewards = np.zeros(self.num_envs)
            ep_mk_gain = 0.0
            ep_td_penalty = 0.0
            
            # [ADDED] Lists to collect all step rewards for Std calculation
            all_mk_rewards = []
            all_td_rewards = []

            while True:
                self.memory.push(state)
                with torch.no_grad():
                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,
                                                             op_mask=state.op_mask_tensor,
                                                             candidate=state.candidate_tensor,
                                                             fea_m=state.fea_m_tensor,
                                                             mch_mask=state.mch_mask_tensor,
                                                             comp_idx=state.comp_idx_tensor,
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                                             fea_pairs=state.fea_pairs_tensor)

                action_envs, action_logprob_envs = sample_action(pi_envs)
                state, reward, done, info = self.env.step(actions=action_envs.cpu().numpy())
                
                # Accumulate Mean
                ep_mk_gain += np.mean(info['reward_mk'])
                ep_td_penalty += np.mean(info['reward_td'])
                
                # Collect for Std
                all_mk_rewards.extend(info['reward_mk'].flatten())
                all_td_rewards.extend(info['reward_td'].flatten())
                
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs.squeeze(-1))
                self.memory.log_probs.append(action_logprob_envs.squeeze(-1))
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss, p_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            mean_tardiness_all_env = np.mean(self.env.accumulated_tardiness)
            
            # [ADDED] Calculate Std
            mk_std = np.std(all_mk_rewards)
            td_std = np.std(all_td_rewards)

            self.log.append([i_update, mean_rewards_all_env])
            
            # [CHANGED] Append raw metrics to detailed log (8 columns)
            self.detailed_log.append([i_update, mean_rewards_all_env, 
                                      ep_mk_gain, mk_std, 
                                      ep_td_penalty, td_std,
                                      mean_makespan_all_env, mean_tardiness_all_env])
                                      
            self.loss_log.append([i_update, loss, v_loss, p_loss])

            if (i_update + 1) % self.validate_timestep == 0:
                # 1. Standard Validation (Fixed Benchmark)
                vali_makespan, vali_tardiness = self.validate_envs_with_various_op_nums()
                vali_result = vali_makespan.mean()
                vali_tardiness_mean = vali_tardiness.mean()
                
                # Calculate weighted objective score for saving best model
                current_score = 0.5 * vali_result + 0.5 * vali_tardiness_mean

                if current_score < self.record:
                    self.save_model()
                    self.record = current_score

                self.validation_log.append(vali_result)
                self.validation_tardiness_log.append(vali_tardiness_mean)
                
                # 2. In-Distribution Validation (Current vs Other Sizes)
                self.run_indist_validation()
                
                self.save_validation_log()
                tqdm.write(f'Vali Quality: {vali_result:.2f} | Tardiness: {vali_tardiness_mean:.2f} | Score: {current_score:.2f} (Best: {self.record:.2f})')

            ep_et = time.time()
            tqdm.write(
                'Episode {}\t R: {:.4f} (Mk: {:.4f}±{:.4f}, Td: {:.4f}±{:.4f})\t makespan: {:.2f}\t tardiness: {:.2f}\t Total_loss: {:.4f}\t P_loss: {:.4f}'.format(
                    i_update + 1, mean_rewards_all_env, ep_mk_gain, mk_std, ep_td_penalty, td_std, mean_makespan_all_env, mean_tardiness_all_env, loss, p_loss))

        self.train_et = time.time()
        self.save_training_log()

    def save_training_log(self):
        log_model_name = f'{self.model_name}_{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(self.config.data_suffix)}'
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + log_model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))
        
        # [ADDED] Save detailed log
        file_writing_obj_d = open(f'./train_log/{self.data_source}/' + 'detailed_reward_' + log_model_name + '.txt', 'w')
        file_writing_obj_d.write(str(self.detailed_log))

        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + log_model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))
        
        file_writing_obj2 = open(f'./train_log/{self.data_source}/' + 'valitardiness_' + log_model_name + '.txt', 'w')
        file_writing_obj2.write(str(self.validation_tardiness_log))
        
        file_writing_obj_loss = open(f'./train_log/{self.data_source}/' + 'loss_' + log_model_name + '.txt', 'w')
        file_writing_obj_loss.write(str(self.loss_log))

        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{log_model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        log_model_name = f'{self.model_name}_{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(self.config.data_suffix)}'
        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + log_model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))
        
        file_writing_obj2 = open(f'./train_log/{self.data_source}/' + 'valitardiness_' + log_model_name + '.txt', 'w')
        file_writing_obj2.write(str(self.validation_tardiness_log))
        
        # [ADDED] Save In-Dist logs for each group with full metadata in filename
        for name, logs in self.vali_indist_logs.items():
            # Find original group info
            g = next(x for x in self.indist_schedule if x['name'] == name)
            # Format metadata tag
            meta_tag = f"{name}_nj{g['n_j']}_k{g['k']}_num{g['num']}"
            f_path = f'./train_log/{self.data_source}/vali_indist_{meta_tag}_{log_model_name}.txt'
            with open(f_path, 'w') as f:
                f.write(str(logs))

    def run_indist_validation(self):
        """
        Runs the model against all fixed in-distribution batches and stores raw [MK, TD].
        """
        self.ppo.policy.eval()
        for name, batch in self.vali_indist_batches.items():
            temp_env = FJSPEnvForVariousOpNums(n_j=batch['n_j'], n_m=self.fixed_n_m)
            # Use normalize_due_date=True for static validation
            state = temp_env.set_initial_data(batch['jl'], batch['pt'], batch['dd'], 
                                              normalize_due_date=True, true_due_date_list=batch['dd'])

            while True:
                with torch.no_grad():
                    batch_idx = ~torch.from_numpy(temp_env.done_flag)
                    if batch_idx.any():
                        pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],
                                                op_mask=state.op_mask_tensor[batch_idx],
                                                candidate=state.candidate_tensor[batch_idx],
                                                fea_m=state.fea_m_tensor[batch_idx],
                                                mch_mask=state.mch_mask_tensor[batch_idx],
                                                comp_idx=state.comp_idx_tensor[batch_idx],
                                                dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                                                fea_pairs=state.fea_pairs_tensor[batch_idx])
                        action = greedy_select_action(pi)
                        state, _, done, _ = temp_env.step(action.cpu().numpy())
                    else:
                        break
                if done.all():
                    break
            
            # Record mean raw metrics for this group
            avg_ms = np.mean(temp_env.current_makespan)
            avg_td = np.mean(temp_env.accumulated_tardiness)
            self.vali_indist_logs[name].append([avg_ms, avg_td])
            
        self.ppo.policy.train()

    def sample_training_instances(self, i_update, k_list=None):
        dataset_JobLength = []
        dataset_OpPT = []
        dataset_DueDate = []
        
        # [FIX] Use the k_list passed from curriculum_schedule. 
        # Fallback to a default if None.
        if k_list is None:
            k_list = [1.2] * self.num_envs
        
        # [NEW] Data Augmentation Strategy
        # 50% Uniform (Base stability)
        # 50% Realistic (Vdata-like: Mixed flexibility + Op-dependent PT)
        generation_modes = ['uniform'] * 50 + ['realistic'] * 50
        
        for i in range(self.num_envs):
            # Deterministic seed for each instance in each update
            instance_seed = self.config.seed_train + i_update * self.num_envs + i
            
            # Randomly select a generation mode for this specific instance
            # Use deterministic choice based on seed to ensure reproducibility if needed
            mode = generation_modes[instance_seed % 100]
            
            JobLength, OpPT, _ = SD2_instance_generator(config=self.config, seed=instance_seed, mode=mode)
            
            # Use the specific k-value for all jobs in this instance
            k_val = k_list[i] if i < len(k_list) else 1.2
            DueDate = generate_due_dates(JobLength, OpPT, tightness=k_val)
            
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)
            dataset_DueDate.append(DueDate)
        return dataset_JobLength, dataset_OpPT, dataset_DueDate

    def validate_envs_with_various_op_nums(self):
        self.ppo.policy.eval()
        all_ms = []
        all_td = []

        for batch in self.vali_data_batches:
            # Re-initialize environment for specific batch size (n_j)
            temp_env = FJSPEnvForVariousOpNums(n_j=batch['n_j'], n_m=self.fixed_n_m)
            state = temp_env.set_initial_data(batch['jl'], batch['pt'], batch['dd'], true_due_date_list=batch['dd'])

            while True:
                with torch.no_grad():
                    batch_idx = ~torch.from_numpy(temp_env.done_flag)
                    if batch_idx.any():
                        pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],
                                                op_mask=state.op_mask_tensor[batch_idx],
                                                candidate=state.candidate_tensor[batch_idx],
                                                fea_m=state.fea_m_tensor[batch_idx],
                                                mch_mask=state.mch_mask_tensor[batch_idx],
                                                comp_idx=state.comp_idx_tensor[batch_idx],
                                                dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                                                fea_pairs=state.fea_pairs_tensor[batch_idx])
                        action = greedy_select_action(pi)
                        state, _, done, _ = temp_env.step(action.cpu().numpy())
                    else:
                        break
                if done.all():
                    break
            
            all_ms.extend(temp_env.current_makespan)
            all_td.extend(temp_env.accumulated_tardiness)

        self.ppo.policy.train()
        return np.array(all_ms), np.array(all_td)

    def save_model(self):
        # save_model_name = 'due_date_ppo'
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}.pth')

    def load_model(self):
        load_model_name = f'{self.model_name}_{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(self.config.data_suffix)}'
        model_path = f'./trained_network/{self.data_source}/{load_model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))


def main():
    # Set seed immediately to ensure validation data is consistent across runs
    setup_seed(configs.seed_train)

    configs.n_j = 10 
    configs.n_m = 5 
    configs.data_source = 'SD2'
    configs.data_suffix = 'mix'

    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
