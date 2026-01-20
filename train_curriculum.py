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
        self.model_name = f'mix_1_div_njob' 

        # --- New: Dynamic Fixed Validation Suite (120 Instances) ---
        self.vali_data_batches = self.generate_fixed_validation_data()
        
        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def generate_fixed_validation_data(self):
        """
        Generates 120 fixed validation instances: 
        Sizes: [10, 15, 20] jobs
        Profiles: [Easy, Moderate, Hard, Crisis]
        Count: 10 per combination.
        Total: 3 * 4 * 10 = 120.
        Batched by 20 for efficient inference.
        """
        print("-" * 25 + "Generating Validation Suite" + "-" * 25)
        print("Creating 120 fixed instances (Sizes: 10, 15, 20 | All Profiles incl. Crisis)")
        
        vali_batches = []
        sizes = [10, 15, 20]
        profiles = ['easy', 'moderate', 'hard', 'crisis']
        num_per_profile = 10
        
        old_n_j = configs.n_j
        
        for n_j in sizes:
            configs.n_j = n_j
            size_jl = []
            size_pt = []
            size_dd = []
            
            for profile in profiles:
                for _ in range(num_per_profile):
                    jl, pt, _ = SD2_instance_generator(configs)
                    # Note: data_utils.generate_due_dates handles 'crisis' specifically if passed as tightness
                    # ensuring crisis instances can still be generated for validation even if excluded from random_mix
                    dd = generate_due_dates(jl, pt, tightness=profile)
                    size_jl.append(jl)
                    size_pt.append(pt)
                    size_dd.append(dd)
            
            # Split 40 instances of this size into 2 batches of 20
            for i in range(0, 40, 20):
                vali_batches.append({
                    'n_j': n_j,
                    'jl': size_jl[i:i+20],
                    'pt': size_pt[i:i+20],
                    'dd': size_dd[i:i+20]
                })
        
        configs.n_j = old_n_j # Restore global config
        print(f"Validation Suite Ready: {len(vali_batches)} batches of 20 instances.")
        return vali_batches

    def train(self):
        """
            train the model following the config, with curriculum learning
        """
        setup_seed(self.config.seed_train)
        self.log = []
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
        current_n_j = self.initial_n_j
        fixed_n_m = self.fixed_n_m
        configs.n_j = current_n_j 
        
        tqdm.write(f"Starting curriculum training with {current_n_j} jobs and {fixed_n_m} machines.")
        self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            if (i_update + 1) % 100 == 0:
                current_n_j = random.randint(10, 20)
                configs.n_j = current_n_j 
                tqdm.write(f"\nCURRICULUM UPDATE: Job count for next instances set to {current_n_j}. (Update @ ep {i_update+1})")

            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt, dataset_due_date = self.sample_training_instances(i_update)
                self.env = FJSPEnvForVariousOpNums(n_j=configs.n_j, n_m=configs.n_m)
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_due_date, true_due_date_list=dataset_due_date)
            else:
                state = self.env.reset()
            
            # # --- Entropy Decay Logic ---
            # ent_start = 0.05
            # ent_end = 0.005
            # # Linear decay from ent_start to ent_end over max_updates
            # current_ent_coef = ent_start - (ent_start - ent_end) * (i_update / max(1, self.max_updates - 1))
            # current_ent_coef = max(ent_end, current_ent_coef) 
            current_ent_coef = 0.01
            
            self.ppo.entloss_coef = current_ent_coef
            # ----------------------------------------------------------------
            
            ep_rewards = - deepcopy(self.env.init_quality)

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
                state, reward, done, _ = self.env.step(actions=action_envs.cpu().numpy())
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

            self.log.append([i_update, mean_rewards_all_env])
            self.loss_log.append([i_update, loss, v_loss, p_loss])

            if (i_update + 1) % self.validate_timestep == 0:
                vali_makespan, vali_tardiness = self.validate_envs_with_various_op_nums()
                vali_result = vali_makespan.mean()
                vali_tardiness_mean = vali_tardiness.mean()
                
                # Calculate weighted objective score for saving best model
                # Using 0.5/0.5 weight as the evaluation standard
                current_score = 0.5 * vali_result + 0.5 * vali_tardiness_mean

                if current_score < self.record:
                    self.save_model()
                    self.record = current_score

                self.validation_log.append(vali_result)
                self.validation_tardiness_log.append(vali_tardiness_mean)
                self.save_validation_log()
                tqdm.write(f'Vali Quality: {vali_result:.2f} | Tardiness: {vali_tardiness_mean:.2f} | Score: {current_score:.2f} (Best: {self.record:.2f})')

            ep_et = time.time()
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t tardiness: {:.2f}\t Total_loss: {:.4f}\t P_loss: {:.4f}\t ent: {:.3f}\t training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, mean_tardiness_all_env, loss, p_loss, current_ent_coef, ep_et - ep_st))

        self.train_et = time.time()
        self.save_training_log()

    def save_training_log(self):
        log_model_name = f'{self.model_name}_{self.initial_n_j}x{self.fixed_n_m}{strToSuffix(self.config.data_suffix)}'
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + log_model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))

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

    def sample_training_instances(self, i_update):
        dataset_JobLength = []
        dataset_OpPT = []
        dataset_DueDate = []
        for i in range(self.num_envs):
            # Deterministic seed for each instance in each update
            instance_seed = self.config.seed_train + i_update * self.num_envs + i
            
            JobLength, OpPT, _ = SD2_instance_generator(config=self.config, seed=instance_seed)
            # --- Ablation Study: Fixed tightness at 1.2 ---
            DueDate = generate_due_dates(JobLength, OpPT, tightness='random_mix')
            
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
    configs.max_updates = 1000 

    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
