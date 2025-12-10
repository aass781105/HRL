from model.main_model_dynamic import *
from common_utils import eval_actions
import torch.nn as nn
import torch
from copy import deepcopy
from params import configs
import numpy as np


class Memory:
    def __init__(self, gamma, gae_lambda):
        """
            the memory used for collect trajectories for PPO training
        :param gamma: discount factor
        :param gae_lambda: GAE parameter for PPO algorithm
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables of DANIEL
        self.fea_j_seq = []  # [N, tensor[sz_b, N, 8]]
        self.op_mask_seq = []  # [N, tensor[sz_b, N, 3]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, 6]]
        self.mch_mask_seq = []  # [N, tensor[sz_b, M, M]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]

        # other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]

    def clear_memory(self):
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def clear_state(self):
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]

    def push(self, state):
        """
            push a state into the memory
        :param state: the MDP state
        :return:
        """
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)

    def transpose_data(self):
        """
            transpose the first and second dimension of collected variables
        """
        # 14
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Compete_m_seq = torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)

        return t_Fea_j_seq, t_op_mask_seq, t_Fea_m_seq, t_mch_mask_seq, t_dynamicMask_seq, \
               t_Compete_m_seq, t_candidate_seq, t_pairMessage_seq, \
               t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq

    def get_gae_advantages(self):
        """
            Compute the generalized advantage estimates
        :return: un-normalized advantage sequences, state value sequence
        """

        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):

            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        # [sz_b, N]
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        # [sz_b, N]
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

        # MODIFICATION: Return un-normalized advantages. Normalization will be done in the update function.
        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    def __init__(self, config):
        """
            The implementation of PPO algorithm
        :param config: a package of parameters
        """
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau

        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size

        self.policy = DANIEL(config)
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

            
    def update(self, memory):
        """
        :param memory: data used for PPO training
        :return: total_loss and critic_loss
        """

        # ===== 1. 先在 CPU 上展平成一大條 =====
        t_data = memory.transpose_data()
        # t_data 是 list，裡面是 fea_j_seq, op_mask_seq, fea_m_seq, ... 等 tensor
        # 全部先丟回 CPU，避免一次卡死 GPU
        t_data = [x.detach().cpu() for x in t_data]

        # 優勢 & value target 也丟回 CPU
        unnormalized_advantage, v_target_seq = memory.get_gae_advantages()  # Now returns un-normalized advantages
        unnormalized_advantage = unnormalized_advantage.detach().cpu()
        v_target_seq    = v_target_seq.detach().cpu()
        
        # 在標準化之前計算統計量
        adv_std_before_norm = unnormalized_advantage.std().item()
        
        # 現在於此處進行標準化
        t_advantage_seq = (unnormalized_advantage - unnormalized_advantage.mean()) / (adv_std_before_norm + 1e-8)

        # 統一成 [N,1]
        if t_advantage_seq.dim() == 1:
            t_advantage_seq = t_advantage_seq.unsqueeze(-1)
        if v_target_seq.dim() == 1:
            v_target_seq = v_target_seq.unsqueeze(-1)

        full_batch_size = len(t_data[-1])  # N
        num_batch = int(np.ceil(full_batch_size / self.minibatch_size))

        loss_epochs = 0.0
        v_loss_epochs = 0.0
        p_loss_epochs = 0.0  # Added for plotting policy loss
        
        with torch.no_grad():
            old_vals_for_log = memory.t_old_val_seq.flatten(0, 1).mean().item()

        for _ in range(self.k_epochs):

            for i in range(num_batch):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size

                # ===== 2. 這個 mini-batch 的資料才搬到 GPU =====
                fea_j_b    = t_data[0][start_idx:end_idx].to(self.device)
                op_mask_b  = t_data[1][start_idx:end_idx].to(self.device)
                fea_m_b    = t_data[2][start_idx:end_idx].to(self.device)
                mch_mask_b = t_data[3][start_idx:end_idx].to(self.device)
                dyn_mask_b = t_data[4][start_idx:end_idx].to(self.device)
                comp_idx_b = t_data[5][start_idx:end_idx].to(self.device)
                cand_b     = t_data[6][start_idx:end_idx].to(self.device)
                fea_pair_b = t_data[7][start_idx:end_idx].to(self.device)

                action_batch = t_data[8][start_idx:end_idx].to(self.device).long()
                old_logprobs = t_data[12][start_idx:end_idx].to(self.device)
                if old_logprobs.dim() == 1:
                    old_logprobs = old_logprobs.unsqueeze(-1)

                advantages_b = t_advantage_seq[start_idx:end_idx].to(self.device)
                v_target_b   = v_target_seq[start_idx:end_idx].to(self.device)

                # ===== 3. 前向、loss 計算都在 GPU 上 =====
                pis, vals = self.policy(
                    fea_j=fea_j_b,
                    op_mask=op_mask_b,
                    candidate=cand_b,
                    fea_m=fea_m_b,
                    mch_mask=mch_mask_b,
                    comp_idx=comp_idx_b,
                    dynamic_pair_mask=dyn_mask_b,
                    fea_pairs=fea_pair_b
                )

                logprobs, ent = eval_actions(pis, action_batch)  # [B,1], [B,1]
                ratios = torch.exp(logprobs - old_logprobs.detach())  # [B,1]

                surr1 = ratios * advantages_b
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_b

                v_loss = self.V_loss_2(vals.squeeze(1), v_target_b.squeeze(1))
                p_loss = -torch.min(surr1, surr2).mean()
                ent_loss = ent.mean()
                ent_reg = -ent_loss

                loss = (
                    self.vloss_coef * v_loss +
                    self.ploss_coef * p_loss +
                    self.entloss_coef * ent_reg
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epochs += loss.detach()
                v_loss_epochs += v_loss.detach()
                p_loss_epochs += p_loss.detach() # Added for plotting policy loss


                # （可選）釋放暫存，稍微幫助碎片整理
                del (fea_j_b, op_mask_b, fea_m_b, mch_mask_b,
                    dyn_mask_b, comp_idx_b, cand_b, fea_pair_b,
                    action_batch, old_logprobs, advantages_b, v_target_b,
                    pis, vals, logprobs, ent, ratios, surr1, surr2, loss, v_loss, p_loss, ent_loss, ent_reg)
                torch.cuda.empty_cache()

        # ===== 4. soft update policy_old =====
        for p_old, p in zip(self.policy_old.parameters(), self.policy.parameters()):
            p_old.data.copy_(self.tau * p_old.data + (1 - self.tau) * p.data)

        # ===== [CRITIC DIAGNOSTIC LOG] =====
        # k_epochs and num_batch could be zero if the batch is empty.
        if self.k_epochs > 0 and num_batch > 0:
            avg_v_loss = v_loss_epochs.item() / (self.k_epochs * num_batch)
        else:
            avg_v_loss = 0.0
            
        mean_actual_return = v_target_seq.mean().item() if v_target_seq.numel() > 0 else 0.0
        batch_steps = len(t_data[0]) if t_data and len(t_data) > 0 else 0

        # print(f"[CRITIC DIAGNOSTIC] Steps: {batch_steps:<4} | Mean Actual Return (G_t): {mean_actual_return:<8.4f} | Mean Predicted Value V(s): {old_vals_for_log:<8.4f} | Mean Value Loss: {avg_v_loss:<8.4f}")

        return loss_epochs.item() / self.k_epochs, v_loss_epochs.item() / self.k_epochs, p_loss_epochs.item() / self.k_epochs



def PPO_initialize():
    ppo = PPO(config=configs)
    return ppo
