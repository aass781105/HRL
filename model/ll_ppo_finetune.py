"""
Finetune-specific PPO implementation.

This module isolates dynamic-finetuning experiments from the main lower-level
PPO used by train_curriculum.py. It adds two finetune-only behaviors:
1. Variable-shape padding so multiple subproblems can be updated together.
2. Episode-aware GAE that resets on done boundaries between subproblems.
"""

from model.ll_ppo import LLPPO as BaseLLPPO
from model.ll_ppo import LLMemory as BaseLLMemory
from params import configs
import torch


class LLFinetuneMemory(BaseLLMemory):
    def num_states(self):
        return len(self.reward_seq)

    def num_subproblems(self):
        if not self.done_seq:
            return 0
        total = 0
        for done_t in self.done_seq:
            total += int(done_t.reshape(-1).to(torch.int64).sum().item())
        return total

    @staticmethod
    def _pad_dim1(tensor, target_len, pad_value):
        if tensor.shape[1] == target_len:
            return tensor
        out_shape = list(tensor.shape)
        out_shape[1] = target_len
        out = tensor.new_full(out_shape, pad_value)
        out[:, :tensor.shape[1], ...] = tensor
        return out

    @staticmethod
    def _pad_last_dim(tensor, target_len, pad_value):
        if tensor.shape[-1] == target_len:
            return tensor
        out_shape = list(tensor.shape)
        out_shape[-1] = target_len
        out = tensor.new_full(out_shape, pad_value)
        out[..., :tensor.shape[-1]] = tensor
        return out

    def transpose_data(self):
        if not self.fea_j_seq:
            raise RuntimeError("finetune memory is empty")

        max_n = max(int(x.shape[1]) for x in self.fea_j_seq)
        max_j = max(int(x.shape[1]) for x in self.candidate_seq)

        t_Fea_j_seq = torch.stack(
            [self._pad_dim1(x, max_n, 0.0).squeeze(0) for x in self.fea_j_seq], dim=0
        )
        t_op_mask_seq = torch.stack(
            [self._pad_dim1(x, max_n, False).squeeze(0) for x in self.op_mask_seq], dim=0
        )
        t_Fea_m_seq = torch.stack([x.squeeze(0) for x in self.fea_m_seq], dim=0)
        t_mch_mask_seq = torch.stack([x.squeeze(0) for x in self.mch_mask_seq], dim=0)
        t_dynamicMask_seq = torch.stack(
            [self._pad_dim1(x, max_j, True).squeeze(0) for x in self.dynamic_pair_mask_seq], dim=0
        )
        t_Compete_m_seq = torch.stack(
            [self._pad_last_dim(x, max_j, False).squeeze(0) for x in self.comp_idx_seq], dim=0
        )
        t_candidate_seq = torch.stack(
            [self._pad_dim1(x, max_j, 0).squeeze(0) for x in self.candidate_seq], dim=0
        )
        t_pairMessage_seq = torch.stack(
            [self._pad_dim1(x, max_j, 0.0).squeeze(0) for x in self.fea_pairs_seq], dim=0
        )
        t_action_seq = torch.stack(self.action_seq, dim=0).reshape(-1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).reshape(-1)
        t_val_seq = torch.stack(self.val_seq, dim=0).reshape(-1)
        t_done_seq = torch.stack(self.done_seq, dim=0).reshape(-1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).reshape(-1)

        return (
            t_Fea_j_seq,
            t_op_mask_seq,
            t_Fea_m_seq,
            t_mch_mask_seq,
            t_dynamicMask_seq,
            t_Compete_m_seq,
            t_candidate_seq,
            t_pairMessage_seq,
            t_action_seq,
            t_reward_seq,
            t_val_seq,
            t_done_seq,
            t_logprobs_seq,
        )

    def get_gae_advantages(self, normalize_vtarget=False):
        reward_arr = torch.stack(self.reward_seq, dim=0).reshape(-1).to(torch.float32)
        values = torch.stack(self.val_seq, dim=0).reshape(-1).to(torch.float32)
        done_arr = torch.stack(self.done_seq, dim=0).reshape(-1).to(torch.bool)
        len_trajectory = int(reward_arr.shape[0])

        advantage = torch.zeros(len_trajectory, device=values.device, dtype=torch.float32)
        running_adv = torch.tensor(0.0, device=values.device, dtype=torch.float32)

        for i in reversed(range(len_trajectory)):
            done_i = bool(done_arr[i].item())
            if i == len_trajectory - 1 or done_i:
                next_value = torch.tensor(0.0, device=values.device, dtype=torch.float32)
                next_adv = torch.tensor(0.0, device=values.device, dtype=torch.float32)
            else:
                next_value = values[i + 1]
                next_adv = running_adv

            delta_t = reward_arr[i] + self.gamma * next_value - values[i]
            running_adv = delta_t + self.gamma * self.gae_lambda * next_adv
            advantage[i] = running_adv

        v_target_seq = advantage + values

        if normalize_vtarget:
            v_target_seq = (v_target_seq - v_target_seq.mean()) / (v_target_seq.std(unbiased=False) + 1e-8)

        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        return advantage, v_target_seq


class LLFinetunePPO(BaseLLPPO):
    """
    Dedicated PPO class for dynamic finetuning.

    Keeps the base PPO update logic, but is paired with the finetune-specific
    Memory above that supports padded variable-shape batching.
    """

    pass


def ll_ppo_finetune_initialize():
    return LLFinetunePPO(config=configs)
