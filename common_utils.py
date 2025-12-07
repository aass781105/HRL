import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


# def sample_action(p):
#     """
#         sample an action by the distribution p
#     :param p: this distribution with the probability of choosing each action
#     :return: an action sampled by p
#     """
#     dist = Categorical(p)
#     s = dist.sample()  # index
#     return s, dist.log_prob(s)

def sample_action(pi: torch.Tensor, action_mask: torch.Tensor = None):
    """
    pi: [B, A]，A = J_max * n_m，為已 softmax 後的機率分佈
    action_mask: 
        - None：不做額外遮罩
        - [B, A] 或 [B, J_max, n_m] 的 Bool tensor
          True  = 允許選
          False = 禁用（例如 dummy job）
    回傳:
        action:       [B, 1] LongTensor
        action_logp:  [B, 1] FloatTensor
    """
    probs = pi

    if action_mask is not None:
        # 將 mask reshape 成 [B, A]
        if action_mask.dim() == 3:
            # [B, J_max, n_m] -> [B, A]
            mask_flat = action_mask.reshape(action_mask.size(0), -1).to(torch.bool)
        else:
            # [B, A]
            mask_flat = action_mask.to(torch.bool)

        if mask_flat.shape != probs.shape:
            raise ValueError(
                f"action_mask shape {mask_flat.shape} != pi shape {probs.shape}"
            )

        # 禁用的 action 機率設為 0
        probs = probs.clone()
        probs = probs * mask_flat  # False 位置變 0

        # 重新 normalize，每個 batch 都要確保 sum>0
        sums = probs.sum(dim=-1, keepdim=True)  # [B,1]
        # 若整行都被 mask 掉（sum=0），fallback 成均勻分配在「原本 mask=True 的位置」
        zero_sum = (sums <= 0)
        if zero_sum.any():
            # 在 zero_sum 的 batch 裡，用 mask_flat 當作均勻分配基底
            fallback = mask_flat.float()
            fallback_sums = fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
            fallback = fallback / fallback_sums
            probs[zero_sum] = fallback[zero_sum]
            sums = probs.sum(dim=-1, keepdim=True)

        probs = probs / sums

    dist = Categorical(probs=probs)
    action = dist.sample()                   # [B]
    log_prob = dist.log_prob(action)         # [B]

    return action.unsqueeze(-1), log_prob.unsqueeze(-1)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


# def greedy_select_action(p):
#     _, index = torch.max(p, dim=1)
#     return index


def greedy_select_action(pi: torch.Tensor, action_mask: torch.Tensor = None):
    """
    pi: [B, A]，通常為 logits 或已 softmax 的分數都可以（只是 argmax）
    action_mask: 
        - None：不做遮罩
        - [B, A] 或 [B, J_max, n_m] 的 Bool tensor
          True  = 允許選
          False = 禁用
    回傳:
        action: [B, 1] LongTensor
    """
    scores = pi

    if action_mask is not None:
        if action_mask.dim() == 3:
            mask_flat = action_mask.reshape(action_mask.size(0), -1).to(torch.bool)
        else:
            mask_flat = action_mask.to(torch.bool)

        if mask_flat.shape != scores.shape:
            raise ValueError(
                f"action_mask shape {mask_flat.shape} != pi shape {scores.shape}"
            )

        # 禁用位置設為極小值，避免被 argmax 選中
        scores = scores.clone()
        scores = scores.masked_fill(~mask_flat, float('-1e9'))

    action = torch.argmax(scores, dim=-1, keepdim=True)  # [B,1]
    return action





def build_job_mask(num_jobs_real: int, num_jobs_max: int, device=None) -> torch.Tensor:
    """
    建立 job mask：
      長度 num_jobs_max
      前 num_jobs_real 個為 True（真實 job），其餘為 False（dummy job）。
    回傳 shape: [1, J_max] BoolTensor
    """
    num_jobs_real = int(num_jobs_real)
    num_jobs_max = int(num_jobs_max)
    if num_jobs_real > num_jobs_max:
        raise ValueError(f"num_jobs_real ({num_jobs_real}) > num_jobs_max ({num_jobs_max})")

    mask = torch.zeros(num_jobs_max, dtype=torch.bool, device=device)
    if num_jobs_real > 0:
        mask[:num_jobs_real] = True
    return mask.unsqueeze(0)  # [1, J_max]



def build_action_mask(job_mask: torch.Tensor, n_m: int, batch_size: int = 1) -> torch.Tensor:
    """
    由 job_mask 建立 action_mask：
      job_mask: [1, J_max] or [B, J_max] BoolTensor
      n_m:      機台數
    回傳:
      action_mask: [B, J_max * n_m] BoolTensor
        True  = 允許選
        False = 禁用（dummy job）
    """
    if job_mask.dim() == 1:
        job_mask = job_mask.unsqueeze(0)  # [1, J_max]

    B_j, J_max = job_mask.shape
    if B_j == 1 and batch_size > 1:
        job_mask = job_mask.repeat(batch_size, 1)
    elif B_j != batch_size:
        # 若 batch_size 跟 job_mask 不一致，先以 job_mask 的 batch 為準
        batch_size = B_j

    # job_mask: [B, J_max]
    # 展開到 [B, J_max, n_m] 再攤平成 [B, J_max * n_m]
    job_mask_expanded = job_mask.unsqueeze(-1).expand(batch_size, J_max, n_m)  # [B, J_max, n_m]
    action_mask = job_mask_expanded.reshape(batch_size, J_max * n_m)

    return action_mask




def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    """
    :param chosen_job: the selected job
    :param env: the production environment
    :return: the machines which can immediately process the chosen job
    """
    mch_state = ~env.candidate_process_relation[0, chosen_job]
    available_mch_list = np.where(mch_state == True)[0]
    mch_free_time = env.mch_free_time[0][available_mch_list]
    job_free_time = env.candidate_free_time[0][chosen_job]
    # case1 eg:
    # JF: 50
    # MchF: 55 60 65 70
    if (job_free_time < mch_free_time).all():
        chosen_mch_list = available_mch_list[min_element_index(mch_free_time)]
    # case2 eg:
    # JF: 50
    # MchF: 35 40 55 60
    else:
        chosen_mch_list = available_mch_list[np.where(mch_free_time <= job_free_time)]

    return chosen_mch_list


def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)

    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[0][available_mchs]
        chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action


"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('123')
