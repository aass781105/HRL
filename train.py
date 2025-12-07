import os
import sys
import time
import random

import numpy as np
from tqdm import tqdm

import torch

from common_utils import (
    strToSuffix,
    setup_seed,
    sample_action,
    greedy_select_action,
)
from params import configs
from model.PPO_dynamic import PPO_initialize, Memory
from PPO_dynamic_wrapper import DynamicToStaticEnvWrapper  # 注意：用你現在的檔名

# 時間字串
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

# GPU 裝置
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
device = torch.device(configs.device)


# ======================= 小工具：對 state 做 padding =======================

def pad_state_for_memory(state, max_ops: int, max_jobs: int):
    """
    將 EnvState 裡「跟 operation 數 N」與「job 數 J」有關的張量 padding 成固定大小：
      - fea_j_tensor:          [B, N, Dj]   -> [B, max_ops, Dj]
      - op_mask_tensor:        [B, N, 3]    -> [B, max_ops, 3]
      - candidate_tensor:      [B, J]       -> [B, max_jobs]
      - dynamic_pair_mask:     [B, J, M]    -> [B, max_jobs, M]
      - fea_pairs_tensor:      [B, J, M, K] -> [B, max_jobs, M, K]
      - comp_idx_tensor:       [B, M, M, J] -> [B, M, M, max_jobs]
    這些都是 PPO.Memory 在 transpose_data() 時會 stack 的維度，
    必須在「進 Memory 之前」就對齊。
    """

    # -------------------- 1) op 維度（N） --------------------
    fj = state.fea_j_tensor               # [B, N, Dj]
    B, N, Dj = fj.shape

    if N > max_ops:
        raise RuntimeError(
            f"[pad_state_for_memory] N={N} > max_ops={max_ops}，"
            f"請把 configs.max_ops_global 設大一點 (例如 n_j*n_m*2)。"
        )

    if N < max_ops:
        pad_N = max_ops - N
        pad_fj = torch.zeros((B, pad_N, Dj), dtype=fj.dtype, device=fj.device)
        state.fea_j_tensor = torch.cat([fj, pad_fj], dim=1)  # [B, max_ops, Dj]

    # op_mask_tensor: [B, N, 3] -> [B, max_ops, 3]
    om = state.op_mask_tensor
    B2, N2, C = om.shape
    assert B2 == B and N2 == N, \
        "[pad_state_for_memory] op_mask_tensor 與 fea_j_tensor 的 batch/N 不一致"

    if N2 < max_ops:
        pad_N = max_ops - N2
        pad_om = torch.zeros((B2, pad_N, C), dtype=om.dtype, device=om.device)
        state.op_mask_tensor = torch.cat([om, pad_om], dim=1)

    # -------------------- 2) job 維度（J） --------------------
    cand = state.candidate_tensor         # [B, J]
    Bc, J = cand.shape

    if J > max_jobs:
        raise RuntimeError(
            f"[pad_state_for_memory] J={J} > max_jobs={max_jobs}，"
            f"請把 configs.max_jobs_global 或 configs.n_j 設大一點。"
        )

    # dynamic_pair_mask: [B, J, M]
    dpm = state.dynamic_pair_mask_tensor  # [B, J, M]
    Bd, Jd, M = dpm.shape
    assert Bd == Bc and Jd == J, \
        "[pad_state_for_memory] dynamic_pair_mask_tensor 與 candidate_tensor 的 batch/J 不一致"

    # fea_pairs: [B, J, M, K]
    fp = state.fea_pairs_tensor           # [B, J, M, K]
    Bf, Jf, Mf, K = fp.shape
    assert Bf == Bc and Jf == J and Mf == M, \
        "[pad_state_for_memory] fea_pairs_tensor 與 candidate/dynamic_pair_mask 不一致"

    # comp_idx: [B, M, M, J]
    ci = state.comp_idx_tensor            # [B, M, M, J]
    Bc2, M1, M2, Jc = ci.shape
    assert Bc2 == Bc and M1 == M and M2 == M and Jc == J, \
        "[pad_state_for_memory] comp_idx_tensor 的 J 或 M 維度不一致"

    if J < max_jobs:
        pad_J = max_jobs - J

        # candidate: padded job 位置設為 0（反正 dynamic_pair_mask 那邊會全部 True 遮掉）
        pad_cand = torch.zeros((Bc, pad_J), dtype=cand.dtype, device=cand.device)
        state.candidate_tensor = torch.cat([cand, pad_cand], dim=1)  # [B, max_jobs]

        # dynamic_pair_mask: padded job 全 True = 全部 pair 不可選
        pad_dpm = torch.ones((Bd, pad_J, M), dtype=dpm.dtype, device=dpm.device)
        state.dynamic_pair_mask_tensor = torch.cat([dpm, pad_dpm], dim=1)  # [B, max_jobs, M]

        # fea_pairs: padded job 全 0 特徵
        pad_fp = torch.zeros((Bf, pad_J, Mf, K), dtype=fp.dtype, device=fp.device)
        state.fea_pairs_tensor = torch.cat([fp, pad_fp], dim=1)            # [B, max_jobs, M, K]

        # comp_idx: padded job 全 0 = 不參與任何競爭
        pad_ci = torch.zeros((Bc2, M1, M2, pad_J), dtype=ci.dtype, device=ci.device)
        state.comp_idx_tensor = torch.cat([ci, pad_ci], dim=3)             # [B, M, M, max_jobs]

    return state


# ================================ Trainer ================================

class Trainer:
    def __init__(self, config):

        self.config = config
        self.n_j = config.n_j
        self.n_m = config.n_m
        self.data_source = config.data_source

        self.max_updates = config.max_updates
        self.validate_timestep = config.validate_timestep

        # ---- padding 上界 ----
        # op 上界（N_max）：預設用 n_j * n_m，你之後可以在 params 加 max_ops_global 來覆蓋
        self.max_ops_global = int(
            getattr(config, "max_ops_global", config.n_j * config.n_m)
        )
        # job 上界（J_max）：預設用 n_j，你也可以加 max_jobs_global 來覆蓋
        self.max_jobs_global = int(
            getattr(config, "max_jobs_global", config.n_j)
        )

        # 建資料夾
        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        # 設定預設 tensor type
        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # 命名
        self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'
        self.model_name = f'dynamic_PPO{strToSuffix(config.model_suffix)}'

        # random seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        # === 動態→靜態環境（訓練 / 驗證） ===
        self.env = DynamicToStaticEnvWrapper(
            cfg=config,
            n_machines=config.n_m,
        )
        self.vali_env = DynamicToStaticEnvWrapper(
            cfg=config,
            n_machines=config.n_m,
        )

        # === PPO / 記憶體 ===
        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    # ------------------------------------------------------------------
    # 訓練主流程
    # ------------------------------------------------------------------
    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source            : {self.data_source}")
        print(f"model name        : {self.model_name}")
        print(f"n_j, n_m          : {self.n_j}, {self.n_m}")
        print(f"max_updates       : {self.max_updates}")
        print(f"validate_timestep : {self.validate_timestep}")
        print(f"device            : {device}")
        print(f"max_ops_global    : {self.max_ops_global}")
        print(f"max_jobs_global   : {self.max_jobs_global}")
        print(f"interarrival_mean : {getattr(self.config, 'interarrival_mean', 'NA')}")
        print(f"burst_size        : {getattr(self.config, 'burst_size', 'NA')}")
        print(f"event_horizon     : {getattr(self.config, 'event_horizon', 'NA')}")
        print(f"cadence_choices   : {getattr(self.config, 'cadence_choices', None)}")
        print(f"cadence_min/max   : {getattr(self.config, 'cadence_min', None)}, "
              f"{getattr(self.config, 'cadence_max', None)}")
        print("\n")

        self.train_st = time.time()

        # === 外層：每個 update 一個 episode ===
        for i_update in tqdm(range(self.max_updates), file=sys.stdout,
                             desc="progress", colour='blue'):
            ep_st = time.time()

            # reset 一個新的動態 episode
            raw_state, info = self.env.reset()
            state = pad_state_for_memory(raw_state, self.max_ops_global, self.max_jobs_global)

            ep_rewards = 0.0
            done_flag = False

            ep_metrics = self.env.get_episode_metrics()
            cadence_N = ep_metrics.get("cadence_N", None)

            # === 內層：與 wrapper 互動直到 episode 結束 ===
            while not done_flag:
                # 1) push state
                self.memory.push(state)

                with torch.no_grad():
                    pi_envs, vals_envs = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor,
                        op_mask=state.op_mask_tensor,
                        candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor,
                        mch_mask=state.mch_mask_tensor,
                        comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                        fea_pairs=state.fea_pairs_tensor
                    )

                # 2) sample action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # 3) 丟給動態 wrapper 走一步
                raw_next_state, reward, done = self.env.step(action_envs.cpu().numpy())
                next_state = pad_state_for_memory(raw_next_state, self.max_ops_global, self.max_jobs_global)

                reward_np = np.asarray(reward, dtype=np.float32)
                done_np = np.asarray(done).astype(bool)

                ep_rewards += float(reward_np.mean())
                reward_t = torch.from_numpy(reward_np).to(device)

                # 4) 收集 transition
                self.memory.done_seq.append(torch.from_numpy(done_np).to(device))
                self.memory.reward_seq.append(reward_t)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                # 5) 更新 state / done_flag
                state = next_state
                done_flag = bool(done_np.all())

            # === 一個 episode 結束 → PPO update ===
            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            ep_metrics = self.env.get_episode_metrics()
            episode_makespan = float(ep_metrics.get("final_makespan", 0.0))

            self.log.append([i_update, ep_rewards])

            # 驗證
            if (i_update + 1) % self.validate_timestep == 0:
                vali_mks = self.validate_dynamic_env()
                vali_result = float(vali_mks.mean())

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(
                    f'[VALIDATE] update={i_update+1}, '
                    f'mean makespan={vali_result:.2f} (best={self.record:.2f})'
                )

            ep_et = time.time()
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t '
                'Mean_loss: {:.8f},  training time: {:.2f}s, cadence_N: {}'.format(
                    i_update + 1, ep_rewards, episode_makespan,
                    loss, ep_et - ep_st, cadence_N
                )
            )

        self.train_et = time.time()
        self.save_training_log()
        self.save_model_final()

    # ------------------------------------------------------------------
    # 驗證：在動態環境中用 greedy policy 跑幾個 episode
    # ------------------------------------------------------------------
    def validate_dynamic_env(self, episodes: int = None) -> np.ndarray:
        self.ppo.policy.eval()

        if episodes is None:
            episodes = int(getattr(self.config, "ddqn_val_episodes", 5))

        makespans = []

        for _ in range(episodes):
            raw_state, info = self.vali_env.reset()
            state = pad_state_for_memory(raw_state, self.max_ops_global, self.max_jobs_global)

            done_flag = False

            while not done_flag:
                with torch.no_grad():
                    pi, _ = self.ppo.policy(
                        fea_j=state.fea_j_tensor,
                        op_mask=state.op_mask_tensor,
                        candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor,
                        mch_mask=state.mch_mask_tensor,
                        comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                        fea_pairs=state.fea_pairs_tensor
                    )
                action = greedy_select_action(pi)
                raw_state, reward, done = self.vali_env.step(action.cpu().numpy())
                state = pad_state_for_memory(raw_state, self.max_ops_global, self.max_jobs_global)

                done_np = np.asarray(done).astype(bool)
                done_flag = bool(done_np.all())

            metrics = self.vali_env.get_episode_metrics()
            makespans.append(float(metrics.get("final_makespan", 0.0)))

        self.ppo.policy.train()
        return np.array(makespans, dtype=np.float32)

    # ------------------------------------------------------------------
    # Log / Model I/O
    # ------------------------------------------------------------------
    def save_training_log(self):
        with open(f'./train_log/{self.data_source}/reward_{self.model_name}.txt', 'w') as f:
            f.write(str(self.log))

        with open(f'./train_log/{self.data_source}/valiquality_{self.model_name}.txt', 'w') as f:
            f.write(str(self.validation_log))

        with open('./train_time.txt', 'a') as f:
            f.write(
                f'model path: ./trained_network/{self.data_source}/{self.model_name}\t\t'
                f'training time: {round((self.train_et - self.train_st), 2)}\t\t'
                f'local time: {str_time}\n'
            )

    def save_validation_log(self):
        with open(f'./train_log/{self.data_source}/valiquality_{self.model_name}.txt', 'w') as f:
            f.write(str(self.validation_log))

    def save_model(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}_dynamic.pth')
    
    def save_model_final(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}_dynamic_final.pth')
        
    def load_model(self):
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location=device))


def main():
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
