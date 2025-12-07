# train_unified.py
# New main script for simultaneously training the DDQN gatekeeper and PPO scheduler.
import os
import math
import random
import copy
import numpy as np
from collections import deque
from typing import Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from params import configs
from model.PPO_dynamic import PPO, Memory
from common_utils import sample_action

# --- Components from existing project files ---
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator, split_matrix_to_jobs
from data_utils import SD2_instance_generator
from PPO_orchestrator_adapter import OrchestratorAdapter

# --- DDQN Components (brought in from train_ddqn.py for self-containment) ---
class QNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=int(capacity))
    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s).astype(np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.stack(s2).astype(np.float32),
                np.array(d, dtype=np.float32))
    def __len__(self):
        return len(self.buf)

# --- Helper function to get DDQN state (from main.py) ---
def _get_gate_obs(orch: GlobalTimelineOrchestrator, n_machines: int, t_now: float, burst_K: int, obs_buffer_cap: int, norm_scale: float) -> np.ndarray:
    buf_size = len(orch.buffer)
    cap = int(obs_buffer_cap) if int(obs_buffer_cap) > 0 else max(1, int(burst_K) * 3)
    o0 = float(buf_size) / float(cap)

    mft_abs = np.asarray(orch.machine_free_time, dtype=float)
    rem = np.maximum(0.0, mft_abs - float(t_now))
    
    total_rem = float(rem.sum()) / n_machines if rem.size > 0 else 0.0
    first_idle_rem = float(rem.min()) if rem.size > 0 else 0.0

    o1 = total_rem / norm_scale
    o2 = first_idle_rem / norm_scale
    
    return np.array([o0, o1, o2], dtype=np.float32)

def _run_final_flush_and_get_cost(
    orchestrator: GlobalTimelineOrchestrator,
    t_now: float,
    heuristic: str,
    max_flush_rounds: int = 16
) -> Tuple[float, float]:
    """
    Schedules all remaining jobs in the orchestrator's buffer and calculates
    the resulting final makespan and idle time.
    """
    if not orchestrator.buffer:
        return (np.max(orchestrator.machine_free_time) if orchestrator.machine_free_time else 0.0), 0.0

    t_flush_start = float(t_now)
    
    flush_round = 0
    while len(orchestrator.buffer) > 0 and flush_round < max_flush_rounds:
        flush_round += 1
        fin = orchestrator.event_release_and_reschedule(t_now, heuristic)
        if fin.get("event") != "batch_finalized":
            break
            
    mk_after = float(np.max(orchestrator.machine_free_time)) if len(orchestrator.machine_free_time) > 0 else t_flush_start
    
    metrics_flush = orchestrator.compute_interval_metrics(t_flush_start, mk_after)
    total_idle_flush = float(metrics_flush.get("total_idle", 0.0))

    return mk_after, total_idle_flush

def plot_unified_training_results(episodes, ddqn_returns, ddqn_losses, ppo_pi_losses, ppo_v_losses, save_dir):
    """
    Plots the training progress for both DDQN and PPO and saves it to a file.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Unified Training Progress')

    # Plot DDQN Return
    axs[0, 0].plot(episodes, ddqn_returns, label='DDQN Episode Return')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Cumulative Return')
    axs[0, 0].set_title('DDQN Performance')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot DDQN Loss
    axs[0, 1].plot(episodes, ddqn_losses, label='DDQN Avg Loss', color='orange')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Average Loss')
    axs[0, 1].set_title('DDQN Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot PPO Policy Loss
    ppo_episodes = [ep for ep, loss in zip(episodes, ppo_pi_losses) if loss is not None]
    valid_pi_losses = [loss for loss in ppo_pi_losses if loss is not None]
    if ppo_episodes:
        axs[1, 0].plot(ppo_episodes, valid_pi_losses, label='PPO Policy Loss', color='red', marker='o', linestyle='-')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Policy Loss')
    axs[1, 0].set_title('PPO Policy Loss (on update)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot PPO Value Loss
    ppo_episodes_v = [ep for ep, loss in zip(episodes, ppo_v_losses) if loss is not None]
    valid_v_losses = [loss for loss in ppo_v_losses if loss is not None]
    if ppo_episodes_v:
        axs[1, 1].plot(ppo_episodes_v, valid_v_losses, label='PPO Value Loss', color='green', marker='o', linestyle='-')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Value Loss')
    axs[1, 1].set_title('PPO Value Loss (on update)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "unified_training_progress.png")
    plt.savefig(save_path)
    print(f"\nTraining plot saved to {save_path}")
    plt.close(fig)

# --- Main Training Orchestrator ---
def train_unified():
    # --- Seeding ---
    seed = int(getattr(configs, "seed_train", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device(configs.device)

    # --- Hyperparameters ---
    # DDQN
    ddqn_episodes = int(getattr(configs, "ddqn_episodes", 500))
    ddqn_batch_size = int(getattr(configs, "ddqn_batch_size", 128))
    ddqn_gamma = float(getattr(configs, "ddqn_gamma", 0.99))
    ddqn_tau = float(getattr(configs, "ddqn_target_tau", 0.005))
    ddqn_lr = float(getattr(configs, "ddqn_lr", 1e-4))
    eps_start = float(getattr(configs, "ddqn_eps_start", 0.4))
    eps_end = float(getattr(configs, "ddqn_eps_end", 0.01))
    eps_decay_episodes = int(getattr(configs, "ddqn_eps_decay_episodes", 300))
    # Env
    event_horizon = int(getattr(configs, "event_horizon", 100))
    interarrival_mean = float(getattr(configs, "interarrival_mean", 25.0))
    burst_K = int(getattr(configs, "burst_size", 1))
    init_jobs = int(getattr(configs, "init_jobs", 10))
    # Normalization & Reward
    norm_scale = float(getattr(configs, "norm_scale", 100.0))
    reward_scale = float(getattr(configs, "reward_scale", 50.0))
    # Penalty
    enable_final_flush_penalty = bool(getattr(configs, "enable_final_flush_penalty", True))

    ppo_model_path =  str(getattr(configs, "ppo_model_path", ""))


    # --- PPO Agent Initialization ---
    ppo_agent = PPO(configs)
    ppo_memory = Memory(gamma=configs.gamma, gae_lambda=configs.gae_lambda)
    
    # Load pre-trained PPO weights
    # ppo_model_path = 'cotrain_network/curriculum_train_10x5+mix.pth'
    if os.path.exists(ppo_model_path):
        print(f"Loading pre-trained PPO model from {ppo_model_path}")
        ppo_agent.policy.load_state_dict(torch.load(ppo_model_path, map_location=device))
        ppo_agent.policy_old.load_state_dict(torch.load(ppo_model_path, map_location=device))
        print("Pre-trained PPO model loaded successfully.")
    else:
        print(f"Warning: Pre-trained PPO model not found at {ppo_model_path}. Starting PPO from scratch.")

    # --- DDQN Agent Initialization ---
    q_net = QNet(obs_dim=3, hidden=128, n_actions=2).to(device)
    q_target = QNet(obs_dim=3, hidden=128, n_actions=2).to(device)
    q_target.load_state_dict(q_net.state_dict())
    ddqn_optimizer = optim.Adam(q_net.parameters(), lr=ddqn_lr)
    ddqn_loss_fn = nn.SmoothL1Loss()
    ddqn_buffer = ReplayBuffer(capacity=int(getattr(configs, "ddqn_buffer_capacity", 100_000)))

    # --- DDQN Helper Functions ---
    def get_epsilon(ep_idx: int) -> float:
        frac = min(1.0, ep_idx / float(eps_decay_episodes))
        return float(eps_start + (eps_end - eps_start) * frac)
    def select_ddqn_action(state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randint(0, 1)
        with torch.no_grad():
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = q_net(s_tensor)
            return int(torch.argmax(q_values, dim=1).item())
    def soft_update(source: nn.Module, target: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    # --- Data Collectors for Plotting ---
    plot_episodes = []
    ddqn_episode_returns = []
    ddqn_episode_avg_losses = []
    ppo_episode_avg_pi_losses = []
    ppo_episode_avg_v_losses = []

    # --- Main Training Loop ---
    for ep in tqdm(range(1, ddqn_episodes + 1), desc="Episodes"):
        # --- Environment & Orchestrator Reset for each episode ---
        ep_seed = seed + ep
        rng = np.random.default_rng(ep_seed)
        job_generator = EventBurstGenerator( 
            sd2_fn=SD2_instance_generator, base_config=copy.deepcopy(configs), n_machines=configs.n_m,
            interarrival_mean=interarrival_mean, k_sampler=lambda _rng: burst_K, rng=rng,
        )
        orchestrator = GlobalTimelineOrchestrator(
            n_machines=configs.n_m, job_generator=job_generator,
            select_from_buffer=lambda buf, o: list(range(len(buf))), release_decider=None, t0=0.0
        )
        adapter = OrchestratorAdapter(orchestrator, n_machines=configs.n_m)

        # Initial jobs
        if init_jobs > 0:
            # This part is simplified; assumes OrchestratorAdapter handles init if needed
            pass 
        
        t_now = 0.0
        events_done = 0
        ep_return = 0.0
        
        # Data collectors for this episode
        ep_ddqn_losses = []
        ep_ppo_pi_losses = []
        ep_ppo_v_losses = []
        
        mk_prev = 0.0
        if len(orchestrator.machine_free_time) > 0:
            mk_prev = float(np.max(orchestrator.machine_free_time))
            
        # --- Event Loop ---
        while events_done < event_horizon:
            # 1. Get DDQN State
            state_ddqn = _get_gate_obs(orchestrator, configs.n_m, t_now, burst_K, 0, norm_scale)

            # 2. DDQN Action
            epsilon = get_epsilon(ep)
            action_ddqn = select_ddqn_action(state_ddqn, epsilon)

            # 3. Execute Action (RELEASE or HOLD)
            if action_ddqn == 1 and orchestrator.buffer: # RELEASE
                                    # PPO Sub-loop
                                    state_ppo = adapter.begin_new_batch(t_now)
                                    if state_ppo:
                                        while True:
                                            ppo_memory.push(state_ppo)
                                            with torch.no_grad():
                                                pi, val = ppo_agent.policy_old(
                                                    fea_j=state_ppo.fea_j_tensor, op_mask=state_ppo.op_mask_tensor,
                                                    candidate=state_ppo.candidate_tensor, fea_m=state_ppo.fea_m_tensor,
                                                    mch_mask=state_ppo.mch_mask_tensor, comp_idx=state_ppo.comp_idx_tensor,
                                                    dynamic_pair_mask=state_ppo.dynamic_pair_mask_tensor, fea_pairs=state_ppo.fea_pairs_tensor
                                                )
                                            
                                            # Use sample_action to get log_prob for PPO update
                                            action_ppo, log_prob_ppo = sample_action(pi)
                                            
                                            # The adapter returns the PPO state and the reward for the PPO step
                                            state_ppo, reward_ppo, sub_done, _ = adapter.step_in_batch(action_ppo.cpu())
                                            
                                            ppo_memory.done_seq.append(torch.from_numpy(np.array([sub_done])).to(device))
                                            ppo_memory.reward_seq.append(torch.from_numpy(reward_ppo).to(device))
                                            ppo_memory.action_seq.append(action_ppo)
                                            ppo_memory.log_probs.append(log_prob_ppo)
                                            ppo_memory.val_seq.append(val.squeeze(1))
                                            
                                            if sub_done:
                                                break
                                        adapter.finalize_batch()
                                        # PPO Update
                                        if len(ppo_memory.reward_seq) > 0:
                                            _, v_loss, p_loss = ppo_agent.update(ppo_memory)
                                            ep_ppo_pi_losses.append(p_loss)
                                            ep_ppo_v_losses.append(v_loss)
                                        ppo_memory.clear_memory()
            else: # HOLD
                adapter.tick_without_release(t_now)
            
            # 4. Environment Transition
            t_next = job_generator.sample_next_time(t_now)
            
            # Use orchestrator to get metrics for DDQN reward
            metrics = orchestrator.compute_interval_metrics(t_now, t_next)
            total_idle = float(metrics.get("total_idle", 0.0))
            mk_now = float(np.max(orchestrator.machine_free_time)) if len(orchestrator.machine_free_time) > 0 else mk_prev
            delta_mk = mk_now - mk_prev
            
            reward_ddqn = (-delta_mk*0.7 - total_idle*0.3) / reward_scale
            
            t_now = t_next
            events_done += 1
            mk_prev = mk_now

            # Arrive new jobs for the *next* step
            new_jobs = job_generator.generate_burst(t_now)
            if new_jobs:
                orchestrator.buffer.extend(new_jobs)

            # 5. DDQN Learning
            next_state_ddqn = _get_gate_obs(orchestrator, configs.n_m, t_now, burst_K, 0, norm_scale)
            done = (events_done >= event_horizon)
            
            # --- Final Flush Penalty ---
            flush_heuristic = ""
            if done and enable_final_flush_penalty and len(orchestrator.buffer) > 0:
                mk_final, _ = _run_final_flush_and_get_cost(orchestrator, t_now, flush_heuristic)
                final_flush_penalty = mk_final / (reward_scale /2)
                # reward_ddqn -= len(orchestrator.buffer) *100
                reward_ddqn -= final_flush_penalty

            ddqn_buffer.push(state_ddqn, action_ddqn, reward_ddqn, next_state_ddqn, float(done))
            ep_return += reward_ddqn

            if len(ddqn_buffer) >= ddqn_batch_size:
                s, a, r, s2, d = ddqn_buffer.sample(ddqn_batch_size)
                s, a, r, s2, d = (torch.from_numpy(s).to(device), torch.from_numpy(a).to(device),
                                  torch.from_numpy(r).to(device), torch.from_numpy(s2).to(device),
                                  torch.from_numpy(d).to(device))
                q_sa = q_net(s).gather(1, a.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    a2 = torch.argmax(q_net(s2), dim=1, keepdim=True)
                    q_s2a2 = q_target(s2).gather(1, a2).squeeze(1)
                    y = r + (1.0 - d) * ddqn_gamma * q_s2a2
                loss = ddqn_loss_fn(q_sa, y)
                ep_ddqn_losses.append(loss.item())
                ddqn_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
                ddqn_optimizer.step()
                soft_update(q_net, q_target, ddqn_tau)

        # --- End of Episode: Aggregate and Log Data ---
        plot_episodes.append(ep)
        ddqn_episode_returns.append(ep_return)
        ddqn_episode_avg_losses.append(np.mean(ep_ddqn_losses) if ep_ddqn_losses else 0)
        ppo_episode_avg_pi_losses.append(np.mean(ep_ppo_pi_losses) if ep_ppo_pi_losses else None)
        ppo_episode_avg_v_losses.append(np.mean(ep_ppo_v_losses) if ep_ppo_v_losses else None)

        if ep % 1 == 0:
            tqdm.write(f"New  [EP {ep:04d}] Episode Return={ep_return:.3f} | Final Makespan={mk_now:.2f}")

    # --- Save Models ---
    print("\nTraining finished.")
    output_dir_ppo = "trained_network"
    output_dir_ddqn = "ddqn_ckpt"
    os.makedirs(os.path.join(output_dir_ppo, "unified"), exist_ok=True)
    os.makedirs(output_dir_ddqn, exist_ok=True)
    path_name = str(getattr(configs, "path_name", ""))
    
    ppo_save_path = os.path.join(output_dir_ppo, "unified", f"unified_ppo_final_{ddqn_episodes}ep_{path_name}.pth")
    ddqn_save_path = os.path.join(output_dir_ddqn, f"unified_ddqn_final_{ddqn_episodes}ep_{path_name}.pth")

    torch.save(ppo_agent.policy.state_dict(), ppo_save_path)
    print(f"[SAVED] Unified PPO model to -> {ppo_save_path}")
    torch.save(q_net.state_dict(), ddqn_save_path)
    print(f"[SAVED] Unified DDQN model to -> {ddqn_save_path}")

    # --- Plot Training Results ---
    plot_unified_training_results(
        episodes=plot_episodes,
        ddqn_returns=ddqn_episode_returns,
        ddqn_losses=ddqn_episode_avg_losses,
        ppo_pi_losses=ppo_episode_avg_pi_losses,
        ppo_v_losses=ppo_episode_avg_v_losses,
        save_dir=f"plots/train_unified{path_name}"
    )

if __name__ == "__main__":
    train_unified()
