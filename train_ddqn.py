# train_ddqn.py
import os
import math
import random
import numpy as np
from collections import deque
from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from event_gate_env import EventGateEnv
from params import configs
from model.ddqn_model import QNet

import matplotlib.pyplot as plt

# --------- Replay Buffer ---------
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

def train_ddqn(
    episodes: int = configs.ddqn_episodes,
    event_horizon: int = int(configs.event_horizon),
    interarrival_mean: float = configs.interarrival_mean,
    burst_K: int = configs.burst_size,
    lr: float = configs.ddqn_lr,
    gamma: float = configs.ddqn_gamma,
    eps_start: float = configs.ddqn_eps_start,
    eps_end: float = configs.ddqn_eps_end,
    eps_decay_episodes: int = configs.ddqn_eps_decay_episodes,
    batch_size: int = configs.ddqn_batch_size,
    buffer_capacity: int = configs.ddqn_buffer_capacity,
    target_tau: float = configs.ddqn_target_tau,
    seed: int = int(getattr(configs, "event_seed", 42)),
    num_envs: int = configs.ddqn_num_envs
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    device = torch.device(getattr(configs, "device", "cpu"))

    def make_env(rank):
        def _thunk():
            return EventGateEnv(
                n_machines=configs.n_m,
                heuristic=configs.scheduler_type,
                interarrival_mean=interarrival_mean,
                burst_K=burst_K,
                event_horizon=event_horizon,
                init_jobs=int(getattr(configs, "init_jobs", 0))
            )
        return _thunk

    # Vectorized Environments (Async for true parallelism)
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    seeds = [seed + i * 100 for i in range(num_envs)]
    states, _ = envs.reset(seed=seeds)
    
    obs_dim = states.shape[1] 
    n_actions = 2

    q = QNet(obs_dim, n_actions=n_actions, hidden=configs.ddqn_hidden_dim, num_layers=configs.ddqn_num_layers).to(device)
    q_tgt = QNet(obs_dim, n_actions=n_actions, hidden=configs.ddqn_hidden_dim, num_layers=configs.ddqn_num_layers).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    buf = ReplayBuffer(capacity=buffer_capacity)

    def epsilon(ep_idx: int) -> float:
        if eps_decay_episodes <= 0: return eps_end
        frac = min(1.0, ep_idx / float(eps_decay_episodes))
        return float(eps_start + (eps_end - eps_start) * frac)

    def select_actions(curr_states: np.ndarray, eps: float) -> np.ndarray:
        actions = []
        for i in range(num_envs):
            if random.random() < eps:
                actions.append(random.randint(0, n_actions - 1))
            else:
                actions.append(-1)
        actions = np.array(actions)
        mask = (actions == -1)
        if mask.any():
            with torch.no_grad():
                s_tensor = torch.as_tensor(curr_states[mask], device=device, dtype=torch.float32)
                qv = q(s_tensor)
                actions[mask] = torch.argmax(qv, dim=1).cpu().numpy()
        return actions

    def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    # Metrics (Final Averaged Points)
    returns, history_loss, history_grad_norm = [], [], []
    train_makespans, train_tardiness, train_rel_counts = [], [], []
    val_episodes, val_makespans, val_tardiness, val_rel_counts = [], [], [], [] 
    history_idle, history_buffer, history_release, history_stability, history_flush = [], [], [], [], []

    # Track data for ongoing episodes in each env
    running_ep_returns = np.zeros(num_envs)
    running_ep_idle = np.zeros(num_envs)
    running_ep_buf = np.zeros(num_envs)
    running_ep_rel = np.zeros(num_envs)
    running_ep_stab = np.zeros(num_envs)
    
    # Buffers to collect data from all envs before averaging
    batch_ep_returns, batch_ep_idle, batch_ep_buf, batch_ep_rel, batch_ep_stab, batch_ep_flush = [], [], [], [], [], []
    batch_ep_td, batch_ep_mk, batch_ep_rel_count = [], [], []
    temp_losses = [] 
    temp_grads = [] # [FIX] Initialize for gradient norm tracking

    pbar = tqdm(total=episodes, desc="Averaged Batches")

    while len(returns) < episodes:
        curr_eps = epsilon(len(returns) + 1)
        actions = select_actions(states, curr_eps)
        
        # Step all environments
        next_states, rewards, dones, truncs, infos = envs.step(actions)

        # Process each environment
        for i in range(num_envs):
            # 1. Push transition to buffer
            buf.push(states[i], actions[i], rewards[i], next_states[i], float(dones[i] or truncs[i]))
            
            # 2. Accumulate metrics safely (Directly from info arrays)
            def get_val(key, idx):
                if key in infos and isinstance(infos[key], (list, np.ndarray)) and idx < len(infos[key]):
                    return float(infos[key][idx])
                return 0.0

            r_idle = get_val("reward_idle_cost", i)
            r_buf = get_val("reward_buffer_penalty", i)
            r_rel = get_val("reward_release_penalty", i)
            r_stab = get_val("reward_stability_penalty", i)

            running_ep_returns[i] += float(rewards[i])
            running_ep_idle[i] += r_idle
            running_ep_buf[i] += r_buf
            running_ep_rel[i] += r_rel
            running_ep_stab[i] += r_stab
            
            if dones[i] or truncs[i]:
                # Extract KPIs directly from the flat info arrays (as per debug output)
                ep_td = get_val("episode_tardiness", i)
                ep_mk = get_val("episode_makespan", i)
                ep_rel_count = int(get_val("release_count", i))
                ep_flush = get_val("reward_final_flush_penalty", i)
                
                # Add to batch buffers
                batch_ep_returns.append(running_ep_returns[i])
                batch_ep_idle.append(running_ep_idle[i])
                batch_ep_buf.append(running_ep_buf[i])
                batch_ep_rel.append(running_ep_rel[i])
                batch_ep_stab.append(running_ep_stab[i])
                batch_ep_flush.append(ep_flush)
                batch_ep_td.append(ep_td)
                batch_ep_mk.append(ep_mk)
                batch_ep_rel_count.append(ep_rel_count)

                # Reset running counters for this env
                running_ep_returns[i] = 0.0
                running_ep_idle[i] = 0.0; running_ep_buf[i] = 0.0
                running_ep_rel[i] = 0.0; running_ep_stab[i] = 0.0

        # Training Step (Sample from Buffer)
        # [MOD] Replay Ratio: Update the network multiple times per step to accelerate convergence
        replay_ratio = int(getattr(configs, "ddqn_replay_ratio", 2))
        if len(buf) >= batch_size:
            for _ in range(replay_ratio):
                s_b, a_b, r_b, s2_b, d_b = buf.sample(batch_size)
                s_b = torch.as_tensor(s_b, device=device); a_b = torch.as_tensor(a_b, device=device)
                r_b = torch.as_tensor(r_b, device=device); s2_b = torch.as_tensor(s2_b, device=device); d_b = torch.as_tensor(d_b, device=device)

                q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    a2 = torch.argmax(q(s2_b), dim=1, keepdim=True)
                    y = r_b + (1.0 - d_b) * gamma * q_tgt(s2_b).gather(1, a2).squeeze(1)

                loss = loss_fn(q_sa, y)
                opt.zero_grad(); loss.backward()

                # [MOD] Gradient Norm Logging: Track the magnitude of updates
                total_norm = 0.0
                for p in q.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                temp_grads.append(total_norm)

                # [MOD] Gradient Norm Clipping: Stabilizes updates during high-frequency learning
                torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)

                opt.step()
                soft_update(q, q_tgt, target_tau)
                temp_losses.append(loss.item()) # Store step loss


        # [CRITICAL] Check Averaging Logic OUTSIDE the per-env loop
        if len(batch_ep_returns) >= num_envs:
            avg_ret = np.mean(batch_ep_returns)
            avg_td = np.mean(batch_ep_td)
            avg_mk = np.mean(batch_ep_mk)
            avg_rel = np.mean(batch_ep_rel_count)
            
            returns.append(avg_ret)
            history_idle.append(np.mean(batch_ep_idle))
            history_buffer.append(np.mean(batch_ep_buf))
            history_release.append(np.mean(batch_ep_rel))
            history_stability.append(np.mean(batch_ep_stab))
            history_flush.append(np.mean(batch_ep_flush))
            train_tardiness.append(avg_td)
            train_makespans.append(avg_mk)
            train_rel_counts.append(avg_rel)

            # Record average loss and grad norm for this episode batch
            if temp_losses:
                history_loss.append(np.mean(temp_losses))
                temp_losses = []
            else:
                history_loss.append(0.0)
                
            if temp_grads:
                history_grad_norm.append(np.mean(temp_grads))
                temp_grads = []
            else:
                history_grad_norm.append(0.0)
            
            # Clear buffers
            batch_ep_returns, batch_ep_idle, batch_ep_buf, batch_ep_rel, batch_ep_stab, batch_ep_flush = [], [], [], [], [], []
            batch_ep_td, batch_ep_mk, batch_ep_rel_count = [], [], []

            curr_idx = len(returns)
            # Validation Logic
            if curr_idx % 10 == 0:
                val_env = EventGateEnv(n_machines=configs.n_m, heuristic=configs.scheduler_type,
                                       interarrival_mean=interarrival_mean, burst_K=burst_K,
                                       event_horizon=event_horizon, init_jobs=int(configs.init_jobs))
                vs, _ = val_env.reset(seed=999)
                vd = False
                while not vd:
                    with torch.no_grad():
                        va = int(torch.argmax(q(torch.as_tensor(vs, device=device, dtype=torch.float32).unsqueeze(0)), dim=1).item())
                    vs, _, vd, _, vi = val_env.step(va)
                v_kpi = val_env.orch.get_final_kpi_stats(val_env.all_job_due_dates)
                val_episodes.append(curr_idx); val_makespans.append(v_kpi["makespan"]); val_tardiness.append(v_kpi["tardiness"]); val_rel_counts.append(vi.get("release_count", 0))
                tqdm.write(f">>> [VAL {curr_idx:04d}] MK={v_kpi['makespan']:7.1f} | TD={v_kpi['tardiness']:8.1f} | Rel={vi.get('release_count', 0)}")

            # Log EP
            avg_R_10 = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
            curr_L = history_loss[-1] if history_loss else 0.0
            tqdm.write(f"[EP {curr_idx:04d}] R={avg_ret:8.2f} (avg10={avg_R_10:8.2f}) | L={curr_L:6.4f} | MK={avg_mk:7.1f} | TD={avg_td:8.1f} | Rel={avg_rel:.0f}")
            pbar.update(1)

        states = next_states

    pbar.close()
    
    # Save & Plot
    plot_dir = os.path.join("plots", "train_ddqn"); os.makedirs(plot_dir, exist_ok=True)
    plot_name = getattr(configs, "ddqn_name", "ddqn_gate_latest")
    
    # [LOGGING SETUP]
    import csv
    log_path = os.path.join(plot_dir, f"train_log_{plot_name}.csv")
    val_log_path = os.path.join(plot_dir, f"val_log_{plot_name}.csv")
    
    train_headers = ["Episode", "Return", "Loss", "Makespan", "Tardiness", "Release_Count", "Idle_Reward", "Buffer_Reward", "Release_Reward", "Stability_Reward", "Flush_Reward"]
    val_headers = ["Episode", "Makespan", "Tardiness", "Release_Count"]
    
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(train_headers)
        for i in range(len(returns)):
            writer.writerow([
                i+1, returns[i], history_loss[i] if i < len(history_loss) else 0.0,
                train_makespans[i], train_tardiness[i], train_rel_counts[i],
                history_idle[i], history_buffer[i], history_release[i], history_stability[i], history_flush[i]
            ])
            
    with open(val_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(val_headers)
        for i in range(len(val_episodes)):
            writer.writerow([val_episodes[i], val_makespans[i], val_tardiness[i], val_rel_counts[i]])

    def get_ma(data, window=10): return [np.mean(data[max(0, i-window):i]) for i in range(1, len(data) + 1)]
    x = np.arange(1, len(returns) + 1)

    fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
    axs[0].plot(x, returns, alpha=0.3, color='gray'); axs[0].plot(x, get_ma(returns), color='red'); axs[0].set_title("Training Return")
    axs[1].plot(np.arange(1, len(history_loss)+1), history_loss, alpha=0.3, color='gray'); axs[1].plot(np.arange(1, len(history_loss)+1), get_ma(history_loss), color='orange'); axs[1].set_title("Training Loss")
    axs[2].plot(val_episodes, val_makespans, marker='o', color='blue'); axs[2].set_title("Validation Makespan")
    axs[3].plot(val_episodes, val_tardiness, marker='o', color='green'); axs[3].set_title("Validation Tardiness")
    axs[4].plot(val_episodes, val_rel_counts, marker='o', color='purple'); axs[4].set_title("Validation Release Count")
    for ax in axs: ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, f"curve_{plot_name}.png"), dpi=150); plt.close(fig)

    fig_t, axs_t = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    t_m = [("Train Makespan", train_makespans, "blue"), ("Train Tardiness", train_tardiness, "green"), ("Train Release Count", train_rel_counts, "purple")]
    for i, (title, data, color) in enumerate(t_m):
        xx = np.arange(1, len(data)+1)
        axs_t[i].plot(xx, data, alpha=0.3, color=color); axs_t[i].plot(xx, get_ma(data), color='red'); axs_t[i].set_title(title); axs_t[i].grid(True, alpha=0.2)
    fig_t.tight_layout(); fig_t.savefig(os.path.join(plot_dir, f"train_metrics_{plot_name}.png"), dpi=150); plt.close(fig_t)

    fig_c, ax_c = plt.subplots(figsize=(12, 7))
    comps = [("Idle", history_idle, "blue"), ("Buffer", history_buffer, "green"), ("Release", history_release, "red"), ("Flush", history_flush, "orange"), ("Stability", history_stability, "purple")]
    for label, h_data, color in comps: 
        if len(h_data) > 0: ax_c.plot(np.arange(1, len(h_data)+1), h_data, label=label, color=color, alpha=0.8)
    ax_c.legend(); ax_c.set_title("Reward Components"); ax_c.grid(True, alpha=0.2); fig_c.tight_layout(); fig_c.savefig(os.path.join(plot_dir, f"components_{plot_name}.png"), dpi=150); plt.close(fig_c)
    torch.save(q.state_dict(), os.path.join("ddqn_ckpt", f"{plot_name}.pth"))

if __name__ == "__main__":
    train_ddqn(
        episodes=configs.ddqn_episodes, 
        event_horizon=configs.event_horizon, 
        interarrival_mean=configs.interarrival_mean, 
        burst_K=configs.burst_size, 
        lr=configs.ddqn_lr, 
        gamma=configs.ddqn_gamma, 
        eps_start=configs.ddqn_eps_start, 
        eps_end=configs.ddqn_eps_end, 
        eps_decay_episodes=configs.ddqn_eps_decay_episodes, 
        batch_size=configs.ddqn_batch_size, 
        buffer_capacity=configs.ddqn_buffer_capacity, 
        target_tau=configs.ddqn_target_tau, 
        seed=int(getattr(configs, "event_seed", 42)),
        num_envs=configs.ddqn_num_envs
    )
