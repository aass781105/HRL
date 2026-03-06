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
    episodes: int = 200,
    event_horizon: int = 200,
    interarrival_mean: float = 0.1,
    burst_K: int = 10,
    lr: float = 1e-3,
    gamma: float = 0.99,
    eps_start: float = 0.2,
    eps_end: float = 0.01,
    eps_decay_episodes: int = 150,
    batch_size: int = 128,
    buffer_capacity: int = 100_000,
    target_tau: float = 0.005,
    seed: int = 42
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    event_horizon = int(event_horizon)
    env = EventGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=interarrival_mean,
        burst_K=burst_K,
        event_horizon=event_horizon,
        init_jobs=int(getattr(configs, "init_jobs", 0))
    )

    obs, _ = env.reset(seed=seed)
    obs_dim = int(obs.shape[0]) # Should be 17
    n_actions = 2

    device = torch.device(getattr(configs, "device", "cpu"))
    q = QNet(obs_dim, hidden=128, n_actions=n_actions).to(device)
    q_tgt = QNet(obs_dim, hidden=128, n_actions=n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    buf = ReplayBuffer(capacity=buffer_capacity)

    def epsilon(ep_idx: int) -> float:
        if eps_decay_episodes <= 0: return eps_end
        frac = min(1.0, ep_idx / float(eps_decay_episodes))
        return float(eps_start + (eps_end - eps_start) * frac)

    def select_action(state: np.ndarray, eps: float) -> int:
        if random.random() < eps: return random.randint(0, n_actions - 1)
        with torch.no_grad():
            s = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            qv = q(s)
            return int(torch.argmax(qv, dim=1).item())

    def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    returns, history_loss = [], []
    train_makespans, train_tardiness, train_rel_counts = [], [], []
    val_episodes, val_makespans, val_tardiness, val_rel_counts = [], [], [], [] 
    history_idle, history_buffer, history_release, history_stability, history_flush = [], [], [], [], []

    # [FIXED] Define training base seed to avoid overlap with main seed 42
    training_base_seed = 1000 if seed == 42 else seed

    for ep in tqdm(range(1, episodes + 1), desc="Episodes"):
        # [NEW] Change problem instance every 10 episodes
        instance_seed = training_base_seed + ((ep - 1) // 10)
        state, _ = env.reset(seed=instance_seed)
        ep_return, ep_idle, ep_buf, ep_rel, ep_stab, ep_flush = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ep_losses = [] 
        
        while True:
            a = select_action(state, epsilon(ep))
            ns, r, done, trunc, info = env.step(a)

            ep_idle += float(info.get("reward_idle_cost", 0.0))
            ep_buf += float(info.get("reward_buffer_penalty", 0.0))
            ep_rel += float(info.get("reward_release_penalty", 0.0))
            ep_stab += float(info.get("reward_stability_penalty", 0.0))
            if done: ep_flush += float(info.get("reward_final_flush_penalty", 0.0))

            buf.push(state, a, r, ns, float(done))
            ep_return += float(r); state = ns

            if len(buf) >= batch_size:
                s_b, a_b, r_b, s2_b, d_b = buf.sample(batch_size)
                s_b = torch.as_tensor(s_b, device=device); a_b = torch.as_tensor(a_b, device=device)
                r_b = torch.as_tensor(r_b, device=device); s2_b = torch.as_tensor(s2_b, device=device)
                d_b = torch.as_tensor(d_b, device=device)

                q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    a2 = torch.argmax(q(s2_b), dim=1, keepdim=True)
                    q_next = q_tgt(s2_b).gather(1, a2).squeeze(1)
                    y = r_b + (1.0 - d_b) * gamma * q_next

                loss = loss_fn(q_sa, y)
                opt.zero_grad(); loss.backward(); opt.step()
                soft_update(q, q_tgt, target_tau)
                ep_losses.append(loss.item())

            if done or trunc: break

        returns.append(ep_return)
        history_loss.append(np.mean(ep_losses) if ep_losses else 0.0)
        history_idle.append(ep_idle); history_buffer.append(ep_buf)
        history_release.append(ep_rel); history_stability.append(ep_stab); history_flush.append(ep_flush)
        
        ep_mk = float(np.max(env.orch.machine_free_time))
        ep_td = info.get("episode_tardiness", 0.0)
        ep_rel_count = info.get("release_count", 0)
        train_makespans.append(ep_mk); train_tardiness.append(ep_td); train_rel_counts.append(ep_rel_count)

        if ep % 10 == 0:
            # [FIXED] Single Greedy Validation on a fixed "Hold-out" instance (Seed 999)
            val_fixed_seed = 999
            v_mks, v_tds, v_rels = [], [], []
            
            vs, _ = env.reset(seed=val_fixed_seed)
            vd = False
            while not vd:
                va = select_action(vs, 0.0) # Greedy
                vs, _, vd, _, vi = env.step(va)
            vst = env.orch.get_final_kpi_stats(env.all_job_due_dates)
            
            amk, atd, arel = vst["makespan"], vst["tardiness"], vi.get("release_count", 0)
            val_episodes.append(ep); val_makespans.append(amk); val_tardiness.append(atd); val_rel_counts.append(arel)
            tqdm.write(f">>> [VAL {ep:04d}] MK={amk:7.1f} | TD={atd:8.1f} | Rel={arel:4.1f}")

        avg_R_10 = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
        avg_L_10 = np.mean(history_loss[-10:]) if len(history_loss) >= 10 else np.mean(history_loss)
        tqdm.write(f"[EP {ep:04d}] R={ep_return:8.2f} (avg10={avg_R_10:8.2f}) | L={avg_L_10:6.4f} | MK={ep_mk:7.1f} | TD={ep_td:8.1f} | Rel={ep_rel_count}")

    plot_dir = os.path.join("plots", "train_ddqn"); os.makedirs(plot_dir, exist_ok=True)
    plot_name = getattr(configs, "ddqn_name", "ddqn_gate_latest")
    def get_ma(data, window=10): return [np.mean(data[max(0, i-window):i]) for i in range(1, len(data) + 1)]
    x_train = np.arange(1, len(returns) + 1)

    fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
    axs[0].plot(x_train, returns, alpha=0.3, color='gray'); axs[0].plot(x_train, get_ma(returns), color='red'); axs[0].set_title("Training Return")
    axs[1].plot(x_train, history_loss, alpha=0.3, color='gray'); axs[1].plot(x_train, get_ma(history_loss), color='orange'); axs[1].set_title("Training Loss")
    axs[2].plot(val_episodes, val_makespans, marker='o', color='blue'); axs[2].set_title("Validation Makespan")
    axs[3].plot(val_episodes, val_tardiness, marker='o', color='green'); axs[3].set_title("Validation Tardiness")
    axs[4].plot(val_episodes, val_rel_counts, marker='o', color='purple'); axs[4].set_title("Validation Release Count")
    for ax in axs: ax.legend(); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, f"curve_{plot_name}.png"), dpi=150); plt.close(fig)

    fig_t, axs_t = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    t_metrics = [("Train Makespan", train_makespans, "blue"), ("Train Tardiness", train_tardiness, "green"), ("Train Release Count", train_rel_counts, "purple")]
    for i, (title, data, color) in enumerate(t_metrics):
        axs_t[i].plot(x_train, data, alpha=0.3, color=color); axs_t[i].plot(x_train, get_ma(data), color='red'); axs_t[i].set_title(title); axs_t[i].grid(True, alpha=0.2)
    fig_t.tight_layout(); fig_t.savefig(os.path.join(plot_dir, f"train_metrics_{plot_name}.png"), dpi=150); plt.close(fig_t)

    fig_c, ax_c = plt.subplots(figsize=(12, 7))
    comps = [("Idle", history_idle, "blue"), ("Buffer", history_buffer, "green"), ("Release", history_release, "red"), ("Flush", history_flush, "orange"), ("Stability", history_stability, "purple")]
    for label, h_data, color in comps: ax_c.plot(x_train, h_data, label=label, color=color, alpha=0.8)
    ax_c.legend(); ax_c.set_title("Reward Components"); ax_c.grid(True, alpha=0.2); fig_c.tight_layout(); fig_c.savefig(os.path.join(plot_dir, f"components_{plot_name}.png"), dpi=150); plt.close(fig_c)
    
    torch.save(q.state_dict(), os.path.join("ddqn_ckpt", f"{plot_name}.pth"))

if __name__ == "__main__":
    train_ddqn(episodes=configs.ddqn_episodes, event_horizon=configs.event_horizon, interarrival_mean=configs.interarrival_mean, burst_K=configs.burst_size, lr=configs.ddqn_lr, gamma=configs.ddqn_gamma, eps_start=configs.ddqn_eps_start, eps_end=configs.ddqn_eps_end, eps_decay_episodes=configs.ddqn_eps_decay_episodes, batch_size=configs.ddqn_batch_size, buffer_capacity=configs.ddqn_buffer_capacity, target_tau=configs.ddqn_target_tau, seed=int(getattr(configs, "event_seed", 42)))

