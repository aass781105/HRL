# train_ddqn.py
# =============================== [ADDED/CHANGED] ===============================
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

# [ADDED] for plotting
import matplotlib.pyplot as plt
import csv


# --------- Q-Network ---------
class QNet(nn.Module):  # [KEPT]
    def __init__(self, obs_dim: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# --------- Replay Buffer ---------
class ReplayBuffer:  # [KEPT]
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


# --------- 訓練主程式 ---------
def train_ddqn(
    episodes: int = 200,
    event_horizon: int = 200,
    interarrival_mean: float = 0.1,
    burst_K: int = 10,
    reward_mode: str = "pure",  # "pure" or "augmented"
    lr: float = 1e-3,
    gamma: float = 0.99,
    eps_start: float = 0.2,
    eps_end: float = 0.01,
    eps_decay_episodes: int = 150,
    batch_size: int = 128,
    buffer_capacity: int = 100_000,
    target_tau: float = 0.005,  # soft update
    seed: int = 42
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # [CHANGED] 統一用唯一終止條件：到達事件上限（確保是 int）
    event_horizon = int(event_horizon)  # ← 你的 parser 是 float，這裡統一轉 int


    # 建立環境
    env = EventGateEnv(
        n_machines=configs.n_m,
        heuristic=(configs.test_method[0] if hasattr(configs, "test_method") and configs.test_method else "SPT"),
        interarrival_mean=interarrival_mean,
        burst_K=burst_K,
        event_horizon=event_horizon,
        init_jobs=int(getattr(configs, "init_jobs", getattr(configs, "initial_jobs", 0))),
        reward_mode=reward_mode
    )

    obs, _ = env.reset(seed=seed)
    obs_dim = int(obs.shape[0])
    n_actions = 2

    device = torch.device(getattr(configs, "device", "cpu"))
    q = QNet(obs_dim, hidden=128, n_actions=n_actions).to(device)
    q_tgt = QNet(obs_dim, hidden=128, n_actions=n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    buf = ReplayBuffer(capacity=buffer_capacity)

    def epsilon(ep_idx: int) -> float:
        if eps_decay_episodes <= 0:
            return eps_end
        frac = min(1.0, ep_idx / float(eps_decay_episodes))
        return float(eps_start + (eps_end - eps_start) * frac)

    def select_action(state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randint(0, n_actions - 1)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(device)
            qv = q(s)
            return int(torch.argmax(qv, dim=1).item())

    def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    # [ADDED] 記錄曲線
    log = []                   # (ep, return, steps)
    returns = []               # 純回報列表
    makespans = []
    steps_list = []            # 每集步數（=到達事件數）
    sparse_mode = (reward_mode == "sparse_makespan")

# === Episodes 進度條 ===
    for ep in tqdm(range(1, episodes + 1), desc="Episodes", dynamic_ncols=True):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_steps = 0
        ep_traj = []  # 稀疏回饋時暫存 transition（s, a, ns, done）
        debug = []
        # === 每集步數（到達事件）進度條 ===
        with tqdm(total=event_horizon, desc=f"Ep{ep} steps", leave=False, dynamic_ncols=True) as pbar:
            while True:
                a = select_action(state, epsilon(ep))
                ns, r, done, trunc, info = env.step(a)

                if sparse_mode:
                    # 稀疏：先不把 transition 丟 replay，reward 暫設 0
                    ep_traj.append((state.copy(), int(a), 0.0, ns.copy(), float(done)))
                else:
                    buf.push(state, a, r, ns, float(done))
                    ep_return += float(r)
                debug.append((float(state[0]), float(state[1]), int(a), float(r)))

                state = ns
                ep_steps += 1

                # 更新步數進度條
                pbar.update(1)
                if (ep_steps % 10) == 0:
                    pbar.set_postfix(R=f"{ep_return:.2f}")

                # 參數更新
                if len(buf) >= batch_size:
                    s, a_, r_, s2, d_ = buf.sample(batch_size)
                    s  = torch.from_numpy(s).to(device)
                    a_ = torch.from_numpy(a_).to(device)
                    r_ = torch.from_numpy(r_).to(device)
                    s2 = torch.from_numpy(s2).to(device)
                    d_ = torch.from_numpy(d_).to(device)

                    q_sa = q(s).gather(1, a_.view(-1, 1)).squeeze(1)
                    with torch.no_grad():
                        a2 = torch.argmax(q(s2), dim=1, keepdim=True)   # Double DQN：online 選 a'
                        q_s2a2 = q_tgt(s2).gather(1, a2).squeeze(1)     # target 評估 Q(s', a')
                        y = r_ + (1.0 - d_) * gamma * q_s2a2

                    loss = loss_fn(q_sa, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), max_norm=5.0)
                    opt.step()
                    soft_update(q, q_tgt, target_tau)

                if done or trunc:
                    # last_r = float(r)
                    # # 這一步是否有 flush penalty（如果有開 enable_final_flush_penalty）
                    # final_pen = float(info.get("final_flush_penalty", 0.0))
                    # delta_mk_flush = float(info.get("delta_mk_flush", 0.0))
                    # idle_flush = float(info.get("idle_flush", 0.0))


                    # print(
                    #     f"[Ep {ep:03d} DONE] steps={ep_steps} "
                    #     f"ep_return={ep_return:.3f} last_r={last_r:.3f} "
                    #     f"final_flush_penalty={final_pen:.3f} "
                    #     f"delta_mk_flush={delta_mk_flush:.3f} idle_flush={idle_flush:.3f}"
                    # )
                    break


        # ====== 稀疏回饋：終局後計算 makespan，均分回報並入 replay ======
        if sparse_mode:
            try:
                ep_mk = float(np.max(getattr(env.orch, "machine_free_time", np.array([0.0]))))
            except Exception:
                ep_mk = 0.0

            if len(ep_traj) > 0:
                per_step_r = - ep_mk / float(len(ep_traj))
                ep_return = - ep_mk
                for (s0, a0, _r_ignored, s1, d0) in ep_traj:
                    buf.push(s0, a0, per_step_r, s1, d0)

                # 終局後做幾次額外更新，加速吸收稀疏訊號
                updates = min(len(ep_traj), 10)
                for _ in range(updates):
                    if len(buf) < batch_size:
                        break
                    s, a_, r_, s2, d_ = buf.sample(batch_size)
                    s  = torch.from_numpy(s).to(device)
                    a_ = torch.from_numpy(a_).to(device)
                    r_ = torch.from_numpy(r_).to(device)
                    s2 = torch.from_numpy(s2).to(device)
                    d_ = torch.from_numpy(d_).to(device)
                    
                    q_sa = q(s).gather(1, a_.view(-1, 1)).squeeze(1)
                    with torch.no_grad():
                        a2 = torch.argmax(q(s2), dim=1, keepdim=True)
                        q_s2a2 = q_tgt(s2).gather(1, a2).squeeze(1)
                        y = r_ + (1.0 - d_) * gamma * q_s2a2
                    
                    loss = loss_fn(q_sa, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), max_norm=5.0)
                    opt.step()
                    soft_update(q, q_tgt, target_tau)
            else:
                ep_return = 0.0
        # for i, (o0, o1, aa, rr) in enumerate(debug, 1):
        #     tqdm.write(f"{i:4d} {o0:9.3f} {o1:11.3f} {aa:3d} {rr:10.4f}")

        # 紀錄本集統計
        log.append((ep, ep_return, ep_steps))
        try:
            ep_mk = float(np.max(getattr(env.orch, "machine_free_time", np.array([]))))
        except Exception:
            ep_mk = float("nan")
        makespans.append(ep_mk)
        returns.append(ep_return)
        steps_list.append(ep_steps)

        if ep % 10 == 0:
            avg_R = np.mean(returns[-10:]) if len(returns) >= 10 else ep_return
            tqdm.write(f"[EP {ep:04d}] return={ep_return:.3f}  avgR(10)={avg_R:.3f}  steps={ep_steps}")


    # （可選）儲存權重（建議：同時存 meta 以利推論）  ----------------------- [CHANGED]
    plot_dir = os.path.join("plots", "train_ddqn")
    os.makedirs(plot_dir, exist_ok=True)

    final_penalty = bool(getattr(configs, "enable_final_flush_penalty", False))
    idle_penalty = bool(getattr(configs, "enable_full_idle_penalty", False))
    # [CHANGED] x 軸、移動平均（視窗=10）
    x = np.arange(1, len(returns) + 1)
    mv_ret = np.array([np.mean(returns[max(0, i-10):i]) for i in range(1, len(returns) + 1)])  # [CHANGED]
    mv_mk  = np.array([np.mean(makespans[max(0, i-10):i]) for i in range(1, len(makespans) + 1)])  # [ADDED]

    # [CHANGED] 兩個子圖：上=Return，下=Makespan
    fig, axs = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)  # [CHANGED]

    # --- 子圖1：Return ---
    axs[0].plot(x, returns, label="Return per Episode")
    axs[0].plot(x, mv_ret, linestyle="--", label="Moving Avg (10)")
    axs[0].set_ylabel("Return")
    axs[0].set_title(f"DDQN Training — Reward & Makespan ({reward_mode})")
    axs[0].legend(loc="best")
    axs[0].grid(True, alpha=0.3)

    # --- 子圖2：Makespan ---
    axs[1].plot(x, makespans, label="Dynamic Makespan")
    axs[1].plot(x, mv_mk, linestyle="--", label="Moving Avg (10)")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Makespan")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()

    png_path = os.path.join(plot_dir, f"train_curve_{reward_mode}_{episodes}ep_for_{event_horizon}events_with_makespan.png")  # [CHANGED]
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] saved plots to {png_path}")
    

    # ====================== [CHANGED] 只存 state_dict，檔名含 reward_mode 與 event_horizon ======================
    out_dir = getattr(configs, "ddqn_out_dir", "ddqn_ckpt")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_name = f"ddqn_gate_{reward_mode}_{episodes}ep{int(event_horizon)}.pth"   # 例如 ddqn_gate_pure_200.pth
    ckpt_path = os.path.join(out_dir, ckpt_name)

    torch.save(q.state_dict(), ckpt_path)  # 只存 online Q 的 state_dict（與 PPO 一樣的做法）
    print(f"[SAVED] DDQN weights -> {ckpt_path}")






if __name__ == "__main__":
    train_ddqn(
        episodes=int(getattr(configs, "ddqn_episodes", 100)),
        event_horizon=int(getattr(configs, "event_horizon", 200)),   # ← 唯一終止條件
        interarrival_mean=float(getattr(configs, "interarrival_mean", 0.1)),
        burst_K=int(getattr(configs, "burst_size", 10)),
        reward_mode=str(getattr(configs, "ddqn_reward_mode", "pure")),   # "pure" 或 "augmented"

        lr=float(getattr(configs, "ddqn_lr", 1e-3)),
        gamma=float(getattr(configs, "ddqn_gamma", 0.99)),
        eps_start=float(getattr(configs, "ddqn_eps_start", 0.2)),
        eps_end=float(getattr(configs, "ddqn_eps_end", 0.01)),
        eps_decay_episodes=int(getattr(configs, "ddqn_eps_decay_episodes", 80)),

        batch_size=int(getattr(configs, "ddqn_batch_size", 128)),
        buffer_capacity=int(getattr(configs, "ddqn_buffer_capacity", 100_000)),
        target_tau=float(getattr(configs, "ddqn_target_tau", 0.005)),

        seed=int(getattr(configs, "event_seed", 42)),
    )

