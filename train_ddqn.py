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
from model.ddqn_model import QNet

# [ADDED] for plotting
import matplotlib.pyplot as plt


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
    tardiness_list = []        # [ADDED]
    weighted_obj_list = []     # [ADDED]
    steps_list = []            # 每集步數（=到達事件數）

# === Episodes 進度條 ===
    for ep in tqdm(range(1, episodes + 1), desc="Episodes", dynamic_ncols=True):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_steps = 0
        debug = []
        info = {} # Init info
        # === 每集步數（到達事件）進度條 ===
        with tqdm(total=event_horizon, desc=f"Ep{ep} steps", leave=False, dynamic_ncols=True) as pbar:
            while True:
                a = select_action(state, epsilon(ep))
                ns, r, done, trunc, info = env.step(a)

                # [ADDED] 準確印出每個 step 的資訊
                state_str = "[" + ", ".join(f"{x:6.3f}" for x in state) + "]"
                # tqdm.write(f"[Ep {ep:03d} Step {ep_steps:03d}] Obs: {state_str} | Act: {a} | Rew: {r:8.4f}")

                buf.push(state, a, r, ns, float(done))
                ep_return += float(r)
                # [CHANGED] 記錄完整 state (4維) + action + reward
                debug.append((
                    float(state[0]), float(state[1]), float(state[2]), float(state[3]),
                    int(a), float(r)
                ))

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
                    break


        # for i, (o0, o1, o2, o3, aa, rr) in enumerate(debug, 1):
        #     tqdm.write(f"{i:4d} | Buf:{o0:5.2f} AvgL:{o1:5.2f} MinL:{o2:5.2f} WIdle:{o3:5.2f} | Act:{aa} Rew:{rr:8.4f}")

        # 紀錄本集統計
        log.append((ep, ep_return, ep_steps))
        try:
            ep_mk = float(np.max(getattr(env.orch, "machine_free_time", np.array([]))))
        except Exception:
            ep_mk = float("nan")
        makespans.append(ep_mk)
        returns.append(ep_return)
        steps_list.append(ep_steps)
        
        # [ADDED] Record Tardiness & Weighted Obj
        ep_td = info.get("episode_tardiness", 0.0)
        tardiness_list.append(ep_td)
        mk_val = ep_mk if not math.isnan(ep_mk) else 0.0
        weighted_obj_list.append(0.5 * mk_val + 0.5 * ep_td)

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
    mv_ret = np.array([np.mean(returns[max(0, i-10):i]) for i in range(1, len(returns) + 1)])  
    mv_mk  = np.array([np.mean(makespans[max(0, i-10):i]) for i in range(1, len(makespans) + 1)]) 
    mv_td  = np.array([np.mean(tardiness_list[max(0, i-10):i]) for i in range(1, len(tardiness_list) + 1)]) 
    mv_w   = np.array([np.mean(weighted_obj_list[max(0, i-10):i]) for i in range(1, len(weighted_obj_list) + 1)])

    # [CHANGED] 4個子圖
    fig, axs = plt.subplots(4, 1, figsize=(9, 12), sharex=True) 

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
    axs[1].set_ylabel("Makespan")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.3)
    
    # --- 子圖3：Tardiness ---
    axs[2].plot(x, tardiness_list, label="Total Tardiness")
    axs[2].plot(x, mv_td, linestyle="--", label="Moving Avg (10)")
    axs[2].set_ylabel("Tardiness")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.3)
    
    # --- 子圖4：Weighted Obj ---
    axs[3].plot(x, weighted_obj_list, label="0.5*MK + 0.5*TD")
    axs[3].plot(x, mv_w, linestyle="--", label="Moving Avg (10)")
    axs[3].set_ylabel("Weighted Obj")
    axs[3].set_xlabel("Episode")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.3)

    fig.tight_layout()

    # [CHANGED] 使用 eval_model_name 命名圖檔
    model_name = str(getattr(configs, "eval_model_name", "default"))
    png_name = f"train_curve_{model_name}_{reward_mode}_{episodes}ep.png"
    png_path = os.path.join(plot_dir, png_name)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] saved plots to {png_path}")
    

    # ====================== [CHANGED] 使用 eval_model_name 作為 ckpt 檔名 ======================
    out_dir = getattr(configs, "ddqn_out_dir", "ddqn_ckpt")
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(configs, "eval_model_name") and configs.eval_model_name:
        ckpt_name = f"{configs.eval_model_name}.pth"
    else:
        ckpt_name = f"ddqn_gate_{reward_mode}_{episodes}ep{int(event_horizon)}.pth"
    
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

