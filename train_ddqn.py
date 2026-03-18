# train_ddqn.py
import os
import math
import random
import time
import csv
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

# --------- Optimized Replay Buffer (Numpy Circular Buffer) ---------
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_eps: float = 1e-3,
        stratified_min_frac: float = 0.0,
    ):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_eps = float(per_eps)
        self.stratified_min_frac = float(np.clip(stratified_min_frac, 0.0, 0.5))
        self.ptr = 0
        self.size = 0
        self.s_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.s2_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.a_buf = np.zeros(self.capacity, dtype=np.int64)
        self.r_buf = np.zeros(self.capacity, dtype=np.float32)
        self.d_buf = np.zeros(self.capacity, dtype=np.float32)
        self.prio_buf = np.ones(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, s, a, r, s2, d):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.s2_buf[self.ptr] = s2
        self.d_buf[self.ptr] = d
        self.prio_buf[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, s, a, r, s2, d):
        n = s.shape[0]
        if self.ptr + n <= self.capacity:
            self.s_buf[self.ptr:self.ptr+n] = s
            self.a_buf[self.ptr:self.ptr+n] = a
            self.r_buf[self.ptr:self.ptr+n] = r
            self.s2_buf[self.ptr:self.ptr+n] = s2
            self.d_buf[self.ptr:self.ptr+n] = d
            self.prio_buf[self.ptr:self.ptr+n] = self.max_priority
        else:
            # Wrap around
            first = self.capacity - self.ptr
            self.s_buf[self.ptr:] = s[:first]
            self.a_buf[self.ptr:] = a[:first]
            self.r_buf[self.ptr:] = r[:first]
            self.s2_buf[self.ptr:] = s2[:first]
            self.d_buf[self.ptr:] = d[:first]
            self.prio_buf[self.ptr:] = self.max_priority
            second = n - first
            self.s_buf[:second] = s[first:]
            self.a_buf[:second] = a[first:]
            self.r_buf[:second] = r[first:]
            self.s2_buf[:second] = s2[first:]
            self.d_buf[:second] = d[first:]
            self.prio_buf[:second] = self.max_priority
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def _draw_from_candidates(self, candidates: np.ndarray, n_draw: int):
        if n_draw <= 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        if candidates.size <= 0:
            candidates = np.arange(self.size, dtype=np.int64)
        replace = candidates.size < n_draw
        if self.use_per:
            pr = np.power(self.prio_buf[candidates] + self.per_eps, self.per_alpha).astype(np.float32)
            total = float(np.sum(pr))
            if total <= 0.0:
                p = np.full(candidates.size, 1.0 / float(candidates.size), dtype=np.float32)
            else:
                p = pr / total
            pick = np.random.choice(candidates.size, size=n_draw, replace=replace, p=p)
            idx = candidates[pick]
            cond_prob = p[pick]
            return idx, cond_prob.astype(np.float32)
        pick = np.random.choice(candidates.size, size=n_draw, replace=replace)
        idx = candidates[pick]
        cond_prob = np.full(n_draw, 1.0 / float(candidates.size), dtype=np.float32)
        return idx, cond_prob

    def _sample_indices(self, batch_size: int):
        valid = np.arange(self.size, dtype=np.int64)
        if self.stratified_min_frac <= 0.0:
            idx, cond_prob = self._draw_from_candidates(valid, batch_size)
            samp_prob = cond_prob
            return idx, samp_prob

        min_n = int(batch_size * self.stratified_min_frac)
        n0 = min_n
        n1 = min_n
        n_rem = batch_size - n0 - n1
        p1_all = float(np.mean(self.a_buf[:self.size] == 1)) if self.size > 0 else 0.5
        n1 += int(round(n_rem * p1_all))
        n0 = batch_size - n1

        cand0 = valid[self.a_buf[:self.size] == 0]
        cand1 = valid[self.a_buf[:self.size] == 1]

        idx0, p0 = self._draw_from_candidates(cand0, n0)
        idx1, p1 = self._draw_from_candidates(cand1, n1)

        samp_prob0 = (float(n0) / float(batch_size)) * p0 if n0 > 0 else np.empty((0,), dtype=np.float32)
        samp_prob1 = (float(n1) / float(batch_size)) * p1 if n1 > 0 else np.empty((0,), dtype=np.float32)

        idx = np.concatenate([idx0, idx1], axis=0)
        samp_prob = np.concatenate([samp_prob0, samp_prob1], axis=0)
        if idx.size > 1:
            perm = np.random.permutation(idx.size)
            idx = idx[perm]
            samp_prob = samp_prob[perm]
        return idx, samp_prob

    def sample(self, batch_size: int):
        s, a, r, s2, d, _, _ = self.sample_with_idx(batch_size, beta=1.0)
        return s, a, r, s2, d

    def sample_with_idx(self, batch_size: int, beta: float = 0.4):
        idx, samp_prob = self._sample_indices(batch_size)
        if self.use_per:
            n = float(max(1, self.size))
            p = np.clip(samp_prob, 1e-12, 1.0)
            is_w = np.power(n * p, -float(beta)).astype(np.float32)
            is_w = is_w / max(1e-12, float(np.max(is_w)))
        else:
            is_w = np.ones(idx.shape[0], dtype=np.float32)
        return (
            self.s_buf[idx],
            self.a_buf[idx],
            self.r_buf[idx],
            self.s2_buf[idx],
            self.d_buf[idx],
            idx,
            is_w,
        )

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray):
        if not self.use_per or idx.size == 0:
            return
        pr = np.asarray(td_abs, dtype=np.float32).reshape(-1)
        if pr.size != idx.size:
            return
        pr = np.clip(pr + self.per_eps, 1e-6, 1e6)
        self.prio_buf[idx] = pr
        self.max_priority = max(self.max_priority, float(np.max(pr)))

    def action_ratio(self) -> float:
        if self.size <= 0:
            return 0.0
        return float(np.mean(self.a_buf[:self.size] == 1))

    def recent_action_ratio(self, frac: float = 0.2) -> float:
        if self.size <= 0:
            return 0.0
        recent_count = max(1, int(self.size * frac))
        idx = (self.ptr - np.arange(1, recent_count + 1)) % self.capacity
        return float(np.mean(self.a_buf[idx] == 1))

    def recent_sample_ratio(self, idx: np.ndarray, frac: float = 0.2) -> float:
        if self.size <= 0 or idx.size == 0:
            return 0.0
        recent_count = max(1, int(self.size * frac))
        age = (self.ptr - idx - 1) % self.capacity
        return float(np.mean(age < recent_count))

    def __len__(self):
        return self.size

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
    train_start_ts = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    device = torch.device(getattr(configs, "device", "cpu"))

    def build_env():
        return EventGateEnv(
            n_machines=configs.n_m,
            heuristic=configs.scheduler_type,
            interarrival_mean=interarrival_mean,
            burst_K=burst_K,
            event_horizon=event_horizon,
            init_jobs=int(getattr(configs, "init_jobs", 0))
        )

    def make_env(rank):
        def _thunk():
            return build_env()
        return _thunk

    env_fns = [make_env(i) for i in range(num_envs)]

    # AsyncVectorEnv can hide worker exceptions on Windows; fall back to SyncVectorEnv if reset fails.
    seeds = [seed + i * 100 for i in range(num_envs)]
    try:
        envs = gym.vector.AsyncVectorEnv(env_fns)
        states, _ = envs.reset(seed=seeds)
    except Exception:
        try:
            envs.close()
        except Exception:
            pass
        envs = gym.vector.SyncVectorEnv(env_fns)
        states, _ = envs.reset(seed=seeds)

    episodes_per_instance = max(1, int(getattr(configs, "ddqn_instance_episodes", 200)))
    same_problem_base_seed = int(seeds[0])
    
    obs_dim = states.shape[1] 
    n_actions = 2

    q = QNet(
        obs_dim,
        n_actions=n_actions,
        hidden=configs.ddqn_hidden_dim,
        num_layers=configs.ddqn_num_layers,
        dueling=bool(getattr(configs, "ddqn_dueling", True)),
    ).to(device)
    q_tgt = QNet(
        obs_dim,
        n_actions=n_actions,
        hidden=configs.ddqn_hidden_dim,
        num_layers=configs.ddqn_num_layers,
        dueling=bool(getattr(configs, "ddqn_dueling", True)),
    ).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    use_per = bool(getattr(configs, "ddqn_use_per", False))
    use_lr_decay = bool(getattr(configs, "ddqn_lr_decay", True))
    lr_end = float(getattr(configs, "ddqn_lr_end", 1e-5))
    per_alpha = float(getattr(configs, "ddqn_per_alpha", 0.6))
    per_beta_start = float(getattr(configs, "ddqn_per_beta_start", 0.4))
    per_beta_end = float(getattr(configs, "ddqn_per_beta_end", 1.0))
    per_eps = float(getattr(configs, "ddqn_per_eps", 1e-3))
    stratified_min_frac = float(getattr(configs, "ddqn_stratified_min_frac", 0.0))
    same_problem_eval_every = max(1, int(getattr(configs, "ddqn_same_problem_eval_every", 1)))
    buf = ReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim,
        use_per=use_per,
        per_alpha=per_alpha,
        per_eps=per_eps,
        stratified_min_frac=stratified_min_frac,
    )

    def epsilon(ep_idx: int) -> float:
        if eps_decay_episodes <= 0: return eps_end
        frac = min(1.0, ep_idx / float(eps_decay_episodes))
        return float(eps_start + (eps_end - eps_start) * frac)

    def current_lr(ep_idx: int) -> float:
        if not use_lr_decay:
            return float(lr)
        if episodes <= 1:
            return float(lr_end)
        frac = min(1.0, max(0.0, (ep_idx - 1) / float(episodes - 1)))
        return float(lr + (lr_end - lr) * frac)

    def per_beta(ep_idx: int) -> float:
        if not use_per:
            return 1.0
        if episodes <= 1:
            return per_beta_end
        frac = min(1.0, max(0.0, (ep_idx - 1) / float(episodes - 1)))
        return float(per_beta_start + (per_beta_end - per_beta_start) * frac)

    def select_actions(curr_states: np.ndarray, eps: float) -> np.ndarray:
        explore_mask = np.random.random(num_envs) < eps
        actions = np.empty(num_envs, dtype=np.int64)
        actions[explore_mask] = np.random.randint(0, n_actions, size=int(explore_mask.sum()))
        mask = ~explore_mask
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

    def build_fixed_probe_states(probe_seed: int, max_states: int = 64) -> np.ndarray:
        probe_env = build_env()
        obs, _ = probe_env.reset(seed=probe_seed)
        collected = [np.asarray(obs, dtype=np.float32).copy()]
        done = False
        step_idx = 0
        while (not done) and (len(collected) < max_states):
            scripted_action = 1 if (step_idx % 2 == 0) else 0
            obs, _, done, _, _ = probe_env.step(scripted_action)
            collected.append(np.asarray(obs, dtype=np.float32).copy())
            step_idx += 1
        return np.asarray(collected, dtype=np.float32)

    def eval_same_problem_greedy(eval_seed: int):
        eval_env = build_env()
        obs, _ = eval_env.reset(seed=eval_seed)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                qv = q(torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0))
                act = int(torch.argmax(qv, dim=1).item())
            obs, _, done, _, info = eval_env.step(act)
        kpi = eval_env.orch.get_final_kpi_stats(eval_env.all_job_due_dates)
        return float(kpi["makespan"]), float(kpi["tardiness"]), int(info.get("release_count", 0))

    probe_states = build_fixed_probe_states(same_problem_base_seed)


    # Metrics (Final Averaged Points)
    returns, history_loss, history_grad_norm = [], [], []
    history_qgap = []
    history_lr = []
    history_rb_a1 = []
    history_rb_recent_a1 = []
    history_samp_a1 = []
    history_samp_recent = []
    history_tdabs_a0 = []
    history_tdabs_a1 = []
    history_same_greedy_mk, history_same_greedy_td, history_same_greedy_rel = [], [], []
    train_makespans, train_tardiness, train_rel_counts = [], [], []
    val_episodes, val_makespans, val_tardiness, val_rel_counts = [], [], [], [] 
    history_buffer, history_shaping, history_gate_total, history_terminal, history_stability, history_flush = [], [], [], [], [], []

    # Track data for ongoing episodes in each env
    running_ep_returns = np.zeros(num_envs)
    running_ep_buf = np.zeros(num_envs)
    running_ep_shape = np.zeros(num_envs)
    running_ep_gate_total = np.zeros(num_envs)
    running_ep_terminal = np.zeros(num_envs)
    running_ep_stab = np.zeros(num_envs)
    env_episode_counts = np.zeros(num_envs, dtype=np.int64)
    # Buffers to collect data from all envs before averaging
    batch_ep_returns, batch_ep_buf, batch_ep_shape, batch_ep_gate_total, batch_ep_terminal, batch_ep_stab, batch_ep_flush = [], [], [], [], [], [], []
    batch_ep_td, batch_ep_mk, batch_ep_rel_count = [], [], []
    temp_losses = [] 
    temp_grads = [] # [FIX] Initialize for gradient norm tracking
    temp_sample_a1 = []
    temp_sample_recent = []
    temp_tdabs_a0 = []
    temp_tdabs_a1 = []

    pbar = tqdm(total=episodes, desc="Averaged Batches")

    while len(returns) < episodes:
        ep_idx = len(returns) + 1
        lr_now = current_lr(ep_idx)
        for g in opt.param_groups:
            g["lr"] = lr_now
        curr_eps = epsilon(ep_idx)
        actions = select_actions(states, curr_eps)
        next_states, rewards, dones, truncs, infos = envs.step(actions)

        # 1. Vectorized Push to Replay Buffer
        buf.push_batch(states, actions, rewards, next_states, (dones | truncs).astype(float))

        # 2. Vectorized Metric Accumulation
        running_ep_returns += rewards.astype(float)
        
        # Helper to get info arrays or zeros
        def get_info_arr(key):
            if key in infos: return np.asarray(infos[key], dtype=float)
            return np.zeros(num_envs)

        running_ep_buf += get_info_arr("reward_buffer_penalty")
        running_ep_shape += get_info_arr("reward_shaping_penalty") if "reward_shaping_penalty" in infos else get_info_arr("reward_release_raw_penalty")
        running_ep_gate_total += get_info_arr("reward_gate_penalty") if "reward_gate_penalty" in infos else get_info_arr("reward_release_penalty")
        running_ep_terminal += get_info_arr("reward_terminal_penalty") if "reward_terminal_penalty" in infos else get_info_arr("reward_td_terminal_penalty")
        running_ep_stab += get_info_arr("reward_stability_penalty")

        # 3. Handle Episode Termination (Vectorized Check)
        term_mask = (dones | truncs)
        if term_mask.any():
            idx_finished = np.where(term_mask)[0]
            for i in idx_finished:
                # Extract terminal metrics
                ep_td = float(infos["episode_tardiness"][i]) if "episode_tardiness" in infos else 0.0
                ep_mk = float(infos["episode_makespan"][i]) if "episode_makespan" in infos else 0.0
                ep_rel_count = int(infos["release_count"][i]) if "release_count" in infos else 0
                ep_flush = float(infos["reward_final_flush_penalty"][i]) if "reward_final_flush_penalty" in infos else 0.0
                
                batch_ep_returns.append(running_ep_returns[i])
                batch_ep_buf.append(running_ep_buf[i])
                batch_ep_shape.append(running_ep_shape[i])
                batch_ep_gate_total.append(running_ep_gate_total[i])
                batch_ep_terminal.append(running_ep_terminal[i])
                batch_ep_stab.append(running_ep_stab[i])
                batch_ep_flush.append(ep_flush)
                batch_ep_td.append(ep_td)
                batch_ep_mk.append(ep_mk)
                batch_ep_rel_count.append(ep_rel_count)
                env_episode_counts[i] += 1

                # Reset running counters
                running_ep_returns[i] = 0.0
                running_ep_buf[i] = 0.0
                running_ep_shape[i] = 0.0
                running_ep_gate_total[i] = 0.0
                running_ep_terminal[i] = 0.0
                running_ep_stab[i] = 0.0

        # Training Step (Sample from Buffer)
        # [MOD] Replay Ratio: Update the network multiple times per step to accelerate convergence
        replay_ratio = int(getattr(configs, "ddqn_replay_ratio", 2))
        if len(buf) >= batch_size:
            for _ in range(replay_ratio):
                beta_now = per_beta(len(returns) + 1)
                s_b, a_b, r_b, s2_b, d_b, idx_b, is_w_b = buf.sample_with_idx(batch_size, beta=beta_now)
                s_b = torch.as_tensor(s_b, device=device); a_b = torch.as_tensor(a_b, device=device)
                r_b = torch.as_tensor(r_b, device=device); s2_b = torch.as_tensor(s2_b, device=device); d_b = torch.as_tensor(d_b, device=device)
                is_w_b = torch.as_tensor(is_w_b, device=device)

                q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    a2 = torch.argmax(q(s2_b), dim=1, keepdim=True)
                    y = r_b + (1.0 - d_b) * gamma * q_tgt(s2_b).gather(1, a2).squeeze(1)

                per_sample_loss = nn.functional.smooth_l1_loss(q_sa, y, reduction='none')
                loss = (per_sample_loss * is_w_b).mean()
                opt.zero_grad(); loss.backward()

                with torch.no_grad():
                    td_abs = torch.abs(y - q_sa).detach().cpu().numpy()
                    buf.update_priorities(idx_b, td_abs)
                    a_np = a_b.detach().cpu().numpy()
                    temp_sample_a1.append(float(np.mean(a_np == 1)))
                    temp_sample_recent.append(float(buf.recent_sample_ratio(idx_b, frac=0.2)))
                    if np.any(a_np == 0):
                        temp_tdabs_a0.append(float(np.mean(td_abs[a_np == 0])))
                    if np.any(a_np == 1):
                        temp_tdabs_a1.append(float(np.mean(td_abs[a_np == 1])))

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
            history_buffer.append(np.mean(batch_ep_buf))
            history_shaping.append(np.mean(batch_ep_shape))
            history_gate_total.append(np.mean(batch_ep_gate_total))
            history_terminal.append(np.mean(batch_ep_terminal))
            history_stability.append(np.mean(batch_ep_stab))
            history_flush.append(np.mean(batch_ep_flush))
            train_tardiness.append(avg_td)
            train_makespans.append(avg_mk)
            train_rel_counts.append(avg_rel)
            history_lr.append(float(opt.param_groups[0]["lr"]))
            history_rb_a1.append(float(buf.action_ratio()))
            history_rb_recent_a1.append(float(buf.recent_action_ratio(frac=0.2)))
            history_samp_a1.append(float(np.mean(temp_sample_a1)) if temp_sample_a1 else 0.0)
            history_samp_recent.append(float(np.mean(temp_sample_recent)) if temp_sample_recent else 0.0)
            history_tdabs_a0.append(float(np.mean(temp_tdabs_a0)) if temp_tdabs_a0 else 0.0)
            history_tdabs_a1.append(float(np.mean(temp_tdabs_a1)) if temp_tdabs_a1 else 0.0)
            with torch.no_grad():
                s_dbg = torch.as_tensor(probe_states, device=device, dtype=torch.float32)
                q_dbg = q(s_dbg)
                q_gap = torch.abs(q_dbg[:, 1] - q_dbg[:, 0]).mean().item()
            history_qgap.append(float(q_gap))
            curr_idx = len(returns)
            do_same_problem_eval = (curr_idx == 1) or (curr_idx % same_problem_eval_every == 0)
            if do_same_problem_eval:
                same_problem_seed = int(same_problem_base_seed + ((curr_idx - 1) // episodes_per_instance))
                greedy_mk, greedy_td, greedy_rel = eval_same_problem_greedy(same_problem_seed)
            elif history_same_greedy_mk:
                greedy_mk = history_same_greedy_mk[-1]
                greedy_td = history_same_greedy_td[-1]
                greedy_rel = history_same_greedy_rel[-1]
            else:
                greedy_mk, greedy_td, greedy_rel = 0.0, 0.0, 0
            history_same_greedy_mk.append(greedy_mk)
            history_same_greedy_td.append(greedy_td)
            history_same_greedy_rel.append(greedy_rel)
            temp_sample_a1 = []
            temp_sample_recent = []
            temp_tdabs_a0 = []
            temp_tdabs_a1 = []

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
            batch_ep_returns, batch_ep_buf, batch_ep_shape, batch_ep_gate_total, batch_ep_terminal, batch_ep_stab, batch_ep_flush = [], [], [], [], [], [], []
            batch_ep_td, batch_ep_mk, batch_ep_rel_count = [], [], []

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
            avg_buf_r = history_buffer[-1]
            avg_shape_r = history_shaping[-1]
            avg_gate_total_r = history_gate_total[-1]
            avg_stab_r = history_stability[-1]
            avg_flush_r = history_flush[-1]
            avg_q_gap = history_qgap[-1] if history_qgap else 0.0
            avg_lr = history_lr[-1] if history_lr else float(opt.param_groups[0]["lr"])
            avg_greedy_td = history_same_greedy_td[-1] if history_same_greedy_td else 0.0
            avg_greedy_mk = history_same_greedy_mk[-1] if history_same_greedy_mk else 0.0
            avg_greedy_rel = history_same_greedy_rel[-1] if history_same_greedy_rel else 0.0
            avg_rb_a1 = history_rb_a1[-1] if history_rb_a1 else 0.0
            avg_rb_recent_a1 = history_rb_recent_a1[-1] if history_rb_recent_a1 else 0.0
            avg_samp_a1 = history_samp_a1[-1] if history_samp_a1 else 0.0
            avg_samp_recent = history_samp_recent[-1] if history_samp_recent else 0.0
            avg_tdabs_a0 = history_tdabs_a0[-1] if history_tdabs_a0 else 0.0
            avg_tdabs_a1 = history_tdabs_a1[-1] if history_tdabs_a1 else 0.0
            tqdm.write(
                f"[EP {curr_idx:04d}] R={avg_ret:8.2f} (avg10={avg_R_10:8.2f}) | "
                f"L={curr_L:6.4f} | LR={avg_lr:.2e} | MK={avg_mk:7.1f} | TD={avg_td:8.1f} | Rel={avg_rel:.0f} | "
                f"Rbuf={avg_buf_r:7.2f} | Rshape={avg_shape_r:7.2f} | Rterm={history_terminal[-1]:7.2f} | "
                f"Rstab={avg_stab_r:7.2f} | Rflush={avg_flush_r:7.2f} | Qgap={avg_q_gap:6.3f}"
            )
            tqdm.write(
                f"           GTD={avg_greedy_td:8.1f} | GMK={avg_greedy_mk:7.1f} | GRel={avg_greedy_rel:4.0f} | Rgate={avg_gate_total_r:7.2f}"
            )
            tqdm.write(
                f"           RB(a1)={avg_rb_a1:5.1%} | RB_recent(a1)={avg_rb_recent_a1:5.1%} | "
                f"Samp(a1)={avg_samp_a1:5.1%} | SampRecent={avg_samp_recent:5.1%} | "
                f"|td|a0={avg_tdabs_a0:6.3f} | |td|a1={avg_tdabs_a1:6.3f}"
            )
            pbar.update(1)

        states = next_states

    pbar.close()
    
    # Save & Plot
    plot_dir = os.path.join("plots", "train_ddqn"); os.makedirs(plot_dir, exist_ok=True)
    plot_name = getattr(configs, "ddqn_name", "ddqn_gate_latest")
    
    # [LOGGING SETUP]
    log_path = os.path.join(plot_dir, f"train_log_{plot_name}.csv")
    val_log_path = os.path.join(plot_dir, f"val_log_{plot_name}.csv")
    
    train_headers = [
        "Episode", "Return", "Loss", "LR", "Q_Gap",
        "Same_Problem_Greedy_MK", "Same_Problem_Greedy_TD", "Same_Problem_Greedy_Rel",
        "Replay_A1_Ratio", "Replay_Recent_A1_Ratio", "Sample_A1_Ratio", "Sample_Recent_Ratio", "Sample_TDAbs_A0", "Sample_TDAbs_A1",
        "Makespan", "Tardiness", "Release_Count",
        "Buffer_Reward", "Shaping_Reward", "Gate_Total_Reward", "Terminal_Reward", "Stability_Reward", "Flush_Reward"
    ]
    val_headers = ["Episode", "Makespan", "Tardiness", "Release_Count"]
    
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(train_headers)
        for i in range(len(returns)):
            writer.writerow([
                i+1, returns[i], history_loss[i] if i < len(history_loss) else 0.0, history_lr[i] if i < len(history_lr) else float(lr),
                history_qgap[i] if i < len(history_qgap) else 0.0,
                history_same_greedy_mk[i] if i < len(history_same_greedy_mk) else 0.0,
                history_same_greedy_td[i] if i < len(history_same_greedy_td) else 0.0,
                history_same_greedy_rel[i] if i < len(history_same_greedy_rel) else 0.0,
                history_rb_a1[i] if i < len(history_rb_a1) else 0.0,
                history_rb_recent_a1[i] if i < len(history_rb_recent_a1) else 0.0,
                history_samp_a1[i] if i < len(history_samp_a1) else 0.0,
                history_samp_recent[i] if i < len(history_samp_recent) else 0.0,
                history_tdabs_a0[i] if i < len(history_tdabs_a0) else 0.0,
                history_tdabs_a1[i] if i < len(history_tdabs_a1) else 0.0,
                train_makespans[i], train_tardiness[i], train_rel_counts[i],
                history_buffer[i], history_shaping[i], history_gate_total[i], history_terminal[i], history_stability[i], history_flush[i]
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
    comps = [("Buffer", history_buffer, "green"), ("Shaping", history_shaping, "red"), ("GateTotal", history_gate_total, "pink"), ("Terminal", history_terminal, "brown"), ("Flush", history_flush, "orange"), ("Stability", history_stability, "purple")]
    for label, h_data, color in comps: 
        if len(h_data) > 0: ax_c.plot(np.arange(1, len(h_data)+1), h_data, label=label, color=color, alpha=0.8)
    ax_c.legend(); ax_c.set_title("Reward Components"); ax_c.grid(True, alpha=0.2); fig_c.tight_layout(); fig_c.savefig(os.path.join(plot_dir, f"components_{plot_name}.png"), dpi=150); plt.close(fig_c)
    torch.save(q.state_dict(), os.path.join("ddqn_ckpt", f"{plot_name}.pth"))

    train_end_ts = time.time()
    elapsed_sec = max(0.0, train_end_ts - train_start_ts)
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
    time_log_path = os.path.join(plot_dir, f"time_{plot_name}.txt")
    with open(time_log_path, "w", encoding="utf-8") as f:
        f.write(f"ddqn_name={plot_name}\n")
        f.write(f"config_path={getattr(configs, 'config', '')}\n")
        f.write(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_ts))}\n")
        f.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_ts))}\n")
        f.write(f"elapsed_seconds={elapsed_sec:.3f}\n")
        f.write(f"elapsed_hms={elapsed_hms}\n")
        f.write(f"episodes={episodes}\n")
        f.write(f"num_envs={num_envs}\n")
        f.write(f"instance_episodes={int(getattr(configs, 'ddqn_instance_episodes', 200))}\n")
        f.write(f"ddqn_dueling={bool(getattr(configs, 'ddqn_dueling', True))}\n")
        f.write(f"ddqn_lr_decay={bool(getattr(configs, 'ddqn_lr_decay', True))}\n")
        f.write(f"ddqn_lr_end={float(getattr(configs, 'ddqn_lr_end', 1e-5))}\n")
        f.write(f"ddqn_use_per={bool(getattr(configs, 'ddqn_use_per', False))}\n")
        f.write(f"ddqn_per_alpha={float(getattr(configs, 'ddqn_per_alpha', 0.6))}\n")
        f.write(f"ddqn_per_beta_start={float(getattr(configs, 'ddqn_per_beta_start', 0.4))}\n")
        f.write(f"ddqn_per_beta_end={float(getattr(configs, 'ddqn_per_beta_end', 1.0))}\n")
        f.write(f"ddqn_per_eps={float(getattr(configs, 'ddqn_per_eps', 1e-3))}\n")
        f.write(f"ddqn_stratified_min_frac={float(getattr(configs, 'ddqn_stratified_min_frac', 0.0))}\n")

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
