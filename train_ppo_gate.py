import os
import csv
import time
import random
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from event_gate_env import EventGateEnv
from model.ppo_gate_model import PPOGateNet
from params import configs


def build_env() -> EventGateEnv:
    return EventGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )


def make_env(rank: int):
    def _thunk():
        return build_env()
    return _thunk


def current_lr(update_idx: int, max_updates: int, lr_start: float, lr_end: float, use_decay: bool) -> float:
    if not use_decay:
        return float(lr_start)
    if max_updates <= 1:
        return float(lr_end)
    frac = min(1.0, max(0.0, (update_idx - 1) / float(max_updates - 1)))
    return float(lr_start + (lr_end - lr_start) * frac)


def current_entropy(update_idx: int, max_updates: int, entropy_start: float, entropy_end: float, use_decay: bool) -> float:
    if not use_decay:
        return float(entropy_start)
    if max_updates <= 1:
        return float(entropy_end)
    frac = min(1.0, max(0.0, (update_idx - 1) / float(max_updates - 1)))
    return float(entropy_start + (entropy_end - entropy_start) * frac)


def compute_gae_and_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t_steps = rewards.shape[0]
    advantages = np.zeros(t_steps, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            next_value = 0.0
            next_non_terminal = 0.0 if dones[t] else 1.0
        else:
            next_value = float(values[t + 1])
            next_non_terminal = 0.0 if dones[t] else 1.0
        delta = float(rewards[t]) + gamma * next_value * next_non_terminal - float(values[t])
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values.astype(np.float32)
    return advantages, returns


def redistribute_release_rewards(
    rewards: np.ndarray,
    actions: np.ndarray,
    shaping_rewards: np.ndarray,
    decay: float,
) -> np.ndarray:
    redistributed = rewards.astype(np.float32).copy()
    shaping_arr = shaping_rewards.astype(np.float32)
    release_idx = np.flatnonzero(actions.astype(np.int64) == 1)
    if release_idx.size == 0:
        return redistributed
    decay = float(decay)
    decay = min(max(decay, 0.0), 1.0)
    prev_release = -1
    for idx in release_idx:
        raw_shape = float(shaping_arr[idx])
        if abs(raw_shape) <= 1e-12:
            prev_release = int(idx)
            continue
        start = prev_release + 1
        end = int(idx)
        span = end - start + 1
        if span <= 0:
            prev_release = int(idx)
            continue
        redistributed[idx] -= raw_shape
        weights = np.array([decay ** (end - step) for step in range(start, end + 1)], dtype=np.float32)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            redistributed[idx] += raw_shape
            prev_release = int(idx)
            continue
        redistributed[start:end + 1] += (weights / weight_sum) * raw_shape
        prev_release = int(idx)
    return redistributed


def resolve_td_credit_mode() -> str:
    explicit = str(getattr(configs, "td_credit_mode", "")).strip().lower()
    if explicit:
        return explicit
    shaping_reward_coef = float(getattr(configs, "shaping_reward_coef", 0.0))
    td_reward_coef = float(getattr(configs, "td_reward_coef", 0.0))
    if abs(shaping_reward_coef) > 1e-12:
        return "redistribute" if bool(getattr(configs, "release_reward_redistribute", False)) else "step_only"
    if abs(td_reward_coef) > 1e-12:
        return "terminal_only"
    return "step_only"


def resolve_stability_mode() -> str:
    explicit = str(getattr(configs, "stability_mode_v2", "")).strip().lower()
    if explicit:
        return explicit
    legacy = str(getattr(configs, "stability_mode", "immediate_all")).strip().lower()
    if abs(float(getattr(configs, "stability_scale", 0.0))) <= 1e-12:
        return "off"
    if legacy == "immediate_all":
        return "immediate_all"
    if bool(getattr(configs, "stability_terminal_only", False)):
        return "free_threshold_terminal"
    return "free_threshold_distributed"


def apply_stability_mode_to_episode(episode: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mode = resolve_stability_mode()
    if mode != "free_threshold_distributed":
        return episode
    total_penalty = float(episode.get("stability_total_penalty", 0.0))
    if abs(total_penalty) <= 1e-12:
        return episode
    actions = episode["actions"].astype(np.int64)
    release_idx = np.flatnonzero(actions == 1)
    if release_idx.size == 0:
        return episode
    rewards = episode["rewards"].astype(np.float32).copy()
    rewards[release_idx] += float(total_penalty) / float(release_idx.size)
    episode["rewards"] = rewards
    episode["episode_return"] = float(np.sum(rewards))
    episode["reward_stability"] = float(total_penalty)
    return episode


def collect_episode(
    env: EventGateEnv,
    model: PPOGateNet,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    obs, _ = env.reset()
    done = False
    states, actions, logps, values, rewards, dones = [], [], [], [], [], []
    event_ids, actual_tds, baseline_step_tds = [], [], []
    step_shape_rewards = []
    ep_buf = 0.0
    ep_shape = 0.0
    ep_term = 0.0
    ep_stab = 0.0
    ep_flush = 0.0
    stability_total_penalty = 0.0
    info = {}
    while not done:
        s_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(s_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            if float(obs[-1]) >= 0.5:
                action = torch.ones_like(action, dtype=torch.int64)
            logp = dist.log_prob(action)
        act = int(action.item())
        next_obs, reward, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)

        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(act)
        logps.append(float(logp.item()))
        values.append(float(value.item()))
        rewards.append(float(reward))
        dones.append(float(done))
        event_ids.append(int(info.get("event_id", len(event_ids) + 1)))
        actual_tds.append(float(info.get("actual_td", 0.0)))
        baseline_step_tds.append(float(info.get("baseline_event_td", 0.0)))
        step_shape_rewards.append(float(info.get("reward_shaping_penalty", 0.0)))
        ep_buf += float(info.get("reward_buffer_penalty", 0.0))
        ep_shape += float(info.get("reward_shaping_penalty", 0.0))
        ep_term += float(info.get("reward_td_penalty", 0.0))
        ep_stab += float(info.get("reward_stability_penalty", 0.0))
        ep_flush += float(info.get("reward_mk_penalty", 0.0))
        stability_total_penalty = float(info.get("reward_stability_total_penalty", stability_total_penalty))
        obs = next_obs
    rewards_arr = np.asarray(rewards, dtype=np.float32)

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "logps": np.asarray(logps, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "rewards": rewards_arr,
        "dones": np.asarray(dones, dtype=np.float32),
        "event_ids": np.asarray(event_ids, dtype=np.int64),
        "actual_tds": np.asarray(actual_tds, dtype=np.float32),
        "baseline_step_tds": np.asarray(baseline_step_tds, dtype=np.float32),
        "shape_rewards": np.asarray(step_shape_rewards, dtype=np.float32),
        "episode_return": float(np.sum(rewards_arr)),
        "episode_mk": float(info.get("episode_makespan", 0.0)),
        "episode_td": float(info.get("episode_tardiness", 0.0)),
        "episode_rel": int(info.get("release_count", 0)),
        "reward_buffer": float(ep_buf),
        "reward_shaping": float(ep_shape),
        "reward_terminal": float(ep_term),
        "reward_stability": float(ep_stab),
        "stability_total_penalty": float(stability_total_penalty),
        "reward_flush": float(ep_flush),
    }


def get_info_value(infos: Dict[str, np.ndarray], key: str, idx: int, default=0.0):
    if key not in infos:
        return default
    mask_key = f"_{key}"
    if mask_key in infos:
        mask = infos[mask_key]
        if idx >= len(mask) or not bool(mask[idx]):
            return default
    values = infos[key]
    try:
        return values[idx]
    except Exception:
        return default


def collect_episode_batch_vectorized(
    envs: gym.vector.VectorEnv,
    states: np.ndarray,
    model: PPOGateNet,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    num_envs = int(states.shape[0])
    completed = np.zeros(num_envs, dtype=np.bool_)
    ep_states = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_logps = [[] for _ in range(num_envs)]
    ep_values = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_dones = [[] for _ in range(num_envs)]
    ep_event_ids = [[] for _ in range(num_envs)]
    ep_actual_tds = [[] for _ in range(num_envs)]
    ep_baseline_step_tds = [[] for _ in range(num_envs)]
    ep_shape_rewards = [[] for _ in range(num_envs)]
    ep_buf = np.zeros(num_envs, dtype=np.float32)
    ep_shape = np.zeros(num_envs, dtype=np.float32)
    ep_term = np.zeros(num_envs, dtype=np.float32)
    ep_stab = np.zeros(num_envs, dtype=np.float32)
    ep_flush = np.zeros(num_envs, dtype=np.float32)
    ep_stab_total = np.zeros(num_envs, dtype=np.float32)
    episode_mk = np.zeros(num_envs, dtype=np.float32)
    episode_td = np.zeros(num_envs, dtype=np.float32)
    episode_rel = np.zeros(num_envs, dtype=np.int64)

    curr_states = np.asarray(states, dtype=np.float32).copy()

    while not np.all(completed):
        active_mask = ~completed
        actions = np.zeros(num_envs, dtype=np.int64)
        logps = np.zeros(num_envs, dtype=np.float32)
        values = np.zeros(num_envs, dtype=np.float32)

        active_idx = np.flatnonzero(active_mask)
        if active_idx.size > 0:
            obs_active = torch.as_tensor(curr_states[active_idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(obs_active)
                dist = Categorical(logits=logits)
                action = dist.sample()
                last_mask = obs_active[:, -1] >= 0.5
                if torch.any(last_mask):
                    action = action.clone()
                    action[last_mask] = 1
                logp = dist.log_prob(action)
            actions[active_idx] = action.cpu().numpy()
            logps[active_idx] = logp.cpu().numpy()
            values[active_idx] = value.cpu().numpy()

        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        done_mask = np.logical_or(terminated, truncated)

        for idx in active_idx:
            done = bool(done_mask[idx])
            ep_states[idx].append(np.asarray(curr_states[idx], dtype=np.float32).copy())
            ep_actions[idx].append(int(actions[idx]))
            ep_logps[idx].append(float(logps[idx]))
            ep_values[idx].append(float(values[idx]))
            ep_rewards[idx].append(float(rewards[idx]))
            ep_dones[idx].append(float(done))
            ep_event_ids[idx].append(int(get_info_value(infos, "event_id", idx, len(ep_event_ids[idx]) + 1)))
            ep_actual_tds[idx].append(float(get_info_value(infos, "actual_td", idx, 0.0)))
            ep_baseline_step_tds[idx].append(float(get_info_value(infos, "baseline_event_td", idx, 0.0)))
            ep_shape_rewards[idx].append(float(get_info_value(infos, "reward_shaping_penalty", idx, 0.0)))
            ep_buf[idx] += float(get_info_value(infos, "reward_buffer_penalty", idx, 0.0))
            ep_shape[idx] += float(get_info_value(infos, "reward_shaping_penalty", idx, 0.0))
            ep_term[idx] += float(get_info_value(infos, "reward_td_penalty", idx, 0.0))
            ep_stab[idx] += float(get_info_value(infos, "reward_stability_penalty", idx, 0.0))
            ep_flush[idx] += float(get_info_value(infos, "reward_mk_penalty", idx, 0.0))
            ep_stab_total[idx] = float(get_info_value(infos, "reward_stability_total_penalty", idx, ep_stab_total[idx]))
            if done:
                completed[idx] = True
                episode_mk[idx] = float(get_info_value(infos, "episode_makespan", idx, 0.0))
                episode_td[idx] = float(get_info_value(infos, "episode_tardiness", idx, 0.0))
                episode_rel[idx] = int(get_info_value(infos, "release_count", idx, 0))

        curr_states = np.asarray(next_states, dtype=np.float32).copy()
        if np.any(done_mask) and not np.all(completed):
            curr_states, _ = envs.reset(
                options={
                    "reset_mask": done_mask.astype(np.bool_),
                    "count_episode": False,
                }
            )
            curr_states = np.asarray(curr_states, dtype=np.float32).copy()

    episodes: List[Dict[str, np.ndarray]] = []
    for idx in range(num_envs):
        rewards_arr = np.asarray(ep_rewards[idx], dtype=np.float32)
        episodes.append({
            "states": np.asarray(ep_states[idx], dtype=np.float32),
            "actions": np.asarray(ep_actions[idx], dtype=np.int64),
            "logps": np.asarray(ep_logps[idx], dtype=np.float32),
            "values": np.asarray(ep_values[idx], dtype=np.float32),
            "rewards": rewards_arr,
            "dones": np.asarray(ep_dones[idx], dtype=np.float32),
            "event_ids": np.asarray(ep_event_ids[idx], dtype=np.int64),
            "actual_tds": np.asarray(ep_actual_tds[idx], dtype=np.float32),
            "baseline_step_tds": np.asarray(ep_baseline_step_tds[idx], dtype=np.float32),
            "shape_rewards": np.asarray(ep_shape_rewards[idx], dtype=np.float32),
            "episode_return": float(np.sum(rewards_arr)),
            "episode_mk": float(episode_mk[idx]),
            "episode_td": float(episode_td[idx]),
            "episode_rel": int(episode_rel[idx]),
            "reward_buffer": float(ep_buf[idx]),
            "reward_shaping": float(ep_shape[idx]),
            "reward_terminal": float(ep_term[idx]),
            "reward_stability": float(ep_stab[idx]),
            "stability_total_penalty": float(ep_stab_total[idx]),
            "reward_flush": float(ep_flush[idx]),
        })
    return episodes


def merge_episode_batch(
    episodes: List[Dict[str, np.ndarray]],
    gamma: float,
    gae_lambda: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, float]]:
    use_release_redistribute = (resolve_td_credit_mode() == "redistribute")
    release_decay = float(getattr(configs, "release_reward_decay", 0.9))
    if use_release_redistribute:
        for ep in episodes:
            ep["rewards"] = redistribute_release_rewards(
                rewards=ep["rewards"],
                actions=ep["actions"],
                shaping_rewards=ep["shape_rewards"],
                decay=release_decay,
            )
            ep["episode_return"] = float(np.sum(ep["rewards"]))
    for ep in episodes:
        apply_stability_mode_to_episode(ep)
    batch = {
        "states": np.concatenate([ep["states"] for ep in episodes], axis=0),
        "actions": np.concatenate([ep["actions"] for ep in episodes], axis=0),
        "logps": np.concatenate([ep["logps"] for ep in episodes], axis=0),
        "values": np.concatenate([ep["values"] for ep in episodes], axis=0),
        "rewards": np.concatenate([ep["rewards"] for ep in episodes], axis=0),
        "dones": np.concatenate([ep["dones"] for ep in episodes], axis=0),
    }
    adv_list, ret_list = [], []
    for ep in episodes:
        adv, ret = compute_gae_and_returns(ep["rewards"], ep["values"], ep["dones"], gamma, gae_lambda)
        adv_list.append(adv)
        ret_list.append(ret)
    advantages = np.concatenate(adv_list, axis=0)
    returns = np.concatenate(ret_list, axis=0)
    summary = {
        "episode_return": float(np.mean([ep["episode_return"] for ep in episodes])),
        "episode_mk": float(np.mean([ep["episode_mk"] for ep in episodes])),
        "episode_td": float(np.mean([ep["episode_td"] for ep in episodes])),
        "episode_rel": float(np.mean([ep["episode_rel"] for ep in episodes])),
        "reward_buffer": float(np.mean([ep["reward_buffer"] for ep in episodes])),
        "reward_shaping": float(np.mean([ep["reward_shaping"] for ep in episodes])),
        "reward_terminal": float(np.mean([ep["reward_terminal"] for ep in episodes])),
        "reward_stability": float(np.mean([ep["reward_stability"] for ep in episodes])),
        "reward_flush": float(np.mean([ep["reward_flush"] for ep in episodes])),
        "samples_per_update": int(batch["states"].shape[0]),
    }
    return batch, advantages, returns, summary


def update_ppo(
    model: PPOGateNet,
    optimizer: Adam,
    device: torch.device,
    batch: Dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    update_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    obs_t = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    act_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
    old_logp_t = torch.as_tensor(batch["logps"], dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    n = obs_t.shape[0]
    losses, p_losses, v_losses, entropies = [], [], [], []
    for _ in range(int(update_epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, int(minibatch_size)):
            idx = perm[start:start + int(minibatch_size)]
            logits, value_pred = model(obs_t[idx])
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_t[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value_pred, ret_t[idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            p_losses.append(float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "policy_loss": float(np.mean(p_losses)) if p_losses else 0.0,
        "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def build_fixed_probe_states(probe_seed: int, max_states: int = 64) -> np.ndarray:
    probe_env = build_env()
    obs, _ = probe_env.reset(seed=probe_seed)
    samples = [np.asarray(obs, dtype=np.float32).copy()]
    done = False
    step_idx = 0
    while (not done) and (len(samples) < max_states):
        action = 1 if (step_idx % 2 == 0) else 0
        obs, _, terminated, truncated, _ = probe_env.step(action)
        done = bool(terminated or truncated)
        samples.append(np.asarray(obs, dtype=np.float32).copy())
        step_idx += 1
    return np.asarray(samples, dtype=np.float32)


def evaluate_policy_gap(model: PPOGateNet, probe_states: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        obs_t = torch.as_tensor(probe_states, dtype=torch.float32, device=device)
        logits, _ = model(obs_t)
        probs = torch.softmax(logits, dim=1)
        gap = torch.abs(probs[:, 1] - probs[:, 0]).mean().item()
    return float(gap)


def eval_same_problem_greedy(model: PPOGateNet, device: torch.device, eval_seed: int) -> Tuple[float, float, int]:
    env = build_env()
    obs, _ = env.reset(seed=int(eval_seed))
    done = False
    info = {}
    while not done:
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            act = int(torch.argmax(logits, dim=1).item())
        if float(obs[-1]) >= 0.5:
            act = 1
        obs, _, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)
    kpi = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    return float(kpi["makespan"]), float(kpi["tardiness"]), int(info.get("release_count", 0))


def validate_greedy(model: PPOGateNet, device: torch.device, seed: int = 999) -> Tuple[float, float, int]:
    env = build_env()
    obs, _ = env.reset(seed=int(seed))
    done = False
    info = {}
    while not done:
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            act = int(torch.argmax(logits, dim=1).item())
        if float(obs[-1]) >= 0.5:
            act = 1
        obs, _, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)
    kpi = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    return float(kpi["makespan"]), float(kpi["tardiness"]), int(info.get("release_count", 0))


def train_ppo_gate():
    start_ts = time.time()
    seed = int(getattr(configs, "ppo_gate_train_seed", getattr(configs, "event_seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(getattr(configs, "device", "cpu"))
    model = PPOGateNet(
        obs_dim=22,
        n_actions=2,
        hidden=int(getattr(configs, "ppo_gate_hidden_dim", 256)),
        num_layers=int(getattr(configs, "ppo_gate_num_layers", 3)),
        separate_trunks=bool(getattr(configs, "ppo_gate_separate_trunks", False)),
        actor_hidden=int(getattr(configs, "ppo_gate_actor_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
        actor_num_layers=int(getattr(configs, "ppo_gate_actor_num_layers", getattr(configs, "ppo_gate_num_layers", 3))),
        critic_hidden=int(getattr(configs, "ppo_gate_critic_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
        critic_num_layers=int(getattr(configs, "ppo_gate_critic_num_layers", getattr(configs, "ppo_gate_num_layers", 3))),
        value_hidden=int(getattr(configs, "ppo_gate_value_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
        value_num_layers=int(getattr(configs, "ppo_gate_value_num_layers", 1)),
    ).to(device)

    updates = int(getattr(configs, "ppo_gate_updates", 200))
    lr_start = float(getattr(configs, "ppo_gate_lr", 1e-4))
    lr_end = float(getattr(configs, "ppo_gate_lr_end", 5e-5))
    lr_decay = bool(getattr(configs, "ppo_gate_lr_decay", True))
    gamma = float(getattr(configs, "ppo_gate_gamma", 0.99))
    gae_lambda = float(getattr(configs, "ppo_gate_gae_lambda", 0.95))
    clip_ratio = float(getattr(configs, "ppo_gate_clip", 0.2))
    value_coef = float(getattr(configs, "ppo_gate_value_coef", 0.5))
    entropy_start = float(getattr(configs, "ppo_gate_entropy_coef", 0.01))
    entropy_end = float(getattr(configs, "ppo_gate_entropy_end", entropy_start))
    entropy_decay = bool(getattr(configs, "ppo_gate_entropy_decay", False))
    update_epochs = int(getattr(configs, "ppo_gate_update_epochs", 5))
    minibatch_size = int(getattr(configs, "ppo_gate_minibatch_size", 64))
    max_grad_norm = float(getattr(configs, "ppo_gate_max_grad_norm", 1.0))
    num_envs = max(1, int(getattr(configs, "ppo_gate_num_envs", 1)))
    use_async_envs = bool(getattr(configs, "ppo_gate_use_async_envs", True))
    same_problem_eval_every = int(getattr(configs, "ppo_gate_same_problem_eval_every", 0))
    validate_every = int(getattr(configs, "ppo_gate_validate_every", 10))
    episodes_per_instance = max(
        1,
        int(getattr(configs, "ppo_gate_instance_episodes", 10)),
    )
    optimizer = Adam(model.parameters(), lr=lr_start)
    env_fns = [make_env(i) for i in range(num_envs)]
    seed_stride = 100_000
    seeds = [int(seed + env_idx * seed_stride) for env_idx in range(num_envs)]
    vector_kwargs = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
    if use_async_envs:
        try:
            envs = gym.vector.AsyncVectorEnv(env_fns, **vector_kwargs)
            states, _ = envs.reset(seed=seeds)
        except Exception:
            try:
                envs.close()
            except Exception:
                pass
            envs = gym.vector.SyncVectorEnv(env_fns, **vector_kwargs)
            states, _ = envs.reset(seed=seeds)
            use_async_envs = False
    else:
        envs = gym.vector.SyncVectorEnv(env_fns, **vector_kwargs)
        states, _ = envs.reset(seed=seeds)
    states = np.asarray(states, dtype=np.float32)

    probe_states = build_fixed_probe_states(seed)
    name = str(getattr(configs, "ppo_gate_name", "ppo_gate_latest"))
    plot_dir = os.path.join("plots", "train_ppo")
    os.makedirs(plot_dir, exist_ok=True)
    history_ret, history_loss, history_ploss, history_vloss, history_entropy = [], [], [], [], []
    history_lr, history_pgap = [], []
    history_mk, history_td, history_rel = [], [], []
    history_buf, history_shape, history_term, history_stab, history_flush = [], [], [], [], []
    history_gmk, history_gtd, history_grel = [], [], []
    history_val_mk, history_val_td, history_val_rel = [], [], []
    val_updates, val_mk, val_td, val_rel = [], [], [], []

    pbar = tqdm(total=updates, desc="PPO Gate Updates")
    for update_idx in range(1, updates + 1):
        lr_now = current_lr(update_idx, updates, lr_start, lr_end, lr_decay)
        entropy_now = current_entropy(update_idx, updates, entropy_start, entropy_end, entropy_decay)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        episodes = collect_episode_batch_vectorized(envs, states, model, device)
        states, _ = envs.reset()
        states = np.asarray(states, dtype=np.float32)
        batch, adv, ret, ep = merge_episode_batch(
            episodes,
            gamma,
            gae_lambda,
        )
        upd = update_ppo(
            model=model,
            optimizer=optimizer,
            device=device,
            batch=batch,
            advantages=adv,
            returns=ret,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_now,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
        )

        history_ret.append(float(ep["episode_return"]))
        history_loss.append(float(upd["loss"]))
        history_ploss.append(float(upd["policy_loss"]))
        history_vloss.append(float(upd["value_loss"]))
        history_entropy.append(float(upd["entropy"]))
        history_lr.append(float(lr_now))
        history_pgap.append(float(evaluate_policy_gap(model, probe_states, device)))
        history_mk.append(float(ep["episode_mk"]))
        history_td.append(float(ep["episode_td"]))
        history_rel.append(float(ep["episode_rel"]))
        history_buf.append(float(ep["reward_buffer"]))
        history_shape.append(float(ep["reward_shaping"]))
        history_term.append(float(ep["reward_terminal"]))
        history_stab.append(float(ep["reward_stability"]))
        history_flush.append(float(ep["reward_flush"]))
        history_val_mk.append(float("nan"))
        history_val_td.append(float("nan"))
        history_val_rel.append(float("nan"))

        do_same_eval = (same_problem_eval_every > 0) and ((update_idx == 1) or (update_idx % same_problem_eval_every == 0))
        if do_same_eval:
            same_seed = int(seed + ((update_idx - 1) // episodes_per_instance))
            gmk, gtd, grel = eval_same_problem_greedy(model, device, same_seed)
        elif history_gmk:
            gmk, gtd, grel = history_gmk[-1], history_gtd[-1], history_grel[-1]
        else:
            gmk, gtd, grel = 0.0, 0.0, 0
        history_gmk.append(float(gmk))
        history_gtd.append(float(gtd))
        history_grel.append(int(grel))

        if (validate_every > 0) and (update_idx % validate_every == 0):
            vmk, vtd, vrel = validate_greedy(model, device, seed=999)
            val_updates.append(update_idx)
            val_mk.append(vmk)
            val_td.append(vtd)
            val_rel.append(vrel)
            history_val_mk[-1] = float(vmk)
            history_val_td[-1] = float(vtd)
            history_val_rel[-1] = float(vrel)
            tqdm.write(f">>> [VAL {update_idx:04d}] MK={vmk:7.1f} | TD={vtd:8.1f} | Rel={vrel:4d}")

        avg10 = float(np.mean(history_ret[-10:]))
        line_parts = [
            f"[UPD {update_idx:04d}] R={history_ret[-1]:8.2f} (avg10={avg10:8.2f})",
            f"L={history_loss[-1]:7.4f}",
            f"PL={history_ploss[-1]:7.4f}",
            f"VL={history_vloss[-1]:7.4f}",
            f"LR={history_lr[-1]:.2e}",
            f"MK={history_mk[-1]:7.1f}",
            f"TD={history_td[-1]:8.1f}",
            f"Rel={history_rel[-1]:4.0f}",
            f"Rbuf={history_buf[-1]:7.2f}",
            f"RTD={history_term[-1]:7.2f}",
            f"Rstab={history_stab[-1]:7.2f}",
            f"Pgap={history_pgap[-1]:6.3f}",
        ]
        if abs(history_shape[-1]) > 1e-12:
            line_parts.insert(9, f"Rshape={history_shape[-1]:7.2f}")
        if abs(history_flush[-1]) > 1e-12:
            insert_idx = len(line_parts) - 1
            line_parts.insert(insert_idx, f"RMK={history_flush[-1]:7.2f}")
        tqdm.write(" | ".join(line_parts))
        if same_problem_eval_every > 0:
            tqdm.write(
                f"           GTD={history_gtd[-1]:8.1f} | GMK={history_gmk[-1]:7.1f} | GRel={history_grel[-1]:4d}"
            )
        pbar.update(1)

    pbar.close()
    envs.close()

    ckpt_dir = "ppo_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{name}.pth"))

    train_log_path = os.path.join(plot_dir, f"train_log_{name}.csv")
    val_log_path = os.path.join(plot_dir, f"val_log_{name}.csv")
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Update", "Return", "Loss", "Policy_Loss", "Value_Loss", "Entropy", "LR", "P_Gap",
            "Val_Makespan", "Val_Tardiness", "Val_Release_Count",
            "Makespan", "Tardiness", "Release_Count",
            "Buffer_Reward", "Shaping_Reward", "TD_Reward", "Stability_Reward", "MK_Reward",
        ])
        for i in range(len(history_ret)):
            writer.writerow([
                i + 1, history_ret[i], history_loss[i], history_ploss[i], history_vloss[i], history_entropy[i], history_lr[i], history_pgap[i],
                history_val_mk[i], history_val_td[i], history_val_rel[i],
                history_mk[i], history_td[i], history_rel[i],
                history_buf[i], history_shape[i], history_term[i], history_stab[i], history_flush[i],
            ])

    with open(val_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Update", "Makespan", "Tardiness", "Release_Count"])
        for i in range(len(val_updates)):
            writer.writerow([val_updates[i], val_mk[i], val_td[i], val_rel[i]])

    x = np.arange(1, len(history_ret) + 1)
    fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
    axs[0].plot(x, history_ret, color="tab:red", alpha=0.8)
    axs[0].set_title("Training Return")
    axs[1].plot(x, history_loss, label="Total", color="tab:orange")
    axs[1].plot(x, history_ploss, label="Policy", color="tab:green")
    axs[1].plot(x, history_vloss, label="Value", color="tab:blue")
    axs[1].legend()
    axs[1].set_title("Training Loss")
    axs[2].plot(x, history_td, color="tab:green", alpha=0.8)
    axs[2].set_title("Train Tardiness")
    axs[3].plot(val_updates, val_rel, marker="o", color="tab:purple", alpha=0.8)
    axs[3].set_title("Validation Release Count")
    axs[4].plot(val_updates, val_td, marker="o", color="tab:blue", alpha=0.8)
    axs[4].set_title("Validation Tardiness")
    for ax in axs:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"curve_{name}.png"), dpi=150)
    plt.close(fig)

    elapsed = max(0.0, time.time() - start_ts)
    with open(os.path.join(plot_dir, f"time_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"ppo_gate_name={name}\n")
        f.write(f"config_path={getattr(configs, 'config', '')}\n")
        f.write(f"elapsed_seconds={elapsed:.3f}\n")
        f.write(f"elapsed_hms={time.strftime('%H:%M:%S', time.gmtime(elapsed))}\n")
        f.write(f"updates={updates}\n")
        f.write(f"ppo_gate_lr={lr_start}\n")
        f.write(f"ppo_gate_lr_end={lr_end}\n")
        f.write(f"ppo_gate_gamma={gamma}\n")
        f.write(f"ppo_gate_gae_lambda={gae_lambda}\n")
        f.write(f"ppo_gate_clip={clip_ratio}\n")
        f.write(f"ppo_gate_entropy_coef={entropy_start}\n")
        f.write(f"ppo_gate_entropy_decay={entropy_decay}\n")
        f.write(f"ppo_gate_entropy_end={entropy_end}\n")
        f.write(f"ppo_gate_value_coef={value_coef}\n")
        f.write(f"ppo_gate_update_epochs={update_epochs}\n")
        f.write(f"ppo_gate_minibatch_size={minibatch_size}\n")
        f.write(f"ppo_gate_num_envs={num_envs}\n")
        f.write(f"ppo_gate_use_async_envs={use_async_envs}\n")
        f.write(f"ppo_gate_instance_episodes={episodes_per_instance}\n")


if __name__ == "__main__":
    train_ppo_gate()
