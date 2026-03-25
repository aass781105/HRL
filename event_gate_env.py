import math
import copy
import numpy as np
from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
from params import configs
from data_utils import SD2_instance_generator, generate_due_dates
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator, split_matrix_to_jobs
from model.gate_state import calculate_gate_state

class EventGateEnv(gym.Env):
    def __init__(self,
                 n_machines: int,
                 heuristic: str = "SPT",
                 interarrival_mean: float = 0.1,
                 burst_K: int = 10,
                 event_horizon: int = 200,
                 init_jobs: int = 0,
                 obs_buffer_cap: Optional[int] = None):
        super().__init__()
        self.M = int(n_machines)
        self.interarrival_mean = float(interarrival_mean)
        self.burst_K = int(burst_K)
        self.event_horizon = int(event_horizon)
        self.init_jobs = int(init_jobs)
        self.mean_pt = (float(configs.low) + float(configs.high)) / 2.0
        self.time_scale = self.mean_pt
        self.reward_base_scale = self.mean_pt
        self.gen = None
        self.orch = None
        self.t_now = 0.0
        self.t_next = 0.0
        self.events_done = 0
        self.episode_tardiness = 0.0
        self.release_count = 0
        self.agent_release_count = 0
        self.all_job_due_dates = {}
        self._prev_arrival_time = 0.0
        self._cached_projected_td = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32) 
        self.action_space = spaces.Discrete(2)

    @staticmethod
    def _compress_rel_tail(raw_value: float, threshold: float = 2.0, tail_scale: float = 1.0) -> float:
        abs_value = abs(float(raw_value))
        if abs_value <= threshold:
            return float(raw_value)
        tail = np.log1p((abs_value - threshold) / tail_scale)
        return float(np.sign(raw_value) * (threshold + tail))

    def _advance_sim_to_next_arrival(self, gen, orch, all_job_due_dates, t_next: float):
        t_event = float(t_next)
        new_jobs = gen.generate_burst(t_event)
        if new_jobs:
            for j in new_jobs:
                all_job_due_dates[j.job_id] = j.meta["due_date"]
            orch.buffer.extend(new_jobs)
        return t_event, float(gen.sample_next_time(t_event))

    def _advance_to_next_arrival(self):
        self.t_now, self.t_next = self._advance_sim_to_next_arrival(
            self.gen, self.orch, self.all_job_due_dates, self.t_next
        )

    @staticmethod
    def _clone_orchestrator(src: GlobalTimelineOrchestrator) -> GlobalTimelineOrchestrator:
        dst = GlobalTimelineOrchestrator.__new__(GlobalTimelineOrchestrator)
        dst.M = int(src.M)
        dst.t = float(src.t)
        dst.generator = src.generator
        dst.select_from_buffer = src.select_from_buffer
        dst.buffer = copy.deepcopy(src.buffer)
        dst.machine_free_time = np.asarray(src.machine_free_time, dtype=float).copy()
        dst._global_rows = [dict(r) for r in src._global_rows]
        dst._global_row_keys = set(src._global_row_keys)
        dst._last_full_rows = [dict(r) for r in src._last_full_rows]
        dst._last_jobs_snapshot = copy.deepcopy(src._last_jobs_snapshot)
        dst._job_history_finishes = dict(src._job_history_finishes)
        dst._release_count = int(src._release_count)
        dst.method = src.method
        dst._ppo = src._ppo
        return dst

    def _estimate_release_now_td(self, orch: GlobalTimelineOrchestrator, t_event: float) -> float:
        actual_td = float(orch.get_total_tardiness_estimate(self.all_job_due_dates))
        if len(orch.buffer) <= 0:
            return actual_td
        shadow = self._clone_orchestrator(orch)
        shadow.event_release_and_reschedule(float(t_event))
        return float(shadow.get_total_tardiness_estimate(self.all_job_due_dates))

    def _build_simulation(self, instance_seed: int):
        rng = np.random.default_rng(int(instance_seed))
        gen = EventBurstGenerator(
            SD2_instance_generator,
            copy.deepcopy(configs),
            self.M,
            self.interarrival_mean,
            lambda _r: int(self.burst_K),
            rng,
        )
        orch = GlobalTimelineOrchestrator(self.M, gen, t0=0.0)
        all_job_due_dates = {}
        release_count = 0
        t_now = 0.0
        if self.init_jobs > 0:
            jl, pt, _ = SD2_instance_generator(copy.deepcopy(configs), rng=rng)
            dd_rel = generate_due_dates(
                jl,
                pt,
                tightness=getattr(configs, "due_date_tightness", 1.2),
                due_date_mode='k',
                rng=rng,
            )
            init_jobs = split_matrix_to_jobs(jl, pt, base_job_id=0, t_arrive=0.0, due_dates=0.0 + dd_rel)
            for j in init_jobs:
                all_job_due_dates[j.job_id] = j.meta["due_date"]
            orch.buffer.extend(init_jobs)
            gen.bump_next_id(max((j.job_id for j in init_jobs)) + 1)
            orch.event_release_and_reschedule(0.0)
            release_count += 1
        t_next = float(gen.sample_next_time(t_now))
        t_now, t_next = self._advance_sim_to_next_arrival(gen, orch, all_job_due_dates, t_next)
        return rng, gen, orch, all_job_due_dates, t_now, t_next, release_count

    def _observe(self) -> np.ndarray:
        t_now = self.t_now
        rem = np.maximum(0.0, self.orch.machine_free_time - t_now)
        mx_l = np.max(rem)
        w_idle, u_idle = self.orch.compute_idle_stats(t_now, mx_l)
        buf_stats = self._get_buffer_stats(t_now)
        wip_stats = self.orch.get_wip_stats(t_now)
        return calculate_gate_state(len(self.orch.buffer), self.orch.machine_free_time, t_now, self.M, 0, self.time_scale, w_idle, u_idle, buf_stats, wip_stats)

    def _get_buffer_stats(self, t_now: float):
        if not self.orch.buffer: return {"tardiness_ratio": 0.0, "min_slack": 0.0, "avg_slack": 0.0, "slack_std": 0.0}
        slacks, neg_count = [], 0
        for j in self.orch.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0)); due = self.all_job_due_dates[j.job_id]; s = due - t_now - mw
            slacks.append(s); 
            if t_now + mw > due: neg_count += 1
        return {"tardiness_ratio": neg_count / len(self.orch.buffer), "min_slack": min(slacks), "avg_slack": sum(slacks) / len(slacks), "slack_std": float(np.std(slacks))}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        options = options or {}
        count_episode = bool(options.get("count_episode", True))
        if seed is not None:
            self.master_seed = int(seed)
            self.internal_episode_counter = 0
        elif not hasattr(self, 'master_seed'):
            self.master_seed = int(getattr(configs, "event_seed", 42))
            self.internal_episode_counter = 0

        episodes_per_instance = max(1, int(getattr(configs, "ppo_gate_instance_episodes", 10)))
        episode_counter = int(self.internal_episode_counter)
        instance_seed = int(self.master_seed + (episode_counter // episodes_per_instance))
        if count_episode:
            self.internal_episode_counter += 1
        
        # Call super().reset for Gymnasium compliance (manages self.np_random)
        super().reset(seed=instance_seed)
        
        _, self.gen, self.orch, self.all_job_due_dates, self.t_now, self.t_next, self.release_count = self._build_simulation(instance_seed)
        self.episode_tardiness, self.events_done = 0.0, 0
        self.agent_release_count = 0
        self._prev_arrival_time = 0.0
        self._cached_projected_td = None

        return self._observe(), {"t_now": self.t_now, "instance_seed": instance_seed}

    def step(self, action: int):
        t_event = float(self.t_now)
        t_next = float(self.t_next)
        current_event_idx = int(self.events_done + 1)
        inter_arrival = float(t_event - self._prev_arrival_time)
        baseline_td_at_step = 0.0
        shaping_reward_coef = float(getattr(configs, "shaping_reward_coef", 0.0))
        td_reward_coef = float(getattr(configs, "td_reward_coef", 0.1))
        use_shaping = abs(shaping_reward_coef) > 1e-12
        if use_shaping:
            phi_before = float(self._cached_projected_td) if self._cached_projected_td is not None else self._estimate_release_now_td(self.orch, t_event)
        else:
            phi_before = 0.0
            self._cached_projected_td = None

        if int(action) == 1:
            self.orch.event_release_and_reschedule(t_event)
            self.release_count += 1
            self.agent_release_count += 1
        else:
            self.orch.tick_without_release(t_event)
        actual_td_now = float(self.orch.get_total_tardiness_estimate(self.all_job_due_dates))

        # Common rewards (Buffer, Stability)
        scale = self.time_scale
        terminal_stability_only = bool(getattr(configs, "stability_terminal_only", False))
        free_releases = max(0, int(getattr(configs, "stability_free_releases", 0)))
        stability_scale = float(configs.stability_scale)
        excess_releases = max(0, int(self.agent_release_count) - free_releases)
        prev_excess_releases = max(0, int(self.agent_release_count) - 1 - free_releases)
        marginal_stab = float(excess_releases * (excess_releases + 1) // 2 - prev_excess_releases * (prev_excess_releases + 1) // 2)
        charged_release = marginal_stab if int(action) == 1 else 0.0
        r_stab = 0.0 if terminal_stability_only else -(float(charged_release) * stability_scale)
        total_neg_slack = 0.0
        for job_state in self.orch.buffer:
            due_abs = float(job_state.meta.get("due_date", t_event))
            rem_work = 0.0
            for op in job_state.operations:
                v = np.array(op.time_row)
                rem_work += float(np.mean(v[v > 0])) if v[v > 0].size else 0.0
            slack = due_abs - t_event - rem_work
            if slack < 0.0:
                total_neg_slack += float(-slack)
            elif slack < 100.0:
                total_neg_slack += float((100.0 - slack) * 0.5)
        r_buf = -(total_neg_slack * float(configs.buffer_penalty_coef)) / self.time_scale

        self.events_done += 1
        done = bool(self.events_done >= self.event_horizon)

        r_td = 0.0
        r_mk = 0.0
        ep_mk = 0.0
        if done:
            while len(self.orch.buffer) > 0:
                self.orch.event_release_and_reschedule(t_next)
                self.release_count += 1
            
            final = self.orch.get_final_kpi_stats(self.all_job_due_dates)
            self.episode_tardiness = final["tardiness"]
            ep_mk = final["makespan"]
            phi_after = float(self.episode_tardiness) if use_shaping else 0.0
            terminal_scale = max(scale * float(self.event_horizon), scale)
            r_td = float((-(self.episode_tardiness) / terminal_scale) * td_reward_coef)
            self._cached_projected_td = None
            
            mk_norm = max(scale * float(self.event_horizon), scale)
            mk_ratio = (ep_mk / mk_norm)
            r_mk_raw = -(((mk_ratio + 1.0) ** 2) - 1.0) * float(configs.mk_reward_coef)
            r_mk = float(r_mk_raw)
            if terminal_stability_only:
                charged_releases = max(0, int(self.agent_release_count) - free_releases)
                cumulative_stab = float(charged_releases * (charged_releases + 1) // 2)
                r_stab = -(cumulative_stab * stability_scale)
        else:
            self._advance_to_next_arrival()
            if use_shaping:
                phi_after = float(self._estimate_release_now_td(self.orch, self.t_now))
                self._cached_projected_td = phi_after
            else:
                phi_after = 0.0
                self._cached_projected_td = None
        self._prev_arrival_time = t_event

        r_shape = float((-(phi_after - phi_before) / scale) * shaping_reward_coef) if use_shaping else 0.0
        reward = r_stab + r_buf + r_shape + r_td + r_mk

        info = {
            "event_id": current_event_idx,
            "time": t_event,
            "inter_arrival": inter_arrival,
            "actual_td": actual_td_now,
            "baseline_td": baseline_td_at_step,
            "episode_tardiness": self.episode_tardiness, 
            "episode_makespan": ep_mk,
            "release_count": self.release_count, 
            "reward_buffer_penalty": r_buf, 
            "reward_shaping_penalty": r_shape,
            "reward_td_penalty": r_td,
            "reward_release_raw_penalty": r_shape,
            "reward_td_terminal_penalty": r_td,
            "reward_stability_penalty": r_stab, 
            "reward_mk_penalty": r_mk,
            "phi_before": float(phi_before),
            "phi_after": float(phi_after),
            "agent_last_release_td": float(0.0),
            "agent_final_td": float(self.episode_tardiness if done else 0.0),
            "baseline_last_release_td": float(0.0),
            "baseline_final_td": float(0.0),
            "baseline_final_mk": float(0.0),
        }
        
        return self._observe(), float(reward), done, False, info
