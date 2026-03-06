import math
import copy
import numpy as np
from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
from params import configs
from data_utils import SD2_instance_generator, generate_due_dates
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator, split_matrix_to_jobs
from model.ddqn_model import calculate_ddqn_state, log_scale_reward

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
        self.shadow_orch = None # [NEW] Counterfactual baseline
        self.t_now = 0.0
        self.events_done = 0
        self.episode_tardiness = 0.0
        self.release_count = 0
        self.all_job_due_dates = {}
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32) 
        self.action_space = spaces.Discrete(2)

    def _observe(self) -> np.ndarray:
        mft_abs = np.asarray(self.orch.machine_free_time, dtype=float)
        rem = np.maximum(0.0, mft_abs - float(self.t_now)); mx_l = np.max(rem)
        w_idle = self.orch.compute_weighted_idle(float(self.t_now), mx_l)
        u_idle = self.orch.compute_unweighted_idle(float(self.t_now), mx_l)
        buf_stats = self._get_buffer_stats(float(self.t_now))
        wip_stats = self.orch.get_wip_stats(float(self.t_now))
        return calculate_ddqn_state(len(self.orch.buffer), self.orch.machine_free_time, float(self.t_now), self.M, 0, self.time_scale, w_idle, u_idle, buf_stats, wip_stats)

    def _get_buffer_stats(self, t_now: float):
        if not self.orch.buffer: return {"tardiness_ratio": 0.0, "min_slack": 0.0, "avg_slack": 0.0, "slack_std": 0.0}
        slacks, neg_count = [], 0
        for j in self.orch.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0)); due = self.all_job_due_dates[j.job_id]; s = due - t_now - mw
            slacks.append(s); 
            if t_now + mw > due: neg_count += 1
        return {"tardiness_ratio": neg_count / len(self.orch.buffer), "min_slack": min(slacks), "avg_slack": sum(slacks) / len(slacks), "slack_std": float(np.std(slacks))}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None: seed = getattr(configs, "event_seed", 42)
        rng = np.random.default_rng(int(seed))
        self.all_job_due_dates, self.episode_tardiness, self.t_now, self.events_done, self.release_count = {}, 0.0, 0.0, 0, 0
        self.gen = EventBurstGenerator(SD2_instance_generator, copy.deepcopy(configs), self.M, self.interarrival_mean, lambda _r: int(self.burst_K), rng)
        self.orch = GlobalTimelineOrchestrator(self.M, self.gen, t0=0.0)
        if self.init_jobs > 0:
            jl, pt, _ = SD2_instance_generator(copy.deepcopy(configs), rng=rng)
            dd_rel = generate_due_dates(jl, pt, tightness=getattr(configs, "due_date_tightness", 1.2), due_date_mode='k', rng=rng)
            init_jobs = split_matrix_to_jobs(jl, pt, base_job_id=0, t_arrive=0.0, due_dates=0.0 + dd_rel)
            for j in init_jobs: self.all_job_due_dates[j.job_id] = j.meta["due_date"]
            self.orch.buffer.extend(init_jobs); self.gen.bump_next_id(max((j.job_id for j in init_jobs)) + 1); self.orch.event_release_and_reschedule(0.0); self.release_count += 1
        
        # [NEW] Initialize shadow simulator as a clone of actual
        self.shadow_orch = self.orch.clone()
        return self._observe(), {"t_now": 0.0}

    def step(self, action: int):
        t_event = float(self.t_now)
        new_jobs = self.gen.generate_burst(t_event)
        if new_jobs:
            for j in new_jobs: 
                self.all_job_due_dates[j.job_id] = j.meta["due_date"]
            self.orch.buffer.extend(new_jobs)
            self.shadow_orch.buffer.extend(copy.deepcopy(new_jobs)) # Shared arrival sequence
            
        t_next = float(self.gen.sample_next_time(t_event))
        dt = t_next - t_event
        
        # [SHADOW ALWAYS RELEASES]
        # In the shadow world, we always release whenever a burst arrives
        self.shadow_orch.event_release_and_reschedule(t_event)
        
        r_rel_component = 0.0
        if int(action) == 1:
            # [ACTION 1: DECISION POINT COMPARISON]
            self.orch.event_release_and_reschedule(t_event)
            self.release_count += 1
            
            actual_td = self.orch.get_total_tardiness_estimate()
            shadow_td = self.shadow_orch.get_total_tardiness_estimate()
            
            # Reward = -(Actual - Shadow) * Coef
            r_rel_component = -((actual_td - shadow_td) * float(configs.release_penalty_coef)) / self.time_scale
            
            # [SYNC] Reset shadow to current actual state for the next cycle
            self.shadow_orch = self.orch.clone()
        else:
            # [ACTION 0: WAIT]
            self.orch.tick_without_release(t_event)
            r_rel_component = 0.0 # No release reward during wait (Option A)

        # Common rewards (Idle, Buffer, Stability)
        met = self.orch.compute_interval_metrics(t_event, t_next); scale = self.time_scale
        r_idle = -(met["total_idle"] * float(configs.idle_penalty_coef)) / scale
        r_stab = -(float(action) * float(configs.stability_scale))
        r_buf = -(len(self.orch.buffer) * dt * float(configs.buffer_penalty_coef)) / self.time_scale
        
        # Release Gain Treatment (Conservative 50% for positive gains)
        if r_rel_component > 0: r_rel_component *= 0.5
        
        reward = r_idle + r_stab + r_buf + r_rel_component
        self.t_now = t_next; self.events_done += 1; done = bool(self.events_done >= self.event_horizon)
        
        r_flush = 0.0
        if done:
            while len(self.orch.buffer) > 0: self.orch.event_release_and_reschedule(self.t_now); self.release_count += 1
            final = self.orch.get_final_kpi_stats(self.all_job_due_dates)
            self.episode_tardiness = final["tardiness"]
            r_flush = (-final["makespan"] / scale * float(configs.flush_penalty_coef))
            reward += r_flush

        info = {"episode_tardiness": self.episode_tardiness, "release_count": self.release_count, "reward_idle_cost": r_idle, "reward_buffer_penalty": r_buf, "reward_release_penalty": r_rel_component, "reward_stability_penalty": r_stab, "reward_final_flush_penalty": r_flush}
        return self._observe(), float(reward), done, False, info
