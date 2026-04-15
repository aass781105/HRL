import math
import copy
import json
import os
import numpy as np
from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
from params import configs
from data_utils import SD2_instance_generator
from dynamic_job_stream import register_initial_jobs, sample_initial_jobs
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator
from model.gate_state import calculate_gate_state


def _baseline_cache_key(
    instance_seed: int,
    cadence: int,
    n_machines: int,
    init_jobs: int,
    event_horizon: int,
    interarrival_mean: float,
    burst_k: int,
) -> str:
    return "|".join(
        [
            str(int(instance_seed)),
            str(int(cadence)),
            str(int(n_machines)),
            str(int(init_jobs)),
            str(int(event_horizon)),
            f"{float(interarrival_mean):.12g}",
            str(int(burst_k)),
        ]
    )


def compute_cadence_baseline_for_seed(
    instance_seed: int,
    *,
    n_machines: int,
    interarrival_mean: float,
    burst_k: int,
    event_horizon: int,
    init_jobs: int,
    cadence: int = 1,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(instance_seed))
    gen = EventBurstGenerator(
        SD2_instance_generator,
        copy.deepcopy(configs),
        int(n_machines),
        float(interarrival_mean),
        lambda _r: int(burst_k),
        rng,
    )
    orch = GlobalTimelineOrchestrator(int(n_machines), gen, t0=0.0)
    all_job_due_dates: Dict[int, float] = {}
    release_count = 0
    t_now = 0.0

    if int(init_jobs) > 0:
        init_cfg = copy.deepcopy(configs)
        setattr(init_cfg, "init_jobs", int(init_jobs))
        init_job_specs = sample_initial_jobs(init_cfg, rng=rng, base_job_id=0, t_arrive=0.0)
        release_count += register_initial_jobs(orch, gen, init_job_specs, all_job_due_dates, t0=0.0)

    t_next = float(gen.sample_next_time(t_now))
    t_now = float(t_next)
    new_jobs = gen.generate_burst(t_now)
    if new_jobs:
        for job in new_jobs:
            all_job_due_dates[job.job_id] = job.meta["due_date"]
        orch.buffer.extend(new_jobs)
    t_next = float(gen.sample_next_time(t_now))

    events_done = 1
    cadence = max(1, int(cadence))
    event_td = []
    while True:
        if events_done % cadence == 0:
            orch.event_release_and_reschedule(float(t_now))
            release_count += 1
        else:
            orch.tick_without_release(float(t_now))
        event_td.append(float(orch.get_total_tardiness_estimate(all_job_due_dates)))

        if events_done >= int(event_horizon):
            break

        t_now = float(t_next)
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            for job in new_jobs:
                all_job_due_dates[job.job_id] = job.meta["due_date"]
            orch.buffer.extend(new_jobs)
        t_next = float(gen.sample_next_time(t_now))
        events_done += 1

    while len(orch.buffer) > 0:
        orch.event_release_and_reschedule(t_next)
        release_count += 1

    final = orch.get_final_kpi_stats(all_job_due_dates)
    return {
        "td": float(final["tardiness"]),
        "mk": float(final["makespan"]),
        "release_count": int(release_count),
        "event_td": [float(x) for x in event_td],
    }


class EventGateEnv(gym.Env):
    _baseline_cache: Dict[str, Dict[str, object]] = {}
    _baseline_cache_loaded_paths = set()

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
        self.baseline_final_td = 0.0
        self.baseline_final_mk = 0.0
        self.baseline_release_count = 0
        self.baseline_event_td = []
        self.baseline_cadence = 1
        self._last_release_event_idx = 0
        self._last_release_td = 0.0
        self._prev_arrival_time = 0.0
        self.instance_seed = 0
        self.steps_since_last_release = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    @staticmethod
    def _compute_total_stability_penalty(agent_release_count: int) -> float:
        stability_scale = float(getattr(configs, "stability_scale", 0.0))
        if abs(stability_scale) <= 1e-12:
            return 0.0
        mode = EventGateEnv._resolve_stability_mode()
        if mode == "off":
            return 0.0
        if mode in ("immediate_all", "immediate_all_terminal"):
            return float(-stability_scale * max(0, int(agent_release_count)))
        free_releases = max(0, int(getattr(configs, "stability_free_releases", 0)))
        excess_releases = max(0, int(agent_release_count) - free_releases)
        return float(-stability_scale * (excess_releases * (excess_releases + 1) / 2.0))

    @staticmethod
    def _resolve_td_signal_source() -> str:
        explicit = str(getattr(configs, "td_signal_source", "")).strip().lower()
        if explicit:
            return explicit
        shaping_reward_coef = float(getattr(configs, "shaping_reward_coef", 0.0))
        td_reward_coef = float(getattr(configs, "td_reward_coef", 0.0))
        if abs(shaping_reward_coef) > 1e-12:
            return "baseline_gap_release_interval"
        if abs(td_reward_coef) > 1e-12:
            return "baseline_gap_final"
        return "none"

    @staticmethod
    def _resolve_td_credit_mode() -> str:
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

    @staticmethod
    def _resolve_stability_mode() -> str:
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

    @staticmethod
    def _resolve_td_step_coef() -> float:
        shaping_reward_coef = float(getattr(configs, "shaping_reward_coef", 0.0))
        if abs(shaping_reward_coef) > 1e-12:
            return shaping_reward_coef
        return float(getattr(configs, "td_reward_coef", 0.0))

    @staticmethod
    def _resolve_td_terminal_coef() -> float:
        td_reward_coef = float(getattr(configs, "td_reward_coef", 0.0))
        if abs(td_reward_coef) > 1e-12:
            return td_reward_coef
        return float(getattr(configs, "shaping_reward_coef", 0.0))

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
            init_cfg = copy.deepcopy(configs)
            setattr(init_cfg, "init_jobs", self.init_jobs)
            init_jobs = sample_initial_jobs(init_cfg, rng=rng, base_job_id=0, t_arrive=0.0)
            release_count += register_initial_jobs(orch, gen, init_jobs, all_job_due_dates, t0=0.0)
        t_next = float(gen.sample_next_time(t_now))
        t_now, t_next = self._advance_sim_to_next_arrival(gen, orch, all_job_due_dates, t_next)
        return rng, gen, orch, all_job_due_dates, t_now, t_next, release_count

    def _run_cadence_baseline(self, instance_seed: int, cadence: Optional[int] = None) -> Dict[str, object]:
        cadence = max(1, int(self.baseline_cadence if cadence is None else cadence))
        cache_key = _baseline_cache_key(
            int(instance_seed),
            int(cadence),
            int(self.M),
            int(self.init_jobs),
            int(self.event_horizon),
            float(self.interarrival_mean),
            int(self.burst_K),
        )
        cache_path = str(os.environ.get("PPO_GATE_BASELINE_CACHE_PATH", "")).strip()
        if cache_path and cache_path not in self._baseline_cache_loaded_paths and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                for key, value in payload.items():
                    self._baseline_cache[key] = {
                        "td": float(value["td"]),
                        "mk": float(value["mk"]),
                        "release_count": int(value["release_count"]),
                        "event_td": [float(x) for x in value.get("event_td", [])],
                    }
                self._baseline_cache_loaded_paths.add(cache_path)
            except Exception:
                pass
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached

        baseline = compute_cadence_baseline_for_seed(
            int(instance_seed),
            n_machines=int(self.M),
            interarrival_mean=float(self.interarrival_mean),
            burst_k=int(self.burst_K),
            event_horizon=int(self.event_horizon),
            init_jobs=int(self.init_jobs),
            cadence=int(cadence),
        )
        result = {
            "td": float(baseline["td"]),
            "mk": float(baseline["mk"]),
            "release_count": int(baseline["release_count"]),
            "event_td": [float(x) for x in baseline.get("event_td", [])],
        }
        self._baseline_cache[cache_key] = result
        return result

    def _observe(self) -> np.ndarray:
        t_now = self.t_now
        rem = np.maximum(0.0, self.orch.machine_free_time - t_now)
        mx_l = np.max(rem)
        w_idle, u_idle = self.orch.compute_idle_stats(t_now, mx_l)
        buf_stats = self._get_buffer_stats(t_now)
        wip_stats = self.orch.get_wip_stats(t_now)
        inter_arrival_scaled = float((t_now - self._prev_arrival_time) / self.time_scale) if self.time_scale > 0 else 0.0
        is_last_step = bool((self.events_done + 1) >= self.event_horizon)
        return calculate_gate_state(
            len(self.orch.buffer),
            self.orch.machine_free_time,
            t_now,
            self.M,
            0,
            self.time_scale,
            w_idle,
            u_idle,
            buf_stats,
            wip_stats,
            inter_arrival_scaled=inter_arrival_scaled,
            steps_since_last_release=self.steps_since_last_release,
            is_last_step=is_last_step,
        )

    def _get_buffer_stats(self, t_now: float):
        if not self.orch.buffer: return {"buffer_neg_slack_ratio": 0.0, "min_slack": 0.0, "avg_slack": 0.0, "slack_std": 0.0, "slack_q25": 0.0}
        slacks, neg_count = [], 0
        for j in self.orch.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0)); due = self.all_job_due_dates[j.job_id]; s = due - t_now - mw
            slacks.append(s); 
            if t_now + mw > due: neg_count += 1
        return {
            "buffer_neg_slack_ratio": neg_count / len(self.orch.buffer),
            "min_slack": min(slacks),
            "avg_slack": sum(slacks) / len(slacks),
            "slack_std": float(np.std(slacks)),
            "slack_q25": float(np.percentile(slacks, 25)),
        }

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
        self.instance_seed = int(instance_seed)
        
        # Call super().reset for Gymnasium compliance (manages self.np_random)
        super().reset(seed=instance_seed)
        
        _, self.gen, self.orch, self.all_job_due_dates, self.t_now, self.t_next, self.release_count = self._build_simulation(instance_seed)
        self.episode_tardiness, self.events_done = 0.0, 0
        self.agent_release_count = 0
        self.baseline_cadence = 1
        td_signal_source = self._resolve_td_signal_source()
        needs_baseline = td_signal_source in ("baseline_gap_final", "baseline_gap_release_interval")
        if needs_baseline:
            baseline = self._run_cadence_baseline(instance_seed)
            self.baseline_final_td = float(baseline["td"])
            self.baseline_final_mk = float(baseline["mk"])
            self.baseline_release_count = int(baseline["release_count"])
            self.baseline_event_td = [float(x) for x in baseline.get("event_td", [])]
        else:
            self.baseline_final_td = 0.0
            self.baseline_final_mk = 0.0
            self.baseline_release_count = 0
            self.baseline_event_td = []
        self._last_release_event_idx = 0
        self._last_release_td = 0.0
        self._prev_arrival_time = 0.0
        self.steps_since_last_release = 0

        return self._observe(), {"t_now": self.t_now, "instance_seed": instance_seed}

    def step(self, action: int):
        t_event = float(self.t_now)
        t_next = float(self.t_next)
        current_event_idx = int(self.events_done + 1)
        inter_arrival = float(t_event - self._prev_arrival_time)
        forced_last_release = bool(current_event_idx >= self.event_horizon)
        executed_action = 1 if forced_last_release else int(action)
        td_signal_source = self._resolve_td_signal_source()
        td_credit_mode = self._resolve_td_credit_mode()
        td_step_coef = self._resolve_td_step_coef()
        td_terminal_coef = self._resolve_td_terminal_coef()

        if int(executed_action) == 1:
            self.orch.event_release_and_reschedule(t_event)
            self.release_count += 1
            self.agent_release_count += 1
            self.steps_since_last_release = 0
        else:
            self.orch.tick_without_release(t_event)
            self.steps_since_last_release += 1
        actual_td_now = float(self.orch.get_total_tardiness_estimate(self.all_job_due_dates))

        # Common rewards (Buffer, Stability)
        scale = self.time_scale
        stability_mode = self._resolve_stability_mode()
        stability_scale = float(configs.stability_scale)
        r_stab = 0.0
        if stability_mode == "immediate_all":
            r_stab = -stability_scale if int(executed_action) == 1 else 0.0
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
        r_stab_total = 0.0
        phi_before = 0.0
        phi_after = 0.0
        td_gap = 0.0
        baseline_step_td = float(self.baseline_event_td[current_event_idx - 1]) if current_event_idx - 1 < len(self.baseline_event_td) else 0.0
        prev_release_event_idx = int(self._last_release_event_idx)
        prev_agent_release_td = float(self._last_release_td)
        prev_baseline_td = float(self.baseline_event_td[prev_release_event_idx - 1]) if prev_release_event_idx > 0 and (prev_release_event_idx - 1) < len(self.baseline_event_td) else 0.0
        agent_td_delta = 0.0
        baseline_td_delta = 0.0
        if done:
            while len(self.orch.buffer) > 0:
                self.orch.event_release_and_reschedule(t_next)
                self.release_count += 1
            
            final = self.orch.get_final_kpi_stats(self.all_job_due_dates)
            self.episode_tardiness = final["tardiness"]
            ep_mk = final["makespan"]
            terminal_scale = max(scale, 1e-8)
            td_gap = float(self.episode_tardiness - self.baseline_final_td)
            if td_credit_mode == "terminal_only":
                if td_signal_source == "baseline_gap_final":
                    r_td = float((-(td_gap) / terminal_scale) * td_terminal_coef)
                elif td_signal_source == "agent_only":
                    r_td = float((-(self.episode_tardiness) / terminal_scale) * td_terminal_coef)
            
            mk_norm = max(scale * float(self.event_horizon), scale)
            mk_ratio = (ep_mk / mk_norm)
            r_mk_raw = -(((mk_ratio + 1.0) ** 2) - 1.0) * float(configs.mk_reward_coef)
            r_mk = float(r_mk_raw)
            if stability_mode in ("immediate_all_terminal", "free_threshold_terminal", "free_threshold_distributed"):
                r_stab_total = self._compute_total_stability_penalty(self.agent_release_count)
                if stability_mode in ("immediate_all_terminal", "free_threshold_terminal"):
                    r_stab = float(r_stab_total)
        else:
            self._advance_to_next_arrival()
        self._prev_arrival_time = t_event

        r_shape = 0.0
        if int(executed_action) == 1 and td_credit_mode in ("step_only", "redistribute") and abs(td_step_coef) > 1e-12:
            agent_td_delta = float(actual_td_now - self._last_release_td)
            baseline_td_delta = float(baseline_step_td - prev_baseline_td)
            if td_signal_source == "agent_only":
                td_signal_value = float(agent_td_delta)
            elif td_signal_source == "baseline_gap_release_interval":
                td_signal_value = float(agent_td_delta - baseline_td_delta)
            else:
                td_signal_value = 0.0
            r_shape = float((-(td_signal_value) / scale) * td_step_coef)
            self._last_release_event_idx = int(current_event_idx)
            self._last_release_td = float(actual_td_now)
            if bool(getattr(configs, "debug_reward_trace", False)) and r_shape > float(getattr(configs, "debug_reward_positive_threshold", 0.5)):
                print(
                    "[REWARD+]"
                    f" seed={self.instance_seed}"
                    f" event={current_event_idx}"
                    f" action=RELEASE"
                    f" prev_rel={prev_release_event_idx}"
                    f" agent_prev_td={prev_agent_release_td:.2f}"
                    f" agent_now_td={actual_td_now:.2f}"
                    f" agent_delta={agent_td_delta:.2f}"
                    f" base_prev_td={prev_baseline_td:.2f}"
                    f" base_now_td={baseline_step_td:.2f}"
                    f" base_delta={baseline_td_delta:.2f}"
                    f" r_shape={r_shape:.4f}"
                )
        elif done and td_signal_source == "agent_only" and td_credit_mode in ("step_only", "redistribute") and abs(td_step_coef) > 1e-12:
            agent_td_delta = float(self.episode_tardiness - self._last_release_td)
            r_shape = float((-(agent_td_delta) / scale) * td_step_coef)
        reward = r_stab + r_buf + r_shape + r_td + r_mk

        info = {
            "event_id": current_event_idx,
            "time": t_event,
            "inter_arrival": inter_arrival,
            "forced_last_release": bool(forced_last_release),
            "requested_action": int(action),
            "executed_action": int(executed_action),
            "actual_td": actual_td_now,
            "baseline_td": float(self.baseline_final_td),
            "baseline_event_td": float(baseline_step_td),
            "episode_tardiness": self.episode_tardiness, 
            "episode_makespan": ep_mk,
            "release_count": self.release_count, 
            "reward_buffer_penalty": r_buf, 
            "reward_shaping_penalty": r_shape,
            "reward_td_penalty": r_td,
            "reward_release_raw_penalty": r_shape,
            "reward_td_terminal_penalty": r_td,
            "reward_stability_penalty": r_stab, 
            "reward_stability_total_penalty": float(r_stab_total),
            "reward_mk_penalty": r_mk,
            "phi_before": float(phi_before),
            "phi_after": float(phi_after),
            "agent_prev_release_event_id": int(prev_release_event_idx),
            "agent_last_release_td": float(prev_agent_release_td),
            "agent_td_delta": float(agent_td_delta),
            "agent_final_td": float(self.episode_tardiness if done else 0.0),
            "baseline_prev_release_event_id": int(prev_release_event_idx),
            "baseline_last_release_td": float(prev_baseline_td),
            "baseline_td_delta": float(baseline_td_delta),
            "baseline_final_td": float(self.baseline_final_td),
            "baseline_final_mk": float(self.baseline_final_mk),
            "baseline_release_count": int(self.baseline_release_count),
            "baseline_cadence": int(self.baseline_cadence),
            "td_gap_vs_baseline_cadence": float(td_gap),
        }
        
        return self._observe(), float(reward), done, False, info
