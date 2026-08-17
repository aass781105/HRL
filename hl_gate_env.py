import math
import copy
import json
import os
import numpy as np
from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import torch
from params import configs
from data_utils import SD2_instance_generator
from dynamic_job_stream import register_initial_jobs, sample_initial_jobs
from hrl_orchestrator import GlobalTimelineOrchestrator, EventBurstGenerator
from ll_fjsp_env import LLFJSPEnv
from model.ll_dan_model import LLMLPNet
from model.hl_gate_state import HL_LL_BUFFER_EMBED_DIM, calculate_hl_gate_state, get_hl_gate_state_dim
from hl_env_scenarios import make_burst_sampler, resolve_hl_env_scenario, scenario_config
from common_utils import resolve_lower_level_weight_path, setup_seed


_LL_ENCODER_MODEL = None
_LL_ENCODER_LOAD_FAILED = False


def _zero_ll_buffer_embedding(config=configs) -> np.ndarray:
    dim = int(getattr(config, "hl_ll_buffer_embedding_dim", HL_LL_BUFFER_EMBED_DIM))
    return np.zeros(dim, dtype=np.float32)


def _get_global_ll_encoder_model(config=configs):
    global _LL_ENCODER_MODEL, _LL_ENCODER_LOAD_FAILED
    if _LL_ENCODER_MODEL is not None:
        return _LL_ENCODER_MODEL
    if _LL_ENCODER_LOAD_FAILED:
        return None

    model_path = resolve_lower_level_weight_path(str(getattr(config, "ll_ppo_model_path", "") or ""), getattr(config, "data_source", "SD2"))
    if not model_path or not os.path.exists(model_path):
        _LL_ENCODER_LOAD_FAILED = True
        return None

    try:
        device = torch.device(getattr(config, "device", "cpu"))
        model = LLMLPNet(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LL_ENCODER_MODEL = model
        return _LL_ENCODER_MODEL
    except Exception as exc:
        _LL_ENCODER_LOAD_FAILED = True
        if bool(getattr(config, "debug_hl_ll_buffer_embedding", False)):
            print(f"[HL-LL-EMB-WARN] failed to load low-level encoder: {exc}")
        return None


def compute_hl_ll_buffer_embedding(orch, t_now: float, n_machines: int, config=configs) -> np.ndarray:
    if not bool(getattr(config, "hl_use_ll_buffer_embedding", False)):
        return np.zeros(0, dtype=np.float32)
    if not orch or not getattr(orch, "buffer", None):
        return _zero_ll_buffer_embedding(config)

    model = _get_global_ll_encoder_model(config)
    if model is None:
        return _zero_ll_buffer_embedding(config)

    target_dim = int(getattr(config, "hl_ll_buffer_embedding_dim", HL_LL_BUFFER_EMBED_DIM))
    try:
        jobs = list(orch.buffer)
        jl_list, pt_list = orch._build_batch(jobs)
        due_rel = [float(job.meta.get("due_date", t_now)) - float(t_now) for job in jobs]
        release_rel = [0.0 for _ in jobs]
        env = LLFJSPEnv(n_j=len(jobs), n_m=int(n_machines))
        state = env.set_initial_data(
            jl_list,
            pt_list,
            due_date_list=[due_rel],
            true_due_date_list=[due_rel],
            release_time_list=[release_rel],
        )
        device = torch.device(getattr(config, "device", "cpu"))
        with torch.no_grad():
            _, _, fea_j_global, fea_m_global = model.feature_exact(
                state.fea_j_tensor.to(device),
                state.op_mask_tensor.to(device),
                state.candidate_tensor.to(device),
                state.fea_m_tensor.to(device),
                state.mch_mask_tensor.to(device),
                state.comp_idx_tensor.to(device),
                state.dynamic_pair_mask_tensor.to(device),
                state.fea_pairs_tensor.to(device),
            )
            emb = torch.cat((fea_j_global[0], fea_m_global[0]), dim=-1).detach().float().cpu().numpy()
        if emb.size < target_dim:
            emb = np.pad(emb, (0, target_dim - emb.size), mode="constant")
        elif emb.size > target_dim:
            emb = emb[:target_dim]
        return emb.astype(np.float32)
    except Exception as exc:
        if bool(getattr(config, "debug_hl_ll_buffer_embedding", False)):
            print(f"[HL-LL-EMB-WARN] failed to compute buffer embedding: {exc}")
        return _zero_ll_buffer_embedding(config)


def _baseline_cache_key(
    instance_seed: int,
    cadence: int,
    n_machines: int,
    init_jobs: int,
    event_horizon: int,
    interarrival_mean: float,
    burst_k: int,
    hl_env_scenario: str,
    hl_burst_size_mode: str,
    hl_burst_size_low: int,
    hl_burst_size_high: int,
    hl_bottleneck_order_prob: float,
    hl_bottleneck_order_machine_count: int,
    hl_bottleneck_exclude_urgent: bool,
    hl_bottleneck_group_sampling: str,
    hl_bottleneck_avoid_prev_machines: bool,
    ll_eval_action_selection: str,
    arrival_mode: str,
    interarrival_uniform_low: float,
    interarrival_uniform_high: float,
    due_date_tightness: float,
    due_date_mode: str,
    due_date_k_low: float,
    due_date_k_high: float,
    urgent_prob: float,
    urgent_k_low: float,
    urgent_k_high: float,
    normal_k_low: float,
    normal_k_high: float,
    pt_low: float,
    pt_high: float,
    op_per_job: float,
) -> str:
    return "|".join(
        [
            "baseline-v4",
            str(int(instance_seed)),
            str(int(cadence)),
            str(int(n_machines)),
            str(int(init_jobs)),
            str(int(event_horizon)),
            f"{float(interarrival_mean):.12g}",
            str(int(burst_k)),
            str(hl_env_scenario),
            str(hl_burst_size_mode),
            str(int(hl_burst_size_low)),
            str(int(hl_burst_size_high)),
            f"{float(hl_bottleneck_order_prob):.12g}",
            str(int(hl_bottleneck_order_machine_count)),
            str(bool(hl_bottleneck_exclude_urgent)),
            str(hl_bottleneck_group_sampling),
            str(bool(hl_bottleneck_avoid_prev_machines)),
            str(ll_eval_action_selection),
            str(arrival_mode),
            f"{float(interarrival_uniform_low):.12g}",
            f"{float(interarrival_uniform_high):.12g}",
            f"{float(due_date_tightness):.12g}",
            str(due_date_mode),
            f"{float(due_date_k_low):.12g}",
            f"{float(due_date_k_high):.12g}",
            f"{float(urgent_prob):.12g}",
            f"{float(urgent_k_low):.12g}",
            f"{float(urgent_k_high):.12g}",
            f"{float(normal_k_low):.12g}",
            f"{float(normal_k_high):.12g}",
            f"{float(pt_low):.12g}",
            f"{float(pt_high):.12g}",
            f"{float(op_per_job):.12g}",
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
    setup_seed(int(instance_seed))
    setattr(configs, "ll_eval_action_selection", "greedy")
    rng = np.random.default_rng(int(instance_seed))
    scenario = resolve_hl_env_scenario(configs, rng)
    base_cfg = scenario_config(configs, scenario)
    gen = EventBurstGenerator(
        SD2_instance_generator,
        copy.deepcopy(base_cfg),
        int(n_machines),
        float(getattr(base_cfg, "interarrival_mean", interarrival_mean)),
        make_burst_sampler(base_cfg),
        rng,
    )
    orch = GlobalTimelineOrchestrator(int(n_machines), gen, t0=0.0)
    all_job_due_dates: Dict[int, float] = {}
    release_count = 0
    t_now = 0.0

    if int(init_jobs) > 0:
        init_cfg = copy.deepcopy(base_cfg)
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


class HLGateEnv(gym.Env):
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
        setattr(configs, "ll_eval_action_selection", "greedy")
        self.baseline_cadence = int(getattr(configs, "baseline_cadence", 1))
        self.baseline_event_cadence = self._effective_baseline_event_cadence(self.baseline_cadence)
        self._last_release_event_idx = 0
        self._last_release_td = 0.0
        self._prev_arrival_time = 0.0
        self.instance_seed = 0
        self.steps_since_last_release = 0
        self.use_ll_buffer_embedding = bool(getattr(configs, "hl_use_ll_buffer_embedding", False))
        self.ll_buffer_embedding_dim = int(getattr(configs, "hl_ll_buffer_embedding_dim", HL_LL_BUFFER_EMBED_DIM))
        self._ll_encoder_model = None
        self._ll_encoder_load_failed = False
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(get_hl_gate_state_dim(configs),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    @staticmethod
    def _compute_total_stability_penalty(agent_release_count: int) -> float:
        stability_scale = float(getattr(configs, "hl_stability_scale", 0.0))
        if abs(stability_scale) <= 1e-12:
            return 0.0
        mode = HLGateEnv._resolve_stability_mode()
        if mode == "off":
            return 0.0
        if mode in ("immediate_all", "immediate_all_terminal"):
            return float(-stability_scale * max(0, int(agent_release_count)))
        free_releases = max(0, int(getattr(configs, "hl_stability_free_releases", 0)))
        excess_releases = max(0, int(agent_release_count) - free_releases)
        stability_power = float(getattr(configs, "hl_stability_power", 0.0))
        if stability_power > 0.0:
            return float(-stability_scale * (float(excess_releases) ** stability_power))
        return float(-stability_scale * (excess_releases * (excess_releases + 1) / 2.0))

    @staticmethod
    def _resolve_td_signal_source() -> str:
        explicit = str(getattr(configs, "hl_td_signal_source", "")).strip().lower()
        if explicit:
            return explicit
        shaping_reward_coef = float(getattr(configs, "hl_shaping_reward_coef", 0.0))
        td_reward_coef = float(getattr(configs, "hl_td_reward_coef", 0.0))
        if abs(shaping_reward_coef) > 1e-12:
            return "baseline_gap_release_interval"
        if abs(td_reward_coef) > 1e-12:
            return "baseline_gap_final"
        return "none"

    @staticmethod
    def _resolve_td_credit_mode() -> str:
        explicit = str(getattr(configs, "hl_td_credit_mode", "")).strip().lower()
        if explicit:
            return explicit
        shaping_reward_coef = float(getattr(configs, "hl_shaping_reward_coef", 0.0))
        td_reward_coef = float(getattr(configs, "hl_td_reward_coef", 0.0))
        if abs(shaping_reward_coef) > 1e-12:
            return "redistribute" if bool(getattr(configs, "hl_release_reward_redistribute", False)) else "step_only"
        if abs(td_reward_coef) > 1e-12:
            return "terminal_only"
        return "step_only"

    @staticmethod
    def _resolve_stability_mode() -> str:
        explicit = str(getattr(configs, "hl_stability_mode_v2", "")).strip().lower()
        if explicit:
            return explicit
        legacy = str(getattr(configs, "hl_stability_mode", "immediate_all")).strip().lower()
        if abs(float(getattr(configs, "hl_stability_scale", 0.0))) <= 1e-12:
            return "off"
        if legacy == "immediate_all":
            return "immediate_all"
        if bool(getattr(configs, "hl_stability_terminal_only", False)):
            return "free_threshold_terminal"
        return "free_threshold_distributed"

    @staticmethod
    def _resolve_td_step_coef() -> float:
        shaping_reward_coef = float(getattr(configs, "hl_shaping_reward_coef", 0.0))
        if abs(shaping_reward_coef) > 1e-12:
            return shaping_reward_coef
        return float(getattr(configs, "hl_td_reward_coef", 0.0))

    @staticmethod
    def _resolve_td_terminal_coef() -> float:
        td_reward_coef = float(getattr(configs, "hl_td_reward_coef", 0.0))
        if abs(td_reward_coef) > 1e-12:
            return td_reward_coef
        return float(getattr(configs, "hl_shaping_reward_coef", 0.0))

    @staticmethod
    def _compress_rel_tail(raw_value: float, threshold: float = 2.0, tail_scale: float = 1.0) -> float:
        abs_value = abs(float(raw_value))
        if abs_value <= threshold:
            return float(raw_value)
        tail = np.log1p((abs_value - threshold) / tail_scale)
        return float(np.sign(raw_value) * (threshold + tail))

    def _is_decision_event(self, event_idx: int) -> bool:
        K = max(1, int(getattr(configs, "hl_gate_decision_interval", 1)))
        horizon = int(getattr(self, "event_horizon", getattr(configs, "event_horizon", 1)))
        event_idx = int(event_idx)
        return bool(event_idx >= horizon or event_idx % K == 0)

    @staticmethod
    def _effective_baseline_event_cadence(decision_cadence: int) -> int:
        decision_interval = max(1, int(getattr(configs, "hl_gate_decision_interval", 1)))
        return max(1, int(decision_cadence)) * decision_interval

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
        setup_seed(int(instance_seed))
        setattr(configs, "ll_eval_action_selection", "greedy")
        rng = np.random.default_rng(int(instance_seed))
        scenario = resolve_hl_env_scenario(configs, rng)
        base_cfg = scenario_config(configs, scenario)
        gen = EventBurstGenerator(
            SD2_instance_generator,
            copy.deepcopy(base_cfg),
            self.M,
            float(getattr(base_cfg, "interarrival_mean", self.interarrival_mean)),
            make_burst_sampler(base_cfg),
            rng,
        )
        orch = GlobalTimelineOrchestrator(self.M, gen, t0=0.0)
        all_job_due_dates = {}
        release_count = 0
        t_now = 0.0
        if self.init_jobs > 0:
            init_cfg = copy.deepcopy(base_cfg)
            setattr(init_cfg, "init_jobs", self.init_jobs)
            init_jobs = sample_initial_jobs(init_cfg, rng=rng, base_job_id=0, t_arrive=0.0)
            release_count += register_initial_jobs(orch, gen, init_jobs, all_job_due_dates, t0=0.0)
        t_next = float(gen.sample_next_time(t_now))
        t_now, t_next = self._advance_sim_to_next_arrival(gen, orch, all_job_due_dates, t_next)
        return rng, gen, orch, all_job_due_dates, t_now, t_next, release_count

    def _run_cadence_baseline(self, instance_seed: int, cadence: Optional[int] = None) -> Dict[str, object]:
        decision_cadence = max(1, int(self.baseline_cadence if cadence is None else cadence))
        event_cadence = self._effective_baseline_event_cadence(decision_cadence)
        cache_key = _baseline_cache_key(
            int(instance_seed),
            int(event_cadence),
            int(self.M),
            int(self.init_jobs),
            int(self.event_horizon),
            float(self.interarrival_mean),
            int(self.burst_K),
            str(getattr(configs, "hl_env_scenario", "baseline")),
            str(getattr(configs, "hl_burst_size_mode", "fixed")),
            int(getattr(configs, "hl_burst_size_low", 1)),
            int(getattr(configs, "hl_burst_size_high", 5)),
            float(getattr(configs, "hl_bottleneck_order_prob", 0.0)),
            int(getattr(configs, "hl_bottleneck_order_machine_count", 1)),
            bool(getattr(configs, "hl_bottleneck_exclude_urgent", False)),
            str(getattr(configs, "hl_bottleneck_group_sampling", "random")),
            bool(getattr(configs, "hl_bottleneck_avoid_prev_machines", False)),
            str(getattr(configs, "ll_eval_action_selection", "greedy")),
            str(getattr(configs, "arrival_mode", "exponential")),
            float(getattr(configs, "interarrival_uniform_low", 10.0)),
            float(getattr(configs, "interarrival_uniform_high", 50.0)),
            float(getattr(configs, "hl_due_date_tightness", 1.2)),
            str(getattr(configs, "ll_due_date_mode", "k")),
            float(getattr(configs, "due_date_k_low", 1.2)),
            float(getattr(configs, "due_date_k_high", 2.0)),
            float(getattr(configs, "hl_due_date_urgent_prob", 0.0)),
            float(getattr(configs, "hl_due_date_k_urgent_low", 1.2)),
            float(getattr(configs, "hl_due_date_k_urgent_high", 2.0)),
            float(getattr(configs, "hl_due_date_k_normal_low", 1.2)),
            float(getattr(configs, "hl_due_date_k_normal_high", 2.0)),
            float(getattr(configs, "low", 1.0)),
            float(getattr(configs, "high", 99.0)),
            float(getattr(configs, "op_per_job", 5.0)),
        )
        cache_path = str(os.environ.get("PPO_GATE_BASELINE_CACHE_PATH", "ppo_gate_baseline_cache.json")).strip()
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
            cadence=int(event_cadence),
        )
        result = {
            "td": float(baseline["td"]),
            "mk": float(baseline["mk"]),
            "release_count": int(baseline["release_count"]),
            "event_td": [float(x) for x in baseline.get("event_td", [])],
        }
        self._baseline_cache[cache_key] = result

        if cache_path:
            try:
                current_data = {}
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        current_data = json.load(f)
                for key, val in self._baseline_cache.items():
                    current_data[key] = val
                temp_path = cache_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(current_data, f, indent=2)
                if os.path.exists(temp_path):
                    try:
                        os.replace(temp_path, cache_path)
                    except OSError:
                        os.remove(temp_path)
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(current_data, f, indent=2)
            except Exception:
                pass
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
        decision_interval = max(1, int(getattr(configs, "hl_gate_decision_interval", 1)))
        decision_steps_elapsed = max(0, int(self.events_done) // decision_interval)
        obs = calculate_hl_gate_state(
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
            release_count_so_far=self.agent_release_count,
            decision_steps_elapsed=decision_steps_elapsed,
            is_last_step=is_last_step,
            buffer_jobs=self.orch.buffer,
        )
        if self.use_ll_buffer_embedding:
            obs = np.concatenate((obs, self._compute_ll_buffer_embedding()), axis=0).astype(np.float32)
        return obs

    def _get_ll_encoder_model(self):
        if self._ll_encoder_model is not None:
            return self._ll_encoder_model
        if self._ll_encoder_load_failed:
            return None

        model_path = resolve_lower_level_weight_path(str(getattr(configs, "ll_ppo_model_path", "") or ""), getattr(configs, "data_source", "SD2"))
        if not model_path or not os.path.exists(model_path):
            self._ll_encoder_load_failed = True
            return None

        try:
            device = torch.device(getattr(configs, "device", "cpu"))
            model = LLMLPNet(configs).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._ll_encoder_model = model
            return self._ll_encoder_model
        except Exception as exc:
            self._ll_encoder_load_failed = True
            if bool(getattr(configs, "debug_hl_ll_buffer_embedding", False)):
                print(f"[HL-LL-EMB-WARN] failed to load low-level encoder: {exc}")
            return None

    def _zero_ll_buffer_embedding(self) -> np.ndarray:
        return np.zeros(self.ll_buffer_embedding_dim, dtype=np.float32)

    def _compute_ll_buffer_embedding(self) -> np.ndarray:
        return compute_hl_ll_buffer_embedding(self.orch, self.t_now, self.M, configs)

    def _get_buffer_stats(self, t_now: float):
        if not self.orch.buffer:
            return {
                "buffer_neg_slack_ratio": 0.0,
                "min_slack": 0.0,
                "avg_slack": 0.0,
                "slack_std": 0.0,
                "slack_q25": 0.0,
                "neg_slack_sum": 0.0,
                "total_work": 0.0,
            }
        slacks, neg_count = [], 0
        neg_slack_sum, total_work = 0.0, 0.0
        for j in self.orch.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0))
            if mw <= 0.0:
                mw = float(sum(float(getattr(op, "avg_proc_time", 0.0)) for op in getattr(j, "operations", []) or []))
            due = self.all_job_due_dates[j.job_id]; s = due - t_now - mw
            total_work += mw
            slacks.append(s)
            if t_now + mw > due:
                neg_count += 1
                neg_slack_sum += float(-s)
        return {
            "buffer_neg_slack_ratio": neg_count / len(self.orch.buffer),
            "min_slack": min(slacks),
            "avg_slack": sum(slacks) / len(slacks),
            "slack_std": float(np.std(slacks)),
            "slack_q25": float(np.percentile(slacks, 25)),
            "neg_slack_sum": neg_slack_sum,
            "total_work": total_work,
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

        episodes_per_instance = max(1, int(getattr(configs, "hl_ppo_instance_episodes", 10)))
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
        setattr(configs, "ll_eval_action_selection", "greedy")
        self.baseline_cadence = int(getattr(configs, "baseline_cadence", 1))
        self.baseline_event_cadence = self._effective_baseline_event_cadence(self.baseline_cadence)
        td_signal_source = self._resolve_td_signal_source()
        needs_baseline = td_signal_source in ("baseline_gap_final", "baseline_gap_release_interval")
        if options.get("needs_baseline", True) is False:
            needs_baseline = False
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

        while not self._is_decision_event(self.events_done + 1):
            _, _, done = self._single_arrival_step(0)
            if done:
                break

        return self._observe(), {"t_now": self.t_now, "instance_seed": instance_seed}

    def step(self, action: int):
        K = int(getattr(configs, "hl_gate_decision_interval", 1))

        while not self._is_decision_event(self.events_done + 1):
            _, _, done = self._single_arrival_step(0)
            if done:
                return self._observe(), 0.0, done, False, {
                    "event_id": self.events_done,
                    "executed_action": 0,
                    "auto_hold_to_decision": True,
                }
        
        # 1. Execute the decision step
        reward, info, done = self._single_arrival_step(action)
        
        # Accumulators for skipped steps
        accumulated_reward = float(reward)
        accumulated_info = info.copy()
        
        # 2. Loop and auto-hold (action=0) for intermediate steps
        while not done:
            if self._is_decision_event(self.events_done + 1):
                break
                
            reward_auto, info_auto, done = self._single_arrival_step(0)
            accumulated_reward += float(reward_auto)
            
            # Accumulate info metrics
            for k in [
                "reward_buffer_penalty", "reward_shaping_penalty", 
                "reward_td_penalty", "reward_release_raw_penalty", 
                "reward_td_terminal_penalty", "reward_stability_penalty", 
                "reward_mk_penalty"
            ]:
                if k in accumulated_info and k in info_auto:
                    accumulated_info[k] = float(accumulated_info[k]) + float(info_auto[k])
            
            # For terminal state information, overwrite with latest
            for k in ["time", "inter_arrival", "event_id", "actual_td", "baseline_event_td"]:
                if k in info_auto:
                    accumulated_info[k] = info_auto[k]
                    
            if done:
                for k in ["episode_tardiness", "episode_makespan", "release_count", "agent_final_td", "reward_stability_total_penalty"]:
                    if k in info_auto:
                        accumulated_info[k] = info_auto[k]
                        
        return self._observe(), accumulated_reward, done, False, accumulated_info

    def _single_arrival_step(self, action: int):
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
        stability_scale = float(getattr(configs, "hl_stability_scale", 0.0))
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
        r_buf = -(total_neg_slack * float(getattr(configs, "hl_buffer_penalty_coef", 0.0))) / self.time_scale

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
            r_mk_raw = -(((mk_ratio + 1.0) ** 2) - 1.0) * float(getattr(configs, "hl_mk_reward_coef", 0.0))
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
            if r_shape > 0.0:
                r_shape = 0.0
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
            if r_shape > 0.0:
                r_shape = 0.0
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
            "baseline_event_cadence": int(getattr(self, "baseline_event_cadence", self._effective_baseline_event_cadence(self.baseline_cadence))),
            "td_gap_vs_baseline_cadence": float(td_gap),
        }
        
        return float(reward), info, done
