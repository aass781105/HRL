# event_gate_env.py
import math
import copy
import numpy as np
from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
from params import configs
from data_utils import SD2_instance_generator
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator, split_matrix_to_jobs


class EventGateEnv(gym.Env):
    """
    事件級 RL 環境：
      - 一步 = 一次『到達事件時間』 t_now
      - action: 0=HOLD, 1=RELEASE
      - reward:
          original           : - total_idle*0.5 - delta_mk *0.5 / scale
          stability          : original - stability_penalty
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 n_machines: int,
                 heuristic: str = "SPT",
                 interarrival_mean: float = 0.1,
                 burst_K: int = 10,
                 event_horizon: int = 200,
                 init_jobs: int = 0,
                 obs_buffer_cap: Optional[int] = None,
                 time_scale: Optional[float] = None,
                 reward_mode: str = "pure"):
        super().__init__()
        self.M = int(n_machines)
        self.heuristic = heuristic
        self.interarrival_mean = float(interarrival_mean)
        self.burst_K = int(burst_K)
        self.event_horizon = int(event_horizon)
        self.init_jobs = int(init_jobs)
        self.reward_mode = str(reward_mode)

        # 正規化尺度
        self.obs_buffer_cap = obs_buffer_cap
        # 用 norm_scale 做時間正規化
        self.time_scale = float(getattr(configs, "norm_scale", 100.0))

        # 生成器與 orchestrator（reset 裡初始化）
        self.gen: Optional[EventBurstGenerator] = None
        self.orch: Optional[GlobalTimelineOrchestrator] = None

        # 內部時鐘
        self.t_now = 0.0
        self.t_next = None
        self.events_done = 0

        self.enable_final_flush_penalty = bool(getattr(configs, "enable_final_flush_penalty", True))

        # 上一輪 makespan
        self._mk_prev = 0.0

        # Gym space（狀態 3 維；動作 {0,1}）
        # state = [ n_buffer_norm, S_norm, L_min_norm ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    # --------------------------- 工具：正規化 ---------------------------
    def _norm_buffer(self, size: int) -> float:
        cap = self.obs_buffer_cap
        if cap is None or cap <= 0:
            cap = max(1, self.burst_K * 3)
        return float(size) / float(cap)

    def _norm_time(self, x: float) -> float:
        scale = self.time_scale
        return float(x) / float(scale)

    # --------------------------- 狀態抽取 ---------------------------
    def _observe(self) -> np.ndarray:
        # 1) buffer 大小
        buf_size = len(self.orch.buffer)

        # 2) 機台剩餘忙碌時間 rem_k = max(machine_free_time_k - t_now, 0)
        mft_abs = np.asarray(self.orch.machine_free_time, dtype=float)
        rem = np.maximum(0.0, mft_abs - float(self.t_now))  # [M]

        if rem.size > 0:
            total_rem = float(rem.sum()) / self.M   # 平均剩餘負載
            first_idle_rem = float(rem.min())       # 離第一台 idle 還有多久
        else:
            total_rem = 0.0
            first_idle_rem = 0.0

        o0 = self._norm_buffer(buf_size)
        o1 = self._norm_time(total_rem)
        o2 = self._norm_time(first_idle_rem)

        return np.array([o0, o1, o2], dtype=np.float32)

    # --------------------------- 工具：目前 makespan ---------------------------
    def _current_makespan(self) -> float:
        """
        優先用 orchestrator 的 row 資訊拿 max(end)，
        找不到再退回 machine_free_time。
        """
        assert self.orch is not None

        # 1) 優先用 _metric_rows（若有維護）
        if hasattr(self.orch, "_metric_rows") and self.orch._metric_rows:
            return max(float(r["end"]) for r in self.orch._metric_rows.values())

        # 2) 再看 _last_full_rows（每次完整重排後的排程）
        last_rows = getattr(self.orch, "_last_full_rows", None)
        if last_rows:
            return max(float(r["end"]) for r in last_rows)

        # 3) 最後才用 machine_free_time 當 fallback
        mft = getattr(self.orch, "machine_free_time", np.array([0.0]))
        return float(np.max(mft))

    # --------------------------- reset ---------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = getattr(configs, "event_seed", 42)
        rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
        base_cfg = copy.deepcopy(configs)

        self.gen = EventBurstGenerator(
            sd2_fn=SD2_instance_generator,
            base_config=base_cfg,
            n_machines=self.M,
            interarrival_mean=self.interarrival_mean,
            k_sampler=lambda _rng: int(self.burst_K),
            rng=rng,
        )
        self.orch = GlobalTimelineOrchestrator(
            n_machines=self.M,
            job_generator=self.gen,
            select_from_buffer=lambda buf, o: list(range(len(buf))),
            release_decider=None,
            t0=0.0,
        )

        self.t_now = 0.0
        self.events_done = 0

        # 初始注入
        if self.init_jobs > 0:
            tmp_cfg = copy.deepcopy(configs)
            old_nj = getattr(tmp_cfg, "n_j", None)
            try:
                setattr(tmp_cfg, "n_j", int(self.init_jobs))
                jl, pt, _ = SD2_instance_generator(tmp_cfg)
            finally:
                if old_nj is not None:
                    setattr(tmp_cfg, "n_j", old_nj)
            init_jobs = split_matrix_to_jobs(jl, pt, base_job_id=0, t_arrive=0.0)
            self.orch.buffer.extend(init_jobs)
            max_existing = max((j.job_id for j in self.orch.buffer), default=-1)
            self.gen.bump_next_id(max_existing + 1)

            # 預設先排一次
            self.orch.event_release_and_reschedule(self.t_now, self.heuristic)

        # 初始化上一輪 makespan
        try:
            self._mk_prev = self._current_makespan()
        except Exception:
            self._mk_prev = 0.0

        # 第一個到達事件
        self.t_next = self.gen.sample_next_time(self.t_now)

        obs = self._observe()
        info = {"t_now": self.t_now, "t_next": self.t_next}
        return obs, info

    # --------------------------- step ---------------------------
    def step(self, action: int):
        assert self.orch is not None and self.gen is not None, "env not reset"

        t_event = float(self.t_now)

        # (1) 到達：生成新工單 → buffer
        new_jobs = self.gen.generate_burst(t_event)
        if new_jobs:
            self.orch.buffer.extend(new_jobs)

        # (2) HOLD / RELEASE
        if int(action) == 1:
            # RELEASE：做一次完整重排
            self.orch.event_release_and_reschedule(t_event, self.heuristic)
        else:
            # HOLD：不放行，只更新 H 與機台占用
            self.orch.tick_without_release(t_event)

        # (3) 下一個事件時間
        t_new = float(self.gen.sample_next_time(t_event))

        # 用目前完整計畫計算 [t_event, t_new) 的區間指標
        met = self.orch.compute_interval_metrics(t_event, t_new)
        time_avg = float(met.get("time_avg", 0.0))
        idle_ratio = float(met.get("idle_ratio", 0.0))
        dt = float(met.get("interval_dt", 0.0))
        total_idle = float(met.get("total_idle", 0.0))
        if dt < 1.0:
            dt = 1.0

        # 目前 makespan
        try:
            mk_now = self._current_makespan()
        except Exception:
            mk_now = self._mk_prev

        # reward_scale：跟訓練時一致
        scale = float(getattr(configs, "reward_scale", 10.0))
        stability_scale = float(getattr(configs, "stability_scale", 5))
        alpha = float(getattr(configs, "reward_alpha", 0.3))

        # (4) reward （依 mode）
        stability_penalty = 0.0
        delta_mk = (mk_now - self._mk_prev)
        
        # Base reward: - [(1-alpha)*idle + alpha*mk_delta] / scale
        base_reward = - ((total_idle * (1.0 - alpha)) + (delta_mk * alpha)) / scale
        
        if self.reward_mode == "original":
            reward = base_reward
        elif self.reward_mode == "stability":
            stability_penalty = int(action) * stability_scale
            reward = base_reward - stability_penalty
        else:
            reward = base_reward # fallback

        # (5) 前進時鐘 & episode 計數
        self.t_now = t_new
        self.events_done += 1

        done = (self.events_done >= self.event_horizon)
        trunc = False
        temp = self._mk_prev
        # 更新上一輪 makespan（給下一步算 ΔMk 用）
        self._mk_prev = mk_now

        # （6）若終局且 buffer 還有工單 → 強制 flush，並把成本加到最後一步 reward
        final_flush_penalty = 0.0
        # delta_mk_flush = 0.0 # Unused in info now
        # idle_flush = 0.0     # Unused in info now
        
        if done and self.enable_final_flush_penalty:
            if len(self.orch.buffer) > 0:
                mk_final, idle_flush = self._run_final_flush_and_get_cost()
            else:
                mk_final = self._current_makespan()

            final_flush_penalty = -mk_final / scale 
            reward += final_flush_penalty

        # 新觀測 & info
        obs = self._observe()
        
        # [CHANGED] Info cleanup
        info = {
            "t_now": self.t_now,
            "buffer_size": len(self.orch.buffer),
            "released": bool(int(action) == 1),
            "total_reward": float(reward),
            
            # Reward breakdown (using alpha)
            "reward_makespan_delta": - (delta_mk * alpha) / scale,
            "reward_idle_cost": - (total_idle * (1.0 - alpha)) / scale,
            "reward_stability_penalty": - stability_penalty,
            "reward_final_flush_penalty": final_flush_penalty,
        }
        
        # print(f"obs:{obs},rewards:{reward:.3},act:{action},t_now:{t_event},t_next:{t_new},mk:{mk_now}, mk_prev:{temp}")
        return obs, float(reward), bool(done), bool(trunc), info

        # --------------------------- 最後 FLUSH + 成本 ---------------------------
    def _run_final_flush_and_get_cost(self,
                                    max_flush_rounds: int = 16) -> Tuple[float, float]:
        """
        將 orch.buffer 中剩餘 job 全部排完，並估算：
        - mk_final         : flush 後的最終 makespan
        - total_idle_flush : 在 [t_flush_start, mk_final) 的 idle
        注意：這裡會真的呼叫 event_release_and_reschedule，
            但對 RL 來說這發生在 done 之後，不會再產生新的 step。
        """
        assert self.orch is not None

        # flush 前的 makespan（現在只當 fallback 用）
        mk_before = float(np.max(self.orch.machine_free_time)) if len(self.orch.machine_free_time) > 0 else 0.0

        # flush 起點時間：用目前 gate 時刻
        t_flush_start = float(self.t_now)

        flush_round = 0
        while len(self.orch.buffer) > 0 and flush_round < max_flush_rounds:
            flush_round += 1
            fin = self.orch.event_release_and_reschedule(self.t_now, self.heuristic)
            if fin.get("event") != "batch_finalized":
                break

        # flush 後的 makespan（= mk_final）
        mk_after = float(np.max(self.orch.machine_free_time)) if len(self.orch.machine_free_time) > 0 else mk_before
        mk_final = mk_after
        t_flush_end = mk_final

        # 用全局排程計算 [t_flush_start, t_flush_end) 的 idle
        metrics_flush = self.orch.compute_interval_metrics(t_flush_start, t_flush_end)
        total_idle_flush = float(metrics_flush.get("total_idle", 0.0))

        return mk_final, total_idle_flush
