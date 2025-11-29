# event_gate_env.py
# =============================== [ADDED] ===============================
import math
import copy
import numpy as np
from typing import Optional, Tuple, Dict

import torch
import gymnasium as gym
from gymnasium import spaces
from params import configs
from data_utils import SD2_instance_generator
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator, split_matrix_to_jobs


class EventGateEnv(gym.Env):  # [ADDED]
    """
    [ADDED] 事件級 RL 環境：
      - 一步 = 一次『到達事件時間』t_cut
      - action: 0=HOLD, 1=RELEASE
      - reward: 以「開始時間在 [t_t, t_{t+1})」的工序平均處理時間 time_avg 來給 r=-time_avg
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 n_machines: int,
                 heuristic: str = "SPT",
                 interarrival_mean: float = 0.1,
                 burst_K: int = 10,
                 event_horizon: int = 200,
                 init_jobs: int = 0,
                 obs_buffer_cap: Optional[int] = None,   # 正規化尺度，用不到就自動估
                 time_scale: Optional[float] = None,     # 正規化尺度，用不到就自動估
                 reward_mode: str = "pure"               # "pure"=-time_avg；"augmented" 會含 idle/time 正規化
                 ):
        super().__init__()
        self.M = int(n_machines)
        self.heuristic = heuristic
        self.interarrival_mean = float(interarrival_mean)
        self.burst_K = int(burst_K)
        self.event_horizon = int(event_horizon)
        self.init_jobs = int(init_jobs)
        self.reward_mode = str(reward_mode)

        # 正規化尺度
        self.obs_buffer_cap = obs_buffer_cap  # 若 None 會動態以 P95 估
        self.time_scale = float(getattr(configs, "norm_scale", 100.0))          

        # 生成器與 orchestrator（reset 裡初始化）
        self.gen = None
        self.orch: Optional[GlobalTimelineOrchestrator] = None

        # 內部時鐘
        self.t_now = 0.0
        self.t_next = None
        self.events_done = 0
        self.enable_full_idle_penalty = bool(getattr(configs, "enable_full_idle_penalty", False))
        self.full_idle_penalty = float(getattr(configs, "full_idle_penalty", 100.0))

        self.enable_final_flush_penalty = bool(getattr(configs, "enable_final_flush_penalty", True))
        self.final_flush_unit_pt = float(getattr(configs, "final_flush_unit_pt", 50.0))  # 先寫死 50，可改 configs

        self._mk_prev = 0.0

        # Gym space（狀態 2 維；動作 {0,1}）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    # --------------------------- 工具：正規化 ---------------------------
    def _norm_buffer(self, size: int) -> float:
        cap = self.obs_buffer_cap
        if cap is None or cap <= 0:
            # 簡易動態尺度：以 burst_K * 3 當參考
            cap = max(1, self.burst_K * 3)
        return float(size) / float(cap)

    def _norm_time(self, x: float) -> float:
        scale = self.time_scale
        return float(x) / float(scale)

    # --------------------------- 狀態抽取 ---------------------------
    def _observe(self) -> np.ndarray:
     # 1) buffer 大小
        buf_size = len(self.orch.buffer)

        # 2) 機台「剩餘忙碌時間」 rem_k = max(machine_free_time_k - t_now, 0)
        mft_abs = np.asarray(self.orch.machine_free_time, dtype=float)  # 絕對 busy-until 時間 T_k
        rem = np.maximum(0.0, mft_abs - float(self.t_now))              # [M]，R_k

        if rem.size > 0:
            # 總剩餘負載 S = Σ R_k
            total_rem = float(rem.sum()) / self.M

            # 離第一台 idle 還有多久 L_min = min R_k
            #   - 若有機器已經 idle，對應 rem=0 => L_min = 0
            #   - 若所有機器都還在忙，L_min > 0
            first_idle_rem = float(rem.min())
        else:
            total_rem = 0.0
            first_idle_rem = 0.0

        # 3) 正規化
        o0 = self._norm_buffer(buf_size)          # buffer 大小 -> [0,1] 之類
        o1 = self._norm_time(total_rem)          # S_norm：整體未來負載
        o2 = self._norm_time(first_idle_rem)     # L_min_norm：離第一台 idle 還有多久

        # 4) high-level state = [ n_buffer, S_norm, L_min_norm ]
        return np.array([o0, o1, o2], dtype=np.float32)


    # --------------------------- reset ---------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = getattr(configs, "event_seed", 42)
        rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
        base_cfg = copy.deepcopy(configs)
        self.gen = EventBurstGenerator(
            sd2_fn=SD2_instance_generator, base_config=base_cfg,
            n_machines=self.M,
            interarrival_mean=self.interarrival_mean,
            k_sampler=lambda _rng: int(self.burst_K),
            rng=rng
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

        # 初始注入（不算 arrive 事件）
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

            # 預設作法：先 release 一次，得到初始完整計畫（可改為 hold 視你需求）
            self.orch.event_release_and_reschedule(self.t_now, self.heuristic)
        # 初始化上一輪 makespan
        try:
            self._mk_prev = float(np.max(getattr(self.orch, "machine_free_time", np.array([0.0]))))
        except Exception:
            self._mk_prev = 0.0

        # 第一個到達事件
        self.t_next = self.gen.sample_next_time(self.t_now)

        obs = self._observe()
        info = {"t_now": self.t_now, "t_next": self.t_next}
        return obs, info

    # --------------------------- step ---------------------------
    def step(self, action: int):
        """
        流程（對齊你定義的 r_t = f([t, t_next)) ）：
          1) 在 t_now 發生到達：把新工單丟進 buffer
          2) 依 action (0/1) 做 HOLD / RELEASE
          3) 取下一個事件時間 t_new，並用『目前完整計畫』計算 [t_now, t_new) 的區間指標
          4) 以 -time_avg（或 augmented）作為 reward
          5) 時鐘前進到 t_new，回傳新觀察
        """
        assert self.orch is not None and self.gen is not None, "env not reset"

        t_event = float(self.t_now)

        # (1) 到達：生成新工單 → buffer
        new_jobs = self.gen.generate_burst(t_event)
        if new_jobs:
            self.orch.buffer.extend(new_jobs)
        
        #  判斷「此刻是否全 idle」
        mft_abs = np.asarray(self.orch.machine_free_time, dtype=float)
        all_idle_now = bool(np.all(mft_abs <= t_event + 1e-9))


        # (2) HOLD / RELEASE
        if int(action) == 1:
            # RELEASE：做一次完整重排
            self.orch.event_release_and_reschedule(t_event, self.heuristic)
        else:
            # HOLD：不放行，只更新 H 與機台占用
            self.orch.tick_without_release(t_event)

        # (3) 下一個事件時間
        t_new = float(self.gen.sample_next_time(t_event))

        # 用「目前的完整計畫（_last_full_rows）」計算 [t_event, t_new) 的區間指標
        met = self.orch.compute_interval_metrics(t_event, t_new)
        time_avg = float(met["time_avg"])
        idle_ratio = float(met["idle_ratio"])
        dt = float(met["interval_dt"])
        total_idle = float(met["total_idle"])
        if dt < 1:
            dt = 1
        #  取得目前 makespan（絕對忙碌最晚時間）
        scale = float(getattr(configs, "reward_scale", 50.0))
        try:
            mk_now = float(np.max(getattr(self.orch, "machine_free_time", np.array([0.0]))))
        except Exception:
            mk_now = self._mk_prev
        # (4) reward
        # (D) reward：三種模式
        if self.reward_mode == "pure":
            reward = (- time_avg)/scale
        elif self.reward_mode == "augmented":
            reward = (- (time_avg / (dt + 1e-9)) - 0.5 * idle_ratio) / scale
        elif self.reward_mode == "delta_makespan":  
            delta_mk = (mk_now - self._mk_prev) 
            reward = (- float(delta_mk)*0.3- total_idle*0.7)/scale
        elif self.reward_mode == "delta_makespan_avg":  
            delta_mk = ((mk_now - self._mk_prev) / dt) 
            reward =( - float(delta_mk)- idle_ratio) / scale
        else:
            # 預設回退 pure
            reward = - time_avg
        # if np.random
        
        #可選：全 idle 懲罰（buffer 有件、全 idle、且選擇 HOLD）
        full_idle_penalized = False
        if self.enable_full_idle_penalty:
            if (int(action) == 0) and (len(self.orch.buffer) > 0) and all_idle_now:
                reward -= float(self.full_idle_penalty)
                full_idle_penalized = True

        # (5) 前進時鐘
        self.t_now = t_new
        self.events_done += 1

        # done?
        done = (self.events_done >= self.event_horizon)

        #  若終局且 buffer 還有工單 → 強制重排懲罰（不真的重排，只給懲罰）
        final_flush_penalized = False
        if done and self.enable_final_flush_penalty:
            leftover = int(len(self.orch.buffer))
            if leftover > 0:
                reward -= float(self.final_flush_unit_pt) * float(leftover)
                final_flush_penalized = True

        # 更新「上一輪 makespan」
        self._mk_prev = mk_now


        obs = self._observe()
        info = {
            "t_now": self.t_now,
            "time_avg": time_avg,
            "idle_ratio": idle_ratio,
            "dt": dt,
            "buffer_size": len(self.orch.buffer),
            "released": bool(int(action) == 1),
        }
        return obs, float(reward), bool(done), False, info  # gymnasium API
