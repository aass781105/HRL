# dynamic_wrapper.py
# 一個 episode = 一次完整動態模擬；固定「arrival 事件次數 N」到達時 release 形成靜態子問題，
# PPO 在子問題內逐步決策（state/ reward/ tensors 與原 FJSPEnv 相容）。
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Callable
import copy
import numpy as np

from params import configs
from data_utils import SD2_instance_generator
from global_env import GlobalTimelineOrchestrator, EventBurstGenerator, split_matrix_to_jobs
from PPO_orchestrator_adapter import OrchestratorAdapter   # 需搭配你將提供的 adapter 檔

# === 小工具 ===
def _spawn_rngs(base_seed: int, episode_idx: int) -> Tuple[np.random.Generator,
                                                           np.random.Generator,
                                                           np.random.Generator]:
    """
    以單一 episode 基種子派生三個獨立 RNG：cadence / arrival / burst
    （不另外新檔，集中於本 wrapper 內，確保重現性且彼此不耦合）
    """
    # 使用 SeedSequence spawn 三條獨立序列
    ss = np.random.SeedSequence(int(base_seed) + int(episode_idx))
    s_cad, s_arr, s_bur = ss.spawn(3)
    return (np.random.default_rng(s_cad),
            np.random.default_rng(s_arr),
            np.random.default_rng(s_bur))


class DynamicToStaticEnvWrapper:
    """
    PPO 視角下的環境：
      - reset() 會建立『第一個靜態子問題』並回傳 state（張量介面與原 FJSPEnv 相同）
      - step(action) 在『當前子問題』走一步；子問題若結束，會自動 finalize 並視『固定 arrival 事件次數 N』
        判斷是否在下一個事件時刻 release 開新子問題；直到全模擬（arrival 事件數達標 + 最後 flush）結束。
    重要約定：
      - 子問題邊界 ≠ episode 結束 → 只有全模擬結束才 done=True
      - reward/ state/ tensors 與你的 FJSPEnvForSameOpNums 完全對齊（Trainer 無需改）
    """

    # ------------------------------- 初始化 -------------------------------
    def __init__(self,
                 cfg,
                 n_machines: Optional[int] = None,
                 sd2_fn: Callable = SD2_instance_generator,
                 base_config: Optional[Any] = None,
                 # 以下若不提供，預設都讀 configs 裡的設定
                 interarrival_mean: Optional[float] = None,
                 burst_K: Optional[int] = None,
                 arrival_target_events: Optional[int] = None,
                 init_jobs: Optional[int] = None,
                 cadence_choices: Optional[list] = None,
                 cadence_min: Optional[int] = None,
                 cadence_max: Optional[int] = None,
                 heuristic_for_preview: str = "SPT"  # 僅在需要預先出第一批時用不到/少用
                 ):
        self.cfg = cfg
        self.M = int(n_machines if n_machines is not None else getattr(self.cfg, "n_m", 10))

        # --- 動態參數（預設讀 params.py） ---
        self.interarrival_mean = float(interarrival_mean if interarrival_mean is not None
                                       else getattr(self.cfg, "interarrival_mean", 1.0))
        self.burst_K = int(burst_K if burst_K is not None
                           else getattr(self.cfg, "burst_size", 1))
        self.arrival_target_events = int(arrival_target_events if arrival_target_events is not None
                                         else getattr(self.cfg, "arrival_target", 100))
        self.init_jobs = int(init_jobs if init_jobs is not None
                             else getattr(self.cfg, "initial_jobs", 0))

        # cadence（固定 arrival 事件次數）
        self.cadence_choices = cadence_choices if cadence_choices is not None \
            else getattr(self.cfg, "cadence_choices", None)
        self.cadence_min = int(cadence_min if cadence_min is not None
                               else getattr(self.cfg, "cadence_min", 1))
        self.cadence_max = int(cadence_max if cadence_max is not None
                               else getattr(self.cfg, "cadence_max", max(1, self.cadence_min)))

        self.heuristic_for_preview = str(heuristic_for_preview)

        # 重現性
        self.episode_seed_base = int(getattr(self.cfg, "episode_seed_base", 12345))
        self._episode_idx = 0  # 每次 reset 後自增

        # 內部組件（reset 時建立）
        self._rng_cad = None
        self._rng_arr = None
        self._rng_bur = None
        self._cadence_N = None               # 本 episode 固定的 arrival cadence
        self._arrival_event_count = 0        # 已觸發的 arrival 事件數
        self._done = False

        self._gen: Optional[EventBurstGenerator] = None
        self._orch: Optional[GlobalTimelineOrchestrator] = None
        self._adapter: Optional[OrchestratorAdapter] = None

        # 時鐘
        self._t_now = 0.0

        # 當前子問題是否存在
        self._has_active_batch = False

        # 指標累計（可給 trainer 拉取做 log）
        self._episode_metrics: Dict[str, Any] = {
            "cadence_N": None,
            "arrival_events": 0,
            "releases": 0,
            "final_makespan": 0.0,
        }

    # ------------------------------- 公開 API -------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        建立一個新的動態模擬 episode，並強保回傳『第一個靜態子問題』的 state。
        若沒有 init_jobs，會自動前推到第一個滿足 cadence 的 arrival 事件並 release。
        """
        # episode index/ seed
        ep_seed_base = int(self.episode_seed_base if seed is None else seed)
        self._rng_cad, self._rng_arr, self._rng_bur = _spawn_rngs(ep_seed_base, self._episode_idx)
        self._episode_idx += 1

        # cadence N
        if self.cadence_choices and len(self.cadence_choices) > 0:
            self._cadence_N = int(self._rng_cad.choice(self.cadence_choices))
        else:
            lo, hi = int(self.cadence_min), int(self.cadence_max)
            if hi < lo:
                hi = lo
            self._cadence_N = int(self._rng_cad.integers(lo, hi + 1))

        # 清狀態
        self._t_now = 0.0
        self._arrival_event_count = 0
        self._done = False
        self._has_active_batch = False
        self._episode_metrics = {
            "cadence_N": self._cadence_N,
            "arrival_events": 0,
            "releases": 0,
            "final_makespan": 0.0,
        }

        # 構建 generator 與 orchestrator
        base_cfg = copy.deepcopy(self.cfg)
        self._gen = EventBurstGenerator(
            sd2_fn=SD2_instance_generator,
            base_config=base_cfg,
            n_machines=self.M,
            interarrival_mean=self.interarrival_mean,
            k_sampler=lambda _rng: int(self.burst_K),
            rng=self._rng_arr,
            starting_job_id=0
        )
        self._orch = GlobalTimelineOrchestrator(
            n_machines=self.M,
            job_generator=self._gen,
            select_from_buffer=lambda buf, o: list(range(len(buf))),
            release_decider=None,
            t0=0.0,
        )
        self._adapter = OrchestratorAdapter(self._orch, self.M)

        # 初始注入（若有）
        if self.init_jobs > 0:
            tmp_cfg = copy.deepcopy(self.cfg)
            old_nj = getattr(tmp_cfg, "n_j", None)
            try:
                setattr(tmp_cfg, "n_j", int(self.init_jobs))
                jl, pt, _ = SD2_instance_generator(tmp_cfg)
            finally:
                if old_nj is not None:
                    setattr(tmp_cfg, "n_j", old_nj)
            init_jobs = split_matrix_to_jobs(jl, pt, base_job_id=0, t_arrive=0.0)
            self._orch.buffer.extend(init_jobs)
            # 對齊下一個 job id
            max_existing = max((j.job_id for j in self._orch.buffer), default=-1)
            self._gen.bump_next_id(max_existing + 1)

            # 直接 release 成第一批，保證 reset 回傳 state
            state0 = self._adapter.begin_new_batch(t_event=self._t_now)
            self._has_active_batch = True
            self._episode_metrics["releases"] += 1
            return state0, {"t_now": self._t_now, "cadence_N": self._cadence_N}

        # 否則：自動前推到第一個滿足 cadence 的 arrival 事件並 release
        while not self._has_active_batch:
            # 產生一個 arrival 事件（注意：此處以『事件數』計 cadence，不以到達筆數）
            t_event = float(self._gen.sample_next_time(self._t_now))
            self._t_now = t_event
            self._arrival_event_count += 1
            self._episode_metrics["arrival_events"] = self._arrival_event_count

            # 到達 -> 放 K 筆到 buffer
            new_jobs = self._gen.generate_burst(self._t_now)
            if new_jobs:
                self._orch.buffer.extend(new_jobs)

            # 是否達到 cadence → release 開第一批
            if (self._arrival_event_count % self._cadence_N) == 0:
                state0 = self._adapter.begin_new_batch(t_event=self._t_now)
                self._has_active_batch = True
                self._episode_metrics["releases"] += 1
                break
            else:
                # 不 release：只把左半段寫入歷史、更新在製
                self._adapter.tick_without_release(self._t_now)

        return state0, {"t_now": self._t_now, "cadence_N": self._cadence_N}

    def step(self, actions: np.ndarray):
        """
        在當前『靜態子問題』走一步。
        - 若子問題尚未完成：回 (next_state, reward, done=False)
        - 若子問題完成：自動 finalize；若尚未達成 episode 終止條件，根據 cadence 推進到下一個 release，
          建立『下一個子問題』並回它的初始 state（仍 done=False）。
        - 當 arrival 事件數達標，且 flush 之後所有工單處理完畢 → done=True。
        """
        assert self._adapter is not None and self._has_active_batch, \
            "環境未 reset 或當前沒有 active 子問題可互動。"

        # 子問題內走一步
        state, reward, sub_done, info = self._adapter.step_in_batch(actions)

        # 仍在同一子問題
        if not bool(sub_done):
            return state, reward, np.array([False], dtype=bool)

        # 子問題結束：先 finalize
        fin = self._adapter.finalize_batch()
        # 更新 makespan（絕對忙到最晚時間）
        try:
            mft = np.asarray(self._orch.machine_free_time, dtype=float)
            self._episode_metrics["final_makespan"] = float(np.max(mft)) if mft.size else 0.0
        except Exception:
            pass

        # 檢查是否已達到 arrival 事件數目標（不再繼續生成 arrival）
        arrival_done = (self._arrival_event_count >= self.arrival_target_events)

        # 如果到達目標，且 buffer 還有件、或 R 內仍可開新批 → 做最後 flush（強制 release）
        if arrival_done:
            # 若 buffer 有件 → 立刻開新批；若沒有 → 表示真的結束
            if len(self._orch.buffer) > 0:
                # 最後一次 release（flush）
                nxt_state = self._adapter.begin_new_batch(t_event=self._t_now)
                self._has_active_batch = True
                self._episode_metrics["releases"] += 1
                # 回新子問題的初始 state（done=False）
                return nxt_state, reward, np.array([False], dtype=bool)
            else:
                # 沒有待處理 → 真正 episode 結束
                self._has_active_batch = False
                self._done = True
                return state, reward, np.array([True], dtype=bool)

        # 還沒達到 arrival 目標，持續前推 arrival 事件直到達到 cadence → 開下一批
        self._has_active_batch = False
        while not self._has_active_batch:
            t_event = float(self._gen.sample_next_time(self._t_now))
            self._t_now = t_event
            self._arrival_event_count += 1
            self._episode_metrics["arrival_events"] = self._arrival_event_count

            # 到達 → 放 K 筆到 buffer
            new_jobs = self._gen.generate_burst(self._t_now)
            if new_jobs:
                self._orch.buffer.extend(new_jobs)

            if (self._arrival_event_count % self._cadence_N) == 0:
                # 到 cadence → 開新批
                nxt_state = self._adapter.begin_new_batch(t_event=self._t_now)
                self._has_active_batch = True
                self._episode_metrics["releases"] += 1
                # 回『新子問題』的初始 state（注意：reward 用剛剛最後一步的 reward）
                return nxt_state, reward, np.array([False], dtype=bool)
            else:
                # 不 release：只把左半段寫入歷史、更新在製（保持上一批完整計畫）
                self._adapter.tick_without_release(self._t_now)

        # 理論到不了這裡
        return state, reward, np.array([False], dtype=bool)

    # ------------------------------- 輔助（Trainer 可選用） -------------------------------
    def get_episode_metrics(self) -> Dict[str, Any]:
        """回傳本 episode 的彙總指標（可在 done=True 後呼叫做紀錄）。"""
        return dict(self._episode_metrics)

    # 可選：讓外部更新某些動態參數（例如做 curriculum）
    def set_cadence_range(self, lo: int, hi: int):
        self.cadence_choices = None
        self.cadence_min = int(lo)
        self.cadence_max = int(hi)

    def set_cadence_choices(self, choices: list):
        self.cadence_choices = list(choices)
