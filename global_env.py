"""
Event-driven rolling-horizon orchestrator for FJSP
==================================================
關鍵差異（本版修正重點）：
- [REMOVED] 不再使用 dt/tick 的時間步進。
- [ADDED] 以「到達事件」驅動：t 直接跳到下一個事件時間（Exponential inter-arrival）。
- [ADDED] 每個事件：用「上一批 *完整* 靜態結果 snapshot」在 t_event 切分 H/R，再把 R ∪ B 開新批，靜態一次排到底（PPO 或派工法則）。
- [FIXED] H/R 切分一律使用「上一批 *原始* s,e」判斷，不會有先被改寫 s 再判斷的問題。
- [ADDED] 兩個時鐘原則：true_* 一律保存「絕對時間」，state/非 true 欄位一律使用「以 t_event 為基準」的正規化值。
- [FIXED] finalize() 不再把 base 加回 rows 與機台時間；因為 true_* 已為絕對時間。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
import copy
import numpy as np
import numpy.ma as ma

# === 依你的實際檔名調整 ===
from fjsp_env_same_op_nums_online import FJSPEnvForSameOpNums

# [ADDED] 直接在檔案最上面 import（依你要求）
import torch
from model.PPO import PPO_initialize
from params import configs
from common_utils import heuristic_select_action, greedy_select_action


# ==============================================================================
# A. 資料結構
# ==============================================================================

@dataclass
class OperationSpec:
    time_row: Optional[List[float]] = None
    machine_times: Optional[Dict[int, float]] = None

    def proc_time_on(self, m: int) -> float:
        if self.time_row is not None:
            v = float(self.time_row[m])
            return v if v > 0 else 0.0
        if self.machine_times is not None:
            return float(self.machine_times.get(m, 0.0))
        return 0.0


@dataclass
class JobSpec:
    job_id: int
    operations: List[OperationSpec]
    meta: Dict = field(default_factory=dict)


def split_matrix_to_jobs(job_length: np.ndarray, op_pt: np.ndarray, *,
                         base_job_id: int = 0, t_arrive: Optional[float] = None) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    J = int(job_length.shape[0])
    cursor = 0
    for j in range(J):
        L = int(job_length[j])
        ops: List[OperationSpec] = []
        for _ in range(L):
            row = op_pt[cursor]
            ops.append(OperationSpec(time_row=row.astype(float).tolist()))
            cursor += 1
        meta = {}
        if t_arrive is not None:
            meta["t_arrive"] = float(t_arrive)
        meta.setdefault("op_offset", 0)  # [ADDED]
        jobs.append(JobSpec(job_id=base_job_id + j, operations=ops, meta=meta))
    return jobs


# ==============================================================================
# [ADDED] B0. 時間正規化器（兩個時鐘原則）
# ==============================================================================

class _TimeNormalizer:
    """[ADDED] 僅用於建 state 與非 true_* 欄位的時間特徵。
       映射：x_abs -> (x_abs - base) / scale
    """
    def __init__(self, base: float, scale: float):
        self.base = float(base)
        self.scale = max(float(scale), 1e-6)

    def f(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.base) / self.scale


# ==============================================================================
# B. 事件驅動到達生成器（Exponential inter-arrival + burst 釋放）
# ==============================================================================

class EventBurstGenerator:
    """
    [ADDED] 事件驅動到達：
      - inter-arrival ~ Exponential(interarrival_mean)：sample_next_time(t_now) → t_next
      - 每個事件一次釋放 K 筆：K 來自 k_sampler（例如固定 K、Poisson、或自定義）
      - 使用 sd2_fn(base_config; 臨時覆寫 n_j=K) → (job_length, op_pt)，再拆成 List[JobSpec]
    """
    def __init__(self, sd2_fn: Callable, base_config, n_machines: int,
                 interarrival_mean: float = 100,
                 k_sampler: Optional[Callable[[np.random.Generator], int]] = None,
                 rng: Optional[np.random.Generator] = None,
                 starting_job_id: int = 0):
        self.sd2_fn = sd2_fn
        self.cfg = base_config
        self.M = int(n_machines)
        self.interarrival_mean = float(interarrival_mean)
        self.rng = rng or np.random.default_rng()
        self.k_sampler = k_sampler or (lambda _rng: 1)
        self._next_id = int(starting_job_id)
        self._initial_starting_job_id = int(starting_job_id) # Store initial value

    def sample_next_time(self, t_now: float) -> float:
        # Exponential inter-arrival with rate λ
        gap = float(self.rng.exponential(self.interarrival_mean))
        return float(t_now + gap)

    def generate_burst(self, t_event: float) -> List[JobSpec]:
        K = int(self.k_sampler(self.rng))
        if K <= 0:
            return []
        cfg = self.cfg
        old_n_j = getattr(cfg, "n_j", None)
        try:
            setattr(cfg, "n_j", K)
            job_length, op_pt, _ = self.sd2_fn(cfg)
        finally:
            if old_n_j is not None:
                setattr(cfg, "n_j", old_n_j)

        jobs = split_matrix_to_jobs(job_length, op_pt, base_job_id=self._next_id, t_arrive=t_event)
        self._next_id += len(jobs)
        return jobs

    def bump_next_id(self, new_min_id: int):
        """[ADDED] 當初始 buffer 已有工單時，對齊下一個可用 job_id。"""
        self._next_id = max(self._next_id, int(new_min_id))

    def reset(self):
        """
        重置生成器的內部狀態，使其可以從頭開始生成工單。
        """
        self._next_id = self._initial_starting_job_id
        # Note: The internal RNG is not re-seeded here. If a new random sequence
        # is desired for each episode, the RNG itself should be re-created or re-seeded
        # externally before passing to EventBurstGenerator's __init__.



# ==============================================================================
# C. 批次排程記錄器：保存每步 (job, op, machine, start, end)
# ==============================================================================

class BatchScheduleRecorder:
    def __init__(self, batch_jobs: List[JobSpec], n_machines: int):
        self.jobs = batch_jobs
        self.M = int(n_machines)
        self.machine_queues: List[List[Tuple[int, int, float, float]]] = [[] for _ in range(self.M)]
        self.rows: List[Dict] = []

    def record_step(self, env: FJSPEnvForSameOpNums, action: int, env_idx: int = 0):
        J, M = env.number_of_jobs, env.number_of_machines
        chosen_job = int(action // M)
        chosen_mch = int(action % M)
        chosen_op_global = int(env.candidate[env.env_idxs, chosen_job][env_idx])
        op_id_in_job = int(chosen_op_global - env.job_first_op_id[env_idx, chosen_job])

        # [ADDED] 全域偏移（跨重排不重覆）
        job_meta = self.jobs[chosen_job].meta or {}
        op_offset = int(job_meta.get("op_offset", 0))
        op_global_in_job = op_id_in_job + op_offset

        # [CHANGED] 使用「絕對時間」的 true_* 來取 start
        start = float(max(env.true_candidate_free_time[env_idx, chosen_job],
                          env.true_mch_free_time[env_idx, chosen_mch]))

        state, reward, done = env.step(np.array([action]))

        # [CHANGED] true_op_ct 也應為「絕對時間」
        end = float(env.true_op_ct[env_idx, chosen_op_global])
        job_id = int(self.jobs[chosen_job].job_id)

        self.machine_queues[chosen_mch].append((job_id, op_global_in_job, start, end))
        self.rows.append({"job": job_id, "op": op_global_in_job, "machine": chosen_mch,
                          "start": start, "end": end, "duration": end - start})
        return state, reward, done

    def to_rows(self) -> List[Dict]:
        return list(self.rows)

    def clear(self):
        self.machine_queues = [[] for _ in range(self.M)]
        self.rows = []


# ==============================================================================
# D. 全局時間軸控制器（事件驅動、H/R/B 三集合）
# ==============================================================================

class GlobalTimelineOrchestrator:
    def __init__(self, n_machines: int,
                 job_generator: EventBurstGenerator,
                 select_from_buffer: Callable[[List[JobSpec], "GlobalTimelineOrchestrator"], List[int]],
                 release_decider=None,
                 t0: float = 0.0,
                 ):
        self.M = int(n_machines)
        self.t = float(t0)
        self.release_decider = release_decider
        self.generator = job_generator
        self.select_from_buffer = select_from_buffer
        self._time_base: Optional[float] = None

        # --- 三集合 ---
        self.buffer: List[JobSpec] = []     # [B]
        self._R_jobs: List[JobSpec] = []    # [R]
        self._R_rows: List[Dict] = []       # [R rows] 給畫圖
        self._global_rows: List[Dict] = []  # [H]
        self._global_row_keys: Set[Tuple[int, int, int, float, float]] = set()

        # 批次狀態/快照
        self.machine_free_time = np.zeros(self.M, dtype=float)  # [CHANGED] 保存「絕對時間」
        self._builder = _FJSPBatchBuilder(self.M)
        self.current_env: Optional[FJSPEnvForSameOpNums] = None
        self._recorder: Optional[BatchScheduleRecorder] = None
        self._committed_jobs: Optional[List[JobSpec]] = None
        self._active_plan: List[Tuple[int, int, int, float, float]] = []
        self._batch_tcut: Optional[float] = None

        # [ADDED] 上一批「完整」靜態結果快照（不在 finalize 時切）
        self._last_full_rows: List[Dict] = []
        self._metric_rows = {} 
        self._last_jobs_snapshot: List[JobSpec] = []

        # [ADDED] PPO 切換與初始化（依你要求，移到頂部 import，這裡不做 try/except）
        self._use_ppo = bool(getattr(configs, "use_ppo", False))
        self._ppo = PPO_initialize() if self._use_ppo else None
        if self._use_ppo:
            _model_path = getattr(configs, "ppo_model_path", None)
            if _model_path:
                self._ppo.policy.load_state_dict(
                    torch.load(_model_path, map_location=getattr(configs, "device", "cpu"))
                )
            self._ppo.policy.eval()

        self._normalizer: Optional[_TimeNormalizer] = None  # [ADDED]

    # ---- 查詢給畫圖 ----
    def get_global_rows(self) -> List[Dict]:
        return list(self._global_rows)

    def get_R_rows(self) -> List[Dict]:
        return list(self._R_rows)

    # ---- 工具 ----
    def _extend_global_rows_dedup(self, rows: List[Dict]):
        for r in rows:
            k = (int(r["job"]), int(r["op"]), int(r["machine"]),
                 float(r["start"]), float(r["end"]))
            if k not in self._global_row_keys:
                self._global_row_keys.add(k)
                self._global_rows.append(r)

    def _plan_to_rows_with_offset(self, jobs: List[JobSpec],
                                  plan: List[Tuple[int, int, int, float, float]]) -> List[Dict]:
        offset_map = {int(j.job_id): int(j.meta.get("op_offset", 0)) for j in jobs}
        out: List[Dict] = []
        for (jid, opk_local, m, s, e) in plan:
            opk_global = int(opk_local) + int(offset_map.get(int(jid), 0))
            out.append({
                "job": int(jid), "op": opk_global, "machine": int(m),
                "start": float(s), "end": float(e), "duration": float(e - s)
            })
        return out

    # [ADDED] 只用 H 在 tcut 的「在製」求每台機器的 busy-until（絕對時間）
    def _compute_mft_from_H(self, tcut: float) -> np.ndarray:
        busy_until = np.full(self.M, float(tcut), dtype=float)
        for r in self._global_rows:
            s, e, m = float(r["start"]), float(r["end"]), int(r["machine"])
            if s < tcut < e:  # 跨切點在製
                busy_until[m] = max(busy_until[m], e)
        return busy_until

    # ---- 初始化/清空 ----
    def reset(self, *, clear_buffer: bool = True, t0: float = 0.0):
        self.t = float(t0)
        if clear_buffer:
            self.buffer.clear()
        self._R_jobs = []
        self._R_rows = []
        self._global_rows = []
        self._global_row_keys = set()
        self.machine_free_time[:] = 0.0
        self.current_env = None
        self._recorder = None
        self._committed_jobs = None
        self._active_plan = []
        self._batch_tcut = None
        self._last_full_rows = []
        self._last_jobs_snapshot = []
        self._normalizer = None  # [ADDED]

    # [REMOVED] tick(dt) 與時間步進相關 API（事件驅動下不需要）

    # ---- 靜態一次排到底（做法 A：PPO；做法 B：派工法則）----
    def solve_current_batch_static(self, env: FJSPEnvForSameOpNums, initial_state, heuristic: str) -> List[Dict]:
        """
        [CHANGED] 支援 PPO 或 Heuristic。
        - PPO 路徑：使用 initial_state → 每步用 PPO 取 action（greedy or sample），更新 state。
        - Heuristic 路徑：沿用 heuristic_select_action(heuristic, env)。
        """
        if env is not self.current_env:
            raise AssertionError("solve_current_batch_static: env 必須是 current_env")
        if self._recorder is None:
            raise AssertionError("solve_current_batch_static: recorder 不存在")

        self._active_plan = []  # 保險：清空上一批紀錄
        done_flag = np.array([0.0])
        state = initial_state

        if self._use_ppo:
            with torch.no_grad():
                while not bool(done_flag.ravel()[0]):
                    pi, _ = self._ppo.policy(
                        fea_j=state.fea_j_tensor,
                        op_mask=state.op_mask_tensor,
                        candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor,
                        mch_mask=state.mch_mask_tensor,
                        comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                        fea_pairs=state.fea_pairs_tensor
                    )

                    # === Debug（可留可註解）===
                    # print(self.current_env.true_mch_free_time[0])  # 絕對
                    # print("mch free (norm):", self.current_env.mch_free_time[0])  # 正規化
                    # print("state m[6] (norm):", state.fea_m_tensor[0, :, 6])

                    act = greedy_select_action(pi)  # 你原本常用的 greedy
                    act_idx = int(np.array(act.cpu().numpy()).reshape(-1)[0])
                    self.capture_planned_op(env, act_idx)
                    state, _, done_flag = self._recorder.record_step(env, act_idx)
        else:
            while not bool(done_flag.ravel()[0]):
                action = heuristic_select_action(heuristic, env)
                self.capture_planned_op(env, action)
                state, _, done_flag = self._recorder.record_step(env, action)

        return self._recorder.to_rows()

    # ---- 記錄一步（把這步規劃的 s,e 先寫入 _active_plan；保留原始 s,e）----  [ADDED]
    def capture_planned_op(self, env: FJSPEnvForSameOpNums, action: int):
        if env is not self.current_env:
            raise AssertionError("capture_planned_op: env 必須是 current_env")

        J = env.number_of_jobs
        M = env.number_of_machines
        job_idx = int(action // M)
        mch_idx = int(action % M)
        if not (0 <= job_idx < J and 0 <= mch_idx < M):
            raise AssertionError("capture_planned_op: action 越界")

        # 單環境索引（你的 env 多半是單一環境）；若有多環境，沿用 env.env_idxs
        env_idx = getattr(env, "env_idxs", 0)

        # 取該 job 目前的「候選」操作（全域 op 下標），轉回 job 內的 local 序號
        chosen_op_flat = int(env.candidate[env_idx, job_idx])
        op_id_in_job = int(chosen_op_flat - env.job_first_op_id[env_idx, job_idx])

        # [CHANGED] true_* 已為絕對時間
        start_true = float(max(
            env.true_candidate_free_time[env_idx, job_idx],
            env.true_mch_free_time[env_idx, mch_idx]
        ))

        pt_true = float(env.true_op_pt[env_idx, chosen_op_flat, mch_idx])
        end_true = float(start_true + pt_true)

        # 對應到全域 job_id（不是 batch 內索引）
        assert self._committed_jobs is not None
        job_id = int(self._committed_jobs[job_idx].job_id)

        # 累積到本批的「原始計畫」（給 finalize() 轉成全域 rows 用）
        self._active_plan.append((job_id, op_id_in_job, mch_idx, start_true, end_true))
        return job_id, op_id_in_job, mch_idx, start_true, end_true

    # ---- 完成一批（不在這裡切 H/R；只存「完整結果快照」供下一次事件切）----
    def finalize(self, env: FJSPEnvForSameOpNums) -> Dict:
        if env is not self.current_env:
            raise AssertionError("finalize() 參數必須是目前 active 的批次環境")

        # [REMOVED] base 加回：不需要；true_* 已是絕對時間
        # base = float(getattr(self, "_time_base", 0.0))  # [REMOVED]

        # [FIXED] 回寫機台可用時間（本批排到底）→ 直接拷貝絕對時間
        self.machine_free_time = env.true_mch_free_time[0].astype(float).copy()  # [CHANGED]

        # 本批完整 rows（全域 op）；存成上一批「完整」快照（start/end 已是絕對）
        batch_rows: List[Dict] = []
        if self._active_plan and self._committed_jobs:
            batch_rows = self._plan_to_rows_with_offset(self._committed_jobs, self._active_plan)



        # [ADDED] 保存上一批完整結果（絕對時間）
        self._last_full_rows = list(batch_rows)
        self._last_jobs_snapshot = copy.deepcopy(self._committed_jobs)

        # [ADDED] 「右半」給畫圖（此批全部當作 newplan 顯示）
        self._R_rows = list(batch_rows)

        # 清理批次狀態
        self.current_env = None
        self._recorder = None
        self._active_plan = []
        self._committed_jobs = None
        self._batch_tcut = None
        self._time_base = None
        self._normalizer = None  # [ADDED] 下一批會重建

        return {
            "event": "batch_finalized",
            "t": self.t,
            "machine_free_time": self.machine_free_time.copy(),
            "rows": batch_rows,
            "t_cut": None
        }

    # ---- 事件處理（在 t_event：切 H/R、R∪B 開新批 → 一次排到底 → finalize）----
    def event_release_and_reschedule(self, t_event: float, heuristic: str) -> Dict:
        """
        [ADDED] 事件驅動的單次完整流程。
        假設：目前沒有活批（我們的主迴圈每個事件都把上一批排完/收尾了）。
        步驟：
          1) 用 _last_full_rows 在 t_event 切 H/R，更新 R_jobs、機台占用（H 在製 → busy_until）。
          2) 新批 = R_jobs ∪ buffer（B 事件當下清空）。
          3) 建 env → solve_current_batch_static(...) → finalize()。
        回傳：{ "event": "batch_finalized", "rows": ..., "t_cut": t_event, ... }
        """
        self.t = float(t_event)
        base = self.t
        self._time_base = base

        # === 1) 先把上一批完整結果在 t_event 切成 H/R ===
        H_add: List[Dict] = []
        by_job: Dict[int, List[Dict]] = {}
        if self._last_full_rows:
            for r in self._last_full_rows:
                if float(r["start"]) < self.t:
                    H_add.append(dict(r))
            if H_add:
                self._extend_global_rows_dedup(H_add)

            for r in self._last_full_rows:
                by_job.setdefault(int(r["job"]), []).append(r)

        # 機台在製佔用：只看 H 的跨切點（s < t < e）→ 回傳「絕對 busy-until」
        self.machine_free_time = self._compute_mft_from_H(self.t)

        # 切出新的 R_jobs
        remain_jobs: List[JobSpec] = []
        if self._last_jobs_snapshot:
            for js in self._last_jobs_snapshot:
                jid = int(js.job_id)
                seq = sorted(by_job.get(jid, []), key=lambda x: x["op"]) if jid in by_job else []
                k_done = sum(1 for r in seq if float(r["start"]) < self.t)
                inprog = [r for r in seq if (float(r["start"]) < self.t < float(r["end"]))]

                if inprog:
                    ready_at = float(inprog[0]["end"])  # 非搶占：下一道等此在製結束（絕對）
                    slice_from = int([r["op"] for r in seq].index(inprog[0]["op"]) + 1)
                else:
                    ready_at = self.t
                    slice_from = int(k_done)

                base_offset = int(js.meta.get("op_offset", 0))
                new_offset = base_offset + slice_from
                ops_left = js.operations[slice_from:]
                if ops_left:
                    js2 = JobSpec(job_id=jid, operations=list(ops_left), meta=dict(js.meta))
                    js2.meta["op_offset"] = new_offset
                    js2.meta["ready_at"] = ready_at  # [CHANGED] 儲存絕對時間
                    remain_jobs.append(js2)

        # === 2) 新批 = R ∪ B（B 事件當下清空） ===
        jobs_from_B = list(self.buffer)
        self.buffer.clear()
        jobs_new = list(remain_jobs) + jobs_from_B
        if not jobs_new:
            # 沒東西可排：只更新 t 即可
            return {"event": "tick", "t": self.t, "buffer": 0, "new_jobs": 0, "t_cut": self.t}

        # === 3) 建 env → 靜態排到底 → finalize ===
        job_length_list, op_pt_list = self._builder.build(jobs_new)
        env_new = FJSPEnvForSameOpNums(n_j=len(jobs_new), n_m=self.M)

        # [ADDED] 估一個時間尺度：以批中處理時間的 95 分位 × 4，避免過小/過大
        pt = op_pt_list[0]
        pt_nonzero = pt[pt > 0]
        q95 = np.percentile(pt_nonzero, 95) if pt_nonzero.size else 1.0
        time_scale = max(1.0, q95 * 4.0)

        # [ADDED] 啟用正規化器（僅用於 state 與非 true_* 欄位）
        self._normalizer = _TimeNormalizer(base=base, scale=time_scale)

        # 初始化靜態資料
        state0 = env_new.set_initial_data(job_length_list, op_pt_list)

        # ===== 核心修正：true_* = 絕對；非 true 欄位 = 正規化 =====
        mft_abs = self.machine_free_time.astype(float)                     # 機台忙到的絕對時間
        env_new.true_mch_free_time[0, :] = mft_abs                         # ✅ 絕對
        env_new.mch_free_time[0, :]       = self._normalizer.f(mft_abs)    # ✅ 正規化

        for j_idx, js in enumerate(jobs_new):
            # job ready 絕對時間（ready_at 優先，否則 t_arrive，且不得早於 t_event）
            r_abs = float(js.meta.get("ready_at", max(float(js.meta.get("t_arrive", 0.0)), self.t)))
            js.meta["ready_at"] = r_abs                                     # meta 存絕對
            env_new.true_candidate_free_time[0, j_idx] = r_abs              # ✅ 絕對
            env_new.candidate_free_time[0, j_idx]      = self._normalizer.f(r_abs)  # ✅ 正規化

        # 以目前 env arrays 重建 state（會讀非 true 欄位）
        state = env_new.rebuild_state_from_current()
        
        # [ADDED] 若你確定 fea_m_tensor[...,6] 正是「機台忙到時間」的成分，可強制對齊

        # 狀態切換到新批
        self._R_jobs = []      # 交付給本批
        self._R_rows = []
        self._committed_jobs = jobs_new
        self._active_plan = []
        self.current_env = env_new
        self._recorder = BatchScheduleRecorder(jobs_new, self.M)
        self._batch_tcut = float(self.t)

        # 一次排到底（PPO / Heuristic）
        _ = self.solve_current_batch_static(env_new, state, heuristic)
        fin = self.finalize(env_new)  # 存完整 rows（絕對）→ 供下一次事件切分

        # 把 tcut 放進回傳（畫圖/統計用）
        fin["t_cut"] = float(self.t)
        fin["reason"] = "release_event"
        fin["event"] = "batch_finalized"
        return fin
    

        # ====================== [ADDED] HOLD 用：不放行、只前進時間 ======================
    def tick_without_release(self, t_event: float) -> Dict:
        """
        [ADDED] 在 t_event 事件時刻『不放行』：
          - 更新 self.t
          - 把上一批完整計畫中，start < t_event 的片段加入歷史 H（只做左半段，R 不動）
          - 用 H 跨切點占用更新 self.machine_free_time
          - 不建立新 env、不 finalize、不改 _last_full_rows（仍然沿用上一批計畫）
        """
        self.t = float(t_event)

        H_add: List[Dict] = []
        if self._last_full_rows:
            for r in self._last_full_rows:
                if float(r["start"]) < self.t:
                    H_add.append(dict(r))
        if H_add:
            self._extend_global_rows_dedup(H_add)

        # 只看 H 的跨切點在製，更新機台 busy-until（絕對時間）
        self.machine_free_time = self._compute_mft_from_H(self.t)

        # 不改 R_rows / 不改 jobs snapshot
        return {
            "event": "hold_tick",
            "t": self.t,
            "t_cut": self.t,
            "rows_added_to_H": len(H_add),
        }

    def compute_interval_metrics(self, t0: float, t1: float) -> Dict:
        """
        以『時間區間 [t0, t1)』計算：
          - time_avg      : 在此區間「開始」的工序之平均處理時間
          - started_ops   : 此區間開始的工序數
          - idle_ratio    : 各機台在此區間的平均閒置比例
          - total_idle    : 區間總閒置時間（所有機台加總）
          - interval_dt   : 區間長度 dt

        這裡不再只用 self._last_full_rows，而是維護一個
        self._metric_rows: Dict[(job, op) -> row]，跨多次重排累加，
        並且把 end < t0 的工序從集合中移除。
        """
        import numpy as np

        eps = 1e-9
        t0 = float(t0)
        t1 = float(t1)
        dt = max(t1 - t0, 0.0)

        # ========= 0) 安全檢查 =========
        if not hasattr(self, "_metric_rows"):
            self._metric_rows = {}

        # ========= 1) 合併最新的 _last_full_rows 到 _metric_rows =========
        # key 用 (job, op)，同一個工序只保留一筆（以最新 schedule 為準）
        if self._last_full_rows:
            for r in self._last_full_rows:
                # 若沒有 job/op 欄位就跳過（理論上你現在都有）
                if ("job" not in r) or ("op" not in r):
                    continue
                key = (int(r["job"]), int(r["op"]))
                self._metric_rows[key] = r

        # ========= 2) 移除已經完全結束、不再影響 [t0, t1) 的工序 =========
        # end < t0 者，在之後的任何 [t0, t1) 都不可能再有 overlap
        to_delete = []
        for key, r in self._metric_rows.items():
            e = float(r["end"])
            if e < t0:
                to_delete.append(key)
        for key in to_delete:
            del self._metric_rows[key]

        # 現在 rows_for_metric 就是「所有尚可能影響 t >= t0 的工序」
        rows_for_metric = list(self._metric_rows.values())

        # ========= 3) 計算 time_avg（只看在 [t0, t1) 開始的工序） =========
        rows = []
        for r in rows_for_metric:
            s = float(r["start"])
            if (t0 <= s) and (s < t1):
                rows.append(r)

        durations = [float(r["end"]) - float(r["start"]) for r in rows]
        started_ops = len(durations)
        time_avg = float(np.mean(durations)) if durations else 0.0

        # ========= 4) 計算 idle / idle_ratio =========
        # 用 overlap_len 計算每台機台在 [t0, t1) 的忙碌長度，再換算成 idle
        def overlap_len(a0, a1, b0, b1):
            return max(0.0, min(a1, b1) - max(a0, b0))

        busy_sum = 0.0
        for m in range(self.M):
            busy_m = 0.0
            for r in rows_for_metric:
                if int(r["machine"]) != m:
                    continue
                s = float(r["start"])
                e = float(r["end"])
                busy_m += overlap_len(s, e, t0, t1)
            # 單一機台在 [t0, t1) 的忙碌時間不應超過 dt
            busy_sum += min(busy_m, dt)

        total_capacity = dt * max(self.M, 1)
        total_idle = max(0.0, total_capacity - busy_sum)
        idle_ratio = (total_idle / (total_capacity + eps)) if dt > 0 else 0.0

        return {
            "total_idle": total_idle,
            "time_avg": time_avg,
            "started_ops": started_ops,
            "idle_ratio": idle_ratio,
            "interval_dt": dt,
        }




# ==============================================================================
# E. 靜態批次建構器（橋接器）— 與你原本一致
# ==============================================================================

class _FJSPBatchBuilder:
    def __init__(self, n_machines: int):
        self.M = int(n_machines)

    def build(self, jobs: List[JobSpec]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(jobs) == 0:
            raise ValueError("_FJSPBatchBuilder.build(): jobs 為空")
        job_lengths = np.array([len(j.operations) for j in jobs], dtype=int)
        N = int(job_lengths.sum())
        op_pt = np.zeros((N, self.M), dtype=float)
        row = 0
        for job in jobs:
            for op in job.operations:
                if op.time_row is not None:
                    tr = np.asarray(op.time_row, dtype=float)
                    if tr.shape[0] != self.M:
                        raise ValueError(f"time_row 長度 {tr.shape[0]} != M={self.M}")
                    op_pt[row, :] = np.where(tr > 0.0, tr, 0.0)
                elif op.machine_times is not None:
                    for m, t in op.machine_times.items():
                        if 0 <= int(m) < self.M:
                            op_pt[row, int(m)] = float(t)
                else:
                    raise ValueError("OperationSpec 需要 time_row 或 machine_times")
                row += 1
        return [job_lengths], [op_pt]
