# orchestrator_adapter.py
# 讓 GlobalTimelineOrchestrator 支援「外部逐步驅動」的最小適配層：
# - begin_new_batch(t_event)  ：在事件時刻切 H/R、建立新靜態子問題並回第一個 state
# - step_in_batch(action)     ：在當前子問題內走一步（不 finalize）
# - finalize_batch()          ：收尾本批（保存完整 rows、同步機台忙到時間）
# - tick_without_release(t)   ：不放行，只更新 H 與在製占用
#
# 重點：完全沿用你在 global_env.py 既有的邏輯與資料結構；不更動獎勵/狀態定義。

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# === 直接沿用你現有檔案內的類別與工具 ===
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from global_env import (
    GlobalTimelineOrchestrator,
    BatchScheduleRecorder,
    _FJSPBatchBuilder,          # 若之後抽出 builders 也可改由那裡匯入
    split_matrix_to_jobs,       # 目前僅在 dynamic wrapper 用，這裡不需要
    # 內部工具：時間正規化器與 H 在製推算；雖然底線開頭，但可安全匯入使用
    _TimeNormalizer,
)


class OrchestratorAdapter:
    """
    對 GlobalTimelineOrchestrator 的輕量適配器：
    - 不「一次排到底」，而是將流程拆成三段，供外部（如 DynamicToStaticEnvWrapper）逐步控制。
    - 嚴格遵循你在 global_env.py 既有的欄位與流程，避免行為漂移。
    """

    def __init__(self, orchestrator: GlobalTimelineOrchestrator, n_machines: int):
        self.o: GlobalTimelineOrchestrator = orchestrator
        self.M = int(n_machines)

    # ---------------------------------------------------------------------
    # 1) 在 t_event 切 H/R，組成新批，回第一個 state（不解題）
    # ---------------------------------------------------------------------
    def begin_new_batch(self, t_event: float):
        """
        在事件時刻 t_event：
          1) 用上一批『完整』rows 切 H/R，更新 machine_free_time（絕對時間）
          2) 新批 = R_jobs ∪ buffer（清空 B）
          3) 建立新的 FJSPEnv（僅 set_initial_data 與重建 state；不「一次排到底」）
        回傳：
          - state0（與你的 FJSPEnv 相同的 state 物件），若此刻沒有可排工單則回 None。
        """
        o = self.o
        o.t = float(t_event)
        base = o.t
        o._time_base = base

        # === (1) 先用上一批完整 rows 在 t_event 切 H/R（左半進 H） ===
        H_add: List[Dict[str, float]] = []
        by_job: Dict[int, List[Dict[str, float]]] = {}
        if o._last_full_rows:
            for r in o._last_full_rows:
                if float(r["start"]) < o.t:
                    H_add.append(dict(r))
            if H_add:
                o._extend_global_rows_dedup(H_add)
            for r in o._last_full_rows:
                by_job.setdefault(int(r["job"]), []).append(r)

        # 機台在製占用：僅看 H 的跨切點（s < t < e）→ 得到「絕對 busy-until」
        o.machine_free_time = o._compute_mft_from_H(o.t)

        # === (1.5) 切出新的 R_jobs（剩餘未做完的片段），保留 op_offset 與 ready_at（絕對） ===
        remain_jobs = []
        if o._last_jobs_snapshot:
            for js in o._last_jobs_snapshot:
                jid = int(js.job_id)
                seq = sorted(by_job.get(jid, []), key=lambda x: x["op"]) if jid in by_job else []
                # 已開始的計數與「是否有在製」
                k_done = sum(1 for r in seq if float(r["start"]) < o.t)
                inprog = [r for r in seq if (float(r["start"]) < o.t < float(r["end"]))]

                if inprog:
                    # 非搶占：在製那一道完工時間之後才可派下一道
                    ready_at = float(inprog[0]["end"])
                    slice_from = int([r["op"] for r in seq].index(inprog[0]["op"]) + 1)
                else:
                    ready_at = o.t
                    slice_from = int(k_done)

                base_offset = int(js.meta.get("op_offset", 0))
                new_offset = base_offset + slice_from
                ops_left = js.operations[slice_from:]

                if ops_left:
                    js2 = type(js)(job_id=jid, operations=list(ops_left), meta=dict(js.meta))
                    js2.meta["op_offset"] = new_offset
                    js2.meta["ready_at"]  = ready_at   # 絕對時間
                    remain_jobs.append(js2)

        # === (2) 新批 = R_jobs ∪ buffer（清空 B） ===
        jobs_from_B = list(o.buffer)
        o.buffer.clear()
        jobs_new = list(remain_jobs) + jobs_from_B
        if not jobs_new:
            # 沒東西可排：不建立 env，讓呼叫端決定是否前推到下一事件
            return None

        # === (3) 建 env（不解題） ===
        job_length_list, op_pt_list = o._builder.build(jobs_new)
        env_new = FJSPEnvForVariousOpNums(n_j=len(jobs_new), n_m=self.M)

        # 估時間正規化尺度：批內加工時間 95 分位 × 4
        pt = op_pt_list[0]
        pt_nonzero = pt[pt > 0]
        q95 = np.percentile(pt_nonzero, 95) if pt_nonzero.size else 1.0
        time_scale = max(1.0, float(q95) * 4.0)

        # 啟用正規化器（僅用於 state 與非 true_* 欄位）
        o._normalizer = _TimeNormalizer(base=base, scale=time_scale)

        # 初始化靜態資料
        state0 = env_new.set_initial_data(job_length_list, op_pt_list)

        # 絕對/正規化對齊（機台忙到時間）
        mft_abs = o.machine_free_time.astype(float)
        env_new.true_mch_free_time[0, :] = mft_abs
        env_new.mch_free_time[0, :]      = o._normalizer.f(mft_abs)

        # job 的 ready_at（絕對）與其正規化對齊
        for j_idx, js in enumerate(jobs_new):
            r_abs = float(js.meta.get("ready_at", max(float(js.meta.get("t_arrive", 0.0)), o.t)))
            js.meta["ready_at"] = r_abs
            env_new.true_candidate_free_time[0, j_idx] = r_abs
            env_new.candidate_free_time[0, j_idx]      = o._normalizer.f(r_abs)

        # 以目前 env arrays 重建 state（會讀非 true_* 欄位）
        state = env_new.rebuild_state_from_current()

        # 設置本批次狀態（與你原本一致）
        o._R_jobs = []
        o._R_rows = []
        o._committed_jobs = jobs_new
        o._active_plan = []
        o.current_env = env_new
        o._recorder = BatchScheduleRecorder(jobs_new, self.M)
        o._batch_tcut = float(o.t)

        return state

    # ---------------------------------------------------------------------
    # 2) 子問題內走一步（不 finalize）
    # ---------------------------------------------------------------------
    def step_in_batch(self, action) -> Tuple[Any, np.ndarray, bool, Dict[str, Any]]:
        """
        在當前子問題內走一步：
          - 捕捉本步計畫（s,e 的絕對時間）
          - 以 env.step(...) 推進一小步
          - 回 (next_state, reward, sub_done, info)
        注意：
          - sub_done=True 表示『子問題完成』（不是 episode 結束）；呼叫端應接著 finalize 或開新批
        """
        o = self.o
        env = o.current_env
        rec = o._recorder
        assert env is not None and rec is not None, "step_in_batch: 尚未 begin_new_batch 或批次狀態遺失。"

        # 支援 numpy / tensor / 標量；取單一整數動作
        if isinstance(action, (np.ndarray,)):
            if action.ndim > 0:
                act_int = int(action.reshape(-1)[0])
            else:
                act_int = int(action)
        else:
            try:
                act_int = int(action)
            except Exception:
                # 例如來自 torch.Tensor
                act_int = int(np.array(action).reshape(-1)[0])

        # 記錄計畫（本步 s,e）
        o.capture_planned_op(env, act_int)

        # 進一步（由 BatchScheduleRecorder 呼叫 env.step 實作）
        next_state, reward, done_flag = rec.record_step(env, act_int)

        # 對齊你的介面：reward 為 numpy array，sub_done 為 bool
        reward_np = np.asarray(reward)
        sub_done = bool(np.array(done_flag).ravel()[0])

        info = {"subproblem_boundary": sub_done}
        return next_state, reward_np, sub_done, info

    # ---------------------------------------------------------------------
    # 3) 完成本批（保存 rows、同步機台忙到時間等）
    # ---------------------------------------------------------------------
    def finalize_batch(self) -> Dict[str, Any]:
        """
        呼叫 orchestrator.finalize(current_env) 完成本批：
          - 機台忙到時間（絕對）同步到 orchestrator.machine_free_time
          - 保存上一批完整 rows（供下一事件切 H/R）
          - 清空本批狀態（current_env / recorder / plan 等）
        回傳：orchestrator.finalize(...) 的結果 dict
        """
        o = self.o
        env = o.current_env
        assert env is not None, "finalize_batch: 無當前批次可 finalize。"
        fin = o.finalize(env)
        return fin

    # ---------------------------------------------------------------------
    # 4) 不放行、只前進時間（左半段進 H；更新在製占用）
    # ---------------------------------------------------------------------
    def tick_without_release(self, t_event: float) -> Dict[str, Any]:
        """
        在事件時刻『不放行』，僅：
          - 更新 orchestrator.t
          - 將 start < t_event 的 rows 左半加入 H
          - 以 H 跨切點更新機台 busy-until（絕對）
        """
        return self.o.tick_without_release(t_event)
