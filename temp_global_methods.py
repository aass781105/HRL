# ---- 完成一批（不在這裡切 H/R；只存「完整結果快照」供下一次事件切）----
    def finalize(self, env: FJSPEnvForVariousOpNums) -> Dict:
        if env is not self.current_env:
            raise AssertionError("finalize() 參數必須是目前 active 的批次環境")

        # 1. 更新機台物理狀態
        self.machine_free_time = env.true_mch_free_time[0].astype(float).copy()

        # 2. 轉換本批計畫為 rows
        batch_rows: List[Dict] = []
        if self._active_plan and self._committed_jobs:
            offset_map = {int(j.job_id): int(j.meta.get("op_offset", 0)) for j in self._committed_jobs}
            for (jid, opk_local, m, s, e) in self._active_plan:
                opk_global = int(opk_local) + int(offset_map.get(int(jid), 0))
                batch_rows.append({
                    "job": int(jid), "op": opk_global, "machine": int(m),
                    "start": float(s), "end": float(e), "duration": float(e - s)
                })

        # 3. 字典化合併：確保 (job, op) 全系統唯一
        full_plan_dict = {(int(r["job"]), int(r["op"])): r for r in self._last_full_rows}
        for r in batch_rows:
            full_plan_dict[(int(r["job"]), int(r["op"]))] = r
            
        new_full_rows = sorted(list(full_plan_dict.values()), key=lambda x: (x["job"], x["op"]))

        # 4. 更新 Snapshot
        committed_job_ids = {js.job_id for js in self._committed_jobs} if self._committed_jobs else set()
        unaffected_jobs = [js for js in self._last_jobs_snapshot if js.job_id not in committed_job_ids]
        
        # [CRITICAL] 這裡放回 Snapshot 的必須是完整定義的 Job (保持 operations 清單完整)
        # 我們會確保在 event_release_and_reschedule 裡才進行切分
        new_snapshot = unaffected_jobs + (self._committed_jobs if self._committed_jobs else [])

        self._last_full_rows = new_full_rows
        self._last_jobs_snapshot = copy.deepcopy(new_snapshot)
        self._R_rows = list(batch_rows)
        batch_tardiness = float(np.sum(env.accumulated_tardiness))

        # 清理
        self.current_env = None; self._recorder = None; self._active_plan = []; self._committed_jobs = None
        self._batch_tcut = None; self._time_base = None; self._normalizer = None

        return {
            "event": "batch_finalized", "t": self.t, "machine_free_time": self.machine_free_time.copy(),
            "rows": batch_rows, "t_cut": None, "batch_tardiness": batch_tardiness
        }

    # ---- 事件處理（在 t_event：切 H/R、R∪B 開新批 → 一次排到底 → finalize）----
    def event_release_and_reschedule(self, t_event: float) -> Dict:
        self.t = float(t_event)
        
        # 1. 物理切分：凡是 start < t_now 的，通通進入歷史 H
        H_add = [dict(r) for r in self._last_full_rows if float(r["start"]) < self.t]
        if H_add: self._extend_global_rows_dedup(H_add)
        self.machine_free_time = self._compute_mft_from_H(self.t)

        by_job = {}
        for r in self._last_full_rows: by_job.setdefault(int(r["job"]), []).append(r)

        # 2. 準備剩餘工序 R
        remain_jobs = []
        if self._last_jobs_snapshot:
            for js in self._last_jobs_snapshot:
                jid = int(js.job_id)
                # 真正的物理進度：該工單在當前計畫中已經「開始」的工序
                # 注意：即便甘特圖上有幽靈工序 (op > len)，這裡也會被 js.operations[n_started:] 導正
                started_ops_rows = [r for r in by_job.get(jid, []) if float(r["start"]) < self.t]
                n_started = len(started_ops_rows)
                
                # 就緒時間
                inprog = [r for r in started_ops_rows if float(r["end"]) > self.t]
                ready_at = float(inprog[0]["end"]) if inprog else self.t
                
                # [FIX] 確保從原始 JobSpec 提取剩餘工序，保證工序數量守恆
                # 如果 JS Snapshot 裡的是切分過的，我們需要補救。
                # 這裡假設 js.operations 在 Snapshot 裡是完整的（由 finalize 保證）
                ops_left = js.operations[n_started:]
                
                if ops_left:
                    # 建立一個給這波 Batch 用的臨時工單物件
                    js_batch = copy.deepcopy(js)
                    js_batch.operations = list(ops_left) # 僅給予剩餘工序
                    js_batch.meta["op_offset"] = n_started # 標記它是從第幾道開始的
                    js_batch.meta["ready_at"] = ready_at
                    remain_jobs.append(js_batch)

        # 3. 組合新批次 R ∪ B
        jobs_new = remain_jobs + list(self.buffer); self.buffer.clear()
        if not jobs_new: return {"event": "tick", "t": self.t, "buffer": 0, "new_jobs": 0, "t_cut": self.t}

        # 4. 靜態環境初始化
        job_length_list, op_pt_list = self._builder.build(jobs_new)
        env_new = FJSPEnvForVariousOpNums(n_j=len(jobs_new), n_m=self.M)
        pt = op_pt_list[0]; pt_nonzero = pt[pt > 0]
        time_scale = max(1.0, np.percentile(pt_nonzero, 95) * 4.0 if pt_nonzero.size else 1.0)
        self._normalizer = _TimeNormalizer(base=self.t, scale=time_scale)
        
        state0 = env_new.set_initial_data(job_length_list, op_pt_list)
        mft_abs = self.machine_free_time.astype(float)
        env_new.true_mch_free_time[0, :] = mft_abs; env_new.mch_free_time[0, :] = self._normalizer.f(mft_abs)
        for j_idx, js in enumerate(jobs_new):
            r_abs = float(js.meta.get("ready_at", self.t))
            env_new.true_candidate_free_time[0, j_idx] = r_abs; env_new.candidate_free_time[0, j_idx] = self._normalizer.f(r_abs)
        
        state = env_new.rebuild_state_from_current()
        self._committed_jobs = jobs_new; self._recorder = BatchScheduleRecorder(jobs_new, self.M); self._batch_tcut = self.t; self.current_env = env_new
        
        # 5. 排程執行
        _ = self.solve_current_batch_static(env_new, state)
        fin = self.finalize(env_new)
        fin.update({"t_cut": self.t, "reason": "release_event", "event": "batch_finalized"})
        return fin
