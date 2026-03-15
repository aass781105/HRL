"""
Event-driven rolling-horizon orchestrator for FJSP (HIGH PERFORMANCE)
=====================================================================
Optimizations:
- Eliminated redundant list concatenations in state/reward calculations.
- Incremental history tracking for KPI stats.
- Efficient interval metric calculation (only scans active rows).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
import copy
import numpy as np

from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
import torch
from model.PPO import PPO_initialize
from params import configs
from common_utils import heuristic_select_action, greedy_select_action, sample_action
from data_utils import generate_due_dates


def _sync_cuda_for_profile():
    if torch.cuda.is_available() and bool(getattr(configs, "profile_cuda_sync", True)):
        torch.cuda.synchronize()

@dataclass
class OperationSpec:
    time_row: Optional[List[float]] = None
    machine_times: Optional[Dict[int, float]] = None
    avg_proc_time: float = 0.0

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
                         base_job_id: int = 0, t_arrive: Optional[float] = None,
                         due_dates: Optional[np.ndarray] = None) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    J = int(job_length.shape[0])
    cursor = 0
    for j in range(J):
        L = int(job_length[j])
        ops: List[OperationSpec] = []
        total_p, min_p = 0.0, 0.0
        for _ in range(L):
            row = op_pt[cursor]
            valid = row[row > 0]
            avg_v = np.mean(valid) if valid.size > 0 else 0.0
            min_v = np.min(valid) if valid.size > 0 else 0.0
            total_p += float(avg_v)
            min_p += float(min_v)
            ops.append(OperationSpec(time_row=row.astype(float).tolist(), avg_proc_time=float(avg_v)))
            cursor += 1
        meta = {"total_proc_time": total_p, "min_total_proc_time": min_p, "total_ops": L, "op_offset": 0}
        if t_arrive is not None: meta["t_arrive"] = float(t_arrive)
        if due_dates is not None: meta["due_date"] = float(due_dates[j])
        jobs.append(JobSpec(job_id=base_job_id + j, operations=ops, meta=meta))
    return jobs

class _TimeNormalizer:
    def __init__(self, base: float, scale: float):
        self.base, self.scale = float(base), max(float(scale), 1e-6)
    def f(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.base) / self.scale

class EventBurstGenerator:
    def __init__(self, sd2_fn: Callable, base_config, n_machines: int,
                 interarrival_mean: float = 100,
                 k_sampler: Optional[Callable] = None,
                 rng: Optional[np.random.Generator] = None,
                 starting_job_id: int = 0):
        self.sd2_fn, self.cfg, self.M = sd2_fn, base_config, int(n_machines)
        self.interarrival_mean = float(interarrival_mean)
        self.rng = rng or np.random.default_rng()
        self.k_sampler = k_sampler or (lambda _rng: 1)
        self._next_id = self._initial_id = int(starting_job_id)

    def sample_next_time(self, t_now: float) -> float:
        return float(t_now + self.rng.exponential(self.interarrival_mean))

    def generate_burst(self, t_event: float) -> List[JobSpec]:
        K = int(self.k_sampler(self.rng))
        if K <= 0: return []
        old_n_j = getattr(self.cfg, "n_j", None)
        try:
            setattr(self.cfg, "n_j", K)
            jl, pt, _ = self.sd2_fn(self.cfg, rng=self.rng)
        finally:
            if old_n_j is not None: setattr(self.cfg, "n_j", old_n_j)
        dd_rel = generate_due_dates(jl, pt, tightness=getattr(configs, "due_date_tightness", 1.2), due_date_mode='k', rng=self.rng)
        jobs = split_matrix_to_jobs(jl, pt, base_job_id=self._next_id, t_arrive=t_event, due_dates=float(t_event)+dd_rel)
        self._next_id += len(jobs)
        return jobs

    def bump_next_id(self, n: int): self._next_id = max(self._next_id, int(n))
    def reset(self): self._next_id = self._initial_id

class BatchScheduleRecorder:
    def __init__(self, batch_jobs: List[JobSpec], n_machines: int):
        self.jobs, self.M, self.rows = batch_jobs, int(n_machines), []

    def record_step(self, env: FJSPEnvForVariousOpNums, action: int):
        # Legacy method, usually not called in the new manual loop
        cj, cm = int(action // self.M), int(action % self.M)
        c_op_g = int(env.candidate[0, cj])
        off = int(self.jobs[cj].meta.get("op_offset", 0))
        op_id = int(c_op_g - env.job_first_op_id[0, cj])
        start = float(max(env.true_candidate_free_time[0, cj], env.true_mch_free_time[0, cm]))
        state, _, done, info = env.step(np.array([action]))
        end = float(env.true_op_ct[0, c_op_g])
        self.rows.append({"job": int(self.jobs[cj].job_id), "op": op_id + off, "machine": cm, "start": start, "end": end, "duration": end - start})
        return state, 0, done

    def record_step_manual(self, env: FJSPEnvForVariousOpNums, action: int, info: Dict):
        """[NEW] Record results using the info dictionary provided by env.step() to avoid double-stepping."""
        cj, cm = int(action // self.M), int(action % self.M)
        det = info["scheduled_op_details"]
        off = int(self.jobs[cj].meta.get("op_offset", 0))
        self.rows.append({
            "job": int(self.jobs[cj].job_id),
            "op": det["op_id_in_job"] + off,
            "machine": cm,
            "start": float(det["start_time"]),
            "end": float(det["end_time"]),
            "duration": float(det["proc_time"])
        })
    
    def to_rows(self): return list(self.rows)

class GlobalTimelineOrchestrator:
    def __init__(self, n_machines: int, job_generator: EventBurstGenerator,
                 select_from_buffer: Callable = None, t0: float = 0.0):
        self.M, self.t, self.generator = int(n_machines), float(t0), job_generator
        self.select_from_buffer = select_from_buffer # [RESTORED]
        self.buffer, self.machine_free_time = [], np.zeros(int(n_machines), dtype=float)
        self._global_rows, self._global_row_keys = [], set()
        self._last_full_rows, self._last_jobs_snapshot = [], []
        self._job_history_finishes: Dict[int, float] = {}
        self._release_count = 0 # [NEW] Track release events
        self.method = str(getattr(configs, "scheduler_type", "PPO")).upper()
        self._ppo = PPO_initialize() if self.method == "PPO" else None
        if self._ppo:
            p = getattr(configs, "ppo_model_path", None)
            if p: self._ppo.policy.load_state_dict(torch.load(p, map_location=getattr(configs, "device", "cpu"), weights_only=True))
            self._ppo.policy.eval()

    def reset(self, *, clear_buffer: bool = True, t0: float = 0.0):
        self.t, self.machine_free_time[:] = float(t0), 0.0
        if clear_buffer: self.buffer.clear()
        self._global_rows, self._global_row_keys, self._last_full_rows, self._last_jobs_snapshot, self._job_history_finishes = [], set(), [], [], {}
        self._release_count = 0 # [NEW]

    def _extend_global_rows_dedup(self, rows: List[Dict]):
        for r in rows:
            k = (int(r["job"]), int(r["op"]), int(r["machine"]), float(r["start"]), float(r["end"]))
            if k not in self._global_row_keys: 
                self._global_row_keys.add(k); self._global_rows.append(r)
                jid = int(r["job"])
                self._job_history_finishes[jid] = max(self._job_history_finishes.get(jid, 0.0), float(r["end"]))

    def solve_current_batch_static(self, env: FJSPEnvForVariousOpNums, state) -> List[Dict]:
        rec = BatchScheduleRecorder(self._committed_jobs, self.M)
        done = False
        
        # [TIMING]
        t_fwd_sum = 0.0
        t_prep_sum = 0.0
        t_f_op_sum = 0.0
        t_f_mch_sum = 0.0
        t_f_pair_sum = 0.0
        t_upd_sum = 0.0
        
        while not done:
            if self.method == "PPO":
                import time
                _sync_cuda_for_profile()
                tfwd0 = time.perf_counter()
                with torch.inference_mode():
                    pi = self._ppo.policy.policy_only(
                        fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor
                    )
                    act = int(pi.argmax(dim=1).item())
                _sync_cuda_for_profile()
                t_fwd_sum += (time.perf_counter() - tfwd0)
            else:
                from common_utils import heuristic_select_action
                act = heuristic_select_action(self.method, env)
            
            # Perform action
            state, _, done, info = env.step(np.array([act]))
            
            # Aggregate timings from the low-level step
            t_prep_sum += info.get("t_prep", 0.0)
            t_f_op_sum += info.get("t_f_op", 0.0)
            t_f_mch_sum += info.get("t_f_mch", 0.0)
            t_f_pair_sum += info.get("t_f_pair", 0.0)
            t_upd_sum += info.get("t_state_upd", 0.0)
            
            # Record for plan
            rec.record_step_manual(env, act, info)
            
        # Store aggregated timings in the orchestrator
        self._last_batch_timings = {
            "t_fwd": t_fwd_sum,
            "t_prep": t_prep_sum,
            "t_f_op": t_f_op_sum,
            "t_f_mch": t_f_mch_sum,
            "t_f_pair": t_f_pair_sum,
            "t_upd": t_upd_sum
        }
        return rec.to_rows()

    def event_release_and_reschedule(self, t_e: float) -> Dict:
        self.t = float(t_e)
        self._release_count += 1 # [NEW] Incremental release count
        H_add = [dict(r) for r in self._last_full_rows if float(r["start"]) < self.t]
        if H_add: self._extend_global_rows_dedup(H_add)
        
        # Compute MFT
        busy = np.full(self.M, self.t, dtype=float)
        for r in self._global_rows:
            if float(r["start"]) < self.t < float(r["end"]): busy[int(r["machine"])] = max(busy[int(r["machine"])], float(r["end"]))
        self.machine_free_time = busy

        # 1. Update snapshot with newly arrived jobs (full versions)
        buffer_jobs = list(self.buffer); self.buffer.clear()
        self._last_jobs_snapshot.extend(buffer_jobs)

        # 2. Filter and Slice for the current PPO Batch
        by_j = {}
        for r in self._last_full_rows: by_j.setdefault(int(r["job"]), []).append(r)
        
        jobs_new = []
        for js in self._last_jobs_snapshot:
            jid = int(js.job_id); rows = by_j.get(jid, [])
            started = [r for r in rows if float(r["start"]) < self.t]
            total_ops = int(js.meta.get("total_ops", len(js.operations)))
            
            if len(started) < total_ops:
                # Optimized replacement for deepcopy
                js_b = JobSpec(job_id=js.job_id, operations=js.operations[len(started):], meta=js.meta.copy())
                js_b.meta["op_offset"] = len(started)
                
                inprog = [r for r in rows if float(r["start"]) <= self.t < float(r["end"])]
                js_b.meta["ready_at"] = float(inprog[0]["end"]) if inprog else max(float(js.meta.get("t_arrive", 0.0)), self.t)
                jobs_new.append(js_b)
        
        if not jobs_new: return {"event": "tick", "t": self.t}
        
        jl, pt = self._build_batch(jobs_new); env = FJSPEnvForVariousOpNums(n_j=len(jobs_new), n_m=self.M)
        pt_scale = (float(configs.low) + float(configs.high)) / 2.0
        norm = _TimeNormalizer(self.t, pt_scale)
        due_dates_ppo = [norm.f(float(j.meta.get("due_date", 0.0))) for j in jobs_new]
        due_dates_abs = [float(j.meta.get("due_date", 0.0)) for j in jobs_new]
        
        state = env.set_initial_data(jl, pt, due_date_list=[due_dates_ppo], normalize_due_date=False, true_due_date_list=[due_dates_abs])
        env.true_mch_free_time[0,:] = self.machine_free_time; env.mch_free_time[0,:] = norm.f(self.machine_free_time)
        for i, js_b in enumerate(jobs_new):
            r_abs = float(js_b.meta.get("ready_at", self.t))
            env.true_candidate_free_time[0,i] = r_abs; env.candidate_free_time[0,i] = norm.f(r_abs)
            
        self._committed_jobs = jobs_new; rows = self.solve_current_batch_static(env, env.rebuild_state_from_current())
        
        # Finalize
        self.machine_free_time = env.true_mch_free_time[0].astype(float).copy()
        f_dict = {(int(r["job"]), int(r["op"])): r for r in self._last_full_rows}
        committed_job_ids = {js.job_id for js in jobs_new}
        to_del = [k for k, r in f_dict.items() if k[0] in committed_job_ids and float(r["start"]) >= self.t]
        for k in to_del: del f_dict[k]
        for r in rows: f_dict[(int(r["job"]), int(r["op"]))] = r
        self._last_full_rows = sorted(list(f_dict.values()), key=lambda x: (x["job"], x["op"]))
        
        fins = set()
        for jid in {js.job_id for js in self._last_jobs_snapshot}:
            j_rows = [r for r in self._last_full_rows if int(r["job"]) == jid]
            if j_rows and max(float(r["end"]) for r in j_rows) <= self.t: fins.add(jid)
        self._last_jobs_snapshot = [js for js in self._last_jobs_snapshot if js.job_id not in fins]
        
        return {"event": "batch_finalized", "t": self.t, "rows": rows}

    def tick_without_release(self, t_e: float) -> Dict:
        self.t = float(t_e)
        H_add = [dict(r) for r in self._last_full_rows if float(r["start"]) < self.t]
        if H_add: self._extend_global_rows_dedup(H_add)
        busy = np.full(self.M, self.t, dtype=float)
        for r in self._last_full_rows: busy[int(r["machine"])] = max(busy[int(r["machine"])], float(r["end"]))
        self.machine_free_time = busy
        return {"event": "hold_tick", "t": self.t}

    def compute_interval_metrics(self, t0: float, t1: float) -> Dict:
        t0, t1, dt = float(t0), float(t1), max(0.0, float(t1-t0))
        # Optimized: Only scan recent history and current plan
        busy_m = np.zeros(self.M)
        for r in self._global_rows[-len(self._last_full_rows)*2:]: # Heuristic: only last few history items matter
            if float(r["end"]) < t0: continue
            busy_m[int(r["machine"])] += max(0.0, min(float(r["end"]), t1) - max(float(r["start"]), t0))
        for r in self._last_full_rows:
            busy_m[int(r["machine"])] += max(0.0, min(float(r["end"]), t1) - max(float(r["start"]), t0))
        return {"total_idle": max(0.0, dt * self.M - np.sum(np.minimum(busy_m, dt))), "interval_dt": dt}

    def compute_idle_stats(self, t_now: float, horizon: float) -> Tuple[float, float]:
        if horizon <= 1e-9:
            return 0.0, 0.0
        t_end, m_ints = t_now + horizon, [[] for _ in range(self.M)]
        for r in self._last_full_rows:
            s, e = max(t_now, float(r["start"])), min(t_end, float(r["end"]))
            if e > s: m_ints[int(r["machine"])].append((s, e))
        total_weighted = 0.0
        total_gap = 0.0
        for m in range(self.M):
            ints = sorted(m_ints[m], key=lambda x: x[0]); merged = []
            if ints:
                cs, ce = ints[0]
                for ns, ne in ints[1:]:
                    if ns < ce: ce = max(ce, ne)
                    else: merged.append((cs, ce)); cs, ce = ns, ne
                merged.append((cs, ce))
            ptr = t_now
            for s, e in merged:
                if s > ptr:
                    total_gap += (s - ptr)
                    total_weighted += (s - ptr) - ((s - t_now)**2 - (ptr - t_now)**2) / (2.0 * horizon)
                ptr = max(ptr, e)
            if ptr < t_end:
                total_gap += (t_end - ptr)
                total_weighted += (t_end - ptr) - ((t_end - t_now)**2 - (ptr - t_now)**2) / (2.0 * horizon)
        return total_weighted / self.M, total_gap / self.M

    def compute_weighted_idle(self, t_now: float, horizon: float) -> float:
        return self.compute_idle_stats(t_now, horizon)[0]

    def compute_unweighted_idle(self, t_now: float, horizon: float) -> float:
        return self.compute_idle_stats(t_now, horizon)[1]

    def get_wip_stats(self, t_now: float) -> Dict[str, float]:
        slacks, n_tardy, n_act, p_td, total_rem_w = [], 0, 0, 0.0, 0.0
        if not self._last_jobs_snapshot: return {"wip_min_slack": 0.0, "wip_avg_slack": 0.0, "wip_tardy_ratio": 0.0, "planned_td": 0.0, "total_rem_work": 0.0, "wip_slack_std": 0.0, "wip_count": 0}
        
        job_rows = {}
        for r in self._last_full_rows: job_rows.setdefault(int(r["job"]), []).append(r)
        
        for js in self._last_jobs_snapshot:
            jid = int(js.job_id); rows = job_rows.get(jid, [])
            if not rows: # New buffer jobs
                n_act += 1; rem_w = 0.0
                for op in js.operations:
                    rem_w += float(op.avg_proc_time)
                total_rem_w += rem_w
                due = float(js.meta.get("due_date", 0.0))
                p_td += max(0.0, t_now + rem_w - due)
                slacks.append(due - (t_now + rem_w))
                continue
            
            # Check if job is truly finished (last planned op ended before t_now)
            p_finish = max(float(r["end"]) for r in rows)
            if p_finish <= t_now: continue
            
            # This is an active WIP job
            n_act += 1
            # [FIX] js.operations is ALREADY sliced in event_release_and_reschedule.
            # Do NOT slice it again with len(started_rows).
            rem_w = 0.0
            for op in js.operations:
                rem_w += float(op.avg_proc_time)
            total_rem_w += rem_w
            
            # Planned TD is based on the FINAL operation's end time in current PPO global plan
            due = float(js.meta.get("due_date", 0.0))
            p_td += max(0.0, p_finish - due)
            
            # Slack based on current time + remaining work
            slacks.append(due - (t_now + rem_w))
            if t_now > due: n_tardy += 1
            
            # [DEBUG] Track Job 1 specifically if needed
            # if jid == 1: print(f"  [DEBUG WIP] Job 1: t_now={t_now:.1f}, p_finish={p_finish:.1f}, rem_w={rem_w:.1f}, due={due:.1f}")

        if n_act == 0: return {"wip_min_slack": 0.0, "wip_avg_slack": 0.0, "wip_tardy_ratio": 0.0, "planned_td": 0.0, "total_rem_work": 0.0, "wip_slack_std": 0.0, "wip_count": 0}
        return {"wip_min_slack": min(slacks), "wip_avg_slack": np.mean(slacks), "wip_tardy_ratio": n_tardy/n_act, "planned_td": p_td, "total_rem_work": total_rem_w, "wip_slack_std": float(np.std(slacks)), "wip_count": n_act}

    def get_final_kpi_stats(self, all_due: Dict[int, float]) -> Dict[str, float]:
        fins = self._job_history_finishes.copy()
        for r in self._last_full_rows: fins[int(r["job"])] = max(fins.get(int(r["job"]), 0.0), float(r["end"]))
        total_td = sum(max(0.0, fins.get(jid, 0.0) - due) for jid, due in all_due.items())
        return {"makespan": float(np.max(self.machine_free_time)), "tardiness": total_td}

    def get_total_tardiness_estimate(self) -> float:
        if not self._last_jobs_snapshot: return 0.0
        fins = self._job_history_finishes.copy()
        for r in self._last_full_rows: fins[int(r["job"])] = max(fins.get(int(r["job"]), 0.0), float(r["end"]))
        return sum(max(0.0, fins.get(int(js.job_id), 0.0) - float(js.meta.get("due_date", 0.0))) for js in self._last_jobs_snapshot)

    def _build_batch(self, jobs: List[JobSpec]):
        jl = np.array([len(j.operations) for j in jobs], dtype=int)
        pt = np.zeros((int(jl.sum()), self.M), dtype=float); row = 0
        for j in jobs:
            for op in j.operations:
                tr = np.asarray(op.time_row, dtype=float)
                pt[row, :] = np.where(tr > 0, tr, 0.0); row += 1
        return [jl], [pt]
