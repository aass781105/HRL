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
            ops.append(OperationSpec(time_row=row.astype(float).tolist()))
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
        cj, cm = int(action // self.M), int(action % self.M)
        c_op_g = int(env.candidate[0, cj])
        off = int(self.jobs[cj].meta.get("op_offset", 0))
        op_id = int(c_op_g - env.job_first_op_id[0, cj])
        start = float(max(env.true_candidate_free_time[0, cj], env.true_mch_free_time[0, cm]))
        state, _, done, _ = env.step(np.array([action]))
        end = float(env.true_op_ct[0, c_op_g])
        self.rows.append({"job": int(self.jobs[cj].job_id), "op": op_id + off, "machine": cm, "start": start, "end": end, "duration": end - start})
        return state, 0, done
    
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
        self.method = str(getattr(configs, "scheduler_type", "PPO")).upper()
        self._ppo = PPO_initialize() if self.method == "PPO" else None
        if self._ppo:
            p = getattr(configs, "ppo_model_path", None)
            if p: self._ppo.policy.load_state_dict(torch.load(p, map_location=getattr(configs, "device", "cpu")))
            self._ppo.policy.eval()

    def clone(self) -> GlobalTimelineOrchestrator:
        """[NEW] Returns a deep clone of the orchestrator state, sharing the PPO reference."""
        new_orch = GlobalTimelineOrchestrator(self.M, self.generator, self.select_from_buffer, self.t)
        new_orch.buffer = copy.deepcopy(self.buffer)
        new_orch.machine_free_time = self.machine_free_time.copy()
        new_orch._global_rows = copy.deepcopy(self._global_rows)
        new_orch._global_row_keys = self._global_row_keys.copy()
        new_orch._last_full_rows = copy.deepcopy(self._last_full_rows)
        new_orch._last_jobs_snapshot = copy.deepcopy(self._last_jobs_snapshot)
        new_orch._job_history_finishes = self._job_history_finishes.copy()
        new_orch._ppo = self._ppo # Share reference
        return new_orch

    def reset(self, *, clear_buffer: bool = True, t0: float = 0.0):
        self.t, self.machine_free_time[:] = float(t0), 0.0
        if clear_buffer: self.buffer.clear()
        self._global_rows, self._global_row_keys, self._last_full_rows, self._last_jobs_snapshot, self._job_history_finishes = [], set(), [], [], {}

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
        while not done:
            if self.method == "PPO":
                with torch.no_grad():
                    pi, _ = self._ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                                             fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
                    act = int(pi.argmax(dim=1).item())
            else: act = heuristic_select_action(self.method, env)
            state, _, done = rec.record_step(env, act)
        return rec.to_rows()

    def event_release_and_reschedule(self, t_e: float) -> Dict:
        self.t = float(t_e)
        H_add = [dict(r) for r in self._last_full_rows if float(r["start"]) < self.t]
        if H_add: self._extend_global_rows_dedup(H_add)
        
        # Compute MFT
        busy = np.full(self.M, self.t, dtype=float)
        for r in self._global_rows:
            if float(r["start"]) < self.t < float(r["end"]): busy[int(r["machine"])] = max(busy[int(r["machine"])], float(r["end"]))
        self.machine_free_time = busy

        # Filter WIP
        by_j = {}
        for r in self._last_full_rows: by_j.setdefault(int(r["job"]), []).append(r)
        rem_j = []
        for js in self._last_jobs_snapshot:
            rows = by_j.get(int(js.job_id), [])
            started = [r for r in rows if float(r["start"]) < self.t]
            if len(started) < len(js.operations):
                js_b = copy.deepcopy(js); js_b.meta["op_offset"] = len(started); js_b.operations = js.operations[len(started):]
                js_b.meta["ready_at"] = float(started[-1]["end"]) if started else self.t
                rem_j.append(js_b)
        
        jobs_new = rem_j + list(self.buffer); self.buffer.clear()
        if not jobs_new: return {"event": "tick", "t": self.t}
        
        jl, pt = self._build_batch(jobs_new); env = FJSPEnvForVariousOpNums(n_j=len(jobs_new), n_m=self.M)
        pt_d = pt[0]; norm = _TimeNormalizer(self.t, max(1.0, np.percentile(pt_d[pt_d>0], 95)*4.0 if pt_d[pt_d>0].size else 50.0))
        state = env.set_initial_data(jl, pt)
        env.true_mch_free_time[0,:] = self.machine_free_time; env.mch_free_time[0,:] = norm.f(self.machine_free_time)
        for i, js in enumerate(jobs_new):
            r_abs = max(float(js.meta.get("ready_at", 0.0)), self.t)
            env.true_candidate_free_time[0,i] = r_abs; env.candidate_free_time[0,i] = norm.f(r_abs)
        self._committed_jobs = jobs_new; rows = self.solve_current_batch_static(env, env.rebuild_state_from_current())
        
        # Finalize
        self.machine_free_time = env.true_mch_free_time[0].astype(float).copy()
        f_dict = {(int(r["job"]), int(r["op"])): r for r in self._last_full_rows}
        for r in rows: f_dict[(int(r["job"]), int(r["op"]))] = r
        self._last_full_rows = sorted(list(f_dict.values()), key=lambda x: (x["job"], x["op"]))
        c_ids = {js.job_id for js in jobs_new}
        self._last_jobs_snapshot = [js for js in self._last_jobs_snapshot if js.job_id not in c_ids] + jobs_new
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

    def compute_weighted_idle(self, t_now: float, horizon: float) -> float:
        if horizon <= 1e-9: return 0.0
        t_end, m_ints = t_now + horizon, [[] for _ in range(self.M)]
        for r in self._last_full_rows:
            s, e = max(t_now, float(r["start"])), min(t_end, float(r["end"]))
            if e > s: m_ints[int(r["machine"])].append((s, e))
        total = 0.0
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
                if s > ptr: total += (s - ptr) - ((s - t_now)**2 - (ptr - t_now)**2) / (2.0 * horizon)
                ptr = max(ptr, e)
            if ptr < t_end: total += (t_end - ptr) - ((t_end - t_now)**2 - (ptr - t_now)**2) / (2.0 * horizon)
        return total / self.M

    def compute_unweighted_idle(self, t_now: float, horizon: float) -> float:
        """[NEW] Returns average gap per machine without weighting."""
        if horizon <= 1e-9: return 0.0
        t_end, m_ints = t_now + horizon, [[] for _ in range(self.M)]
        for r in self._last_full_rows:
            s, e = max(t_now, float(r["start"])), min(t_end, float(r["end"]))
            if e > s: m_ints[int(r["machine"])].append((s, e))
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
                if s > ptr: total_gap += (s - ptr)
                ptr = max(ptr, e)
            if ptr < t_end: total_gap += (t_end - ptr)
        return total_gap / self.M

    def get_wip_stats(self, t_now: float) -> Dict[str, float]:
        slacks, n_tardy, n_act, p_td, t_td = [], 0, 0, 0.0, 0.0
        if not self._last_jobs_snapshot: return {"wip_min_slack": 0.0, "wip_avg_slack": 0.0, "wip_tardy_ratio": 0.0, "planned_td": 0.0, "theoretical_td": 0.0, "wip_slack_std": 0.0, "wip_count": 0}
        job_rows = {}
        for r in self._last_full_rows: job_rows.setdefault(int(r["job"]), []).append(r)
        for js in self._last_jobs_snapshot:
            jid = int(js.job_id); rows = job_rows.get(jid, [])
            if rows and int(js.meta.get("total_ops", 0)) == len(rows) and max(float(r["end"]) for r in rows) <= t_now: continue
            n_act += 1; started = [r for r in rows if float(r["start"]) < t_now]; rem_w = 0.0
            for op in js.operations[len(started):]:
                v = np.array(op.time_row); rem_w += np.mean(v[v>0]) if v[v>0].size else 0.0
            due = float(js.meta.get("due_date", 0.0))
            t_td += max(0.0, t_now + rem_w - due); p_td += max(0.0, (max(float(r["end"]) for r in rows) if rows else t_now + rem_w) - due)
            slacks.append(due - t_now - rem_w)
            if t_now > due: n_tardy += 1
        if n_act == 0: return {"wip_min_slack": 0.0, "wip_avg_slack": 0.0, "wip_tardy_ratio": 0.0, "planned_td": 0.0, "theoretical_td": 0.0, "wip_slack_std": 0.0, "wip_count": 0}
        return {"wip_min_slack": min(slacks), "wip_avg_slack": np.mean(slacks), "wip_tardy_ratio": n_tardy/n_act, "planned_td": p_td, "theoretical_td": t_td, "wip_slack_std": float(np.std(slacks)), "wip_count": n_act}

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
