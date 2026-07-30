from __future__ import annotations
"""
Event-driven rolling-horizon orchestrator for FJSP (HIGH PERFORMANCE)
=====================================================================
Optimizations:
- Eliminated redundant list concatenations in state/reward calculations.
- Incremental history tracking for KPI stats.
- Efficient interval metric calculation (only scans active rows).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
import copy
import numpy as np

from ll_fjsp_env import LLFJSPEnv
import torch
from model.ll_ppo import ll_ppo_initialize
from params import configs
from common_utils import heuristic_select_action, greedy_select_action, sample_action, resolve_lower_level_weight_path
from data_utils import generate_due_dates
from hl_env_scenarios import apply_bottleneck_orders


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
                         due_dates: Optional[np.ndarray] = None,
                         job_info: Optional[Dict] = None) -> List[JobSpec]:
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
        if job_info is not None:
            if "is_urgent" in job_info:
                meta["is_urgent"] = bool(np.asarray(job_info["is_urgent"])[j])
            if "due_date_k" in job_info:
                meta["due_date_k"] = float(np.asarray(job_info["due_date_k"])[j])
            if "job_work" in job_info:
                meta["job_work"] = float(np.asarray(job_info["job_work"])[j])
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
        # Dynamic world should use fixed 5 operations per job.
        setattr(self.cfg, "op_per_job", 5)
        setattr(self.cfg, "enable_op_mixture", False)
        self.interarrival_mean = float(interarrival_mean)
        self.arrival_mode = str(getattr(base_config, "arrival_mode", "exponential")).strip().lower()
        self.interarrival_uniform_low = float(getattr(base_config, "interarrival_uniform_low", 10.0))
        self.interarrival_uniform_high = float(getattr(base_config, "interarrival_uniform_high", 50.0))
        if self.interarrival_uniform_low > self.interarrival_uniform_high:
            self.interarrival_uniform_low, self.interarrival_uniform_high = self.interarrival_uniform_high, self.interarrival_uniform_low
        self.rng = rng or np.random.default_rng()
        self.k_sampler = k_sampler or (lambda _rng: 1)
        self._next_id = self._initial_id = int(starting_job_id)

    def sample_next_time(self, t_now: float) -> float:
        if self.arrival_mode == "uniform":
            interval = self.rng.uniform(self.interarrival_uniform_low, self.interarrival_uniform_high)
        else:
            interval = self.rng.exponential(self.interarrival_mean)
        return float(t_now + max(float(interval), 0.0))

    def generate_burst(self, t_event: float) -> List[JobSpec]:
        K = int(self.k_sampler(self.rng))
        if K <= 0: return []
        same_batch = bool(getattr(self.cfg, "hl_same_batch_jobs", False)) and K > 1
        old_n_j = getattr(self.cfg, "n_j", None)
        try:
            setattr(self.cfg, "n_j", K)
            jl, pt, _ = self.sd2_fn(self.cfg, rng=self.rng)
        finally:
            if old_n_j is not None: setattr(self.cfg, "n_j", old_n_j)

        if same_batch:
            # Treat a burst as one repeated job template, not K independent jobs.
            # The job IDs remain different, but every operation keeps the same
            # machine eligibility and processing-time row.
            jl_arr = np.asarray(jl, dtype=int).reshape(-1)
            pt_arr = np.asarray(pt, dtype=float)
            template_len = int(jl_arr[0])
            template_pt = np.array(pt_arr[:template_len], copy=True)
            jl = np.full(K, template_len, dtype=int)
            pt = np.tile(template_pt, (K, 1))

        dd_rel, due_info = generate_due_dates(
            jl,
            pt,
            tightness=getattr(self.cfg, "hl_due_date_tightness", 1.2),
            due_date_mode='mix_urgent_normal',
            rng=self.rng,
            return_info=True,
            due_config=self.cfg,
        )
        if same_batch:
            # Keep due date, urgency, k, and work metadata identical as well.
            dd_rel = np.full(K, float(np.asarray(dd_rel).reshape(-1)[0]), dtype=float)
            due_info = {
                key: np.full(
                    K,
                    np.asarray(value).reshape(-1)[0],
                    dtype=np.asarray(value).dtype,
                )
                for key, value in due_info.items()
            }

        burst_due_alpha = float(getattr(self.cfg, "hl_burst_due_date_scale_alpha", 0.0))
        if burst_due_alpha != 0.0 and K > 1:
            # Relax larger bursts gradually to reduce seed variance caused by
            # one event receiving several urgent jobs at once.
            max_burst = max(2, int(getattr(self.cfg, "hl_burst_size_high", 5)))
            burst_factor = 1.0 + burst_due_alpha * (K - 1) / float(max_burst - 1)
            dd_rel = np.asarray(dd_rel, dtype=float) * burst_factor
            if "due_date_k" in due_info:
                due_info["due_date_k"] = (
                    np.asarray(due_info["due_date_k"], dtype=float) * burst_factor
                )

        jobs = split_matrix_to_jobs(jl, pt, base_job_id=self._next_id, t_arrive=t_event, due_dates=float(t_event)+dd_rel, job_info=due_info)
        if same_batch:
            # Apply scenario transformations once to the template, then clone
            # the transformed job so bottleneck selection cannot diverge inside
            # the same burst.
            jobs = apply_bottleneck_orders([jobs[0]], self.cfg, self.rng, self.M)
            template = jobs[0]
            for job_id in range(1, K):
                jobs.append(JobSpec(
                    job_id=self._next_id + job_id,
                    operations=copy.deepcopy(template.operations),
                    meta=copy.deepcopy(template.meta),
                ))
        else:
            jobs = apply_bottleneck_orders(jobs, self.cfg, self.rng, self.M)
        self._next_id += len(jobs)
        return jobs

    def bump_next_id(self, n: int): self._next_id = max(self._next_id, int(n))
    def reset(self): self._next_id = self._initial_id

class BatchScheduleRecorder:
    def __init__(self, batch_jobs: List[JobSpec], n_machines: int):
        self.jobs, self.M, self.rows = batch_jobs, int(n_machines), []

    def record_step(self, env: LLFJSPEnv, action: int):
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

    def record_step_manual(self, env: LLFJSPEnv, action: int, info: Dict):
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
        self.last_batch_manifest = None
        self.last_batch_rows = []
        self.method = str(getattr(configs, "scheduler_type", "PPO")).upper()
        self._ppo = ll_ppo_initialize() if self.method == "PPO" else None
        if self._ppo:
            p = getattr(configs, "ll_ppo_model_path", None)
            if p:
                p = resolve_lower_level_weight_path(p, getattr(configs, "data_source", "SD2"))
                self._ppo.policy.load_state_dict(torch.load(p, map_location=getattr(configs, "device", "cpu"), weights_only=True))
            self._ppo.policy.eval()

    def reset(self, *, clear_buffer: bool = True, t0: float = 0.0):
        self.t, self.machine_free_time[:] = float(t0), 0.0
        if clear_buffer: self.buffer.clear()
        self._global_rows, self._global_row_keys, self._last_full_rows, self._last_jobs_snapshot, self._job_history_finishes = [], set(), [], [], {}
        self._release_count = 0 # [NEW]
        self.last_batch_manifest = None
        self.last_batch_rows = []

    def _extend_global_rows_dedup(self, rows: List[Dict]):
        for r in rows:
            k = (int(r["job"]), int(r["op"]), int(r["machine"]), float(r["start"]), float(r["end"]))
            if k not in self._global_row_keys: 
                self._global_row_keys.add(k); self._global_rows.append(r)
                jid = int(r["job"])
                self._job_history_finishes[jid] = max(self._job_history_finishes.get(jid, 0.0), float(r["end"]))

    def _build_last_batch_manifest(
        self,
        *,
        jobs_new: List[JobSpec],
        batch_time: float,
        pt_scale: float,
        jl,
        pt,
        due_dates_state_abs,
        due_dates_abs,
        event_id: Optional[int] = None,
    ) -> Dict:
        batch_time = float(batch_time)
        pt_scale = max(float(pt_scale), 1e-6)
        norm = _TimeNormalizer(batch_time, pt_scale)
        job_length_list = [int(x) for x in np.asarray(jl[0], dtype=int).tolist()] if jl else []
        op_pt_list = np.asarray(pt[0], dtype=float).tolist() if pt else []
        due_dates_abs = [float(x) for x in due_dates_abs]
        due_dates_state_abs = [float(x) for x in due_dates_state_abs]
        ready_times_abs = [float(js.meta.get("ready_at", batch_time)) for js in jobs_new]
        ready_times_rel = [float(x - batch_time) for x in ready_times_abs]
        ready_times_env = [float(norm.f(x)) for x in ready_times_abs]

        jobs_payload = []
        for batch_idx, js in enumerate(jobs_new):
            op_offset = int(js.meta.get("op_offset", 0))
            total_ops = int(js.meta.get("total_ops", op_offset + len(js.operations)))
            arrive_abs = float(js.meta.get("t_arrive", batch_time))
            arrive_rel = float(arrive_abs - batch_time)
            ready_abs = float(js.meta.get("ready_at", batch_time))
            due_abs = float(js.meta.get("due_date", 0.0))
            ops_payload = []
            for local_op_idx, op in enumerate(js.operations):
                machine_times = {}
                if op.time_row is not None:
                    for m_idx, pt_val in enumerate(op.time_row):
                        if float(pt_val) > 0:
                            machine_times[str(int(m_idx))] = float(pt_val)
                elif op.machine_times is not None:
                    for m_idx, pt_val in op.machine_times.items():
                        if float(pt_val) > 0:
                            machine_times[str(int(m_idx))] = float(pt_val)
                ops_payload.append(
                    {
                        "local_op_id": int(local_op_idx),
                        "global_op_id": int(op_offset + local_op_idx),
                        "machine_times": machine_times,
                    }
                )

            jobs_payload.append(
                {
                    "batch_job_index": int(batch_idx),
                    "job_id": int(js.job_id),
                    "started_ops": int(op_offset),
                    "remaining_ops": int(len(js.operations)),
                    "total_ops": int(total_ops),
                    "job_length": int(len(js.operations)),
                    "t_arrive_abs": arrive_abs,
                    "t_arrive_rel": arrive_rel,
                    "ready_at_abs": ready_abs,
                    "ready_at_rel": float(ready_abs - batch_time),
                    "ready_at_env": float(norm.f(ready_abs)),
                    "due_date_abs": due_abs,
                    "due_date_rel": float(due_abs - batch_time),
                    "due_date_env": float(norm.f(due_abs)),
                    "is_urgent": bool(js.meta.get("is_urgent", False)),
                    "due_date_k": float(js.meta.get("due_date_k", 0.0)),
                    "operations": ops_payload,
                }
            )

        manifest = {
            "event": "batch_finalized",
            "event_id": int(event_id) if event_id is not None else None,
            "batch_time_abs": batch_time,
            "release_index": int(self._release_count),
            "pt_scale": pt_scale,
            "n_jobs": int(len(jobs_new)),
            "n_ops": int(sum(len(j.operations) for j in jobs_new)),
            "machine_free_time_abs": [float(x) for x in np.asarray(self.machine_free_time, dtype=float).tolist()],
            "machine_free_time_env": [float(x) for x in np.asarray(norm.f(self.machine_free_time), dtype=float).tolist()],
            "job_length_list": [job_length_list],
            "op_pt_list": [op_pt_list],
            "due_date_list": [due_dates_state_abs],
            "true_due_date_list": [due_dates_abs],
            "candidate_free_time_abs": ready_times_abs,
            "candidate_free_time_env": ready_times_env,
            "candidate_ready_rel": ready_times_rel,
            "jobs": jobs_payload,
        }
        return manifest

    def solve_current_batch_static(self, env: LLFJSPEnv, state) -> Tuple[List[Dict], int]:
        K = env.number_of_envs
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
                    if K > 1:
                        # Multi-path rollout forces sampling mode
                        action_tensor, _ = sample_action(pi)
                        act = action_tensor.squeeze(-1).cpu().numpy()  # shape (K,)
                    else:
                        if str(getattr(configs, "ll_eval_action_selection", "greedy")).lower() == "sample":
                            action_tensor, _ = sample_action(pi)
                            act = int(action_tensor.item())
                        else:
                            act = int(pi.argmax(dim=1).item())
                _sync_cuda_for_profile()
                t_fwd_sum += (time.perf_counter() - tfwd0)
            else:
                if self.method in ("OR-TOOLS", "ORTOOLS", "OR_TOOLS"):
                    raise RuntimeError(
                        "scheduler_type=OR-Tools is not supported by the generic heuristic/PPO orchestrator path. "
                        "Use ortools_tools/dynamic/run_dynamic_ortools_cadence.py for OR-Tools cadence scheduling, or set scheduler_type "
                        "to PPO/SPT/MWKR/FIFO for this path."
                    )
                from common_utils import heuristic_select_action
                if K > 1:
                    act = np.array([heuristic_select_action(self.method, env)] * K)
                else:
                    act = heuristic_select_action(self.method, env)
            
            # Perform action
            if K > 1:
                state, _, done_flag, info = env.step(act)
                done = done_flag.all()
            else:
                state, _, done_flag, info = env.step(np.array([act]))
                done = bool(done_flag[0])
            
            # Aggregate timings from the low-level step
            t_prep_sum += info.get("t_prep", 0.0)
            t_f_op_sum += info.get("t_f_op", 0.0)
            t_f_mch_sum += info.get("t_f_mch", 0.0)
            t_f_pair_sum += info.get("t_f_pair", 0.0)
            t_upd_sum += info.get("t_state_upd", 0.0)
            
        # Store aggregated timings in the orchestrator
        self._last_batch_timings = {
            "t_fwd": t_fwd_sum,
            "t_prep": t_prep_sum,
            "t_f_op": t_f_op_sum,
            "t_f_mch": t_f_mch_sum,
            "t_f_pair": t_f_pair_sum,
            "t_upd": t_upd_sum
        }
        
        # Evaluate to pick the best rollout k using 0.5 * MK + 0.5 * TD
        best_k = 0
        if K > 1:
            scores = []
            for k in range(K):
                makespan = env.true_mch_free_time[k].max() - self.t
                tardiness = env.accumulated_tardiness[k]
                score = 0.5 * float(makespan) + 0.5 * float(tardiness)
                scores.append(score)
            best_k = int(np.argmin(scores))
            
        # Post-hoc reconstruction of best_k's plan rows
        rows = []
        for j_idx, js in enumerate(self._committed_jobs):
            start_op = int(env.job_first_op_id[best_k, j_idx])
            end_op = int(env.job_last_op_id[best_k, j_idx])
            op_offset = int(js.meta.get("op_offset", 0))
            
            for op_id in range(start_op, end_op + 1):
                mch = int(env.op_assigned_mch[best_k, op_id])
                ct = float(env.true_op_ct[best_k, op_id])
                pt = float(env.true_op_pt[best_k, op_id, mch])
                st = ct - pt
                
                rows.append({
                    "job": int(js.job_id),
                    "op": int(op_id - start_op + op_offset),
                    "machine": mch,
                    "start": st,
                    "end": ct,
                    "duration": pt
                })
        rows = sorted(rows, key=lambda x: (x["job"], x["op"]))
        sub_makespan = float(env.true_mch_free_time[best_k].max())
        sub_tardiness = float(env.accumulated_tardiness[best_k])
        return rows, best_k

    def event_release_and_reschedule(self, t_e: float, event_id: Optional[int] = None) -> Dict:
        self.t = float(t_e)
        self.current_event_id = event_id
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
        
        if not jobs_new:
            self.last_batch_manifest = None
            self.last_batch_rows = []
            return {"event": "tick", "t": self.t}
        
        K = int(getattr(configs, "ll_rollout_k", 1))
        jl, pt = self._build_batch(jobs_new)
        jl_duplicated = jl * K
        pt_duplicated = pt * K
        
        pt_scale = (float(configs.low) + float(configs.high)) / 2.0
        norm = _TimeNormalizer(self.t, pt_scale)
        due_dates_abs = [float(j.meta.get("due_date", 0.0)) for j in jobs_new]
        due_dates_state_rel = [float(due - self.t) for due in due_dates_abs]
        due_dates_state_rel_duplicated = [due_dates_state_rel] * K
        due_dates_abs_duplicated = [due_dates_abs] * K

        self.last_batch_manifest = self._build_last_batch_manifest(
            jobs_new=jobs_new,
            batch_time=self.t,
            pt_scale=pt_scale,
            jl=jl,
            pt=pt,
            due_dates_state_abs=due_dates_state_rel,
            due_dates_abs=due_dates_abs,
            event_id=event_id,
        )
        
        env = LLFJSPEnv(n_j=len(jobs_new), n_m=self.M)
        state = env.set_initial_data(
            jl_duplicated, 
            pt_duplicated, 
            due_date_list=due_dates_state_rel_duplicated, 
            true_due_date_list=due_dates_abs_duplicated
        )
        env.true_mch_free_time[:, :] = self.machine_free_time
        env.mch_free_time[:, :] = norm.f(self.machine_free_time)
        if bool(getattr(configs, "enable_gap_insertion", False)):
            fixed_intervals_single = [[] for _ in range(self.M)]
            for r in self._global_rows:
                if float(r["start"]) < self.t < float(r["end"]):
                    fixed_intervals_single[int(r["machine"])].append((float(r["start"]), float(r["end"])))
            fixed_intervals = [copy.deepcopy(fixed_intervals_single) for _ in range(K)]
            env.set_fixed_machine_intervals(fixed_intervals, base_time=[self.t] * K)
        for i, js_b in enumerate(jobs_new):
            r_abs = float(js_b.meta.get("ready_at", self.t))
            env.true_candidate_free_time[:, i] = r_abs
            env.candidate_free_time[:, i] = norm.f(r_abs)
            
        self._committed_jobs = jobs_new; rows, best_k = self.solve_current_batch_static(env, env.rebuild_state_from_current())
        self.last_batch_rows = [dict(r) for r in rows]
        
        # Finalize
        self.machine_free_time = env.true_mch_free_time[best_k].astype(float).copy()
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
        
        sub_makespan = float(env.true_mch_free_time[best_k].max())
        sub_tardiness = float(env.accumulated_tardiness[best_k])
        return {
            "event": "batch_finalized",
            "t": self.t,
            "rows": rows,
            "jobs_count": len(jobs_new),
            "operations_count": len(rows),
            "K": K,
            "sub_makespan": sub_makespan,
            "sub_tardiness": sub_tardiness,
        }

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
        if not self._last_jobs_snapshot:
            return {
                "wip_min_slack": 0.0,
                "wip_avg_slack": 0.0,
                "wip_tardy_ratio": 0.0,
                "planned_td": 0.0,
                "total_rem_work": 0.0,
                "wip_slack_std": 0.0,
                "wip_count": 0,
            }
        
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

        if n_act == 0:
            return {
                "wip_min_slack": 0.0,
                "wip_avg_slack": 0.0,
                "wip_tardy_ratio": 0.0,
                "planned_td": 0.0,
                "total_rem_work": 0.0,
                "wip_slack_std": 0.0,
                "wip_count": 0,
            }
        return {
            "wip_min_slack": min(slacks),
            "wip_avg_slack": np.mean(slacks),
            "wip_tardy_ratio": n_tardy/n_act,
            "planned_td": p_td,
            "total_rem_work": total_rem_w,
            "wip_slack_std": float(np.std(slacks)),
            "wip_count": n_act,
        }

    def get_final_kpi_stats(self, all_due: Dict[int, float]) -> Dict[str, float]:
        fins = self._job_history_finishes.copy()
        for r in self._last_full_rows: fins[int(r["job"])] = max(fins.get(int(r["job"]), 0.0), float(r["end"]))
        total_td = sum(max(0.0, fins.get(jid, 0.0) - due) for jid, due in all_due.items())
        return {"makespan": float(np.max(self.machine_free_time)), "tardiness": total_td}

    def get_total_tardiness_estimate(self, all_due: Dict[int, float]) -> float:
        fins = self._job_history_finishes.copy()
        for r in self._last_full_rows:
            fins[int(r["job"])] = max(fins.get(int(r["job"]), 0.0), float(r["end"]))
        return sum(max(0.0, fins.get(int(jid), 0.0) - float(due)) for jid, due in all_due.items())

    def _build_batch(self, jobs: List[JobSpec]):
        jl = np.array([len(j.operations) for j in jobs], dtype=int)
        pt = np.zeros((int(jl.sum()), self.M), dtype=float); row = 0
        for j in jobs:
            for op in j.operations:
                tr = np.asarray(op.time_row, dtype=float)
                pt[row, :] = np.where(tr > 0, tr, 0.0); row += 1
        return [jl], [pt]
