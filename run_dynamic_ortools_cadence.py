import os
import time
import csv
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from params import configs
from global_env import GlobalTimelineOrchestrator, JobSpec
from dynamic_job_stream import create_dynamic_world, sample_initial_jobs
from model.gate_state import calculate_gate_state
from gantt import plot_global_gantt


def scale_time(value: float, time_scale: int) -> int:
    return int(round(float(value) * max(1, int(time_scale))))


def unscale_time(value: int, time_scale: int) -> float:
    return float(value) / float(max(1, int(time_scale)))


def compute_planned_release_count(max_events: int, cadence: int, has_init_release: bool) -> int:
    event_releases = 0 if int(max_events) <= 0 else int((int(max_events) + int(cadence) - 1) // int(cadence))
    return int(event_releases + (1 if has_init_release else 0))


class ORToolsSolveBudget:
    def __init__(self, total_budget_seconds: float, planned_releases: int):
        self.total_budget_seconds = max(0.0, float(total_budget_seconds))
        self.planned_releases = max(0, int(planned_releases))
        self.used_releases = 0
        self.consumed_seconds = 0.0

    def next_time_limit(self, fallback_limit: float) -> float:
        if self.total_budget_seconds <= 0.0:
            return float(fallback_limit)
        remaining_budget = max(0.0, self.total_budget_seconds - self.consumed_seconds)
        remaining_releases = max(1, self.planned_releases - self.used_releases)
        return float(remaining_budget / float(remaining_releases))

    def record_release(self, actual_solve_seconds: float) -> None:
        if self.total_budget_seconds > 0.0:
            self.consumed_seconds = min(
                self.total_budget_seconds,
                self.consumed_seconds + max(0.0, float(actual_solve_seconds)),
            )
        self.used_releases += 1

    def remaining_budget(self) -> float:
        return max(0.0, self.total_budget_seconds - self.consumed_seconds)


def compute_batch_horizon(jobs: List[JobSpec], machine_free_time: np.ndarray, time_scale: int) -> int:
    if not jobs:
        return 1
    max_ready = 0
    max_due = 0
    min_due = 0
    max_machine_free = max((float(x) for x in machine_free_time), default=0.0)
    total_max_proc = 0
    for job in jobs:
        max_ready = max(max_ready, scale_time(float(job.meta.get("ready_at", job.meta.get("t_arrive", 0.0))), time_scale))
        due_scaled = scale_time(float(job.meta.get("due_date", 0.0)), time_scale)
        max_due = max(max_due, due_scaled)
        min_due = min(min_due, due_scaled)
        for op in job.operations:
            tr = np.asarray(op.time_row, dtype=float)
            feas = tr[tr > 0]
            if feas.size == 0:
                raise ValueError(f"Job {job.job_id} has an operation with no feasible machine.")
            total_max_proc += scale_time(float(np.max(feas)), time_scale)
    horizon = scale_time(max_machine_free, time_scale) + total_max_proc + max(0, max_due) + max(0, -min_due)
    return max(1, int(horizon))


def solve_current_batch_ortools(
    jobs: List[JobSpec],
    *,
    machine_free_time: np.ndarray,
    n_machines: int,
    time_scale: int,
    time_limit: float,
) -> Tuple[List[Dict], Dict]:
    if not jobs:
        return []

    model = cp_model.CpModel()
    horizon = compute_batch_horizon(jobs, machine_free_time, time_scale)
    machine_intervals: List[List] = [[] for _ in range(int(n_machines))]
    all_tasks: Dict[Tuple[int, int], Dict] = {}
    job_end_vars: Dict[int, cp_model.IntVar] = {}
    tardiness_vars: Dict[int, cp_model.IntVar] = {}

    due_scaled_by_job = {int(job.job_id): scale_time(float(job.meta.get("due_date", 0.0)), time_scale) for job in jobs}
    min_due = min(due_scaled_by_job.values()) if due_scaled_by_job else 0
    tardiness_upper = horizon + max(0, -min_due)

    for job in jobs:
        job_id = int(job.job_id)
        op_offset = int(job.meta.get("op_offset", 0))
        ready_at = scale_time(float(job.meta.get("ready_at", job.meta.get("t_arrive", 0.0))), time_scale)
        due_scaled = due_scaled_by_job[job_id]
        total_ops = int(job.meta.get("total_ops", op_offset + len(job.operations)))
        prev_end = None

        for local_op_idx, op in enumerate(job.operations):
            global_op_idx = op_offset + local_op_idx
            op_start = model.NewIntVar(0, horizon, f"j{job_id}_o{global_op_idx}_start")
            op_end = model.NewIntVar(0, horizon, f"j{job_id}_o{global_op_idx}_end")
            choices = []
            tr = np.asarray(op.time_row, dtype=float)
            for machine_idx, pt in enumerate(tr.tolist()):
                duration_scaled = scale_time(pt, time_scale)
                if duration_scaled <= 0:
                    continue
                presence = model.NewBoolVar(f"j{job_id}_o{global_op_idx}_m{machine_idx}_present")
                start_m = model.NewIntVar(0, horizon, f"j{job_id}_o{global_op_idx}_m{machine_idx}_start")
                end_m = model.NewIntVar(0, horizon, f"j{job_id}_o{global_op_idx}_m{machine_idx}_end")
                interval = model.NewOptionalIntervalVar(
                    start_m,
                    duration_scaled,
                    end_m,
                    presence,
                    f"j{job_id}_o{global_op_idx}_m{machine_idx}_interval",
                )
                model.Add(start_m == op_start).OnlyEnforceIf(presence)
                model.Add(end_m == op_end).OnlyEnforceIf(presence)
                model.Add(start_m >= scale_time(float(machine_free_time[machine_idx]), time_scale)).OnlyEnforceIf(presence)
                machine_intervals[machine_idx].append(interval)
                choices.append(
                    {
                        "machine": int(machine_idx),
                        "presence": presence,
                        "duration_scaled": duration_scaled,
                    }
                )

            if not choices:
                raise ValueError(f"Job {job_id} op {global_op_idx} has no valid machine choice.")

            model.AddExactlyOne([choice["presence"] for choice in choices])
            if prev_end is None:
                model.Add(op_start >= ready_at)
            else:
                model.Add(op_start >= prev_end)
            prev_end = op_end
            all_tasks[(job_id, global_op_idx)] = {
                "start": op_start,
                "end": op_end,
                "choices": choices,
                "due_date": float(job.meta.get("due_date", 0.0)),
                "num_ops": total_ops,
                "is_last_op": bool(global_op_idx == total_ops - 1),
            }

        job_end_vars[job_id] = prev_end
        tardiness = model.NewIntVar(0, tardiness_upper, f"j{job_id}_tardiness")
        model.Add(tardiness >= prev_end - due_scaled)
        tardiness_vars[job_id] = tardiness

    for machine_idx in range(int(n_machines)):
        model.AddNoOverlap(machine_intervals[machine_idx])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(job_end_vars.values()))
    total_tardiness_upper = tardiness_upper * max(1, len(tardiness_vars))
    total_tardiness = model.NewIntVar(0, total_tardiness_upper, "total_tardiness")
    model.Add(total_tardiness == sum(tardiness_vars.values()))
    model.Minimize(makespan + total_tardiness)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solve_start_ts = time.time()
    status = solver.Solve(model)
    solve_wall_time = max(0.0, time.time() - solve_start_ts)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools subproblem solver failed with status={solver.StatusName(status)}")

    rows: List[Dict] = []
    for (job_id, global_op_idx), task in sorted(all_tasks.items(), key=lambda x: (x[0][0], x[0][1])):
        chosen_machine = None
        for choice in task["choices"]:
            if solver.Value(choice["presence"]):
                chosen_machine = int(choice["machine"])
                break
        if chosen_machine is None:
            raise RuntimeError(f"No machine selected for job {job_id} op {global_op_idx}")
        start_time = unscale_time(solver.Value(task["start"]), time_scale)
        end_time = unscale_time(solver.Value(task["end"]), time_scale)
        rows.append(
            {
                "job": int(job_id),
                "op": int(global_op_idx),
                "machine": int(chosen_machine),
                "start": float(start_time),
                "end": float(end_time),
                "duration": float(end_time - start_time),
            }
        )
    total_tardiness = sum(
        unscale_time(solver.Value(tardiness_vars[job_id]), time_scale)
        for job_id in tardiness_vars.keys()
    )
    makespan = unscale_time(solver.Value(makespan), time_scale)
    solve_info = {
        "status": solver.StatusName(status),
        "solve_time_seconds": float(solve_wall_time),
        "solver_wall_time_seconds": float(solver.WallTime()),
        "makespan": float(makespan),
        "total_tardiness": float(total_tardiness),
        "objective_value": float(makespan + total_tardiness),
    }
    return rows, solve_info


def event_release_and_reschedule_ortools(
    orch: GlobalTimelineOrchestrator,
    *,
    t_e: float,
    all_due: Dict[int, float],
    time_scale: int,
    time_limit: float,
    event_id: Optional[int] = None,
) -> Dict:
    orch.t = float(t_e)
    orch._release_count += 1
    history_add = [dict(r) for r in orch._last_full_rows if float(r["start"]) < orch.t]
    if history_add:
        orch._extend_global_rows_dedup(history_add)

    busy = np.full(orch.M, orch.t, dtype=float)
    for r in orch._global_rows:
        if float(r["start"]) < orch.t < float(r["end"]):
            busy[int(r["machine"])] = max(busy[int(r["machine"])], float(r["end"]))
    orch.machine_free_time = busy

    buffer_jobs = list(orch.buffer)
    orch.buffer.clear()
    orch._last_jobs_snapshot.extend(buffer_jobs)

    by_j = {}
    for r in orch._last_full_rows:
        by_j.setdefault(int(r["job"]), []).append(r)

    jobs_new: List[JobSpec] = []
    for js in orch._last_jobs_snapshot:
        jid = int(js.job_id)
        rows = by_j.get(jid, [])
        started = [r for r in rows if float(r["start"]) < orch.t]
        total_ops = int(js.meta.get("total_ops", len(js.operations)))
        if len(started) < total_ops:
            js_b = JobSpec(job_id=js.job_id, operations=js.operations[len(started):], meta=js.meta.copy())
            js_b.meta["op_offset"] = len(started)
            inprog = [r for r in rows if float(r["start"]) <= orch.t < float(r["end"])]
            js_b.meta["ready_at"] = float(inprog[0]["end"]) if inprog else max(float(js.meta.get("t_arrive", 0.0)), orch.t)
            jobs_new.append(js_b)

    if not jobs_new:
        return {"event": "tick", "t": orch.t}

    orch._committed_jobs = jobs_new
    rows, solve_info = solve_current_batch_ortools(
        jobs_new,
        machine_free_time=orch.machine_free_time,
        n_machines=orch.M,
        time_scale=time_scale,
        time_limit=time_limit,
    )

    new_machine_free = orch.machine_free_time.astype(float).copy()
    for r in rows:
        m = int(r["machine"])
        new_machine_free[m] = max(new_machine_free[m], float(r["end"]))
    orch.machine_free_time = new_machine_free

    f_dict = {(int(r["job"]), int(r["op"])): r for r in orch._last_full_rows}
    committed_job_ids = {js.job_id for js in jobs_new}
    to_del = [k for k, r in f_dict.items() if k[0] in committed_job_ids and float(r["start"]) >= orch.t]
    for k in to_del:
        del f_dict[k]
    for r in rows:
        f_dict[(int(r["job"]), int(r["op"]))] = r
    orch._last_full_rows = sorted(list(f_dict.values()), key=lambda x: (x["job"], x["op"]))

    fins = set()
    for jid in {js.job_id for js in orch._last_jobs_snapshot}:
        j_rows = [r for r in orch._last_full_rows if int(r["job"]) == jid]
        if j_rows and max(float(r["end"]) for r in j_rows) <= orch.t:
            fins.add(jid)
    orch._last_jobs_snapshot = [js for js in orch._last_jobs_snapshot if js.job_id not in fins]
    return {
        "event": "batch_finalized",
        "t": orch.t,
        "rows": rows,
        "solve_info": solve_info,
        "event_id": int(event_id) if event_id is not None else None,
    }


def run_event_driven_ortools_cadence(
    *,
    max_events: int,
    interarrival_mean: float,
    burst_k: int = 1,
    plot_global_dir: Optional[str] = None,
):
    fast_mode = bool(getattr(configs, "fast_mode", True))
    seed = int(getattr(configs, "event_seed", 42))
    cadence = max(1, int(getattr(configs, "gate_cadence", 3)))
    time_scale = max(1, int(getattr(configs, "ortools_time_scale", 1)))
    subproblem_time_limit = float(getattr(configs, "ortools_subproblem_time_limit", 30.0))
    total_solve_time_budget = float(getattr(configs, "ortools_total_solve_time_budget", 0.0))

    rng, gen, orch = create_dynamic_world(
        configs,
        interarrival_mean=float(interarrival_mean),
        burst_k=int(burst_k),
        seed=seed,
    )
    all_job_due_dates: Dict[int, float] = {}
    reward_scale = (float(configs.low) + float(configs.high)) / 2.0

    suffix = f"ORTCadence_{cadence}"
    base_plot_dir = plot_global_dir or "plots/global"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = f"ortcad{cadence}"
    override_name = str(getattr(configs, "plot_run_name", "")).strip()
    if override_name:
        safe_name = override_name
    safe_name = safe_name.replace(" ", "_")[:48]
    csv_dir = os.path.join(base_plot_dir, f"{timestamp}_{safe_name}")
    os.makedirs(csv_dir, exist_ok=True)

    csv_prefix = safe_name
    raw_csv_file = open(os.path.join(csv_dir, f"{csv_prefix}_raw_state.csv"), "w", newline="", encoding="utf-8")
    raw_csv_writer = csv.writer(raw_csv_file)
    obs_csv_file = open(os.path.join(csv_dir, f"{csv_prefix}_agent_state.csv"), "w", newline="", encoding="utf-8")
    obs_csv_writer = csv.writer(obs_csv_file)
    ort_csv_file = open(os.path.join(csv_dir, f"{csv_prefix}_ortools_release_log.csv"), "w", newline="", encoding="utf-8")
    ort_csv_writer = csv.writer(ort_csv_file)

    raw_headers = [
        "Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str",
        "Raw_Buffer_Count", "Raw_Avg_Load", "Raw_Min_Load", "Raw_Max_Load", "Raw_Load_Std",
        "Raw_Weighted_Idle", "Raw_Unweighted_Idle",
        "Raw_Buffer_NegSlack_Ratio", "Raw_Buf_Min_Slack", "Raw_Buf_Avg_Slack", "Raw_Buf_Slack_Std", "Raw_Buf_Slack_Q25",
        "Raw_WIP_Job_Count", "Raw_WIP_Tardy_Ratio", "Raw_WIP_Min_Slack", "Raw_WIP_Avg_Slack", "Raw_WIP_Slack_Std",
        "Raw_WIP_Planned_TD", "Raw_WIP_Total_Rem_Work",
        "Baseline_Step_TD", "Baseline_Prev_Release_Event_ID", "Baseline_Prev_Release_TD", "Baseline_TD_Delta",
        "Agent_Prev_Release_Event_ID", "Agent_Prev_Release_TD", "Agent_TD_Delta",
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD", "TD_Gap_vs_Baseline_Cadence",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
    ]
    obs_headers = [
        "Event_ID", "Time", "Inter_Arrival", "Action", "Action_Str",
        "Log_Buffer_Count", "Norm_Avg_Load", "Norm_Min_Load", "Norm_Load_Range", "Norm_Load_Std",
        "Norm_Weighted_Idle", "Norm_Unweighted_Idle",
        "Buffer_NegSlack_Ratio", "Norm_Buf_Min_Slack", "Norm_Buf_Avg_Slack", "Norm_Buf_Slack_Std", "Norm_Buf_Slack_Q25",
        "WIP_Job_Count", "WIP_Tardy_Ratio", "Norm_WIP_Min_Slack", "Norm_WIP_Avg_Slack", "Norm_WIP_Slack_Std",
        "Clipped_Planned_TD_Ratio", "Avg_WIP_Slack_Per_Job",
        "Scaled_Inter_Arrival", "Log_Steps_Since_Last_Release", "Is_Last_Step",
    ] + [
        "Baseline_Step_TD", "Baseline_Prev_Release_Event_ID", "Baseline_Prev_Release_TD", "Baseline_TD_Delta",
        "Agent_Prev_Release_Event_ID", "Agent_Prev_Release_TD", "Agent_TD_Delta",
        "Actual_TD",
        "Reward_Total", "Reward_Stab", "Reward_Buffer", "Reward_Shaping", "Reward_Terminal", "Reward_Flush",
        "Phi_Before", "Phi_After", "Agent_Final_TD", "TD_Gap_vs_Baseline_Cadence",
        "Final_Makespan", "Final_Tardiness", "Release_Count",
    ]
    obs_csv_order = [0, 1, 2, 15, 10, 3, 17, 4, 5, 6, 12, 14, 9, 7, 8, 13, 11, 16, 18, 19, 20, 21]
    raw_csv_writer.writerow(raw_headers)
    obs_csv_writer.writerow(obs_headers)
    ort_csv_writer.writerow([
        "Event_ID",
        "Release_Time",
        "Time_Limit_Seconds",
        "Solver_Status",
        "Solve_Time_Seconds",
        "Solver_Wall_Time_Seconds",
        "Objective_MK_Plus_TD",
        "Makespan",
        "Total_Tardiness",
        "Num_Committed_Jobs",
        "Num_Rows",
    ])

    def get_raw_state_info(orchestrator, t_now):
        b_slacks, b_neg = [], 0
        for j in orchestrator.buffer:
            mw = float(j.meta.get("total_proc_time", 0.0))
            due = all_job_due_dates[j.job_id]
            s = due - t_now - mw
            b_slacks.append(s)
            if t_now + mw > due:
                b_neg += 1
        b_stats = (
            b_neg / len(orchestrator.buffer),
            min(b_slacks),
            sum(b_slacks) / len(b_slacks),
            np.std(b_slacks),
            np.percentile(b_slacks, 25),
        ) if orchestrator.buffer else (0.0, 0.0, 0.0, 0.0, 0.0)
        wip = orchestrator.get_wip_stats(t_now)
        mft = np.asarray(orchestrator.machine_free_time, dtype=float)
        rem = np.maximum(0.0, mft - t_now)
        mx_l = np.max(rem)
        w_idle = orchestrator.compute_weighted_idle(t_now, float(mx_l)) if mx_l > 0 else 0.0
        u_idle = orchestrator.compute_unweighted_idle(t_now, float(mx_l)) if mx_l > 0 else 0.0
        return [
            len(orchestrator.buffer),
            np.mean(rem),
            np.min(rem),
            mx_l,
            np.std(rem),
            w_idle,
            u_idle,
            b_stats[0],
            b_stats[1],
            b_stats[2],
            b_stats[3],
            b_stats[4],
            wip["wip_count"],
            wip["wip_tardy_ratio"],
            wip["wip_min_slack"],
            wip["wip_avg_slack"],
            wip["wip_slack_std"],
            wip["planned_td"],
            wip["total_rem_work"],
        ]

    def save_details(t_marker: float, seq: int, label: str = ""):
        if fast_mode:
            return
        unique_rows = {}

        def row_status(row):
            return "History" if float(row["start"]) < float(t_marker) else "NewPlan"

        def should_replace(existing, new_row):
            if existing is None:
                return True
            existing_row, existing_status = existing
            new_status = row_status(new_row)
            if existing_status == "History" and new_status == "NewPlan":
                return False
            if existing_status == "NewPlan" and new_status == "History":
                return True
            return True

        for r in orch._global_rows:
            key = (int(r["job"]), int(r["op"]))
            if should_replace(unique_rows.get(key), r):
                unique_rows[key] = (dict(r), row_status(r))
        for r in orch._last_full_rows:
            key = (int(r["job"]), int(r["op"]))
            if should_replace(unique_rows.get(key), r):
                unique_rows[key] = (dict(r), row_status(r))

        job_max_op = {}
        for jid, opid in unique_rows.keys():
            job_max_op[jid] = max(job_max_op.get(jid, -1), opid)

        full_data_rows = []
        for (jid, opid), (row, status) in sorted(unique_rows.items()):
            due_date = all_job_due_dates.get(jid, 0.0)
            is_last = (opid == job_max_op[jid])
            row.update(
                {
                    "status": status,
                    "due_date": f"{due_date:.2f}",
                    "tardiness": f"{max(0.0, float(row['end']) - due_date):.2f}" if is_last else "0.00",
                }
            )
            full_data_rows.append(row)
        fname = f"details_r{seq:03d}_t{int(t_marker):05d}{label}.csv"
        df = pd.DataFrame(full_data_rows)
        df.to_csv(os.path.join(csv_dir, fname), index=False)
        csv_sum_td = df["tardiness"].astype(float).sum() if "tardiness" in df.columns else 0.0
        print(f"  [CSV Export] {fname} | Unique Jobs: {len(job_max_op)} | Total TD: {csv_sum_td:.2f}")

    def build_plot_rows(t_marker: float):
        return [
            dict(
                r,
                phase="history" if float(r["start"]) < float(t_marker) else "newplan",
                due_date=float(all_job_due_dates.get(int(r["job"]), 0.0)),
            )
            for r in (orch._global_rows + orch._last_full_rows)
        ]

    release_count = 0
    plot_seq = 0
    steps_since_last_release = 0
    pending_raw_rows = []
    pending_obs_rows = []

    init_jobs = sample_initial_jobs(configs, rng=rng, base_job_id=0, t_arrive=0.0)
    solve_budget = ORToolsSolveBudget(
        total_budget_seconds=total_solve_time_budget,
        planned_releases=compute_planned_release_count(
            max_events=int(max_events),
            cadence=int(cadence),
            has_init_release=bool(init_jobs),
        ),
    )
    if init_jobs:
        for job in init_jobs:
            all_job_due_dates[job.job_id] = float(job.meta.get("due_date", 0.0))
        orch.buffer.extend(init_jobs)
        gen.bump_next_id(max((job.job_id for job in init_jobs), default=-1) + 1)
        init_time_limit = solve_budget.next_time_limit(subproblem_time_limit)
        init_result = event_release_and_reschedule_ortools(
            orch,
            t_e=0.0,
            all_due=all_job_due_dates,
            time_scale=time_scale,
            time_limit=init_time_limit,
            event_id=0,
        )
        release_count += 1
        init_info = init_result.get("solve_info", {})
        solve_budget.record_release(init_info.get("solve_time_seconds", 0.0))
        ort_csv_writer.writerow([
            0,
            f"{0.0:.4f}",
            f"{float(init_time_limit):.6f}",
            init_info.get("status", ""),
            f"{float(init_info.get('solve_time_seconds', 0.0)):.6f}",
            f"{float(init_info.get('solver_wall_time_seconds', 0.0)):.6f}",
            f"{float(init_info.get('objective_value', 0.0)):.4f}",
            f"{float(init_info.get('makespan', 0.0)):.4f}",
            f"{float(init_info.get('total_tardiness', 0.0)):.4f}",
            len(orch._committed_jobs),
            len(init_result.get("rows", [])),
        ])
        if not fast_mode:
            save_details(0.0, plot_seq + 1, "_INIT")
            plot_global_gantt(build_plot_rows(0.0), os.path.join(csv_dir, f"global_r{plot_seq:03d}_t0.png"), t_now=0.0, title="Initial")
        plot_seq += 1

    stats = {"arrive": 0}
    t_now, t_prev = 0.0, 0.0
    t_next = gen.sample_next_time(t_now)
    while stats["arrive"] < int(max_events):
        t_now = float(t_next)
        inter_arrival = t_now - t_prev
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            for job in new_jobs:
                all_job_due_dates[job.job_id] = float(job.meta.get("due_date", 0.0))
            orch.buffer.extend(new_jobs)
        stats["arrive"] += 1
        is_last_step = bool(stats["arrive"] >= int(max_events))

        raw_s = get_raw_state_info(orch, t_now)
        b_dict = {
            "buffer_neg_slack_ratio": raw_s[7],
            "min_slack": raw_s[8],
            "avg_slack": raw_s[9],
            "slack_std": raw_s[10],
            "slack_q25": raw_s[11],
        }
        w_dict = {
            "wip_count": raw_s[12],
            "wip_tardy_ratio": raw_s[13],
            "wip_min_slack": raw_s[14],
            "wip_avg_slack": raw_s[15],
            "wip_slack_std": raw_s[16],
            "planned_td": raw_s[17],
            "total_rem_work": raw_s[18],
        }
        obs = calculate_gate_state(
            len(orch.buffer),
            orch.machine_free_time,
            t_now,
            configs.n_m,
            0,
            reward_scale,
            raw_s[5],
            raw_s[6],
            b_dict,
            w_dict,
            inter_arrival_scaled=(float(inter_arrival) / float(reward_scale)) if reward_scale > 0 else 0.0,
            steps_since_last_release=steps_since_last_release,
            is_last_step=is_last_step,
        )

        act = 1 if (is_last_step or (stats["arrive"] % cadence == 0)) else 0
        action_str = "RELEASE" if act == 1 else "HOLD"

        if act == 1:
            release_time_limit = solve_budget.next_time_limit(subproblem_time_limit)
            result = event_release_and_reschedule_ortools(
                orch,
                t_e=t_now,
                all_due=all_job_due_dates,
                time_scale=time_scale,
                time_limit=release_time_limit,
                event_id=int(stats["arrive"]),
            )
            release_count += 1
            steps_since_last_release = 0
            solve_info = result.get("solve_info", {})
            solve_budget.record_release(solve_info.get("solve_time_seconds", 0.0))
            ort_csv_writer.writerow([
                int(stats["arrive"]),
                f"{t_now:.4f}",
                f"{float(release_time_limit):.6f}",
                solve_info.get("status", ""),
                f"{float(solve_info.get('solve_time_seconds', 0.0)):.6f}",
                f"{float(solve_info.get('solver_wall_time_seconds', 0.0)):.6f}",
                f"{float(solve_info.get('objective_value', 0.0)):.4f}",
                f"{float(solve_info.get('makespan', 0.0)):.4f}",
                f"{float(solve_info.get('total_tardiness', 0.0)):.4f}",
                len(orch._committed_jobs),
                len(result.get("rows", [])),
            ])
            if not fast_mode:
                save_details(t_now, plot_seq + 1)
                plot_global_gantt(
                    build_plot_rows(t_now),
                    os.path.join(csv_dir, f"global_r{plot_seq:03d}_t{int(t_now):05d}.png"),
                    t_now=t_now,
                    title=f"Event #{stats['arrive']}",
                )
                plot_seq += 1
        else:
            orch.tick_without_release(t_now)
            steps_since_last_release += 1

        actual_td_logged = float(orch.get_total_tardiness_estimate(all_job_due_dates))
        common_tail = [
            "", "", "", "", "", "", "", f"{actual_td_logged:.2f}",
            "", "", "", "", "", "", "", "", "", "", "", "", release_count,
        ]
        raw_row = [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, action_str] + [f"{float(x):.6f}" for x in raw_s] + common_tail
        obs_row = [stats["arrive"], f"{t_now:.2f}", f"{inter_arrival:.2f}", act, action_str] + [f"{float(obs[i]):.6f}" for i in obs_csv_order] + common_tail
        pending_raw_rows.append(raw_row)
        pending_obs_rows.append(obs_row)

        t_prev = t_now
        t_next = float(gen.sample_next_time(t_now))

    t_flush = float(t_next)
    while len(orch.buffer) > 0:
        flush_time_limit = solve_budget.next_time_limit(subproblem_time_limit)
        flush_result = event_release_and_reschedule_ortools(
            orch,
            t_e=t_flush,
            all_due=all_job_due_dates,
            time_scale=time_scale,
            time_limit=flush_time_limit,
            event_id=int(max_events) + 1,
        )
        release_count += 1
        flush_info = flush_result.get("solve_info", {})
        solve_budget.record_release(flush_info.get("solve_time_seconds", 0.0))
        ort_csv_writer.writerow([
            int(max_events) + 1,
            f"{t_flush:.4f}",
            f"{float(flush_time_limit):.6f}",
            flush_info.get("status", ""),
            f"{float(flush_info.get('solve_time_seconds', 0.0)):.6f}",
            f"{float(flush_info.get('solver_wall_time_seconds', 0.0)):.6f}",
            f"{float(flush_info.get('objective_value', 0.0)):.4f}",
            f"{float(flush_info.get('makespan', 0.0)):.4f}",
            f"{float(flush_info.get('total_tardiness', 0.0)):.4f}",
            len(orch._committed_jobs),
            len(flush_result.get("rows", [])),
        ])
        if not fast_mode:
            save_details(t_flush, plot_seq + 1, "_FLUSH")
            plot_global_gantt(build_plot_rows(t_flush), os.path.join(csv_dir, f"global_r{plot_seq:03d}_FLUSH.png"), t_now=t_flush, title="FLUSH")
            plot_seq += 1

    final_stats = orch.get_final_kpi_stats(all_job_due_dates)
    total_td = float(final_stats["tardiness"])
    final_mk = float(final_stats["makespan"])

    for row in pending_raw_rows:
        raw_csv_writer.writerow(row)
    for row in pending_obs_rows:
        obs_csv_writer.writerow(row)

    raw_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 18 + [
        f"{total_td:.2f}", "", "", "", "", "", "", "", "", "", f"{final_mk:.2f}", f"{total_td:.2f}", release_count
    ]
    obs_summary = ["END", f"{t_now:.2f}", "", "", "SUMMARY"] + [""] * 18 + [
        f"{total_td:.2f}", "", "", "", "", "", "", "", "", "", f"{final_mk:.2f}", f"{total_td:.2f}", release_count
    ]
    raw_csv_writer.writerow(raw_summary)
    obs_csv_writer.writerow(obs_summary)
    raw_csv_file.close()
    obs_csv_file.close()
    ort_csv_file.close()

    summary = {
        "policy": suffix,
        "event_seed": seed,
        "gate_cadence": cadence,
        "event_horizon": int(max_events),
        "init_jobs": int(getattr(configs, "init_jobs", 0)),
        "release_count": int(release_count),
        "makespan": float(final_mk),
        "total_tardiness": float(total_td),
        "subproblem_time_limit": float(subproblem_time_limit),
        "total_solve_time_budget": float(total_solve_time_budget),
        "remaining_solve_time_budget": float(solve_budget.remaining_budget()),
        "time_scale": int(time_scale),
        "output_dir": csv_dir,
    }
    with open(os.path.join(csv_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{suffix}] MK={final_mk:.2f} | TD={total_td:.2f} | Releases={release_count} | Output={csv_dir}")
    return final_mk, {"release_count": release_count, "total_tardiness": total_td}


def main():
    print("-------------------- Dynamic FJSP OR-Tools Cadence --------------------")
    run_event_driven_ortools_cadence(
        max_events=int(configs.event_horizon),
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        plot_global_dir=str(getattr(configs, "plot_global_dir", "plots/global")),
    )


if __name__ == "__main__":
    main()
