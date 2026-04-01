import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model


def parse_solver_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=os.path.join("or_tools_solutions", "dynamic"))
    parser.add_argument("--instance_output", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--time_limit", type=float, default=7200)
    parser.add_argument("--time_scale", type=int, default=1)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


solver_args = parse_solver_args()

from dynamic_job_stream import dynamic_job_stream_to_dict, generate_dynamic_job_stream
from ortools_gantt import plot_ortools_gantt_with_due_dates
from params import configs


class IntermediateSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var, tardiness_var, time_scale: int):
        super().__init__()
        self._makespan_var = makespan_var
        self._tardiness_var = tardiness_var
        self._time_scale = max(1, int(time_scale))
        self._solutions = 0
        self._start_ts = time.time()

    def on_solution_callback(self):
        self._solutions += 1
        elapsed = time.time() - self._start_ts
        mk = self.Value(self._makespan_var) / self._time_scale
        td = self.Value(self._tardiness_var) / self._time_scale
        obj = 0.5 * mk + 0.5 * td
        print(
            f"    [Sol #{self._solutions} @ {elapsed:.1f}s] "
            f"Obj={obj:.3f} (MK={mk:.3f}, TD={td:.3f})"
        )


def scale_time(value: float, time_scale: int) -> int:
    return int(round(float(value) * max(1, int(time_scale))))


def unscale_time(value: int, time_scale: int) -> float:
    return float(value) / float(max(1, int(time_scale)))


def infer_n_machines(payload: Dict) -> int:
    meta_n_m = int(payload.get("meta", {}).get("n_machines", 0))
    if meta_n_m > 0:
        return meta_n_m

    max_m = -1
    for job in payload.get("jobs", []):
        for op in job.get("operations", []):
            for m_key in op.get("machine_times", {}).keys():
                max_m = max(max_m, int(m_key))
    if max_m < 0:
        raise ValueError("Failed to infer n_machines from instance payload.")
    return max_m + 1


def derive_instance_name(payload: Dict, instance_json_path: str = "", explicit_name: str = "") -> str:
    if explicit_name:
        return explicit_name
    if instance_json_path:
        return os.path.splitext(os.path.basename(instance_json_path))[0]
    meta = payload.get("meta", {})
    seed = int(meta.get("seed", getattr(configs, "event_seed", 42)))
    horizon = int(meta.get("event_horizon", getattr(configs, "event_horizon", 0)))
    burst = int(meta.get("burst_size", getattr(configs, "burst_size", 1)))
    return f"dynamic_seed{seed}_h{horizon}_k{burst}"


def validate_unique_job_ids(payload: Dict):
    seen = set()
    duplicates = []
    for job in payload.get("jobs", []):
        job_id = int(job.get("job_id", -1))
        if job_id in seen:
            duplicates.append(job_id)
        seen.add(job_id)
    if duplicates:
        dup_str = ", ".join(str(x) for x in sorted(set(duplicates))[:10])
        raise ValueError(
            "Duplicate job_id values found in dynamic instance payload "
            f"({dup_str}). Regenerate the instance JSON after the init_jobs fix."
        )


def save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_or_generate_instance() -> Tuple[Dict, str]:
    instance_json_path = str(solver_args.instance_json or "").strip()
    if instance_json_path:
        with open(instance_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        validate_unique_job_ids(payload)
        return payload, instance_json_path

    stream = generate_dynamic_job_stream(
        configs,
        max_events=int(configs.event_horizon),
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        seed=int(getattr(configs, "event_seed", 42)),
    )
    payload = dynamic_job_stream_to_dict(stream, config=configs)
    validate_unique_job_ids(payload)
    return payload, ""


def compute_horizon(payload: Dict, time_scale: int) -> int:
    jobs = payload.get("jobs", [])
    if not jobs:
        return 1

    max_release = 0
    max_due = 0
    min_due = 0
    total_max_proc = 0
    for job in jobs:
        max_release = max(max_release, scale_time(job.get("arrive_time", 0.0), time_scale))
        due_scaled = scale_time(job.get("due_date", 0.0), time_scale)
        max_due = max(max_due, due_scaled)
        min_due = min(min_due, due_scaled)
        for op in job.get("operations", []):
            machine_times = [float(v) for v in op.get("machine_times", {}).values() if float(v) > 0]
            if not machine_times:
                raise ValueError(f"Job {job.get('job_id')} op {op.get('op_id')} has no feasible machine.")
            total_max_proc += scale_time(max(machine_times), time_scale)

    horizon = max_release + total_max_proc + max(0, max_due) + max(0, -min_due)
    return max(1, int(horizon))


def build_dynamic_fjsp_model(payload: Dict, time_scale: int):
    jobs = sorted(payload.get("jobs", []), key=lambda x: int(x.get("job_id", 0)))
    if not jobs:
        raise ValueError("Dynamic instance has no jobs to solve.")

    n_machines = infer_n_machines(payload)
    horizon = compute_horizon(payload, time_scale)

    due_scaled_by_job = {int(job["job_id"]): scale_time(job.get("due_date", 0.0), time_scale) for job in jobs}
    min_due = min(due_scaled_by_job.values()) if due_scaled_by_job else 0
    tardiness_upper = horizon + max(0, -min_due)

    model = cp_model.CpModel()
    machine_intervals: List[List] = [[] for _ in range(n_machines)]
    all_tasks: Dict[Tuple[int, int], Dict] = {}
    job_end_vars: Dict[int, cp_model.IntVar] = {}
    tardiness_vars: Dict[int, cp_model.IntVar] = {}

    for job in jobs:
        job_id = int(job["job_id"])
        arrive_scaled = scale_time(job.get("arrive_time", 0.0), time_scale)
        due_scaled = due_scaled_by_job[job_id]
        operations = sorted(job.get("operations", []), key=lambda x: int(x.get("op_id", 0)))
        if not operations:
            raise ValueError(f"Job {job_id} has no operations.")

        prev_end = None
        num_ops = len(operations)
        for op in operations:
            op_id = int(op["op_id"])
            op_start = model.NewIntVar(0, horizon, f"j{job_id}_o{op_id}_start")
            op_end = model.NewIntVar(0, horizon, f"j{job_id}_o{op_id}_end")

            choices = []
            machine_times = op.get("machine_times", {})
            for m_key, pt in sorted(machine_times.items(), key=lambda x: int(x[0])):
                duration_scaled = scale_time(pt, time_scale)
                if duration_scaled <= 0:
                    continue
                machine_idx = int(m_key)
                presence = model.NewBoolVar(f"j{job_id}_o{op_id}_m{machine_idx}_present")
                start_m = model.NewIntVar(0, horizon, f"j{job_id}_o{op_id}_m{machine_idx}_start")
                end_m = model.NewIntVar(0, horizon, f"j{job_id}_o{op_id}_m{machine_idx}_end")
                interval = model.NewOptionalIntervalVar(
                    start_m, duration_scaled, end_m, presence, f"j{job_id}_o{op_id}_m{machine_idx}_interval"
                )
                model.Add(start_m == op_start).OnlyEnforceIf(presence)
                model.Add(end_m == op_end).OnlyEnforceIf(presence)
                machine_intervals[machine_idx].append(interval)
                choices.append(
                    {
                        "machine": machine_idx,
                        "presence": presence,
                        "start": start_m,
                        "end": end_m,
                        "duration_scaled": duration_scaled,
                        "duration_original": float(pt),
                    }
                )

            if not choices:
                raise ValueError(f"Job {job_id} op {op_id} has no valid machine choices after scaling.")

            model.AddExactlyOne([choice["presence"] for choice in choices])
            if op_id == 0:
                model.Add(op_start >= arrive_scaled)
            if prev_end is not None:
                model.Add(op_start >= prev_end)

            prev_end = op_end
            all_tasks[(job_id, op_id)] = {
                "job_id": job_id,
                "op_id": op_id,
                "start": op_start,
                "end": op_end,
                "choices": choices,
                "arrive_time": float(job.get("arrive_time", 0.0)),
                "due_date": float(job.get("due_date", 0.0)),
                "num_ops": num_ops,
                "is_last_op": bool(op_id == num_ops - 1),
            }

        job_end_vars[job_id] = prev_end
        tardiness = model.NewIntVar(0, tardiness_upper, f"j{job_id}_tardiness")
        model.Add(tardiness >= prev_end - due_scaled)
        tardiness_vars[job_id] = tardiness

    for machine_idx in range(n_machines):
        model.AddNoOverlap(machine_intervals[machine_idx])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(job_end_vars.values()))

    total_tardiness_upper = tardiness_upper * max(1, len(tardiness_vars))
    total_tardiness = model.NewIntVar(0, total_tardiness_upper, "total_tardiness")
    model.Add(total_tardiness == sum(tardiness_vars.values()))

    # 0.5 * MK + 0.5 * TD is equivalent to MK + TD for optimization, because 0.5 is a common factor.
    model.Minimize(makespan + total_tardiness)

    return {
        "model": model,
        "jobs": jobs,
        "n_machines": n_machines,
        "horizon": horizon,
        "all_tasks": all_tasks,
        "job_end_vars": job_end_vars,
        "tardiness_vars": tardiness_vars,
        "makespan_var": makespan,
        "total_tardiness_var": total_tardiness,
    }


def extract_solution_rows(solver: cp_model.CpSolver, model_data: Dict, time_scale: int) -> List[Dict]:
    rows = []
    all_tasks = model_data["all_tasks"]
    tardiness_vars = model_data["tardiness_vars"]

    for (job_id, op_id), task in sorted(all_tasks.items(), key=lambda x: (x[0][0], x[0][1])):
        chosen_machine = None
        for choice in task["choices"]:
            if solver.Value(choice["presence"]):
                chosen_machine = int(choice["machine"])
                break

        if chosen_machine is None:
            raise RuntimeError(f"Solver returned no machine assignment for job {job_id} op {op_id}.")

        start_scaled = solver.Value(task["start"])
        end_scaled = solver.Value(task["end"])
        start_time = unscale_time(start_scaled, time_scale)
        end_time = unscale_time(end_scaled, time_scale)
        due_date = float(task["due_date"])
        tardiness = unscale_time(solver.Value(tardiness_vars[job_id]), time_scale) if task["is_last_op"] else 0.0

        rows.append(
            {
                "Job": int(job_id),
                "Op": int(op_id),
                "Machine": int(chosen_machine),
                "Start": start_time,
                "End": end_time,
                "Duration": float(end_time - start_time),
                "Arrive_Time": float(task["arrive_time"]),
                "Due_Date": due_date,
                "Is_Last_Op": int(task["is_last_op"]),
                "Tardiness": tardiness,
                "Job_Total_Ops": int(task["num_ops"]),
            }
        )

    return rows


def write_detail_csv(csv_path: str, rows: List[Dict], *, sort_mode: str = "job"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if str(sort_mode).lower() == "machine":
        ordered_rows = sorted(rows, key=lambda r: (int(r["Machine"]), float(r["Start"]), int(r["Job"]), int(r["Op"])))
    else:
        ordered_rows = sorted(rows, key=lambda r: (int(r["Job"]), int(r["Op"])))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Job",
                "Op",
                "Machine",
                "Start",
                "End",
                "Duration",
                "Arrive_Time",
                "Due_Date",
                "Is_Last_Op",
                "Tardiness",
                "Job_Total_Ops",
            ],
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "Job": int(row["Job"]),
                    "Op": int(row["Op"]),
                    "Machine": int(row["Machine"]),
                    "Start": f"{float(row['Start']):.4f}",
                    "End": f"{float(row['End']):.4f}",
                    "Duration": f"{float(row['Duration']):.4f}",
                    "Arrive_Time": f"{float(row['Arrive_Time']):.4f}",
                    "Due_Date": f"{float(row['Due_Date']):.4f}",
                    "Is_Last_Op": int(row["Is_Last_Op"]),
                    "Tardiness": f"{float(row['Tardiness']):.4f}",
                    "Job_Total_Ops": int(row["Job_Total_Ops"]),
                }
            )


def write_summary_json(summary_path: str, summary: Dict):
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def solve_dynamic_instance(payload: Dict, *, output_dir: str, instance_name: str, time_scale: int, time_limit: float):
    model_data = build_dynamic_fjsp_model(payload, time_scale)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)

    printer = IntermediateSolutionPrinter(
        model_data["makespan_var"],
        model_data["total_tardiness_var"],
        time_scale=time_scale,
    )
    start_ts = time.time()
    status = solver.Solve(model_data["model"], printer)
    wall_time = max(0.0, time.time() - start_ts)

    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        summary = {
            "instance_name": instance_name,
            "status": status_name,
            "solve_time_seconds": wall_time,
            "objective_definition": "0.5 * makespan + 0.5 * total_tardiness",
            "objective_optimized_as": "makespan + total_tardiness",
            "time_scale": int(time_scale),
        }
        write_summary_json(os.path.join(output_dir, f"{instance_name}_summary.json"), summary)
        return summary

    rows = extract_solution_rows(solver, model_data, time_scale)
    total_tardiness = sum(float(row["Tardiness"]) for row in rows if int(row["Is_Last_Op"]) == 1)
    makespan = unscale_time(solver.Value(model_data["makespan_var"]), time_scale)
    objective_value = 0.5 * makespan + 0.5 * total_tardiness

    detail_csv_path = os.path.join(output_dir, f"{instance_name}_detail.csv")
    detail_machine_csv_path = os.path.join(output_dir, f"{instance_name}_detail_by_machine.csv")
    gantt_png_path = os.path.join(output_dir, f"{instance_name}_gantt.png")
    write_detail_csv(detail_csv_path, rows, sort_mode="job")
    write_detail_csv(detail_machine_csv_path, rows, sort_mode="machine")
    plot_ortools_gantt_with_due_dates(
        rows,
        gantt_png_path,
        title=f"OR-Tools Dynamic Schedule\nMK={makespan:.2f}, TD={total_tardiness:.2f}",
    )

    summary = {
        "instance_name": instance_name,
        "status": status_name,
        "solve_time_seconds": wall_time,
        "solver_wall_time_seconds": float(solver.WallTime()),
        "num_jobs": len(payload.get("jobs", [])),
        "num_operations": len(rows),
        "n_machines": int(model_data["n_machines"]),
        "time_scale": int(time_scale),
        "objective_definition": "0.5 * makespan + 0.5 * total_tardiness",
        "objective_optimized_as": "makespan + total_tardiness",
        "objective_value": objective_value,
        "makespan": makespan,
        "total_tardiness": total_tardiness,
        "detail_csv": detail_csv_path,
        "detail_by_machine_csv": detail_machine_csv_path,
        "gantt_png": gantt_png_path,
    }
    write_summary_json(os.path.join(output_dir, f"{instance_name}_summary.json"), summary)
    return summary


def main():
    payload, instance_json_path = load_or_generate_instance()
    output_dir = str(solver_args.output_dir or os.path.join("or_tools_solutions", "dynamic"))
    os.makedirs(output_dir, exist_ok=True)

    instance_name = derive_instance_name(payload, instance_json_path=instance_json_path, explicit_name=str(solver_args.name or ""))
    time_scale = max(1, int(solver_args.time_scale))

    if solver_args.time_limit is not None and float(solver_args.time_limit) > 0:
        time_limit = float(solver_args.time_limit)
    else:
        time_limit = float(getattr(configs, "max_solve_time", 1800))

    if instance_json_path:
        used_instance_path = instance_json_path
    else:
        used_instance_path = solver_args.instance_output or os.path.join(output_dir, f"{instance_name}_instance.json")
        save_json(used_instance_path, payload)

    summary = solve_dynamic_instance(
        payload,
        output_dir=output_dir,
        instance_name=instance_name,
        time_scale=time_scale,
        time_limit=time_limit,
    )
    summary["instance_json"] = used_instance_path
    write_summary_json(os.path.join(output_dir, f"{instance_name}_summary.json"), summary)

    print(f"Instance JSON: {used_instance_path}")
    print(
        f"[{summary['status']}] Obj={summary.get('objective_value', float('nan')):.4f} | "
        f"MK={summary.get('makespan', float('nan')):.4f} | "
        f"TD={summary.get('total_tardiness', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
