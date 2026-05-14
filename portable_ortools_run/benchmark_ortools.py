
import os
import json
import time
import argparse
import re
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from ortools_gantt import plot_ortools_gantt_with_due_dates


def text_to_matrix(text):
    n_j = int(re.findall(r"\d+\.?\d*", text[0])[0])
    n_m = int(re.findall(r"\d+\.?\d*", text[0])[1])
    job_length = np.zeros(n_j, dtype="int32")
    op_pt = []

    for i in range(n_j):
        content = np.array([int(s) for s in re.findall(r"\d+\.?\d*", text[i + 1])])
        job_length[i] = content[0]
        idx = 1
        for _ in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype="int32")
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt
            idx = next_idx
            op_pt.append(op_pt_row)

    return job_length, np.array(op_pt)

class IntermediateSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, mk_var, td_vars):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__mk_var = mk_var
        self.__td_vars = td_vars
        self.__solutions = 0
        self.__start_time = time.time()

    def on_solution_callback(self):
        self.__solutions += 1
        elapsed = time.time() - self.__start_time
        mk = self.Value(self.__mk_var)
        td = sum(self.Value(t) for t in self.__td_vars)
        print(f"    [Sol #{self.__solutions} @ {elapsed:.1f}s] Obj: {self.ObjectiveValue():.1f} (MK: {mk}, TD: {td})")

def solve_fjsp_with_ortools(fjs_path, json_path, time_limit=300, log_solutions=False):
    # 1. Load Data
    with open(fjs_path, "r") as f:
        content = f.readlines()
    jl, pt = text_to_matrix(content)
    
    with open(json_path, "r") as f:
        due_data = json.load(f)
    due_dates = due_data["due_dates"]
    
    n_j, n_m = len(jl), pt.shape[1]
    model = cp_model.CpModel()
    horizon = int(np.sum(np.max(pt, axis=1))) + max(0, int(max(due_dates)))
    
    # 2. Variables & Constraints
    # all_tasks_vars: (j, o) -> list of (machine_idx, presence_bool, start_var, end_var)
    all_tasks_vars = {} 
    machine_intervals = [[] for _ in range(n_m)]
    job_ends = []
    op_ptr = 0
    
    for j in range(n_j):
        last_op_end = None
        for o in range(jl[j]):
            compatible_mchs = np.where(pt[op_ptr] > 0)[0]
            op_start = model.NewIntVar(0, horizon, f'j{j}_o{o}_s')
            op_end = model.NewIntVar(0, horizon, f'j{j}_o{o}_e')
            
            choices = []
            for m in compatible_mchs:
                p = model.NewBoolVar(f'j{j}_o{o}_m{m}_p')
                duration = int(round(pt[op_ptr, m]))
                
                start_m = model.NewIntVar(0, horizon, f'j{j}_o{o}_m{m}_s')
                end_m = model.NewIntVar(0, horizon, f'j{j}_o{o}_m{m}_e')
                interval = model.NewOptionalIntervalVar(start_m, duration, end_m, p, f'j{j}_o{o}_m{m}_i')
                
                model.Add(start_m == op_start).OnlyEnforceIf(p)
                model.Add(end_m == op_end).OnlyEnforceIf(p)
                
                machine_intervals[m].append(interval)
                choices.append((m, p, start_m, end_m))
            
            model.AddExactlyOne([c[1] for c in choices])
            if last_op_end is not None:
                model.Add(op_start >= last_op_end)
            
            last_op_end = op_end
            all_tasks_vars[(j, o)] = choices
            op_ptr += 1
        job_ends.append(last_op_end)

    for m in range(n_m):
        model.AddNoOverlap(machine_intervals[m])

    # Objectives
    total_tardiness_vars = []
    for j in range(n_j):
        t = model.NewIntVar(0, horizon, f'j{j}_td')
        model.Add(t >= job_ends[j] - int(round(due_dates[j])))
        total_tardiness_vars.append(t)
    
    makespan = model.NewIntVar(0, horizon, 'mk')
    model.AddMaxEquality(makespan, job_ends)
    
    model.Minimize(makespan + sum(total_tardiness_vars))

    # 3. Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if log_solutions:
        printer = IntermediateSolutionPrinter(makespan, total_tardiness_vars)
        status = solver.Solve(model, printer)
    else:
        status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol_rows = []
        job_completion = {}
        for (j, o), choices in all_tasks_vars.items():
            for m_idx, p, s_var, e_var in choices:
                if solver.Value(p):
                    start_val = solver.Value(s_var)
                    end_val = solver.Value(e_var)
                    job_completion[j] = max(job_completion.get(j, 0), end_val)
                    sol_rows.append({
                        "Job": j, "Op": o, "Machine": m_idx,
                        "Start": start_val, "End": end_val,
                        "Duration": end_val - start_val,
                        "Due_Date": float(due_dates[j]),
                    })
        for row in sol_rows:
            job = int(row["Job"])
            row["Job_Completion"] = float(job_completion.get(job, 0.0))
            row["Job_Tardiness"] = max(0.0, row["Job_Completion"] - float(row["Due_Date"]))
        return {
            "makespan": solver.Value(makespan),
            "total_tardiness": sum(solver.Value(t) for t in total_tardiness_vars),
            "obj": 0.5 * solver.Value(makespan) + 0.5 * sum(solver.Value(t) for t in total_tardiness_vars),
            "status": solver.StatusName(status),
            "time": solver.WallTime(),
            "solution_rows": sol_rows
        }
    return None

def _parse_scale(scale_name):
    parts = str(scale_name).split("_", 1)
    size = parts[0]
    due_setting = parts[1] if len(parts) > 1 else ""
    if "x" in size:
        n_j, n_m = size.split("x", 1)
        return int(n_j), int(n_m), due_setting
    return None, None, due_setting


def _safe_name(*parts):
    return "_".join(str(p).replace(" ", "_").replace("/", "_").replace("\\", "_") for p in parts if str(p) != "")


def run_benchmark(base_dir="or_instances_uniform_test_30_50", time_limit=7200, output_root="or_tools_solutions", make_gantt=True, log_solutions=False, resume_dir=""):
    output_dir = str(resume_dir).strip()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_root, f"static_ortools_{timestamp}")
    detail_dir = os.path.join(output_dir, "schedule_details")
    gantt_dir = os.path.join(output_dir, "gantt")
    os.makedirs(detail_dir, exist_ok=True)
    os.makedirs(gantt_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "ortools_static_instance_results.csv")
    results = []
    processed = set()
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        if not old_df.empty and {"scale", "instance"}.issubset(old_df.columns):
            results = old_df.to_dict("records")
            processed = set(zip(old_df["scale"].astype(str), old_df["instance"].astype(str)))
            print(f"Resume enabled. Found {len(processed)} completed instances in {output_path}")
    failed = []

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Static instance directory not found: {base_dir}")

    subdirs = sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)))
    print("Static OR-Tools benchmark")
    print(f"Input: {base_dir}")
    print(f"Output: {output_dir}")
    print(f"Time limit per instance: {float(time_limit):.1f}s")

    for scale in subdirs:
        path = os.path.join(base_dir, scale)
        files = sorted(f for f in os.listdir(path) if f.endswith(".fjs"))
        n_j_from_name, n_m_from_name, due_setting_from_name = _parse_scale(scale)
        print(f"\n--- Scale: {scale} ---")
        for fjs_file in files:
            base = fjs_file.replace(".fjs", "")
            if (str(scale), str(base)) in processed:
                print(f"Skipping completed {scale}/{base}")
                continue
            print(f"Solving {fjs_file}...")
            f_path = os.path.join(path, fjs_file)
            j_path = os.path.join(path, f"{base}.json")

            if not os.path.exists(j_path):
                failed.append({"scale": scale, "instance": base, "reason": "missing_json"})
                print(f"  Missing JSON: {j_path}")
                continue

            with open(j_path, "r", encoding="utf-8") as f:
                due_data = json.load(f)

            due_setting = str(due_data.get("due_setting", due_setting_from_name))
            n_j = int(due_data.get("n_j", n_j_from_name if n_j_from_name is not None else 0))
            n_m = int(due_data.get("n_m", n_m_from_name if n_m_from_name is not None else 0))
            instance_seed = due_data.get("instance_seed", "")
            due_seed = due_data.get("due_seed", "")
            range_low = due_data.get("range_low", "")
            range_high = due_data.get("range_high", "")

            start_wall = time.time()
            res = solve_fjsp_with_ortools(f_path, j_path, time_limit=float(time_limit), log_solutions=log_solutions)
            elapsed_wall = time.time() - start_wall
            if res:
                sol_df = pd.DataFrame(res["solution_rows"])
                file_prefix = _safe_name(scale, base)
                detail_path = os.path.join(detail_dir, f"{file_prefix}_schedule.csv")
                gantt_path = os.path.join(gantt_dir, f"{file_prefix}_gantt.png")
                sol_df.to_csv(detail_path, index=False)
                if make_gantt:
                    plot_ortools_gantt_with_due_dates(
                        res["solution_rows"],
                        gantt_path,
                        title=f"OR-Tools {scale} {base}",
                    )

                summary = {
                    "scale": scale,
                    "n_j": n_j,
                    "n_m": n_m,
                    "due_setting": due_setting,
                    "instance": base,
                    "instance_seed": instance_seed,
                    "due_seed": due_seed,
                    "range_low": range_low,
                    "range_high": range_high,
                    "time_limit_seconds": float(time_limit),
                    "makespan": res["makespan"],
                    "total_tardiness": res["total_tardiness"],
                    "obj": res["obj"],
                    "status": res["status"],
                    "solver_wall_time": round(res["time"], 3),
                    "elapsed_wall_time": round(elapsed_wall, 3),
                    "schedule_detail_csv": detail_path,
                    "gantt_png": gantt_path if make_gantt else "",
                }
                results.append(summary)
                pd.DataFrame(results).to_csv(output_path, index=False)
                print(f"  Result: MK={res['makespan']}, TD={res['total_tardiness']}, Obj={res['obj']:.1f} ({res['status']})")
            else:
                failed.append({"scale": scale, "instance": base, "reason": "no_feasible_solution"})
                print(f"  Failed to solve {fjs_file}")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise RuntimeError(f"No OR-Tools results produced under {base_dir}")

    by_scale = result_df.groupby(["scale", "n_j", "n_m", "due_setting"], as_index=False).agg({
        "makespan": "mean",
        "total_tardiness": "mean",
        "obj": "mean",
        "solver_wall_time": "mean",
    })
    by_due = result_df.groupby(["due_setting"], as_index=False).agg({
        "makespan": "mean",
        "total_tardiness": "mean",
        "obj": "mean",
        "solver_wall_time": "mean",
    })
    by_size = result_df.groupby(["n_j", "n_m"], as_index=False).agg({
        "makespan": "mean",
        "total_tardiness": "mean",
        "obj": "mean",
        "solver_wall_time": "mean",
    })

    by_scale.to_csv(os.path.join(output_dir, "ortools_static_summary_by_scale.csv"), index=False)
    by_due.to_csv(os.path.join(output_dir, "ortools_static_summary_by_due.csv"), index=False)
    by_size.to_csv(os.path.join(output_dir, "ortools_static_summary_by_size.csv"), index=False)
    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(output_dir, "ortools_static_failed.csv"), index=False)

    print("\n--- Summary by scale ---")
    print(by_scale)
    print(f"\nInstance results: {output_path}")
    print(f"Schedule details: {detail_dir}")
    print(f"Gantt charts: {gantt_dir}")
    print(f"Summary folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OR-Tools on generated static FJSP test instances.")
    parser.add_argument("--base_dir", type=str, default="or_instances_uniform_test_30_50")
    parser.add_argument("--time_limit", type=float, default=7200.0, help="Time limit per instance in seconds.")
    parser.add_argument("--output_root", type=str, default="or_tools_solutions")
    parser.add_argument("--no_gantt", action="store_true", help="Skip Gantt chart generation.")
    parser.add_argument("--log_solutions", action="store_true", help="Print every intermediate OR-Tools solution.")
    parser.add_argument("--resume_dir", type=str, default="", help="Existing output folder to resume from.")
    args = parser.parse_args()
    run_benchmark(
        base_dir=args.base_dir,
        time_limit=args.time_limit,
        output_root=args.output_root,
        make_gantt=not args.no_gantt,
        log_solutions=args.log_solutions,
        resume_dir=args.resume_dir,
    )
