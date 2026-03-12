
import os
import json
import time
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from data_utils import text_to_matrix

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

def solve_fjsp_with_ortools(fjs_path, json_path, time_limit=300):
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
    printer = IntermediateSolutionPrinter(makespan, total_tardiness_vars)
    status = solver.Solve(model, printer)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol_rows = []
        for (j, o), choices in all_tasks_vars.items():
            for m_idx, p, s_var, e_var in choices:
                if solver.Value(p):
                    start_val = solver.Value(s_var)
                    end_val = solver.Value(e_var)
                    sol_rows.append({
                        "Job": j, "Op": o, "Machine": m_idx,
                        "Start": start_val, "End": end_val,
                        "Duration": end_val - start_val
                    })
        return {
            "makespan": solver.Value(makespan),
            "total_tardiness": sum(solver.Value(t) for t in total_tardiness_vars),
            "status": solver.StatusName(status),
            "time": solver.WallTime(),
            "solution_rows": sol_rows
        }
    return None

def run_benchmark(base_dir="or_instances_uniform"):
    output_path = "or_tools_benchmark_results.csv"
    solution_dir = "or_tools_solutions"
    os.makedirs(solution_dir, exist_ok=True)
    
    results = []
    processed = set()
    if os.path.exists(output_path):
        try:
            old = pd.read_csv(output_path)
            results = old.to_dict('records')
            processed = set(old['instance'].tolist())
            print(f"Resuming. Found {len(processed)} existing results.")
        except: pass

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for scale in subdirs:
        path = os.path.join(base_dir, scale)
        files = [f for f in os.listdir(path) if f.endswith(".fjs")]
        print(f"\n--- Scale: {scale} ---")
        for fjs_file in files:
            base = fjs_file.replace(".fjs", "")
            if base in processed: continue
            
            print(f"Solving {fjs_file}...")
            f_path = os.path.join(path, fjs_file)
            j_path = os.path.join(path, f"{base}.json")
            
            res = solve_fjsp_with_ortools(f_path, j_path, time_limit=7200) 
            if res:
                sol_df = pd.DataFrame(res["solution_rows"])
                sol_df.to_csv(os.path.join(solution_dir, f"{base}_sol.csv"), index=False)
                
                summary = {
                    "scale": scale, "instance": base,
                    "makespan": res["makespan"], "total_tardiness": res["total_tardiness"],
                    "status": res["status"], "solve_time": round(res["time"], 2)
                }
                results.append(summary)
                pd.DataFrame(results).to_csv(output_path, index=False)
                print(f"  Result: MK={res['makespan']}, TD={res['total_tardiness']} ({res['status']})")
            else:
                print(f"  Failed to solve {fjs_file}")

if __name__ == "__main__":
    run_benchmark()
