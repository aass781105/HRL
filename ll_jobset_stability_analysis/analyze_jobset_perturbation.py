import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_local_args():
    parser = argparse.ArgumentParser(
        description="Test LL PPO schedule stability when adding/removing a few jobs."
    )
    parser.add_argument(
        "--config",
        default=r"yaml跑不同種子碼\train_ll_u1030_esttd_odprog.yml",
        help="YAML config used to load the lower-level PPO model.",
    )
    parser.add_argument(
        "--instance_root",
        default=r"instances\or_instances_uniform_test_30_50_due_scaled",
        help="Root folder containing 30x5_tight/40x5_tight/50x5_tight instances.",
    )
    parser.add_argument("--scales", nargs="+", default=["30x5", "40x5", "50x5"])
    parser.add_argument("--due", default="tight")
    parser.add_argument("--deltas", nargs="+", type=int, default=[-2, -1, 0, 1, 2])
    parser.add_argument(
        "--random_trials",
        type=int,
        default=1,
        help="Random variants per nonzero delta. Slack-large uses one deterministic variant.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        default=r"analysis_results\ll_jobset_stability",
        help="Output folder root.",
    )
    return parser.parse_args()


LOCAL_ARGS = parse_local_args()

# params.py parses sys.argv at import time. Keep only --config visible to project config parsing.
sys.argv = [sys.argv[0], "--config", LOCAL_ARGS.config]
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common_utils import greedy_select_action, resolve_lower_level_weight_path  # noqa: E402
from data_utils import text_to_matrix  # noqa: E402
from evaluate_range_due_test_batch import extract_schedule_rows, load_due_dates_json  # noqa: E402
from ll_fjsp_env import LLFJSPEnv  # noqa: E402
from model.ll_ppo import ll_ppo_initialize  # noqa: E402
from params import configs  # noqa: E402


def positive_mean(row):
    vals = np.asarray(row, dtype=np.float64)
    vals = vals[vals > 0]
    return float(vals.mean()) if vals.size else 0.0


def split_jobs(job_lengths, op_pt, due_dates):
    jobs = []
    start = 0
    for job_id, length in enumerate(np.asarray(job_lengths, dtype=int).tolist()):
        end = start + int(length)
        pt_block = np.asarray(op_pt[start:end], dtype=np.float64).copy()
        total_mean_pt = float(sum(positive_mean(row) for row in pt_block))
        due = float(due_dates[job_id])
        jobs.append(
            {
                "source_job_id": int(job_id),
                "ops": pt_block,
                "length": int(length),
                "due": due,
                "total_mean_pt": total_mean_pt,
                "slack": due - total_mean_pt,
            }
        )
        start = end
    return jobs


def merge_jobs(jobs):
    job_lengths = np.asarray([int(job["length"]) for job in jobs], dtype=np.int64)
    op_pt = np.vstack([np.asarray(job["ops"], dtype=np.float64) for job in jobs])
    due_dates = np.asarray([float(job["due"]) for job in jobs], dtype=np.float64)
    return job_lengths, op_pt, due_dates


def load_instance(txt_path):
    txt_path = Path(txt_path)
    json_path = txt_path.with_suffix(".json")
    with txt_path.open("r", encoding="utf-8") as f:
        job_lengths, op_pt = text_to_matrix(f.readlines())
    with json_path.open("r", encoding="utf-8") as f:
        due_data = json.load(f)
    due_dates = load_due_dates_json(due_data)
    return np.asarray(job_lengths, dtype=np.int64), np.asarray(op_pt, dtype=np.float64), due_dates


def find_instances(instance_root, scale, due):
    folder = Path(instance_root) / f"{scale}_{due}"
    files = sorted(folder.glob("*.fjs"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least 2 .fjs files for base/donor under {folder}")
    return files


def choose_job_indices(jobs, count, mode, rng):
    count = int(count)
    if count <= 0:
        return []
    if count > len(jobs):
        raise ValueError(f"Cannot choose {count} jobs from pool of {len(jobs)}")
    if mode == "random":
        return sorted(rng.choice(len(jobs), size=count, replace=False).astype(int).tolist())
    if mode == "slack_large":
        ranked = sorted(range(len(jobs)), key=lambda idx: (jobs[idx]["slack"], idx), reverse=True)
        return sorted(ranked[:count])
    raise ValueError(f"Unknown selection mode: {mode}")


def build_variant(base_jobs, donor_jobs, delta, mode, rng):
    variant_jobs = []
    metadata = {
        "removed_source_ids": [],
        "added_donor_ids": [],
        "selected_slacks": [],
    }

    if delta < 0:
        remove_indices = set(choose_job_indices(base_jobs, abs(delta), mode, rng))
        metadata["removed_source_ids"] = [base_jobs[idx]["source_job_id"] for idx in sorted(remove_indices)]
        metadata["selected_slacks"] = [base_jobs[idx]["slack"] for idx in sorted(remove_indices)]
        for idx, job in enumerate(base_jobs):
            if idx not in remove_indices:
                copied = dict(job)
                copied["origin"] = "base"
                copied["base_job_id"] = int(job["source_job_id"])
                variant_jobs.append(copied)
    elif delta > 0:
        for job in base_jobs:
            copied = dict(job)
            copied["origin"] = "base"
            copied["base_job_id"] = int(job["source_job_id"])
            variant_jobs.append(copied)

        add_indices = choose_job_indices(donor_jobs, delta, mode, rng)
        metadata["added_donor_ids"] = [donor_jobs[idx]["source_job_id"] for idx in add_indices]
        metadata["selected_slacks"] = [donor_jobs[idx]["slack"] for idx in add_indices]
        for idx in add_indices:
            job = donor_jobs[idx]
            copied = dict(job)
            copied["origin"] = "donor"
            copied["base_job_id"] = None
            variant_jobs.append(copied)
    else:
        for job in base_jobs:
            copied = dict(job)
            copied["origin"] = "base"
            copied["base_job_id"] = int(job["source_job_id"])
            variant_jobs.append(copied)

    mapping = [job.get("base_job_id") for job in variant_jobs]
    return variant_jobs, mapping, metadata


def load_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    configs.ll_eval_action_selection = "greedy"
    configs.ll_rollout_k = 1

    ppo = ll_ppo_initialize()
    model_path = str(getattr(configs, "ppo_model_path", "") or getattr(configs, "ll_ppo_model_path", ""))
    model_path = resolve_lower_level_weight_path(model_path, getattr(configs, "data_source", "SD2"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Lower-level PPO model not found: {model_path}")
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ppo.policy.to(device)
    ppo.policy.eval()
    return ppo, model_path, device


def run_greedy_episode(ppo, job_lengths, op_pt, due_dates):
    n_j = int(len(job_lengths))
    n_m = int(op_pt.shape[1])
    env = LLFJSPEnv(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(
        job_length_list=[job_lengths],
        op_pt_list=[op_pt],
        due_date_list=[due_dates],
        true_due_date_list=[due_dates],
    )

    while True:
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
            action = greedy_select_action(pi)
        state, _, done, _ = env.step(action.cpu().numpy())
        if bool(done.all()):
            break

    rows = extract_schedule_rows(env, env_idx=0)
    return {
        "makespan": float(env.current_makespan[0]),
        "tardiness": float(env.accumulated_tardiness[0]),
        "obj": 0.5 * float(env.current_makespan[0]) + 0.5 * float(env.accumulated_tardiness[0]),
        "rows": rows,
    }


def schedule_maps(rows, variant_to_base):
    completion = {}
    tardiness = {}
    op_machine = {}
    op_start = {}
    op_end = {}

    for row in rows:
        variant_job = int(row["Job"])
        if variant_job >= len(variant_to_base):
            continue
        base_job = variant_to_base[variant_job]
        if base_job is None:
            continue
        base_job = int(base_job)
        op = int(row["Op"])
        completion[base_job] = max(completion.get(base_job, -np.inf), float(row["Job_Completion"]))
        tardiness[base_job] = max(tardiness.get(base_job, 0.0), float(row["Job_Tardiness"]))
        op_machine[(base_job, op)] = int(row["Machine"])
        op_start[(base_job, op)] = float(row["Start"])
        op_end[(base_job, op)] = float(row["End"])
    return completion, tardiness, op_machine, op_start, op_end


def order_flip_rate(base_completion, variant_completion, shared_ids):
    ids = sorted(shared_ids)
    total = 0
    flips = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = ids[i]
            b = ids[j]
            base_sign = np.sign(base_completion[a] - base_completion[b])
            var_sign = np.sign(variant_completion[a] - variant_completion[b])
            if base_sign == 0 or var_sign == 0:
                continue
            total += 1
            if base_sign != var_sign:
                flips += 1
    return float(flips / total) if total else 0.0


def compare_to_base(base_result, variant_result, base_mapping, variant_mapping):
    base_completion, base_tardiness, base_op_machine, _, _ = schedule_maps(base_result["rows"], base_mapping)
    var_completion, var_tardiness, var_op_machine, _, _ = schedule_maps(variant_result["rows"], variant_mapping)

    shared_ids = sorted(set(base_completion.keys()) & set(var_completion.keys()))
    shifts = np.asarray([abs(var_completion[j] - base_completion[j]) for j in shared_ids], dtype=np.float64)
    shared_base_td = float(sum(base_tardiness.get(j, 0.0) for j in shared_ids))
    shared_var_td = float(sum(var_tardiness.get(j, 0.0) for j in shared_ids))

    common_ops = sorted(set(base_op_machine.keys()) & set(var_op_machine.keys()))
    machine_changes = [base_op_machine[key] != var_op_machine[key] for key in common_ops]

    summary = {
        "shared_jobs": len(shared_ids),
        "shared_td_base": shared_base_td,
        "shared_td_variant": shared_var_td,
        "shared_td_delta": shared_var_td - shared_base_td,
        "mean_abs_completion_shift": float(shifts.mean()) if shifts.size else 0.0,
        "p90_abs_completion_shift": float(np.percentile(shifts, 90)) if shifts.size else 0.0,
        "max_abs_completion_shift": float(shifts.max()) if shifts.size else 0.0,
        "order_flip_rate": order_flip_rate(base_completion, var_completion, shared_ids),
        "machine_assignment_change_rate": float(np.mean(machine_changes)) if machine_changes else 0.0,
    }

    drift_rows = []
    for job_id in shared_ids:
        drift_rows.append(
            {
                "base_job_id": job_id,
                "base_completion": base_completion[job_id],
                "variant_completion": var_completion[job_id],
                "completion_shift": var_completion[job_id] - base_completion[job_id],
                "abs_completion_shift": abs(var_completion[job_id] - base_completion[job_id]),
                "base_tardiness": base_tardiness.get(job_id, 0.0),
                "variant_tardiness": var_tardiness.get(job_id, 0.0),
                "tardiness_delta": var_tardiness.get(job_id, 0.0) - base_tardiness.get(job_id, 0.0),
            }
        )
    return summary, drift_rows


def main():
    rng = np.random.default_rng(int(LOCAL_ARGS.seed))
    out_dir = Path(LOCAL_ARGS.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo, model_path, _ = load_ppo()
    print(f"Loaded lower-level PPO: {model_path}")
    print(f"Output: {out_dir}")

    summary_rows = []
    drift_rows_all = []
    selected_rows = []

    for scale in LOCAL_ARGS.scales:
        files = find_instances(LOCAL_ARGS.instance_root, scale, LOCAL_ARGS.due)
        base_file = files[0]
        donor_files = files[1:]
        base_jl, base_pt, base_due = load_instance(base_file)
        base_jobs = split_jobs(base_jl, base_pt, base_due)

        donor_jobs = []
        for donor_file in donor_files:
            donor_jl, donor_pt, donor_due = load_instance(donor_file)
            donor_jobs.extend(split_jobs(donor_jl, donor_pt, donor_due))

        base_mapping = [job["source_job_id"] for job in base_jobs]
        base_result = run_greedy_episode(ppo, base_jl, base_pt, base_due)

        selected_rows.append(
            {
                "scale": scale,
                "base_file": str(base_file),
                "donor_files": " | ".join(str(p) for p in donor_files),
                "base_makespan": base_result["makespan"],
                "base_tardiness": base_result["tardiness"],
                "base_obj": base_result["obj"],
            }
        )

        for delta in LOCAL_ARGS.deltas:
            if delta == 0:
                jobs, mapping, meta = build_variant(base_jobs, donor_jobs, 0, "base", rng)
                jl, pt, due = merge_jobs(jobs)
                result = base_result
                compare_summary, drift_rows = compare_to_base(base_result, result, base_mapping, mapping)
                summary_rows.append(
                    {
                        "scale": scale,
                        "delta": 0,
                        "selection_mode": "base",
                        "trial": 0,
                        "n_jobs": int(len(jl)),
                        "makespan": result["makespan"],
                        "tardiness": result["tardiness"],
                        "obj": result["obj"],
                        "mk_delta": 0.0,
                        "td_delta": 0.0,
                        "obj_delta": 0.0,
                        "removed_source_ids": "",
                        "added_donor_ids": "",
                        "selected_slacks": "",
                        **compare_summary,
                    }
                )
                for row in drift_rows:
                    drift_rows_all.append({"scale": scale, "delta": 0, "selection_mode": "base", "trial": 0, **row})
                continue

            modes = ["random", "slack_large"]
            for mode in modes:
                trials = int(LOCAL_ARGS.random_trials) if mode == "random" else 1
                for trial in range(trials):
                    jobs, mapping, meta = build_variant(base_jobs, donor_jobs, delta, mode, rng)
                    jl, pt, due = merge_jobs(jobs)
                    result = run_greedy_episode(ppo, jl, pt, due)
                    compare_summary, drift_rows = compare_to_base(base_result, result, base_mapping, mapping)

                    row_prefix = {
                        "scale": scale,
                        "delta": int(delta),
                        "selection_mode": mode,
                        "trial": int(trial),
                    }
                    summary_rows.append(
                        {
                            **row_prefix,
                            "n_jobs": int(len(jl)),
                            "makespan": result["makespan"],
                            "tardiness": result["tardiness"],
                            "obj": result["obj"],
                            "mk_delta": result["makespan"] - base_result["makespan"],
                            "td_delta": result["tardiness"] - base_result["tardiness"],
                            "obj_delta": result["obj"] - base_result["obj"],
                            "removed_source_ids": ",".join(map(str, meta["removed_source_ids"])),
                            "added_donor_ids": ",".join(map(str, meta["added_donor_ids"])),
                            "selected_slacks": ",".join(f"{x:.3f}" for x in meta["selected_slacks"]),
                            **compare_summary,
                        }
                    )
                    for drift in drift_rows:
                        drift_rows_all.append({**row_prefix, **drift})

        print(f"Finished {scale}: base={base_file.name}")

    pd.DataFrame(selected_rows).to_csv(out_dir / "selected_instances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(drift_rows_all).to_csv(out_dir / "shared_job_drift.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote: {out_dir / 'summary.csv'}")
    print(f"Wrote: {out_dir / 'shared_job_drift.csv'}")


if __name__ == "__main__":
    main()
