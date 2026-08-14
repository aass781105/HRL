"""Analyze lower-level PPO schedule stability after adding jobs.

The script keeps one base instance per scale and evaluates nested job additions
of +1, +5, +10, +20, and +30 jobs. It reports operation start-time drift,
machine assignment changes, and machine-level operation sequence changes.
"""

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
        description="Analyze lower-level PPO schedule stability after adding jobs."
    )
    parser.add_argument(
        "--config",
        default=r"yaml跑不同種子碼\train_ll_u1030_esttd_odprog.yml",
        help="YAML config used to initialize and load the lower-level PPO model.",
    )
    parser.add_argument(
        "--instance_root",
        default=r"instances\or_instances_uniform_test_30_50_due_scaled",
    )
    parser.add_argument("--scales", nargs="+", default=["30x5", "40x5", "50x5"])
    parser.add_argument("--due", default="tight")
    parser.add_argument(
        "--add_counts",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 30],
    )
    parser.add_argument(
        "--selection_modes",
        nargs="+",
        default=["random", "slack_large"],
        choices=["random", "slack_large"],
    )
    parser.add_argument("--random_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        default=r"analysis_results\ll_jobset_stability_schedule",
    )
    return parser.parse_args()


LOCAL_ARGS = parse_local_args()

# params.py parses command-line arguments during import. Keep only its config.
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
    values = np.asarray(row, dtype=np.float64)
    values = values[values > 0]
    return float(values.mean()) if values.size else 0.0


def split_jobs(job_lengths, op_pt, due_dates, source_instance):
    jobs = []
    cursor = 0
    for source_job_id, length in enumerate(np.asarray(job_lengths, dtype=int).tolist()):
        end = cursor + int(length)
        pt_block = np.asarray(op_pt[cursor:end], dtype=np.float64).copy()
        total_mean_pt = float(sum(positive_mean(row) for row in pt_block))
        due = float(due_dates[source_job_id])
        jobs.append(
            {
                "source_instance": str(source_instance),
                "source_job_id": int(source_job_id),
                "ops": pt_block,
                "length": int(length),
                "due": due,
                "total_mean_pt": total_mean_pt,
                "slack": due - total_mean_pt,
            }
        )
        cursor = end
    return jobs


def merge_jobs(jobs):
    job_lengths = np.asarray([int(job["length"]) for job in jobs], dtype=np.int64)
    op_pt = np.vstack([np.asarray(job["ops"], dtype=np.float64) for job in jobs])
    due_dates = np.asarray([float(job["due"]) for job in jobs], dtype=np.float64)
    return job_lengths, op_pt, due_dates


def load_instance(txt_path):
    txt_path = Path(txt_path)
    json_path = txt_path.with_suffix(".json")
    with txt_path.open("r", encoding="utf-8") as file:
        job_lengths, op_pt = text_to_matrix(file.readlines())
    with json_path.open("r", encoding="utf-8") as file:
        due_data = json.load(file)
    due_dates = load_due_dates_json(due_data)
    return (
        np.asarray(job_lengths, dtype=np.int64),
        np.asarray(op_pt, dtype=np.float64),
        due_dates,
    )


def find_instances(instance_root, scale, due):
    folder = Path(instance_root) / f"{scale}_{due}"
    files = sorted(folder.glob("*.fjs"))
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 .fjs files for base/donor under {folder}"
        )
    return files


def donor_order(donor_jobs, mode, rng):
    if mode == "random":
        return rng.permutation(len(donor_jobs)).astype(int).tolist()
    if mode == "slack_large":
        return sorted(
            range(len(donor_jobs)),
            key=lambda index: (donor_jobs[index]["slack"], index),
            reverse=True,
        )
    raise ValueError(f"Unknown selection mode: {mode}")


def build_added_variant(base_jobs, donor_jobs, ordered_indices, add_count):
    add_count = int(add_count)
    if add_count <= 0:
        raise ValueError("add_count must be positive")
    if add_count > len(ordered_indices):
        raise ValueError(
            f"Cannot add {add_count} jobs; donor pool has {len(ordered_indices)} jobs"
        )

    variant_jobs = []
    for job in base_jobs:
        copied = dict(job)
        copied["origin"] = "base"
        copied["base_job_id"] = int(job["source_job_id"])
        variant_jobs.append(copied)

    selected_jobs = []
    for index in ordered_indices[:add_count]:
        job = donor_jobs[index]
        copied = dict(job)
        copied["origin"] = "donor"
        copied["base_job_id"] = None
        copied["added_rank"] = len(selected_jobs) + 1
        variant_jobs.append(copied)
        selected_jobs.append(copied)

    variant_to_base = [job["base_job_id"] for job in variant_jobs]
    return variant_jobs, variant_to_base, selected_jobs


def load_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)
    configs.ll_eval_action_selection = "greedy"
    configs.ll_rollout_k = 1

    ppo = ll_ppo_initialize()
    model_path = str(
        getattr(configs, "ppo_model_path", "")
        or getattr(configs, "ll_ppo_model_path", "")
    )
    model_path = resolve_lower_level_weight_path(
        model_path, getattr(configs, "data_source", "SD2")
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Lower-level PPO model not found: {model_path}")
    ppo.policy.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    ppo.policy.to(device)
    ppo.policy.eval()
    return ppo, model_path


def run_greedy_episode(ppo, job_lengths, op_pt, due_dates):
    env = LLFJSPEnv(n_j=len(job_lengths), n_m=op_pt.shape[1])
    state = env.set_initial_data(
        job_length_list=[job_lengths],
        op_pt_list=[op_pt],
        due_date_list=[due_dates],
        true_due_date_list=[due_dates],
    )

    while True:
        with torch.no_grad():
            policy_output, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
            action = greedy_select_action(policy_output)
        state, _, done, _ = env.step(action.cpu().numpy())
        if bool(np.asarray(done).all()):
            break

    rows = extract_schedule_rows(env, env_idx=0)
    makespan = float(env.current_makespan[0])
    tardiness = float(env.accumulated_tardiness[0])
    return {
        "makespan": makespan,
        "tardiness": tardiness,
        "obj": 0.5 * makespan + 0.5 * tardiness,
        "rows": rows,
    }


def build_operation_map(rows, variant_to_base):
    operations = {}
    for row in rows:
        variant_job = int(row["Job"])
        if variant_job >= len(variant_to_base):
            continue
        base_job = variant_to_base[variant_job]
        if base_job is None:
            continue
        key = (int(base_job), int(row["Op"]))
        record = {
            "job_id": int(base_job),
            "op": int(row["Op"]),
            "machine": int(row["Machine"]),
            "start": float(row["Start"]),
            "end": float(row["End"]),
            "duration": float(row["Duration"]),
        }
        operations[key] = record
    return operations


def build_full_machine_sequences(rows, variant_to_base):
    """Keep added jobs in the sequence while preserving common-job keys."""
    sequences = {}
    sortable = {}
    for row in rows:
        variant_job = int(row["Job"])
        if variant_job >= len(variant_to_base):
            continue
        base_job = variant_to_base[variant_job]
        if base_job is None:
            key = ("new", variant_job, int(row["Op"]))
        else:
            key = (int(base_job), int(row["Op"]))
        machine = int(row["Machine"])
        sequences.setdefault(machine, []).append(key)
        sortable[key] = (
            float(row["Start"]),
            float(row["End"]),
            str(key),
        )

    for machine, keys in sequences.items():
        sequences[machine] = sorted(keys, key=lambda key: sortable[key])
    return sequences


def sequence_key_label(key):
    if isinstance(key[0], str):
        return f"N{key[1]}O{key[2]}"
    return f"J{key[0]}O{key[1]}"


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else 0.0


def compare_operations(base_rows, variant_rows, base_mapping, variant_mapping):
    base_ops = build_operation_map(base_rows, base_mapping)
    variant_ops = build_operation_map(variant_rows, variant_mapping)
    base_sequences = build_full_machine_sequences(base_rows, base_mapping)
    variant_sequences = build_full_machine_sequences(variant_rows, variant_mapping)
    common_keys = sorted(set(base_ops) & set(variant_ops))

    operation_rows = []
    machine_changes = 0
    for job_id, op in common_keys:
        base = base_ops[(job_id, op)]
        variant = variant_ops[(job_id, op)]
        machine_changed = int(base["machine"] != variant["machine"])
        machine_changes += machine_changed
        operation_rows.append(
            {
                "job_id": job_id,
                "op": op,
                "base_machine": base["machine"],
                "variant_machine": variant["machine"],
                "machine_changed": machine_changed,
                "base_start": base["start"],
                "variant_start": variant["start"],
                "start_shift": variant["start"] - base["start"],
                "abs_start_shift": abs(variant["start"] - base["start"]),
                "base_end": base["end"],
                "variant_end": variant["end"],
                "end_shift": variant["end"] - base["end"],
            }
        )

    same_machine_keys = [
        key
        for key in common_keys
        if base_ops[key]["machine"] == variant_ops[key]["machine"]
    ]
    base_positions = {
        key: position
        for machine_keys in base_sequences.values()
        for position, key in enumerate(machine_keys)
    }
    variant_positions = {
        key: position
        for machine_keys in variant_sequences.values()
        for position, key in enumerate(machine_keys)
    }

    position_shifts = []
    sequence_pairs = 0
    sequence_flips = 0
    for index, key_a in enumerate(same_machine_keys):
        for key_b in same_machine_keys[index + 1 :]:
            if base_ops[key_a]["machine"] != base_ops[key_b]["machine"]:
                continue
            if variant_ops[key_a]["machine"] != variant_ops[key_b]["machine"]:
                continue
            base_order = np.sign(base_positions[key_a] - base_positions[key_b])
            variant_order = np.sign(
                variant_positions[key_a] - variant_positions[key_b]
            )
            if base_order == 0 or variant_order == 0:
                continue
            sequence_pairs += 1
            if base_order != variant_order:
                sequence_flips += 1

    for key in same_machine_keys:
        position_shifts.append(variant_positions[key] - base_positions[key])

    start_shifts = [row["start_shift"] for row in operation_rows]
    abs_start_shifts = [row["abs_start_shift"] for row in operation_rows]
    return {
        "operation_rows": operation_rows,
        "machine_sequence_rows": build_machine_sequence_rows(
            base_sequences,
            variant_sequences,
        ),
        "common_ops": len(common_keys),
        "machine_changed_ops": machine_changes,
        "machine_change_rate": (
            float(machine_changes / len(common_keys)) if common_keys else 0.0
        ),
        "mean_start_shift": float(np.mean(start_shifts)) if start_shifts else 0.0,
        "mean_abs_start_shift": (
            float(np.mean(abs_start_shifts)) if abs_start_shifts else 0.0
        ),
        "p50_abs_start_shift": percentile(abs_start_shifts, 50),
        "p90_abs_start_shift": percentile(abs_start_shifts, 90),
        "max_abs_start_shift": max(abs_start_shifts) if abs_start_shifts else 0.0,
        "sequence_common_ops": len(same_machine_keys),
        "sequence_pairs": sequence_pairs,
        "sequence_flip_rate": (
            float(sequence_flips / sequence_pairs) if sequence_pairs else 0.0
        ),
        "mean_sequence_position_shift": (
            float(np.mean(np.abs(position_shifts))) if position_shifts else 0.0
        ),
        "p90_sequence_position_shift": percentile(np.abs(position_shifts), 90),
        "max_sequence_position_shift": (
            float(np.max(np.abs(position_shifts))) if position_shifts else 0.0
        ),
    }


def build_machine_sequence_rows(base_sequences, variant_sequences):
    rows = []
    for machine in sorted(set(base_sequences) | set(variant_sequences)):
        base_keys = base_sequences.get(machine, [])
        variant_keys = variant_sequences.get(machine, [])
        common_keys = sorted(set(base_keys) & set(variant_keys))
        base_pos = {key: index for index, key in enumerate(base_keys)}
        variant_pos = {key: index for index, key in enumerate(variant_keys)}
        shifts = [variant_pos[key] - base_pos[key] for key in common_keys]
        rows.append(
            {
                "machine": machine,
                "base_sequence_length": len(base_keys),
                "variant_sequence_length": len(variant_keys),
                "common_sequence_ops": len(common_keys),
                "mean_abs_position_shift": (
                    float(np.mean(np.abs(shifts))) if shifts else 0.0
                ),
                "p90_abs_position_shift": percentile(np.abs(shifts), 90),
                "max_abs_position_shift": (
                    float(np.max(np.abs(shifts))) if shifts else 0.0
                ),
                "base_sequence": "|".join(
                    sequence_key_label(key) for key in base_keys
                ),
                "variant_sequence": "|".join(
                    sequence_key_label(key) for key in variant_keys
                ),
            }
        )
    return rows


def selected_job_rows(scale, mode, trial, add_count, selected_jobs):
    rows = []
    for job in selected_jobs:
        rows.append(
            {
                "scale": scale,
                "selection_mode": mode,
                "trial": trial,
                "add_count": add_count,
                "added_rank": int(job["added_rank"]),
                "source_instance": job["source_instance"],
                "source_job_id": int(job["source_job_id"]),
                "due": float(job["due"]),
                "total_mean_pt": float(job["total_mean_pt"]),
                "slack": float(job["slack"]),
            }
        )
    return rows


def main():
    output_dir = Path(LOCAL_ARGS.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(LOCAL_ARGS.seed))
    ppo, model_path = load_ppo()

    print(f"Loaded lower-level PPO: {model_path}")
    print(f"Output: {output_dir}")

    summary_rows = []
    operation_rows_all = []
    machine_sequence_rows_all = []
    selected_rows_all = []

    for scale in LOCAL_ARGS.scales:
        files = find_instances(LOCAL_ARGS.instance_root, scale, LOCAL_ARGS.due)
        base_file = files[0]
        base_jl, base_pt, base_due = load_instance(base_file)
        base_jobs = split_jobs(base_jl, base_pt, base_due, base_file.name)
        donor_jobs = []
        for donor_file in files[1:]:
            donor_jl, donor_pt, donor_due = load_instance(donor_file)
            donor_jobs.extend(
                split_jobs(donor_jl, donor_pt, donor_due, donor_file.name)
            )

        base_mapping = [job["source_job_id"] for job in base_jobs]
        base_result = run_greedy_episode(ppo, base_jl, base_pt, base_due)
        summary_rows.append(
            {
                "scale": scale,
                "selection_mode": "base",
                "trial": 0,
                "add_count": 0,
                "base_jobs": len(base_jobs),
                "variant_jobs": len(base_jobs),
                "makespan": base_result["makespan"],
                "tardiness": base_result["tardiness"],
                "obj": base_result["obj"],
                "common_ops": 0,
                "machine_change_rate": 0.0,
                "mean_abs_start_shift": 0.0,
                "p90_abs_start_shift": 0.0,
                "max_abs_start_shift": 0.0,
                "sequence_flip_rate": 0.0,
                "mean_sequence_position_shift": 0.0,
                "p90_sequence_position_shift": 0.0,
                "max_sequence_position_shift": 0.0,
            }
        )

        for mode in LOCAL_ARGS.selection_modes:
            for trial in range(int(LOCAL_ARGS.random_trials)):
                mode_rng = rng if mode == "random" else np.random.default_rng(0)
                ordered_indices = donor_order(donor_jobs, mode, mode_rng)
                for add_count in sorted(set(map(int, LOCAL_ARGS.add_counts))):
                    jobs, variant_mapping, selected_jobs = build_added_variant(
                        base_jobs,
                        donor_jobs,
                        ordered_indices,
                        add_count,
                    )
                    variant_jl, variant_pt, variant_due = merge_jobs(jobs)
                    variant_result = run_greedy_episode(
                        ppo, variant_jl, variant_pt, variant_due
                    )
                    comparison = compare_operations(
                        base_result["rows"],
                        variant_result["rows"],
                        base_mapping,
                        variant_mapping,
                    )
                    summary_rows.append(
                        {
                            "scale": scale,
                            "selection_mode": mode,
                            "trial": trial,
                            "add_count": add_count,
                            "base_jobs": len(base_jobs),
                            "variant_jobs": len(jobs),
                            "makespan": variant_result["makespan"],
                            "tardiness": variant_result["tardiness"],
                            "obj": variant_result["obj"],
                            "mk_delta": variant_result["makespan"]
                            - base_result["makespan"],
                            "td_delta": variant_result["tardiness"]
                            - base_result["tardiness"],
                            "obj_delta": variant_result["obj"] - base_result["obj"],
                            "added_total_work": float(
                                sum(job["total_mean_pt"] for job in selected_jobs)
                            ),
                            **{
                                key: value
                                for key, value in comparison.items()
                                if key
                                not in {"operation_rows", "machine_sequence_rows"}
                            },
                        }
                    )
                    prefix = {
                        "scale": scale,
                        "selection_mode": mode,
                        "trial": trial,
                        "add_count": add_count,
                    }
                    operation_rows_all.extend(
                        {**prefix, **row} for row in comparison["operation_rows"]
                    )
                    machine_sequence_rows_all.extend(
                        {**prefix, **row}
                        for row in comparison["machine_sequence_rows"]
                    )
                    selected_rows_all.extend(
                        selected_job_rows(
                            scale, mode, trial, add_count, selected_jobs
                        )
                    )

        print(f"Finished {scale}: base={base_file.name}")

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(operation_rows_all).to_csv(
        output_dir / "operation_drift.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(machine_sequence_rows_all).to_csv(
        output_dir / "machine_sequence_drift.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selected_rows_all).to_csv(
        output_dir / "selected_added_jobs.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Wrote: {output_dir / 'summary.csv'}")
    print(f"Wrote: {output_dir / 'operation_drift.csv'}")
    print(f"Wrote: {output_dir / 'machine_sequence_drift.csv'}")
    print(f"Wrote: {output_dir / 'selected_added_jobs.csv'}")


if __name__ == "__main__":
    main()
