"""Read-only audit for lower-level PPO time/state normalization.

This script does not change the environment implementation. It compares the
physical ``true_*`` values with the values exposed to the low-level policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--base-time", type=float, default=200.0)
    return parser.parse_args()


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
    }


def make_instance(rng, jobs, machines):
    job_lengths = np.full(jobs, machines, dtype=int)
    op_pt = rng.integers(10, 91, size=(jobs * machines, machines), dtype=int)
    # Keep every operation flexible, while preserving different PT values.
    return job_lengths, op_pt


def due_dates(job_lengths, op_pt, factor=1.35):
    means = np.mean(op_pt.reshape(len(job_lengths), -1, op_pt.shape[1]), axis=(1, 2))
    total_work = means * job_lengths
    return total_work * factor


def finite_min(masked_values, mask):
    values = np.where(mask, masked_values, np.inf)
    return np.min(values, axis=(1, 2))


def audit_snapshot(env, base_time, label):
    pt_scale = float(env.pt_scale)
    valid_pt = env.process_relation
    pt_expected = env.true_op_pt / pt_scale
    pt_error = env.unmasked_op_pt[valid_pt] - pt_expected[valid_pt]
    pt_ratio = env.unmasked_op_pt[valid_pt] / np.maximum(pt_expected[valid_pt], 1e-12)

    due_expected = (env.true_due_date - base_time) / pt_scale
    due_error = env.due_date - due_expected

    work_expected = env.true_op_match_job_remain_work / pt_scale
    work_mask = env.true_op_match_job_remain_work > 1e-12
    work_error = env.op_match_job_remain_work[work_mask] - work_expected[work_mask]
    work_ratio = env.op_match_job_remain_work[work_mask] / np.maximum(work_expected[work_mask], 1e-12)

    machine_expected = (env.true_mch_free_time - base_time) / pt_scale
    machine_error = env.mch_free_time - machine_expected

    ready_expected = (env.true_candidate_free_time - base_time) / pt_scale
    ready_error = env.candidate_free_time - ready_expected

    true_pair = env._compute_pair_free_time(true_time=True)
    true_next_abs = finite_min(true_pair, ~env.candidate_process_relation)
    true_next_expected = (true_next_abs - base_time) / pt_scale
    next_error = env.next_schedule_time - true_next_expected

    candidate_ops = env.candidate[0]
    state_slack = env.raw_fea_j[0, candidate_ops, 11]
    candidate_work = env.true_op_match_job_remain_work[0, candidate_ops]
    candidate_due = env.true_due_date[0]
    slack_expected = (candidate_due - base_time) / pt_scale - true_next_expected[0] - candidate_work / pt_scale
    slack_error = state_slack - slack_expected

    return {
        "label": label,
        "pt_scale": pt_scale,
        "batch_pt_min": float(np.min(env.true_op_pt[valid_pt])),
        "batch_pt_max": float(np.max(env.true_op_pt[valid_pt])),
        "batch_pt_range": float(np.ptp(env.true_op_pt[valid_pt])),
        "pt_error": stats(pt_error),
        "pt_ratio": stats(pt_ratio),
        "due_error": stats(due_error),
        "remaining_work_error": stats(work_error),
        "remaining_work_ratio": stats(work_ratio),
        "machine_free_error": stats(machine_error),
        "candidate_ready_error": stats(ready_error),
        "next_schedule_error": stats(next_error),
        "candidate_slack_error": stats(slack_error),
        "sample": {
            "state_slack": np.round(state_slack, 6).tolist(),
            "expected_slack": np.round(slack_expected, 6).tolist(),
            "state_next_schedule_time": np.round(env.next_schedule_time, 6).tolist(),
            "expected_next_schedule_time": np.round(true_next_expected, 6).tolist(),
        },
    }


def run_case(job_lengths, op_pt, due, base_time, label, args):
    # Import after parsing so --device can be passed through params.py.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.argv = [sys.argv[0], "--device", str(args.device)]
    from ll_fjsp_env import LLFJSPEnv

    env = LLFJSPEnv(n_j=len(job_lengths), n_m=op_pt.shape[1])
    env.set_initial_data(
        [job_lengths],
        [op_pt],
        due_date_list=[due],
        true_due_date_list=[due],
    )

    reports = [audit_snapshot(env, base_time=0.0, label=f"{label}_initial")]

    valid = np.argwhere(~env.dynamic_pair_mask[0])
    job, machine = map(int, valid[0])
    env.step(np.asarray([job * env.number_of_machines + machine]))
    reports.append(audit_snapshot(env, base_time=0.0, label=f"{label}_after_one_step"))

    # Rebuild a dynamic-like state at a nonzero current time. The true values
    # remain absolute while the policy-facing time values are relative.
    dyn = LLFJSPEnv(n_j=len(job_lengths), n_m=op_pt.shape[1])
    dyn.set_initial_data(
        [job_lengths],
        [op_pt],
        due_date_list=[due - base_time],
        true_due_date_list=[due],
    )
    true_machine_free = np.asarray([[base_time + 37.0, base_time + 82.0, base_time + 11.0, base_time + 121.0, base_time + 54.0]])
    true_ready = np.linspace(base_time, base_time + 45.0, len(job_lengths), dtype=float)[None, :]
    dyn.true_mch_free_time[:, :] = true_machine_free
    dyn.mch_free_time[:, :] = (true_machine_free - base_time) / dyn.pt_scale
    dyn.true_candidate_free_time[:, :] = true_ready
    dyn.candidate_free_time[:, :] = (true_ready - base_time) / dyn.pt_scale
    dyn.rebuild_state_from_current()
    reports.append(audit_snapshot(dyn, base_time=base_time, label=f"{label}_dynamic_like"))
    return reports


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    job_lengths, op_pt = make_instance(rng, args.jobs, args.machines)
    due = due_dates(job_lengths, op_pt)
    reports = run_case(job_lengths, op_pt, due, args.base_time, "synthetic", args)

    print(json.dumps({"config": vars(args), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
