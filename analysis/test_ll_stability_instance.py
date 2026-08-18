"""Run the legacy lower-level PPO on one stability-training instance.

The instance is generated through the same helpers used by
``train_ll_stability_finetune.py``.  The legacy checkpoint is used only for
inference, so this script does not enable the new stability features or
stability reward.  Outputs are written to one timestamped directory:

* ``schedule_details.csv``: one row per scheduled operation, including the
  selected action and per-step reward components.
* ``job_details.csv``: job-level due date, release time, and old/new source.
* ``summary.csv`` and ``summary.json``: final makespan, tardiness, and metadata.
* ``gantt.png``: the resulting static schedule.

Examples from the repository root::

    python analysis/test_ll_stability_instance.py
    python analysis/test_ll_stability_instance.py --kind fresh --due-mode range3_tight --target-jobs 35
    python analysis/test_ll_stability_instance.py --kind reschedule --due-mode range3_mixed --seed 7 --action-mode sample
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "yaml_config" / "train_ll_stability_finetune.yml"),
        help="Configuration used by the current stability-training generator.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Legacy lower-level PPO checkpoint. Defaults to the YAML reference model.",
    )
    parser.add_argument(
        "--kind",
        choices=("fresh", "reschedule", "mixed"),
        default="reschedule",
        help="Sample type. mixed follows the configured fresh/reschedule ratio.",
    )
    parser.add_argument(
        "--due-mode",
        choices=("auto", "range3_loose", "range3_mixed", "range3_tight"),
        default="auto",
        help="Due-date mode. auto follows the configured hold schedule.",
    )
    parser.add_argument(
        "--target-jobs",
        type=int,
        default=None,
        help="Target job count. Defaults to the configured target range sampler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Generation seed. Defaults to seed_train in the YAML.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Training-style sample/update index used by the generator.",
    )
    parser.add_argument(
        "--require-overdue",
        action="store_true",
        help="Retry different seeds until the generated sample has at least one due<0 job.",
    )
    parser.add_argument(
        "--max-sample-retries",
        type=int,
        default=100,
        help="Maximum seed retries when --require-overdue is enabled.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("greedy", "sample"),
        default="greedy",
        help="Lower-level PPO action selection mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to analysis_results/lower_level_stability_test/.",
    )
    return parser.parse_args()


def _scalar(value) -> float:
    """Convert a scalar numpy/torch value into a Python float."""
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _resolve_checkpoint(path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("No legacy lower-level checkpoint was configured.")
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _build_op_maps(job_lengths) -> tuple[list[int], list[int]]:
    op_to_job: list[int] = []
    op_to_local: list[int] = []
    for job_id, length in enumerate(job_lengths):
        for op_id in range(int(length)):
            op_to_job.append(int(job_id))
            op_to_local.append(int(op_id))
    return op_to_job, op_to_local


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_policy(env, state, ppo, action_mode: str):
    import numpy as np
    import torch
    from common_utils import greedy_select_action, sample_action

    model_inputs = (
        state.fea_j_tensor,
        state.op_mask_tensor,
        state.candidate_tensor,
        state.fea_m_tensor,
        state.mch_mask_tensor,
        state.comp_idx_tensor,
        state.dynamic_pair_mask_tensor,
        state.fea_pairs_tensor,
    )
    with torch.no_grad():
        pi = ppo.policy.policy_only(*model_inputs)
        if action_mode == "sample":
            action_tensor, _ = sample_action(pi)
        else:
            action_tensor = greedy_select_action(pi)
    action_tensor = action_tensor.reshape(-1)
    action = int(action_tensor[0].item())
    action_prob = float(pi.reshape(-1)[action].item())
    return np.asarray([action], dtype=np.int64), action, action_prob


def _plot_gantt(rows: list[dict], path: Path, title: str) -> None:
    from gantt import plot_global_gantt

    plot_rows = []
    for row in rows:
        plot_rows.append(
            {
                "job": int(row["job_id"]),
                "op": int(row["op_id_in_job"]),
                "machine": int(row["machine_id"]),
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
                "duration": float(row["proc_time"]),
                "due_date": float(row["due_date"]),
            }
        )
    plot_global_gantt(plot_rows, str(path), title=title)


def main() -> None:
    import numpy as np
    import torch

    args = parse_args()

    # Importing this module is safe: its argument parser is only called by its
    # own main(). Reusing its helpers keeps this test aligned with training.
    from train_ll_stability_finetune import (
        build_env,
        clone_policy_config,
        generate_batch,
        effective_due_mode,
        load_policy_checkpoint,
        load_project_config,
        set_global_config,
        set_seed,
    )

    configured, config_path = load_project_config(args.config)
    if args.seed is not None:
        configured.seed_train = int(args.seed)
    set_seed(int(getattr(configured, "seed_train", 3)))

    # The old checkpoint has the legacy 20/8 state dimensions. Keep the
    # analysis environment on that interface; stability features are disabled.
    configured.fea_j_input_dim = 20
    configured.fea_pair_input_dim = 8
    configured.ll_stability_enable = False
    configured.ll_stability_critic_summary = False
    set_global_config(configured, configured)

    from model.ll_ppo import LLPPO

    policy_config = clone_policy_config(configured, stability=False)
    ppo = LLPPO(policy_config)
    checkpoint = _resolve_checkpoint(
        args.model
        or getattr(configured, "ll_stability_reference_model_path", None)
        or getattr(configured, "ll_ppo_model_path", None)
    )
    loaded_checkpoint = load_policy_checkpoint(ppo, checkpoint, torch)
    ppo.policy.eval()

    pattern = None
    if args.kind == "fresh":
        pattern = [True]
    elif args.kind == "reschedule":
        pattern = [False]

    due_mode = None if args.due_mode == "auto" else args.due_mode
    base_seed = int(getattr(configured, "seed_train", 3))
    max_retries = max(1, int(args.max_sample_retries))
    samples = None
    selected_seed = base_seed
    for attempt in range(max_retries if args.require_overdue else 1):
        selected_seed = base_seed + attempt
        configured.seed_train = selected_seed
        candidate_samples = generate_batch(
            configured,
            ppo,
            sample_index=int(args.sample_index),
            target_jobs=args.target_jobs,
            pattern=pattern,
            due_mode=due_mode,
            num_envs=1,
        )
        candidate = candidate_samples[0]
        has_overdue = bool(np.any(np.asarray(candidate["due"], dtype=np.float64) < 0.0))
        if not args.require_overdue or has_overdue:
            samples = candidate_samples
            break
    if samples is None:
        raise RuntimeError(
            "Could not generate a sample with due<0 within "
            f"{max_retries} seed attempts."
        )
    sample = samples[0]
    env, state = build_env(configured, samples, stability_enabled=False)

    op_to_job, op_to_local = _build_op_maps(sample["job_lengths"])
    reference = sample["reference"]
    op_count = len(op_to_job)
    rows: list[dict] = []
    done = False
    step = 0
    while not done:
        action_array, action, action_prob = _run_policy(
            env, state, ppo, args.action_mode
        )
        state, reward, done_flags, info = env.step(action_array)
        detail = info.get("scheduled_op_details_all", [None])[0]
        if detail is None:
            detail = info["scheduled_op_details"]

        job_id = int(detail["job_id"])
        op_global = int(detail["op_global_id"])
        op_local = int(detail["op_id_in_job"])
        if not (0 <= op_global < op_count):
            raise RuntimeError(
                f"Invalid operation id {op_global}; expected [0, {op_count})."
            )
        source = "new" if float(reference["is_new_job_op"][op_global]) > 0.5 else "retained_old"
        due = float(sample["due"][job_id])
        is_last = op_local == int(sample["job_lengths"][job_id]) - 1
        end_time = float(detail["end_time"])
        row = {
            "schedule_step": int(step),
            "action_index": int(action),
            "action_probability": float(action_prob),
            "job_id": job_id,
            "op_id_in_job": op_local,
            "op_global_id": op_global,
            "machine_id": int(detail["machine_id"]),
            "start_time": float(detail["start_time"]),
            "end_time": end_time,
            "proc_time": float(detail["proc_time"]),
            "due_date": due,
            "release_time": float(sample["release"][job_id]),
            "source": source,
            "has_old_reference": int(bool(reference["ref_mask"][op_global])),
            "old_machine": int(reference["old_machine"][op_global]),
            "old_rank": int(reference["old_rank"][op_global]),
            "old_rank_norm": float(reference["old_rank_norm"][op_global]),
            "is_last_op": int(is_last),
            "op_tardiness": float(max(0.0, end_time - due) if is_last else 0.0),
            "reward": _scalar(np.asarray(reward)[0]),
            "reward_mk_step": _scalar(np.asarray(info["reward_mk_step"])[0]),
            "reward_td_step": _scalar(np.asarray(info["reward_td_step"])[0]),
            "reward_od_step": _scalar(np.asarray(info["reward_od_step"])[0]),
            "reward_stability_step": _scalar(np.asarray(info["reward_stability_step"])[0]),
        }
        rows.append(row)
        step += 1
        done = bool(np.asarray(done_flags, dtype=bool)[0])

    # Recompute final KPIs from exported operations so the CSV and summary use
    # exactly the same source data.
    job_finish: dict[int, float] = {}
    for row in rows:
        job_finish[int(row["job_id"])] = max(
            job_finish.get(int(row["job_id"]), 0.0), float(row["end_time"])
        )
    makespan = max((float(row["end_time"]) for row in rows), default=0.0)
    tardiness = sum(
        max(0.0, finish - float(sample["due"][job_id]))
        for job_id, finish in job_finish.items()
    )
    objective = 0.5 * makespan + 0.5 * tardiness

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    else:
        output_dir = ROOT / "analysis_results" / "lower_level_stability_test" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_fields = list(rows[0].keys()) if rows else [
        "schedule_step", "action_index", "job_id", "op_id_in_job"
    ]
    _write_csv(output_dir / "schedule_details.csv", rows, detail_fields)

    job_rows = []
    offset = 0
    for job_id, length in enumerate(sample["job_lengths"]):
        length = int(length)
        block = slice(offset, offset + length)
        sources = reference["is_new_job_op"][block]
        job_rows.append(
            {
                "job_id": int(job_id),
                "op_count": length,
                "due_date": float(sample["due"][job_id]),
                "release_time": float(sample["release"][job_id]),
                "source": "new" if np.all(sources > 0.5) else "retained_old",
                "has_old_reference_ops": int(np.sum(reference["ref_mask"][block])),
                "job_tardiness": float(max(0.0, job_finish.get(job_id, 0.0) - sample["due"][job_id])),
            }
        )
        offset += length
    _write_csv(
        output_dir / "job_details.csv",
        job_rows,
        list(job_rows[0].keys()) if job_rows else ["job_id", "op_count"],
    )

    inferred_due_mode = (
        effective_due_mode(configured, int(args.sample_index))
        if args.due_mode == "auto"
        else args.due_mode
    )
    summary = {
        "config": str(config_path),
        "legacy_checkpoint": str(loaded_checkpoint),
        "kind_requested": args.kind,
        "kind_generated": str(sample.get("kind", "unknown")),
        "due_mode_requested": args.due_mode,
        "due_mode_effective": inferred_due_mode,
        "seed": int(selected_seed),
        "sample_index": int(args.sample_index),
        "action_mode": args.action_mode,
        "target_jobs": int(len(sample["job_lengths"])),
        "scheduled_operations": int(len(rows)),
        "retained_jobs": int(sample.get("retained_jobs", 0)),
        "new_jobs": int(sum(1 for row in job_rows if row["source"] == "new")),
        "overdue_jobs": int(sum(1 for row in job_rows if float(row["due_date"]) < 0.0)),
        "overdue_job_ratio": float(
            sum(1 for row in job_rows if float(row["due_date"]) < 0.0)
            / max(len(job_rows), 1)
        ),
        "cut_time": float(sample.get("cut_time", 0.0)),
        "makespan": float(makespan),
        "tardiness": float(tardiness),
        "objective": float(objective),
    }
    _write_csv(output_dir / "summary.csv", [summary], list(summary.keys()))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_gantt(
        rows,
        output_dir / "gantt.png",
        title=(
            f"Legacy lower PPO | {summary['kind_generated']} | "
            f"MK={makespan:.1f}, TD={tardiness:.1f}"
        ),
    )

    print(f"[CONFIG] {config_path}")
    print(f"[LEGACY PPO] {loaded_checkpoint}")
    print(
        f"[RESULT] kind={summary['kind_generated']} seed={summary['seed']} "
        f"jobs={summary['target_jobs']} ops={summary['scheduled_operations']} "
        f"MK={makespan:.3f} TD={tardiness:.3f} Obj={objective:.3f}"
    )
    print(f"[OUTPUT] {output_dir}")


if __name__ == "__main__":
    main()
