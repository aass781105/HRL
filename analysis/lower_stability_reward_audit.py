"""Audit lower-level PPO stability metrics over virtual-cut/target grids.

The script intentionally uses the existing dynamic rescheduling path without
changing the training environment or PPO implementation:

1. Generate a 30-job uniform base instance with the same generator used by
   ``train_ll_curriculum.py``.
2. Solve the complete base instance with the configured old lower-level PPO.
3. Reuse that reference schedule and apply virtual cuts at several fractions
   of its makespan.
4. Add jobs until each requested target-job count is reached.
5. Reschedule through ``GlobalTimelineOrchestrator`` and measure global
   makespan/tardiness plus old-operation machine changes and pair flips.

The reference policy is loaded from the supplied lower-level YAML. The script
is an analysis tool only; it does not update weights or modify core code.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_local_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit lower-level PPO MK/TD/flip/machine-change distributions."
    )
    parser.add_argument(
        "--config",
        default=r"yaml跑不同種子碼\train_ll_u1030_esttd_odprog.yml",
        help="Lower-level YAML used for data generation and old PPO loading.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional old lower-level checkpoint override.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Base-instance seeds. Defaults to 1..10.",
    )
    parser.add_argument(
        "--cut_fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
        help="Virtual cut as a fraction of reference makespan.",
    )
    parser.add_argument(
        "--target_jobs",
        nargs="+",
        type=int,
        default=[30, 32, 35, 38, 40],
        help="Final target job counts for the reschedule sample.",
    )
    parser.add_argument(
        "--initial_jobs",
        type=int,
        default=30,
        help="Base job count. The first protocol version uses 30.",
    )
    parser.add_argument(
        "--due_mode",
        default="auto",
        help="Due mode. auto follows the current training mode at update 0.",
    )
    parser.add_argument(
        "--output_root",
        default=r"analysis_results\lower_stability_reward_audit",
        help="Output root for CSV, JSON, and plots.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip matplotlib distribution plots.",
    )
    return parser.parse_args()


LOCAL_ARGS = parse_local_args()
CONFIG_PATH = Path(LOCAL_ARGS.config)
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = REPO_ROOT / CONFIG_PATH
CONFIG_PATH = CONFIG_PATH.resolve()
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Lower-level config not found: {CONFIG_PATH}")

# params.py parses command-line arguments at import time. Do not expose the
# analysis script's grid arguments to the project parser.
sys.argv = [sys.argv[0], "--config", str(CONFIG_PATH)]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from common_utils import resolve_lower_level_weight_path  # noqa: E402
from data_utils import SD2_instance_generator, generate_due_dates  # noqa: E402
from hrl_orchestrator import (  # noqa: E402
    EventBurstGenerator,
    GlobalTimelineOrchestrator,
    split_matrix_to_jobs,
)
from params import configs  # noqa: E402


def resolve_due_mode(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested != "auto":
        return requested
    configured = str(getattr(configs, "ll_due_date_mode", "k")).strip().lower()
    if configured == "range3_hold":
        # train_ll_curriculum starts the hold cycle with loose.
        return "range3_loose"
    return configured


def configure_runtime() -> str:
    if not torch.cuda.is_available():
        configs.device = "cpu"
    configs.scheduler_type = "PPO"
    configs.ll_eval_action_selection = "greedy"
    configs.ll_rollout_k = 1
    configs.data_source = "SD2"
    configs.op_per_job = 5
    configs.enable_op_mixture = False
    configs.n_j = int(LOCAL_ARGS.initial_jobs)
    if LOCAL_ARGS.model:
        configs.ll_ppo_model_path = LOCAL_ARGS.model

    configured_path = str(
        getattr(configs, "ll_ppo_model_path", "")
        or getattr(configs, "ppo_model_path", "")
    )
    resolved_path = resolve_lower_level_weight_path(
        configured_path,
        getattr(configs, "data_source", "SD2"),
    )
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            "Old lower-level PPO checkpoint not found: "
            f"{resolved_path}. Pass --model or set ll_ppo_model_path in the YAML."
        )
    return str(Path(resolved_path).resolve())


def generate_uniform_instance(seed: int, job_count: int):
    old_n_j = getattr(configs, "n_j", None)
    try:
        configs.n_j = int(job_count)
        job_lengths, op_pt, _ = SD2_instance_generator(
            config=configs,
            seed=int(seed),
            mode="uniform",
        )
    finally:
        if old_n_j is not None:
            configs.n_j = old_n_j
    return np.asarray(job_lengths, dtype=np.int64), np.asarray(op_pt, dtype=float)


def generate_base_jobs(seed: int, due_mode: str):
    job_lengths, op_pt = generate_uniform_instance(seed, LOCAL_ARGS.initial_jobs)
    due_dates = np.asarray(
        generate_due_dates(
            job_lengths,
            op_pt,
            due_date_mode=due_mode,
            seed=int(seed),
        ),
        dtype=float,
    )
    jobs = split_matrix_to_jobs(
        job_lengths,
        op_pt,
        base_job_id=0,
        t_arrive=0.0,
        due_dates=due_dates,
    )
    return jobs, due_dates


def generate_new_job_pool(seed: int, target_jobs: int, t_cut: float):
    """Generate the protocol's tight new-job pool for one target count."""
    job_seed = int(seed) * 100_000 + int(target_jobs) * 101 + 17
    job_lengths, op_pt = generate_uniform_instance(job_seed, target_jobs)
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    due_scale = float(getattr(configs, "ll_due_range_scale", 0.7))
    a_new = due_scale * float(target_jobs) * mean_pt
    due_rng = np.random.default_rng(job_seed + 1)
    due_rel = due_rng.uniform(-0.1 * a_new, 1.2 * a_new, size=target_jobs)
    due_abs = float(t_cut) + due_rel
    return split_matrix_to_jobs(
        job_lengths,
        op_pt,
        base_job_id=int(LOCAL_ARGS.initial_jobs),
        t_arrive=float(t_cut),
        due_dates=due_abs,
    )


def make_orchestrator(seed: int) -> GlobalTimelineOrchestrator:
    rng = np.random.default_rng(int(seed) + 700_001)
    generator = EventBurstGenerator(
        SD2_instance_generator,
        configs,
        int(configs.n_m),
        interarrival_mean=1.0,
        rng=rng,
    )
    return GlobalTimelineOrchestrator(int(configs.n_m), generator, t0=0.0)


def reset_to_reference(
    orch: GlobalTimelineOrchestrator,
    reference_rows: Sequence[Mapping[str, object]],
    base_jobs: Sequence[object],
) -> None:
    """Restore only the orchestrator state needed by event rescheduling."""
    orch.reset(clear_buffer=True, t0=0.0)
    orch._last_full_rows = [dict(row) for row in copy.deepcopy(reference_rows)]
    orch._last_jobs_snapshot = copy.deepcopy(list(base_jobs))
    orch._release_count = 1


def reference_schedule(
    orch: GlobalTimelineOrchestrator,
    base_jobs: Sequence[object],
) -> List[dict]:
    reset_to_reference(orch, [], [])
    orch.buffer.extend(copy.deepcopy(list(base_jobs)))
    orch.event_release_and_reschedule(0.0, event_id=0)
    return [dict(row) for row in copy.deepcopy(orch._last_full_rows)]


def kpis(rows: Iterable[Mapping[str, object]], due_by_job: Mapping[int, float]) -> Tuple[float, float]:
    end_by_job: Dict[int, float] = {}
    all_ends: List[float] = []
    for row in rows:
        job_id = int(row["job"])
        end = float(row["end"])
        end_by_job[job_id] = max(end_by_job.get(job_id, -math.inf), end)
        all_ends.append(end)
    makespan = max(all_ends) if all_ends else 0.0
    tardiness = sum(
        max(0.0, end - float(due_by_job[job_id]))
        for job_id, end in end_by_job.items()
        if job_id in due_by_job
    )
    return float(makespan), float(tardiness)


def future_rows(rows: Iterable[Mapping[str, object]], t_cut: float) -> List[dict]:
    return [dict(row) for row in rows if float(row["start"]) > float(t_cut)]


def op_key(row: Mapping[str, object]) -> Tuple[int, int]:
    return int(row["job"]), int(row["op"])


def sequence_by_machine(rows: Iterable[Mapping[str, object]]) -> Dict[int, List[Tuple[int, int]]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["machine"]), []).append(row)
    return {
        machine: [
            op_key(row)
            for row in sorted(
                machine_rows,
                key=lambda item: (
                    float(item["start"]),
                    float(item["end"]),
                    int(item["job"]),
                    int(item["op"]),
                ),
            )
        ]
        for machine, machine_rows in grouped.items()
    }


def stability_counts(
    before_rows: Iterable[Mapping[str, object]],
    after_rows: Iterable[Mapping[str, object]],
    t_cut: float,
) -> Dict[str, float]:
    """Match the project's dynamic stability definition for future old ops."""
    before = future_rows(before_rows, t_cut)
    after = future_rows(after_rows, t_cut)
    before_map = {op_key(row): row for row in before}
    after_map = {op_key(row): row for row in after}
    common = sorted(set(before_map) & set(after_map))

    machine_changes = sum(
        int(int(before_map[key]["machine"]) != int(after_map[key]["machine"]))
        for key in common
    )

    before_seq = sequence_by_machine(before)
    after_seq = sequence_by_machine(after)
    pair_count = 0
    flip_count = 0
    for machine in sorted(set(before_seq) | set(after_seq)):
        old_pos = {key: idx for idx, key in enumerate(before_seq.get(machine, []))}
        new_pos = {key: idx for idx, key in enumerate(after_seq.get(machine, []))}
        comparable = [
            key
            for key in before_seq.get(machine, [])
            if key in common
            and key in new_pos
            and int(after_map[key]["machine"]) == machine
        ]
        for first, second in itertools.combinations(comparable, 2):
            pair_count += 1
            before_delta = old_pos[first] - old_pos[second]
            after_delta = new_pos[first] - new_pos[second]
            if before_delta * after_delta < 0:
                flip_count += 1

    common_count = len(common)
    return {
        "common_future_ops": int(common_count),
        "comparable_pairs": int(pair_count),
        "flip_count": int(flip_count),
        "machine_change_count": int(machine_changes),
        "flip_rate": float(flip_count / pair_count) if pair_count else 0.0,
        "machine_change_rate": (
            float(machine_changes / common_count) if common_count else 0.0
        ),
    }


def retained_stats(reference_rows: Sequence[Mapping[str, object]], t_cut: float, base_jobs) -> Dict[str, int]:
    rows_by_job: Dict[int, List[Mapping[str, object]]] = {}
    for row in reference_rows:
        rows_by_job.setdefault(int(row["job"]), []).append(row)

    retained_jobs = 0
    retained_ops = 0
    in_progress_ops = 0
    for job in base_jobs:
        job_id = int(job.job_id)
        rows = rows_by_job.get(job_id, [])
        started = [row for row in rows if float(row["start"]) < float(t_cut)]
        future = [row for row in rows if float(row["start"]) > float(t_cut)]
        if len(started) < int(job.meta.get("total_ops", len(job.operations))):
            retained_jobs += 1
            retained_ops += len(future)
        in_progress_ops += sum(
            int(float(row["start"]) <= float(t_cut) < float(row["end"]))
            for row in rows
        )
    return {
        "retained_old_jobs": int(retained_jobs),
        "retained_old_ops": int(retained_ops),
        "in_progress_ops": int(in_progress_ops),
    }


def base_row(
    seed: int,
    due_mode: str,
    target_jobs: int,
    cut_fraction: float,
    reference_mk: float,
    reference_td: float,
    t_cut: float,
    retained: Mapping[str, int],
) -> Dict[str, object]:
    return {
        "seed": int(seed),
        "due_mode": str(due_mode),
        "initial_jobs": int(LOCAL_ARGS.initial_jobs),
        "target_jobs": int(target_jobs),
        "cut_fraction": float(cut_fraction),
        "t_cut": float(t_cut),
        "reference_makespan": float(reference_mk),
        "reference_tardiness": float(reference_td),
        **{key: int(value) for key, value in retained.items()},
        "new_jobs": int(target_jobs - retained["retained_old_jobs"]),
        "reference_objective": 0.5 * (reference_mk + reference_td),
        "status": "invalid_target",
    }


def run_case(
    orch: GlobalTimelineOrchestrator,
    reference_rows: Sequence[Mapping[str, object]],
    base_jobs: Sequence[object],
    base_due: Mapping[int, float],
    seed: int,
    due_mode: str,
    cut_fraction: float,
    target_jobs: int,
    reference_mk: float,
    reference_td: float,
) -> Dict[str, object]:
    t_cut = float(cut_fraction) * float(reference_mk)
    retained = retained_stats(reference_rows, t_cut, base_jobs)
    row = base_row(
        seed,
        due_mode,
        target_jobs,
        cut_fraction,
        reference_mk,
        reference_td,
        t_cut,
        retained,
    )
    if retained["retained_old_jobs"] > int(target_jobs):
        return row

    new_pool = generate_new_job_pool(seed, int(target_jobs), t_cut)
    new_count = int(target_jobs) - int(retained["retained_old_jobs"])
    new_jobs = new_pool[:new_count]
    if len(new_jobs) != new_count:
        row["status"] = "error_new_job_pool"
        row["error"] = f"requested={new_count}, generated={len(new_jobs)}"
        return row

    reset_to_reference(orch, reference_rows, base_jobs)
    orch.buffer.extend(copy.deepcopy(new_jobs))
    orch.event_release_and_reschedule(float(t_cut), event_id=int(round(cut_fraction * 100)))
    after_rows = [dict(item) for item in copy.deepcopy(orch._last_full_rows)]
    after_due = dict(base_due)
    after_due.update(
        {
            int(job.job_id): float(job.meta.get("due_date", 0.0))
            for job in new_jobs
        }
    )
    after_mk, after_td = kpis(after_rows, after_due)
    stability = stability_counts(reference_rows, after_rows, t_cut)

    row.update(
        {
            "status": "ok",
            "new_ops": int(sum(len(job.operations) for job in new_jobs)),
            "after_makespan": float(after_mk),
            "after_tardiness": float(after_td),
            "after_objective": 0.5 * (after_mk + after_td),
            "mk_delta": float(after_mk - reference_mk),
            "td_delta": float(after_td - reference_td),
            **stability,
        }
    )
    return row


SUMMARY_METRICS = [
    "reference_makespan",
    "reference_tardiness",
    "after_makespan",
    "after_tardiness",
    "mk_delta",
    "td_delta",
    "flip_count",
    "machine_change_count",
    "flip_rate",
    "machine_change_rate",
]


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    valid = samples[samples["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame()
    group_cols = ["due_mode", "cut_fraction", "target_jobs"]
    rows = []
    for keys, group in valid.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["samples"] = int(len(group))
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_q25"] = float(np.percentile(values, 25))
            row[f"{metric}_median"] = float(np.percentile(values, 50))
            row[f"{metric}_q75"] = float(np.percentile(values, 75))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(samples: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = samples[samples["status"] == "ok"].copy()
    if valid.empty:
        return
    plot_specs = [
        ("after_makespan", "After makespan"),
        ("after_tardiness", "After tardiness"),
        ("flip_count", "Pair flip count"),
        ("machine_change_count", "Machine change count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, plot_specs):
        values = pd.to_numeric(valid[metric], errors="coerce").dropna().to_numpy()
        axis.hist(values, bins="auto", color="#2d6cdf", alpha=0.82, edgecolor="white")
        if values.size:
            axis.axvline(float(np.median(values)), color="#d64545", linestyle="--", label="median")
        axis.set_title(title)
        axis.set_xlabel(metric)
        axis.set_ylabel("count")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Lower-level stability reward audit distributions")
    fig.savefig(output_dir / "metric_distributions.png", dpi=160)
    plt.close(fig)

    # A second plot keeps cut/target effects visible without requiring a
    # separate figure for every metric.
    labels = [
        f"c{cut:.2f}-j{target}"
        for cut, target in sorted(
            valid[["cut_fraction", "target_jobs"]].drop_duplicates().itertuples(index=False, name=None)
        )
    ]
    if labels:
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
        groups = [
            valid[(valid["cut_fraction"] == cut) & (valid["target_jobs"] == target)]
            for cut, target in sorted(
                valid[["cut_fraction", "target_jobs"]].drop_duplicates().itertuples(index=False, name=None)
            )
        ]
        for axis, (metric, title) in zip(axes.flat, plot_specs):
            data = [pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy() for group in groups]
            axis.boxplot(data, tick_labels=labels, showfliers=False)
            axis.set_title(f"{title} by cut/target")
            axis.tick_params(axis="x", labelrotation=75)
            axis.grid(axis="y", alpha=0.25)
        fig.savefig(output_dir / "metric_distributions_by_cut_target.png", dpi=160)
        plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    resolved_model = configure_runtime()
    due_mode = resolve_due_mode(LOCAL_ARGS.due_mode)
    cut_fractions = sorted({float(value) for value in LOCAL_ARGS.cut_fractions})
    target_jobs = sorted({int(value) for value in LOCAL_ARGS.target_jobs})
    if any(value <= 0 or value > 1 for value in cut_fractions):
        raise ValueError("cut_fractions must be in (0, 1].")
    if any(value < int(LOCAL_ARGS.initial_jobs) for value in target_jobs):
        raise ValueError("target_jobs must be >= initial_jobs for this protocol.")

    output_dir = REPO_ROOT / LOCAL_ARGS.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": str(CONFIG_PATH),
        "reference_model": resolved_model,
        "reference_action_selection": "greedy",
        "initial_jobs": int(LOCAL_ARGS.initial_jobs),
        "op_per_job": 5,
        "seeds": [int(value) for value in LOCAL_ARGS.seeds],
        "cut_fractions": cut_fractions,
        "target_jobs": target_jobs,
        "due_mode_requested": str(LOCAL_ARGS.due_mode),
        "due_mode_resolved": due_mode,
        "new_job_due_rule": "Uniform(-0.1*a_new, 1.2*a_new), a_new=ll_due_range_scale*target_jobs*mean_pt",
        "metric_definition": "future old operations only; new jobs excluded from flip/machine-change comparisons",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    all_rows: List[Dict[str, object]] = []
    total_seeds = len(LOCAL_ARGS.seeds)
    total_cases = len(cut_fractions) * len(target_jobs)
    for seed_index, seed in enumerate(LOCAL_ARGS.seeds, start=1):
        print(f"[SEED {seed_index}/{total_seeds}] seed={seed}: generating base and reference schedule")
        base_jobs, base_due_array = generate_base_jobs(int(seed), due_mode)
        base_due = {int(job_id): float(value) for job_id, value in enumerate(base_due_array)}
        orch = make_orchestrator(int(seed))
        reference_rows = reference_schedule(orch, base_jobs)
        reference_mk, reference_td = kpis(reference_rows, base_due)
        print(
            f"  reference MK={reference_mk:.3f} TD={reference_td:.3f} "
            f"cases={total_cases}"
        )

        case_index = 0
        for cut_fraction in cut_fractions:
            for target in target_jobs:
                case_index += 1
                row = run_case(
                    orch,
                    reference_rows,
                    base_jobs,
                    base_due,
                    int(seed),
                    due_mode,
                    float(cut_fraction),
                    int(target),
                    reference_mk,
                    reference_td,
                )
                all_rows.append(row)
                print(
                    f"  [CASE {case_index:02d}/{total_cases}] "
                    f"cut={cut_fraction:.2f} target={target} "
                    f"retained={row['retained_old_jobs']} status={row['status']}"
                )

    samples = pd.DataFrame(all_rows)
    samples.to_csv(output_dir / "samples.csv", index=False, encoding="utf-8-sig")
    summary = summarize(samples)
    summary.to_csv(output_dir / "summary_by_cut_target.csv", index=False, encoding="utf-8-sig")
    if not LOCAL_ARGS.no_plots:
        write_plots(samples, output_dir)

    valid_count = int((samples["status"] == "ok").sum()) if not samples.empty else 0
    invalid_count = int((samples["status"] != "ok").sum()) if not samples.empty else 0
    elapsed = time.perf_counter() - started
    print(f"[DONE] output={output_dir}")
    print(f"[DONE] valid_cases={valid_count} invalid_or_error_cases={invalid_count} elapsed={elapsed:.2f}s")
    print(f"[DONE] samples={output_dir / 'samples.csv'}")
    print(f"[DONE] summary={output_dir / 'summary_by_cut_target.csv'}")


if __name__ == "__main__":
    main()
