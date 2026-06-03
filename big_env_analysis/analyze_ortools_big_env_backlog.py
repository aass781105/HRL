import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ORTOOLS_ROOT = "plots/global/ortool_big_env"
DEFAULT_INSTANCE_PATTERN = "dynamic_instances/dynamic_seed42_init50_h80_U20_50_due20_80_seed*.json"
DEFAULT_OUTPUT_DIR = "big_env_analysis/results/ortools_big_env_backlog"


def infer_seed_from_text(text: str) -> str:
    matches = re.findall(r"seed(\d+)", str(text))
    return matches[-1] if matches else str(text)


def load_instances(pattern: str):
    instances = {}
    for path in sorted(Path().glob(pattern) if not any(ch in pattern for ch in "*?[") else map(Path, __import__("glob").glob(pattern))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        seed = str(payload.get("meta", {}).get("seed", "") or infer_seed_from_text(path.stem))
        job_map = {}
        for job in payload.get("jobs", []) or []:
            jid = int(job["job_id"])
            ops = sorted(job.get("operations", []), key=lambda x: int(x.get("op_id", 0)))
            op_avg = []
            op_min = []
            op_machine_times = {}
            for op in ops:
                mt = {int(k): float(v) for k, v in (op.get("machine_times", {}) or {}).items()}
                vals = [v for v in mt.values() if v > 0]
                op_avg.append(float(np.mean(vals)) if vals else 0.0)
                op_min.append(float(np.min(vals)) if vals else 0.0)
                op_machine_times[int(op.get("op_id", len(op_machine_times)))] = mt
            job_map[jid] = {
                "job_id": jid,
                "arrive_time": float(job.get("arrive_time", job.get("t_arrive_abs", 0.0))),
                "due_date": float(job.get("due_date", 0.0)),
                "total_proc_time": float(sum(op_avg)),
                "min_total_proc_time": float(sum(op_min)),
                "total_ops": int(job.get("total_ops", len(ops))),
                "op_avg": op_avg,
                "op_min": op_min,
                "op_machine_times": op_machine_times,
            }
        instances[seed] = {"path": str(path), "payload": payload, "jobs": job_map}
    return instances


def read_release_log(run_dir: Path) -> pd.DataFrame:
    logs = sorted(run_dir.glob("*release_log*.csv"))
    if not logs:
        raise FileNotFoundError(f"No release log found in {run_dir}")
    df = pd.read_csv(logs[0])
    df["release_index"] = np.arange(1, len(df) + 1)
    df["release_label"] = df["Event_ID"].apply(lambda x: "INIT" if int(x) == 0 else f"EVENT_{int(x)}")
    return df


def read_details(run_dir: Path):
    details = {}
    for path in sorted(run_dir.glob("details_*.csv")):
        m = re.search(r"details_r(\d+)_t(\d+)", path.name)
        if not m:
            continue
        release_index = int(m.group(1))
        df = pd.read_csv(path)
        for col in ("job", "op", "machine"):
            if col in df.columns:
                df[col] = df[col].astype(int)
        for col in ("start", "end", "duration", "due_date", "tardiness"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        details[release_index] = {"path": path, "df": df}
    return details


def safe_num(row, col, default=0.0):
    return float(row[col]) if col in row and pd.notna(row[col]) else float(default)


def summarize_details(seed: str, release_row, detail_df: pd.DataFrame, job_map: dict):
    t = safe_num(release_row, "Release_Time")
    status = detail_df["status"].astype(str) if "status" in detail_df.columns else pd.Series([""] * len(detail_df))
    new_df = detail_df[status.eq("NewPlan")]
    hist_df = detail_df[status.eq("History")]
    fixed_df = detail_df[detail_df["start"].astype(float) < t] if "start" in detail_df.columns else hist_df
    completed_df = detail_df[detail_df["end"].astype(float) <= t] if "end" in detail_df.columns else hist_df

    arrived_jobs = [jid for jid, job in job_map.items() if float(job["arrive_time"]) <= t + 1e-9]
    fixed_counts = fixed_df.groupby("job")["op"].nunique().to_dict() if len(fixed_df) else {}
    completed_counts = completed_df.groupby("job")["op"].nunique().to_dict() if len(completed_df) else {}
    completed_jobs = 0
    fixed_complete_jobs = 0
    for jid in arrived_jobs:
        total_ops = int(job_map[jid]["total_ops"])
        if int(completed_counts.get(jid, 0)) >= total_ops:
            completed_jobs += 1
        if int(fixed_counts.get(jid, 0)) >= total_ops:
            fixed_complete_jobs += 1

    return {
        "seed": seed,
        "release_index": int(release_row["release_index"]),
        "release_label": release_row["release_label"],
        "event_id": int(release_row["Event_ID"]),
        "release_time": t,
        "objective": safe_num(release_row, "Global_Objective_0p5MK_0p5TD", safe_num(release_row, "Objective_0p5MK_0p5TD")),
        "makespan": safe_num(release_row, "Global_Makespan", safe_num(release_row, "Makespan")),
        "tardiness": safe_num(release_row, "Global_Total_Tardiness", safe_num(release_row, "Total_Tardiness")),
        "subproblem_job_count": int(safe_num(release_row, "Subproblem_Job_Count")),
        "repeated_job_count": int(safe_num(release_row, "Repeated_Job_Count")),
        "num_rows": int(safe_num(release_row, "Num_Rows")),
        "detail_unique_jobs": int(detail_df["job"].nunique()) if "job" in detail_df.columns else 0,
        "newplan_jobs": int(new_df["job"].nunique()) if "job" in new_df.columns else 0,
        "newplan_ops": int(len(new_df)),
        "history_jobs": int(hist_df["job"].nunique()) if "job" in hist_df.columns else 0,
        "history_ops": int(len(hist_df)),
        "fixed_started_ops_before_release": int(len(fixed_df)),
        "completed_ops_before_release": int(len(completed_df)),
        "arrived_jobs_by_release": int(len(arrived_jobs)),
        "completed_jobs_before_release": int(completed_jobs),
        "fixed_complete_jobs_before_release": int(fixed_complete_jobs),
    }


def compute_candidate_slack(seed: str, release_row, detail_df: pd.DataFrame, job_map: dict):
    t = safe_num(release_row, "Release_Time")
    status = detail_df["status"].astype(str) if "status" in detail_df.columns else pd.Series([""] * len(detail_df))
    new_df = detail_df[status.eq("NewPlan")].copy()
    hist_df = detail_df[detail_df["start"].astype(float) < t].copy() if "start" in detail_df.columns else detail_df[status.eq("History")].copy()
    if new_df.empty:
        return []

    machine_ready = {}
    if not hist_df.empty:
        for m, g in hist_df.groupby("machine"):
            machine_ready[int(m)] = max(float(t), float(g["end"].max()))

    job_ready = {}
    if not hist_df.empty:
        for jid, g in hist_df.groupby("job"):
            job_ready[int(jid)] = max(float(t), float(g["end"].max()))

    rows = []
    for jid, g in new_df.groupby("job"):
        jid = int(jid)
        if jid not in job_map:
            continue
        current_op = int(g["op"].min())
        job = job_map[jid]
        op_machine_times = job["op_machine_times"].get(current_op, {})
        if not op_machine_times:
            continue
        due = float(job["due_date"])
        arrive = float(job["arrive_time"])
        ready_j = float(job_ready.get(jid, t))
        remain_after_mean = float(sum(job["op_avg"][current_op + 1 :]))
        remain_after_min = float(sum(job["op_min"][current_op + 1 :]))
        for machine, pt in sorted(op_machine_times.items()):
            ready_m = float(machine_ready.get(int(machine), t))
            start_est = max(ready_j, ready_m)
            finish_est = start_est + float(pt)
            slack_mean = due - finish_est - remain_after_mean
            slack_min = due - finish_est - remain_after_min
            rows.append(
                {
                    "seed": seed,
                    "release_index": int(release_row["release_index"]),
                    "release_label": release_row["release_label"],
                    "event_id": int(release_row["Event_ID"]),
                    "release_time": t,
                    "job": jid,
                    "op": current_op,
                    "machine": int(machine),
                    "arrive_time": arrive,
                    "due_date": due,
                    "relative_due": due - t,
                    "job_ready_time": ready_j,
                    "machine_ready_time": ready_m,
                    "start_est": start_est,
                    "finish_est": finish_est,
                    "pt": float(pt),
                    "remain_after_mean": remain_after_mean,
                    "remain_after_min": remain_after_min,
                    "candidate_slack_mean": slack_mean,
                    "candidate_slack_min": slack_min,
                    "candidate_neg_slack_mean": max(0.0, -slack_mean),
                    "candidate_neg_slack_min": max(0.0, -slack_min),
                }
            )
    return rows


def summarize_candidate_slack(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    def q25(s):
        return float(s.quantile(0.25))

    pair_summary = (
        pair_df.groupby(["seed", "release_index", "release_label", "event_id", "release_time"])
        .agg(
            candidate_pairs=("candidate_slack_mean", "count"),
            pair_neg_slack_ratio=("candidate_slack_mean", lambda s: float((s < 0).mean())),
            pair_mean_neg_slack=("candidate_neg_slack_mean", "mean"),
            pair_min_slack=("candidate_slack_mean", "min"),
            pair_q25_slack=("candidate_slack_mean", q25),
            pair_mean_slack=("candidate_slack_mean", "mean"),
        )
        .reset_index()
    )

    best_rows = []
    for keys, sub in pair_df.groupby(["seed", "release_index", "release_label", "event_id", "release_time", "job", "op"]):
        best = sub.loc[sub["candidate_slack_mean"].idxmax()]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)
    best_summary = (
        best_df.groupby(["seed", "release_index", "release_label", "event_id", "release_time"])
        .agg(
            candidate_jobs=("candidate_slack_mean", "count"),
            best_neg_slack_ratio=("candidate_slack_mean", lambda s: float((s < 0).mean())),
            best_mean_neg_slack=("candidate_neg_slack_mean", "mean"),
            best_min_slack=("candidate_slack_mean", "min"),
            best_q25_slack=("candidate_slack_mean", q25),
            best_mean_slack=("candidate_slack_mean", "mean"),
        )
        .reset_index()
    )
    return pair_summary.merge(best_summary, on=["seed", "release_index", "release_label", "event_id", "release_time"], how="outer")


def compute_machine_conflict(seed: str, release_row, detail_df: pd.DataFrame, job_map: dict):
    status = detail_df["status"].astype(str) if "status" in detail_df.columns else pd.Series([""] * len(detail_df))
    new_df = detail_df[status.eq("NewPlan")].copy()
    all_machines = sorted(
        {
            int(m)
            for job in job_map.values()
            for mt in job.get("op_machine_times", {}).values()
            for m in mt.keys()
        }
    )
    if not all_machines:
        all_machines = [0]
    machine_load = {m: 0.0 for m in all_machines}
    feasible_counts = []
    single_machine_ops = 0
    used_ops = 0

    for _, row in new_df.iterrows():
        jid = int(row["job"])
        op_id = int(row["op"])
        job = job_map.get(jid)
        if not job:
            continue
        mt = job.get("op_machine_times", {}).get(op_id, {})
        vals = [float(v) for v in mt.values() if float(v) > 0]
        machines = [int(m) for m, v in mt.items() if float(v) > 0]
        if not vals or not machines:
            continue
        op_mean_pt = float(np.mean(vals))
        split_load = op_mean_pt / float(len(machines))
        for m in machines:
            machine_load[m] = machine_load.get(m, 0.0) + split_load
        feasible_counts.append(len(machines))
        single_machine_ops += int(len(machines) == 1)
        used_ops += 1

    loads = np.asarray([machine_load[m] for m in sorted(machine_load)], dtype=float)
    total_load = float(loads.sum())
    mean_load = float(loads.mean()) if loads.size else 0.0
    max_load = float(loads.max()) if loads.size else 0.0
    positive = loads[loads > 0]
    nonzero_machine_count = int(positive.size)
    load_imbalance = max_load / mean_load if mean_load > 1e-12 else 0.0
    bottleneck_share = max_load / total_load if total_load > 1e-12 else 0.0
    if total_load > 1e-12 and loads.size > 1:
        p = loads / total_load
        p = p[p > 0]
        normalized_entropy = float(-(p * np.log(p)).sum() / np.log(loads.size))
    else:
        normalized_entropy = 0.0

    return {
        "seed": seed,
        "release_index": int(release_row["release_index"]),
        "release_label": release_row["release_label"],
        "event_id": int(release_row["Event_ID"]),
        "release_time": safe_num(release_row, "Release_Time"),
        "machine_conflict_ops": int(used_ops),
        "machine_total_load_mean_split": total_load,
        "machine_mean_load_mean_split": mean_load,
        "machine_max_load_mean_split": max_load,
        "machine_load_imbalance": load_imbalance,
        "machine_bottleneck_share": bottleneck_share,
        "machine_load_entropy": normalized_entropy,
        "machine_conflict_entropy": 1.0 - normalized_entropy,
        "single_machine_op_ratio": single_machine_ops / used_ops if used_ops else 0.0,
        "avg_feasible_machine_count": float(np.mean(feasible_counts)) if feasible_counts else 0.0,
        "min_feasible_machine_count": float(np.min(feasible_counts)) if feasible_counts else 0.0,
        "nonzero_machine_count": nonzero_machine_count,
        **{f"machine_load_m{m}": float(machine_load[m]) for m in sorted(machine_load)},
    }


def build_init_completion_profile(seed: str, release_df: pd.DataFrame, init_detail_df: pd.DataFrame):
    rows = []
    init_jobs = sorted(init_detail_df["job"].unique().tolist())
    init_total_ops = int(len(init_detail_df))
    job_total_ops = init_detail_df.groupby("job")["op"].nunique().to_dict()
    job_completion = init_detail_df.groupby("job")["end"].max().to_dict()
    for _, rel in release_df.iterrows():
        t = safe_num(rel, "Release_Time")
        completed_ops = int((init_detail_df["end"].astype(float) <= t).sum())
        completed_jobs = int(sum(1 for jid in init_jobs if float(job_completion.get(jid, np.inf)) <= t))
        rows.append(
            {
                "seed": seed,
                "release_index": int(rel["release_index"]),
                "release_label": rel["release_label"],
                "event_id": int(rel["Event_ID"]),
                "release_time": t,
                "init_completed_ops": completed_ops,
                "init_total_ops": init_total_ops,
                "init_completed_op_ratio": completed_ops / init_total_ops if init_total_ops else 0.0,
                "init_completed_jobs": completed_jobs,
                "init_total_jobs": len(init_jobs),
                "init_completed_job_ratio": completed_jobs / len(init_jobs) if init_jobs else 0.0,
                "init_remaining_jobs": len(init_jobs) - completed_jobs,
                "init_remaining_ops": init_total_ops - completed_ops,
                "init_avg_ops_per_remaining_job": (
                    sum(max(0, int(job_total_ops[jid]) - int((init_detail_df[(init_detail_df["job"] == jid) & (init_detail_df["end"] <= t)]["op"].nunique()))) for jid in init_jobs)
                    / max(1, len(init_jobs) - completed_jobs)
                ),
            }
        )
    return rows


def plot_release_lines(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    metrics = [
        ("release_time", "Release Time"),
        ("tardiness", "Global Tardiness"),
        ("newplan_jobs", "NewPlan Jobs"),
        ("newplan_ops", "NewPlan Ops"),
        ("repeated_job_count", "Repeated Jobs"),
        ("completed_jobs_before_release", "Completed Jobs Before Release"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        for seed, sub in df.groupby("seed"):
            ax.plot(sub["event_id"], sub[col], marker="o", label=f"seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("Event ID")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "release_backlog_comparison.png", dpi=180)
    plt.close(fig)


def plot_slack_lines(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    metrics = [
        ("best_neg_slack_ratio", "Best-Machine Negative Slack Ratio"),
        ("best_mean_neg_slack", "Best-Machine Mean Negative Slack"),
        ("best_min_slack", "Best-Machine Min Slack"),
        ("best_q25_slack", "Best-Machine Q25 Slack"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        for seed, sub in df.groupby("seed"):
            ax.plot(sub["event_id"], sub[col], marker="o", label=f"seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("Event ID")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_slack_comparison.png", dpi=180)
    plt.close(fig)


def plot_machine_conflict_lines(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    metrics = [
        ("machine_load_imbalance", "Machine Load Imbalance"),
        ("machine_bottleneck_share", "Bottleneck Share"),
        ("single_machine_op_ratio", "Single-Machine Op Ratio"),
        ("avg_feasible_machine_count", "Avg Feasible Machine Count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        for seed, sub in df.groupby("seed"):
            ax.plot(sub["event_id"], sub[col], marker="o", label=f"seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("Event ID")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "machine_conflict_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze why OR-Tools big-env seeds produce different tardiness.")
    parser.add_argument("--ortools_root", default=DEFAULT_ORTOOLS_ROOT)
    parser.add_argument("--instance_pattern", default=DEFAULT_INSTANCE_PATTERN)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.instance_pattern)
    release_rows = []
    slack_rows = []
    machine_conflict_rows = []
    init_profile_rows = []

    run_dirs = [p for p in sorted(Path(args.ortools_root).iterdir()) if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No OR-Tools run folders found under {args.ortools_root}")

    for run_dir in run_dirs:
        seed = infer_seed_from_text(run_dir.name)
        if seed not in instances:
            print(f"[WARN] Skip {run_dir.name}: no matching dynamic instance JSON for seed {seed}")
            continue
        job_map = instances[seed]["jobs"]
        release_df = read_release_log(run_dir)
        details = read_details(run_dir)
        for _, rel in release_df.iterrows():
            release_index = int(rel["release_index"])
            if release_index not in details:
                print(f"[WARN] Missing details for {run_dir.name} release index {release_index}")
                continue
            detail_df = details[release_index]["df"]
            release_rows.append(summarize_details(seed, rel, detail_df, job_map))
            slack_rows.extend(compute_candidate_slack(seed, rel, detail_df, job_map))
            machine_conflict_rows.append(compute_machine_conflict(seed, rel, detail_df, job_map))

        if 1 in details:
            init_profile_rows.extend(build_init_completion_profile(seed, release_df, details[1]["df"]))

    release_enriched = pd.DataFrame(release_rows)
    candidate_pairs = pd.DataFrame(slack_rows)
    candidate_summary = summarize_candidate_slack(candidate_pairs)
    machine_conflict_summary = pd.DataFrame(machine_conflict_rows)
    init_profile = pd.DataFrame(init_profile_rows)

    if not candidate_summary.empty and not release_enriched.empty:
        release_enriched = release_enriched.merge(
            candidate_summary,
            on=["seed", "release_index", "release_label", "event_id", "release_time"],
            how="left",
        )
    if not machine_conflict_summary.empty and not release_enriched.empty:
        release_enriched = release_enriched.merge(
            machine_conflict_summary,
            on=["seed", "release_index", "release_label", "event_id", "release_time"],
            how="left",
        )
    if not init_profile.empty and not release_enriched.empty:
        release_enriched = release_enriched.merge(
            init_profile[
                [
                    "seed",
                    "release_index",
                    "event_id",
                    "init_completed_op_ratio",
                    "init_completed_job_ratio",
                    "init_remaining_jobs",
                    "init_remaining_ops",
                ]
            ],
            on=["seed", "release_index", "event_id"],
            how="left",
        )

    release_enriched.to_csv(out_dir / "release_enriched.csv", index=False)
    candidate_pairs.to_csv(out_dir / "candidate_slack_pairs.csv", index=False)
    candidate_summary.to_csv(out_dir / "candidate_slack_summary.csv", index=False)
    machine_conflict_summary.to_csv(out_dir / "machine_conflict_summary.csv", index=False)
    init_profile.to_csv(out_dir / "init_completion_profile.csv", index=False)
    plot_release_lines(release_enriched, out_dir)
    plot_slack_lines(candidate_summary, out_dir)
    plot_machine_conflict_lines(machine_conflict_summary, out_dir)

    cols = [
        "seed",
        "release_label",
        "release_time",
        "tardiness",
        "newplan_jobs",
        "newplan_ops",
        "repeated_job_count",
        "init_completed_job_ratio",
        "best_neg_slack_ratio",
        "best_mean_neg_slack",
        "best_q25_slack",
        "machine_load_imbalance",
        "machine_bottleneck_share",
        "single_machine_op_ratio",
        "avg_feasible_machine_count",
    ]
    print(f"Outputs written to: {out_dir.resolve()}")
    print(release_enriched[[c for c in cols if c in release_enriched.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
