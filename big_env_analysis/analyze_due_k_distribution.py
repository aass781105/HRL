import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATTERN = "dynamic_instances/dynamic_seed42_init50_h80_U20_50_due20_80_seed*.json"


def infer_seed(path: str, payload: dict) -> str:
    meta_seed = payload.get("meta", {}).get("seed", None)
    if meta_seed is not None:
        return str(meta_seed)
    matches = re.findall(r"seed(\d+)", Path(path).stem)
    if matches:
        return matches[-1]
    return Path(path).stem


def iter_jobs(payload: dict):
    jobs = payload.get("jobs", None)
    if isinstance(jobs, list) and jobs:
        for job in jobs:
            arrive_time = float(job.get("arrive_time", job.get("t_arrive_abs", 0.0)))
            yield ("init" if abs(arrive_time) < 1e-9 else "arrival"), job
        return

    for kind, key in (("init", "init_jobs"), ("arrival", "arrival_jobs")):
        for job in payload.get(key, []) or []:
            yield kind, job

    for event in payload.get("events", []) or []:
        event_jobs = event.get("jobs", [])
        for job in event_jobs:
            yield "arrival", job


def compute_job_total_pt_from_machine_mean(job: dict):
    total_mean = 0.0
    total_min = 0.0
    valid_ops = 0
    for op in sorted(job.get("operations", []) or [], key=lambda x: int(x.get("op_id", 0))):
        vals = [float(v) for v in (op.get("machine_times", {}) or {}).values() if float(v) > 0]
        if not vals:
            continue
        total_mean += float(np.mean(vals))
        total_min += float(np.min(vals))
        valid_ops += 1
    if valid_ops == 0:
        return np.nan, np.nan
    return total_mean, total_min


def load_job_rows(json_paths):
    rows = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        seed = infer_seed(path, payload)
        for job_type, job in iter_jobs(payload):
            arrive_time = float(job.get("arrive_time", job.get("t_arrive_abs", 0.0)))
            due_date = float(job["due_date"])
            total_proc_time, min_total_proc_time = compute_job_total_pt_from_machine_mean(job)
            k = (due_date - arrive_time) / total_proc_time if total_proc_time > 0 else np.nan
            k_min = (
                (due_date - arrive_time) / min_total_proc_time
                if np.isfinite(min_total_proc_time) and min_total_proc_time > 0
                else np.nan
            )
            rows.append(
                {
                    "seed": seed,
                    "source_file": Path(path).name,
                    "job_type": job_type,
                    "job_id": int(job["job_id"]),
                    "arrive_time": arrive_time,
                    "due_date": due_date,
                    "total_proc_time": total_proc_time,
                    "min_total_proc_time": min_total_proc_time,
                    "total_ops": int(job.get("total_ops", len(job.get("operations", [])))),
                    "k": k,
                    "k_min": k_min,
                }
            )
    return pd.DataFrame(rows)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("seed")["k"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            q25=lambda s: s.quantile(0.25),
            median="median",
            q75=lambda s: s.quantile(0.75),
            max="max",
        )
        .reset_index()
    )
    return summary


def plot_histograms(df: pd.DataFrame, out_path: Path, bin_low: float, bin_high: float, bin_width: float):
    seeds = sorted(df["seed"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    bins = np.arange(bin_low, bin_high + bin_width * 0.5, bin_width)
    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 2.8 * len(seeds)), sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        vals = df.loc[df["seed"] == seed, "k"].dropna().to_numpy()
        weights = np.ones_like(vals, dtype=float) / max(len(vals), 1) * 100.0
        ax.hist(vals, bins=bins, weights=weights, edgecolor="black", alpha=0.75)
        ax.set_title(f"Seed {seed} k distribution")
        ax.set_ylabel("Percent (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("k = (due_date - arrive_time) / total_proc_time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ecdf_box(df: pd.DataFrame, out_path: Path):
    seeds = sorted(df["seed"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for seed in seeds:
        vals = np.sort(df.loc[df["seed"] == seed, "k"].dropna().to_numpy())
        y = np.arange(1, len(vals) + 1) / max(len(vals), 1)
        ax.step(vals, y, where="post", label=f"seed {seed}")
    ax.set_title("ECDF of k by seed")
    ax.set_xlabel("k")
    ax.set_ylabel("Cumulative probability")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    data = [df.loc[df["seed"] == seed, "k"].dropna().to_numpy() for seed in seeds]
    try:
        ax.boxplot(data, tick_labels=[f"seed {s}" for s in seeds], showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=[f"seed {s}" for s in seeds], showmeans=True)
    ax.set_title("Boxplot of k by seed")
    ax.set_ylabel("k")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_path: Path):
    seeds = sorted(df["seed"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for seed in seeds:
        sub = df[df["seed"] == seed]
        axes[0].scatter(sub["arrive_time"], sub["k"], s=18, alpha=0.7, label=f"seed {seed}")
        axes[1].scatter(sub["total_proc_time"], sub["k"], s=18, alpha=0.7, label=f"seed {seed}")
    axes[0].set_title("k vs arrive_time")
    axes[0].set_xlabel("arrive_time")
    axes[0].set_ylabel("k")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("k vs total_proc_time")
    axes[1].set_xlabel("total_proc_time")
    axes[1].set_ylabel("k")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze due-date factor k distribution for dynamic big-env JSONs.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for dynamic instance JSON files.")
    parser.add_argument("--output_dir", default="big_env_analysis/results", help="Output folder.")
    parser.add_argument("--bin_low", type=float, default=2.0, help="Fixed histogram lower edge.")
    parser.add_argument("--bin_high", type=float, default=8.0, help="Fixed histogram upper edge.")
    parser.add_argument("--bin_width", type=float, default=0.5, help="Fixed histogram bin width.")
    args = parser.parse_args()

    json_paths = sorted(glob.glob(args.pattern))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files matched pattern: {args.pattern}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_job_rows(json_paths)
    summary = make_summary(df)

    df.to_csv(out_dir / "due_k_job_level.csv", index=False)
    summary.to_csv(out_dir / "due_k_summary_by_seed.csv", index=False)
    plot_histograms(df, out_dir / "due_k_histogram_fixed_bins.png", args.bin_low, args.bin_high, args.bin_width)
    plot_ecdf_box(df, out_dir / "due_k_ecdf_boxplot.png")
    plot_scatter(df, out_dir / "due_k_scatter.png")

    print(f"Analyzed {len(json_paths)} files, {len(df)} jobs.")
    print(f"Outputs written to: {out_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
