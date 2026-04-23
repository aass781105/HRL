import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import SD2_instance_generator
from params import configs


# Edit these declarations directly when you want to change the analysis setup.
INSTANCE_SIZES = [10, 20, 30, 40, 50]
INSTANCE_NUM = 100
N_MACHINES = 5
GEN_MODE = "uniform"
SEED_BASE = 4200
OUTPUT_DIR = "batch_static_exports"


MU_RANGE_LABELS = ["low", "mid", "high"]
SIGMA_RANGE_LABELS = ["low", "mid", "high"]


def compute_job_work(job_length, op_pt):
    job_work = []
    op_idx = 0
    for j in range(int(job_length.shape[0])):
        work = 0.0
        for _ in range(int(job_length[j])):
            pt_row = op_pt[op_idx]
            compat = pt_row[pt_row > 0]
            if compat.size > 0:
                work += float(np.mean(compat))
            op_idx += 1
        job_work.append(work)
    return np.asarray(job_work, dtype=float)


def build_three_point_values(lower, upper):
    lower = float(lower)
    upper = float(upper)
    middle = (lower + upper) / 2.0
    return [lower, middle, upper]


def generate_norm_due_dates_by_combo(n_j, rng, mu_bin_idx, sigma_bin_idx):
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    a = float(N_MACHINES) * mean_pt

    mu_values = build_three_point_values(-a /2.0 , a * 1.5)
    sigma_values = build_three_point_values(a / 2.0, a)

    mu_value = float(mu_values[int(mu_bin_idx)])
    sigma_value = float(sigma_values[int(sigma_bin_idx)])

    mu = np.full(shape=(int(n_j),), fill_value=mu_value, dtype=float)
    sigma = np.full(shape=(int(n_j),), fill_value=sigma_value, dtype=float)
    due_date = rng.normal(loc=mu, scale=sigma, size=int(n_j))
    return due_date, mu_value, sigma_value


def collect_distribution_rows(instance_sizes):
    rows = []
    old_n_j = int(configs.n_j)
    old_n_m = int(configs.n_m)
    try:
        configs.n_m = int(N_MACHINES)
        for size_idx, n_j in enumerate(instance_sizes):
            configs.n_j = int(n_j)
            for inst_idx in range(int(INSTANCE_NUM)):
                seed = int(SEED_BASE) + size_idx * 100000 + inst_idx
                jl, pt, _ = SD2_instance_generator(configs, seed=seed, mode=str(GEN_MODE))
                job_work = compute_job_work(jl, pt)

                for mu_bin_idx, mu_label in enumerate(MU_RANGE_LABELS):
                    for sigma_bin_idx, sigma_label in enumerate(SIGMA_RANGE_LABELS):
                        combo_seed = seed + mu_bin_idx * 10000 + sigma_bin_idx * 1000
                        rng = np.random.default_rng(combo_seed)
                        dd, mu_value, sigma_value = generate_norm_due_dates_by_combo(
                            n_j=int(jl.shape[0]),
                            rng=rng,
                            mu_bin_idx=mu_bin_idx,
                            sigma_bin_idx=sigma_bin_idx,
                        )
                        combo_id = f"mu_{mu_label}__sigma_{sigma_label}"

                        for job_id in range(int(jl.shape[0])):
                            rows.append(
                                {
                                    "instance_size": int(n_j),
                                    "instance_label": f"{int(n_j)}x{int(N_MACHINES)}",
                                    "instance_id": int(inst_idx),
                                    "combo_id": combo_id,
                                    "mu_bin": str(mu_label),
                                    "sigma_bin": str(sigma_label),
                                    "job_id": int(job_id),
                                    "job_op_count": int(jl[job_id]),
                                    "job_work": float(job_work[job_id]),
                                    "due_date": float(dd[job_id]),
                                    "slack": float(dd[job_id] - job_work[job_id]),
                                    "mu_value": float(mu_value),
                                    "sigma_value": float(sigma_value),
                                }
                            )
    finally:
        configs.n_j = old_n_j
        configs.n_m = old_n_m

    return pd.DataFrame(rows)


def build_combo_boxplot(df, value_col, ylabel, title, output_path):
    combo_order = [
        f"mu_{mu_label}__sigma_{sigma_label}"
        for mu_label in MU_RANGE_LABELS
        for sigma_label in SIGMA_RANGE_LABELS
    ]
    labels = []
    grouped = []
    for combo_id in combo_order:
        values = df.loc[df["combo_id"] == combo_id, value_col].dropna()
        if values.empty:
            continue
        grouped.append(values.astype(float).tolist())
        labels.append(combo_id.replace("__", "\n"))

    if not grouped:
        raise ValueError(f"No data collected for {value_col}.")

    fig, ax = plt.subplots(figsize=(16, 8))
    box = ax.boxplot(grouped, tick_labels=labels, patch_artist=True, showfliers=True)
    palette = ["#9ecae1", "#fdae6b", "#a1d99b", "#fdd0a2", "#bcbddc", "#fa9fb5", "#c7e9c0", "#c6dbef", "#fdd49e"]
    for idx, patch in enumerate(box["boxes"]):
        patch.set(facecolor=palette[idx % len(palette)], edgecolor="#4d4d4d", linewidth=1.0)
    for median in box["medians"]:
        median.set(color="#b22222", linewidth=1.2)
    for whisker in box["whiskers"]:
        whisker.set(color="#4d4d4d", linewidth=0.9)
    for cap in box["caps"]:
        cap.set(color="#4d4d4d", linewidth=0.9)

    if value_col == "slack":
        ax.axhline(0, color="#b22222", linestyle="--", linewidth=0.9)

    ax.set_title(title)
    ax.set_xlabel("mu / sigma Combination")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = collect_distribution_rows(INSTANCE_SIZES)

    stem = f"due_date_norm_combo_{GEN_MODE}_{len(INSTANCE_SIZES)}sizes_{INSTANCE_NUM}inst"
    raw_csv = os.path.join(OUTPUT_DIR, f"{stem}_raw.csv")
    summary_csv = os.path.join(OUTPUT_DIR, f"{stem}_summary.csv")
    due_png = os.path.join(OUTPUT_DIR, f"{stem}_due_date_boxplot.png")
    slack_png = os.path.join(OUTPUT_DIR, f"{stem}_slack_boxplot.png")

    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    summary = df.groupby(["combo_id", "instance_label"])[["due_date", "slack", "job_work"]].describe()
    summary.to_csv(summary_csv, encoding="utf-8-sig")

    build_combo_boxplot(
        df,
        value_col="due_date",
        ylabel="Due Date",
        title=f"Due Date Boxplots by mu/sigma Combination | mode={GEN_MODE}",
        output_path=due_png,
    )
    build_combo_boxplot(
        df,
        value_col="slack",
        ylabel="Slack (Due Date - Job Work)",
        title=f"Slack Boxplots by mu/sigma Combination | mode={GEN_MODE}",
        output_path=slack_png,
    )

    print(f"Saved raw data to: {raw_csv}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved due-date boxplot grid to: {due_png}")
    print(f"Saved slack boxplot grid to: {slack_png}")


if __name__ == "__main__":
    main()
