import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot due-date boxplots for selected batch events.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("batch_static_exports", "batch_static_jobs_runs10_cad1.csv"),
        help="Input CSV exported from export_dynamic_batch_static_csv.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path. Defaults to batch_static_exports/<csv_stem>_due_date_boxplot.png",
    )
    return parser.parse_args()


def build_target_events():
    return ["init"] + [f"event_{idx}" for idx in range(10, 101, 10)]


def main():
    args = parse_args()
    csv_path = str(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"batch_id", "due_date_rel"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    target_events = build_target_events()
    selected = df[df["batch_id"].isin(target_events)].copy()
    if selected.empty:
        raise ValueError("No rows found for init/event_10/.../event_100 in the input CSV.")

    labels = []
    series = []
    for event_name in target_events:
        event_values = selected.loc[selected["batch_id"] == event_name, "due_date_rel"].dropna()
        if event_values.empty:
            continue
        labels.append(event_name)
        series.append(event_values.astype(float).tolist())

    if not series:
        raise ValueError("Selected events exist, but all due_date_rel values are empty.")

    plt.figure(figsize=(14, 7))
    box = plt.boxplot(series, tick_labels=labels, patch_artist=True, showfliers=True)

    for patch in box["boxes"]:
        patch.set(facecolor="#9ecae1", edgecolor="#2b6c99", linewidth=1.2)
    for median in box["medians"]:
        median.set(color="#b22222", linewidth=1.4)
    for whisker in box["whiskers"]:
        whisker.set(color="#2b6c99", linewidth=1.0)
    for cap in box["caps"]:
        cap.set(color="#2b6c99", linewidth=1.0)

    plt.title("Relative Due Date Distribution by Batch Event")
    plt.xlabel("Batch Event")
    plt.ylabel("Relative Due Date")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    output_path = str(args.output or "").strip()
    if not output_path:
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join("batch_static_exports", f"{csv_stem}_due_date_boxplot.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved boxplot to: {output_path}")


if __name__ == "__main__":
    main()
