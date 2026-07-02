import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _color_for_job(job_id: int, cmap_name: str = "tab20"):
    cmap = plt.get_cmap(cmap_name)
    n_colors = getattr(cmap, "N", 20)
    return cmap(int(job_id) % max(1, int(n_colors)))


def plot_ortools_gantt_with_due_dates(detail_rows: List[Dict], save_path: str, *, title: str = "OR-Tools Schedule"):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(14, 6))

    if not detail_rows:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return

    machines = sorted({int(row["Machine"]) for row in detail_rows})
    machine_to_y = {machine: idx for idx, machine in enumerate(machines)}
    due_dates = {}
    rows_sorted = sorted(detail_rows, key=lambda r: (int(r["Machine"]), float(r["Start"]), int(r["Job"]), int(r["Op"])))
    min_start = min(float(row["Start"]) for row in rows_sorted)
    max_end = max(float(row["End"]) for row in rows_sorted)
    x_span = max(1.0, max_end - min_start)
    label_fontsize = 6 if len(rows_sorted) < 120 else 5
    label_overlap_threshold = max(1.0, x_span * 0.03)
    label_offset_step = 0.18
    prev_label_center_by_machine = {}
    prev_label_offset_by_machine = {}

    for row in rows_sorted:
        job = int(row["Job"])
        op = int(row["Op"])
        machine = int(row["Machine"])
        start = float(row["Start"])
        end = float(row["End"])
        duration = float(row["Duration"])
        due_date = float(row["Due_Date"])
        y = machine_to_y[machine]
        color = _color_for_job(job)
        is_op4 = (op == 4)
        label_center = start + duration / 2.0
        label_y = y

        prev_center = prev_label_center_by_machine.get(machine)
        if prev_center is not None and abs(label_center - prev_center) < label_overlap_threshold:
            prev_offset = prev_label_offset_by_machine.get(machine, -label_offset_step)
            label_y = y + (label_offset_step if prev_offset <= 0 else -label_offset_step)
        prev_label_center_by_machine[machine] = label_center
        prev_label_offset_by_machine[machine] = label_y - y

        ax.barh(
            y,
            duration,
            left=start,
            color=color,
            edgecolor="black" if is_op4 else "none",
            linewidth=1.1 if is_op4 else 0.0,
            alpha=0.85,
            height=0.8,
        )
        ax.text(
            label_center,
            label_y,
            f"J{job}-O{op}",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color="black",
            clip_on=False,
        )
        due_dates[job] = due_date

    for job, due_date in sorted(due_dates.items()):
        color = _color_for_job(job)
        ax.axvline(x=due_date, color=color, linestyle=":", linewidth=1.5, alpha=0.6)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"M{machine}" for machine in machines])

    ymin, _ = ax.get_ylim()
    close_due_threshold = max(0.8, x_span * 0.02)
    label_levels = [ymin - 0.15, ymin - 0.42]
    prev_due_date = None
    current_level = 0
    for job, due_date in sorted(due_dates.items(), key=lambda x: x[1]):
        color = _color_for_job(job)
        if prev_due_date is not None and abs(due_date - prev_due_date) < close_due_threshold:
            current_level = 1 - current_level
        else:
            current_level = 0
        ax.text(
            due_date,
            label_levels[current_level],
            f"D{job}",
            color=color,
            rotation=90,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False,
        )
        prev_due_date = due_date

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
