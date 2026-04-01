# plotters.py
import os
from typing import List, Dict, Optional
import matplotlib
mpl_backend = matplotlib.get_backend()
# 強制使用 Agg 後端以避免 Windows 上的 Tkinter 執行緒衝突
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _color_for_job(job_id: int, cmap_name: str = "tab20"):
    cmap = mpl.cm.get_cmap(cmap_name)
    N = getattr(cmap, "N", 20)
    idx = int(job_id) % max(1, int(N))
    return cmap(idx)

def _plot_rows(ax, rows: List[Dict], *, machines_order=None, alpha=1.0, linestyle="-", edge="black", lw=0.5, label_prefix="", highlight_op: Optional[int] = None):
    if not rows:
        return {}, []

    unique_rows = []
    seen = set()
    for r in rows:
        key = (
            int(r["job"]),
            int(r["op"]),
            int(r["machine"]),
            round(float(r["start"]), 8),
            round(float(r["end"]), 8),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(r)

    machines = machines_order or sorted({int(r["machine"]) for r in unique_rows})
    m_to_y = {m: i for i, m in enumerate(machines)}
    patches = []
    rows_sorted = sorted(unique_rows, key=lambda r: (int(r["machine"]), float(r["start"]), int(r["job"]), int(r["op"])))
    min_start = min(float(r["start"]) for r in rows_sorted)
    max_end = max(float(r["end"]) for r in rows_sorted)
    x_span = max(1.0, max_end - min_start)
    label_fontsize = 6 if len(rows_sorted) < 120 else 5
    label_overlap_threshold = max(1.0, x_span * 0.03)
    label_offset_step = 0.18
    prev_label_center_by_machine = {}
    prev_label_offset_by_machine = {}

    for r in rows_sorted:
        m = int(r["machine"]); y = m_to_y[m]
        s = float(r["start"]); e = float(r["end"])
        dur = float(r.get("duration", e - s))
        job = int(r["job"])
        op = int(r["op"])
        color = _color_for_job(job)
        is_highlight = (highlight_op is not None and op == int(highlight_op))
        label_center = s + dur / 2.0
        label_y = y

        prev_center = prev_label_center_by_machine.get(m)
        if prev_center is not None and abs(label_center - prev_center) < label_overlap_threshold:
            prev_offset = prev_label_offset_by_machine.get(m, -label_offset_step)
            label_y = y + (label_offset_step if prev_offset <= 0 else -label_offset_step)
        prev_label_center_by_machine[m] = label_center
        prev_label_offset_by_machine[m] = label_y - y

        ax.broken_barh([(s, dur)], (y - 0.4, 0.8),
                       facecolors=color,
                       edgecolor=("black" if is_highlight else "none"),
                       linewidth=(1.1 if is_highlight else 0.0),
                       alpha=alpha,
                       linestyle=linestyle)
        ax.text(label_center, label_y, f"{label_prefix}J{job}-O{op}",
                ha="center", va="center", fontsize=label_fontsize, color="black", clip_on=False)
        patches.append((y, s, dur))
    return m_to_y, patches

def plot_global_gantt(global_rows: List[Dict], save_path: str, *,
                      t_now: Optional[float] = None, title: str = "Global schedule"):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(12, 5))

    if not global_rows:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("time"); ax.set_ylabel("machine")
    else:
        machines = sorted({int(r["machine"]) for r in global_rows})
        ax.set_yticks(range(len(machines))); ax.set_yticklabels([f"M{m}" for m in machines])
        _plot_rows(ax, global_rows, machines_order=machines, alpha=0.85, linestyle="-", edge="none", lw=0.0, highlight_op=4)

        due_dates = {}
        for r in global_rows:
            if "due_date" not in r:
                continue
            try:
                due_dates[int(r["job"])] = float(r["due_date"])
            except Exception:
                continue

        for job, due_date in sorted(due_dates.items()):
            color = _color_for_job(job)
            ax.axvline(x=due_date, color=color, linestyle=":", linewidth=1.5, alpha=0.6)

        if t_now is not None:
            ax.axvline(t_now, linestyle="--", linewidth=1.0)
            ymin, ymax = ax.get_ylim()
            ax.text(t_now, ymax + 0.1, f"t={t_now:.2f}", ha="center", va="bottom", fontsize=9, clip_on=False)

        if due_dates:
            ymin, _ = ax.get_ylim()
            x_values = [float(r["end"]) for r in global_rows] + [float(v) for v in due_dates.values()]
            x_span = max(1.0, max(x_values) - min(float(r["start"]) for r in global_rows))
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

        ax.set_title(title)
        ax.set_xlabel("time")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# [ADDED] 同一張圖上畫「歷史 + 計畫」
def plot_global_gantt_with_plan(history_rows: List[Dict], plan_rows: List[Dict], save_path: str, *,
                                t_now: Optional[float] = None,
                                title: str = "Global (history + plan)"):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(13, 5))

    all_rows = (history_rows or []) + (plan_rows or [])
    if not all_rows:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("time"); ax.set_ylabel("machine")
    else:
        machines = sorted({int(r["machine"]) for r in all_rows})
        ax.set_yticks(range(len(machines))); ax.set_yticklabels([f"M{m}" for m in machines])

        # 歷史：實線、實心
        _plot_rows(ax, history_rows or [], machines_order=machines,
                   alpha=1.0, linestyle="-", edge="black", lw=0.6, label_prefix="")

        # 計畫：半透明、虛線框
        _plot_rows(ax, plan_rows or [], machines_order=machines,
                   alpha=0.45, linestyle="-", edge="gray", lw=0.6, label_prefix="")

        if t_now is not None:
            ax.axvline(t_now, linestyle="--", linewidth=1.0)
            ymin, ymax = ax.get_ylim()
            ax.text(t_now, ymax + 0.1, f"t={t_now:.2f}", ha="center", va="bottom", fontsize=9, clip_on=False)

        ax.set_title(title)
        ax.set_xlabel("time")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_batch_gantt(batch_rows: List[Dict], save_path: str, *,
                     t_now: Optional[float] = None, title: str = "Batch result"):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    # 這張圖只有計畫本身，所以直接用 global 的單層繪製
    plot_global_gantt(batch_rows, save_path, t_now=t_now, title=title)
    plt.close('all')
