# plotters.py
import os
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _color_for_job(job_id: int, cmap_name: str = "tab20"):
    cmap = mpl.cm.get_cmap(cmap_name)
    N = getattr(cmap, "N", 20)
    idx = int(job_id) % max(1, int(N))
    return cmap(idx)

def _plot_rows(ax, rows: List[Dict], *, machines_order=None, alpha=1.0, linestyle="-", edge="black", lw=0.5, label_prefix=""):
    if not rows:
        return {}, []

    machines = machines_order or sorted({int(r["machine"]) for r in rows})
    m_to_y = {m: i for i, m in enumerate(machines)}
    patches = []

    for r in rows:
        m = int(r["machine"]); y = m_to_y[m]
        s = float(r["start"]); e = float(r["end"])
        dur = float(r.get("duration", e - s))
        job = int(r["job"])
        color = _color_for_job(job)
        ax.broken_barh([(s, dur)], (y - 0.4, 0.8),
                       facecolors=color, edgecolor=edge, linewidth=lw, alpha=alpha, linestyle=linestyle)
        ax.text(s + dur/2.0, y, f"{label_prefix}J{job}-O{int(r['op'])}",
                ha="center", va="center", fontsize=8, color="black")
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
        _plot_rows(ax, global_rows, machines_order=machines, alpha=1.0, linestyle="-", edge="black", lw=0.5)

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
