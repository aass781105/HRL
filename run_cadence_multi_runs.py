import csv
import json
import os
from datetime import datetime
from statistics import mean, pstdev

from params import configs
from main import run_event_driven_until_nevents


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def main():
    # Force high-level policy to cadence.
    configs.gate_policy = "cadence"

    num_runs = max(1, _safe_int(getattr(configs, "eval_runs_per_instance", 10)))
    cadence = max(1, _safe_int(getattr(configs, "gate_cadence", 1)))
    base_seed = _safe_int(getattr(configs, "event_seed", 42))

    # Keep outputs separated from main run folders.
    out_dir = os.path.join("evaluation_results", "cadence_multi_runs")
    os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"cad{cadence}_runs{num_runs}_seed{base_seed}"
    per_run_csv = os.path.join(out_dir, f"{stamp}_{tag}_per_run.csv")
    summary_json = os.path.join(out_dir, f"{stamp}_{tag}_summary.json")

    rows = []
    mk_vals = []
    td_vals = []
    rel_vals = []

    print("-" * 30)
    print("Cadence Multi-Run Evaluation")
    print(f"Policy=cadence | cadence={cadence} | runs={num_runs} | base_seed={base_seed}")
    print("-" * 30)

    for run_id in range(1, num_runs + 1):
        run_seed = base_seed + (run_id - 1)
        configs.event_seed = run_seed
        configs.eval_model_name = f"cadence_run_{run_id:03d}"

        mk, stats = run_event_driven_until_nevents(
            max_events=int(configs.event_horizon),
            interarrival_mean=configs.interarrival_mean,
            burst_K=configs.burst_size,
            plot_global_dir=getattr(configs, "plot_global_dir", "plots/global"),
        )

        td = _safe_float(stats.get("total_tardiness", 0.0))
        rc = _safe_int(stats.get("release_count", 0))

        mk_vals.append(float(mk))
        td_vals.append(td)
        rel_vals.append(rc)

        rows.append(
            {
                "run_id": run_id,
                "event_seed": run_seed,
                "cadence": cadence,
                "makespan": float(mk),
                "total_tardiness": td,
                "release_count": rc,
            }
        )
        print(f"[Run {run_id:03d}] seed={run_seed} | MK={mk:.3f} | TD={td:.3f} | Releases={rc}")

    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "event_seed", "cadence", "makespan", "total_tardiness", "release_count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "policy": "cadence",
        "cadence": cadence,
        "num_runs": num_runs,
        "base_seed": base_seed,
        "event_horizon": _safe_int(getattr(configs, "event_horizon", 0)),
        "interarrival_mean": _safe_float(getattr(configs, "interarrival_mean", 0.0)),
        "burst_size": _safe_int(getattr(configs, "burst_size", 1)),
        "metrics": {
            "makespan_mean": mean(mk_vals),
            "makespan_std": pstdev(mk_vals) if len(mk_vals) > 1 else 0.0,
            "makespan_min": min(mk_vals),
            "makespan_max": max(mk_vals),
            "tardiness_mean": mean(td_vals),
            "tardiness_std": pstdev(td_vals) if len(td_vals) > 1 else 0.0,
            "tardiness_min": min(td_vals),
            "tardiness_max": max(td_vals),
            "release_mean": mean(rel_vals),
            "release_std": pstdev(rel_vals) if len(rel_vals) > 1 else 0.0,
            "release_min": min(rel_vals),
            "release_max": max(rel_vals),
        },
        "per_run_csv": per_run_csv,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print("Done.")
    print(f"Per-run CSV: {per_run_csv}")
    print(f"Summary JSON: {summary_json}")
    print(
        f"MK(mean±std)={summary['metrics']['makespan_mean']:.3f}±{summary['metrics']['makespan_std']:.3f} | "
        f"TD(mean±std)={summary['metrics']['tardiness_mean']:.3f}±{summary['metrics']['tardiness_std']:.3f}"
    )


if __name__ == "__main__":
    main()
