import csv
import math
import re
from pathlib import Path


ROOT = Path(r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\瓶頸")
SUMMARY_NAME = "sample_runs_summary.csv"
METRICS = ["makespan", "total_tardiness", "obj", "release_count", "elapsed_time_sec"]


def to_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def seed_from_name(name):
    match = re.search(r"seed[_-]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_summary(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    selected = next((r for r in rows if str(r.get("run", "")).lower() == "mean"), None)
    if selected is None:
        selected = next((r for r in rows if str(r.get("run", "")) == "1"), rows[0])
    return selected


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, scenario):
    out = {"scenario": scenario, "n": len(rows)}
    for metric in METRICS:
        values = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
        if not values:
            out[f"{metric}_mean"] = ""
            out[f"{metric}_std"] = ""
            out[f"{metric}_min"] = ""
            out[f"{metric}_max"] = ""
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = math.sqrt(var)
        out[f"{metric}_min"] = min(values)
        out[f"{metric}_max"] = max(values)
    return out


def main():
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    all_rows = []
    stats_rows = []
    scenario_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        rows = []
        for seed_dir in sorted([p for p in scenario_dir.iterdir() if p.is_dir()], key=lambda p: (seed_from_name(p.name) is None, seed_from_name(p.name) or p.name)):
            summary_csv = seed_dir / SUMMARY_NAME
            if not summary_csv.exists():
                continue
            data = read_summary(summary_csv)
            seed = seed_from_name(seed_dir.name)
            if seed is None and data.get("env_seed") not in ("", None):
                seed = int(float(data["env_seed"]))
            row = {
                "scenario": scenario,
                "seed": seed if seed is not None else "",
                "run_folder": seed_dir.name,
                "ppo_model_name": data.get("ppo_model_name", ""),
                "summary_csv": str(summary_csv),
            }
            for metric in METRICS:
                row[metric] = to_float(data.get(metric))
            rows.append(row)
            all_rows.append(row)

        rows.sort(key=lambda r: (r["seed"] == "", r["seed"]))
        fields = ["scenario", "seed", "run_folder", "ppo_model_name", *METRICS, "summary_csv"]
        write_csv(scenario_dir / "comparison_summary.csv", rows, fields)
        stats_rows.append(summarize(rows, scenario))

    all_fields = ["scenario", "seed", "run_folder", "ppo_model_name", *METRICS, "summary_csv"]
    write_csv(ROOT / "comparison_all.csv", sorted(all_rows, key=lambda r: (r["scenario"], r["seed"])), all_fields)

    pivot = {}
    for row in all_rows:
        seed = row["seed"]
        if seed == "":
            continue
        dst = pivot.setdefault(seed, {"seed": seed})
        scenario = row["scenario"]
        for metric in METRICS:
            dst[f"{scenario}_{metric}"] = row[metric]
    pivot_fields = ["seed"]
    for scenario_dir in scenario_dirs:
        for metric in METRICS:
            pivot_fields.append(f"{scenario_dir.name}_{metric}")
    write_csv(ROOT / "comparison_by_seed.csv", [pivot[k] for k in sorted(pivot)], pivot_fields)

    stats_fields = ["scenario", "n"]
    for metric in METRICS:
        stats_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    write_csv(ROOT / "comparison_stats.csv", stats_rows, stats_fields)

    print(f"Wrote {len(all_rows)} seed rows from {len(scenario_dirs)} scenarios under {ROOT}")


if __name__ == "__main__":
    main()
