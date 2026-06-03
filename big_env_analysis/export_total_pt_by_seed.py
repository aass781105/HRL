import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PATTERN = "dynamic_instances/dynamic_seed42_init50_h80_U20_50_due20_80_seed*.json"
DEFAULT_OUTPUT = "big_env_analysis/results/total_pt_by_seed.csv"


def infer_seed(path: Path, payload: dict) -> str:
    meta_seed = payload.get("meta", {}).get("seed")
    if meta_seed is not None:
        return str(meta_seed)
    matches = re.findall(r"seed(\d+)", path.stem)
    return matches[-1] if matches else path.stem


def compute_total_pt_by_op_machine_mean(job: dict) -> float:
    total = 0.0
    for op in sorted(job.get("operations", []) or [], key=lambda x: int(x.get("op_id", 0))):
        vals = [float(v) for v in (op.get("machine_times", {}) or {}).values() if float(v) > 0]
        if vals:
            total += float(np.mean(vals))
    return total


def main():
    parser = argparse.ArgumentParser(description="Export dynamic job total_pt table by seed.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    rows = {}
    for raw_path in paths:
        path = Path(raw_path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        seed = infer_seed(path, payload)
        col = f"seed{seed}"
        for job in payload.get("jobs", []) or []:
            job_id = int(job["job_id"])
            rows.setdefault(job_id, {"job_id": job_id})
            rows[job_id][col] = compute_total_pt_by_op_machine_mean(job)

    df = pd.DataFrame([rows[job_id] for job_id in sorted(rows)])
    seed_cols = sorted([c for c in df.columns if c.startswith("seed")], key=lambda x: int(re.findall(r"\d+", x)[0]))
    df = df[["job_id"] + seed_cols]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
