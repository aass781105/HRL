import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "dynamic_instances"
BASE_CONFIG = ROOT / "yaml_config" / "run_big_env_PPO1.yml"
GENERATED_CONFIG_DIR = ROOT / "_dynamic_batch_configs"
SUMMARY_DIR = ROOT / "or_tools_solutions" / "dynamic_ortools_batch"

SUBPROBLEM_TIME_LIMIT_SEC = 7200.0


def load_base_config():
    if BASE_CONFIG.exists():
        with BASE_CONFIG.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    return data


def write_config(instance_path, base_config):
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rel_instance_path = os.path.relpath(instance_path, ROOT)
    cfg = dict(base_config)
    cfg.update(
        {
            "scheduler_type": "OR-Tools",
            "instance_json": rel_instance_path,
            "ortools_subproblem_time_limit": SUBPROBLEM_TIME_LIMIT_SEC,
            "ortools_total_solve_time_budget": 0.0,
            "fast_mode": False,
            "plot_run_name": f"ortools_{instance_path.stem}",
        }
    )
    config_path = GENERATED_CONFIG_DIR / f"{instance_path.stem}_ortools_2h.yml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return config_path


def append_summary(row):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "dynamic_ortools_batch_summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "instance_json",
                "config",
                "return_code",
                "elapsed_sec",
                "status",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    os.chdir(ROOT)
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    instances = sorted(INSTANCE_DIR.glob("*.json"))
    if not instances:
        print(f"No dynamic instance JSON files found under: {INSTANCE_DIR}")
        print("Put exported dynamic .json files into this folder, then run again.")
        return 1

    base_config = load_base_config()
    print("======================================================")
    print("Dynamic OR-Tools batch runner")
    print(f"Folder       : {ROOT}")
    print(f"Input folder : {INSTANCE_DIR}")
    print(f"Instances    : {len(instances)}")
    print(f"Time limit   : {SUBPROBLEM_TIME_LIMIT_SEC:.0f} sec per reschedule")
    print("======================================================")

    failures = 0
    for idx, instance_path in enumerate(instances, start=1):
        config_path = write_config(instance_path, base_config)
        print()
        print(f"[{idx}/{len(instances)}] Solving {instance_path.name}")
        print(f"Config: {config_path}")

        start = time.time()
        completed = subprocess.run(
            [sys.executable, "run_dynamic_ortools_cadence.py", "--config", str(config_path)],
            cwd=ROOT,
        )
        elapsed = time.time() - start
        status = "ok" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            failures += 1

        append_summary(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "instance_json": str(instance_path),
                "config": str(config_path),
                "return_code": completed.returncode,
                "elapsed_sec": f"{elapsed:.3f}",
                "status": status,
            }
        )
        print(f"Finished {instance_path.name}: {status}, elapsed={elapsed:.1f}s")

    print()
    print("======================================================")
    print("All dynamic OR-Tools instances done.")
    print(f"Failures: {failures}")
    print(f"Batch summary: {SUMMARY_DIR / 'dynamic_ortools_batch_summary.csv'}")
    print("Per-instance outputs are under plots/global.")
    print("======================================================")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
