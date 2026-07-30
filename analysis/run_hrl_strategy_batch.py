"""Run baseline/urgent high-level strategy evaluations in the workspace.

Results stay under plots/global until the complete batch is verified and moved
in one operation. The already completed baseline PPO batch is skipped.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
HRL_MAIN = PROJECT_ROOT / "hrl_main.py"
CONFIG_DIR = PROJECT_ROOT / "yaml_config"
TEMP_CONFIG_DIR = CONFIG_DIR / "batch_tmp_release_eval"
PLOT_DIR = PROJECT_ROOT / "plots" / "global"
MANIFEST_PATH = PROJECT_ROOT / "analysis_results" / "hrl_strategy_batch_manifest.csv"

SCENARIOS = ("baseline", "urgent")
STRATEGIES = ("ppo", "cad1", "cad5", "slack0")
SEEDS = tuple(range(1, 11))

MODEL_PATHS = {
    "baseline": r"trained_weights\high_level\hlgate_baseline_stab05_e16_newstate.pth",
    "urgent": r"trained_weights\high_level\hlgate_urgent_stab05_e16_newstate.pth",
}

# These ten runs were already completed and moved to the external results
# directory before this batch runner was created.
SKIP_COMPLETED = {("baseline", "ppo")}


def base_config_path(seed: int) -> Path:
    return CONFIG_DIR / f"eval_baseline_seed{seed}_greedy_cadence1_1run.yml"


def make_config(scenario: str, strategy: str, seed: int) -> Path:
    source = base_config_path(seed)
    if not source.exists():
        raise FileNotFoundError(f"Missing seed config: {source}")

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config["event_seed"] = int(seed)
    config["hl_env_scenario"] = scenario
    config["hl_ppo_model_path"] = MODEL_PATHS[scenario]
    config["hl_ppo_name"] = f"hlgate_{scenario}_stab05_e16_newstate"
    config["hl_eval_action_selection"] = "greedy"

    if strategy == "ppo":
        config["hl_gate_policy"] = "ppo"
    elif strategy == "cad1":
        config["hl_gate_policy"] = "cadence"
        config["hl_gate_cadence"] = 1
    elif strategy == "cad5":
        config["hl_gate_policy"] = "cadence"
        config["hl_gate_cadence"] = 5
    elif strategy == "slack0":
        config["hl_gate_policy"] = "slack_threshold"
        config["hl_buffer_slack_release_threshold"] = 0.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    config["plot_run_name"] = f"{scenario}_{strategy}_seed{seed:02d}"
    config["main_sample_runs"] = 1
    config["eval_runs_per_instance"] = 1

    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output = TEMP_CONFIG_DIR / f"{scenario}_{strategy}_seed{seed:02d}.yml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return output


def locate_output(scenario: str, strategy: str, seed: int) -> str:
    prefix = f"{scenario}_{strategy}_seed{seed:02d}"
    matches = sorted(
        (path for path in PLOT_DIR.iterdir() if path.is_dir() and prefix in path.name),
        key=lambda path: path.stat().st_mtime,
    )
    return str(matches[-1].relative_to(PROJECT_ROOT)) if matches else ""


def append_manifest(row: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "strategy", "seed", "status", "output_dir"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_completed() -> dict[tuple[str, str, int], dict[str, str]]:
    """Load the latest successful record for each combination for resume runs."""
    if not MANIFEST_PATH.exists():
        return {}

    completed: dict[tuple[str, str, int], dict[str, str]] = {}
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["strategy"], int(row["seed"]))
            status = row["status"]
            output_dir = row.get("output_dir", "")
            if status == "already_moved":
                completed[key] = row
            elif status == "ok" and output_dir:
                output_path = PROJECT_ROOT / output_dir
                if output_path.is_dir():
                    completed[key] = row
    return completed


def main() -> None:
    if not HRL_MAIN.exists():
        raise FileNotFoundError(HRL_MAIN)

    total = len(SCENARIOS) * len(STRATEGIES) * len(SEEDS)
    completed = 0
    existing = load_completed()
    print(f"[BATCH] total combinations: {total}", flush=True)
    print(f"[BATCH] workspace output: {PLOT_DIR}", flush=True)
    print(f"[BATCH] resumable combinations: {len(existing)}", flush=True)

    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                completed += 1
                key = (scenario, strategy)
                resume_key = (scenario, strategy, seed)
                if resume_key in existing:
                    previous = existing[resume_key]
                    print(
                        f"[{completed}/{total}] RESUME-SKIP scenario={scenario} "
                        f"strategy={strategy} seed={seed:02d} "
                        f"status={previous['status']}",
                        flush=True,
                    )
                    continue
                if key in SKIP_COMPLETED:
                    print(
                        f"[{completed}/{total}] SKIP scenario={scenario} "
                        f"strategy={strategy} seed={seed:02d}",
                        flush=True,
                    )
                    append_manifest(
                        {
                            "scenario": scenario,
                            "strategy": strategy,
                            "seed": seed,
                            "status": "already_moved",
                            "output_dir": "external_results/baseline/ppo",
                        }
                    )
                    continue

                config_path = make_config(scenario, strategy, seed)
                print(
                    f"[{completed}/{total}] RUN scenario={scenario} "
                    f"strategy={strategy} seed={seed:02d}",
                    flush=True,
                )
                started = time.perf_counter()
                result = subprocess.run(
                    [str(PYTHON), str(HRL_MAIN), "--config", str(config_path)],
                    cwd=PROJECT_ROOT,
                    check=False,
                )
                output_dir = locate_output(scenario, strategy, seed)
                status = "ok" if result.returncode == 0 else f"failed_{result.returncode}"
                append_manifest(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "seed": seed,
                        "status": status,
                        "output_dir": output_dir,
                    }
                )
                print(
                    f"[BATCH] status={status} elapsed={time.perf_counter() - started:.1f}s "
                    f"output={output_dir}",
                    flush=True,
                )
                if result.returncode != 0:
                    raise SystemExit(result.returncode)

    print(f"[BATCH] completed. Manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
