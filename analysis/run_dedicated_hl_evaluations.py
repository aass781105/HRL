"""Run scenario-specific high-level gate evaluations without PowerShell path parsing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260819")
OLD_URGENT_ROOT = RESULTS_ROOT / "urgent"
BASE_CONFIG = REPO_ROOT / "yaml_config" / "eval_baseline_seed{seed}_greedy_cadence1_1run.yml"
LOW_LEVEL_WEIGHT = r"trained_weights\lower_level\ll_u1030_esttd_odprog_ptscale.pth"

SCENARIO_WEIGHTS = {
    "baseline": r"trained_weights\high_level\hlgate_baseline_stab05_e16_newstate.pth",
    "urgent": r"trained_weights\high_level\hlgate_urgent_stab05_e16_newstate.pth",
    "burst_cluster": r"trained_weights\high_level\hlgate_multijob_stab05_e16_newstate.pth",
}

NEW_SCENARIO_WEIGHTS = {
    "baseline": r"trained_weights\high_level\hlgate_baseline_stab05_e16_ptscale.pth",
    "urgent": r"trained_weights\high_level\hlgate_urgent_stab05_e16_ptscale.pth",
    "burst_cluster": r"trained_weights\high_level\hlgate_multijob_stab05_e16_ptscale.pth",
}


def choose_output_root() -> Path:
    root = RESULTS_ROOT / "dedicated"
    root.mkdir(parents=True, exist_ok=True)
    return root


def copy_existing_urgent_baselines(output_root: Path) -> None:
    """Reuse the already completed urgent non-PPO runs without rerunning them."""
    target_root = output_root / "urgent"
    target_root.mkdir(parents=True, exist_ok=True)
    for strategy in ("cad1", "cad5", "slack0"):
        source = OLD_URGENT_ROOT / strategy
        target = target_root / strategy
        if target.exists():
            existing_target_files = list(target.rglob("sample_runs_summary.csv"))
            if len(existing_target_files) == 10:
                print(f"[REUSE] urgent/{strategy}: dedicated copy already exists")
                continue
            raise RuntimeError(f"Partial target already exists: {target}")
        if not source.is_dir():
            raise FileNotFoundError(f"Existing urgent strategy directory not found: {source}")
        sample_files = list(source.rglob("sample_runs_summary.csv"))
        if len(sample_files) != 10:
            raise RuntimeError(
                f"Expected 10 existing urgent {strategy} runs, found {len(sample_files)}"
            )
        shutil.copytree(source, target)
        print(f"[REUSE] urgent/{strategy}: copied {len(sample_files)} existing runs")


def strategy_args(strategy: str, scenario: str, weight: str) -> list[str]:
    args = ["--hl_env_scenario", scenario]
    if strategy in {"ppo", "ppo_new"}:
        args += ["--hl_gate_policy", "ppo", "--hl_ppo_model_path", weight]
    elif strategy == "cad1":
        args += ["--hl_gate_policy", "cadence", "--hl_gate_cadence", "1"]
    elif strategy == "cad5":
        args += ["--hl_gate_policy", "cadence", "--hl_gate_cadence", "5"]
    elif strategy == "slack0":
        args += [
            "--hl_gate_policy", "slack_threshold",
            "--hl_buffer_slack_release_threshold", "0.0",
        ]
    else:
        raise ValueError(strategy)
    return args


def run_one(*, scenario: str, strategy: str, seed: int, output_root: Path) -> None:
    config = Path(str(BASE_CONFIG).format(seed=seed))
    if not config.is_file():
        raise FileNotFoundError(config)

    output_dir = output_root / scenario / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_suffix = f"_{scenario}_{strategy}_seed{seed:02d}_seed{seed:03d}"
    completed_dirs = [
        path for path in output_dir.iterdir()
        if path.is_dir()
        and path.name.endswith(expected_suffix)
        and (path / "sample_runs_summary.csv").is_file()
    ]
    if completed_dirs:
        print(f"[SKIP] {scenario}/{strategy}/seed{seed:02d} already completed")
        return
    if strategy == "ppo_new":
        weight = NEW_SCENARIO_WEIGHTS[scenario]
    else:
        weight = SCENARIO_WEIGHTS[scenario]
    command = [
        sys.executable,
        str(REPO_ROOT / "hrl_main.py"),
        "--config", str(config),
        "--plot_global_dir", str(output_dir),
        "--plot_run_name", f"{scenario}_{strategy}_seed{seed:02d}",
        "--event_seed", str(seed),
        "--eval_seed", str(seed),
        "--main_sample_runs", "1",
        "--fast_mode", "true",
        "--ll_ppo_model_path", LOW_LEVEL_WEIGHT,
        *strategy_args(strategy, scenario, weight),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-40:])
        raise RuntimeError(
            f"Failed: scenario={scenario}, strategy={strategy}, seed={seed}\n{tail}"
        )

    summary_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(("[PPO-GATE]", "Run 01/", "Sample x1"))
    ]
    print(f"[DONE] {scenario}/{strategy}/seed{seed:02d}")
    for line in summary_lines[-3:]:
        print(f"       {line}")


def main() -> None:
    output_root = choose_output_root()
    print(f"[OUTPUT] {output_root}")
    copy_existing_urgent_baselines(output_root)

    tasks = [("urgent", "ppo")]
    for scenario in ("baseline", "burst_cluster"):
        for strategy in ("ppo", "cad1", "cad5", "slack0"):
            tasks.append((scenario, strategy))
    for scenario in ("baseline", "urgent", "burst_cluster"):
        tasks.append((scenario, "ppo_new"))

    for index, (scenario, strategy) in enumerate(tasks, start=1):
        print(f"[GROUP {index}/{len(tasks)}] {scenario}/{strategy}")
        for seed in range(1, 11):
            print(f"[RUN] {index}/{len(tasks)} {scenario}/{strategy}/seed{seed:02d}")
            run_one(scenario=scenario, strategy=strategy, seed=seed, output_root=output_root)

    print(f"[COMPLETE] {len(tasks)} strategy groups, 10 seeds each")
    print(f"[RESULT_ROOT] {output_root}")


if __name__ == "__main__":
    main()
