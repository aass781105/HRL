"""Plot event-level strategy differences against cadence1.

The script reuses exported ``*_ppo_release_log.csv`` files.  Global metrics
are forward-filled between releases, so HOLD events keep the previous global
schedule value.  Release events are marked on the step plots.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_ROOT = Path(
    r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260729\結果"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_results" / "event_level_difference"
STRATEGIES = ("ppo", "cadence5", "slack0")
STRATEGY_DIRS = {
    "cadence1": "cad1",
    "ppo": "ppo",
    "cadence5": "cad5",
    "slack0": "slack0",
}
SCENARIOS = ("baseline", "urgent", "多單")
SEEDS = tuple(range(1, 11))
STRATEGY_DIRS_BY_SCENARIO = {
    "baseline": {
        "cadence1": "cad1",
        "ppo": "ppo",
        "cadence5": "cad5",
        "slack0": "slack0",
    },
    "urgent": {
        "cadence1": "cad1",
        "ppo": "ppo",
        "cadence5": "cad5",
        "slack0": "slack0",
    },
    "多單": {
        "cadence1": "cad1",
        "ppo": "ppo_term",
        "cadence5": "cad5",
        "slack0": "slack0",
    },
}


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def find_strategy_dir(result_root: Path, scenario: str, strategy: str, seed: int) -> Path:
    root = result_root / scenario / STRATEGY_DIRS_BY_SCENARIO[scenario][strategy]
    if not root.is_dir():
        raise FileNotFoundError(root)

    markers = (f"seed{seed:03d}", f"seed{seed:02d}")
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and any(marker in path.name for marker in markers)
        and any(path.glob("*_ppo_release_log.csv"))
    )
    if not candidates:
        raise FileNotFoundError(f"Missing {scenario}/{strategy}/seed{seed:02d}")
    return candidates[-1]


def read_release_log(folder: Path) -> dict[int, dict[str, object]]:
    files = sorted(folder.glob("*_ppo_release_log.csv"))
    if not files:
        raise FileNotFoundError(f"Release log missing: {folder}")

    releases: dict[int, dict[str, object]] = {}
    with files[-1].open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                event = int(row["Event_ID"])
            except (KeyError, TypeError, ValueError):
                continue
            global_mk = parse_float(row.get("Global_Makespan"))
            global_td = parse_float(row.get("Global_Total_Tardiness"))
            global_obj = parse_float(row.get("Global_Objective_0p5MK_0p5TD"))
            if math.isnan(global_obj) and not math.isnan(global_mk) and not math.isnan(global_td):
                global_obj = 0.5 * global_mk + 0.5 * global_td
            releases[event] = {
                "mk": global_mk,
                "td": global_td,
                "obj": global_obj,
                "release_type": row.get("Release_Type", ""),
                "release_time": parse_float(row.get("Release_Time")),
            }
    if not releases:
        raise RuntimeError(f"No numeric release events in {files[-1]}")
    return releases


def forward_fill(releases: dict[int, dict[str, object]], event_horizon: int) -> list[dict[str, object]]:
    current: dict[str, object] = {"mk": math.nan, "td": math.nan, "obj": math.nan}
    rows: list[dict[str, object]] = []
    for event in range(event_horizon + 1):
        release = releases.get(event)
        if release is not None:
            current = {key: release[key] for key in ("mk", "td", "obj")}
        rows.append(
            {
                "event": event,
                "mk": current["mk"],
                "td": current["td"],
                "obj": current["obj"],
                "released": int(release is not None),
                "release_type": release.get("release_type", "") if release else "",
                "release_time": release.get("release_time", math.nan) if release else math.nan,
            }
        )
    return rows


def difference_rows(
    scenario: str,
    seed: int,
    strategy: str,
    strategy_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    last_delta = {"mk": math.nan, "td": math.nan, "obj": math.nan}
    anchor_event: int | None = None
    for current, reference in zip(strategy_rows, reference_rows):
        if current["released"]:
            for metric in ("mk", "td", "obj"):
                strategy_value = current[metric]
                reference_value = reference[metric]
                last_delta[metric] = (
                    strategy_value - reference_value
                    if not math.isnan(strategy_value) and not math.isnan(reference_value)
                    else math.nan
                )
            anchor_event = int(current["event"])
        row: dict[str, object] = {
            "scenario": scenario,
            "seed": seed,
            "event": current["event"],
            "strategy": strategy,
            "strategy_mk": current["mk"],
            "cad1_mk": reference["mk"],
            "delta_mk": last_delta["mk"],
            "strategy_td": current["td"],
            "cad1_td": reference["td"],
            "delta_td": last_delta["td"],
            "strategy_obj": current["obj"],
            "cad1_obj": reference["obj"],
            "delta_obj": last_delta["obj"],
            "strategy_released": current["released"],
            "release_type": current["release_type"],
            "release_time": current["release_time"],
            "delta_anchor_event": anchor_event,
        }
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_seed(
    path: Path,
    scenario: str,
    seed: int,
    rows_by_strategy: dict[str, list[dict[str, object]]],
) -> None:
    colors = {"ppo": "#2563eb", "cadence5": "#dc2626", "slack0": "#16a34a"}
    metrics = (("delta_td", "Delta TD"), ("delta_mk", "Delta MK"), ("delta_obj", "Delta Obj"))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (column, title) in zip(axes, metrics):
        for strategy in STRATEGIES:
            rows = rows_by_strategy[strategy]
            events = [row["event"] for row in rows]
            values = [row[column] for row in rows]
            ax.step(events, values, where="post", color=colors[strategy], label=strategy)
            release_events = [
                row["event"]
                for row in rows
                if row["strategy_released"] and not math.isnan(row[column])
            ]
            release_values = [
                row[column]
                for row in rows
                if row["strategy_released"] and not math.isnan(row[column])
            ]
            ax.scatter(
                release_events,
                release_values,
                color=colors[strategy],
                s=22,
                zorder=3,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Event")
    fig.suptitle(f"Event-level difference vs cadence1 | {scenario} | seed {seed:02d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seed", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--event-horizon", type=int, default=160)
    args = parser.parse_args()

    output_root = args.output_root or (
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    for scenario in args.scenario:
        for seed in args.seed:
            reference = forward_fill(
                read_release_log(
                    find_strategy_dir(args.result_root, scenario, "cadence1", seed)
                ),
                args.event_horizon,
            )
            rows_by_strategy: dict[str, list[dict[str, object]]] = {}
            for strategy in STRATEGIES:
                strategy_rows = forward_fill(
                    read_release_log(
                        find_strategy_dir(args.result_root, scenario, strategy, seed)
                    ),
                    args.event_horizon,
                )
                rows_by_strategy[strategy] = difference_rows(
                    scenario, seed, strategy, strategy_rows, reference
                )

            scenario_dir = output_root / scenario
            for strategy, rows in rows_by_strategy.items():
                write_csv(scenario_dir / f"{scenario}_seed_{seed:02d}_{strategy}.csv", rows)
            plot_seed(
                scenario_dir / f"{scenario}_seed_{seed:02d}_event_difference.png",
                scenario,
                seed,
                rows_by_strategy,
            )
            print(f"[DONE] scenario={scenario} seed={seed:02d}")

    print(f"[OUTPUT] {output_root}")


if __name__ == "__main__":
    main()
