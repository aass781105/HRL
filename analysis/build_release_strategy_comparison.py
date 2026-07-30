"""Build comparison CSVs for the baseline and urgent strategy batches."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "analysis_results" / "hrl_strategy_batch_manifest.csv"
EXTERNAL_ROOT = (
    Path(r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260729\結果")
)
STAGE_ROOT = PROJECT_ROOT / "analysis_results" / "comparison_stage"
SCENARIOS = ("baseline", "urgent")
STRATEGIES = ("ppo", "cad1", "cad5", "slack0")
SEEDS = tuple(range(1, 11))


def latest_manifest_rows() -> dict[tuple[str, str, int], dict[str, str]]:
    rows: dict[tuple[str, str, int], dict[str, str]] = {}
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[(row["scenario"], row["strategy"], int(row["seed"]))] = row
    return rows


def summary_path(
    scenario: str,
    strategy: str,
    seed: int,
    manifest: dict[tuple[str, str, int], dict[str, str]],
) -> Path:
    row = manifest[(scenario, strategy, seed)]
    if row["status"] == "already_moved":
        root = EXTERNAL_ROOT / scenario / strategy
        candidates = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path / "sample_runs_summary.csv").exists()
            and (
                f"seed{seed:03d}" in path.name
                or f"seed{seed:02d}" in path.name
            )
        )
        if not candidates:
            raise FileNotFoundError(
                f"Missing external result for {scenario}/{strategy}/seed{seed}"
            )
        return candidates[-1] / "sample_runs_summary.csv"

    path = PROJECT_ROOT / row["output_dir"] / "sample_runs_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_metrics(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    return {
        "mk": float(row["makespan"]),
        "td": float(row["total_tardiness"]),
        "obj": float(row["obj"]),
        "rel": float(row["release_count"]),
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def write_comparisons(
    scenario: str,
    metrics: dict[tuple[str, int], dict[str, float]],
) -> None:
    output_dir = STAGE_ROOT / scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    wide_fields = ["Seed"]
    for strategy in STRATEGIES:
        wide_fields.extend(
            [
                f"{strategy}_MK",
                f"{strategy}_TD",
                f"{strategy}_Obj",
                f"{strategy}_Rel",
            ]
        )
    wide_rows: list[dict[str, str]] = []
    for seed in SEEDS:
        row = {"Seed": f"seed{seed}"}
        for strategy in STRATEGIES:
            values = metrics[(strategy, seed)]
            row.update(
                {
                    f"{strategy}_MK": fmt(values["mk"]),
                    f"{strategy}_TD": fmt(values["td"]),
                    f"{strategy}_Obj": fmt(values["obj"]),
                    f"{strategy}_Rel": fmt(values["rel"], 0),
                }
            )
        wide_rows.append(row)

    for label, aggregate in (("mean", statistics.mean), ("std", statistics.stdev)):
        row = {"Seed": label}
        for strategy in STRATEGIES:
            for metric in ("mk", "td", "obj", "rel"):
                values = [metrics[(strategy, seed)][metric] for seed in SEEDS]
                column = {
                    "mk": "MK",
                    "td": "TD",
                    "obj": "Obj",
                    "rel": "Rel",
                }[metric]
                row[f"{strategy}_{column}"] = fmt(
                    aggregate(values), 4 if label == "std" else 3
                )
        wide_rows.append(row)

    wide_path = output_dir / "strategy_metrics_comparison.csv"
    with wide_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)

    legacy_fields = ["Seed", *STRATEGIES]
    legacy_rows: list[dict[str, str]] = []
    for seed in SEEDS:
        row = {"Seed": f"seed{seed}"}
        for strategy in STRATEGIES:
            values = metrics[(strategy, seed)]
            row[strategy] = f"{values['td']:.0f} / {values['rel']:.0f}"
        legacy_rows.append(row)
    for label, aggregate in (("mean", statistics.mean), ("std", statistics.stdev)):
        row = {"Seed": label}
        for strategy in STRATEGIES:
            td_values = [metrics[(strategy, seed)]["td"] for seed in SEEDS]
            rel_values = [metrics[(strategy, seed)]["rel"] for seed in SEEDS]
            rel_digits = 1 if label == "mean" else 4
            row[strategy] = (
                f"{aggregate(td_values):.0f} / "
                f"{aggregate(rel_values):.{rel_digits}f}"
            )
        legacy_rows.append(row)

    legacy_path = output_dir / "tardiness_releases_comparison.csv"
    with legacy_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerows(legacy_rows)

    print(f"[COMPARE] {scenario}: {wide_path}")
    print(f"[COMPARE] {scenario}: {legacy_path}")


def main() -> None:
    manifest = latest_manifest_rows()
    for scenario in SCENARIOS:
        metrics: dict[tuple[str, int], dict[str, float]] = {}
        for strategy in STRATEGIES:
            for seed in SEEDS:
                metrics[(strategy, seed)] = read_metrics(
                    summary_path(scenario, strategy, seed, manifest)
                )
        write_comparisons(scenario, metrics)


if __name__ == "__main__":
    main()
