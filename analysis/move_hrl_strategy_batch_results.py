"""Safely move the completed HRL strategy batch to the presentation folder."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_ROOT = (PROJECT_ROOT / "plots" / "global").resolve()
MANIFEST_PATH = PROJECT_ROOT / "analysis_results" / "hrl_strategy_batch_manifest.csv"
COMPARISON_ROOT = PROJECT_ROOT / "analysis_results" / "comparison_stage"
EXTERNAL_ROOT = (
    Path(r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260729\結果")
)
SCENARIOS = ("baseline", "urgent")
STRATEGIES = ("ppo", "cad1", "cad5", "slack0")
SEEDS = tuple(range(1, 11))


def latest_manifest_rows() -> dict[tuple[str, str, int], dict[str, str]]:
    rows: dict[tuple[str, str, int], dict[str, str]] = {}
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[(row["scenario"], row["strategy"], int(row["seed"]))] = row
    return rows


def assert_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes intended root: {path}") from exc


def main() -> None:
    manifest = latest_manifest_rows()
    moves: list[tuple[Path, Path]] = []

    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                row = manifest.get((scenario, strategy, seed))
                if row is None or row["status"] not in {"ok", "already_moved"}:
                    raise RuntimeError(f"Missing successful manifest row: {scenario}/{strategy}/{seed}")
                if row["status"] == "already_moved":
                    continue

                source = (PROJECT_ROOT / row["output_dir"]).resolve()
                assert_inside(source, PLOT_ROOT, "Source")
                if not source.is_dir():
                    raise FileNotFoundError(source)
                if not (source / "sample_runs_summary.csv").exists():
                    raise RuntimeError(f"Missing summary CSV: {source}")

                destination_root = EXTERNAL_ROOT / scenario / strategy
                destination = destination_root / source.name
                if destination.exists():
                    raise FileExistsError(f"Destination already exists: {destination}")
                moves.append((source, destination))

    for source, destination in moves:
        destination_root = destination.parent.resolve()
        assert_inside(destination_root, EXTERNAL_ROOT.resolve(), "Destination")
        destination_root.mkdir(parents=True, exist_ok=True)

    for source, destination in moves:
        shutil.move(str(source), str(destination))

    for scenario in SCENARIOS:
        stage = COMPARISON_ROOT / scenario
        destination_root = EXTERNAL_ROOT / scenario
        for name in ("strategy_metrics_comparison.csv", "tardiness_releases_comparison.csv"):
            source = stage / name
            destination = destination_root / name
            if not source.exists():
                raise FileNotFoundError(source)
            if destination.exists():
                raise FileExistsError(f"Comparison destination already exists: {destination}")
            destination_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    print(f"Moved result folders: {len(moves)}")
    print(f"Copied comparison files: {len(SCENARIOS) * 2}")
    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            count = len([p for p in (EXTERNAL_ROOT / scenario / strategy).iterdir() if p.is_dir()])
            print(f"{scenario}/{strategy}: {count} folders")


if __name__ == "__main__":
    main()
