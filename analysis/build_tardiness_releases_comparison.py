"""Build a per-seed tardiness/release comparison CSV from evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_ROOT = Path(
    r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260819\urgent"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "tardiness_releases_comparison.csv"
STRATEGIES = ("ppo", "cad1", "cad5", "slack0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    matches = re.findall(r"seed0*(\d+)", str(path), flags=re.IGNORECASE)
    if not matches:
        raise ValueError(f"Cannot identify seed from: {path}")
    return int(matches[-1])


def read_strategy(root: Path, strategy: str) -> dict[int, tuple[float, float]]:
    strategy_dir = root / strategy
    files = sorted(strategy_dir.rglob("sample_runs_summary.csv"))
    if len(files) != 10:
        raise RuntimeError(
            f"{strategy}: expected 10 sample_runs_summary.csv files, found {len(files)}"
        )

    result: dict[int, tuple[float, float]] = {}
    for path in files:
        seed = seed_from_path(path)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        row = next((item for item in rows if str(item.get("run", "")) in {"1", "1.0"}), None)
        if row is None:
            raise RuntimeError(f"Missing run=1 in {path}")
        if seed in result:
            raise RuntimeError(f"Duplicate seed {seed} in {strategy}")
        result[seed] = (
            float(row["total_tardiness"]),
            float(row["release_count"]),
        )

    if set(result) != set(range(1, 11)):
        raise RuntimeError(f"{strategy}: expected seeds 1..10, found {sorted(result)}")
    return result


def cell(td: float, releases: float, *, mean: bool = False) -> str:
    release_format = ".1f" if mean else ".0f"
    return f"{td:.0f} / {releases:{release_format}}"


def build_csv(root: Path, output: Path) -> None:
    values = {strategy: read_strategy(root, strategy) for strategy in STRATEGIES}
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Seed", *STRATEGIES))
        for seed in range(1, 11):
            writer.writerow(
                [
                    f"seed{seed}",
                    *[
                        cell(values[strategy][seed][0], values[strategy][seed][1])
                        for strategy in STRATEGIES
                    ],
                ]
            )

        writer.writerow(
            [
                "mean",
                *[
                    cell(
                        sum(values[strategy][seed][0] for seed in range(1, 11)) / 10,
                        sum(values[strategy][seed][1] for seed in range(1, 11)) / 10,
                        mean=True,
                    )
                    for strategy in STRATEGIES
                ],
            ]
        )

    print(f"Wrote: {output}")


if __name__ == "__main__":
    args = parse_args()
    build_csv(args.root, args.output)
