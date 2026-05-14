import argparse
import csv
import json
import os
from typing import Dict, Tuple

import numpy as np

from data_utils import SD2_instance_generator, matrix_to_text
from params import configs


DUE_SETTINGS: Dict[str, Tuple[float, float]] = {
    "mixed": (-1.0, 1.0),
    "loose": (0.0, 1.0),
    "tight": (-1.0, 0.0),
}


def generate_range_due_dates(n_j: int, setting: str, rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    mean_pt = (float(configs.low) + float(configs.high)) / 2.0
    a = float(n_j) * mean_pt
    low_mul, high_mul = DUE_SETTINGS[setting]
    low = low_mul * a
    high = high_mul * a
    return rng.uniform(low, high, size=int(n_j)), low, high


def write_fjs(path: str, job_length, op_pt, op_per_mch) -> None:
    lines = matrix_to_text(job_length, op_pt, op_per_mch)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def generate_test_instances(
    *,
    output_dir: str,
    sizes,
    n_m: int,
    instances_per_combo: int,
    seed_base: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    for n_j in sizes:
        configs.n_j = int(n_j)
        configs.n_m = int(n_m)

        generated_instances = {}
        for idx in range(1, int(instances_per_combo) + 1):
            instance_seed = int(seed_base) + int(n_j) * 1000 + idx
            generated_instances[idx] = SD2_instance_generator(configs, seed=instance_seed, mode="uniform")

        for setting in DUE_SETTINGS:
            scale_name = f"{int(n_j)}x{int(n_m)}_{setting}"
            scale_dir = os.path.join(output_dir, scale_name)
            os.makedirs(scale_dir, exist_ok=True)

            for idx, (job_length, op_pt, op_per_mch) in generated_instances.items():
                instance_seed = int(seed_base) + int(n_j) * 1000 + idx
                due_seed = instance_seed
                due_rng = np.random.default_rng(due_seed)
                due_dates, range_low, range_high = generate_range_due_dates(int(n_j), setting, due_rng)

                instance_id = f"{idx:03d}"
                base_name = f"instance_{scale_name}_{instance_id}"
                fjs_path = os.path.join(scale_dir, f"{base_name}.fjs")
                json_path = os.path.join(scale_dir, f"{base_name}.json")

                write_fjs(fjs_path, job_length, op_pt, op_per_mch)
                payload = {
                    "due_dates": due_dates.tolist(),
                    "due_setting": setting,
                    "range_low": float(range_low),
                    "range_high": float(range_high),
                    "n_j": int(n_j),
                    "n_m": int(n_m),
                    "instance_seed": int(instance_seed),
                    "due_seed": int(due_seed),
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=4, ensure_ascii=False)

                manifest_rows.append(
                    {
                        "scale": scale_name,
                        "n_j": int(n_j),
                        "n_m": int(n_m),
                        "due_setting": setting,
                        "instance_id": instance_id,
                        "instance_seed": int(instance_seed),
                        "due_seed": int(due_seed),
                        "range_low": float(range_low),
                        "range_high": float(range_high),
                        "fjs_path": fjs_path,
                        "json_path": json_path,
                    }
                )

            print(f"Generated {instances_per_combo} instances in {scale_dir}")

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total pairs: {len(manifest_rows)}")


def main():
    parser = argparse.ArgumentParser(description="Generate 30/40/50 x 5 range-due test instances.")
    parser.add_argument("--output_dir", default="or_instances_uniform_test_30_50")
    parser.add_argument("--sizes", nargs="+", type=int, default=[30, 40, 50])
    parser.add_argument("--n_m", type=int, default=5)
    parser.add_argument("--instances_per_combo", type=int, default=3)
    parser.add_argument("--seed_base", type=int, default=9000)
    args = parser.parse_args()

    generate_test_instances(
        output_dir=args.output_dir,
        sizes=args.sizes,
        n_m=args.n_m,
        instances_per_combo=args.instances_per_combo,
        seed_base=args.seed_base,
    )


if __name__ == "__main__":
    main()
