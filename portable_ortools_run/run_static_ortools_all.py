import os

from benchmark_ortools import run_benchmark


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    run_benchmark(
        base_dir="or_instances_uniform_test_30_50",
        time_limit=7200.0,
        output_root="or_tools_solutions",
        make_gantt=True,
        log_solutions=True,
    )


if __name__ == "__main__":
    main()
