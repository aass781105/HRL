import argparse
import json
import os
import sys


def parse_export_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, default="")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


export_args = parse_export_args()

from dynamic_job_stream import dynamic_job_stream_to_dict, generate_dynamic_job_stream
from params import configs


def default_output_path() -> str:
    name = (
        f"dynamic_instance_seed{int(getattr(configs, 'event_seed', 42))}"
        f"_h{int(getattr(configs, 'event_horizon', 0))}"
        f"_k{int(getattr(configs, 'burst_size', 1))}.json"
    )
    return os.path.join("evaluation_results", name)


def main():
    output_path = export_args.output or default_output_path()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    stream = generate_dynamic_job_stream(
        configs,
        max_events=int(configs.event_horizon),
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        seed=int(getattr(configs, "event_seed", 42)),
    )
    payload = dynamic_job_stream_to_dict(stream, config=configs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved dynamic instance to: {output_path}")
    print(f"Jobs: {len(payload['jobs'])}, Events: {len(payload['events'])}, InitJobs: {len(payload['init_jobs'])}")


if __name__ == "__main__":
    main()
