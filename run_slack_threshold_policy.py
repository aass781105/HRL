import os

from params import configs
from main import run_event_driven_until_nevents


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))
    setattr(configs, "gate_policy", "slack_threshold")
    print("-" * 20 + " Dynamic FJSP Slack-Threshold Evaluation " + "-" * 20)
    print(f"[SlackThreshold] buffer_min_slack < {float(getattr(configs, 'buffer_slack_release_threshold', 0.0)):g} -> RELEASE")
    mk, stats = run_event_driven_until_nevents(
        max_events=int(configs.event_horizon),
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        plot_global_dir=getattr(configs, "plot_global_dir", "plots/global"),
    )
    print(f"\nMakespan: {mk:.3f}, Tardiness: {stats['total_tardiness']:.3f}, Releases: {stats['release_count']}")


if __name__ == "__main__":
    main()
