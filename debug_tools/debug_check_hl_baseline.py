from params import configs
from hl_gate_env import HLGateEnv, compute_cadence_baseline_for_seed


def main():
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    seed = int(getattr(configs, "event_seed", 1))

    cached = env._run_cadence_baseline(seed)
    direct = compute_cadence_baseline_for_seed(
        seed,
        n_machines=int(configs.n_m),
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
        cadence=int(env.baseline_event_cadence),
    )

    obs, _ = env.reset(seed=seed, options={"needs_baseline": False})
    done = False
    info = {}
    steps = 0
    while not done:
        obs, _, terminated, truncated, info = env.step(1)
        done = bool(terminated or truncated)
        steps += 1

    final = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    print("seed", seed)
    print("decision_interval", getattr(configs, "hl_gate_decision_interval", None))
    print("baseline_cadence_decision_steps", env.baseline_cadence)
    print("baseline_cadence_events", env.baseline_event_cadence)
    print("cached_baseline", cached)
    print("direct_baseline", direct)
    print(
        "always_release_agent",
        {
            "td": float(final["tardiness"]),
            "mk": float(final["makespan"]),
            "release_count": int(info.get("release_count", 0)),
            "steps": int(steps),
        },
    )


if __name__ == "__main__":
    main()
