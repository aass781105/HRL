import copy
from typing import Callable

import numpy as np


VALID_HL_ENV_SCENARIOS = ("baseline", "custom", "burst_cluster", "bottleneck_order", "mixed3")


def resolve_hl_env_scenario(config, rng: np.random.Generator = None) -> str:
    resolved = str(getattr(config, "hl_env_scenario_resolved", "")).strip().lower()
    if resolved:
        return resolved
    scenario = str(getattr(config, "hl_env_scenario", "baseline")).strip().lower()
    if scenario == "mixed3":
        choices = ("baseline", "burst_cluster", "bottleneck_order")
        rng = rng or np.random.default_rng()
        scenario = str(rng.choice(choices))
    if scenario not in VALID_HL_ENV_SCENARIOS:
        raise ValueError(f"Unknown hl_env_scenario={scenario!r}; expected one of {VALID_HL_ENV_SCENARIOS}")
    return scenario


def scenario_config(config, scenario: str):
    cfg = copy.deepcopy(config)
    scenario = str(scenario).strip().lower()

    if scenario == "baseline":
        # Fixed reference environment. Keep baseline comparisons independent from
        # ad-hoc YAML environment tweaks; use hl_env_scenario=custom for tuning.
        setattr(cfg, "init_jobs", 50)
        setattr(cfg, "burst_size", 1)
        setattr(cfg, "arrival_mode", "uniform")
        setattr(cfg, "interarrival_uniform_low", 20.0)
        setattr(cfg, "interarrival_uniform_high", 45.0)
        setattr(cfg, "hl_bottleneck_order_prob", 0.0)
        setattr(cfg, "hl_bottleneck_order_machine_count", 1)
        setattr(cfg, "hl_due_date_urgent_prob", 0.3)
        setattr(cfg, "hl_due_date_k_urgent_low", 1.5)
        setattr(cfg, "hl_due_date_k_urgent_high", 3.0)
        setattr(cfg, "hl_due_date_k_normal_low", 7.0)
        setattr(cfg, "hl_due_date_k_normal_high", 10.0)
    elif scenario == "custom":
        # Fully YAML-driven environment for sensitivity tests.
        pass
    elif scenario == "burst_cluster":
        setattr(cfg, "init_jobs", 30)
        setattr(cfg, "burst_size", 1)
        setattr(cfg, "arrival_mode", "uniform")
        setattr(cfg, "interarrival_uniform_low", 45.0)
        setattr(cfg, "interarrival_uniform_high", 100.0)
        setattr(cfg, "hl_burst_size_mode", "inverse")
        setattr(cfg, "hl_burst_size_low", 1)
        setattr(cfg, "hl_burst_size_high", 5)
        setattr(cfg, "hl_bottleneck_order_prob", 0.0)
        setattr(cfg, "hl_bottleneck_order_machine_count", 1)
        setattr(cfg, "hl_bottleneck_exclude_urgent", True)
        setattr(cfg, "hl_bottleneck_group_sampling", "rolling_freq")
        setattr(cfg, "hl_due_date_urgent_prob", 0.3)
        setattr(cfg, "hl_due_date_k_urgent_low", 1.5)
        setattr(cfg, "hl_due_date_k_urgent_high", 3.0)
        setattr(cfg, "hl_due_date_k_normal_low", 7.0)
        setattr(cfg, "hl_due_date_k_normal_high", 10.0)
    elif scenario == "bottleneck_order":
        setattr(cfg, "init_jobs", 30)
        setattr(cfg, "burst_size", 1)
        setattr(cfg, "arrival_mode", "uniform")
        setattr(cfg, "interarrival_uniform_low", 20.0)
        setattr(cfg, "interarrival_uniform_high", 45.0)
        setattr(cfg, "hl_burst_size_mode", "fixed")
        setattr(cfg, "hl_bottleneck_order_prob", 0.5)
        setattr(cfg, "hl_bottleneck_order_machine_count", 1)
        setattr(cfg, "hl_bottleneck_exclude_urgent", True)
        setattr(cfg, "hl_bottleneck_group_sampling", "rolling_freq")
        setattr(cfg, "hl_due_date_urgent_prob", 0.3)
        setattr(cfg, "hl_due_date_k_urgent_low", 1.5)
        setattr(cfg, "hl_due_date_k_urgent_high", 3.0)
        setattr(cfg, "hl_due_date_k_normal_low", 6.0)
        setattr(cfg, "hl_due_date_k_normal_high", 10.0)
    else:
        raise ValueError(f"Unknown concrete scenario={scenario!r}")
    setattr(cfg, "hl_env_scenario_resolved", scenario)
    return cfg


def make_burst_sampler(config) -> Callable[[np.random.Generator], int]:
    mode = str(getattr(config, "hl_burst_size_mode", "fixed")).strip().lower()
    if mode in ("uniform", "inverse"):
        low = int(getattr(config, "hl_burst_size_low", 1))
        high = int(getattr(config, "hl_burst_size_high", max(low, int(getattr(config, "burst_size", 1)))))
        if low > high:
            low, high = high, low
        values = np.arange(low, high + 1, dtype=int)

        if mode == "inverse":
            weights = 1.0 / values.astype(float)
            probs = weights / np.sum(weights)

            def _sample(rng: np.random.Generator) -> int:
                return int(rng.choice(values, p=probs))

            return _sample

        def _sample(rng: np.random.Generator) -> int:
            return int(rng.integers(low, high + 1))

        return _sample

    fixed_k = int(getattr(config, "burst_size", 1))

    def _fixed(_rng: np.random.Generator) -> int:
        return int(fixed_k)

    return _fixed


def apply_bottleneck_orders(jobs, config, rng: np.random.Generator, n_machines: int):
    prob = float(getattr(config, "hl_bottleneck_order_prob", 0.0))
    if prob <= 0.0 or not jobs:
        return jobs

    exclude_urgent = bool(getattr(config, "hl_bottleneck_exclude_urgent", False))
    sampling_mode = str(getattr(config, "hl_bottleneck_group_sampling", "random")).strip().lower()
    avoid_prev = bool(getattr(config, "hl_bottleneck_avoid_prev_machines", False)) or sampling_mode == "avoid_prev"
    machine_count = max(1, min(int(getattr(config, "hl_bottleneck_order_machine_count", 2)), int(n_machines)))
    for job in jobs:
        is_urgent = bool(job.meta.get("is_urgent", False))
        if (exclude_urgent and is_urgent) or float(rng.random()) >= prob:
            job.meta["is_bottleneck_order"] = False
            continue
        machine_pool = np.arange(int(n_machines))
        if sampling_mode == "rolling_freq":
            counts = np.asarray(getattr(config, "_bottleneck_machine_counts", np.zeros(int(n_machines), dtype=int)), dtype=int)
            if counts.shape[0] != int(n_machines):
                counts = np.zeros(int(n_machines), dtype=int)
            tie_noise = rng.random(int(n_machines)) * 1e-6
            order = np.lexsort((tie_noise, counts))
            bottleneck_machines = np.asarray(order[:machine_count], dtype=int)
        else:
            prev = np.asarray(getattr(config, "_last_bottleneck_machines", []), dtype=int)
            if avoid_prev and prev.size > 0:
                candidate_pool = np.setdiff1d(machine_pool, prev, assume_unique=False)
                if candidate_pool.size >= machine_count:
                    machine_pool = candidate_pool
            bottleneck_machines = rng.choice(machine_pool, size=machine_count, replace=False)
        counts = np.asarray(getattr(config, "_bottleneck_machine_counts", np.zeros(int(n_machines), dtype=int)), dtype=int)
        if counts.shape[0] != int(n_machines):
            counts = np.zeros(int(n_machines), dtype=int)
        counts[np.asarray(bottleneck_machines, dtype=int)] += 1
        setattr(config, "_bottleneck_machine_counts", counts)
        setattr(config, "_last_bottleneck_machines", [int(m) for m in bottleneck_machines])
        for op in job.operations:
            row = np.asarray(op.time_row, dtype=float)
            valid = np.flatnonzero(row > 0)
            keep = [int(m) for m in bottleneck_machines if int(m) in set(valid.tolist())]
            if not keep and valid.size > 0:
                # Preserve feasibility even if the sampled bottleneck machine cannot process this op.
                keep = [int(valid[np.argmin(row[valid])])]
            new_row = np.zeros_like(row)
            for m in keep:
                new_row[m] = row[m]
            op.time_row = new_row.astype(float).tolist()
            vals = new_row[new_row > 0]
            op.avg_proc_time = float(np.mean(vals)) if vals.size else 0.0
        job.meta["is_bottleneck_order"] = True
        job.meta["bottleneck_machines"] = [int(m) for m in bottleneck_machines]
        vals = []
        for op in job.operations:
            row = np.asarray(op.time_row, dtype=float)
            if np.any(row > 0):
                vals.append(float(np.mean(row[row > 0])))
        job.meta["total_proc_time"] = float(sum(vals))
        job.meta["min_total_proc_time"] = float(sum(float(np.min(np.asarray(op.time_row, dtype=float)[np.asarray(op.time_row, dtype=float) > 0])) for op in job.operations if np.any(np.asarray(op.time_row, dtype=float) > 0)))
    return jobs
