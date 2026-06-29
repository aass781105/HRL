import numpy as np

HL_GATE_STATE_DIM = 25


def _buffer_machine_demand_stats(buffer_jobs, machine_free_time, t_now: float, n_machines: int):
    n_machines = int(n_machines)
    if n_machines <= 0:
        return 0.0, 0.0

    demand = np.zeros(n_machines, dtype=float)
    for job in buffer_jobs or []:
        for op in getattr(job, "operations", []) or []:
            time_row = getattr(op, "time_row", None)
            if time_row is None:
                machine_times = getattr(op, "machine_times", None)
                if machine_times:
                    row = np.zeros(n_machines, dtype=float)
                    for key, val in machine_times.items():
                        try:
                            idx = int(key)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= idx < n_machines and float(val) > 0:
                            row[idx] = float(val)
                    time_row = row
            if time_row is None:
                continue
            row = np.asarray(time_row, dtype=float)
            if row.size < n_machines:
                row = np.pad(row, (0, n_machines - row.size), mode="constant")
            elif row.size > n_machines:
                row = row[:n_machines]
            valid = row > 0
            compat_count = int(np.sum(valid))
            if compat_count <= 0:
                continue
            demand[valid] += row[valid] / float(compat_count)

    total_demand = float(np.sum(demand))
    if total_demand <= 1e-12:
        return 0.0, 0.0

    demand_norm = demand / total_demand
    demand_max_share = float(np.max(demand_norm))

    load = np.maximum(0.0, np.asarray(machine_free_time, dtype=float)[:n_machines] - float(t_now))
    if load.size < n_machines:
        load = np.pad(load, (0, n_machines - load.size), mode="constant")
    total_load = float(np.sum(load))
    if total_load <= 1e-12:
        load_overlap = 0.0
    else:
        load_norm = load / total_load
        load_overlap = float(np.sum(demand_norm * load_norm))

    return demand_max_share, load_overlap


def calculate_hl_gate_state(
    buffer_size: int,
    machine_free_time: np.ndarray,
    t_now: float,
    n_machines: int,
    obs_buffer_cap: int,
    time_scale: float,
    weighted_idle: float,
    unweighted_idle: float,
    buffer_stats: dict = None,
    wip_stats: dict = None,
    inter_arrival_scaled: float = 0.0,
    steps_since_last_release: int = 0,
    release_count_so_far: int = 0,
    decision_steps_elapsed: int = 0,
    is_last_step: bool = False,
    buffer_jobs=None,
) -> np.ndarray:
    """
    State vector for the high-level gate agent.
    """
    o0 = float(np.log1p(buffer_size))
    mft_abs = np.asarray(machine_free_time, dtype=float)
    rem = np.maximum(0.0, mft_abs - float(t_now))
    if rem.size > 0:
        avg_load, min_load, max_load, load_std = float(rem.mean()), float(rem.min()), float(rem.max()), float(np.std(rem))
    else:
        avg_load = min_load = max_load = load_std = 0.0

    o1, o2, o15, o10 = avg_load / time_scale, min_load / time_scale, (max_load - min_load) / time_scale, load_std / time_scale
    o3 = weighted_idle / time_scale
    o17 = unweighted_idle / time_scale

    buf_neg = buf_min = buf_avg = buf_std = 0.0
    buf_q25 = 0.0
    w_min = w_avg = w_rat = c_log = w_std = w_cnt = s_den = 0.0

    if buffer_stats:
        buf_neg = float(buffer_stats.get("buffer_neg_slack_ratio", 0.0))
        buf_min = float(buffer_stats.get("min_slack", 0.0)) / time_scale
        buf_avg = float(buffer_stats.get("avg_slack", 0.0)) / time_scale
        buf_std = float(buffer_stats.get("slack_std", 0.0)) / time_scale
        buf_q25 = float(buffer_stats.get("slack_q25", 0.0)) / time_scale

    if wip_stats:
        w_min = float(wip_stats.get("wip_min_slack", 0.0)) / time_scale
        w_avg = float(wip_stats.get("wip_avg_slack", 0.0)) / time_scale
        w_rat = float(wip_stats.get("wip_tardy_ratio", 0.0))
        p_td = float(wip_stats.get("planned_td", 0.0))
        total_rem_work = float(wip_stats.get("total_rem_work", 0.0))
        c_log_raw = p_td / (total_rem_work + 1.0)
        c_log = float(np.clip(c_log_raw, 0.0, 10.0))
        w_std = float(wip_stats.get("wip_slack_std", 0.0)) / time_scale
        w_cnt_raw = max(0.0, float(wip_stats.get("wip_count", 0.0)))
        w_cnt = float(np.log1p(w_cnt_raw))
        s_den = w_avg / (w_cnt_raw + 1.0)

    o18 = float(inter_arrival_scaled)
    o19 = float(np.log1p(max(0, int(steps_since_last_release))))
    decision_den = max(5.0, float(max(0, int(decision_steps_elapsed))))
    o20 = float(max(0, int(release_count_so_far)) / decision_den)
    o21, o22 = _buffer_machine_demand_stats(buffer_jobs, mft_abs, t_now, n_machines)
    o23 = float(1.0 if bool(is_last_step) else 0.0)

    return np.array(
        [o0, o1, o2, o3, buf_neg, buf_min, buf_avg, w_min, w_avg, w_rat, o10, c_log, buf_std, w_std, w_cnt, o15, s_den, o17, o18, o19, o20, o21, o22, buf_q25, o23],
        dtype=np.float32,
    )
