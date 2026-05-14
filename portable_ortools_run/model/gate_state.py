import numpy as np


def calculate_gate_state(
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
    is_last_step: bool = False,
) -> np.ndarray:
    """
    22-dimensional state for the high-level gate agent.
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
    o20 = float(1.0 if bool(is_last_step) else 0.0)

    return np.array(
        [o0, o1, o2, o3, buf_neg, buf_min, buf_avg, w_min, w_avg, w_rat, o10, c_log, buf_std, w_std, w_cnt, o15, s_den, o17, o18, o19, buf_q25, o20],
        dtype=np.float32,
    )
