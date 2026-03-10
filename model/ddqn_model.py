import numpy as np
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 2, hidden: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(hidden))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def log_scale_reward(x: float) -> float:
    return float(np.sign(x) * np.log1p(np.abs(x)))

def calculate_ddqn_state(
    buffer_size: int,
    machine_free_time: np.ndarray,
    t_now: float,
    n_machines: int,
    obs_buffer_cap: int,
    time_scale: float,
    weighted_idle: float,
    unweighted_idle: float, # [NEW]
    buffer_stats: dict = None,
    wip_stats: dict = None
) -> np.ndarray:
    """
    18-dimensional ultimate state for the DDQN Gate Agent.
    """
    o0 = float(np.log1p(buffer_size))
    mft_abs = np.asarray(machine_free_time, dtype=float)
    rem = np.maximum(0.0, mft_abs - float(t_now))
    if rem.size > 0:
        avg_load, min_load, max_load, load_std = float(rem.mean()), float(rem.min()), float(rem.max()), float(np.std(rem))
    else:
        avg_load = min_load = max_load = load_std = 0.0

    o1, o2, o15, o10 = avg_load/time_scale, min_load/time_scale, (max_load-min_load)/time_scale, load_std/time_scale
    o3 = weighted_idle / time_scale
    o17 = unweighted_idle / time_scale # [NEW] Unweighted Idle

    buf_neg = buf_min = buf_avg = buf_std = 0.0
    w_min = w_avg = w_rat = c_log = w_std = w_cnt = s_den = 0.0
    
    if buffer_stats:
        buf_neg = float(buffer_stats.get("tardiness_ratio", 0.0))
        buf_min, buf_avg, buf_std = float(buffer_stats.get("min_slack", 0.0))/time_scale, float(buffer_stats.get("avg_slack", 0.0))/time_scale, float(buffer_stats.get("slack_std", 0.0))/time_scale
        
    if wip_stats:
        w_min, w_avg, w_rat = float(wip_stats.get("wip_min_slack", 0.0))/time_scale, float(wip_stats.get("wip_avg_slack", 0.0))/time_scale, float(wip_stats.get("wip_tardy_ratio", 0.0))
        p_td = float(wip_stats.get("planned_td", 0.0))
        total_rem_work = float(wip_stats.get("total_rem_work", 0.0))
        
        # [MOD] o11: Tardiness Density Ratio (Planned TD / Total Processing Work)
        # This provides a stable indicator of how 'overdue' the pending work is.
        c_log = p_td / (total_rem_work + 1.0)
        
        w_std, w_cnt = float(wip_stats.get("wip_slack_std", 0.0))/time_scale, float(wip_stats.get("wip_count", 0.0))
        s_den = w_avg / (w_cnt + 1.0)

    # Note: Re-ordered to match 18-dim requirements
    return np.array([o0, o1, o2, o3, buf_neg, buf_min, buf_avg, w_min, w_avg, w_rat, o10, c_log, buf_std, w_std, w_cnt, o15, s_den, o17], dtype=np.float32)
