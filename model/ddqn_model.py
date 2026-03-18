import numpy as np
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 2, hidden: int = 128, num_layers: int = 3, dueling: bool = True):
        super().__init__()
        self.dueling = bool(dueling)
        trunk_layers = []
        last_dim = obs_dim
        for _ in range(max(0, num_layers - 1)):
            trunk_layers.append(nn.Linear(last_dim, hidden))
            trunk_layers.append(nn.ReLU(inplace=True))
            trunk_layers.append(nn.LayerNorm(hidden))
            last_dim = hidden

        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        if self.dueling:
            head_hidden = max(32, hidden // 2)
            head_hidden2 = max(32, head_hidden // 2)
            self.value_head = nn.Sequential(
                nn.Linear(last_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, head_hidden2),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden2, 1),
            )
            self.adv_head = nn.Sequential(
                nn.Linear(last_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, head_hidden2),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden2, n_actions),
            )
            self.q_head = None
        else:
            self.q_head = nn.Linear(last_dim, n_actions)
            self.value_head = None
            self.adv_head = None

    def forward(self, x):
        h = self.trunk(x)
        if self.dueling:
            v = self.value_head(h)
            a = self.adv_head(h)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        return self.q_head(h)

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
        # Clip only (no additional scaling) to avoid rare extreme spikes.
        c_log_raw = p_td / (total_rem_work + 1.0)
        c_log = float(np.clip(c_log_raw, 0.0, 10.0))
        
        w_std, w_cnt = float(wip_stats.get("wip_slack_std", 0.0))/time_scale, float(wip_stats.get("wip_count", 0.0))
        s_den = w_avg / (w_cnt + 1.0)

    # Note: Re-ordered to match 18-dim requirements
    return np.array([o0, o1, o2, o3, buf_neg, buf_min, buf_avg, w_min, w_avg, w_rat, o10, c_log, buf_std, w_std, w_cnt, o15, s_den, o17], dtype=np.float32)
