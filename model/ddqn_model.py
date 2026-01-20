import numpy as np
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

def calculate_ddqn_state(
    buffer_size: int,
    machine_free_time: np.ndarray,
    t_now: float,
    n_machines: int,
    obs_buffer_cap: int,
    time_scale: float,
    weighted_idle: float,  # [CHANGED] Passed from orchestrator
    buffer_stats: dict = None, # [ADDED]
    wip_stats: dict = None     # [ADDED]
) -> np.ndarray:
    """
    Centralized logic for calculating the 8-dimensional state for the DDQN Gate Agent.
    
    State:
    o0: Normalized buffer size
    o1: Normalized average remaining load
    o2: Normalized time to first idle
    o3: Normalized weighted idle time (reflecting load imbalance & fragmentation)
    o4: Buffer Tardiness Ratio (0~1)
    o5: Buffer Min Slack (Normalized)
    o6: Buffer Avg Slack (Normalized)
    o7: WIP Min Slack (Normalized) - Urgency of jobs already on the floor but not finished
    """
    # 1) Buffer Normalization
    cap = float(obs_buffer_cap) if obs_buffer_cap > 0 else 1.0 
    o0 = float(buffer_size) / cap

    # 2) Remaining Load Calculation
    mft_abs = np.asarray(machine_free_time, dtype=float)
    rem = np.maximum(0.0, mft_abs - float(t_now))

    if rem.size > 0:
        total_rem = float(rem.sum()) / n_machines
        first_idle_rem = float(rem.min())
    else:
        total_rem = 0.0
        first_idle_rem = 0.0

    o1 = total_rem / time_scale
    o2 = first_idle_rem / time_scale

    # 3) Weighted Idle (o3) - Pre-calculated by orchestrator
    o3 = weighted_idle / time_scale

    # 4) Buffer & WIP Urgency Features
    buf_tardiness_ratio = 0.0
    buf_min_slack = 0.0 
    buf_avg_slack = 0.0 
    wip_min_slack = 0.0 
    
    if buffer_stats:
        buf_tardiness_ratio = float(buffer_stats.get("tardiness_ratio", 0.0))
        # Normalize Slack by time_scale
        buf_min_slack = float(buffer_stats.get("min_slack", 0.0)) / time_scale
        buf_avg_slack = float(buffer_stats.get("avg_slack", 0.0)) / time_scale
        
    if wip_stats:
        wip_min_slack = float(wip_stats.get("wip_min_slack", 0.0)) / time_scale

    return np.array([o0, o1, o2, o3, 
                     buf_tardiness_ratio, buf_min_slack, buf_avg_slack, wip_min_slack], 
                    dtype=np.float32)
