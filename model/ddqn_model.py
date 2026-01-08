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
    time_scale: float
) -> np.ndarray:
    """
    Centralized logic for calculating the 4-dimensional state for the DDQN Gate Agent.
    
    State:
    o0: Normalized buffer size
    o1: Normalized average remaining load
    o2: Normalized time to first idle
    o3: Normalized weighted idle time (reflecting load imbalance)
    """
    # 1) Buffer Normalization
    # If obs_buffer_cap is not provided (<=0), use a heuristic fallback
    # However, for consistency, the caller should ideally provide a valid cap.
    # Here we assume the caller handles the fallback logic for 'cap' if it depends on burst_K.
    # If passed 0, we might need a fallback, but let's assume the caller provides the effective cap.
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

    # 3) Weighted Idle (o3)
    # Horizon H = total_rem (Average Remaining Load)
    # We calculate weighted idle for machines that finish earlier than the average.
    # Weight w(t) = 1 - t/H
    H = total_rem
    if H > 1e-6:
        early_machines_mask = rem < H
        if np.any(early_machines_mask):
            rem_early = rem[early_machines_mask]
            r_ratios = rem_early / H
            # Integral of (1 - t/H) from rem[m] to H is (H/2) * (1 - rem[m]/H)^2
            w_idle_vals = (H / 2.0) * ((1.0 - r_ratios) ** 2)
            
            # Average over ALL machines to reflect global utilization gap
            avg_w_idle = float(w_idle_vals.sum()) / n_machines
            o3 = avg_w_idle / time_scale
        else:
            o3 = 0.0
    else:
        o3 = 0.0

    return np.array([o0, o1, o2, o3], dtype=np.float32)
