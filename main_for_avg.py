import os
import time
import copy
import numpy as np
from tqdm import tqdm
from typing import Optional

from params import configs
from common_utils import *   # heuristic_select_action(...)
from global_env import (
    GlobalTimelineOrchestrator,
    EventBurstGenerator,     # [ADDED] 事件驅動生成器
    split_matrix_to_jobs,
)
from data_utils import SD2_instance_generator

# --- Minimal reproducible seed ---
SEED = int(getattr(configs, "event_seed", 42))
np.random.seed(SEED)
import random
random.seed(SEED)

# ======================= [ADDED] DDQN 推論用最小 QNetwork =======================
import torch
import torch.nn as nn

class _QNet(nn.Module):  # [ADDED]
    def __init__(self, obs_dim: int = 2, hidden: int = 128, n_actions: int = 2):
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

# -----------------------------------------------------------------------------

def fixed_k_sampler(K: int):
    """[ADDED] 固定一次釋放 K 筆的 sampler。"""
    def _fn(rng: np.random.Generator) -> int:
        return int(K)
    return _fn


# ======================= [ADDED] Gate 觀測量（與訓練環境一致） =======================
def _gate_obs(orch: GlobalTimelineOrchestrator,
              n_machines: int,
              t_now: float,
              burst_K: int,
              interarrival_mean: float,
              buf_cap_cfg: int = 0,
              time_scale_cfg: float = 0.0) -> np.ndarray:
    """
    o0：buffer 大小正規化（分母：gate_obs_buffer_cap 或自動 = burst_K*3）
    o1：機台剩餘忙碌時間的方差（var(rem)）除以 time_scale^2
    """
    scale = float(getattr(configs, "norm_scale", 100.0))
    # buffer
    buf_size = len(orch.buffer)
    cap = int(buf_cap_cfg) if int(buf_cap_cfg) > 0 else max(1, int(burst_K) * 3)
    o0 = float(buf_size) / float(cap)

    # time variance
    mft_abs = np.asarray(orch.machine_free_time, dtype=float)
    rem = np.maximum(0.0, mft_abs - float(t_now))
    if rem.size > 0:
        # 總剩餘負載 S = Σ R_k
        total_rem = float(rem.sum()) /n_machines

        # 離第一台 idle 還有多久 L_min = min R_k
        #   - 若有機器已經 idle，對應 rem=0 => L_min = 0
        #   - 若所有機器都還在忙，L_min > 0
        first_idle_rem = float(rem.min())
    else:
        total_rem = 0.0
        first_idle_rem = 0.0

     # 3) 正規化
    o0 = float(buf_size) / cap          # buffer 大小 -> [0,1] 之類
    o1 = float(total_rem) /scale          # S_norm：整體未來負載
    o2 = float(first_idle_rem) / scale    # L_min_norm：離第一台 idle 還有多久

    # 4) high-level state = [ n_buffer, S_norm, L_min_norm ]
    return np.array([o0, o1, o2], dtype=np.float32)


def run_event_driven_until_nevents(*,
                                   max_events: int,
                                   heuristic: str,
                                   interarrival_mean: float,
                                   burst_K: int = 1,
                                   seed_offset: int = 0):
    """移除繪圖版本；新增 seed_offset 讓多次執行不會完全相同。"""

    # ---- 建到達生成器（事件驅動） ----
    base_cfg = copy.deepcopy(configs)
    base_seed = int(getattr(configs, "event_seed", 42))
    rng = np.random.default_rng(base_seed + int(seed_offset))
    gen = EventBurstGenerator(
        sd2_fn=SD2_instance_generator,
        base_config=base_cfg,
        n_machines=configs.n_m,
        interarrival_mean=interarrival_mean,
        k_sampler=fixed_k_sampler(int(burst_K)),
        rng=rng,
    )

    # ---- 建 orchestrator（H/R/B）----
    orch = GlobalTimelineOrchestrator(
        n_machines=configs.n_m,
        job_generator=gen,
        select_from_buffer=lambda buf, o: list(range(len(buf))),  # 不靠 selector
        release_decider=None,
        t0=0.0,
    )

    # ======================= [ADDED] Gate 策略：DDQN / 時間門檻 =======================
    gate_policy = str(getattr(configs, "gate_policy", "always")).lower()
    gate_time_threshold = float(getattr(configs, "gate_time_threshold", 50.0))  # 事件間隔累加門檻
    gate_obs_buffer_cap = int(getattr(configs, "gate_obs_buffer_cap", 0))
    gate_time_scale = float(getattr(configs, "gate_time_scale", 0.0))

    # DDQN 推論初始化（僅當 gate_policy='ddqn'）
    ddqn_model = None
    ddqn_device = torch.device(getattr(configs, "device", "cpu"))
    if gate_policy == "ddqn":  # [ADDED]
        ddqn_model_path = str(getattr(configs, "ddqn_model_path", "ddqn_ckpt/ddqn_gate_pure.pth"))
        ddqn_model = _QNet(obs_dim=3, hidden=128, n_actions=2).to(ddqn_device)
        try:
            sd = torch.load(ddqn_model_path, map_location=ddqn_device)
            ddqn_model.load_state_dict(sd)
            ddqn_model.eval()
            print(f"[DDQN] Loaded weights from: {ddqn_model_path}")
        except Exception as e:
            print(f"[WARN] Failed to load DDQN weights: {e}")
            print("       Fallback to 'always' release policy.")
            ddqn_model = None
            gate_policy = "always"
    # -------------------------------------------------------------------------

    # ---- t=0 初始工單（可選；無繪圖）----
    INIT_J = int(getattr(configs, "init_jobs", getattr(configs, "initial_jobs", 0)))
    if INIT_J > 0:
        init_cfg = copy.deepcopy(configs)
        old_nj = getattr(init_cfg, "n_j", None)
        try:
            setattr(init_cfg, "n_j", INIT_J)
            job_length, op_pt, _ = SD2_instance_generator(init_cfg)
        finally:
            if old_nj is not None:
                setattr(init_cfg, "n_j", old_nj)
        initial_jobs = split_matrix_to_jobs(job_length, op_pt, base_job_id=0, t_arrive=0.0)
        orch.buffer.extend(initial_jobs)
        max_existing = max((j.job_id for j in orch.buffer), default=-1)
        gen.bump_next_id(max_existing + 1)  # 對齊 next_id
        # 一開始先排一次（符合你舊行為）
        orch.event_release_and_reschedule(0.0, heuristic)

    # ---- 統計 ----
    stats = {
        "arrive": 0,
        "arrived_jobs_total": 0,
        "release": 0,
        "scheduled_ops": 0,
        "wall_time_sec": 0.0,
        "event_horizon": int(max_events),
    }

    t_prev = 0.0                           # [ADDED] 前一個事件時間（用來算間隔）
    t_now = 0.0
    t_next = gen.sample_next_time(t_now)   # 第一個到達事件
    t0_wall = time.time()

    # [ADDED] 針對 gate_policy='time' 累積時間的計數器
    acc_since_release = 0.0

    while stats["arrive"] < int(max_events):
        t_now = float(t_next)
        dt_interval = float(t_now - t_prev)   # 這次與上次事件的間隔
        t_prev = t_now

        # 事件：生成 K 筆、進 buffer
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            orch.buffer.extend(new_jobs)
            stats["arrived_jobs_total"] += len(new_jobs)
            stats["arrive"] += 1

        # ===================== 放行決策（多策略） =====================
        if gate_policy == "ddqn" and ddqn_model is not None:
            # 用與訓練一致的 2D 狀態
            obs = _gate_obs(
                orch=orch,
                n_machines=configs.n_m,
                t_now=t_now,
                burst_K=burst_K,
                interarrival_mean=interarrival_mean,
                buf_cap_cfg=gate_obs_buffer_cap,
                time_scale_cfg=gate_time_scale,
            )
            with torch.no_grad():
                qv = ddqn_model(torch.from_numpy(obs).float().unsqueeze(0).to(ddqn_device))
                act = int(torch.argmax(qv, dim=1).item())  # 0=HOLD, 1=RELEASE
            if stats["arrive"] % 10 == 0:
                mft = np.asarray(orch.machine_free_time, dtype=float)
                rem = np.maximum(0.0, mft - float(t_now))
                # print(f"[DDQN] t={t_now:.2f}  buf={len(orch.buffer)}  "
                #       f"var(rem)={float(np.var(rem)):.4f}  obs={obs.tolist()}  "
                #       f"Q={qv.cpu().numpy().round(4).tolist()}  act={act}")
            decide_release = (act == 1)


        elif gate_policy == "time":
            # 累加事件間隔，達門檻才 release
            acc_since_release += dt_interval
            if acc_since_release >= gate_time_threshold:
                decide_release = True
                acc_since_release = 0.0  # 釋放後歸零
            else:
                decide_release = False

        else:
            # 'always' 或不認得的策略 → 一律放行
            decide_release = True
        # =============================================================

        if decide_release:
            fin = orch.event_release_and_reschedule(t_now, heuristic)
            if fin.get("event") == "batch_finalized":
                stats["scheduled_ops"] += len(fin.get("rows", []))
                stats["release"] += 1
        # 下一個事件時間
        t_next = gen.sample_next_time(t_now)

    # ===== 最後一個到達事件後，若 buffer 尚有工單 → 強制釋放全排（Flush） =====
    if len(orch.buffer) > 0:
        print(f"[FLUSH] last arrival reached; buffer still has {len(orch.buffer)} jobs → force release until empty.")
        max_flush_rounds = 16
        flush_round = 0
        while len(orch.buffer) > 0 and flush_round < max_flush_rounds:
            flush_round += 1
            fin = orch.event_release_and_reschedule(t_now, heuristic)
            if fin.get("event") == "batch_finalized":
                stats["scheduled_ops"] += len(fin.get("rows", []))
                stats["release"] += 1
            else:
                print("[FLUSH] finalize did not return 'batch_finalized' — stop flushing.")
                break

        if len(orch.buffer) > 0:
            print(f"[FLUSH][WARN] after {flush_round} rounds, buffer still has {len(orch.buffer)} jobs.")

    # ===== [END FLUSH] =====

    stats["wall_time_sec"] = time.time() - t0_wall
    dynamic_makespan = float(np.max(orch.machine_free_time))
    return dynamic_makespan, stats


# === 更新 main()：用事件數上限（event_horizon），連跑 5 次並輸出平均 ===
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))

    E_MAX     = int(getattr(configs, "event_horizon", 200))          # 到達事件上限
    interarrival_mean = float(getattr(configs, "interarrival_mean", 0.100))
    BURST_K   = int(getattr(configs, "burst_size", 1))

    if hasattr(configs, "test_method") and len(configs.test_method) > 0:
        HEURISTIC = configs.test_method[0]
    else:
        HEURISTIC = "SPT"

    INIT_J = int(getattr(configs, "init_jobs", getattr(configs, "initial_jobs", 10)))

    print("-" * 20 + " Event-driven Orchestrated Scheduling (Event Horizon) " + "-" * 20)
    print(f"Machines (M)        : {configs.n_m}")
    print(f"Rule / PPO          : {('PPO' if getattr(configs, 'use_ppo', False) else HEURISTIC)}")
    print(f"Event horizon (E)   : {E_MAX}")
    print(f"Exponential interarrival mean: {interarrival_mean}")
    print(f"Burst size K/event  : {BURST_K}")
    print(f"Initial jobs @ t=0  : {INIT_J}")
    print(f"Gate policy         : {getattr(configs, 'gate_policy', 'always')}")

    runs = 30
    mk_list = []
    sum_arrive = 0
    sum_release = 0
    sum_eh = 0
    sum_ops = 0
    sum_wall = 0.0

    for i in range(runs):
        print(f"\n===== Run {i+1}/{runs} =====")
        # 想讓 5 次完全相同 → 把 seed_offset=i 改成 0
        mk, stats = run_event_driven_until_nevents(
            max_events=E_MAX,
            heuristic=HEURISTIC,
            interarrival_mean=interarrival_mean,
            burst_K=BURST_K,
            seed_offset=i
        )
        mk_list.append(mk)
        sum_arrive  += stats.get('arrive', 0)
        sum_release += stats.get('release', 0)
        sum_eh      += stats.get('event_horizon', 0)
        sum_ops     += stats.get('scheduled_ops', 0)
        sum_wall    += stats.get('wall_time_sec', 0.0)

        # 單次 Summary
        print("\n== Summary ==")
        print(f"Dynamic makespan     : {mk:.3f}")
        print(f"Arrive events        : {stats.get('arrive')}")
        print(f"Release decisions    : {stats.get('release')}")
        print(f"Event horizon (E)    : {stats.get('event_horizon')}")
        print(f"Total ops scheduled  : {stats['scheduled_ops']}")
        print(f"Wall-time (sec)      : {stats['wall_time_sec']:.2f}")

    # 平均
    avg_mk   = float(np.mean(mk_list)) if mk_list else 0.0
    std_mk = float(np.std(mk_list, ddof=1)) if len(mk_list) > 1 else 0.0
    max_mk = float(np.max(mk_list)) if len(mk_list) > 1 else 0.0
    min_mk = float(np.min(mk_list)) if len(mk_list) > 1 else 0.0
    avg_arr  = sum_arrive / runs
    avg_rel  = sum_release / runs
    avg_eh   = sum_eh / runs
    avg_ops  = sum_ops / runs
    avg_wall = sum_wall / runs

    print(f"\n===== Average over {runs} runs =====")
    print(f"Dynamic makespan (avg) : {avg_mk:.3f}")
    print(f"Dynamic makespan (std) : {std_mk:.3f}")
    print(f"Dynamic makespan (max) : {max_mk:.3f}")
    print(f"Dynamic makespan (min) : {min_mk:.3f}")
    print(f"Arrive events (avg)    : {avg_arr:.2f}")
    print(f"Release decisions (avg): {avg_rel:.2f}")
    print(f"Event horizon (avg)    : {avg_eh:.2f}")
    print(f"Total ops scheduled(avg): {avg_ops:.2f}")
    print(f"Wall-time (sec) (avg)  : {avg_wall:.2f}")


if __name__ == "__main__":
    main()
