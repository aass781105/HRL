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

# 繪圖
from gantt import plot_global_gantt, plot_batch_gantt


def fixed_k_sampler(K: int):
    """[ADDED] 固定一次釋放 K 筆的 sampler。"""
    def _fn(rng: np.random.Generator) -> int:
        return int(K)
    return _fn



def run_event_driven_until_nevents(*,
                                max_events: int,               # [ADDED] 最大模擬時間 T_MAX
                                heuristic: str,
                                interarrival_mean: float,
                                burst_K: int = 1,
                                plot_global_dir: Optional[str] = None,
                                plot_batch_dir: Optional[str] = None):

    # ---- 建到達生成器（事件驅動） ----
    base_cfg = copy.deepcopy(configs)
    rng = np.random.default_rng(int(getattr(configs, "event_seed", 42)))
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

    # ---- t=0 初始工單（可選）----
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

        # [ADDED] 若一開始就有件，直接在 t=0 先排一次（符合你舊行為）
        fin0 = orch.event_release_and_reschedule(0.0, heuristic)
        # 可選：這裡不畫圖，統一由下方事件循環畫
        # --- 立刻畫圖（global + batch）---
        tcut = float(fin0.get("t_cut", 0.0))
        tcut_tag = f"{int(round(tcut * 100)):08d}"

        # Global（把 H 與本批一起畫）
        hist   = [dict(r, phase="history") for r in orch.get_global_rows()]
        plan_R = [dict(r, phase="newplan") for r in orch.get_R_rows()]   # 等同 fin0["rows"]，本批完整
        global_rows_snapshot = hist + plan_R

        plot_global_dir = plot_global_dir or getattr(configs, "plot_global_dir", "plots/global")
        os.makedirs(plot_global_dir, exist_ok=True)
        global_path = os.path.join(plot_global_dir, f"global_event0_t{tcut_tag}.png")
        plot_global_gantt(
            global_rows_snapshot,
            save_path=global_path,
            t_now=tcut,
            title=f"Global schedule @ t={tcut:.2f} (event #0)"
        )


    # ---- 圖檔輸出資料夾 ----  
    plot_global_dir = plot_global_dir or getattr(configs, "plot_global_dir", "plots/global")
    plot_batch_dir  = plot_batch_dir  or getattr(configs, "plot_batch_dir",  "plots/batch")
    os.makedirs(plot_global_dir, exist_ok=True)
    os.makedirs(plot_batch_dir,  exist_ok=True)

    # ---- 統計 ----
    # ---- stats 初始化 ----
    stats = {
        "arrive": 0,                 
        "arrived_jobs_total": 0,   
        "release": 0,               
        "scheduled_ops": 0,
        "wall_time_sec": 0.0,
        "event_horizon": int(max_events),
    }

    plot_seq = 0
    t_now = 0.0
    t_next = gen.sample_next_time(t_now)   # 第一個到達事件
    t0_wall = time.time()

    # [CHANGED] 以 arrive（到來次數）作為終止條件
    while stats["arrive"] < int(max_events):
        t_now = float(t_next)

        # 事件：生成 K 筆、進 buffer
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            orch.buffer.extend(new_jobs)
            stats["arrived_jobs_total"] += len(new_jobs)   
            stats["arrive"] += 1                              

        # [ADDED] 放行決策（先用固定 True；之後接 DDQN 的動作）
        decide_release = True  # TODO: 接 DDQN：True=Release, False=Hold

        if decide_release:
            # 事件處理：切 H/R、R∪B 開新批 → 一次排到底 → finalize
            fin = orch.event_release_and_reschedule(t_now, heuristic)
            if fin.get("event") == "batch_finalized":
                stats["scheduled_ops"] += len(fin.get("rows", []))
                stats["release"] += 1               # [ADDED] 成功 finalize 的「一次放行」

                # ---- 畫圖（t_now = tcut）----
                plot_seq += 1
                tcut = float(fin.get("t_cut", t_now))
                tcut_tag = f"{int(round(tcut * 100)):08d}"

                # A：歷史（H）
                hist = [dict(r, phase="history") for r in orch.get_global_rows()]
                # C：本批結果（新規劃）
                plan_R = [dict(r, phase="newplan") for r in orch.get_R_rows()]

                # 合併與繪製
                global_rows_snapshot = hist + plan_R
                global_path = os.path.join(
                    plot_global_dir, f"global_r{plot_seq:03d}_t{tcut_tag}.png"
                )
                plot_global_gantt(
                    global_rows_snapshot,
                    save_path=global_path,
                    t_now=tcut,
                    title=f"Global schedule @ t={tcut:.2f} (event #{plot_seq})"
                )

                # 批次圖（整批靜態結果）
                if fin.get("rows"):
                    batch_path = os.path.join(
                        plot_batch_dir,
                        f"batch_r{plot_seq:03d}_tcut{tcut_tag}_final_t{int(round(orch.t*100)):08d}.png"
                    )
                    plot_batch_gantt(
                        fin["rows"],
                        save_path=batch_path,
                        t_now=tcut,
                        title=f"Batch result | cut@{tcut:.2f}"
                    )
        else:
            pass

        t_next = gen.sample_next_time(t_now)   # 下一個事件時間

    stats["wall_time_sec"] = time.time() - t0_wall
    dynamic_makespan = float(np.max(orch.machine_free_time))
    return dynamic_makespan, stats



# === 更新 main()：用時間地平線取代事件次數 ===
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))

    E_MAX     = int(getattr(configs, "event_horizon", 200))          # [ADDED] 到達事件上限
    interarrival_mean = float(getattr(configs, "interarrival_mean", 0.100))
    BURST_K   = int(getattr(configs, "burst_size", 1))

    if hasattr(configs, "test_method") and len(configs.test_method) > 0:
        HEURISTIC = configs.test_method[0]
    else:
        HEURISTIC = "SPT"

    PLOT_GLOBAL_DIR = getattr(configs, "plot_global_dir", "plots/global")
    PLOT_BATCH_DIR  = getattr(configs, "plot_batch_dir",  "plots/batch")
    INIT_J = int(getattr(configs, "init_jobs", getattr(configs, "initial_jobs", 10)))

    print("-" * 20 + " Event-driven Orchestrated Scheduling (Event Horizon) " + "-" * 20)  # [CHANGED]
    print(f"Machines (M)        : {configs.n_m}")
    print(f"Rule / PPO          : {('PPO' if getattr(configs, 'use_ppo', False) else HEURISTIC)}")
    print(f"Event horizon (E)   : {E_MAX}")                                                # [ADDED]
    print(f"Exponential interarrival mean: {interarrival_mean}")                           # [CHANGED] 語意更清楚
    print(f"Burst size K/event  : {BURST_K}")
    print(f"Initial jobs @ t=0  : {INIT_J}")
    print(f"Global plots dir    : {PLOT_GLOBAL_DIR}")
    print(f"Batch  plots dir    : {PLOT_BATCH_DIR}")


    mk, stats = run_event_driven_until_nevents(                                         # [CHANGED] 呼叫事件數版本
        max_events=E_MAX,                                                               # [CHANGED]
        heuristic=HEURISTIC,
        interarrival_mean=interarrival_mean,
        burst_K=BURST_K,
        plot_global_dir=PLOT_GLOBAL_DIR,
        plot_batch_dir=PLOT_BATCH_DIR,
    )

    print("\n== Summary ==")
    print(f"Dynamic makespan     : {mk:.3f}")
    print(f"Arrive events        : {stats.get('arrive')}")                                # [ADDED]
    print(f"Release decisions    : {stats.get('release')}")                               # [ADDED]
    # print(f"Time horizon (T)     : {stats['horizon_time']}")                            # [REMOVED]
    print(f"Event horizon (E)    : {stats.get('event_horizon')}")                         # [ADDED]
    print(f"Total ops scheduled  : {stats['scheduled_ops']}")
    print(f"Wall-time (sec)      : {stats['wall_time_sec']:.2f}")




if __name__ == "__main__":
    main()
