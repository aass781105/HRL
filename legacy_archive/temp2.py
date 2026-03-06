# main.py
import os
import time
import copy
import numpy as np
from tqdm import tqdm
from typing import Optional

from params import configs
from common_utils import *                   # 需要 heuristic_select_action(...)
from global_env import (
    GlobalTimelineOrchestrator,
    OnTheFlySD2Generator,
    split_matrix_to_jobs,                    # [ADDED] t0 產初始工單要用
)
from data_utils import SD2_instance_generator

# 匯入畫圖工具  使用gantt
from gantt import plot_global_gantt, plot_batch_gantt


def binary_release_decider(buffer, orch) -> int:
    p = getattr(configs, "release_probability", None)
    if p is not None:
        return int(np.random.rand() < float(p))
    n = getattr(configs, "release_every_n_steps", None)
    if n is not None and int(n) > 1:
        return int(orch.step_idx % int(n) == 0)
    return 1


def run_dynamic_simulation(*,
                           steps: int,
                           dt: float,
                           heuristic: str,
                           lam: float,
                           release_K: Optional[int] = None,
                           plot_global_dir: Optional[str] = None,
                           plot_batch_dir: Optional[str] = None):

    # ---- 建立「到達生成器」 ----
    base_cfg = copy.deepcopy(configs)
    gen = OnTheFlySD2Generator(
        sd2_fn=SD2_instance_generator,
        base_config=base_cfg,
        n_machines=configs.n_m,
        lam=lam,
        lam_unit=getattr(configs, "arrival_rate_unit", "per_time"),
    )

    # ---- 建立全局 Orchestrator（H/R/B 版本）----
    orch = GlobalTimelineOrchestrator(
        n_machines=configs.n_m,
        job_generator=gen,
        select_from_buffer=lambda buf, o: list(range(len(buf))),  # 此版 _start_static_batch 會用 R∪B，不靠這個選擇器
        release_decider=binary_release_decider,
        t0=0.0,
    )

    # ---- t0 產生「初始工單」並直接放入 buffer ----
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
        gen._next_id = max(gen._next_id, max_existing + 1)  # [ADDED] 避免新到達 job_id 與既有重複

    # ---- 圖檔輸出資料夾 ----
    plot_global_dir = plot_global_dir or getattr(configs, "plot_global_dir", "plots/global")
    plot_batch_dir  = plot_batch_dir  or getattr(configs, "plot_batch_dir",  "plots/batch")
    os.makedirs(plot_global_dir, exist_ok=True)
    os.makedirs(plot_batch_dir,  exist_ok=True)

    # ---- 統計 ----
    stats = {
        "ticks": 0,
        "batches_started": 0,
        "batches_finalized": 0,
        "scheduled_ops": 0,
        "wall_time_sec": 0.0,
        "reschedules": 0,              # [KEPT] 若你仍要追蹤 reschedule 次數可保留
        "schedule_events": 0,          # [ADDED] 真正的「排程事件」次數（start + reschedule）
    }

    # 繪圖序號（初始=1，每次排程事件+1）
    plot_seq = 0

    # 內部：排完整批 + 畫圖
    # [CHANGED] 新增 initial_state 參數，把事件傳入的 state 交給 solve_current_batch_static
    def _run_batch_to_completion(env, *, reason: str, resched_tcut: Optional[float], initial_state):
        nonlocal stats, plot_seq

        # --- 一次排到底（用 orchestrator 的 solve_current_batch_static；內部會依 configs.use_ppo 決定 PPO / heuristic） ---
        # [CHANGED] 統一呼叫 solve_current_batch_static，並把 initial_state 與 heuristic 一併傳入
        batch_rows_full = orch.solve_current_batch_static(
            env,
            initial_state=initial_state,     # [ADDED]
            heuristic=heuristic              # [ADDED] 當 use_ppo=False 時才會用到
        )
        stats["scheduled_ops"] += len(batch_rows_full)

        # 完成本批（依 orchestrator 內部的 t_cut 來分 H/R）
        fin = orch.finalize(env)
        stats["batches_finalized"] += 1
        stats["schedule_events"]  += 1      # [ADDED] 每完成一批就代表一次「排程事件」

        # --- 全局圖：每次排程事件都畫（初始也算一次）---
        plot_seq += 1

        # 一律使用呼叫方傳入的 resched_tcut；初始傳 0.0；start/reschedule 事件 tick 會帶 t_cut
        tcut = float(resched_tcut or 0.0)
        tcut_tag = f"{int(round(tcut * 100)):08d}"

        # A：歷史（H；包含 s<t_cut 的整段 [s,e]）
        hist = [dict(r, phase="history") for r in orch.get_global_rows()]

        # C：R（這一批在 t_cut 之後的排程區段；一律畫到 e）——由 orchestrator 在 finalize 時分好
        plan_R = [dict(r, phase="newplan") for r in orch.get_R_rows()]

        # 合併與繪製（不會重覆畫 s<t_cut 的區段）
        global_rows_snapshot = hist + plan_R
        global_path = os.path.join(
            plot_global_dir, f"global_r{plot_seq:03d}_step{orch.step_idx:06d}_t{tcut_tag}.png"
        )
        plot_global_gantt(
            global_rows_snapshot,
            save_path=global_path,
            t_now=tcut,
            title=f"Global schedule @ t={tcut:.2f} (resched #{plot_seq})"
        )

        # 批次圖：重排時才畫（整批的靜態結果，畫到 e）
        if reason == "reschedule" and fin.get("rows"):
            batch_path = os.path.join(
                plot_batch_dir,
                f"batch_r{plot_seq:03d}_tcut{tcut_tag}_final_t{int(round(orch.t*100)):08d}.png"
            )
            plot_batch_gantt(
                fin["rows"],
                save_path=batch_path,
                t_now=tcut,
                title=f"Rescheduled batch result | cut@{tcut:.2f}"
            )

    # ---- t=0：若 buffer 有件 → 立刻開批（R∪B）→ 一次排到底 → 畫圖 ----
    if len(orch.buffer) > 0:
        ev0 = orch._start_static_batch(list(range(len(orch.buffer))))
        stats["batches_started"] += 1
        _run_batch_to_completion(
            ev0["env"],
            reason="start",
            resched_tcut=0.0,             # 初始固定畫 t=0 線（避免與 H 重疊）
            initial_state=ev0.get("state") # [ADDED]
        )

    # ---- 主回圈 ----
    t0_wall = time.time()
    for _ in tqdm(range(steps), desc="Sim ticks"):
        ev = orch.tick(dt=dt)
        stats["ticks"] += 1

        if ev.get("event") == "batch_ready":
            stats["batches_started"] += 1
            reason = ev.get("reason", "start")

            # [CHANGED] 仍保留 reschedules 統計，但你要的「排程次數」會在 _run_batch_to_completion 內以 schedule_events 統一加總
            if reason == "reschedule":
                stats["reschedules"] += 1

            _run_batch_to_completion(
                ev["env"],
                reason=reason,
                resched_tcut=ev.get("t_cut", 0.0),
                initial_state=ev.get("state")   # [ADDED]
            )

        else:
            # 保險：如果意外還有活批，直接排到底並畫圖
            if orch.current_env is not None:
                _run_batch_to_completion(
                    orch.current_env,
                    reason="start",
                    resched_tcut=0.0,
                    initial_state=None            # [ADDED] 非事件觸發，通常沒有 state；傳 None
                )

    stats["wall_time_sec"] = time.time() - t0_wall
    dynamic_makespan = float(np.max(orch.machine_free_time))
    return dynamic_makespan, stats


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))

    STEPS      = int(getattr(configs, "dyn_steps", 200))
    DT         = float(getattr(configs, "dt", 1.0))
    LAM        = float(getattr(configs, "arrival_lambda", 0.8))
    RELEASE_K  = getattr(configs, "dyn_release_K", None)

    # [CHANGED] 若 use_ppo=True，顯示 PPO；否則顯示派工法則
    if bool(getattr(configs, "use_ppo", False)):
        RULE_STR = "PPO"
        HEURISTIC = "N/A"  # 不使用
    else:
        if hasattr(configs, "test_method") and len(configs.test_method) > 0:
            HEURISTIC = configs.test_method[0]
        else:
            HEURISTIC = "SPT"
        RULE_STR = HEURISTIC

    if getattr(configs, "release_probability", None) is not None:
        release_mode = f"prob={configs.release_probability}"
    elif getattr(configs, "release_every_n_steps", None) not in (None, 1):
        release_mode = f"every {configs.release_every_n_steps} steps"
    else:
        release_mode = "every step"

    PLOT_GLOBAL_DIR = getattr(configs, "plot_global_dir", "plots/global")
    PLOT_BATCH_DIR  = getattr(configs, "plot_batch_dir",  "plots/batch")
    INIT_J = int(getattr(configs, "init_jobs", getattr(configs, "initial_jobs", 0)))
    RATE_UNIT = getattr(configs, "arrival_rate_unit", "per_time")

    print("-" * 25 + " Dynamic Orchestrated Scheduling " + "-" * 25)  # [CHANGED] 標題泛化
    print(f"Machines (M)        : {configs.n_m}")
    print(f"Rule                : {RULE_STR}")  # [CHANGED] 可能是 PPO 或 Heuristic 名稱
    print(f"Steps               : {STEPS} (dt={DT})")
    print(f"Arrival λ (Poisson) : {LAM} ({RATE_UNIT})")
    print(f"Release/Resched     : {release_mode}")
    print(f"Initial jobs @ t0   : {INIT_J}")
    print(f"Global plots dir    : {PLOT_GLOBAL_DIR}")
    print(f"Batch  plots dir    : {PLOT_BATCH_DIR}")

    mk, stats = run_dynamic_simulation(
        steps=STEPS,
        dt=DT,
        heuristic=HEURISTIC,                 # [KEPT] 當 use_ppo=False 時會在 solve_current_batch_static 用到
        lam=LAM,
        release_K=RELEASE_K,
        plot_global_dir=PLOT_GLOBAL_DIR,
        plot_batch_dir=PLOT_BATCH_DIR,
    )

    print("\n== Summary ==")
    print(f"Dynamic makespan        : {mk:.3f}")
    print(f"Sim ticks               : {stats['ticks']}")
    print(f"Batches started         : {stats['batches_started']}")
    print(f"Batches finalized       : {stats['batches_finalized']}")
    print(f"Reschedules (only)      : {stats['reschedules']}")       # [KEPT]
    print(f"Scheduling events (all) : {stats['schedule_events']}")   # [ADDED] 這就是你要的「排程次數」
    print(f"Total ops scheduled     : {stats['scheduled_ops']}")
    print(f"Wall-time (sec)         : {stats['wall_time_sec']:.2f}")


if __name__ == "__main__":
    main()
