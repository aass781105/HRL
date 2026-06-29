from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ll_fjsp_env import LLFJSPEnv
from gantt import plot_global_gantt
from model.ll_ppo import ll_ppo_initialize
from params import configs


CASE_PATH = Path("debug/dynamic_r60_candidate_features/r060_event118_t03691_fixed_ll_case.npz")
OUTPUT_DIR = Path("debug/dynamic_r60_fixed_ll_eval")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    try:
        f = path.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        f = alt.open("w", newline="", encoding="utf-8-sig")
        print(f"[FIXED-LL] target locked; wrote alternate CSV {alt}")
    with f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_gantt(path: Path, schedule_rows: List[Dict]) -> None:
    if not schedule_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = [
        {
            "job": int(row["job_id"]),
            "op": int(row["op"]),
            "machine": int(row["machine"]),
            "start": float(row["start_rel"]),
            "end": float(row["end_rel"]),
            "duration": float(row["duration"]),
            "due_date": float(row["due_rel"]),
            "phase": "newplan",
        }
        for row in schedule_rows
    ]
    try:
        plot_global_gantt(
            plot_rows,
            str(path),
            t_now=0.0,
            title="Fixed r60 lower-PPO rollout (relative time)",
        )
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        plot_global_gantt(
            plot_rows,
            str(alt),
            t_now=0.0,
            title="Fixed r60 lower-PPO rollout (relative time)",
        )
        print(f"[FIXED-LL] target locked; wrote alternate Gantt {alt}")


def build_env(case):
    job_lengths = case["job_lengths"].astype(np.int64)
    op_pt = case["op_pt"].astype(np.float64)
    due_rel = case["due_rel"].astype(np.float64)
    ready_rel = case["ready_rel"].astype(np.float64)
    machine_free_rel = case["machine_free_rel"].astype(np.float64)
    pt_scale = (float(configs.low) + float(configs.high)) / 2.0

    env = LLFJSPEnv(n_j=int(len(job_lengths)), n_m=int(op_pt.shape[1]))
    state = env.set_initial_data(
        [job_lengths],
        [op_pt],
        due_date_list=[due_rel],
        true_due_date_list=[due_rel],
    )
    env.true_mch_free_time[:, :] = machine_free_rel
    env.mch_free_time[:, :] = machine_free_rel / max(pt_scale, 1e-8)
    env.true_candidate_free_time[:, :] = ready_rel
    env.candidate_free_time[:, :] = ready_rel / max(pt_scale, 1e-8)
    state = env.rebuild_state_from_current()
    return env, state


def load_policy():
    model_path = str(getattr(configs, "ll_ppo_model_path", "") or "").strip()
    if not model_path:
        raise ValueError("Missing ll_ppo_model_path. Pass --config with the desired lower PPO model.")
    ppo = ll_ppo_initialize()
    ppo.policy.load_state_dict(torch.load(model_path, map_location=getattr(configs, "device", "cpu"), weights_only=True))
    ppo.policy.eval()
    return ppo


def policy_probs(ppo, state):
    with torch.inference_mode():
        pi = ppo.policy.policy_only(
            fea_j=state.fea_j_tensor,
            op_mask=state.op_mask_tensor,
            candidate=state.candidate_tensor,
            fea_m=state.fea_m_tensor,
            mch_mask=state.mch_mask_tensor,
            comp_idx=state.comp_idx_tensor,
            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
            fea_pairs=state.fea_pairs_tensor,
        )
    return pi.detach().cpu().numpy()[0]


def dump_first_step_ranking(env, probs, case) -> List[Dict]:
    order = np.argsort(-probs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    job_ids = case["job_ids"].astype(np.int64)
    op_offsets = case["op_offsets"].astype(np.int64)
    due_rel = case["due_rel"].astype(np.float64)
    ready_rel = case["ready_rel"].astype(np.float64)
    rows = []
    for j_idx, job_id in enumerate(job_ids):
        cand_op = int(env.candidate[0, j_idx])
        local_op = int(cand_op - env.job_first_op_id[0, j_idx])
        global_op = int(op_offsets[j_idx] + local_op)
        raw_op = env.raw_fea_j[0, cand_op, :]
        for m in range(env.number_of_machines):
            action = int(j_idx * env.number_of_machines + m)
            rows.append(
                {
                    "policy_rank": int(ranks[action]),
                    "policy_prob": float(probs[action]),
                    "action": action,
                    "job_id": int(job_id),
                    "batch_job_index": int(j_idx),
                    "machine": int(m),
                    "candidate_global_op": global_op,
                    "candidate_local_op": local_op,
                    "true_candidate_pt": float(env.true_op_pt[0, cand_op, m]),
                    "ready_rel": float(ready_rel[j_idx]),
                    "due_rel": float(due_rel[j_idx]),
                    "raw_feat_slack": float(raw_op[11]),
                    "raw_feat_is_tardy": float(raw_op[13]),
                    "raw_feat_job_current_tardiness": float(raw_op[14]),
                    "raw_feat_overdue_weight": float(raw_op[19]) if raw_op.shape[0] > 19 else "",
                    "pair_est_lateness_log": float(env.fea_pairs[0, j_idx, m, 7]),
                    "dynamic_pair_mask_raw": int(env.dynamic_pair_mask[0, j_idx, m]),
                }
            )
    return sorted(rows, key=lambda r: r["policy_rank"])


def run_rollout(env, state, ppo, case):
    batch_time = float(case["batch_time"])
    job_ids = case["job_ids"].astype(np.int64)
    op_offsets = case["op_offsets"].astype(np.int64)
    due_rel = case["due_rel"].astype(np.float64)
    done = False
    step = 0
    decision_rows = []
    schedule_rows = []
    reward_rows = []
    cum_mk = 0.0
    cum_td = 0.0
    cum_od = 0.0
    cum_wait_od = 0.0
    cum_total = 0.0

    while not done:
        probs = policy_probs(ppo, state)
        action = int(np.argmax(probs))
        j_idx = int(action // env.number_of_machines)
        m = int(action % env.number_of_machines)
        cand_op = int(env.candidate[0, j_idx])
        local_op = int(cand_op - env.job_first_op_id[0, j_idx])
        global_op = int(op_offsets[j_idx] + local_op)
        raw_op = env.raw_fea_j[0, cand_op, :]

        decision_row = {
            "step": step,
            "action": action,
            "job_id": int(job_ids[j_idx]),
            "batch_job_index": j_idx,
            "machine": m,
            "candidate_global_op": global_op,
            "candidate_local_op": local_op,
            "policy_prob": float(probs[action]),
            "due_rel": float(due_rel[j_idx]),
            "raw_feat_slack": float(raw_op[11]),
            "raw_feat_overdue_weight": float(raw_op[19]) if raw_op.shape[0] > 19 else "",
            "true_candidate_pt": float(env.true_op_pt[0, cand_op, m]),
            "candidate_free_rel": float(env.true_candidate_free_time[0, j_idx]),
            "machine_free_rel": float(env.true_mch_free_time[0, m]),
        }

        state, _, done_flag, info = env.step(np.array([action]))
        reward_mk_step = float(np.asarray(info.get("reward_mk_step", [0.0])).reshape(-1)[0])
        reward_td_step = float(np.asarray(info.get("reward_td_step", [0.0])).reshape(-1)[0])
        reward_od_step = float(np.asarray(info.get("reward_od_step", [0.0])).reshape(-1)[0])
        reward_wait_od_step = float(np.asarray(info.get("reward_wait_od_step", [0.0])).reshape(-1)[0])
        reward_total_step = reward_mk_step + reward_td_step + reward_od_step + reward_wait_od_step
        cum_mk += reward_mk_step
        cum_td += reward_td_step
        cum_od += reward_od_step
        cum_wait_od += reward_wait_od_step
        cum_total += reward_total_step
        raw_mk_gain = float(info.get("raw_mk_gain", 0.0))
        raw_local_tardiness = float(info.get("raw_local_tardiness", 0.0))
        raw_accumulated_tardiness = float(info.get("raw_accumulated_tardiness", 0.0))

        det = info.get("scheduled_op_details", {})
        start = float(det.get("start_time", 0.0))
        end = float(det.get("end_time", 0.0))
        decision_row.update(
            {
                "reward_mk_step": reward_mk_step,
                "reward_td_step": reward_td_step,
                "reward_od_step": reward_od_step,
                "reward_wait_od_step": reward_wait_od_step,
                "reward_total_step": reward_total_step,
                "cum_reward_mk": cum_mk,
                "cum_reward_td": cum_td,
                "cum_reward_od": cum_od,
                "cum_reward_wait_od": cum_wait_od,
                "cum_reward_total": cum_total,
                "raw_mk_gain": raw_mk_gain,
                "raw_local_tardiness": raw_local_tardiness,
                "raw_accumulated_tardiness": raw_accumulated_tardiness,
            }
        )
        decision_rows.append(decision_row)
        reward_rows.append(
            {
                "step": step,
                "job_id": int(job_ids[j_idx]),
                "op": global_op,
                "machine": m,
                "due_rel": float(due_rel[j_idx]),
                "start_rel": start,
                "end_rel": end,
                "duration": float(det.get("proc_time", 0.0)),
                "reward_mk_step": reward_mk_step,
                "reward_td_step": reward_td_step,
                "reward_od_step": reward_od_step,
                "reward_wait_od_step": reward_wait_od_step,
                "reward_total_step": reward_total_step,
                "cum_reward_mk": cum_mk,
                "cum_reward_td": cum_td,
                "cum_reward_od": cum_od,
                "cum_reward_wait_od": cum_wait_od,
                "cum_reward_total": cum_total,
                "raw_mk_gain": raw_mk_gain,
                "raw_local_tardiness": raw_local_tardiness,
                "raw_accumulated_tardiness": raw_accumulated_tardiness,
            }
        )
        schedule_rows.append(
            {
                "step": step,
                "job_id": int(job_ids[j_idx]),
                "batch_job_index": j_idx,
                "op": global_op,
                "machine": m,
                "start_rel": start,
                "end_rel": end,
                "start_abs": start + batch_time,
                "end_abs": end + batch_time,
                "duration": float(det.get("proc_time", 0.0)),
                "due_rel": float(due_rel[j_idx]),
                "tardiness_rel_if_last": max(0.0, end - float(due_rel[j_idx])),
            }
        )
        done = bool(done_flag[0])
        step += 1

    return decision_rows, schedule_rows, reward_rows


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))
    case_path = Path(getattr(configs, "debug_ll_case_path", str(CASE_PATH)))
    if not case_path.exists():
        raise FileNotFoundError(f"Missing case file: {case_path}")
    case = np.load(case_path, allow_pickle=False)
    ppo = load_policy()
    env, state = build_env(case)

    first_probs = policy_probs(ppo, state)
    ranking_rows = dump_first_step_ranking(env, first_probs, case)
    decision_rows, schedule_rows, reward_rows = run_rollout(env, state, ppo, case)

    prefix = case_path.stem
    write_csv(OUTPUT_DIR / f"{prefix}_first_step_ranking.csv", ranking_rows)
    write_csv(OUTPUT_DIR / f"{prefix}_decisions.csv", decision_rows)
    write_csv(OUTPUT_DIR / f"{prefix}_schedule.csv", schedule_rows)
    write_csv(OUTPUT_DIR / f"{prefix}_reward_steps.csv", reward_rows)
    write_gantt(OUTPUT_DIR / f"{prefix}_gantt.png", schedule_rows)
    if reward_rows:
        summary_rows = [
            {
                "steps": len(reward_rows),
                "total_reward_mk": reward_rows[-1]["cum_reward_mk"],
                "total_reward_td": reward_rows[-1]["cum_reward_td"],
                "total_reward_od": reward_rows[-1]["cum_reward_od"],
                "total_reward_wait_od": reward_rows[-1]["cum_reward_wait_od"],
                "total_reward": reward_rows[-1]["cum_reward_total"],
                "final_raw_accumulated_tardiness": reward_rows[-1]["raw_accumulated_tardiness"],
            }
        ]
        write_csv(OUTPUT_DIR / f"{prefix}_reward_summary.csv", summary_rows)

    top = ranking_rows[0]
    overdue_top = [r for r in ranking_rows if float(r["due_rel"]) < 0][:10]
    print(f"[FIXED-LL] case={case_path}")
    print(f"[FIXED-LL] first action: rank1 job={top['job_id']} op={top['candidate_global_op']} m={top['machine']} prob={top['policy_prob']:.6f}")
    print("[FIXED-LL] overdue candidates among first-step ranking:")
    for row in overdue_top:
        print(
            f"  rank={row['policy_rank']:>3} job={row['job_id']} op={row['candidate_global_op']} "
            f"m={row['machine']} due_rel={float(row['due_rel']):.2f} slack={float(row['raw_feat_slack']):.2f} "
            f"prob={float(row['policy_prob']):.6f}"
        )
    print(f"[FIXED-LL] wrote {OUTPUT_DIR}")
    if reward_rows:
        print(
            "[FIXED-LL] reward totals: "
            f"MK={reward_rows[-1]['cum_reward_mk']:.6f}, "
            f"TD={reward_rows[-1]['cum_reward_td']:.6f}, "
            f"OD={reward_rows[-1]['cum_reward_od']:.6f}, "
            f"WaitOD={reward_rows[-1]['cum_reward_wait_od']:.6f}, "
            f"Total={reward_rows[-1]['cum_reward_total']:.6f}"
        )


if __name__ == "__main__":
    main()
