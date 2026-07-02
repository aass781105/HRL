from __future__ import annotations

import csv
import json
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

import dynamic_job_stream
import hrl_main
from hrl_orchestrator import GlobalTimelineOrchestrator
from params import configs


TARGET_RELEASE_INDEX = 60
OUTPUT_DIR = Path("debug") / "dynamic_r60_candidate_features"


OP_FEATURE_NAMES = [
    "op_scheduled_flag",
    "op_ct_lb",
    "op_min_pt",
    "pt_span",
    "op_mean_pt",
    "op_waiting_time",
    "op_remain_work",
    "op_match_job_left_op_nums",
    "op_match_job_remain_work",
    "op_available_mch_nums",
    "feat_rem_time",
    "feat_slack",
    "feat_cr_log",
    "feat_is_tardy",
    "feat_job_current_tardiness",
    "feat_slack_rank",
    "feat_slack_gap_to_min",
    "feat_remaining_flex_min",
    "feat_remaining_flex_mean",
    "feat_overdue_weight",
]

PAIR_FEATURE_NAMES = [
    "candidate_pt",
    "pt_over_op_max",
    "pt_over_mch_candidate_max",
    "pt_over_global_remain_max",
    "pt_over_mch_remain_max",
    "pt_over_pair_max",
    "pt_over_job_remain_work",
    "pair_est_lateness_log",
]

MCH_FEATURE_NAMES = [
    "mch_free_time",
    "mch_workload",
    "mch_next_workload",
    "mch_available_op_nums",
    "mch_available_job_nums",
    "global_slack_mean",
    "global_slack_std",
    "global_tardy_ratio",
    "global_pressure",
]


class DumpComplete(RuntimeError):
    pass


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    try:
        f = path.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        f = alt.open("w", newline="", encoding="utf-8-sig")
        print(f"[DUMP] target locked; wrote alternate CSV {alt}")
    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DumpingOrchestrator(GlobalTimelineOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_release_index = 0

    def solve_current_batch_static(self, env, state):
        self._debug_release_index += 1
        if self._debug_release_index == TARGET_RELEASE_INDEX:
            self._dump_initial_candidate_features(env, state)
            raise DumpComplete(f"Dumped release {TARGET_RELEASE_INDEX}")
        return super().solve_current_batch_static(env, state)

    def _dump_initial_candidate_features(self, env, state) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        manifest = getattr(self, "last_batch_manifest", {}) or {}
        jobs_payload = manifest.get("jobs", [])
        batch_time = float(manifest.get("batch_time", self.t))
        pt_scale = float(manifest.get("pt_scale", (float(configs.low) + float(configs.high)) / 2.0))
        event_id = manifest.get("event_id", "")

        with torch.inference_mode():
            pi = self._ppo.policy.policy_only(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
        probs = pi.detach().cpu().numpy()[0]
        order = np.argsort(-probs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        greedy_action = int(order[0])

        rows: List[Dict] = []
        summary_rows: List[Dict] = []
        e = 0
        n_m = int(env.number_of_machines)
        candidate_ops = env.candidate[e]

        for j_idx, job in enumerate(self._committed_jobs):
            job_payload = jobs_payload[j_idx] if j_idx < len(jobs_payload) else {}
            candidate_op = int(candidate_ops[j_idx])
            op_offset = int(job.meta.get("op_offset", 0))
            local_op = int(candidate_op - env.job_first_op_id[e, j_idx])
            global_op = int(op_offset + local_op)
            due_abs = float(job.meta.get("due_date", 0.0))
            ready_abs = float(job.meta.get("ready_at", batch_time))
            due_rel = due_abs - batch_time
            ready_rel = ready_abs - batch_time
            raw_op = env.raw_fea_j[e, candidate_op, :]
            norm_op = env.fea_j[e, candidate_op, :]

            summary = {
                "release_index": TARGET_RELEASE_INDEX,
                "event_id": event_id,
                "batch_time": batch_time,
                "pt_scale": pt_scale,
                "batch_job_index": int(j_idx),
                "job_id": int(job.job_id),
                "candidate_local_op": local_op,
                "candidate_global_op": global_op,
                "op_offset": op_offset,
                "remaining_ops": int(len(job.operations)),
                "total_ops": int(job.meta.get("total_ops", op_offset + len(job.operations))),
                "arrive_abs": float(job.meta.get("t_arrive", batch_time)),
                "ready_abs": ready_abs,
                "ready_rel": ready_rel,
                "due_abs": due_abs,
                "due_rel": due_rel,
                "due_rel_over_pt_scale": due_rel / max(pt_scale, 1e-8),
                "is_ready_now": int(ready_abs <= batch_time + 1e-9),
                "is_overdue_now": int(due_abs < batch_time),
                "raw_feat_rem_time": float(raw_op[10]),
                "raw_feat_slack": float(raw_op[11]),
                "raw_feat_is_tardy": float(raw_op[13]),
                "raw_feat_job_current_tardiness": float(raw_op[14]),
                "raw_feat_overdue_weight": float(raw_op[19]) if raw_op.shape[0] > 19 else "",
                "manifest_due_rel": job_payload.get("due_date_rel", ""),
                "manifest_ready_rel": job_payload.get("ready_at_rel", ""),
            }
            summary_rows.append(summary)

            for m in range(n_m):
                action = int(j_idx * n_m + m)
                pair_mask = bool(env.dynamic_pair_mask[e, j_idx, m])
                row = dict(summary)
                row.update(
                    {
                        "machine": int(m),
                        "action": action,
                        "is_valid_pair": int(pair_mask),
                        "policy_prob": float(probs[action]),
                        "policy_rank": int(ranks[action]),
                        "is_greedy_action": int(action == greedy_action),
                        "true_candidate_pt": float(env.true_op_pt[e, candidate_op, m]),
                        "true_candidate_free_time": float(env.true_candidate_free_time[e, j_idx]),
                        "true_mch_free_time": float(env.true_mch_free_time[e, m]),
                    }
                )
                for idx, name in enumerate(OP_FEATURE_NAMES):
                    row[f"raw_op_{idx:02d}_{name}"] = float(raw_op[idx]) if idx < raw_op.shape[0] else ""
                    row[f"norm_op_{idx:02d}_{name}"] = float(norm_op[idx]) if idx < norm_op.shape[0] else ""
                pair = env.fea_pairs[e, j_idx, m, :]
                for idx, name in enumerate(PAIR_FEATURE_NAMES):
                    row[f"pair_{idx:02d}_{name}"] = float(pair[idx]) if idx < pair.shape[0] else ""
                mch = env.fea_m[e, m, :]
                for idx, name in enumerate(MCH_FEATURE_NAMES):
                    row[f"mch_{idx:02d}_{name}"] = float(mch[idx]) if idx < mch.shape[0] else ""
                rows.append(row)

        prefix = f"r{TARGET_RELEASE_INDEX:03d}_event{event_id}_t{int(round(batch_time)):05d}"
        _write_csv(OUTPUT_DIR / f"{prefix}_candidate_pairs.csv", rows)
        _write_csv(OUTPUT_DIR / f"{prefix}_candidate_jobs.csv", summary_rows)

        top_rows = sorted(rows, key=lambda r: int(r["policy_rank"]))[:50]
        _write_csv(OUTPUT_DIR / f"{prefix}_top50_policy_pairs.csv", top_rows)
        self._save_fixed_case(env, manifest, prefix)
        print(f"[DUMP] wrote {OUTPUT_DIR / (prefix + '_candidate_pairs.csv')}")
        print(f"[DUMP] wrote {OUTPUT_DIR / (prefix + '_candidate_jobs.csv')}")
        print(f"[DUMP] wrote {OUTPUT_DIR / (prefix + '_top50_policy_pairs.csv')}")

    def _save_fixed_case(self, env, manifest: Dict, prefix: str) -> None:
        batch_time = float(manifest.get("batch_time", self.t))
        n_ops = int(env.env_number_of_ops[0])
        job_lengths = np.asarray([len(job.operations) for job in self._committed_jobs], dtype=np.int64)
        op_pt = np.asarray(env.true_op_pt[0, :n_ops, :], dtype=np.float64)
        due_abs = np.asarray([float(job.meta.get("due_date", 0.0)) for job in self._committed_jobs], dtype=np.float64)
        due_rel = due_abs - batch_time
        ready_abs = np.asarray([float(job.meta.get("ready_at", batch_time)) for job in self._committed_jobs], dtype=np.float64)
        ready_rel = ready_abs - batch_time
        machine_free_abs = np.asarray(env.true_mch_free_time[0], dtype=np.float64)
        machine_free_rel = machine_free_abs - batch_time
        job_ids = np.asarray([int(job.job_id) for job in self._committed_jobs], dtype=np.int64)
        op_offsets = np.asarray([int(job.meta.get("op_offset", 0)) for job in self._committed_jobs], dtype=np.int64)
        total_ops = np.asarray(
            [int(job.meta.get("total_ops", int(op_offsets[i] + job_lengths[i]))) for i, job in enumerate(self._committed_jobs)],
            dtype=np.int64,
        )
        arrive_abs = np.asarray([float(job.meta.get("t_arrive", batch_time)) for job in self._committed_jobs], dtype=np.float64)

        npz_path = OUTPUT_DIR / f"{prefix}_fixed_ll_case.npz"
        np.savez_compressed(
            npz_path,
            batch_time=np.asarray(batch_time, dtype=np.float64),
            event_id=np.asarray(int(manifest.get("event_id", -1)), dtype=np.int64),
            job_lengths=job_lengths,
            op_pt=op_pt,
            due_abs=due_abs,
            due_rel=due_rel,
            ready_abs=ready_abs,
            ready_rel=ready_rel,
            machine_free_abs=machine_free_abs,
            machine_free_rel=machine_free_rel,
            job_ids=job_ids,
            op_offsets=op_offsets,
            total_ops=total_ops,
            arrive_abs=arrive_abs,
        )
        meta_path = OUTPUT_DIR / f"{prefix}_fixed_ll_case_meta.json"
        meta = {
            "case_npz": str(npz_path),
            "batch_time": batch_time,
            "event_id": int(manifest.get("event_id", -1)),
            "job_count": int(len(job_lengths)),
            "op_count": int(n_ops),
            "machine_count": int(env.number_of_machines),
            "model_path": str(getattr(configs, "ll_ppo_model_path", "")),
            "note": "Times in *_rel are shifted so the dynamic release time is static t=0.",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[DUMP] wrote {npz_path}")
        print(f"[DUMP] wrote {meta_path}")


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(configs, "device_id", ""))
    configs.scheduler_type = "PPO"
    configs.ll_rollout_k = 1
    hrl_main.GlobalTimelineOrchestrator = DumpingOrchestrator
    dynamic_job_stream.GlobalTimelineOrchestrator = DumpingOrchestrator
    try:
        hrl_main.run_event_driven_until_nevents(
            max_events=int(configs.event_horizon),
            interarrival_mean=float(configs.interarrival_mean),
            burst_K=int(configs.burst_size),
            plot_global_dir=str(getattr(configs, "plot_global_dir", "plots/global")),
            write_outputs=False,
            seed_override=int(getattr(configs, "event_seed", 42)),
            sample_seed_override=int(getattr(configs, "event_seed", 42)),
        )
    except DumpComplete as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
