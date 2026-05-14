import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--n_j", type=int, default=None)
    parser.add_argument("--n_m", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=os.path.join("diagnostics", "state_features"))
    parser.add_argument("--action_mode", type=str, default="random", choices=["random", "spt"])
    parser.add_argument("--inspect_model_input", type=str, default="true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


LOCAL_ARGS = parse_local_args()

from params import configs  # noqa: E402
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums  # noqa: E402
from data_utils import SD2_instance_generator, generate_due_dates  # noqa: E402
from model.PPO import PPO_initialize  # noqa: E402


OP_FEATURE_NAMES = [
    "op_scheduled_flag",
    "op_ct_lb",
    "op_min_pt",
    "pt_span",
    "op_mean_pt",
    "op_waiting_time",
    "op_remain_work",
    "job_left_op_nums_scaled",
    "job_remain_work",
    "op_available_mch_nums",
    "due_remaining_time_signed",
    "slack_signed",
    "critical_ratio_log",
    "is_tardy_flag",
    "job_current_tardiness_log",
    "slack_rank",
    "slack_gap_to_min_log",
    "remaining_flex_min",
    "remaining_flex_mean",
]

MCH_FEATURE_NAMES = [
    "available_job_count_z",
    "available_op_count_z",
    "mch_min_pt_z",
    "mch_mean_pt_z",
    "mch_waiting_time_z",
    "mch_tardiness_pressure",
    "global_slack_mean_signed_log",
    "global_slack_std_log",
    "global_congestion_log",
]

PAIR_FEATURE_NAMES = [
    "candidate_pt",
    "candidate_pt_over_op_max",
    "candidate_pt_over_mch_candidate_max",
    "candidate_pt_over_global_remain_max",
    "candidate_pt_over_mch_remain_max",
    "candidate_pt_over_pair_max",
    "candidate_pt_over_job_remain_work",
    "pair_est_lateness_log",
]


def feature_names(prefix: str, names: List[str], dim: int) -> List[str]:
    out = []
    for i in range(dim):
        label = names[i] if i < len(names) else f"dim{i}"
        out.append(f"{prefix}{i:02d}_{label}")
    return out


def finite_stats(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = int(flat.size)
    finite = np.isfinite(flat)
    good = flat[finite]
    if good.size == 0:
        return {
            "count": total,
            "finite_count": 0,
            "nan_ratio": float(np.isnan(flat).mean()) if total else 0.0,
            "inf_ratio": float(np.isinf(flat).mean()) if total else 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "zero_ratio": np.nan,
            "negative_ratio": np.nan,
            "positive_ratio": np.nan,
            "unique_count": 0,
        }
    return {
        "count": total,
        "finite_count": int(good.size),
        "nan_ratio": float(np.isnan(flat).mean()) if total else 0.0,
        "inf_ratio": float(np.isinf(flat).mean()) if total else 0.0,
        "mean": float(np.mean(good)),
        "std": float(np.std(good)),
        "min": float(np.min(good)),
        "p1": float(np.percentile(good, 1)),
        "p5": float(np.percentile(good, 5)),
        "p50": float(np.percentile(good, 50)),
        "p95": float(np.percentile(good, 95)),
        "p99": float(np.percentile(good, 99)),
        "max": float(np.max(good)),
        "zero_ratio": float(np.mean(good == 0.0)),
        "negative_ratio": float(np.mean(good < 0.0)),
        "positive_ratio": float(np.mean(good > 0.0)),
        "unique_count": int(np.unique(np.round(good, decimals=8)).size),
    }


def add_feature_stats(rows: List[Dict], group: str, names: List[str], data: np.ndarray, valid_mask: np.ndarray = None):
    arr = np.asarray(data)
    dim = int(arr.shape[-1])
    flat = arr.reshape(-1, dim)
    if valid_mask is not None:
        valid = np.asarray(valid_mask).reshape(-1)
        flat = flat[valid]
    for i, name in enumerate(feature_names("", names, dim)):
        row = {"group": group, "feature_index": i, "feature": name}
        row.update(finite_stats(flat[:, i] if flat.size else np.array([])))
        rows.append(row)


def choose_actions(env: FJSPEnvForVariousOpNums, rng: np.random.Generator, mode: str) -> np.ndarray:
    actions = np.zeros(env.number_of_envs, dtype=int)
    for e in range(env.number_of_envs):
        if bool(env.done_flag[e]):
            actions[e] = 0
            continue
        valid = np.argwhere(~env.dynamic_pair_mask[e])
        if valid.size == 0:
            actions[e] = 0
            continue
        if mode == "spt":
            pts = env.true_op_pt[e, env.candidate[e, valid[:, 0]], valid[:, 1]]
            idx = int(np.argmin(pts))
        else:
            idx = int(rng.integers(0, valid.shape[0]))
        job, mch = valid[idx]
        actions[e] = int(job * env.number_of_machines + mch)
    return actions


def collect_snapshot(env: FJSPEnvForVariousOpNums, episode: int, step: int, warnings: List[str]) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    deleted_op_nodes = getattr(env, "deleted_op_nodes", np.zeros_like(env.mask_dummy_node, dtype=bool))
    valid_ops = ~(env.mask_dummy_node | deleted_op_nodes.astype(bool))
    valid_pairs = ~env.dynamic_pair_mask

    if not np.array_equal(env.state.dynamic_pair_mask_tensor.detach().cpu().numpy(), env.dynamic_pair_mask):
        warnings.append(f"episode={episode} step={step}: state.dynamic_pair_mask_tensor mismatches env.dynamic_pair_mask")

    valid_pair_counts = valid_pairs.reshape(env.number_of_envs, -1).sum(axis=1)
    for e, cnt in enumerate(valid_pair_counts):
        if not env.done_flag[e] and cnt <= 0:
            warnings.append(f"episode={episode} step={step} env={e}: no valid candidate pair while not done")

    if np.any(env.candidate < 0) or np.any(env.candidate >= env.max_number_of_ops):
        warnings.append(f"episode={episode} step={step}: candidate index out of op range")

    raw = getattr(env, "raw_fea_j", None)
    if raw is not None and raw.shape[-1] >= 14:
        raw_rem = raw[:, :, 10]
        raw_slack = raw[:, :, 11]
        raw_is_tardy = raw[:, :, 13]
        norm_rem = env.fea_j[:, :, 10]
        norm_slack = env.fea_j[:, :, 11]
        rem_sign_mismatch = np.logical_and(valid_ops, np.sign(raw_rem) != np.sign(norm_rem))
        slack_sign_mismatch = np.logical_and(valid_ops, np.sign(raw_slack) != np.sign(norm_slack))
        if np.any(rem_sign_mismatch):
            warnings.append(f"episode={episode} step={step}: due_remaining sign changed after normalization")
        if np.any(slack_sign_mismatch):
            warnings.append(f"episode={episode} step={step}: slack sign changed after normalization")
        tardy_flag_bad = np.logical_and(valid_ops, raw_is_tardy != (raw_rem < 0).astype(float))
        if np.any(tardy_flag_bad):
            warnings.append(f"episode={episode} step={step}: is_tardy flag does not match raw due_remaining_time < 0")

    if np.any(env.fea_pairs[:, :, :, 7][valid_pairs] < -1e-8):
        warnings.append(f"episode={episode} step={step}: pair_est_lateness_log has negative values")

    step_rows = []
    if raw is not None:
        raw_slack = raw[:, :, 11]
        raw_rem = raw[:, :, 10]
        raw_tardy = raw[:, :, 13]
        step_rows.append(
            {
                "episode": episode,
                "step": step,
                "remaining_ops": int(np.sum(env.op_scheduled_flag == 0)),
                "valid_pairs_mean": float(np.mean(valid_pair_counts)),
                "valid_pairs_min": int(np.min(valid_pair_counts)),
                "valid_pairs_max": int(np.max(valid_pair_counts)),
                "raw_due_rem_mean": float(np.mean(raw_rem[valid_ops])) if np.any(valid_ops) else 0.0,
                "raw_slack_mean": float(np.mean(raw_slack[valid_ops])) if np.any(valid_ops) else 0.0,
                "raw_slack_min": float(np.min(raw_slack[valid_ops])) if np.any(valid_ops) else 0.0,
                "raw_slack_neg_ratio": float(np.mean(raw_slack[valid_ops] < 0)) if np.any(valid_ops) else 0.0,
                "raw_is_tardy_ratio": float(np.mean(raw_tardy[valid_ops] > 0.5)) if np.any(valid_ops) else 0.0,
            }
        )
    return step_rows, {"valid_ops": valid_ops, "valid_pairs": valid_pairs}


def plot_distributions(stats_df: pd.DataFrame, step_df: pd.DataFrame, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sel = stats_df[stats_df["group"].isin(["raw_op", "norm_op"])]
    for group, grp in sel.groupby("group"):
        ax.plot(grp["feature_index"], grp["std"], marker="o", label=group)
    ax.set_title("Feature std by op feature index")
    ax.set_xlabel("feature index")
    ax.set_ylabel("std")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "op_feature_std.png"), dpi=150)
    plt.close(fig)

    if not step_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(step_df["step"], step_df["raw_slack_mean"], label="raw_slack_mean")
        ax.plot(step_df["step"], step_df["raw_slack_min"], label="raw_slack_min")
        ax2 = ax.twinx()
        ax2.plot(step_df["step"], step_df["raw_slack_neg_ratio"], color="tab:red", label="slack_neg_ratio")
        ax.set_title("Slack over rollout steps")
        ax.set_xlabel("step")
        ax.set_ylabel("slack")
        ax2.set_ylabel("ratio")
        ax.grid(True, linestyle=":")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "slack_over_steps.png"), dpi=150)
        plt.close(fig)


def main():
    rng = np.random.default_rng(int(LOCAL_ARGS.seed))
    n_j = int(LOCAL_ARGS.n_j or getattr(configs, "n_j", 10))
    n_m = int(LOCAL_ARGS.n_m or getattr(configs, "n_m", 5))
    configs.n_j = n_j
    configs.n_m = n_m

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOCAL_ARGS.output_dir, f"state_diag_j{n_j}_m{n_m}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    stats_rows: List[Dict] = []
    model_input_rows: List[Dict] = []
    step_rows: List[Dict] = []
    warnings: List[str] = []
    inspect_model_input = str(LOCAL_ARGS.inspect_model_input).lower() in ("1", "true", "yes", "y")
    ppo = None
    if inspect_model_input:
        try:
            ppo = PPO_initialize()
            import torch
            model_path = str(getattr(configs, "ppo_model_path", "") or "")
            if model_path and os.path.exists(model_path):
                ppo.policy.load_state_dict(torch.load(model_path, map_location=getattr(configs, "device", "cpu"), weights_only=True))
            ppo.policy.eval()
        except Exception as exc:
            warnings.append(f"model input inspection disabled: {exc}")
            ppo = None

    for episode in range(int(LOCAL_ARGS.episodes)):
        jl, pt, _ = SD2_instance_generator(configs, rng=rng)
        due_mode = str(getattr(configs, "due_date_mode", "range"))
        due = generate_due_dates(jl, pt, due_date_mode=due_mode, rng=rng)

        env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
        state = env.set_initial_data(
            job_length_list=[jl],
            op_pt_list=[pt],
            due_date_list=[due],
            true_due_date_list=[due],
        )

        done = False
        step = 0
        while not bool(np.all(done)):
            rows, masks = collect_snapshot(env, episode, step, warnings)
            step_rows.extend(rows)

            if hasattr(env, "raw_fea_j"):
                add_feature_stats(stats_rows, "raw_op", OP_FEATURE_NAMES, env.raw_fea_j, masks["valid_ops"])
            add_feature_stats(stats_rows, "norm_op", OP_FEATURE_NAMES, env.fea_j, masks["valid_ops"])
            add_feature_stats(stats_rows, "machine", MCH_FEATURE_NAMES, env.fea_m, None)
            add_feature_stats(stats_rows, "pair_valid", PAIR_FEATURE_NAMES, env.fea_pairs, masks["valid_pairs"])
            if ppo is not None:
                try:
                    import torch
                    with torch.no_grad():
                        candidate_feature, global_feature = ppo.policy._compute_policy_features(
                            state.fea_j_tensor,
                            state.op_mask_tensor,
                            state.candidate_tensor,
                            state.fea_m_tensor,
                            state.mch_mask_tensor,
                            state.comp_idx_tensor,
                            state.dynamic_pair_mask_tensor,
                            state.fea_pairs_tensor,
                        )
                    valid_pair_flat = (~state.dynamic_pair_mask_tensor).detach().cpu().numpy().reshape(-1)
                    cand_np = candidate_feature.detach().cpu().numpy().reshape(-1, candidate_feature.shape[-1])
                    cand_valid = cand_np[valid_pair_flat]
                    for idx in range(cand_valid.shape[-1]):
                        row = {"group": "actor_candidate_feature", "feature_index": idx, "feature": f"actor_input_{idx:03d}"}
                        row.update(finite_stats(cand_valid[:, idx]))
                        model_input_rows.append(row)
                    glob_np = global_feature.detach().cpu().numpy().reshape(-1, global_feature.shape[-1])
                    for idx in range(glob_np.shape[-1]):
                        row = {"group": "critic_global_feature", "feature_index": idx, "feature": f"critic_input_{idx:03d}"}
                        row.update(finite_stats(glob_np[:, idx]))
                        model_input_rows.append(row)
                except Exception as exc:
                    warnings.append(f"episode={episode} step={step}: model input inspection failed: {exc}")
                    ppo = None

            actions = choose_actions(env, rng, LOCAL_ARGS.action_mode)
            _, _, done, _ = env.step(actions)
            step += 1
            if step > int(np.sum(jl)) + 5:
                warnings.append(f"episode={episode}: rollout exceeded expected op count; stopping")
                break

    stats_df = pd.DataFrame(stats_rows)
    # Aggregate repeated per-step stats into one final row per feature.
    numeric_cols = [c for c in stats_df.columns if c not in ("group", "feature")]
    agg = stats_df.groupby(["group", "feature_index", "feature"], as_index=False)[numeric_cols].mean(numeric_only=True)
    step_df = pd.DataFrame(step_rows)

    stats_path = os.path.join(out_dir, "state_feature_stats.csv")
    model_input_path = os.path.join(out_dir, "model_input_feature_stats.csv")
    step_path = os.path.join(out_dir, "state_feature_by_step.csv")
    warnings_path = os.path.join(out_dir, "state_feature_warnings.txt")
    agg.to_csv(stats_path, index=False, encoding="utf-8-sig")
    model_input_df = pd.DataFrame(model_input_rows)
    if not model_input_df.empty:
        model_numeric_cols = [c for c in model_input_df.columns if c not in ("group", "feature")]
        model_agg = model_input_df.groupby(["group", "feature_index", "feature"], as_index=False)[model_numeric_cols].mean(numeric_only=True)
        model_agg.to_csv(model_input_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(model_input_path, index=False, encoding="utf-8-sig")
    step_df.to_csv(step_path, index=False, encoding="utf-8-sig")

    auto_warnings = []
    for _, row in agg.iterrows():
        feature = f"{row['group']}:{row['feature']}"
        if float(row.get("nan_ratio", 0.0)) > 0:
            auto_warnings.append(f"{feature}: nan_ratio={row['nan_ratio']:.6f}")
        if float(row.get("inf_ratio", 0.0)) > 0:
            auto_warnings.append(f"{feature}: inf_ratio={row['inf_ratio']:.6f}")
        if float(row.get("std", 0.0)) < 1e-8:
            auto_warnings.append(f"{feature}: near-zero std")
        if abs(float(row.get("max", 0.0))) > 1e4 or abs(float(row.get("min", 0.0))) > 1e4:
            auto_warnings.append(f"{feature}: extreme magnitude min={row['min']:.3g}, max={row['max']:.3g}")

    with open(warnings_path, "w", encoding="utf-8") as f:
        f.write("Manual semantic warnings\n")
        f.write("========================\n")
        for w in warnings:
            f.write(w + "\n")
        f.write("\nAutomatic distribution warnings\n")
        f.write("===============================\n")
        for w in auto_warnings:
            f.write(w + "\n")

    plot_distributions(agg, step_df, out_dir)

    print(f"State diagnostics written to: {out_dir}")
    print(f"Stats: {stats_path}")
    print(f"Model input stats: {model_input_path}")
    print(f"Steps: {step_path}")
    print(f"Warnings: {warnings_path}")
    print(f"Warnings count: semantic={len(warnings)}, distribution={len(auto_warnings)}")


if __name__ == "__main__":
    main()
