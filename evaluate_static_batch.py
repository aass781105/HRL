import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from params import configs
from model.PPO import PPO_initialize
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from data_utils import text_to_matrix
from common_utils import sample_action


def run_sample_episode(ppo, jl, pt, due_dates_abs, n_j, n_m, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(
        job_length_list=[jl],
        op_pt_list=[pt],
        due_date_list=[due_dates_abs],
        true_due_date_list=[due_dates_abs],
    )

    done = False
    while not done:
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
            action, _ = sample_action(pi)
        state, _, done, _ = env.step(action.cpu().numpy())

    return float(env.current_makespan[0]), float(env.accumulated_tardiness[0])


def evaluate_on_stored_instances(base_dir="or_instances_uniform"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = str(device)

    ppo = PPO_initialize()
    if os.path.exists(configs.ppo_model_path):
        ppo.policy.load_state_dict(torch.load(configs.ppo_model_path, map_location=device, weights_only=True))
        print(f"Loaded PPO model from {configs.ppo_model_path}")
    ppo.policy.to(device)
    ppo.policy.eval()

    eval_runs = int(getattr(configs, "eval_runs_per_instance", 10))
    if eval_runs <= 0:
        eval_runs = 10
    print(f"Static PPO evaluation action mode: sample | runs per instance: {eval_runs}")

    results = []
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found scales: {subdirs}")

    for scale in subdirs:
        curr_path = os.path.join(base_dir, scale)
        fjs_files = [f for f in os.listdir(curr_path) if f.endswith(".fjs")]

        print(f"\nEvaluating {scale} ({len(fjs_files)} instances)...")

        for fjs_name in tqdm(fjs_files):
            base_name = fjs_name.replace(".fjs", "")
            json_path = os.path.join(curr_path, f"{base_name}.json")
            fjs_path = os.path.join(curr_path, fjs_name)

            if not os.path.exists(json_path):
                continue

            with open(fjs_path, "r") as f:
                jl, pt = text_to_matrix(f.readlines())

            with open(json_path, "r") as f:
                due_data = json.load(f)
            due_dates_abs = np.array(due_data["due_dates"])

            n_j = jl.shape[0]
            n_m = pt.shape[1]
            mk_runs = []
            td_runs = []
            obj_runs = []
            seed_base = int(getattr(configs, "eval_seed", 42))
            instance_idx = len(results)

            for run_idx in range(eval_runs):
                run_seed = seed_base + instance_idx * 1000 + run_idx
                mk, td = run_sample_episode(ppo, jl, pt, due_dates_abs, n_j, n_m, seed=run_seed)
                mk_runs.append(mk)
                td_runs.append(td)
                obj_runs.append(0.5 * mk + 0.5 * td)

            results.append({
                "scale": scale,
                "instance": base_name,
                "runs": eval_runs,
                "makespan_mean": round(float(np.mean(mk_runs)), 2),
                "makespan_std": round(float(np.std(mk_runs, ddof=0)), 2),
                "total_tardiness_mean": round(float(np.mean(td_runs)), 2),
                "total_tardiness_std": round(float(np.std(td_runs, ddof=0)), 2),
                "obj_mean": round(float(np.mean(obj_runs)), 2),
                "obj_std": round(float(np.std(obj_runs, ddof=0)), 2),
            })

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    model_filename = os.path.basename(configs.ppo_model_path).replace(".pth", "")
    detail_csv = os.path.join(output_dir, f"ppo_static_bench_{model_filename}_sample{eval_runs}_details.csv")
    summary_csv = os.path.join(output_dir, f"ppo_static_bench_{model_filename}_sample{eval_runs}_summary.csv")

    df = pd.DataFrame(results)
    df["ppo_model"] = configs.ppo_model_path
    df.to_csv(detail_csv, index=False)

    summary_df = df.groupby("scale").agg({
        "makespan_mean": "mean",
        "makespan_std": "mean",
        "total_tardiness_mean": "mean",
        "total_tardiness_std": "mean",
        "obj_mean": "mean",
        "obj_std": "mean",
    }).reset_index()
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n--- Sample Evaluation Summary (Model: {model_filename}, runs={eval_runs}) ---")
    print(summary_df)
    print(f"\nDetailed results saved to: {detail_csv}")
    print(f"Summary results saved to: {summary_csv}")


if __name__ == "__main__":
    evaluate_on_stored_instances()
