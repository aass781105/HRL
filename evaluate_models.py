# evaluate_models.py
# This script benchmarks a PPO model on a fixed set of dynamically generated instances.
# It reads all configuration from the global `configs` object (from params.py)
# to determine which model to test, how many times to run each instance, and
# parameters for the dynamic environment. The high-level gate policy supports
# PPO gate or cadence-based release decisions.
# The results are saved to a uniquely named CSV file based on the model name.

import os
import csv
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import re

# --- Core components from the project ---
from params import configs
from model.PPO_dynamic import PPO
from common_utils import sample_action, greedy_select_action
from data_utils import SD2_instance_generator
from global_env import (
    GlobalTimelineOrchestrator,
    EventBurstGenerator,
    BatchScheduleRecorder,
)
from PPO_orchestrator_adapter import OrchestratorAdapter

# ================== Monkey Patch for BatchScheduleRecorder.record_step ==================
# This patch is necessary because the environment's `step` method returns 4 values,
# but the original `record_step` in `global_env.py` only expects 3, causing a ValueError.
# By applying this patch at runtime, we fix the issue without modifying the original environment file.

def patched_record_step(self, env, action: int, env_idx: int = 0):
    """ Corrected version of record_step that handles 4 return values from env.step(). """
    J, M = env.number_of_jobs, env.number_of_machines
    chosen_job = int(action // M)
    chosen_mch = int(action % M)
    chosen_op_global = int(env.candidate[env.env_idxs, chosen_job][env_idx])
    op_id_in_job = int(chosen_op_global - env.job_first_op_id[env_idx, chosen_job])

    job_meta = self.jobs[chosen_job].meta or {}
    op_offset = int(job_meta.get("op_offset", 0))
    op_global_in_job = op_id_in_job + op_offset

    start = float(max(env.true_candidate_free_time[env_idx, chosen_job],
                      env.true_mch_free_time[env_idx, chosen_mch]))

    # THE FIX: Unpack 4 values, ignoring the 4th (info).
    state, reward, done, _ = env.step(np.array([action]))

    end = float(env.true_op_ct[env_idx, chosen_op_global])
    job_id = int(self.jobs[chosen_job].job_id)

    self.machine_queues[chosen_mch].append((job_id, op_global_in_job, start, end))
    self.rows.append({"job": job_id, "op": op_global_in_job, "machine": chosen_mch,
                      "start": start, "end": end, "duration": end - start})
    return state, reward, done

# Apply the patch
BatchScheduleRecorder.record_step = patched_record_step
# ============================== End of Monkey Patch ===============================


from model.gate_state import calculate_gate_state
from model.ppo_gate_model import PPOGateNet

def fixed_k_sampler(K: int):
    """[ADDED] 固定一次釋放 K 筆的 sampler。"""
    def _fn(rng: np.random.Generator) -> int:
        return int(K)
    return _fn

def _gate_obs(orch: GlobalTimelineOrchestrator, n_machines: int, t_now: float,
              burst_K: int, interarrival_mean: float,
              buf_cap_cfg: int = 0) -> np.ndarray:
    """
    Calculates the observation for the PPO high-level gate policy.
    """
    scale = (float(configs.low) + float(configs.high)) / 2.0
    cap = int(buf_cap_cfg) if int(buf_cap_cfg) > 0 else max(1, int(burst_K) * 3)
    
    mft_abs = np.asarray(orch.machine_free_time, dtype=float)
    rem = np.maximum(0.0, mft_abs - float(t_now))
    horizon = float(rem.min()) if rem.size > 0 else 0.0
    
    w_idle = orch.compute_weighted_idle(t_now, horizon)

    buf_stats = {"tardiness_ratio": 0.0, "min_slack": 0.0, "avg_slack": 0.0, "slack_std": 0.0}
    if orch.buffer:
        slacks = []
        neg = 0
        for j in orch.buffer:
            total_proc = float(j.meta.get("total_proc_time", 0.0))
            due = float(j.meta.get("due_date", 0.0))
            slack = due - t_now - total_proc
            slacks.append(slack)
            if t_now + total_proc > due:
                neg += 1
        buf_stats = {
            "tardiness_ratio": neg / len(orch.buffer),
            "min_slack": min(slacks),
            "avg_slack": float(np.mean(slacks)),
            "slack_std": float(np.std(slacks)),
        }

    wip_stats = orch.get_wip_stats(t_now)

    return calculate_gate_state(
        buffer_size=len(orch.buffer),
        machine_free_time=orch.machine_free_time,
        t_now=t_now,
        n_machines=n_machines,
        obs_buffer_cap=cap,
        time_scale=scale,
        weighted_idle=w_idle,
        unweighted_idle=orch.compute_unweighted_idle(t_now, horizon) if horizon > 0 else 0.0,
        buffer_stats=buf_stats,
        wip_stats=wip_stats,
    )


def run_dynamic_ppo_episode(adapter, ppo_policy, action_selection, device, max_events,
                            gate_policy, ppo_gate_model, burst_size, interarrival_mean, job_gen):
    """
    Runs a single dynamic episode with PPO or cadence gate decisions.
    """
    orch = adapter.o
    release_count = 0
    t_prev = 0.0

    num_arrivals = 0
    t_now = 0.0
    t_next = job_gen.sample_next_time(t_now)

    while num_arrivals < max_events and t_next >= 0:
        t_now = t_next
        dt_interval = t_now - t_prev
        t_prev = t_now
        
        new_jobs = job_gen.generate_burst(t_now)
        if new_jobs:
            orch.buffer.extend(new_jobs)
            num_arrivals += 1
            
        # --- Gating Decision ---
        decide_release = False
        if gate_policy == 'ppo' and ppo_gate_model is not None and len(orch.buffer) > 0:
            gate_obs_buffer_cap = int(getattr(configs, "gate_obs_buffer_cap", 0))
            obs = _gate_obs(orch, orch.M, t_now, burst_size, interarrival_mean,
                            buf_cap_cfg=gate_obs_buffer_cap)
            with torch.no_grad():
                logits, _ = ppo_gate_model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                if action_selection == 'sample':
                    act = torch.distributions.Categorical(logits=logits).sample().item()
                else:
                    act = torch.argmax(logits, dim=1).item()
            decide_release = (act == 1)
        else:
            cadence = max(1, int(getattr(configs, "gate_cadence", 1)))
            decide_release = (num_arrivals % cadence == 0)
        
        # --- Scheduling (if released) ---
        if decide_release:
            state_ppo = adapter.begin_new_batch(t_now)
            if state_ppo:
                release_count += 1
                while True:
                    with torch.no_grad():
                        pi, _ = ppo_policy(
                            fea_j=state_ppo.fea_j_tensor, op_mask=state_ppo.op_mask_tensor,
                            candidate=state_ppo.candidate_tensor, fea_m=state_ppo.fea_m_tensor,
                            mch_mask=state_ppo.mch_mask_tensor, comp_idx=state_ppo.comp_idx_tensor,
                            dynamic_pair_mask=state_ppo.dynamic_pair_mask_tensor, fea_pairs=state_ppo.fea_pairs_tensor
                        )
                    
                    if action_selection == 'sample':
                        action_ppo, _ = sample_action(pi)
                    else:
                        action_ppo = greedy_select_action(pi)

                    state_ppo, _, sub_done, _ = adapter.step_in_batch(action_ppo.cpu())
                    
                    if sub_done:
                        break
                adapter.finalize_batch()
        
        t_next = job_gen.sample_next_time(t_now)
            
    # --- Final Flush ---
    flush_rounds = 0
    while len(orch.buffer) > 0 and flush_rounds < 50:
        flush_rounds += 1
        release_count += 1
        t_flush = orch.t
        state_ppo = adapter.begin_new_batch(t_flush)
        if not state_ppo: break
            
        while True:
            with torch.no_grad():
                pi, _ = ppo_policy(
                    fea_j=state_ppo.fea_j_tensor, op_mask=state_ppo.op_mask_tensor, candidate=state_ppo.candidate_tensor,
                    fea_m=state_ppo.fea_m_tensor, mch_mask=state_ppo.mch_mask_tensor, comp_idx=state_ppo.comp_idx_tensor,
                    dynamic_pair_mask=state_ppo.dynamic_pair_mask_tensor, fea_pairs=state_ppo.fea_pairs_tensor
                )
            if action_selection == 'sample':
                action_ppo, _ = sample_action(pi)
            else:
                action_ppo = greedy_select_action(pi)

            state_ppo, _, sub_done, _ = adapter.step_in_batch(action_ppo.cpu())
            if sub_done: break
        adapter.finalize_batch()

    final_makespan = np.max(adapter.o.machine_free_time) if len(adapter.o.machine_free_time) > 0 else 0.0
    return final_makespan, release_count


def main():
    # --- 1. Load Configuration from params.configs ---
    test_seed = getattr(configs, 'eval_seed', 42)
    runs_per_instance = getattr(configs, 'eval_runs_per_instance', 10)
    num_test_instances = getattr(configs, 'eval_num_instances', 10)
    
    event_horizon = configs.event_horizon
    interarrival_mean = configs.interarrival_mean
    burst_size = configs.burst_size

    model_name = getattr(configs, 'eval_model_name', 'default_model')
    ppo_model_path = getattr(configs, 'ppo_model_path', None)
    action_selection = getattr(configs, 'eval_action_selection', 'sample')

    gate_policy = getattr(configs, 'gate_policy', 'ppo').lower()
    ppo_gate_model_path = getattr(configs, 'ppo_gate_model_path', None)
    
    if not ppo_model_path:
        tqdm.write("Error: PPO model path not specified. Please set --eval_ppo_model_path.")
        return

    device = torch.device(configs.device)
    
    # --- Load PPO Gate Model (if applicable) ---
    ppo_gate_model = None
    if gate_policy == 'ppo':
        if not ppo_gate_model_path or not os.path.exists(ppo_gate_model_path):
            tqdm.write(f"[WARN] gate_policy is 'ppo' but ppo_gate_model_path is invalid: {ppo_gate_model_path}")
            tqdm.write("       Falling back to cadence gate policy.")
            gate_policy = 'cadence'
        else:
            ppo_gate_model = PPOGateNet(
                obs_dim=18,
                n_actions=2,
                hidden=int(getattr(configs, "ppo_gate_hidden_dim", 256)),
                num_layers=int(getattr(configs, "ppo_gate_num_layers", 3)),
                separate_trunks=bool(getattr(configs, "ppo_gate_separate_trunks", False)),
                actor_hidden=int(getattr(configs, "ppo_gate_actor_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
                actor_num_layers=int(getattr(configs, "ppo_gate_actor_num_layers", getattr(configs, "ppo_gate_num_layers", 3))),
                critic_hidden=int(getattr(configs, "ppo_gate_critic_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
                critic_num_layers=int(getattr(configs, "ppo_gate_critic_num_layers", getattr(configs, "ppo_gate_num_layers", 3))),
                value_hidden=int(getattr(configs, "ppo_gate_value_hidden_dim", getattr(configs, "ppo_gate_hidden_dim", 256))),
                value_num_layers=int(getattr(configs, "ppo_gate_value_num_layers", 1)),
            ).to(device)
            ppo_gate_model.load_state_dict(torch.load(ppo_gate_model_path, map_location=device, weights_only=True))
            ppo_gate_model.eval()
            tqdm.write(f"Successfully loaded PPO gate model from: {ppo_gate_model_path}")

    tqdm.write("-" * 25 + " Starting Dynamic Model Evaluation " + "-" * 25)
    tqdm.write(f"PPO Model: {model_name}")
    tqdm.write(f"Gate Policy: {gate_policy}")
    tqdm.write(f"Test instances seed: {test_seed}")
    tqdm.write(f"Runs per instance: {runs_per_instance}")
    tqdm.write(f"Number of test instances: {num_test_instances}")
    tqdm.write(f"Events per instance: {event_horizon}")
    tqdm.write(f"Inter-arrival mean: {interarrival_mean}")
    tqdm.write(f"Jobs per arrival: {burst_size}")
    
    # --- 2. Generate Fixed Seeds for Test Instances ---
    tqdm.write(f"Generating {num_test_instances} fixed instance seeds...")
    instance_seed_rng = np.random.default_rng(test_seed)
    test_instance_seeds = [instance_seed_rng.integers(low=0, high=1_000_000) for _ in range(num_test_instances)]
    tqdm.write("Instance seeds generated successfully.")
    
    all_results = []

    # --- 3. Main Evaluation Logic ---
    tqdm.write(f"\n--- Testing Model: {model_name} ---")
    if not os.path.exists(ppo_model_path):
        tqdm.write(f"FATAL: PPO Model path not found: '{ppo_model_path}'. Exiting.")
        return
        
    ppo_policy = PPO(configs).policy
    ppo_policy.load_state_dict(torch.load(ppo_model_path, map_location=device, weights_only=True))
    ppo_policy.to(device)
    ppo_policy.eval()

    for i in tqdm(range(num_test_instances), desc=f"Instances for {model_name}"):
        instance_seed = test_instance_seeds[i]
        instance_makespans = []
        instance_release_counts = []
        for run_num in trange(runs_per_instance, desc="Runs", leave=False):
            run_rng = np.random.default_rng(instance_seed)
            job_generator = EventBurstGenerator(
                sd2_fn=SD2_instance_generator, base_config=copy.deepcopy(configs),
                n_machines=configs.n_m, interarrival_mean=interarrival_mean,
                k_sampler=fixed_k_sampler(int(burst_size)), rng=run_rng,
            )
            orchestrator = GlobalTimelineOrchestrator(
                n_machines=configs.n_m, job_generator=job_generator,
                select_from_buffer=lambda buf, o: list(range(len(buf))), t0=0.0
            )
            adapter = OrchestratorAdapter(orchestrator, n_machines=configs.n_m)
            
            makespan, rel_cnt = run_dynamic_ppo_episode(
                adapter, ppo_policy, action_selection, device, event_horizon,
                gate_policy, ppo_gate_model, burst_size, interarrival_mean,
                job_generator
            )
            instance_makespans.append(makespan)
            instance_release_counts.append(rel_cnt)

        avg_makespan = np.mean(instance_makespans)
        std_makespan = np.std(instance_makespans)
        avg_release_count = np.mean(instance_release_counts)
        
        instance_result = {
            "model_name": model_name,
            "instance_id": i + 1,
            "avg_makespan": avg_makespan,
            "std_makespan": std_makespan,
            "avg_release_count": avg_release_count,
        }
        for r_idx, r_makespan in enumerate(instance_makespans):
            instance_result[f"run{r_idx + 1}"] = r_makespan
            
        all_results.append(instance_result)

    # --- 4. Output Results ---
    tqdm.write("\n" + "=" * 25 + " Evaluation Results " + "=" * 25)
    tqdm.write(f"\nModel: {model_name}")
    tqdm.write("-" * 55)
    tqdm.write(f"{'Instance':<12} | {'Avg Makespan':<15} | {'Std Dev':<15} | {'Avg Rel Cnt':<15}")
    tqdm.write("-" * 55)
    for res in all_results:
        tqdm.write(f"{res['instance_id']:<12} | {res['avg_makespan']:<15.2f} | {res['std_makespan']:<15.2f} | {res['avg_release_count']:<15.2f}")

    safe_model_name = re.sub(r'[\\/*?:"<>|]', "", model_name)
    csv_file = f'dynamic_evaluation_results_{safe_model_name}.csv'
    tqdm.write(f"\nSaving detailed results to {csv_file}...")
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["Model Name", "Instance ID"]
            header.extend([f"Run {j+1}" for j in range(runs_per_instance)])
            header.extend(["Avg Makespan", "Std Dev Makespan", "Avg Release Count"])
            writer.writerow(header)
            for res in all_results:
                row_data = [res['model_name'], res['instance_id']]
                row_data.extend([res.get(f"run{j+1}") for j in range(runs_per_instance)])
                row_data.extend([res['avg_makespan'], res['std_makespan'], res['avg_release_count']])
                writer.writerow(row_data)
        tqdm.write("Successfully saved results to CSV.")
    except IOError as e:
        tqdm.write(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    main()
