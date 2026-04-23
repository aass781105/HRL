import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from common_utils import greedy_select_action, sample_action
from dynamic_job_stream import create_dynamic_world, sample_initial_jobs, register_initial_jobs
from FJSPEnvForVariousOpNums import FJSPEnvForVariousOpNums
from model.PPO import Memory, PPO_initialize
from params import configs

# ---- Local defaults (edit here directly) ----
DEFAULT_CADENCES = "1,3,5"
DEFAULT_NUM_DYNAMIC_INSTANCES = 30
DEFAULT_VAL_CADENCE = 3
DEFAULT_VAL_SEED = 42
DEFAULT_VAL_NUM_ENVS = 3
DEFAULT_SAVE_NAME = "llmk1_dynft_best.pth"
DEFAULT_FT_EPS_CLIP = 0.05
DEFAULT_FT_K_EPOCHS = 1
DEFAULT_FT_VLOSS_COEF = 0.01
DEFAULT_FT_ENTLOSS_COEF = 0.01
DEFAULT_UPDATE_EVERY_INSTANCES = 3
DEFAULT_FIXED_TRAIN_CADENCE = False
DEFAULT_RANDOMIZE_TRAIN_CADENCE = True
DEFAULT_MULTI_CAD_VAL = True
DEFAULT_TRAIN_EPISODES_PER_INSTANCE = 1
DEFAULT_VALIDATE_EVERY_EPISODES = 3
DEFAULT_DIFFERENT_POLICY_SEED_PER_LOCAL_EP = True


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cadences", type=str, default=DEFAULT_CADENCES, help="Training cadence set, comma-separated, e.g. 1,3,5")
    parser.add_argument("--num_dynamic_instances", type=int, default=DEFAULT_NUM_DYNAMIC_INSTANCES)
    parser.add_argument("--val_cadence", type=int, default=DEFAULT_VAL_CADENCE)
    parser.add_argument("--val_seed", type=int, default=DEFAULT_VAL_SEED)
    parser.add_argument("--val_num_envs", type=int, default=DEFAULT_VAL_NUM_ENVS)
    parser.add_argument("--finetune_lr", type=float, default=-1.0, help="If <=0, use 0.025 * configs.lr")
    parser.add_argument("--output_dir", type=str, default=os.path.join("evaluation_results", "ll_finetune_dynamic_cadence"))
    parser.add_argument("--save_name", type=str, default=DEFAULT_SAVE_NAME)
    args, remaining = parser.parse_known_args()
    import sys
    sys.argv = [sys.argv[0]] + remaining
    return args


def _parse_cadences(value: str) -> List[int]:
    raw = str(value or "").strip()
    if not raw:
        return [max(1, int(getattr(configs, "gate_cadence", 1)))]
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    vals = [max(1, v) for v in vals]
    return vals if vals else [max(1, int(getattr(configs, "gate_cadence", 1)))]


def _build_env_from_manifest(manifest: Dict, n_m: int):
    jl = np.asarray(manifest["job_length_list"][0], dtype=np.int32)
    pt = np.asarray(manifest["op_pt_list"][0], dtype=np.float64)
    due_ppo = np.asarray(manifest["due_date_list"][0], dtype=np.float64)
    due_abs = np.asarray(manifest["true_due_date_list"][0], dtype=np.float64)

    env = FJSPEnvForVariousOpNums(n_j=int(len(jl)), n_m=int(n_m))
    state = env.set_initial_data(
        job_length_list=[jl],
        op_pt_list=[pt],
        due_date_list=[due_ppo],
        normalize_due_date=False,
        true_due_date_list=[due_abs],
    )

    env.true_mch_free_time[0, :] = np.asarray(manifest["machine_free_time_abs"], dtype=np.float64)
    env.mch_free_time[0, :] = np.asarray(manifest["machine_free_time_env"], dtype=np.float64)
    for j_idx, job in enumerate(manifest.get("jobs", [])):
        env.true_candidate_free_time[0, j_idx] = float(job.get("ready_at_abs", manifest["batch_time_abs"]))
        env.candidate_free_time[0, j_idx] = float(job.get("ready_at_env", 0.0))
    state = env.rebuild_state_from_current()
    return env, state


def _state_shape_key(state) -> Tuple:
    return (
        tuple(state.fea_j_tensor.shape[1:]),
        tuple(state.op_mask_tensor.shape[1:]),
        tuple(state.fea_m_tensor.shape[1:]),
        tuple(state.mch_mask_tensor.shape[1:]),
        tuple(state.dynamic_pair_mask_tensor.shape[1:]),
        tuple(state.comp_idx_tensor.shape[1:]),
        tuple(state.candidate_tensor.shape[1:]),
        tuple(state.fea_pairs_tensor.shape[1:]),
    )


def _collect_one_batch_rollout(ppo, manifest: Dict, n_m: int, device: torch.device, memory_buckets: Dict[Tuple, Memory]):
    env, state = _build_env_from_manifest(manifest, n_m)
    key = _state_shape_key(state)
    if key not in memory_buckets:
        memory_buckets[key] = Memory(gamma=configs.gamma, gae_lambda=configs.gae_lambda)
    memory = memory_buckets[key]
    reward_sum = 0.0

    while True:
        memory.push(state)
        with torch.no_grad():
            batch_idx = ~torch.from_numpy(env.done_flag).to(state.fea_j_tensor.device)
            pi_valid, vals_valid = ppo.policy_old(
                fea_j=state.fea_j_tensor[batch_idx],
                op_mask=state.op_mask_tensor[batch_idx],
                candidate=state.candidate_tensor[batch_idx],
                fea_m=state.fea_m_tensor[batch_idx],
                mch_mask=state.mch_mask_tensor[batch_idx],
                comp_idx=state.comp_idx_tensor[batch_idx],
                dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                fea_pairs=state.fea_pairs_tensor[batch_idx],
            )
        act_valid, logp_valid = sample_action(pi_valid)

        full_actions = torch.zeros((1, 1), dtype=act_valid.dtype, device=act_valid.device)
        full_logp = torch.zeros((1, 1), dtype=logp_valid.dtype, device=logp_valid.device)
        full_vals = torch.zeros((1, 1), dtype=vals_valid.dtype, device=vals_valid.device)
        full_actions[batch_idx] = act_valid
        full_logp[batch_idx] = logp_valid
        full_vals[batch_idx] = vals_valid

        state, reward, done, _ = env.step(full_actions.cpu().numpy())
        reward_sum += float(np.sum(reward))
        memory.done_seq.append(torch.from_numpy(done).to(device))
        memory.reward_seq.append(torch.from_numpy(reward).to(device))
        memory.action_seq.append(full_actions.squeeze(-1))
        memory.log_probs.append(full_logp.squeeze(-1))
        memory.val_seq.append(full_vals.squeeze(1))
        if bool(done.all()):
            break

    return float(reward_sum)


def _evaluate_one_batch(ppo, manifest: Dict, n_m: int):
    env, state = _build_env_from_manifest(manifest, n_m)
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
        action = greedy_select_action(pi)
        state, _, done, _ = env.step(action.cpu().numpy())


def _run_dynamic_episode(
    ppo,
    cadence: int,
    seed: int,
    train_mode: bool,
    memory_buckets: Optional[Dict[Tuple, Memory]] = None,
    policy_seed: Optional[int] = None,
):
    rng, gen, orch = create_dynamic_world(
        configs,
        interarrival_mean=float(configs.interarrival_mean),
        burst_k=int(configs.burst_size),
        seed=int(seed),
    )
    # Keep the dynamic world fixed by `seed`, but allow policy sampling to diverge
    # across repeated local episodes on the same world.
    if train_mode and policy_seed is not None:
        torch.manual_seed(int(policy_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(policy_seed))
    orch.method = "PPO"
    orch._ppo = ppo

    all_due: Dict[int, float] = {}
    init_jobs = sample_initial_jobs(configs, rng=rng, base_job_id=0, t_arrive=0.0)
    register_initial_jobs(orch, gen, init_jobs, all_due, t0=0.0)

    t_now = 0.0
    t_next = float(gen.sample_next_time(t_now))
    arrival_count = 0
    batch_rewards = []

    for event_idx in range(1, int(configs.event_horizon) + 1):
        t_now = float(t_next)
        new_jobs = gen.generate_burst(t_now)
        for job in new_jobs:
            all_due[job.job_id] = float(job.meta["due_date"])
        if new_jobs:
            orch.buffer.extend(new_jobs)

        arrival_count += 1
        should_release = bool((arrival_count % cadence == 0) or (event_idx >= int(configs.event_horizon)))
        if should_release:
            result = orch.event_release_and_reschedule(t_now, event_id=event_idx)
            if result.get("event") == "batch_finalized" and orch.last_batch_manifest:
                if train_mode:
                    if memory_buckets is None:
                        raise RuntimeError("memory_buckets must be provided in train_mode=True")
                    r_sum = _collect_one_batch_rollout(
                        ppo, orch.last_batch_manifest, int(configs.n_m), torch.device(configs.device), memory_buckets
                    )
                    batch_rewards.append(r_sum)
                else:
                    _evaluate_one_batch(ppo, orch.last_batch_manifest, int(configs.n_m))
        else:
            orch.tick_without_release(t_now)
        t_next = float(gen.sample_next_time(t_now))

    while len(orch.buffer) > 0:
        result = orch.event_release_and_reschedule(float(orch.t), event_id=int(configs.event_horizon) + 1)
        if result.get("event") == "batch_finalized" and orch.last_batch_manifest:
            if train_mode:
                if memory_buckets is None:
                    raise RuntimeError("memory_buckets must be provided in train_mode=True")
                r_sum = _collect_one_batch_rollout(
                    ppo, orch.last_batch_manifest, int(configs.n_m), torch.device(configs.device), memory_buckets
                )
                batch_rewards.append(r_sum)
            else:
                _evaluate_one_batch(ppo, orch.last_batch_manifest, int(configs.n_m))
        if result.get("event") != "batch_finalized":
            break

    final_stats = orch.get_final_kpi_stats(all_due)
    return final_stats, batch_rewards


def _validate_multi_envs(ppo, val_cadence: int, val_seed: int, val_num_envs: int):
    mk_list = []
    td_list = []
    obj_list = []
    for k in range(max(1, int(val_num_envs))):
        seed_k = int(val_seed) + k
        stats_k, _ = _run_dynamic_episode(ppo, val_cadence, seed_k, train_mode=False)
        mk = float(stats_k["makespan"])
        td = float(stats_k["tardiness"])
        obj = 0.5 * mk + 0.5 * td
        mk_list.append(mk)
        td_list.append(td)
        obj_list.append(obj)
    return {
        "mk_mean": float(np.mean(mk_list)),
        "td_mean": float(np.mean(td_list)),
        "obj_mean": float(np.mean(obj_list)),
        "mk_list": mk_list,
        "td_list": td_list,
        "obj_list": obj_list,
    }


def _validate_multi_cadences(ppo, cadence_list: List[int], val_seed: int, val_num_envs: int):
    per_cad = {}
    mk_vals = []
    td_vals = []
    obj_vals = []
    for cad in cadence_list:
        stats = _validate_multi_envs(ppo, int(cad), val_seed, val_num_envs)
        per_cad[str(int(cad))] = stats
        mk_vals.append(float(stats["mk_mean"]))
        td_vals.append(float(stats["td_mean"]))
        obj_vals.append(float(stats["obj_mean"]))
    return {
        "mk_mean": float(np.mean(mk_vals)) if mk_vals else 0.0,
        "td_mean": float(np.mean(td_vals)) if td_vals else 0.0,
        "obj_mean": float(np.mean(obj_vals)) if obj_vals else 0.0,
        "per_cadence": per_cad,
    }


def main():
    args = parse_args()
    cadence_set = _parse_cadences(args.cadences)
    num_instances = max(1, int(args.num_dynamic_instances))
    val_cadence = max(1, int(args.val_cadence))
    val_seed = int(args.val_seed)
    val_num_envs = max(1, int(args.val_num_envs))
    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Fine-tune stability overrides (local to this script)
    configs.eps_clip = float(DEFAULT_FT_EPS_CLIP)
    configs.k_epochs = int(DEFAULT_FT_K_EPOCHS)
    configs.vloss_coef = float(DEFAULT_FT_VLOSS_COEF)
    configs.entloss_coef = float(DEFAULT_FT_ENTLOSS_COEF)
    ppo = PPO_initialize()
    model_path = str(getattr(configs, "ppo_model_path", "") or "").strip()
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Base low-level PPO model not found: {model_path}")
    device = torch.device(getattr(configs, "device", "cpu"))
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ppo.policy_old.load_state_dict(ppo.policy.state_dict())
    ppo.policy.to(device)
    ppo.policy_old.to(device)
    ppo.policy.train()
    ppo.policy_old.train()

    base_lr = float(getattr(configs, "lr", 3e-4))
    finetune_lr = float(args.finetune_lr) if float(args.finetune_lr) > 0 else base_lr * 0.025
    for g in ppo.optimizer.param_groups:
        g["lr"] = finetune_lr

    save_name = str(args.save_name or "").strip()
    if not save_name:
        save_name = f"{str(getattr(configs, 'eval_model_name', 'llppo'))}_ftdyn_best.pth"
    if not save_name.lower().endswith(".pth"):
        save_name += ".pth"
    save_path = os.path.join("trained_network", str(getattr(configs, "data_source", "SD2")), save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base_seed = int(getattr(configs, "event_seed", 42))
    val_seed_end = val_seed + val_num_envs - 1
    train_seed_start = base_seed
    if max(train_seed_start, val_seed) <= min(train_seed_start + num_instances - 1, val_seed_end):
        train_seed_start = val_seed_end + 1
    cadence_rng = np.random.default_rng(train_seed_start + 9973)
    best_val_obj = float("inf")
    history = []

    print("-" * 30)
    print("LL PPO Dynamic Fine-Tune (Cadence Set)")
    val_mode_desc = "cadence_set" if bool(DEFAULT_MULTI_CAD_VAL) else f"single({val_cadence})"
    print(
        f"cadence_set={cadence_set} | num_dynamic_instances={num_instances} | val_mode={val_mode_desc} | "
        f"val_seed={val_seed}..{val_seed_end} | train_seed_start={train_seed_start}"
    )
    print(
        f"finetune_lr={finetune_lr:.6g} | eps_clip={configs.eps_clip} | k_epochs={configs.k_epochs} | "
        f"vloss_coef={configs.vloss_coef} | entloss_coef={configs.entloss_coef}"
    )
    print(
        f"update_every_instances={int(DEFAULT_UPDATE_EVERY_INSTANCES)} | "
        f"fixed_train_cadence={bool(DEFAULT_FIXED_TRAIN_CADENCE)} | "
        f"randomize_train_cadence={bool(DEFAULT_RANDOMIZE_TRAIN_CADENCE)} | "
        f"multi_cad_val={bool(DEFAULT_MULTI_CAD_VAL)} | "
        f"train_eps_per_instance={int(DEFAULT_TRAIN_EPISODES_PER_INSTANCE)} | "
        f"validate_every_eps={int(DEFAULT_VALIDATE_EVERY_EPISODES)} | "
        f"diff_policy_seed_per_local_ep={bool(DEFAULT_DIFFERENT_POLICY_SEED_PER_LOCAL_EP)}"
    )
    if bool(DEFAULT_MULTI_CAD_VAL):
        print("note: --val_cadence is ignored when multi_cad_val=True (validation uses cadence_set).")
    print(f"validation aggregation: {val_num_envs} envs per cadence, objective = mean(0.5*MK + 0.5*TD)")
    print("-" * 30)

    train_memory_buckets: Dict[Tuple, Memory] = {}
    pending_instances = 0
    last_update_loss = 0.0
    last_update_v = 0.0
    last_update_p = 0.0
    last_grad_shared = 0.0
    last_grad_actor = 0.0
    last_grad_critic = 0.0
    last_grad_ratio = 0.0
    last_update_total_states = 0
    last_update_num_buckets = 0
    last_update_bucket_state_sizes = []

    # Baseline validation before any fine-tuning update.
    ppo.policy.eval()
    ppo.policy_old.eval()
    if bool(DEFAULT_MULTI_CAD_VAL):
        baseline_val = _validate_multi_cadences(ppo, cadence_set, val_seed, val_num_envs)
    else:
        baseline_val = _validate_multi_envs(ppo, val_cadence, val_seed, val_num_envs)
    ppo.policy.train()
    ppo.policy_old.train()
    best_val_obj = float(baseline_val["obj_mean"])
    torch.save(ppo.policy.state_dict(), save_path)
    print(
        f"[Baseline Val] MK={baseline_val['mk_mean']:.3f} | TD={baseline_val['td_mean']:.3f} | "
        f"Obj={baseline_val['obj_mean']:.3f} (saved as initial best)"
    )
    if bool(DEFAULT_MULTI_CAD_VAL):
        per_cad_base = baseline_val.get("per_cadence", {})
        cad_line = " | ".join(
            [f"cad{cad}:Obj={float(v.get('obj_mean', 0.0)):.3f}" for cad, v in per_cad_base.items()]
        )
        if cad_line:
            print(f"[Baseline Per-Cad] {cad_line}")

    episode_counter = 0
    for i in range(num_instances):
        if bool(DEFAULT_FIXED_TRAIN_CADENCE):
            cadence = int(cadence_set[0])
        elif bool(DEFAULT_RANDOMIZE_TRAIN_CADENCE):
            cadence = int(cadence_rng.choice(cadence_set))
        else:
            cadence = int(cadence_set[i % len(cadence_set)])
        train_seed = train_seed_start + i
        instance_rewards = []
        instance_losses = []
        instance_vs = []
        instance_ps = []
        instance_grad_shared = []
        instance_grad_actor = []
        instance_grad_critic = []
        instance_grad_ratio = []

        for local_ep in range(int(DEFAULT_TRAIN_EPISODES_PER_INSTANCE)):
            episode_counter += 1
            policy_seed = None
            if bool(DEFAULT_DIFFERENT_POLICY_SEED_PER_LOCAL_EP):
                policy_seed = int(train_seed_start + i * 1000 + local_ep)
            _, batch_rewards = _run_dynamic_episode(
                ppo,
                cadence,
                train_seed,
                train_mode=True,
                memory_buckets=train_memory_buckets,
                policy_seed=policy_seed,
            )
            pending_instances += 1
            instance_rewards.extend(batch_rewards)

            did_update = False
            if pending_instances >= int(DEFAULT_UPDATE_EVERY_INSTANCES):
                bucket_losses = []
                bucket_state_sizes = []
                total_states = 0
                for mem in train_memory_buckets.values():
                    if len(mem.reward_seq) == 0:
                        continue
                    steps = len(mem.reward_seq)
                    envs = int(mem.reward_seq[0].shape[0]) if steps > 0 else 0
                    states_in_bucket = int(steps * envs)
                    bucket_state_sizes.append(states_in_bucket)
                    total_states += states_in_bucket
                    l, v, p = ppo.update(mem)
                    bucket_losses.append((float(l), float(v), float(p)))
                    mem.clear_memory()
                if bucket_losses:
                    last_update_loss = float(np.mean([x[0] for x in bucket_losses]))
                    last_update_v = float(np.mean([x[1] for x in bucket_losses]))
                    last_update_p = float(np.mean([x[2] for x in bucket_losses]))
                    g = getattr(ppo, "last_grad_stats", {}) or {}
                    last_grad_shared = float(g.get("shared_grad_norm", 0.0))
                    last_grad_actor = float(g.get("actor_grad_norm", 0.0))
                    last_grad_critic = float(g.get("critic_grad_norm", 0.0))
                    last_grad_ratio = float(g.get("critic_actor_ratio", 0.0))
                    last_update_total_states = int(total_states)
                    last_update_num_buckets = int(len(bucket_state_sizes))
                    last_update_bucket_state_sizes = list(bucket_state_sizes)
                    instance_losses.append(last_update_loss)
                    instance_vs.append(last_update_v)
                    instance_ps.append(last_update_p)
                    instance_grad_shared.append(last_grad_shared)
                    instance_grad_actor.append(last_grad_actor)
                    instance_grad_critic.append(last_grad_critic)
                    instance_grad_ratio.append(last_grad_ratio)
                pending_instances = 0
                did_update = True

            # validate every N episodes
            if episode_counter % int(DEFAULT_VALIDATE_EVERY_EPISODES) != 0:
                print(
                    f"[EP {episode_counter:03d}] cad={cadence} seed={train_seed} local_ep={local_ep+1}/"
                    f"{int(DEFAULT_TRAIN_EPISODES_PER_INSTANCE)} | train(upd={int(did_update)})"
                )
                continue

            ppo.policy.eval()
            ppo.policy_old.eval()
            if bool(DEFAULT_MULTI_CAD_VAL):
                val_stats = _validate_multi_cadences(ppo, cadence_set, val_seed, val_num_envs)
            else:
                val_stats = _validate_multi_envs(ppo, val_cadence, val_seed, val_num_envs)
            ppo.policy.train()
            ppo.policy_old.train()

            val_td = float(val_stats["td_mean"])
            val_mk = float(val_stats["mk_mean"])
            val_obj = float(val_stats["obj_mean"])
            if val_obj < best_val_obj:
                best_val_obj = val_obj
                torch.save(ppo.policy.state_dict(), save_path)

            avg_loss = float(np.mean(instance_losses)) if instance_losses else float(last_update_loss)
            avg_v = float(np.mean(instance_vs)) if instance_vs else float(last_update_v)
            avg_p = float(np.mean(instance_ps)) if instance_ps else float(last_update_p)
            avg_r = float(np.mean(instance_rewards)) if instance_rewards else 0.0
            avg_gs = float(np.mean(instance_grad_shared)) if instance_grad_shared else float(last_grad_shared)
            avg_ga = float(np.mean(instance_grad_actor)) if instance_grad_actor else float(last_grad_actor)
            avg_gc = float(np.mean(instance_grad_critic)) if instance_grad_critic else float(last_grad_critic)
            avg_gr = float(np.mean(instance_grad_ratio)) if instance_grad_ratio else float(last_grad_ratio)
            bucket_sizes = list(last_update_bucket_state_sizes)
            bucket_sizes_sorted = sorted(bucket_sizes)
            bucket_min = int(bucket_sizes_sorted[0]) if bucket_sizes_sorted else 0
            bucket_med = int(bucket_sizes_sorted[len(bucket_sizes_sorted) // 2]) if bucket_sizes_sorted else 0
            bucket_max = int(bucket_sizes_sorted[-1]) if bucket_sizes_sorted else 0

            row = {
                "instance_idx": i + 1,
                "episode_idx": episode_counter,
                "train_seed": train_seed,
                "train_cadence": cadence,
                "train_batches": len(instance_rewards),
                "train_avg_loss": avg_loss,
                "train_avg_v_loss": avg_v,
                "train_avg_p_loss": avg_p,
                "train_avg_reward_sum": avg_r,
                "did_update": bool(did_update),
                "pending_instances_since_last_update": int(pending_instances),
                "grad_shared_norm": avg_gs,
                "grad_actor_norm": avg_ga,
                "grad_critic_norm": avg_gc,
                "grad_critic_actor_ratio": avg_gr,
                "update_total_states": int(last_update_total_states),
                "update_num_buckets": int(last_update_num_buckets),
                "update_bucket_state_sizes": bucket_sizes,
                "val_seed": val_seed,
                "val_cadence": val_cadence,
                "val_num_envs": val_num_envs,
                "val_makespan": val_mk,
                "val_tardiness": val_td,
                "val_objective": val_obj,
                "val_per_cadence": val_stats.get("per_cadence", {}),
                "best_val_objective_so_far": best_val_obj,
            }
            history.append(row)
            per_cad = val_stats.get("per_cadence", {})
            per_cad_line = ""
            if per_cad:
                per_cad_line = " | " + ", ".join(
                    [f"c{cad}:{float(v.get('obj_mean', 0.0)):.1f}" for cad, v in per_cad.items()]
                )
            print(
                f"[EP {episode_counter:03d}] inst={i+1:03d}/{num_instances} cad={cadence} seed={train_seed} | "
                f"train(loss={avg_loss:.4f}, v={avg_v:.4f}, p={avg_p:.4f}) | "
                f"grad(shared={avg_gs:.3e}, actor={avg_ga:.3e}, critic={avg_gc:.3e}, c/a={avg_gr:.2f}) | "
                f"update(states={last_update_total_states}, buckets={last_update_num_buckets}, "
                f"b[min/med/max]={bucket_min}/{bucket_med}/{bucket_max}) | "
                f"val(MK={val_mk:.3f}, TD={val_td:.3f}, Obj={val_obj:.3f}){per_cad_line} | bestObj={best_val_obj:.3f}"
            )

    # Flush remaining collected rollouts (if any) so all train data contributes.
    has_pending_memory = any(len(mem.reward_seq) > 0 for mem in train_memory_buckets.values())
    if pending_instances > 0 and has_pending_memory:
        bucket_losses = []
        for mem in train_memory_buckets.values():
            if len(mem.reward_seq) == 0:
                continue
            l, v, p = ppo.update(mem)
            bucket_losses.append((float(l), float(v), float(p)))
            mem.clear_memory()
        if bucket_losses:
            flush_loss = float(np.mean([x[0] for x in bucket_losses]))
            flush_v = float(np.mean([x[1] for x in bucket_losses]))
            flush_p = float(np.mean([x[2] for x in bucket_losses]))
        else:
            flush_loss = flush_v = flush_p = 0.0
        print(
            f"[Flush Update] pending_instances={pending_instances} | "
            f"loss={flush_loss:.4f}, v={flush_v:.4f}, p={flush_p:.4f}"
        )

    summary = {
        "base_model": model_path,
        "saved_best_model": save_path,
        "num_dynamic_instances": num_instances,
        "cadence_set": cadence_set,
        "train_seed_start": train_seed_start,
        "val_seed": val_seed,
        "val_cadence": val_cadence,
        "val_num_envs": val_num_envs,
        "finetune_lr": finetune_lr,
        "update_every_instances": int(DEFAULT_UPDATE_EVERY_INSTANCES),
        "fixed_train_cadence": bool(DEFAULT_FIXED_TRAIN_CADENCE),
        "randomize_train_cadence": bool(DEFAULT_RANDOMIZE_TRAIN_CADENCE),
        "multi_cad_val": bool(DEFAULT_MULTI_CAD_VAL),
        "baseline_val": baseline_val,
        "best_val_objective": best_val_obj,
        "history": history,
    }
    summary_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(save_name))[0]}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print("Done.")
    print(f"Best model: {save_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
