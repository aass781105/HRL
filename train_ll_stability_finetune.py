"""Fine-tune the lower-level PPO with schedule-stability supervision.

The script deliberately keeps the legacy lower-level checkpoint frozen as a
reference-policy generator.  Trainable rollouts use the confirmed 30% fresh /
70% virtual-cut reschedule mixture and the confirmed incremental stability
penalty.  It is intentionally separate from ``train_ll_curriculum.py`` so the
legacy training path remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_local_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "yaml_config" / "train_ll_stability_finetune.yml"),
    )
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def load_project_config(config_path):
    """Let the project's params parser load the requested YAML only."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    original_argv = sys.argv[:]
    sys.argv = [original_argv[0], "--config", str(config_path)]
    try:
        from params import configs
    finally:
        sys.argv = original_argv
    return configs, config_path


def set_global_config(global_config, configured):
    """Copy the fine-tune configuration into the project's global namespace."""
    for key, value in vars(configured).items():
        setattr(global_config, key, value)


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clone_policy_config(configured, stability):
    cfg = copy.deepcopy(configured)
    if stability:
        cfg.fea_j_input_dim = 22
        cfg.fea_pair_input_dim = 12
        cfg.ll_stability_enable = True
        cfg.ll_stability_critic_summary = True
    else:
        cfg.fea_j_input_dim = 20
        cfg.fea_pair_input_dim = 8
        cfg.ll_stability_enable = False
        cfg.ll_stability_critic_summary = False
    return cfg


def load_policy_checkpoint(ppo, path, torch_module):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference checkpoint not found: {path}")
    try:
        state_dict = torch_module.load(
            str(path), map_location=ppo.device, weights_only=True
        )
    except TypeError:
        state_dict = torch_module.load(str(path), map_location=ppo.device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    ppo.policy.load_state_dict(state_dict, strict=True)
    ppo.policy_old.load_state_dict(ppo.policy.state_dict())
    return path


def stability_hold_schedule(configured):
    """Return the explicit target-size and due-setting update schedule."""
    due_hold = max(1, int(getattr(configured, "ll_due_setting_hold_updates", 20)))
    size_hold = max(1, int(getattr(configured, "ll_mixed_size_hold_updates", 60)))
    due_modes = ("range3_loose", "range3_mixed", "range3_tight")
    expected_size_hold = due_hold * len(due_modes)
    if size_hold != expected_size_hold:
        raise ValueError(
            "ll_mixed_size_hold_updates must equal 3 * "
            f"ll_due_setting_hold_updates ({expected_size_hold}), got {size_hold}."
        )
    return size_hold, due_hold, due_modes


def effective_due_mode(configured, update):
    mode = str(getattr(configured, "ll_due_date_mode", "k"))
    if mode == "range3_hold":
        _, due_hold, due_modes = stability_hold_schedule(configured)
        due_slot = (int(update) % (due_hold * len(due_modes))) // due_hold
        return due_modes[due_slot]
    return mode


def generate_instance(configured, n_jobs, seed):
    from data_utils import SD2_instance_generator

    generator_config = copy.deepcopy(configured)
    generator_config.n_j = int(n_jobs)
    return SD2_instance_generator(generator_config, seed=int(seed), mode="uniform")


def generate_due(
    configured,
    job_lengths,
    op_pt,
    mode,
    seed,
    allow_overdue_injection=True,
):
    from data_utils import generate_due_dates

    return generate_due_dates(
        job_lengths,
        op_pt,
        due_date_mode=mode,
        seed=int(seed),
        due_config=configured,
        allow_overdue_injection=allow_overdue_injection,
    )


def empty_reference(op_count):
    import numpy as np

    return {
        "ref_mask": np.zeros(int(op_count), dtype=bool),
        "is_new_job_op": np.ones(int(op_count), dtype=np.float32),
        "old_machine": np.full(int(op_count), -1, dtype=np.int32),
        "old_rank": np.full(int(op_count), -1, dtype=np.int32),
        "old_rank_norm": np.zeros(int(op_count), dtype=np.float32),
    }


def make_fresh_sample(
    configured,
    target_jobs,
    seed,
    due_mode,
    allow_overdue=True,
):
    import numpy as np

    job_lengths, op_pt, _ = generate_instance(configured, target_jobs, seed)
    due = generate_due(
        configured,
        job_lengths,
        op_pt,
        due_mode,
        seed,
        allow_overdue_injection=allow_overdue,
    )
    if not allow_overdue:
        due = np.maximum(np.asarray(due, dtype=np.float64), 0.0)
    return {
        "job_lengths": np.asarray(job_lengths, dtype=np.int64),
        "op_pt": np.asarray(op_pt, dtype=np.float64),
        "due": np.asarray(due, dtype=np.float64),
        "release": np.zeros(int(target_jobs), dtype=np.float64),
        "machine_free": np.zeros(int(configured.n_m), dtype=np.float64),
        "reference": empty_reference(int(np.sum(job_lengths))),
        "kind": "fresh",
    }


def greedy_reference_schedule(reference_ppo, configured, job_lengths, op_pt, due):
    """Generate one old-policy schedule and return operation rows."""
    import numpy as np
    import torch
    from ll_fjsp_env import LLFJSPEnv
    from common_utils import greedy_select_action

    env = LLFJSPEnv(
        n_j=int(job_lengths.shape[0]),
        n_m=int(op_pt.shape[1]),
        stability_enabled=False,
    )
    state = env.set_initial_data(
        [job_lengths],
        [op_pt],
        due_date_list=[due],
        true_due_date_list=[due],
    )
    rows = []
    done = False
    while not done:
        with torch.no_grad():
            pi = reference_ppo.policy.policy_only(
                state.fea_j_tensor,
                state.op_mask_tensor,
                state.candidate_tensor,
                state.fea_m_tensor,
                state.mch_mask_tensor,
                state.comp_idx_tensor,
                state.dynamic_pair_mask_tensor,
                state.fea_pairs_tensor,
            )
            action = greedy_select_action(pi).reshape(-1)
        action_int = int(action[0].item())
        next_state, _, done_flag, info = env.step(np.asarray([action_int], dtype=np.int64))
        detail = info["scheduled_op_details"]
        rows.append({
            "job": int(detail["job_id"]),
            "op": int(detail["op_id_in_job"]),
            "op_global": int(detail["op_global_id"]),
            "machine": int(detail["machine_id"]),
            "start": float(detail["start_time"]),
            "end": float(detail["end_time"]),
            "duration": float(detail["proc_time"]),
        })
        state = next_state
        done = bool(done_flag[0])
    return rows


def make_reschedule_sample(
    configured,
    reference_ppo,
    target_jobs,
    seed,
    due_mode,
    rng,
    allow_overdue=True,
):
    """Create a relative-time virtual-cut subproblem from an old schedule."""
    import numpy as np

    base_jobs = int(getattr(configured, "ll_stability_initial_jobs", 30))
    base_lengths, base_pt, _ = generate_instance(configured, base_jobs, seed)
    base_due = generate_due(
        configured,
        base_lengths,
        base_pt,
        due_mode,
        seed,
        allow_overdue_injection=allow_overdue,
    )
    if not allow_overdue:
        base_due = np.maximum(np.asarray(base_due, dtype=np.float64), 0.0)
    rows = greedy_reference_schedule(reference_ppo, configured, base_lengths, base_pt, base_due)
    if not rows:
        return make_fresh_sample(
            configured,
            target_jobs,
            seed + 1,
            due_mode,
            allow_overdue=allow_overdue,
        )

    reference_makespan = max(float(row["end"]) for row in rows)
    cut_low = max(0.0, float(getattr(configured, "ll_stability_cut_low", 0.10)))
    cut_high = max(cut_low, float(getattr(configured, "ll_stability_cut_high", 0.50)))
    cut_time = float(rng.uniform(cut_low, cut_high)) * reference_makespan

    rows_by_job = {job_id: [] for job_id in range(base_jobs)}
    for row in rows:
        rows_by_job.setdefault(int(row["job"]), []).append(row)

    machine_free = np.zeros(int(configured.n_m), dtype=np.float64)
    for machine_id in range(int(configured.n_m)):
        active = [
            row for row in rows
            if int(row["machine"]) == machine_id
            and float(row["start"]) <= cut_time < float(row["end"])
        ]
        if active:
            machine_free[machine_id] = max(
                0.0, max(float(row["end"]) for row in active) - cut_time
            )

    retained = []
    for job_id in range(base_jobs):
        job_rows = rows_by_job.get(job_id, [])
        future = sorted(
            [row for row in job_rows if float(row["start"]) > cut_time + 1e-9],
            key=lambda row: (int(row["op"]), float(row["start"])),
        )
        if not future:
            continue
        active = [
            row for row in job_rows
            if float(row["start"]) <= cut_time < float(row["end"])
        ]
        ready_after_cut = 0.0
        if active:
            ready_after_cut = max(
                0.0, max(float(row["end"]) for row in active) - cut_time
            )
        relative_due = float(base_due[job_id]) - cut_time
        if not allow_overdue:
            relative_due = max(0.0, relative_due)
        retained.append({
            "job": job_id,
            "future": future,
            "ready": ready_after_cut,
            "due": relative_due,
        })

    job_lengths = []
    op_blocks = []
    due_dates = []
    release_times = []
    ref_masks = []
    new_flags = []
    old_machines = []
    old_ranks = []
    old_rank_norms = []

    for item in retained:
        future = item["future"]
        job_lengths.append(len(future))
        op_blocks.append(np.asarray([base_pt[int(row["op_global"])] for row in future], dtype=np.float64))
        due_dates.append(item["due"])
        release_times.append(item["ready"])

        by_machine = {}
        for row in future:
            by_machine.setdefault(int(row["machine"]), []).append(row)
        rank_by_key = {}
        for machine_rows in by_machine.values():
            machine_rows.sort(key=lambda row: (float(row["start"]), int(row["op"])))
            denom = max(len(machine_rows) - 1, 1)
            for rank, row in enumerate(machine_rows):
                key = (int(row["job"]), int(row["op"]))
                rank_by_key[key] = (rank, float(rank) / float(denom))

        for row in future:
            rank, rank_norm = rank_by_key[(int(row["job"]), int(row["op"]))]
            ref_masks.append(True)
            new_flags.append(0.0)
            old_machines.append(int(row["machine"]))
            old_ranks.append(rank)
            old_rank_norms.append(rank_norm)

    new_count = max(0, int(target_jobs) - len(retained))
    if new_count:
        new_lengths, new_pt, _ = generate_instance(
            configured,
            new_count,
            int(seed) * 1009 + int(target_jobs) * 17 + 7,
        )
        new_due = generate_due(
            configured,
            new_lengths,
            new_pt,
            due_mode,
            int(seed) * 1013 + 11,
            allow_overdue_injection=False,
        )
        if not allow_overdue:
            new_due = np.maximum(np.asarray(new_due, dtype=np.float64), 0.0)
        for job_length, block, due in zip(new_lengths, np.split(new_pt, np.cumsum(new_lengths)[:-1]), new_due):
            job_lengths.append(int(job_length))
            op_blocks.append(np.asarray(block, dtype=np.float64))
            due_dates.append(float(due))
            release_times.append(0.0)
            ref_masks.extend([False] * int(job_length))
            new_flags.extend([1.0] * int(job_length))
            old_machines.extend([-1] * int(job_length))
            old_ranks.extend([-1] * int(job_length))
            old_rank_norms.extend([0.0] * int(job_length))

    if not op_blocks:
        return make_fresh_sample(
            configured,
            target_jobs,
            seed + 2,
            due_mode,
            allow_overdue=allow_overdue,
        )

    return {
        "job_lengths": np.asarray(job_lengths, dtype=np.int64),
        "op_pt": np.concatenate(op_blocks, axis=0),
        "due": np.asarray(due_dates, dtype=np.float64),
        "release": np.asarray(release_times, dtype=np.float64),
        "machine_free": machine_free,
        "reference": {
            "ref_mask": np.asarray(ref_masks, dtype=bool),
            "is_new_job_op": np.asarray(new_flags, dtype=np.float32),
            "old_machine": np.asarray(old_machines, dtype=np.int32),
            "old_rank": np.asarray(old_ranks, dtype=np.int32),
            "old_rank_norm": np.asarray(old_rank_norms, dtype=np.float32),
        },
        "kind": "reschedule",
        "cut_time": cut_time,
        "retained_jobs": len(retained),
    }


def generate_batch(
    configured,
    reference_ppo,
    sample_index,
    target_jobs=None,
    pattern=None,
    due_mode=None,
    num_envs=None,
    allow_overdue=True,
):
    import numpy as np

    if num_envs is None:
        num_envs = int(getattr(configured, "ll_num_envs", 1))
    else:
        num_envs = int(num_envs)
    low = int(getattr(configured, "ll_stability_target_jobs_low", 30))
    high = int(getattr(configured, "ll_stability_target_jobs_high", 40))
    if target_jobs is None:
        target_rng = np.random.default_rng(int(configured.seed_train) + int(sample_index) * 100003)
        target_jobs = int(target_rng.integers(min(low, high), max(low, high) + 1))
    target_jobs = max(1, int(target_jobs))
    if due_mode is None:
        due_mode = effective_due_mode(configured, sample_index)

    fresh_ratio = max(0.0, float(getattr(configured, "ll_stability_fresh_ratio", 0.30)))
    reschedule_ratio = max(0.0, float(getattr(configured, "ll_stability_reschedule_ratio", 0.70)))
    total_ratio = max(fresh_ratio + reschedule_ratio, 1e-8)
    fresh_ratio /= total_ratio

    samples = []
    for env_idx in range(num_envs):
        sample_seed = int(configured.seed_train) + int(sample_index) * 100003 + env_idx * 1009
        rng = np.random.default_rng(sample_seed)
        if pattern is not None:
            is_fresh = bool(pattern[env_idx % len(pattern)])
        else:
            is_fresh = bool(rng.random() < fresh_ratio)
        if is_fresh:
            sample = make_fresh_sample(
                configured,
                target_jobs,
                sample_seed,
                due_mode,
                allow_overdue=allow_overdue,
            )
        else:
            sample = make_reschedule_sample(
                configured,
                reference_ppo,
                target_jobs,
                sample_seed,
                due_mode,
                rng,
                allow_overdue=allow_overdue,
            )
        samples.append(sample)
    return samples


def build_env(configured, samples, stability_enabled=True):
    import numpy as np
    from ll_fjsp_env import LLFJSPEnv

    env = LLFJSPEnv(
        n_j=int(samples[0]["job_lengths"].shape[0]),
        n_m=int(configured.n_m),
        stability_enabled=stability_enabled,
    )
    env.set_initial_data(
        [sample["job_lengths"] for sample in samples],
        [sample["op_pt"] for sample in samples],
        due_date_list=[sample["due"] for sample in samples],
        true_due_date_list=[sample["due"] for sample in samples],
        release_time_list=[sample["release"] for sample in samples],
        stability_reference=[sample["reference"] for sample in samples],
    )

    machine_free = np.asarray([sample["machine_free"] for sample in samples], dtype=np.float64)
    ready = np.asarray([sample["release"] for sample in samples], dtype=np.float64)
    scale = max(float(env.pt_scale), 1e-8)
    env.true_mch_free_time[:, :] = machine_free
    env.mch_free_time[:, :] = machine_free / scale
    env.true_candidate_free_time[:, :] = ready
    env.candidate_free_time[:, :] = ready / scale
    return env, env.rebuild_state_from_current()


def calculate_stability_rates(env, flip_counts, machine_change_counts):
    """Return per-environment pair-flip and machine-change rates."""
    import numpy as np

    ref_mask = np.asarray(env.stability_ref_mask, dtype=bool)
    old_machine = np.asarray(env.stability_old_machine, dtype=np.int32)
    assigned_machine = np.asarray(env.op_assigned_mch, dtype=np.int32)
    common_old_ops = np.sum(ref_mask, axis=1).astype(np.float64)
    comparable_pairs = np.zeros(env.number_of_envs, dtype=np.float64)

    for env_idx in range(env.number_of_envs):
        for machine_idx in range(env.number_of_machines):
            comparable = (
                ref_mask[env_idx]
                & (old_machine[env_idx] == machine_idx)
                & (assigned_machine[env_idx] == machine_idx)
            )
            count = float(np.sum(comparable))
            comparable_pairs[env_idx] += count * max(count - 1.0, 0.0) / 2.0

    flip_rates = np.divide(
        np.asarray(flip_counts, dtype=np.float64),
        comparable_pairs,
        out=np.zeros_like(comparable_pairs),
        where=comparable_pairs > 0,
    )
    machine_change_rates = np.divide(
        np.asarray(machine_change_counts, dtype=np.float64),
        common_old_ops,
        out=np.zeros_like(common_old_ops),
        where=common_old_ops > 0,
    )
    return flip_rates, machine_change_rates


class StabilityMemory:
    """LLMemory with done-aware GAE for variable operation counts."""

    def __init__(self, gamma, gae_lambda, ll_memory_cls):
        self._base = ll_memory_cls(gamma, gae_lambda)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name == "_base":
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)

    def get_gae_advantages(self, normalize_vtarget=False):
        import torch

        reward_arr = torch.stack(self.reward_seq, dim=0)
        done_arr = torch.stack(self.done_seq, dim=0).to(dtype=reward_arr.dtype)
        values = self.t_old_val_seq.transpose(0, 1)
        len_trajectory = reward_arr.shape[0]
        advantage = torch.zeros(reward_arr.shape[1], device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                next_value = torch.zeros_like(values[i])
            else:
                next_value = values[i + 1]
            not_done = 1.0 - done_arr[i]
            delta = reward_arr[i] + self.gamma * not_done * next_value - values[i]
            advantage = delta + self.gamma * self.gae_lambda * not_done * advantage
            advantage_seq.insert(0, advantage)

        advantages = torch.stack(advantage_seq, dim=0).transpose(0, 1).float()
        v_targets = advantages + self.t_old_val_seq
        if normalize_vtarget:
            v_targets = (v_targets - v_targets.mean(dim=1, keepdim=True)) / (
                v_targets.std(dim=1, keepdim=True) + 1e-8
            )
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (
            advantages.std(dim=1, keepdim=True) + 1e-8
        )
        return advantages.flatten(0, 1), v_targets.flatten(0, 1)


def safe_state_for_active_rows(state, done):
    """Avoid forwarding all-masked completed environments through the policy."""
    import torch
    import numpy as np

    done = np.asarray(done, dtype=bool)
    active = np.where(~done)[0]
    fields = (
        state.fea_j_tensor,
        state.op_mask_tensor,
        state.candidate_tensor,
        state.fea_m_tensor,
        state.mch_mask_tensor,
        state.comp_idx_tensor,
        state.dynamic_pair_mask_tensor,
        state.fea_pairs_tensor,
    )
    if not done.any() or not len(active):
        return fields
    source = int(active[0])
    safe = []
    for field in fields:
        copied = field.clone()
        copied[torch.as_tensor(done, device=field.device)] = field[source]
        safe.append(copied)
    return tuple(safe)


def collect_rollout(ppo, env, ll_memory_cls):
    import numpy as np
    import torch
    from common_utils import sample_action
    from params import configs as active_configs

    memory = StabilityMemory(ppo.gamma, ppo.gae_lambda, ll_memory_cls)
    state = env.state
    done = np.zeros(env.number_of_envs, dtype=bool)
    total_reward = np.zeros(env.number_of_envs, dtype=np.float64)
    component_totals = {
        "mk": np.zeros(env.number_of_envs, dtype=np.float64),
        "td": np.zeros(env.number_of_envs, dtype=np.float64),
        "od": np.zeros(env.number_of_envs, dtype=np.float64),
        "wait_od": np.zeros(env.number_of_envs, dtype=np.float64),
        "flip_stability": np.zeros(env.number_of_envs, dtype=np.float64),
        "machine_change_stability": np.zeros(env.number_of_envs, dtype=np.float64),
    }
    component_abs_sums = {name: 0.0 for name in component_totals}
    component_nonzero = {"od": 0, "wait_od": 0}
    component_samples = {"od": 0, "wait_od": 0}
    total_flips = np.zeros(env.number_of_envs, dtype=np.float64)
    total_machine_changes = np.zeros(env.number_of_envs, dtype=np.float64)
    flip_coef = float(getattr(active_configs, "ll_stability_flip_coef", 1.0))
    machine_change_coef = float(
        getattr(active_configs, "ll_stability_machine_change_coef", 3.0)
    )
    reward_divisor = max(float(getattr(active_configs, "ll_reward_divisor", 10.0)), 1e-8)

    while not bool(done.all()):
        memory.push(state)
        model_state = safe_state_for_active_rows(state, done)
        with torch.no_grad():
            pi, values = ppo.policy_old(*model_state)
            actions, log_probs = sample_action(pi)
        actions = actions.reshape(-1)
        log_probs = log_probs.reshape(-1)
        values = values.reshape(-1)

        done_tensor = torch.as_tensor(done, device=actions.device, dtype=torch.bool)
        actions = torch.where(done_tensor, torch.zeros_like(actions), actions)
        log_probs = torch.where(done_tensor, torch.zeros_like(log_probs), log_probs)
        values = torch.where(done_tensor, torch.zeros_like(values), values)

        next_state, reward, done_flag, info = env.step(actions.detach().cpu().numpy())
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=actions.device)
        done_after = np.asarray(done_flag, dtype=bool)
        memory.action_seq.append(actions.detach())
        memory.log_probs.append(log_probs.detach())
        memory.val_seq.append(values.detach())
        memory.reward_seq.append(reward_tensor)
        memory.done_seq.append(torch.as_tensor(done_after, device=actions.device, dtype=torch.bool))

        total_reward += np.asarray(reward, dtype=np.float64)
        step_components = {
            "mk": np.asarray(info.get("reward_mk_step", np.zeros(env.number_of_envs)), dtype=np.float64),
            "td": np.asarray(info.get("reward_td_step", np.zeros(env.number_of_envs)), dtype=np.float64),
            "od": np.asarray(info.get("reward_od_step", np.zeros(env.number_of_envs)), dtype=np.float64),
            "wait_od": np.asarray(info.get("reward_wait_od_step", np.zeros(env.number_of_envs)), dtype=np.float64),
            "flip_stability": -flip_coef * np.asarray(
                info["stability_flip_increment"], dtype=np.float64
            ) / reward_divisor,
            "machine_change_stability": -machine_change_coef * np.asarray(
                info["stability_machine_change_increment"], dtype=np.float64
            ) / reward_divisor,
        }
        for name, values in step_components.items():
            values = values.reshape(-1)
            component_totals[name] += values
            component_abs_sums[name] += float(np.sum(np.abs(values)))
            if name in component_nonzero:
                component_nonzero[name] += int(np.count_nonzero(np.abs(values) > 1e-12))
                component_samples[name] += int(values.size)
        total_flips += np.asarray(info["stability_flip_increment"], dtype=np.float64)
        total_machine_changes += np.asarray(info["stability_machine_change_increment"], dtype=np.float64)
        state = next_state
        done = done_after

    abs_total = max(float(sum(component_abs_sums.values())), 1e-12)
    component_means = {
        name: float(values.mean()) for name, values in component_totals.items()
    }
    component_shares = {
        name: float(component_abs_sums[name] / abs_total) for name in component_totals
    }
    component_hits = {
        name: float(component_nonzero[name] / max(component_samples[name], 1))
        for name in component_nonzero
    }
    flip_rates, machine_change_rates = calculate_stability_rates(
        env, total_flips, total_machine_changes
    )

    return memory, {
        "reward": float(total_reward.mean()),
        "mk_reward": component_means["mk"],
        "td_reward": component_means["td"],
        "od_reward": component_means["od"],
        "wait_od_reward": component_means["wait_od"],
        "flip_stability_reward": component_means["flip_stability"],
        "machine_change_stability_reward": component_means["machine_change_stability"],
        "mk_share": component_shares["mk"],
        "td_share": component_shares["td"],
        "od_share": component_shares["od"],
        "wait_od_share": component_shares["wait_od"],
        "flip_stability_share": component_shares["flip_stability"],
        "machine_change_stability_share": component_shares["machine_change_stability"],
        "od_hit": component_hits["od"],
        "wait_od_hit": component_hits["wait_od"],
        "flip_count": float(total_flips.mean()),
        "machine_change_count": float(total_machine_changes.mean()),
        "flip_rate": float(flip_rates.mean()),
        "machine_change_rate": float(machine_change_rates.mean()),
        "train_makespan": float(np.mean(env.current_makespan)),
        "train_tardiness": float(np.mean(env.accumulated_tardiness)),
        "steps": len(memory.reward_seq),
    }


def evaluate_batch(ppo, configured, samples):
    import numpy as np
    import torch

    env, state = build_env(configured, samples, stability_enabled=True)
    number_of_envs = len(samples)
    total_stability = np.zeros(number_of_envs, dtype=np.float64)
    total_flips = np.zeros(number_of_envs, dtype=np.float64)
    total_machine_changes = np.zeros(number_of_envs, dtype=np.float64)
    done = np.zeros(number_of_envs, dtype=bool)
    ppo.policy.eval()
    while not bool(done.all()):
        model_state = safe_state_for_active_rows(state, done)
        with torch.no_grad():
            pi = ppo.policy.policy_only(
                *model_state,
            )
            action = torch.argmax(pi, dim=1)
            action[torch.as_tensor(done, device=action.device)] = 0
        state, _, done_flag, info = env.step(action.cpu().numpy())
        done = np.asarray(done_flag, dtype=bool)
        total_stability += np.asarray(info["reward_stability_step"], dtype=np.float64)
        total_flips += np.asarray(info["stability_flip_increment"], dtype=np.float64)
        total_machine_changes += np.asarray(
            info["stability_machine_change_increment"], dtype=np.float64
        )

    makespans = np.zeros(number_of_envs, dtype=np.float64)
    tardiness = np.zeros(number_of_envs, dtype=np.float64)
    for env_idx in range(number_of_envs):
        op_count = int(env.env_number_of_ops[env_idx])
        makespans[env_idx] = float(np.max(env.true_op_ct[env_idx, :op_count]))
        for job_idx in range(env.number_of_jobs):
            last_op = int(env.job_last_op_id[env_idx, job_idx])
            tardiness[env_idx] += max(
                0.0,
                float(env.true_op_ct[env_idx, last_op])
                - float(env.true_due_date[env_idx, job_idx]),
            )

    flip_rates, machine_change_rates = calculate_stability_rates(
        env, total_flips, total_machine_changes
    )
    ppo.policy.train()
    return [
        {
            "makespan": float(makespans[idx]),
            "tardiness": float(tardiness[idx]),
            "objective": float(0.5 * makespans[idx] + 0.5 * tardiness[idx]),
            "stability_reward": float(total_stability[idx]),
            "flip_count": float(total_flips[idx]),
            "machine_change_count": float(total_machine_changes[idx]),
            "flip_rate": float(flip_rates[idx]),
            "machine_change_rate": float(machine_change_rates[idx]),
        }
        for idx in range(number_of_envs)
    ]


def build_validation_suite(configured, reference_ppo):
    """Build 3 vectorized batches with 9 logged due-date subgroups."""
    target_sizes = (30, 35, 40)
    due_modes = ("range3_loose", "range3_mixed", "range3_tight")
    suite = []
    group_index = 0
    for target_jobs in target_sizes:
        batch_samples = []
        groups = []
        for due_mode in due_modes:
            samples = generate_batch(
                configured,
                reference_ppo,
                sample_index=900000 + group_index,
                target_jobs=target_jobs,
                pattern=[False],
                due_mode=due_mode,
                num_envs=5,
                allow_overdue=False,
            )
            start = len(batch_samples)
            batch_samples.extend(samples)
            groups.append(
                {
                    "due_mode": due_mode,
                    "start": start,
                    "end": len(batch_samples),
                }
            )
            group_index += 1
        suite.append(
            {
                "target_jobs": target_jobs,
                "samples": batch_samples,
                "groups": groups,
            }
        )
    return suite


def save_line(path, line):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main():
    import numpy as np
    import torch
    from tqdm import tqdm

    local_args = parse_local_args()
    configs, config_path = load_project_config(local_args.config)
    if local_args.max_updates is not None:
        configs.ll_max_updates = int(local_args.max_updates)
    if local_args.num_envs is not None:
        configs.ll_num_envs = int(local_args.num_envs)

    # The new trainable policy consumes the confirmed stability state.
    configs.fea_j_input_dim = 22
    configs.fea_pair_input_dim = 12
    configs.ll_stability_enable = True
    configs.ll_stability_critic_summary = True
    configs.ll_reward_divisor = float(getattr(configs, "ll_reward_divisor", 10.0))
    set_seed(int(getattr(configs, "seed_train", 3)))

    from model.ll_ppo import LLPPO, LLMemory
    from common_utils import lower_level_log_dir

    train_config = clone_policy_config(configs, stability=True)
    reference_config = clone_policy_config(configs, stability=False)
    reference_ppo = LLPPO(reference_config)
    reference_path = load_policy_checkpoint(
        reference_ppo,
        getattr(configs, "ll_stability_reference_model_path"),
        torch,
    )

    train_ppo = LLPPO(train_config)
    loaded_path = load_policy_checkpoint(train_ppo, reference_path, torch)

    output_path = Path(getattr(configs, "ll_stability_output_model_path"))
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = output_path.stem
    log_dir = ROOT / lower_level_log_dir()
    reward_log = log_dir / f"reward_{model_name}.txt"
    detail_log = log_dir / f"detailed_reward_{model_name}.txt"
    reward_log.write_text("", encoding="utf-8")
    detail_log.write_text("", encoding="utf-8")

    start_clock = time.perf_counter()
    start_time = dt.datetime.now().isoformat(timespec="seconds")
    validation_suite = None
    size_hold, due_hold, due_modes = stability_hold_schedule(configs)
    max_updates = int(getattr(configs, "ll_max_updates", 1))
    validate_every = max(1, int(getattr(configs, "validate_timestep", 10)))
    cached_stage = None
    cached_samples = None
    target_jobs = None

    print(f"[CONFIG] {config_path}")
    print(f"[REFERENCE] {reference_path}")
    print(f"[TRAIN OUTPUT] {output_path}")
    print(
        f"[STABILITY] fresh={getattr(configs, 'll_stability_fresh_ratio', 0.3):.2f} "
        f"reschedule={getattr(configs, 'll_stability_reschedule_ratio', 0.7):.2f} "
        f"flip={getattr(configs, 'll_stability_flip_coef', 1.0):.2f} "
        f"machine_change={getattr(configs, 'll_stability_machine_change_coef', 3.0):.2f} "
        f"divisor={getattr(configs, 'll_reward_divisor', 10.0):.2f}"
    )

    progress = tqdm(
        range(max_updates),
        desc="LL Stability PPO",
        unit="upd",
        dynamic_ncols=True,
    )
    for update in progress:
        size_cycle = update // size_hold
        due_slot = (update % size_hold) // due_hold
        stage = (size_cycle, due_slot)
        if stage != cached_stage:
            if cached_stage is None or size_cycle != cached_stage[0]:
                target_rng = np.random.default_rng(
                    int(configs.seed_train) + int(size_cycle) * 100003
                )
                low = int(getattr(configs, "ll_stability_target_jobs_low", 30))
                high = int(getattr(configs, "ll_stability_target_jobs_high", 40))
                target_jobs = int(target_rng.integers(min(low, high), max(low, high) + 1))

            sample_index = size_cycle * len(due_modes) + due_slot
            due_mode = due_modes[due_slot]
            cached_samples = generate_batch(
                configs,
                reference_ppo,
                sample_index=sample_index,
                target_jobs=target_jobs,
                due_mode=due_mode,
            )
            cached_stage = stage
            kinds = {kind: sum(sample["kind"] == kind for sample in cached_samples) for kind in ("fresh", "reschedule")}
            progress.write(
                f"[SAMPLE] update={update + 1} size_cycle={size_cycle + 1} "
                f"due={due_mode} ({due_slot + 1}/3) "
                f"target_jobs={target_jobs} fresh={kinds['fresh']} "
                f"reschedule={kinds['reschedule']} "
                f"hold={due_hold}/{size_hold}"
            )

        env, _ = build_env(configs, cached_samples, stability_enabled=True)
        memory, rollout_stats = collect_rollout(train_ppo, env, LLMemory)
        # Pass the wrapper so LLPPO.update uses done-aware GAE.
        loss, v_loss, _ = train_ppo.update(memory)
        memory.clear_memory()

        log_line = (
            f"Update {update + 1}/{max_updates} | R: {rollout_stats['reward']: .4f} | "
            f"Loss: {float(loss): .4f} | V-Loss: {float(v_loss): .4f} | "
            f"MK_r: {rollout_stats['mk_share'] * 100:4.1f}% | "
            f"TD_r: {rollout_stats['td_share'] * 100:4.1f}% | "
            f"OD_r: {rollout_stats['od_share'] * 100:4.1f}% | "
            f"FlipR: {rollout_stats['flip_stability_share'] * 100:4.1f}% | "
            f"MChgR: {rollout_stats['machine_change_stability_share'] * 100:4.1f}% | "
            f"Train MK: {rollout_stats['train_makespan']: .1f} | "
            f"Train TD: {rollout_stats['train_tardiness']: .1f} | "
            f"Flip: {rollout_stats['flip_count']: .2f} "
            f"({rollout_stats['flip_rate'] * 100:4.1f}%) | "
            f"MChg: {rollout_stats['machine_change_count']: .2f} "
            f"({rollout_stats['machine_change_rate'] * 100:4.1f}%) | "
            f"Steps: {rollout_stats['steps']}"
        )
        progress.write(log_line)
        progress.set_postfix(
            due=due_mode,
            jobs=target_jobs,
            R=f"{rollout_stats['reward']:.2f}",
        )
        save_line(reward_log, log_line)
        save_line(
            detail_log,
            f"update={update + 1},size_cycle={size_cycle + 1},due_mode={due_mode},"
            f"target_jobs={target_jobs},fresh_ratio="
            f"{getattr(configs, 'll_stability_fresh_ratio', 0.3):.4f},"
            f"reward={rollout_stats['reward']:.8f},"
            f"mk_share={rollout_stats['mk_share']:.8f},"
            f"td_share={rollout_stats['td_share']:.8f},"
            f"od_share={rollout_stats['od_share']:.8f},"
            f"wait_od_share={rollout_stats['wait_od_share']:.8f},"
            f"flip_stability_share={rollout_stats['flip_stability_share']:.8f},"
            f"machine_change_stability_share="
            f"{rollout_stats['machine_change_stability_share']:.8f},"
            f"flip_count={rollout_stats['flip_count']:.8f},"
            f"flip_rate={rollout_stats['flip_rate']:.8f},"
            f"machine_change_count={rollout_stats['machine_change_count']:.8f},"
            f"machine_change_rate={rollout_stats['machine_change_rate']:.8f},"
            f"train_makespan={rollout_stats['train_makespan']:.8f},"
            f"train_tardiness={rollout_stats['train_tardiness']:.8f}"
        )

        if not local_args.skip_validation and (update + 1) % validate_every == 0:
            if validation_suite is None:
                progress.write("[VAL] building 3 batches x 15 reschedule instances")
                validation_suite = build_validation_suite(configs, reference_ppo)

            val_results = []
            for batch in validation_suite:
                batch_results = evaluate_batch(train_ppo, configs, batch["samples"])
                val_results.extend(batch_results)
                for subgroup in batch["groups"]:
                    group_results = batch_results[subgroup["start"] : subgroup["end"]]
                    group_mk = float(np.mean([result["makespan"] for result in group_results]))
                    group_td = float(np.mean([result["tardiness"] for result in group_results]))
                    group_obj = 0.5 * group_mk + 0.5 * group_td
                    group_flip = float(np.mean([result["flip_count"] for result in group_results]))
                    group_mchg = float(
                        np.mean([result["machine_change_count"] for result in group_results])
                    )
                    group_flip_rate = float(
                        np.mean([result["flip_rate"] for result in group_results])
                    )
                    group_mchg_rate = float(
                        np.mean([result["machine_change_rate"] for result in group_results])
                    )
                    save_line(
                        detail_log,
                        f"validation_group=jobs_{batch['target_jobs']}_{subgroup['due_mode']},"
                        f"update={update + 1},n=5,mk={group_mk:.8f},td={group_td:.8f},"
                        f"obj={group_obj:.8f},flip={group_flip:.8f},"
                        f"flip_rate={group_flip_rate:.8f},"
                        f"machine_change={group_mchg:.8f},"
                        f"machine_change_rate={group_mchg_rate:.8f}",
                    )

            val_mk = float(np.mean([result["makespan"] for result in val_results]))
            val_td = float(np.mean([result["tardiness"] for result in val_results]))
            val_obj = 0.5 * val_mk + 0.5 * val_td
            val_flip = float(np.mean([r["flip_count"] for r in val_results]))
            val_flip_rate = float(np.mean([r["flip_rate"] for r in val_results]))
            val_mchg = float(np.mean([r["machine_change_count"] for r in val_results]))
            val_mchg_rate = float(
                np.mean([r["machine_change_rate"] for r in val_results])
            )
            val_line = (
                f"Vali n=45 Reschedule | MK: {val_mk:.3f} | TD: {val_td:.3f} | "
                f"Vali Obj: {val_obj:.3f} | Vali Flip: {val_flip:.3f} "
                f"({val_flip_rate * 100:.1f}%) | Vali MChg: {val_mchg:.3f} "
                f"({val_mchg_rate * 100:.1f}%)"
            )
            progress.write(val_line)
            save_line(reward_log, val_line)

    # Keep the final policy, not the validation-best policy.
    torch.save(train_ppo.policy.state_dict(), str(output_path))
    print(f"[CHECKPOINT] saved final update={max_updates} -> {output_path}")

    elapsed = time.perf_counter() - start_clock
    end_time = dt.datetime.now().isoformat(timespec="seconds")
    train_time_path = ROOT / "train_time.txt"
    save_line(
        train_time_path,
        "\n" + "=" * 80 + "\n"
        + f"model_name: {model_name}\n"
        + f"script: train_ll_stability_finetune.py\n"
        + f"config: {config_path}\n"
        + f"reference_checkpoint: {loaded_path}\n"
        + f"start_time: {start_time}\n"
        + f"end_time: {end_time}\n"
        + f"train_seconds: {elapsed:.3f}\n"
        + f"train_minutes: {elapsed / 60.0:.3f}\n"
        + f"train_hours: {elapsed / 3600.0:.3f}\n"
    )
    print(f"[DONE] elapsed={elapsed:.3f}s")

    # Auto-generate training and comparison plots
    try:
        from plot_train import plot_stability_finetune_results, plot_stability_models_comparison
        plot_stability_finetune_results(model_name)
        plot_stability_models_comparison()
    except Exception as e:
        print(f"[PLOT] Warning: Failed to auto-generate plots: {e}")


if __name__ == "__main__":
    main()
