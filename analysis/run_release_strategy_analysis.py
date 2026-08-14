"""Run the event-40 release strategy analysis.

The script runs the same seed under four high-level policies:
  - ppo
  - cadence1
  - cadence5
  - slack0

For each scenario and seed it writes one CSV containing checkpoints at
events 0, 40, 80, 120, and 160.  CSV columns intentionally follow
analysis/release_strategy_analysis_spec.md.
"""

from __future__ import annotations

import copy
import csv
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT,
    "yaml_config",
    "eval_baseline_seed1_greedy_cadence1_1run.yml",
)

SCENARIOS = ("baseline", "urgent", "burst_cluster")
SEEDS = tuple(range(1, 11))
STRATEGIES = ("ppo", "cadence5", "slack0")
CHECKPOINTS = (0, 40, 80, 120, 160)

# Each scenario uses the high-level policy trained for that scenario.
SCENARIO_MODEL_PATHS = {
    "baseline": os.path.join(
        PROJECT_ROOT,
        "trained_weights",
        "high_level",
        "hlgate_baseline_stab05_e16_newstate.pth",
    ),
    "urgent": os.path.join(
        PROJECT_ROOT,
        "trained_weights",
        "high_level",
        "hlgate_urgent_stab05_e16_newstate.pth",
    ),
    "burst_cluster": os.path.join(
        PROJECT_ROOT,
        "trained_weights",
        "high_level",
        "hlgate_multijob_stab05_e16_newstate.pth",
    ),
}

# The short CSV names and their required order are defined by the analysis MD.
CSV_COLUMNS = [
    "strat",
    "seed",
    "event",
    "time",
    "mk",
    "td",
    "obj",
    "b_jobs",
    "b_work",
    "b_smin",
    "b_smean",
    "b_sq25",
    "b_sstd",
    "b_neg_n",
    "b_neg_r",
    "w_jobs",
    "w_ops",
    "w_work",
    "w_td",
    "w_smin",
    "w_smean",
    "w_sq25",
    "w_sstd",
    "m_lmean",
    "m_lstd",
    "m_lmin",
    "m_lmax",
    "m_imb",
    "rn_mk",
    "rn_td",
    "rn_obj",
    "arr_jobs",
    "urg_jobs",
    "in_smin",
    "in_sq25",
    "in_smean",
    "rel_n",
]

EVENT_COLUMNS = [
    "strat",
    "seed",
    "event",
    "time",
    "action",
    "released",
    "td_before",
    "td",
    "b_jobs",
    "b_ops",
    "b_work",
    "b_smin",
    "b_smean",
    "b_sq25",
    "b_sstd",
    "b_neg_sum",
    "b_lt0_n",
    "b_lt100_n",
    "b_lt0_r",
    "b_lt100_r",
    "b_critical_work",
    "b_near_critical_work",
    "w_jobs",
    "w_ops",
    "w_work",
    "w_smin",
    "w_smean",
    "w_sq25",
    "w_sstd",
    "w_neg_sum",
    "w_lt0_n",
    "w_lt100_n",
    "w_lt0_r",
    "w_lt100_r",
    "w_critical_work",
    "w_near_critical_work",
    "bw_smin",
    "bw_smean",
    "bw_sq25",
    "bw_sstd",
    "bw_jobs",
    "bw_neg_sum",
    "bw_lt0_n",
    "bw_lt100_n",
    "bw_lt0_r",
    "bw_lt100_r",
    "bw_critical_work",
    "bw_near_critical_work",
    "m_lmax",
]

PLOT_STRATEGIES = ("ppo", "cadence5", "slack0")

PLOT_METRIC_SPECS = (
    (
        "slack_lt0_count",
        {"b": "b_lt0_n", "w": "w_lt0_n", "bw": "bw_lt0_n"},
    ),
    (
        "slack_lt100_count",
        {"b": "b_lt100_n", "w": "w_lt100_n", "bw": "bw_lt100_n"},
    ),
    (
        "negative_slack_burden",
        {"b": "b_neg_sum", "w": "w_neg_sum", "bw": "bw_neg_sum"},
    ),
    (
        "critical_work_lt0",
        {
            "b": "b_critical_work",
            "w": "w_critical_work",
            "bw": "bw_critical_work",
        },
    ),
    (
        "near_critical_work_lt100",
        {
            "b": "b_near_critical_work",
            "w": "w_near_critical_work",
            "bw": "bw_near_critical_work",
        },
    ),
)


# Parse the optional config before importing params.py.  params.py parses
# sys.argv itself, so it must only receive its own --config argument.
_raw_args = list(sys.argv[1:])
if "--config" in _raw_args:
    _config_idx = _raw_args.index("--config")
    if _config_idx + 1 >= len(_raw_args):
        raise SystemExit("--config requires a YAML path")
    CONFIG_PATH = os.path.abspath(_raw_args[_config_idx + 1])
else:
    CONFIG_PATH = DEFAULT_CONFIG

if "--seed" in _raw_args:
    _seed_idx = _raw_args.index("--seed")
    if _seed_idx + 1 >= len(_raw_args):
        raise SystemExit("--seed requires an integer")
    try:
        _requested_seed = int(_raw_args[_seed_idx + 1])
    except ValueError as exc:
        raise SystemExit("--seed requires an integer") from exc
    if _requested_seed < 1:
        raise SystemExit("--seed must be >= 1")
    SELECTED_SEEDS = (_requested_seed,)
else:
    SELECTED_SEEDS = SEEDS

if "--output-root" in _raw_args:
    _output_idx = _raw_args.index("--output-root")
    if _output_idx + 1 >= len(_raw_args):
        raise SystemExit("--output-root requires a directory")
    OUTPUT_ROOT = os.path.abspath(_raw_args[_output_idx + 1])
else:
    OUTPUT_ROOT = None

if "--replot-root" in _raw_args:
    _replot_idx = _raw_args.index("--replot-root")
    if _replot_idx + 1 >= len(_raw_args):
        raise SystemExit("--replot-root requires an existing analysis directory")
    REPLOT_ROOT = os.path.abspath(_raw_args[_replot_idx + 1])
else:
    REPLOT_ROOT = None

sys.argv = [sys.argv[0], "--config", CONFIG_PATH]

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common_utils import resolve_high_level_weight_path, setup_seed  # noqa: E402
from data_utils import SD2_instance_generator  # noqa: E402
from dynamic_job_stream import register_initial_jobs, sample_initial_jobs  # noqa: E402
from hl_env_scenarios import make_burst_sampler, resolve_hl_env_scenario, scenario_config  # noqa: E402
from hl_gate_env import HLGateEnv  # noqa: E402
from hrl_orchestrator import EventBurstGenerator, GlobalTimelineOrchestrator  # noqa: E402
from model.hl_gate_state import HL_LL_BUFFER_EMBED_DIM, get_hl_gate_state_dim  # noqa: E402
from model.hl_ppo_gate_model import HLPPOGateNet  # noqa: E402
from params import configs  # noqa: E402


def _job_work(job) -> float:
    value = float(getattr(job, "meta", {}).get("total_proc_time", 0.0))
    if value > 0.0:
        return value
    return float(
        sum(float(getattr(op, "avg_proc_time", 0.0)) for op in getattr(job, "operations", []) or [])
    )


def _job_slack(job, all_due: Mapping[int, float], now: float) -> float:
    return float(all_due.get(int(job.job_id), 0.0)) - float(now) - _job_work(job)


def _job_map(orch) -> Dict[int, object]:
    result: Dict[int, object] = {}
    sources = [
        getattr(orch, "buffer", []),
        getattr(orch, "_last_jobs_snapshot", []),
        getattr(orch, "_committed_jobs", []),
    ]
    for source in sources:
        for job in source or []:
            result[int(job.job_id)] = job
    return result


def _slack_summary(values: Sequence[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.min()), float(arr.mean()), float(np.percentile(arr, 25)), float(arr.std())


def _wip_details(orch, all_due: Mapping[int, float], now: float) -> Dict[str, float]:
    """Mirror GlobalTimelineOrchestrator.get_wip_stats and add op/q25 counts."""
    rows_by_job: Dict[int, List[dict]] = {}
    for row in getattr(orch, "_last_full_rows", []) or []:
        rows_by_job.setdefault(int(row["job"]), []).append(row)

    slack_values: List[float] = []
    wip_jobs = 0
    wip_ops = 0
    total_work = 0.0
    planned_td = 0.0
    neg_slack_sum = 0.0
    slack_lt0_count = 0
    slack_lt100_count = 0
    critical_work = 0.0
    near_critical_work = 0.0

    for job in getattr(orch, "_last_jobs_snapshot", []) or []:
        jid = int(job.job_id)
        rows = rows_by_job.get(jid, [])
        if rows:
            planned_finish = max(float(row["end"]) for row in rows)
            if planned_finish <= float(now):
                continue
        rem_work = _job_work(job)
        wip_jobs += 1
        wip_ops += len(getattr(job, "operations", []) or [])
        total_work += rem_work
        due = float(all_due.get(jid, 0.0))
        slack = due - float(now) - rem_work
        slack_values.append(slack)
        if slack < 0.0:
            neg_slack_sum += -slack
            slack_lt0_count += 1
            critical_work += rem_work
        if slack < 100.0:
            slack_lt100_count += 1
            near_critical_work += rem_work
        if rows:
            planned_td += max(0.0, planned_finish - due)

    smin, smean, sq25, sstd = _slack_summary(slack_values)
    return {
        "jobs": float(wip_jobs),
        "ops": float(wip_ops),
        "work": float(total_work),
        "td": float(planned_td),
        "neg_sum": float(neg_slack_sum),
        "lt0_count": float(slack_lt0_count),
        "lt100_count": float(slack_lt100_count),
        "critical_work": float(critical_work),
        "near_critical_work": float(near_critical_work),
        "smin": smin,
        "smean": smean,
        "sq25": sq25,
        "sstd": sstd,
        "slacks": slack_values,
    }


def collect_metrics(env: HLGateEnv, *, orch=None, now: Optional[float] = None) -> Dict[str, float]:
    """Collect the checkpoint fields from the current schedule state."""
    orch = env.orch if orch is None else orch
    now = float(env.t_now if now is None else now)
    all_due = env.all_job_due_dates

    buffer_jobs = list(getattr(orch, "buffer", []) or [])
    buffer_slacks = [_job_slack(job, all_due, now) for job in buffer_jobs]
    b_smin, b_smean, b_sq25, b_sstd = _slack_summary(buffer_slacks)
    b_neg_n = sum(1 for value in buffer_slacks if value < 0.0)
    b_lt100_n = sum(1 for value in buffer_slacks if value < 100.0)
    b_jobs = len(buffer_jobs)
    b_ops = sum(len(getattr(job, "operations", []) or []) for job in buffer_jobs)
    b_neg_sum = float(sum(-value for value in buffer_slacks if value < 0.0))
    b_critical_work = float(
        sum(_job_work(job) for job, slack in zip(buffer_jobs, buffer_slacks) if slack < 0.0)
    )
    b_near_critical_work = float(
        sum(_job_work(job) for job, slack in zip(buffer_jobs, buffer_slacks) if slack < 100.0)
    )

    machine_free = np.asarray(getattr(orch, "machine_free_time", []), dtype=float)
    machine_load = np.maximum(0.0, machine_free - now)
    if machine_load.size:
        m_lmean = float(machine_load.mean())
        m_lstd = float(machine_load.std())
        m_lmin = float(machine_load.min())
        m_lmax = float(machine_load.max())
        m_imb = float(m_lmax - m_lmin)
        current_mk = float(machine_free.max())
    else:
        m_lmean = m_lstd = m_lmin = m_lmax = m_imb = current_mk = 0.0

    wip = _wip_details(orch, all_due, now)
    bw_smin, bw_smean, bw_sq25, bw_sstd = _slack_summary(
        buffer_slacks + list(wip["slacks"])
    )
    bw_jobs = b_jobs + float(wip["jobs"])
    bw_neg_sum = b_neg_sum + float(wip["neg_sum"])
    bw_lt0_n = float(b_neg_n) + float(wip["lt0_count"])
    bw_lt100_n = float(b_lt100_n) + float(wip["lt100_count"])
    bw_critical_work = b_critical_work + float(wip["critical_work"])
    bw_near_critical_work = b_near_critical_work + float(wip["near_critical_work"])
    current_td = float(orch.get_total_tardiness_estimate(all_due))
    current_obj = 0.5 * current_mk + 0.5 * current_td
    return {
        "mk": current_mk,
        "td": current_td,
        "obj": current_obj,
        "b_jobs": float(b_jobs),
        "b_ops": float(b_ops),
        "b_work": float(sum(_job_work(job) for job in buffer_jobs)),
        "b_smin": b_smin,
        "b_smean": b_smean,
        "b_sq25": b_sq25,
        "b_sstd": b_sstd,
        "b_neg_n": float(b_neg_n),
        "b_neg_r": float(b_neg_n / b_jobs) if b_jobs else 0.0,
        "b_lt100_n": float(b_lt100_n),
        "b_lt0_r": float(b_neg_n / b_jobs) if b_jobs else 0.0,
        "b_lt100_r": float(b_lt100_n / b_jobs) if b_jobs else 0.0,
        "b_neg_sum": b_neg_sum,
        "b_critical_work": b_critical_work,
        "b_near_critical_work": b_near_critical_work,
        "w_jobs": wip["jobs"],
        "w_ops": wip["ops"],
        "w_work": wip["work"],
        "w_td": wip["td"],
        "w_neg_sum": wip["neg_sum"],
        "w_lt0_n": wip["lt0_count"],
        "w_lt100_n": wip["lt100_count"],
        "w_lt0_r": float(wip["lt0_count"] / wip["jobs"]) if wip["jobs"] else 0.0,
        "w_lt100_r": float(wip["lt100_count"] / wip["jobs"]) if wip["jobs"] else 0.0,
        "w_critical_work": wip["critical_work"],
        "w_near_critical_work": wip["near_critical_work"],
        "w_smin": wip["smin"],
        "w_smean": wip["smean"],
        "w_sq25": wip["sq25"],
        "w_sstd": wip["sstd"],
        "m_lmean": m_lmean,
        "m_lstd": m_lstd,
        "m_lmin": m_lmin,
        "m_lmax": m_lmax,
        "m_imb": m_imb,
        "bw_smin": bw_smin,
        "bw_smean": bw_smean,
        "bw_sq25": bw_sq25,
        "bw_sstd": bw_sstd,
        "bw_jobs": bw_jobs,
        "bw_neg_sum": bw_neg_sum,
        "bw_lt0_n": bw_lt0_n,
        "bw_lt100_n": bw_lt100_n,
        "bw_lt0_r": float(bw_lt0_n / bw_jobs) if bw_jobs else 0.0,
        "bw_lt100_r": float(bw_lt100_n / bw_jobs) if bw_jobs else 0.0,
        "bw_critical_work": bw_critical_work,
        "bw_near_critical_work": bw_near_critical_work,
    }


def clone_orchestrator(orch):
    """Clone scheduler state while sharing the loaded lower-level PPO model."""
    ll_ppo = getattr(orch, "_ppo", None)
    setattr(orch, "_ppo", None)
    try:
        cloned = copy.deepcopy(orch)
    finally:
        setattr(orch, "_ppo", ll_ppo)
    setattr(cloned, "_ppo", ll_ppo)
    return cloned


def release_now_metrics(env: HLGateEnv, now: float) -> Dict[str, float]:
    """Evaluate a hypothetical release at the checkpoint without changing env."""
    probe_orch = clone_orchestrator(env.orch)
    probe_orch.event_release_and_reschedule(float(now))
    return collect_metrics(env, orch=probe_orch, now=now)


def setup_analysis_env(scenario: str, seed: int) -> Tuple[HLGateEnv, List[object]]:
    """Create the same initial state as HLGateEnv.reset, but expose event 0."""
    setup_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    resolved = resolve_hl_env_scenario(configs, rng)
    if resolved != scenario:
        raise RuntimeError(f"Scenario resolution mismatch: requested={scenario}, resolved={resolved}")
    base_cfg = scenario_config(configs, scenario)
    generator = EventBurstGenerator(
        SD2_instance_generator,
        copy.deepcopy(base_cfg),
        int(configs.n_m),
        float(getattr(base_cfg, "interarrival_mean", configs.interarrival_mean)),
        make_burst_sampler(base_cfg),
        rng,
    )
    orch = GlobalTimelineOrchestrator(int(configs.n_m), generator, t0=0.0)
    all_due: Dict[int, float] = {}
    init_cfg = copy.deepcopy(base_cfg)
    setattr(init_cfg, "init_jobs", int(getattr(configs, "init_jobs", 0)))
    initial_jobs = sample_initial_jobs(init_cfg, rng=rng, base_job_id=0, t_arrive=0.0)
    if initial_jobs:
        register_initial_jobs(orch, generator, initial_jobs, all_due, t0=0.0)

    env = HLGateEnv(
        n_machines=int(configs.n_m),
        heuristic=str(configs.scheduler_type),
        interarrival_mean=float(configs.interarrival_mean),
        burst_K=int(configs.burst_size),
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    env.gen = generator
    env.orch = orch
    env.all_job_due_dates = all_due
    env.t_now = 0.0
    env.t_next = float(generator.sample_next_time(0.0))
    env.events_done = 0
    env.release_count = 1 if initial_jobs else 0
    env.agent_release_count = 0
    env.baseline_final_td = 0.0
    env.baseline_final_mk = 0.0
    env.baseline_event_td = []
    env._last_release_event_idx = 0
    env._last_release_td = 0.0
    env._prev_arrival_time = 0.0
    env.steps_since_last_release = 0
    return env, list(initial_jobs)


def apply_event_without_advance(env: HLGateEnv, action: int) -> None:
    """Apply one event while keeping t_now fixed for checkpoint accounting."""
    event_time = float(env.t_now)
    action = int(action)
    if action == 1:
        env.orch.event_release_and_reschedule(event_time)
        env.release_count += 1
        env.agent_release_count += 1
        env.steps_since_last_release = 0
    else:
        env.orch.tick_without_release(event_time)
        env.steps_since_last_release += 1
    env.events_done += 1
    env._prev_arrival_time = event_time


def observe_arrivals(
    env: HLGateEnv,
    seen_ids: set,
    segment_jobs: List[object],
) -> None:
    """Append newly generated jobs to the current segment summary."""
    new_ids = [int(jid) for jid in env.all_job_due_dates if int(jid) not in seen_ids]
    if not new_ids:
        return
    jobs = _job_map(env.orch)
    for jid in new_ids:
        job = jobs.get(jid)
        if job is not None:
            segment_jobs.append(job)
        seen_ids.add(jid)


def load_high_level_model(scenario: str, device: torch.device):
    path = SCENARIO_MODEL_PATHS[scenario]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing high-level weight for {scenario}: {path}")
    model = HLPPOGateNet(
        obs_dim=get_hl_gate_state_dim(configs),
        n_actions=2,
        hidden=int(getattr(configs, "hl_ppo_hidden_dim", 256)),
        num_layers=int(getattr(configs, "hl_ppo_num_layers", 3)),
        separate_trunks=bool(getattr(configs, "hl_ppo_separate_trunks", False)),
        actor_hidden=int(getattr(configs, "hl_ppo_actor_hidden_dim", getattr(configs, "hl_ppo_hidden_dim", 256))),
        actor_num_layers=int(getattr(configs, "hl_ppo_actor_num_layers", getattr(configs, "hl_ppo_num_layers", 3))),
        critic_hidden=int(getattr(configs, "hl_ppo_critic_hidden_dim", getattr(configs, "hl_ppo_hidden_dim", 256))),
        critic_num_layers=int(getattr(configs, "hl_ppo_critic_num_layers", getattr(configs, "hl_ppo_num_layers", 3))),
        value_hidden=int(getattr(configs, "hl_ppo_value_hidden_dim", getattr(configs, "hl_ppo_hidden_dim", 256))),
        value_num_layers=int(getattr(configs, "hl_ppo_value_num_layers", 1)),
        use_residual=bool(getattr(configs, "hl_ppo_use_residual", False)),
        use_glu=bool(getattr(configs, "hl_ppo_use_glu", False)),
        pre_norm=bool(getattr(configs, "hl_ppo_pre_norm", False)),
        manual_obs_dim=get_hl_gate_state_dim(configs),
        ll_embed_raw_dim=int(getattr(configs, "hl_ll_buffer_embedding_dim", HL_LL_BUFFER_EMBED_DIM))
        if bool(getattr(configs, "hl_use_ll_buffer_embedding", False))
        else 0,
        ll_embed_proj_dim=int(getattr(configs, "hl_ll_buffer_projection_dim", 16)),
        initial_release_prob=float(getattr(configs, "hl_initial_release_prob", -1.0)),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def ppo_action(model, obs: np.ndarray, device: torch.device) -> int:
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        return int(torch.argmax(logits, dim=1).item())


def choose_action(
    strategy: str,
    event_id: int,
    env: HLGateEnv,
    obs: np.ndarray,
    model,
    device: torch.device,
) -> int:
    if event_id >= int(env.event_horizon):
        return 1
    if not env._is_decision_event(event_id):
        return 0
    if strategy == "ppo":
        return ppo_action(model, obs, device)
    if strategy == "cadence1":
        return 1
    if strategy == "cadence5":
        interval = max(1, int(getattr(configs, "hl_gate_decision_interval", 1)))
        decision_step = int(event_id) // interval
        return int(decision_step % 5 == 0)
    if strategy == "slack0":
        return int(bool(env.orch.buffer) and float(env._get_buffer_stats(env.t_now)["min_slack"]) < 0.0)
    raise ValueError(f"Unknown strategy: {strategy}")


def segment_summary(segment_jobs: Sequence[object], release_count: int) -> Dict[str, object]:
    if not segment_jobs:
        return {
            "arr_jobs": "",
            "urg_jobs": "",
            "in_smin": "",
            "in_sq25": "",
            "in_smean": "",
            "rel_n": int(release_count),
        }
    initial_slacks = []
    urgent_count = 0
    for job in segment_jobs:
        due = float(getattr(job, "meta", {}).get("due_date", 0.0))
        arrive = float(getattr(job, "meta", {}).get("t_arrive", 0.0))
        initial_slacks.append(due - arrive - _job_work(job))
        urgent_count += int(bool(getattr(job, "meta", {}).get("is_urgent", False)))
    smin, smean, sq25, _ = _slack_summary(initial_slacks)
    return {
        "arr_jobs": len(segment_jobs),
        "urg_jobs": urgent_count,
        "in_smin": smin,
        "in_sq25": sq25,
        "in_smean": smean,
        "rel_n": int(release_count),
    }


def rounded_row(
    strategy: str,
    seed: int,
    event_id: int,
    now: float,
    current: Mapping[str, float],
    release_now: Mapping[str, float],
    segment: Mapping[str, object],
) -> Dict[str, object]:
    def integer(key: str):
        return int(round(float(current[key])))

    def std(key: str):
        return round(float(current[key]), 2)

    row = {
        "strat": strategy,
        "seed": int(seed),
        "event": int(event_id),
        "time": int(round(float(now))),
        "mk": integer("mk"),
        "td": integer("td"),
        "obj": integer("obj"),
        "b_jobs": integer("b_jobs"),
        "b_work": integer("b_work"),
        "b_smin": integer("b_smin"),
        "b_smean": integer("b_smean"),
        "b_sq25": integer("b_sq25"),
        "b_sstd": std("b_sstd"),
        "b_neg_n": integer("b_neg_n"),
        "b_neg_r": round(float(current["b_neg_r"]), 3),
        "w_jobs": integer("w_jobs"),
        "w_ops": integer("w_ops"),
        "w_work": integer("w_work"),
        "w_td": integer("w_td"),
        "w_smin": integer("w_smin"),
        "w_smean": integer("w_smean"),
        "w_sq25": integer("w_sq25"),
        "w_sstd": std("w_sstd"),
        "m_lmean": integer("m_lmean"),
        "m_lstd": std("m_lstd"),
        "m_lmin": integer("m_lmin"),
        "m_lmax": integer("m_lmax"),
        "m_imb": integer("m_imb"),
        "rn_mk": int(round(float(release_now["mk"]))),
        "rn_td": int(round(float(release_now["td"]))),
        "rn_obj": int(round(float(release_now["obj"]))),
    }
    row.update(segment)
    return {key: row.get(key, "") for key in CSV_COLUMNS}


def run_strategy(
    scenario: str,
    seed: int,
    strategy: str,
    model,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    strategy_started_at = time.perf_counter()
    print(
        f"    [START] scenario={scenario} seed={seed} strategy={strategy} event=0/{CHECKPOINTS[-1]}",
        flush=True,
    )
    env, initial_jobs = setup_analysis_env(scenario, seed)
    seen_ids = {int(job.job_id) for job in initial_jobs}
    segment_jobs: List[object] = list(initial_jobs)
    segment_releases = 0
    rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    # Event 0 is the initial post-registration state.  Initial jobs are also
    # included in the first 0-40 segment summary.
    current0 = collect_metrics(env, now=0.0)
    release0 = release_now_metrics(env, 0.0)
    event_rows.append(
        {
            "strat": strategy,
            "seed": int(seed),
            "event": 0,
            "time": 0,
            "action": "INIT",
            "released": 0,
            "td_before": round(float(current0["td"]), 3),
            "td": round(float(current0["td"]), 3),
            "b_jobs": int(round(current0["b_jobs"])),
            "b_ops": int(round(current0["b_ops"])),
            "b_work": round(float(current0["b_work"]), 3),
            "b_smin": round(float(current0["b_smin"]), 3),
            "b_smean": round(float(current0["b_smean"]), 3),
            "b_sq25": round(float(current0["b_sq25"]), 3),
            "b_sstd": round(float(current0["b_sstd"]), 3),
            "b_neg_sum": round(float(current0["b_neg_sum"]), 3),
            "b_lt0_n": int(round(current0["b_neg_n"])),
            "b_lt100_n": int(round(current0["b_lt100_n"])),
            "b_lt0_r": round(float(current0["b_lt0_r"]), 4),
            "b_lt100_r": round(float(current0["b_lt100_r"]), 4),
            "b_critical_work": round(float(current0["b_critical_work"]), 3),
            "b_near_critical_work": round(float(current0["b_near_critical_work"]), 3),
            "w_jobs": int(round(current0["w_jobs"])),
            "w_ops": int(round(current0["w_ops"])),
            "w_work": round(float(current0["w_work"]), 3),
            "w_smin": round(float(current0["w_smin"]), 3),
            "w_smean": round(float(current0["w_smean"]), 3),
            "w_sq25": round(float(current0["w_sq25"]), 3),
            "w_sstd": round(float(current0["w_sstd"]), 3),
            "w_neg_sum": round(float(current0["w_neg_sum"]), 3),
            "w_lt0_n": int(round(current0["w_lt0_n"])),
            "w_lt100_n": int(round(current0["w_lt100_n"])),
            "w_lt0_r": round(float(current0["w_lt0_r"]), 4),
            "w_lt100_r": round(float(current0["w_lt100_r"]), 4),
            "w_critical_work": round(float(current0["w_critical_work"]), 3),
            "w_near_critical_work": round(float(current0["w_near_critical_work"]), 3),
            "bw_smin": round(float(current0["bw_smin"]), 3),
            "bw_smean": round(float(current0["bw_smean"]), 3),
            "bw_sq25": round(float(current0["bw_sq25"]), 3),
            "bw_sstd": round(float(current0["bw_sstd"]), 3),
            "bw_jobs": int(round(current0["bw_jobs"])),
            "bw_neg_sum": round(float(current0["bw_neg_sum"]), 3),
            "bw_lt0_n": int(round(current0["bw_lt0_n"])),
            "bw_lt100_n": int(round(current0["bw_lt100_n"])),
            "bw_lt0_r": round(float(current0["bw_lt0_r"]), 4),
            "bw_lt100_r": round(float(current0["bw_lt100_r"]), 4),
            "bw_critical_work": round(float(current0["bw_critical_work"]), 3),
            "bw_near_critical_work": round(float(current0["bw_near_critical_work"]), 3),
            "m_lmax": round(float(current0["m_lmax"]), 3),
        }
    )
    rows.append(
        rounded_row(
            strategy,
            seed,
            0,
            0.0,
            current0,
            release0,
            segment_summary(segment_jobs, segment_releases),
        )
    )

    while env.events_done < int(env.event_horizon):
        env._advance_to_next_arrival()
        event_id = int(env.events_done + 1)
        observe_arrivals(env, seen_ids, segment_jobs)
        obs = env._observe()
        event_time = float(env.t_now)
        action = choose_action(strategy, event_id, env, obs, model, device)
        before = collect_metrics(env, now=event_time)

        # The release-now fields are only needed at output checkpoints.  Avoid
        # cloning and re-solving the scheduler for intermediate events.
        release_now = None
        if event_id in CHECKPOINTS[1:]:
            release_now = release_now_metrics(env, event_time)

        apply_event_without_advance(env, action)
        current = collect_metrics(env, now=event_time)
        if action == 1:
            segment_releases += 1

        event_rows.append(
            {
                "strat": strategy,
                "seed": int(seed),
                "event": int(event_id),
                "time": round(event_time, 3),
                "action": "RELEASE" if action else "HOLD",
                "released": int(action == 1),
                "td_before": round(float(before["td"]), 3),
                "td": round(float(current["td"]), 3),
                "b_jobs": int(round(before["b_jobs"])),
                "b_ops": int(round(before["b_ops"])),
                "b_work": round(float(before["b_work"]), 3),
                "b_smin": round(float(before["b_smin"]), 3),
                "b_smean": round(float(before["b_smean"]), 3),
                "b_sq25": round(float(before["b_sq25"]), 3),
                "b_sstd": round(float(before["b_sstd"]), 3),
                "b_neg_sum": round(float(before["b_neg_sum"]), 3),
                "b_lt0_n": int(round(before["b_neg_n"])),
                "b_lt100_n": int(round(before["b_lt100_n"])),
                "b_lt0_r": round(float(before["b_lt0_r"]), 4),
                "b_lt100_r": round(float(before["b_lt100_r"]), 4),
                "b_critical_work": round(float(before["b_critical_work"]), 3),
                "b_near_critical_work": round(float(before["b_near_critical_work"]), 3),
                "w_jobs": int(round(before["w_jobs"])),
                "w_ops": int(round(before["w_ops"])),
                "w_work": round(float(before["w_work"]), 3),
                "w_smin": round(float(before["w_smin"]), 3),
                "w_smean": round(float(before["w_smean"]), 3),
                "w_sq25": round(float(before["w_sq25"]), 3),
                "w_sstd": round(float(before["w_sstd"]), 3),
                "w_neg_sum": round(float(before["w_neg_sum"]), 3),
                "w_lt0_n": int(round(before["w_lt0_n"])),
                "w_lt100_n": int(round(before["w_lt100_n"])),
                "w_lt0_r": round(float(before["w_lt0_r"]), 4),
                "w_lt100_r": round(float(before["w_lt100_r"]), 4),
                "w_critical_work": round(float(before["w_critical_work"]), 3),
                "w_near_critical_work": round(float(before["w_near_critical_work"]), 3),
                "bw_smin": round(float(before["bw_smin"]), 3),
                "bw_smean": round(float(before["bw_smean"]), 3),
                "bw_sq25": round(float(before["bw_sq25"]), 3),
                "bw_sstd": round(float(before["bw_sstd"]), 3),
                "bw_jobs": int(round(before["bw_jobs"])),
                "bw_neg_sum": round(float(before["bw_neg_sum"]), 3),
                "bw_lt0_n": int(round(before["bw_lt0_n"])),
                "bw_lt100_n": int(round(before["bw_lt100_n"])),
                "bw_lt0_r": round(float(before["bw_lt0_r"]), 4),
                "bw_lt100_r": round(float(before["bw_lt100_r"]), 4),
                "bw_critical_work": round(float(before["bw_critical_work"]), 3),
                "bw_near_critical_work": round(float(before["bw_near_critical_work"]), 3),
                "m_lmax": round(float(before["m_lmax"]), 3),
            }
        )

        if event_id in CHECKPOINTS[1:]:
            rows.append(
                rounded_row(
                    strategy,
                    seed,
                    event_id,
                    event_time,
                    current,
                    release_now,
                    segment_summary(segment_jobs, segment_releases),
                )
            )
            segment_jobs = []
            segment_releases = 0

        if event_id % 10 == 0 or event_id in CHECKPOINTS[1:]:
            print(
                f"    [EVENT] scenario={scenario} seed={seed} strategy={strategy} "
                f"event={event_id}/{env.event_horizon}",
                flush=True,
            )

    print(
        f"    [DONE] scenario={scenario} seed={seed} strategy={strategy} "
        f"rows={len(rows)} elapsed={time.perf_counter() - strategy_started_at:.1f}s",
        flush=True,
    )
    return rows, event_rows


def write_seed_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_event_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def plot_tardiness_indicators(
    path: str,
    scenario: str,
    seed: int,
    slack_stat: str,
    rows_by_strategy: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Plot one slack statistic for all comparison strategies."""
    colors = {
        "ppo": "#2563eb",
        "cadence5": "#dc2626",
        "slack0": "#16a34a",
    }
    stat_fields = {"min": "smin", "mean": "smean", "q25": "sq25"}
    if slack_stat not in stat_fields:
        raise ValueError(f"Unsupported slack statistic: {slack_stat}")
    field_suffix = stat_fields[slack_stat]
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    for strategy in PLOT_STRATEGIES:
        rows = rows_by_strategy.get(strategy, [])
        if not rows:
            continue
        events = [int(row["event"]) for row in rows]
        color = colors[strategy]
        release_rows = [row for row in rows if int(row["released"]) == 1]
        release_events = [int(row["event"]) for row in release_rows]

        axes[0].step(
            events,
            [float(row["td"]) for row in rows],
            where="post",
            color=color,
            label=strategy,
        )

        def mark_releases(axis, field: str, *, show_label: bool = False) -> None:
            axis.scatter(
                release_events,
                [float(row[field]) for row in release_rows],
                color=color,
                s=20,
                zorder=3,
                label=f"{strategy} release" if show_label else "_nolegend_",
            )

        def plot_slack(axis, prefix: str, label_prefix: str) -> None:
            field = f"{prefix}_{field_suffix}"
            axis.plot(
                events,
                [float(row[field]) for row in rows],
                color=color,
                label=f"{strategy} {label_prefix} {slack_stat}",
            )

        mark_releases(axes[0], "td", show_label=True)
        for axis, prefix, label_prefix in (
            (axes[1], "b", "buffer slack"),
            (axes[2], "w", "WIP slack"),
            (axes[3], "bw", "buffer+WIP slack"),
        ):
            plot_slack(axis, prefix, label_prefix)
            mark_releases(
                axis,
                f"{prefix}_{field_suffix}",
                show_label=False,
            )

    axes[0].set_ylabel("Global TD")
    axes[1].set_ylabel(f"Buffer slack ({slack_stat})")
    axes[2].set_ylabel(f"WIP slack ({slack_stat})")
    axes[3].set_ylabel(f"Buffer+WIP slack ({slack_stat})")
    axes[3].set_xlabel("Event")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=2)
    fig.suptitle(
        f"Tardiness and {slack_stat} slack | {scenario} | seed {int(seed):02d}"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_indicators(
    path: str,
    scenario: str,
    seed: int,
    metric_label: str,
    fields: Mapping[str, str],
    rows_by_strategy: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Plot one non-slack metric with TD and three population views."""
    colors = {
        "ppo": "#2563eb",
        "cadence5": "#dc2626",
        "slack0": "#16a34a",
    }
    labels = {
        "b": "Buffer",
        "w": "WIP",
        "bw": "Buffer+WIP",
    }
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    for strategy in PLOT_STRATEGIES:
        rows = rows_by_strategy.get(strategy, [])
        if not rows:
            continue
        events = [int(row["event"]) for row in rows]
        color = colors[strategy]
        release_rows = [row for row in rows if int(row["released"]) == 1]
        release_events = [int(row["event"]) for row in release_rows]

        axes[0].step(
            events,
            [float(row["td"]) for row in rows],
            where="post",
            color=color,
            label=strategy,
        )
        axes[0].scatter(
            release_events,
            [float(row["td"]) for row in release_rows],
            color=color,
            s=20,
            zorder=3,
            label=f"{strategy} release",
        )

        for axis, prefix in zip(axes[1:], ("b", "w", "bw")):
            field = fields[prefix]
            axis.plot(
                events,
                [float(row[field]) for row in rows],
                color=color,
                label=f"{strategy} {labels[prefix]}",
            )
            axes_index = ("b", "w", "bw").index(prefix) + 1
            axes[axes_index].scatter(
                release_events,
                [float(row[field]) for row in release_rows],
                color=color,
                s=20,
                zorder=3,
                label="_nolegend_",
            )

    axes[0].set_ylabel("Global TD")
    axes[1].set_ylabel(f"Buffer {metric_label}")
    axes[2].set_ylabel(f"WIP {metric_label}")
    axes[3].set_ylabel(f"Buffer+WIP {metric_label}")
    axes[3].set_xlabel("Event")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=2)
    fig.suptitle(
        f"Tardiness and {metric_label} | {scenario} | seed {int(seed):02d}"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def replot_existing_results(root: str) -> None:
    """Regenerate plots from existing event CSVs without rerunning simulations."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Analysis result directory not found: {root}")

    print(f"[REPLOT] {root}")
    for scenario in SCENARIOS:
        scenario_dir = os.path.join(root, scenario)
        if not os.path.isdir(scenario_dir):
            print(f"  [SKIP] missing scenario directory: {scenario}")
            continue
        for seed in SELECTED_SEEDS:
            event_rows_by_strategy: Dict[str, List[Dict[str, str]]] = {}
            missing = []
            for strategy in STRATEGIES:
                event_path = os.path.join(
                    scenario_dir,
                    f"{scenario}_seed_{int(seed):02d}_event_metrics_{strategy}.csv",
                )
                if not os.path.isfile(event_path):
                    missing.append(strategy)
                    continue
                with open(event_path, "r", newline="", encoding="utf-8-sig") as handle:
                    event_rows_by_strategy[strategy] = list(csv.DictReader(handle))
            if missing:
                print(
                    f"  [SKIP] {scenario} seed={int(seed):02d} "
                    f"missing event CSV: {', '.join(missing)}"
                )
                continue

            for slack_stat in ("min", "mean", "q25"):
                plot_tardiness_indicators(
                    os.path.join(
                        scenario_dir,
                        f"{scenario}_seed_{int(seed):02d}_tardiness_indicators_{slack_stat}.png",
                    ),
                    scenario,
                    seed,
                    slack_stat,
                    event_rows_by_strategy,
                )
            for metric_name, metric_fields in PLOT_METRIC_SPECS:
                plot_metric_indicators(
                    os.path.join(
                        scenario_dir,
                        f"{scenario}_seed_{int(seed):02d}_tardiness_indicators_{metric_name}.png",
                    ),
                    scenario,
                    seed,
                    metric_name,
                    metric_fields,
                    event_rows_by_strategy,
                )
            print(f"  [PLOT] {scenario} seed={int(seed):02d}")


def main() -> None:
    if REPLOT_ROOT:
        replot_existing_results(REPLOT_ROOT)
        return

    output_root = OUTPUT_ROOT or os.path.join(
        PROJECT_ROOT,
        "analysis_results",
        "release_strategy_analysis",
        time.strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_root, exist_ok=True)
    device = torch.device(getattr(configs, "device", "cpu"))

    print(f"[CONFIG] {CONFIG_PATH}")
    print(f"[OUTPUT] {output_root}")
    print(f"[SCENARIOS] {', '.join(SCENARIOS)}")
    if len(SELECTED_SEEDS) == 1:
        print(f"[SEEDS] {SELECTED_SEEDS[0]}")
    else:
        print(f"[SEEDS] {SELECTED_SEEDS[0]}~{SELECTED_SEEDS[-1]}")
    print(f"[STRATEGIES] {', '.join(STRATEGIES)}")

    for scenario in SCENARIOS:
        configs.hl_env_scenario = scenario
        configs.hl_ppo_model_path = SCENARIO_MODEL_PATHS[scenario]
        scenario_dir = os.path.join(output_root, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        model = load_high_level_model(scenario, device)
        print(f"[SCENARIO] {scenario} | high-level={SCENARIO_MODEL_PATHS[scenario]}")

        for seed in SELECTED_SEEDS:
            output_path = os.path.join(
                scenario_dir, f"{scenario}_seed_{int(seed):02d}.csv"
            )
            required_paths = [
                output_path,
                *(
                    os.path.join(
                        scenario_dir,
                        f"{scenario}_seed_{int(seed):02d}_event_metrics_{strategy}.csv",
                    )
                    for strategy in STRATEGIES
                ),
                *(
                    os.path.join(
                        scenario_dir,
                        f"{scenario}_seed_{int(seed):02d}_tardiness_indicators_{name}.png",
                    )
                    for name in (
                        "min",
                        "mean",
                        "q25",
                        "slack_lt0_count",
                        "slack_lt100_count",
                        "negative_slack_burden",
                        "critical_work_lt0",
                        "near_critical_work_lt100",
                    )
                ),
            ]
            if all(os.path.isfile(path) for path in required_paths):
                print(f"  [SKIP] {scenario} seed={int(seed):02d} already complete")
                continue

            seed_rows: List[Dict[str, object]] = []
            event_rows_by_strategy: Dict[str, List[Dict[str, object]]] = {}
            for strategy in STRATEGIES:
                print(f"  [RUN] scenario={scenario} seed={seed} strategy={strategy}")
                policy_model = model if strategy == "ppo" else None
                checkpoint_rows, event_rows = run_strategy(
                    scenario,
                    seed,
                    strategy,
                    policy_model,
                    device,
                )
                seed_rows.extend(checkpoint_rows)
                event_rows_by_strategy[strategy] = event_rows
            # Group rows by checkpoint first so strategies can be compared at
            # the same event before moving to the next checkpoint.
            strategy_order = {name: index for index, name in enumerate(STRATEGIES)}
            seed_rows.sort(
                key=lambda row: (
                    int(row["event"]),
                    strategy_order[str(row["strat"])],
                )
            )
            write_seed_csv(output_path, seed_rows)
            print(f"  [CSV] {output_path}")
            for strategy, event_rows in event_rows_by_strategy.items():
                event_path = os.path.join(
                    scenario_dir,
                    f"{scenario}_seed_{int(seed):02d}_event_metrics_{strategy}.csv",
                )
                write_event_csv(event_path, event_rows)
            for slack_stat in ("min", "mean", "q25"):
                plot_path = os.path.join(
                    scenario_dir,
                    f"{scenario}_seed_{int(seed):02d}_tardiness_indicators_{slack_stat}.png",
                )
                plot_tardiness_indicators(
                    plot_path,
                    scenario,
                    seed,
                    slack_stat,
                    event_rows_by_strategy,
                )
            metric_specs = (
                (
                    "slack_lt0_count",
                    {"b": "b_lt0_n", "w": "w_lt0_n", "bw": "bw_lt0_n"},
                ),
                (
                    "slack_lt100_count",
                    {"b": "b_lt100_n", "w": "w_lt100_n", "bw": "bw_lt100_n"},
                ),
                (
                    "negative_slack_burden",
                    {"b": "b_neg_sum", "w": "w_neg_sum", "bw": "bw_neg_sum"},
                ),
                (
                    "critical_work_lt0",
                    {
                        "b": "b_critical_work",
                        "w": "w_critical_work",
                        "bw": "bw_critical_work",
                    },
                ),
                (
                    "near_critical_work_lt100",
                    {
                        "b": "b_near_critical_work",
                        "w": "w_near_critical_work",
                        "bw": "bw_near_critical_work",
                    },
                ),
            )
            for metric_name, metric_fields in metric_specs:
                plot_path = os.path.join(
                    scenario_dir,
                    f"{scenario}_seed_{int(seed):02d}_tardiness_indicators_{metric_name}.png",
                )
                plot_metric_indicators(
                    plot_path,
                    scenario,
                    seed,
                    metric_name,
                    metric_fields,
                    event_rows_by_strategy,
                )
            print(f"  [EVENT CSV/PNG] {scenario} seed={int(seed):02d}")

    print("[DONE] Release strategy analysis completed.")


if __name__ == "__main__":
    main()
