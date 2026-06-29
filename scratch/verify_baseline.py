import sys
import os
import copy
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from params import configs
from hl_gate_env import compute_cadence_baseline_for_seed, HLGateEnv

def verify():
    # Force evaluation config
    setattr(configs, "ll_rollout_k", 1)
    setattr(configs, "eval_action_selection", "greedy")
    setattr(configs, "hl_td_signal_source", "baseline_gap_final")
    setattr(configs, "hl_td_credit_mode", "terminal_only")
    
    seed = 142 # The seed used for UPD 0002 (since train_seed is 142)
    print("--- 1. Running Cadence-1 Baseline ---")
    baseline = compute_cadence_baseline_for_seed(
        seed,
        n_machines=configs.n_m,
        interarrival_mean=configs.interarrival_mean,
        burst_k=configs.burst_size,
        event_horizon=configs.event_horizon,
        init_jobs=configs.init_jobs,
        cadence=1
    )
    print(f"Baseline TD: {baseline['td']:.2f}")
    print(f"Baseline Makespan: {baseline['mk']:.2f}")
    print(f"Baseline Release Count: {baseline['release_count']}")
    
    print("\n--- 2. Checking if any logical difference exists ---")
    # Let's inspect if GlobalTimelineOrchestrator logic runs identically inside env step
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=configs.event_horizon,
        init_jobs=configs.init_jobs
    )
    obs, info = env.reset(seed=seed)
    
    # Perform a dummy run of all RELEASE (action=1) to see if it matches Cadence-1 baseline
    done = False
    step_idx = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(1) # Force all release
        done = terminated or truncated
        step_idx += 1
    
    final_stats = env.orch.get_final_kpi_stats(env.all_job_due_dates)
    print("--- 3. Running Env with Forced All-RELEASE (Action=1) ---")
    print(f"Forced Release TD: {final_stats['tardiness']:.2f}")
    print(f"Forced Release Makespan: {final_stats['makespan']:.2f}")
    print(f"Forced Release Count: {env.release_count}")

    # --- Debugging detailed job count differences ---
    baseline_fins = baseline.get("event_td", []) # wait, compute_cadence_baseline_for_seed returns a dict
    # Let's rebuild job finish dicts for both
    baseline_orch = compute_cadence_baseline_for_seed_orch(seed) # Let's print out dict sizes
    agent_fins_dict = env.orch._job_history_finishes.copy()
    for r in env.orch._last_full_rows:
        agent_fins_dict[int(r["job"])] = max(agent_fins_dict.get(int(r["job"]), 0.0), float(r["end"]))
        
    base_fins_dict = baseline_orch._job_history_finishes.copy()
    for r in baseline_orch._last_full_rows:
        base_fins_dict[int(r["job"])] = max(base_fins_dict.get(int(r["job"]), 0.0), float(r["end"]))

    print("\n--- 4. Detail Checklist ---")
    print(f"Baseline total due dates in dictionary: {len(env.all_job_due_dates)}")
    print(f"Baseline finished jobs count: {len(base_fins_dict)}")
    print(f"Agent finished jobs count: {len(agent_fins_dict)}")
    
    # Check if there are any jobs in due dates that are missing in finishes
    missing_in_base = [jid for jid in env.all_job_due_dates if jid not in base_fins_dict]
    missing_in_agent = [jid for jid in env.all_job_due_dates if jid not in agent_fins_dict]
    
    print(f"Jobs missing in Baseline finishes: {missing_in_base}")
    print(f"Jobs missing in Agent finishes: {missing_in_agent}")

def compute_cadence_baseline_for_seed_orch(instance_seed: int):
    # Helper to return the orchestrator instance of baseline
    from dynamic_job_stream import EventBurstGenerator
    from hrl_orchestrator import GlobalTimelineOrchestrator
    from dynamic_job_stream import SD2_instance_generator, sample_initial_jobs, register_initial_jobs
    
    rng = np.random.default_rng(int(instance_seed))
    gen = EventBurstGenerator(
        SD2_instance_generator,
        copy.deepcopy(configs),
        configs.n_m,
        configs.interarrival_mean,
        lambda _r: int(configs.burst_size),
        rng,
    )
    orch = GlobalTimelineOrchestrator(configs.n_m, gen, t0=0.0)
    all_job_due_dates = {}
    release_count = 0
    t_now = 0.0

    if int(configs.init_jobs) > 0:
        init_cfg = copy.deepcopy(configs)
        setattr(init_cfg, "init_jobs", int(configs.init_jobs))
        init_job_specs = sample_initial_jobs(init_cfg, rng=rng, base_job_id=0, t_arrive=0.0)
        release_count += register_initial_jobs(orch, gen, init_job_specs, all_job_due_dates, t0=0.0)

    t_next = float(gen.sample_next_time(t_now))
    t_now = float(t_next)
    new_jobs = gen.generate_burst(t_now)
    if new_jobs:
        for job in new_jobs:
            all_job_due_dates[job.job_id] = job.meta["due_date"]
        orch.buffer.extend(new_jobs)
    t_next = float(gen.sample_next_time(t_now))

    events_done = 1
    while True:
        orch.event_release_and_reschedule(float(t_now))
        if events_done >= int(configs.event_horizon):
            break
        t_now = float(t_next)
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            for job in new_jobs:
                all_job_due_dates[job.job_id] = job.meta["due_date"]
            orch.buffer.extend(new_jobs)
        t_next = float(gen.sample_next_time(t_now))
        events_done += 1
    return orch

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


