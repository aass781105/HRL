import os
import sys
import numpy as np
import yaml
import copy

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from params import configs, parser

def get_seed_physics_stats(seed_val):
    config_path = os.path.join(project_root, "yaml_config", f"eval_baseline_seed{seed_val}_greedy_cadence1_1run.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
        
    configs.__dict__.update(file_cfg)
    parser.set_defaults(**file_cfg)
    cfg = parser.parse_args(args=["--config", config_path])
    
    from hl_gate_env import resolve_hl_env_scenario, scenario_config, make_burst_sampler, EventBurstGenerator
    from dynamic_job_stream import sample_initial_jobs
    
    rng = np.random.default_rng(int(seed_val))
    scenario = resolve_hl_env_scenario(configs, rng)
    base_cfg = scenario_config(configs, scenario)
    
    # Generate actual jobs to inspect operations
    from data_utils import SD2_instance_generator
    gen = EventBurstGenerator(
        SD2_instance_generator,
        copy.deepcopy(base_cfg),
        int(cfg.n_m),
        float(getattr(base_cfg, "interarrival_mean", 25.0)),
        make_burst_sampler(base_cfg),
        rng,
    )
    
    rng_init = np.random.default_rng(int(seed_val))
    init_job_specs = sample_initial_jobs(base_cfg, rng=rng_init, base_job_id=0, t_arrive=0.0)
    all_jobs = list(init_job_specs)
    
    t_now = 0.0
    for event_id in range(1, int(cfg.event_horizon) + 1):
        t_next = float(gen.sample_next_time(t_now))
        t_now = t_next
        new_jobs = gen.generate_burst(t_now)
        if new_jobs:
            all_jobs.extend(new_jobs)
            
    # Calculate machine statistics
    # 1. Total PT sum for compatibilities
    # 2. Minimum PT sum if we always choose the fastest machine for each op
    mch_min_pt = {m: 0.0 for m in range(cfg.n_m)}
    mch_avg_pt = {m: 0.0 for m in range(cfg.n_m)}
    
    total_ops_count = 0
    for j in all_jobs:
        for op in j.operations:
            v = np.array(op.time_row)
            compat_mchs = np.where(v > 0)[0]
            total_ops_count += 1
            # If we choose min PT
            min_mch = compat_mchs[np.argmin(v[compat_mchs])]
            mch_min_pt[min_mch] += v[min_mch]
            for m in compat_mchs:
                mch_avg_pt[m] += v[m]
                
    return {
        "total_jobs": len(all_jobs),
        "total_ops": total_ops_count,
        "mch_min_pt": mch_min_pt,
        "mch_avg_pt": mch_avg_pt
    }

def main():
    s4 = get_seed_physics_stats(4)
    s10 = get_seed_physics_stats(10)
    
    print("\n================ MACHINE LOAD AND COMPATIBILITY EVIDENCE ================")
    print(f"Total Operations: Seed 4 = {s4['total_ops']} | Seed 10 = {s10['total_ops']}")
    print("-" * 65)
    print("If each Operation is assigned to its FASTEST compatible machine (Min PT Load):")
    print(f"{'Machine':<10} | {'Seed 4 Min PT Load':<22} | {'Seed 10 Min PT Load':<22}")
    print("-" * 65)
    for m in range(5):
        print(f"Machine {m:<2} | {s4['mch_min_pt'][m]:<22.2f} | {s10['mch_min_pt'][m]:<22.2f}")
        
    print("\nTotal Sum of Min PT Load:")
    print(f"Seed 4  Total Min PT: {sum(s4['mch_min_pt'].values()):.2f}")
    print(f"Seed 10 Total Min PT: {sum(s10['mch_min_pt'].values()):.2f}")
    
    print("\nLoad Unbalance (Standard Deviation of Min PT Load across 5 machines):")
    s4_std = np.std(list(s4['mch_min_pt'].values()))
    s10_std = np.std(list(s10['mch_min_pt'].values()))
    print(f"Seed 4  Load Std: {s4_std:.2f}")
    print(f"Seed 10 Load Std: {s10_std:.2f}")

if __name__ == "__main__":
    main()
