import sys
import os
import csv

# Find the project root directory (parent of scratch/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set CLI args programmatically to load the baseline config
sys.argv = ["", "--config", os.path.join(project_root, "yaml_config", "eval_baseline_seed1_greedy_cadence1_1run.yml")]

from params import configs
# Force the environment to run baseline calculations
configs.hl_td_signal_source = "baseline_gap_final"

from hl_gate_env import HLGateEnv

def main():
    # Output file path (saved in the same scratch directory)
    output_csv = os.path.join(os.path.dirname(__file__), "eval_baseline_i0_i9_comparison.csv")
    
    # We will evaluate I0 to I9 (all 10 batches in the 200 epochs)
    base_seed = 42
    seed_stride = 100_000
    num_envs = 4
    
    print("Initializing environment (loading PyTorch model weights once)...")
    # Instantiate the env once to avoid multiple torch model loads
    env = HLGateEnv(
        n_machines=configs.n_m,
        heuristic=configs.scheduler_type,
        interarrival_mean=configs.interarrival_mean,
        burst_K=configs.burst_size,
        event_horizon=int(configs.event_horizon),
        init_jobs=int(getattr(configs, "init_jobs", 0)),
    )
    
    print("Starting baseline evaluations for all 10 batches (I0 to I9) under Cadence 1...")
    print(f"Results will be saved to: {output_csv}\n")
    
    headers = [
        "Group", "Env", "Seed", 
        "Makespan_Cadence1", "Tardiness_Cadence1", "Releases_Cadence1"
    ]
    
    records = []
    
    # Loop through batches I0 to I9
    for group_idx in range(10):
        group_name = f"I{group_idx}"
        print(f"================ Processing Batch {group_name} ================")
        
        # Loop through parallel environments Env 0 to Env 3
        for env_idx in range(num_envs):
            seed = base_seed + env_idx * seed_stride + group_idx
            print(f"  -> Evaluating Env {env_idx} (Seed {seed}) [Cadence 1]...", end="", flush=True)
            
            # Run Cadence 1
            configs.baseline_cadence = 1
            env.reset(seed=seed)
            mk1 = env.baseline_final_mk
            td1 = env.baseline_final_td
            rel1 = env.baseline_release_count
            
            print(f" Done. Makespan: {mk1:.2f}, Tardiness: {td1:.2f}")
            
            records.append({
                "Group": group_name,
                "Env": f"Env {env_idx}",
                "Seed": seed,
                "Makespan_Cadence1": f"{mk1:.2f}",
                "Tardiness_Cadence1": f"{td1:.2f}",
                "Releases_Cadence1": rel1
            })
            
    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
            
    print(f"\nSuccess! All 40 instances evaluated under Cadence 1 and written to: {output_csv}")

if __name__ == "__main__":
    main()
