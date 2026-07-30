import os
import yaml
import pandas as pd

project_root = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
yaml_dir = os.path.join(project_root, "yaml_config")

# Load all 10 yaml configs
configs_dict = {}
for i in range(1, 11):
    filename = f"eval_baseline_seed{i}_greedy_cadence1_1run.yml"
    filepath = os.path.join(yaml_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            configs_dict[f"seed{i}"] = cfg

# Find all unique keys across all configs
all_keys = set()
for cfg in configs_dict.values():
    all_keys.update(cfg.keys())

# Compare values for each key
differences = {}
for key in sorted(all_keys):
    # Get values for this key across all 10 seeds
    values = {seed: configs_dict[seed].get(key) for seed in configs_dict.keys()}
    
    # Check if there are different values (ignoring event_seed which is naturally different)
    unique_values = set(values.values())
    if len(unique_values) > 1 and key != "event_seed":
        differences[key] = values

# Print results
if not differences:
    print("\nAll 10 seeds have identical configurations (excluding event_seed)!")
else:
    print(f"\nFound {len(differences)} parameter differences across the 10 seeds:")
    # Build a DataFrame for nice display
    df = pd.DataFrame(differences).T
    print(df.to_string())
