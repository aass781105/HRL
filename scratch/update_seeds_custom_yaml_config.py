import os

dir_path = "yaml_config"

for i in range(1, 11):
    filename = f"eval_baseline_seed{i}_greedy_cadence1_1run.yml"
    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if line.startswith("hl_env_scenario:"):
                new_lines.append("hl_env_scenario: custom\n")
            else:
                new_lines.append(line)
                
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Updated {filename} to custom scenario.")
    else:
        print(f"File not found: {filepath}")
