import os
import re

config_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\yaml_config"

# Targets: seed1 to seed10
filenames = [f"eval_baseline_seed{i}_greedy_cadence1_1run.yml" for i in range(1, 11)]

replacements = {
    r"^hl_gate_policy\s*:.*$": "hl_gate_policy: slack_threshold",
    r"^hl_buffer_slack_release_threshold\s*:.*$": "hl_buffer_slack_release_threshold: 0.0"
}

for fname in filenames:
    path = os.path.join(config_dir, fname)
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        continue
        
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    updated = {k: False for k in replacements.keys()}
    
    for line in lines:
        matched = False
        for pattern, replacement in replacements.items():
            if re.match(pattern, line.strip()):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}{replacement}\n")
                matched = True
                updated[pattern] = True
                break
        if not matched:
            new_lines.append(line)
            
    for pattern, replacement in replacements.items():
        if not updated[pattern]:
            new_lines.append(f"{replacement}\n")
            
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Updated policy to slack_threshold in: {fname}")

print("All config files updated to slack_threshold successfully!")
