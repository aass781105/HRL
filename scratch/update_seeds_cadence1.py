import os

dir_path = "yaml_config"
files = [f for f in os.listdir(dir_path) if f.startswith("eval_baseline_seed") and f.endswith(".yml")]

for filename in files:
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if line.startswith("hl_gate_cadence:"):
            new_lines.append("hl_gate_cadence: 1\n")
        elif line.startswith("baseline_cadence:"):
            new_lines.append("baseline_cadence: 1\n")
        else:
            new_lines.append(line)
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Updated {filename} to cadence 1.")
