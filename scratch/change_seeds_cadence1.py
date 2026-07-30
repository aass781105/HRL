import os

dest_dir = "yaml_config"
files = ["eval_baseline_seed1_greedy_cadence1_1run.yml", "eval_baseline_seed10_greedy_cadence1_1run.yml"]

for filename in files:
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
        
    # Read the file line-by-line
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    # Modify line-by-line based on matching key prefixes
    for line in lines:
        if line.startswith("hl_gate_policy:"):
            new_lines.append("hl_gate_policy: cadence\n")
        elif line.startswith("hl_gate_cadence:"):
            new_lines.append("hl_gate_cadence: 1\n")
        elif line.startswith("baseline_cadence:"):
            new_lines.append("baseline_cadence: 1\n")
        else:
            new_lines.append(line)
            
    # Write the modified lines back
    with open(filepath, "w", encoding="utf-8", newline="\r\n") as f:
        f.writelines(new_lines)
    print(f"Updated {filename} to cadence 1 line-by-line.")
