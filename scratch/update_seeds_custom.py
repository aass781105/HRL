import os
import sys

# Dynamically find the folder starting with 'yaml' but not related to 'config'
dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('yaml') and 'config' not in d]
if not dirs:
    print("Error: Target folder not found!")
    sys.exit(1)

dir_path = dirs[0]
print(f"Dynamically detected folder: {repr(dir_path)}")

files = [f for f in os.listdir(dir_path) if f.startswith("eval_baseline_seed") and f.endswith(".yml")]

for filename in files:
    filepath = os.path.join(dir_path, filename)
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
