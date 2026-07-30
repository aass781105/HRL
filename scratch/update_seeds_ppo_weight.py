import os

dir_path = "yaml_config"
files = [f for f in os.listdir(dir_path) if f.startswith("eval_baseline_seed") and f.endswith(".yml")]

for filename in files:
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if line.startswith("hl_gate_policy:"):
            new_lines.append("hl_gate_policy: ppo\n")
        elif line.startswith("hl_ppo_model_path:"):
            new_lines.append(r"hl_ppo_model_path: trained_weights\high_level\hlgate_custom_seedenv_newstate_immediate_s01_e8.pth" + "\n")
        else:
            new_lines.append(line)
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Updated {filename} to PPO policy and weight path.")
