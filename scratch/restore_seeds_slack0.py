import os
import re

dest_dir = "yaml_config"

# Update seed 1 and seed 10 with custom scenario, weights, and slack0 policy
seeds = [1, 10]
for s in seeds:
    filename = f"eval_baseline_seed{s}_greedy_cadence1_1run.yml"
    filepath = os.path.join(dest_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        continue
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace hl_gate_policy to slack_threshold
    content = re.sub(r"^hl_gate_policy\s*:.*$", "hl_gate_policy: slack_threshold", content, flags=re.MULTILINE)
    # Replace hl_buffer_slack_release_threshold to 0.0
    content = re.sub(r"^hl_buffer_slack_release_threshold\s*:.*$", "hl_buffer_slack_release_threshold: 0.0", content, flags=re.MULTILINE)
    
    # Replace hl_ppo_model_path to custom path (escaping backslashes for re.sub)
    weight_path = r"hl_ppo_model_path: trained_weights\high_level\hlgate_custom_seedenv_newstate_immediate_s01_e8.pth"
    weight_path_escaped = weight_path.replace("\\", "\\\\")
    content = re.sub(r"^hl_ppo_model_path\s*:.*$", weight_path_escaped, content, flags=re.MULTILINE)
    
    # Replace hl_env_scenario to custom
    content = re.sub(r"^hl_env_scenario\s*:.*$", "hl_env_scenario: custom", content, flags=re.MULTILINE)
    
    with open(filepath, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(content)
    print(f"Updated {filename} successfully.")
