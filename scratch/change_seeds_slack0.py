import os
import re

dir_path = "yaml_config"
files = ["eval_baseline_seed1_greedy_cadence1_1run.yml", "eval_baseline_seed10_greedy_cadence1_1run.yml"]

for filename in files:
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace hl_gate_policy to slack_threshold
    content = re.sub(r"^hl_gate_policy\s*:.*$", "hl_gate_policy: slack_threshold", content, flags=re.MULTILINE)
    # Replace hl_buffer_slack_release_threshold to 0.0
    content = re.sub(r"^hl_buffer_slack_release_threshold\s*:.*$", "hl_buffer_slack_release_threshold: 0.0", content, flags=re.MULTILINE)
    
    with open(filepath, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(content)
    print(f"Updated {filename} to slack0 policy.")
