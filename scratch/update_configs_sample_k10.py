import os
import re

config_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\yaml_config"

def main():
    if not os.path.exists(config_dir):
        print(f"Directory not found: {config_dir}")
        return
        
    files = [f for f in os.listdir(config_dir) if f.endswith(".yml") and "eval_baseline_seed" in f]
    print(f"Found {len(files)} config files to update:")
    
    for filename in files:
        filepath = os.path.join(config_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if re.match(r"^ll_eval_action_selection\s*:", line):
                line = "ll_eval_action_selection: sample\n"
            elif re.match(r"^ll_rollout_k\s*:", line):
                line = "ll_rollout_k: 10\n"
            new_lines.append(line)
            
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(new_lines)
        print(f"  Successfully updated {filename} to sample, k=10")

if __name__ == "__main__":
    main()
