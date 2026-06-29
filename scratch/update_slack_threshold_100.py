import os

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
            content = f.read()
            
        # Replace the threshold parameter
        new_content = content.replace("hl_buffer_slack_release_threshold: 0.0", "hl_buffer_slack_release_threshold: 100.0")
        
        # In case it is written as integer or different spacing
        if new_content == content:
            new_content = content.replace("hl_buffer_slack_release_threshold: 0", "hl_buffer_slack_release_threshold: 100.0")
            
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_content)
        print(f"  Updated {filename}")

if __name__ == "__main__":
    main()
