import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_config_dir = os.path.join(project_root, "yaml_config")
    
    files_updated = 0
    print(f"Updating configs in {yaml_config_dir}...")
    
    for filename in os.listdir(yaml_config_dir):
        if filename.startswith("eval_baseline_seed") and filename.endswith(".yml"):
            file_path = os.path.join(yaml_config_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            updated_lines = []
            for line in lines:
                # Check for hl_env_scenario
                if line.strip().startswith("hl_env_scenario:"):
                    updated_lines.append("hl_env_scenario: bottleneck_order\n")
                # Check for hl_ppo_model_path
                elif line.strip().startswith("hl_ppo_model_path:"):
                    updated_lines.append("hl_ppo_model_path: trained_weights\\high_level\\hlgate_scn_bottleneck_newstate_s07_p15_e8.pth\n")
                else:
                    updated_lines.append(line)
                    
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                f.writelines(updated_lines)
                
            print(f"  [UPDATED] {filename}")
            files_updated += 1
            
    print(f"Batch update completed. Total files updated: {files_updated}")

if __name__ == "__main__":
    main()
