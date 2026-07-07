import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_config_dir = os.path.join(project_root, "yaml_config")
    
    # Target path from last step to replace with the new path
    old_target = "trained_weights\\high_level\\hlgate_scn_base_newstate_deltagap_actormask_criticextra_s07_p15_e8.pth"
    new_target = "trained_weights\\high_level\\hlgate_scn_base_s05_e16.pth"
    
    files_updated = 0
    
    print(f"Scanning configs in {yaml_config_dir}...")
    for filename in os.listdir(yaml_config_dir):
        if filename.startswith("eval_baseline_seed") and filename.endswith(".yml"):
            file_path = os.path.join(yaml_config_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            if old_target in content:
                new_content = content.replace(old_target, new_target)
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"  [UPDATED] {filename}")
                files_updated += 1
            else:
                fallback_targets = [
                    "ppo_ckpt\\hlgate_scn_base_s05_e4.pth",
                    "trained_weights\\high_level\\hlgate_scn_base_s05_e8.pth",
                    "trained_weights\\high_level\\hlgate_scn_base_s05_p15_e4.pth",
                    "trained_weights\\high_level\\hlgate_scn_base_newstate_s07_p15_e8.pth"
                ]
                updated = False
                for target in fallback_targets:
                    if target in content:
                        new_content = content.replace(target, new_target)
                        with open(file_path, "w", newline="", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"  [UPDATED via fallback] {filename}")
                        files_updated += 1
                        updated = True
                        break
                if not updated:
                    print(f"  [SKIPPED] {filename} (target path not found)")
                
    print(f"Batch update completed. Total files updated: {files_updated}")

if __name__ == "__main__":
    main()
