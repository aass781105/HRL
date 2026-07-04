import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = "train_hl_gate_scn_baseline_env8.yml"
    
    found_path = None
    for root, dirs, files in os.walk(project_root):
        # Skip heavy dirs
        dirs[:] = [d for d in dirs if d not in ('.git', 'trained_network', 'plots', 'ppo_ckpt')]
        if target_file in files:
            found_path = os.path.join(root, target_file)
            break
            
    if not found_path:
        print(f"Could not find {target_file}")
        return
        
    print(f"Reading file: {found_path}")
    with open(found_path, "r", encoding="utf-8") as f:
        print(f.read())

if __name__ == "__main__":
    main()
