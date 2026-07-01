import subprocess
import os
import sys

python_exe = r"C:\Users\123\anaconda3\envs\newest_environment\python.exe"
config_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\yaml_config"

def main():
    for s in range(1, 11):
        config_path = os.path.join(config_dir, f"eval_baseline_seed{s}_greedy_cadence1_1run.yml")
        cmd = [python_exe, "hrl_main.py", "--config", config_path]
        print(f"[{s}/10] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error on seed {s}", file=sys.stderr)
            sys.exit(result.returncode)
    print("All 10 seeds evaluated successfully.")

if __name__ == "__main__":
    main()
