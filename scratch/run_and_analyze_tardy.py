import os
import re
import csv
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

# Config directory and paths
config_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\yaml_config"
workspace_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
global_plots_dir = os.path.join(workspace_dir, "plots", "global")
python_exe = r"C:\Users\123\anaconda3\envs\newest_environment\python.exe"

seeds = [1, 5]
policies = {
    "cadence5": {
        "hl_gate_policy": "cadence",
        "hl_gate_cadence": 5,
        "baseline_cadence": 5,
        "hl_buffer_slack_release_threshold": 0.0
    },
    "slack_0": {
        "hl_gate_policy": "slack_threshold",
        "hl_gate_cadence": 5,
        "baseline_cadence": 5,
        "hl_buffer_slack_release_threshold": 0.0
    },
    "slack_100": {
        "hl_gate_policy": "slack_threshold",
        "hl_gate_cadence": 5,
        "baseline_cadence": 5,
        "hl_buffer_slack_release_threshold": 100.0
    }
}


def backup_config(seed_num):
    filepath = os.path.join(config_dir, f"eval_baseline_seed{seed_num}_greedy_cadence1_1run.yml")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def restore_config(seed_num, content):
    filepath = os.path.join(config_dir, f"eval_baseline_seed{seed_num}_greedy_cadence1_1run.yml")
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

def modify_config(seed_num, policy_name, policy_cfg):
    filepath = os.path.join(config_dir, f"eval_baseline_seed{seed_num}_greedy_cadence1_1run.yml")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Modify parameters
    content = re.sub(r"^hl_gate_policy\s*:.*$", f"hl_gate_policy: {policy_cfg['hl_gate_policy']}", content, flags=re.MULTILINE)
    content = re.sub(r"^hl_gate_cadence\s*:.*$", f"hl_gate_cadence: {policy_cfg['hl_gate_cadence']}", content, flags=re.MULTILINE)
    content = re.sub(r"^baseline_cadence\s*:.*$", f"baseline_cadence: {policy_cfg['baseline_cadence']}", content, flags=re.MULTILINE)
    content = re.sub(r"^hl_buffer_slack_release_threshold\s*:.*$", f"hl_buffer_slack_release_threshold: {policy_cfg['hl_buffer_slack_release_threshold']}", content, flags=re.MULTILINE)
    
    # Force fast_mode: false for CSV details generation
    content = re.sub(r"^fast_mode\s*:.*$", "fast_mode: false", content, flags=re.MULTILINE)
    
    # Force single run
    content = re.sub(r"^main_sample_runs\s*:.*$", "main_sample_runs: 1", content, flags=re.MULTILINE)
    content = re.sub(r"^eval_runs_per_instance\s*:.*$", "eval_runs_per_instance: 1", content, flags=re.MULTILINE)

    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

def run_simulation(seed_num):
    config_name = f"eval_baseline_seed{seed_num}_greedy_cadence1_1run.yml"
    cmd = [python_exe, "hrl_main.py", "--config", f"yaml_config/{config_name}"]
    print(f"Running simulation: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=workspace_dir, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"Error running simulation for seed {seed_num}:\n{result.stderr}")
    return result.stdout

def get_latest_folder(before_folders):
    after_folders = set(os.listdir(global_plots_dir))
    new_folders = list(after_folders - before_folders)
    if new_folders:
        # Return the one with the newest modification time or name
        new_folders.sort(key=lambda x: os.path.getmtime(os.path.join(global_plots_dir, x)))
        return new_folders[-1]
    return None

def get_final_details_csv(folder_path):
    # Find all CSV files matching details_r*.csv
    files = [f for f in os.listdir(folder_path) if f.startswith("details_r") and f.endswith(".csv")]
    if not files:
        return None
    
    # Find the file with the highest r number
    highest_r = -1
    best_file = None
    for f in files:
        m = re.match(r"details_r(\d+)_", f)
        if m:
            r_num = int(m.group(1))
            if r_num > highest_r:
                highest_r = r_num
                best_file = f
    if best_file:
        return os.path.join(folder_path, best_file)
    return None

def extract_tardy_jobs(csv_path):
    tardy_jobs = set()
    all_jobs = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = int(row["job"])
            all_jobs.add(job_id)
            # Tardiness is stored as float string in details CSV
            # Note: tardiness column is only non-zero on the last op of the job
            tardiness = float(row.get("tardiness", 0.0))
            if tardiness > 0.0:
                tardy_jobs.add(job_id)
    return all_jobs, tardy_jobs

def main():
    # Make sure global_plots_dir exists
    if not os.path.exists(global_plots_dir):
        os.makedirs(global_plots_dir)

    results = {} # seed_num -> {policy_name -> set_of_tardy_jobs}
    seed_all_jobs = {} # seed_num -> set_of_all_jobs

    for seed_num in seeds:
        print(f"\n=================== Processing Seed {seed_num} ===================")
        results[seed_num] = {}
        
        # Backup original config
        original_config = backup_config(seed_num)
        
        try:
            for policy_name, policy_cfg in policies.items():
                print(f"\n--- Policy: {policy_name} ---")
                
                # Get current folders list before run
                before_folders = set(os.listdir(global_plots_dir))
                
                # Modify config
                modify_config(seed_num, policy_name, policy_cfg)
                
                # Run simulation
                run_simulation(seed_num)
                
                # Detect the output folder
                output_folder = get_latest_folder(before_folders)
                if not output_folder:
                    print(f"Error: No new output folder detected for Seed {seed_num}, Policy {policy_name}!")
                    continue
                
                folder_path = os.path.join(global_plots_dir, output_folder)
                print(f"Output folder detected: {output_folder}")
                
                # Find the final details CSV
                csv_path = get_final_details_csv(folder_path)
                if not csv_path:
                    print(f"Error: No details CSV found in {folder_path}!")
                    continue
                
                print(f"Found final details CSV: {os.path.basename(csv_path)}")
                
                # Extract tardy jobs
                all_jobs, tardy_jobs = extract_tardy_jobs(csv_path)
                print(f"Total jobs: {len(all_jobs)}, Tardy jobs count: {len(tardy_jobs)}")
                
                results[seed_num][policy_name] = tardy_jobs
                if seed_num not in seed_all_jobs:
                    seed_all_jobs[seed_num] = all_jobs
                else:
                    seed_all_jobs[seed_num].update(all_jobs)
                    
        finally:
            # Restore original config
            restore_config(seed_num, original_config)
            print(f"Restored configuration for Seed {seed_num}")

    # Process categorization and plot
    for seed_num in seeds:
        print(f"\n=================== Categorization Analysis for Seed {seed_num} ===================")
        all_jobs = list(seed_all_jobs.get(seed_num, set()))
        if not all_jobs:
            print(f"No jobs found for Seed {seed_num}!")
            continue
            
        tardy_sets = results[seed_num]
        
        # We need to make sure all policies ran successfully for this seed
        ran_policies = list(tardy_sets.keys())
        print(f"Policies successfully evaluated: {ran_policies}")
        if len(ran_policies) < len(policies):
            print("Skipping categorization because some policies failed.")
            continue

            
        always_tardy = []
        policy_sensitive = []
        never_tardy = []
        
        for job_id in all_jobs:
            tardy_in_policies = [policy for policy in ran_policies if job_id in tardy_sets[policy]]
            
            if len(tardy_in_policies) == len(ran_policies):
                always_tardy.append(job_id)
            elif len(tardy_in_policies) > 0:
                policy_sensitive.append(job_id)
            else:
                never_tardy.append(job_id)
                
        print(f"Always Tardy Count: {len(always_tardy)}")
        print(f"Policy Sensitive Count: {len(policy_sensitive)}")
        print(f"Never Tardy Count: {len(never_tardy)}")
        print(f"Total Unique Jobs: {len(all_jobs)}")
        
        # Plot separate chart
        categories = ['Always Tardy', 'Policy Sensitive', 'Never Tardy']
        counts = [len(always_tardy), len(policy_sensitive), len(never_tardy)]
        percentages = [c / len(all_jobs) * 100 for c in counts]
        
        plt.figure(figsize=(8, 6))
        colors = ['#e74c3c', '#f1c40f', '#2ecc71'] # Sleek red, yellow, green
        
        bars = plt.bar(categories, counts, color=colors, edgecolor='none', width=0.55)
        
        # Style layout
        plt.title(f'Tardy Jobs Classification Analysis - Seed {seed_num}', fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Number of Jobs', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels on top of bars
        for bar, pct in zip(bars, percentages):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts) * 0.015), 
                     f'{int(yval)} ({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.ylim(0, max(counts) * 1.15)
        plt.tight_layout()
        
        plot_path = os.path.join(workspace_dir, "plots", f"seed{seed_num}_tardy_categories.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved bar chart for Seed {seed_num} to: {plot_path}")

        # Also write a text summary file in the workspace
        txt_path = os.path.join(workspace_dir, f"tardy_summary_seed{seed_num}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Tardy Job Analysis Summary for Seed {seed_num}\n")
            f.write("=========================================\n")
            f.write(f"Total Unique Jobs: {len(all_jobs)}\n")
            f.write(f"Always Tardy (Always Tardy in all 4 policies): {len(always_tardy)} ({percentages[0]:.2f}%)\n")
            f.write(f"Policy Sensitive (Tardy in some policies): {len(policy_sensitive)} ({percentages[1]:.2f}%)\n")
            f.write(f"Never Tardy (Never Tardy in all 4 policies): {len(never_tardy)} ({percentages[2]:.2f}%)\n\n")
            f.write(f"Always Tardy Job IDs: {sorted(always_tardy)}\n")
            f.write(f"Policy Sensitive Job IDs: {sorted(policy_sensitive)}\n")
            f.write(f"Never Tardy Job IDs: {sorted(never_tardy)}\n")
        print(f"Saved text report for Seed {seed_num} to: {txt_path}")

if __name__ == "__main__":
    main()
