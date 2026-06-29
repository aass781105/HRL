import os
import re
import csv
import matplotlib.pyplot as plt

workspace_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"
global_plots_dir = os.path.join(workspace_dir, "plots", "global")

# Existing folders mapped to policies
run_folders = {
    1: {
        "cadence5": "20260623_113314_odprog_seed001",
        "slack_0": "20260623_113419_odprog_seed001",
        "slack_100": "20260623_113518_odprog_seed001"
    },
    5: {
        "cadence5": "20260623_114136_odprog_seed005",
        "slack_0": "20260623_114243_odprog_seed005",
        "slack_100": "20260623_114341_odprog_seed005"
    }
}

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
            tardiness = float(row.get("tardiness", 0.0))
            if tardiness > 0.0:
                tardy_jobs.add(job_id)
    return all_jobs, tardy_jobs

def main():
    for seed_num, policies in run_folders.items():
        print(f"\n=================== Analyzing Seed {seed_num} (Excluded Cadence 1) ===================")
        results = {}
        seed_all_jobs = set()
        
        for policy_name, folder_name in policies.items():
            folder_path = os.path.join(global_plots_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"Error: Folder {folder_path} does not exist!")
                continue
                
            csv_path = get_final_details_csv(folder_path)
            if not csv_path:
                print(f"Error: No details CSV found in {folder_path}!")
                continue
                
            all_jobs, tardy_jobs = extract_tardy_jobs(csv_path)
            results[policy_name] = tardy_jobs
            seed_all_jobs.update(all_jobs)
            print(f"  Policy {policy_name} -> Total jobs: {len(all_jobs)}, Tardy jobs count: {len(tardy_jobs)}")
            
        all_jobs_list = sorted(list(seed_all_jobs))
        if not all_jobs_list:
            print("No jobs found, skipping.")
            continue
            
        ran_policies = list(results.keys())
        always_tardy = []
        policy_sensitive = []
        never_tardy = []
        
        for job_id in all_jobs_list:
            tardy_in_policies = [policy for policy in ran_policies if job_id in results[policy]]
            
            if len(tardy_in_policies) == len(ran_policies):
                always_tardy.append(job_id)
            elif len(tardy_in_policies) > 0:
                policy_sensitive.append(job_id)
            else:
                never_tardy.append(job_id)
                
        print(f"  Always Tardy Count: {len(always_tardy)}")
        print(f"  Policy Sensitive Count: {len(policy_sensitive)}")
        print(f"  Never Tardy Count: {len(never_tardy)}")
        print(f"  Total Unique Jobs: {len(all_jobs_list)}")
        
        # Plot separate chart
        categories = ['Always Tardy', 'Policy Sensitive', 'Never Tardy']
        counts = [len(always_tardy), len(policy_sensitive), len(never_tardy)]
        percentages = [c / len(all_jobs_list) * 100 for c in counts]
        
        plt.figure(figsize=(8, 6))
        colors = ['#e74c3c', '#f1c40f', '#2ecc71'] # Sleek red, yellow, green
        
        bars = plt.bar(categories, counts, color=colors, edgecolor='none', width=0.55)
        
        # Style layout
        plt.title(f'Tardy Jobs Classification Analysis - Seed {seed_num}\n(Policies: cadence5, slack_0, slack_100)', fontsize=13, fontweight='bold', pad=15)
        plt.ylabel('Number of Jobs', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels on top of bars
        for bar, pct in zip(bars, percentages):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts) * 0.015), 
                     f'{int(yval)} ({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.ylim(0, max(counts) * 1.15)
        plt.tight_layout()
        
        plot_path = os.path.join(workspace_dir, "plots", f"seed{seed_num}_tardy_categories_no_cadence1.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved bar chart for Seed {seed_num} to: {plot_path}")

        # Also write a text summary file in the workspace
        txt_path = os.path.join(workspace_dir, f"tardy_summary_seed{seed_num}_no_cadence1.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Tardy Job Analysis Summary for Seed {seed_num} (Excluded Cadence 1)\n")
            f.write("===============================================================\n")
            f.write(f"Total Unique Jobs: {len(all_jobs_list)}\n")
            f.write(f"Always Tardy (Always Tardy in all 3 policies): {len(always_tardy)} ({percentages[0]:.2f}%)\n")
            f.write(f"Policy Sensitive (Tardy in some policies): {len(policy_sensitive)} ({percentages[1]:.2f}%)\n")
            f.write(f"Never Tardy (Never Tardy in all 3 policies): {len(never_tardy)} ({percentages[2]:.2f}%)\n\n")
            f.write(f"Always Tardy Job IDs: {sorted(always_tardy)}\n")
            f.write(f"Policy Sensitive Job IDs: {sorted(policy_sensitive)}\n")
            f.write(f"Never Tardy Job IDs: {sorted(never_tardy)}\n")
        print(f"  Saved text report for Seed {seed_num} to: {txt_path}")

if __name__ == "__main__":
    main()
