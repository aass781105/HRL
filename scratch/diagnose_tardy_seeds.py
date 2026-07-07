import os
import pandas as pd

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    
    # We will look at slack0瓶頸300 runs
    slack_path = os.path.join(meeting_dir, "slack0瓶頸300")
    
    # Target seeds to compare: Seed 8 (extremely high TD: 19378) and Seed 9 (extremely low TD: 1242)
    seeds_to_compare = {
        8: "20260706_180847_odprog_seed008",
        9: "20260706_180921_odprog_seed009",
        2: "20260706_180602_odprog_seed002"
    }
    
    for seed, folder in seeds_to_compare.items():
        folder_path = os.path.join(slack_path, folder)
        print(f"\n================ DIAGNOSING SEED {seed} (Folder: {folder}) ================")
        if not os.path.exists(folder_path):
            print("Folder does not exist!")
            continue
            
        # Read the raw state to see the max load on machines during the run
        raw_state_file = os.path.join(folder_path, "odprog_raw_state.csv")
        if os.path.exists(raw_state_file):
            df_raw = pd.read_csv(raw_state_file)
            print(f"Total decision steps in run: {len(df_raw)}")
            # Get the final row's metrics
            last_row = df_raw.iloc[-1]
            print(f"Final WIP Count: {last_row.get('Raw_WIP_Job_Count', 'N/A')}")
            print(f"Final WIP Tardy Ratio: {last_row.get('Raw_WIP_Tardy_Ratio', 'N/A')}")
            print(f"Final Planned Tardiness: {last_row.get('Raw_WIP_Planned_TD', 'N/A')}")
            print(f"Final Actual Tardiness (Cumulative): {last_row.get('Actual_TD', 'N/A')}")
            print(f"Final Release Count: {last_row.get('Release_Count', 'N/A')}")
            
        # Read the env_jobs list if it exists to analyze the job tardiness distribution
        env_jobs_file = os.path.join(folder_path, "odprog_env_jobs.csv")
        if os.path.exists(env_jobs_file):
            # Wait, let's check what headers env_jobs has
            try:
                df_jobs = pd.read_csv(env_jobs_file)
                print("Job count in env_jobs:", len(df_jobs))
                print("Headers in env_jobs:", df_jobs.columns.tolist()[:10])
                # Print stats of jobs: finished tardiness, etc.
                # If there is a column for tardiness, makespan, due date, end time
                tardy_jobs = df_jobs[df_jobs["tardiness"] > 0] if "tardiness" in df_jobs.columns else []
                print(f"Number of tardy jobs: {len(tardy_jobs)}")
                if len(tardy_jobs) > 0:
                    print("Top 5 most tardy jobs:")
                    print(tardy_jobs.sort_values(by="tardiness", ascending=False)[["job_id", "is_urgent", "due_date", "tardiness"]].head(5).to_string(index=False))
                    print(f"Sum of tardiness of all tardy jobs: {tardy_jobs['tardiness'].sum():.2f}")
            except Exception as e:
                print("Error reading env_jobs:", e)

if __name__ == "__main__":
    main()
