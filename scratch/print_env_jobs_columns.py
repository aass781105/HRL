import os
import pandas as pd

def main():
    project_root = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(project_root))
    meeting_dir = os.path.join(parent_dir, "meeting_ppt", "20260708")
    
    env_jobs_file = os.path.join(meeting_dir, "slack0瓶頸300", "20260706_180847_odprog_seed008", "odprog_env_jobs.csv")
    if os.path.exists(env_jobs_file):
        df = pd.read_csv(env_jobs_file)
        print("All columns:")
        print(df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

if __name__ == "__main__":
    main()
