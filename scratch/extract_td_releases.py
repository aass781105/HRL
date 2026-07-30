import os
import pandas as pd

root_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260715\新環境情境"
policies = ["PPO", "cad1", "cad5", "slack0"]

# We will read each comparison file and extract the 'Tardiness / Releases' column
combined_data = {}

for policy in policies:
    filepath = os.path.join(root_dir, f"{policy}_10seeds_comparison.csv")
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        continue
        
    df = pd.read_csv(filepath)
    
    # Store 'Seed' column once
    if "Seed" not in combined_data:
        combined_data["Seed"] = df["Seed"].tolist()
        
    # Store the target column named as the policy
    combined_data[policy] = df["Tardiness / Releases"].tolist()

# Create summary DataFrame
df_summary = pd.DataFrame(combined_data)

# Save to target CSV
output_path = os.path.join(root_dir, "tardiness_releases_comparison.csv")
df_summary.to_csv(output_path, index=False)
print(f"Successfully generated comparison CSV: {output_path}")
