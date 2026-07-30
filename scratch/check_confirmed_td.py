import os
import pandas as pd

def check_file(file_path, label):
    if not os.path.exists(file_path):
        print(f"{label} file not found!")
        return
        
    df = pd.read_csv(file_path)
    # Check by Seed
    decrease_instances = []
    print(f"\n--- Checking {label} ---")
    
    for seed in sorted(df["Seed"].unique()):
        df_seed = df[df["Seed"] == seed].sort_values(by="Event_ID")
        confirmed_vals = df_seed["Confirmed_TD_Before"].values
        event_ids = df_seed["Event_ID"].values
        
        # Check if it ever decreases
        for i in range(1, len(confirmed_vals)):
            if confirmed_vals[i] < confirmed_vals[i-1]:
                decrease_instances.append((seed, event_ids[i-1], confirmed_vals[i-1], event_ids[i], confirmed_vals[i]))
                
        print(f"Seed {seed:2d}: Min={confirmed_vals.min():.2f}, Max={confirmed_vals.max():.2f}, Final={confirmed_vals[-1]:.2f}")
        
    if decrease_instances:
        print(f"🔴 WARNING: Found {len(decrease_instances)} instances where Confirmed_TD_Before DECREASED!")
        for inst in decrease_instances[:5]:
            print(f"  Seed {inst[0]}: Event {inst[1]} ({inst[2]:.2f}) -> Event {inst[3]} ({inst[4]:.2f})")
    else:
        print("🟢 SUCCESS: Confirmed_TD_Before never decreases! It is strictly monotonically increasing.")

def main():
    target_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\analysis_results\policy_compare\20260708_153037"
    check_file(os.path.join(target_dir, "decision_trace_ppo.csv"), "PPO Trace")
    check_file(os.path.join(target_dir, "decision_trace_slack0.csv"), "Slack0 Trace")

if __name__ == "__main__":
    main()
