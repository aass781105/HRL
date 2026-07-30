import os
import shutil
import pandas as pd

def main():
    target_dir = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\analysis_results\policy_compare\20260708_153037"
    
    if not os.path.exists(target_dir):
        print(f"Target directory {target_dir} does not exist!")
        return
        
    # 1. Split decision_trace.csv
    trace_path = os.path.join(target_dir, "decision_trace.csv")
    if os.path.exists(trace_path):
        df_trace = pd.read_csv(trace_path)
        df_trace[df_trace["Policy"] == "ppo"].to_csv(os.path.join(target_dir, "decision_trace_ppo.csv"), index=False)
        df_trace[df_trace["Policy"] == "slack0"].to_csv(os.path.join(target_dir, "decision_trace_slack0.csv"), index=False)
        os.remove(trace_path)
        print("Split and removed decision_trace.csv")
    else:
        print("decision_trace.csv not found or already split.")
        
    # 2. Split divergence_windows.csv
    div_path = os.path.join(target_dir, "divergence_windows.csv")
    if os.path.exists(div_path):
        df_div = pd.read_csv(div_path)
        df_div[df_div["Policy"] == "ppo"].to_csv(os.path.join(target_dir, "divergence_windows_ppo.csv"), index=False)
        df_div[df_div["Policy"] == "slack0"].to_csv(os.path.join(target_dir, "divergence_windows_slack0.csv"), index=False)
        os.remove(div_path)
        print("Split and removed divergence_windows.csv")
    else:
        print("divergence_windows.csv not found or already split.")
        
    # 3. Split release_aligned.csv
    rel_path = os.path.join(target_dir, "release_aligned.csv")
    if os.path.exists(rel_path):
        df_rel = pd.read_csv(rel_path)
        df_rel[df_rel["Policy"] == "ppo"].to_csv(os.path.join(target_dir, "release_aligned_ppo.csv"), index=False)
        df_rel[df_rel["Policy"] == "slack0"].to_csv(os.path.join(target_dir, "release_aligned_slack0.csv"), index=False)
        os.remove(rel_path)
        print("Split and removed release_aligned.csv")
    else:
        print("release_aligned.csv not found or already split.")
        
    # 4. Remove plots directory
    plots_dir = os.path.join(target_dir, "plots")
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
        print("Removed plots directory")
    else:
        print("plots directory not found or already removed.")
        
    print("\nSuccessfully finished separating all CSVs!")

if __name__ == "__main__":
    main()
