
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_simulation_summary_stats(all_data, output_dir):
    """
    Plots boxplots for Due Dates and Slacks collected during simulation.
    all_data: List of dicts containing 'due_date' and 'slack'
    """
    if not all_data:
        print("[WARN] No data collected for summary plots.")
        return

    df = pd.DataFrame(all_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Boxplot of Due Dates
    df[['due_date']].boxplot(ax=axes[0])
    axes[0].grid(False)
    axes[0].set_title('Distribution of All Job Due Dates')
    axes[0].set_ylabel('Absolute Time')
    
    # Plot 2: Boxplot of Slacks
    df[['slack']].boxplot(ax=axes[1])
    axes[1].grid(False)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Distribution of All Job Slacks (Due - Ready - RemainingWork)')
    axes[1].set_ylabel('Buffer Time')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'simulation_summary_boxplots.png')
    plt.savefig(save_path)
    print(f"\n[INFO] Summary boxplots saved to '{save_path}'")
    # plt.show() # Uncomment if you want to see it immediately
