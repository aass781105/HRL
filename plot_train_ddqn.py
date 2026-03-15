# plot_train_ddqn.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from params import configs

def plot_ddqn_training():
    # 1. Setup paths
    plot_name = getattr(configs, "ddqn_name", "ddqn_gate_latest")
    log_dir = os.path.join("plots", "train_ddqn")
    csv_path = os.path.join(log_dir, f"log_{plot_name}.csv")
    output_path = os.path.join(log_dir, f"plot_{plot_name}.png")

    if not os.path.exists(csv_path):
        print(f"[ERROR] DDQN log file not found at: {csv_path}")
        print("Please ensure train_ddqn.py is running and saving logs.")
        return

    # 2. Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    if df.empty:
        print("[WARN] Log file is empty.")
        return

    # 3. Create Plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f"DDQN Training Analysis: {plot_name}", fontsize=16)

    # Window for moving average
    window = max(1, len(df) // 10)

    # --- Subplot 1: Episode Return ---
    ax1 = axes[0]
    ax1.plot(df['episode'], df['return'], color='blue', alpha=0.3, label='Raw Return')
    ax1.plot(df['episode'], df['return'].rolling(window=window).mean(), color='red', linewidth=2, label=f'MA (w={window})')
    ax1.set_title("Learning Progress: Episode Return")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Makespan ---
    ax2 = axes[1]
    ax2.plot(df['episode'], df['makespan'], color='green', marker='o', markersize=2, alpha=0.6)
    ax2.set_title("System Efficiency: Average Makespan")
    ax2.set_ylabel("Makespan (min)")
    ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Tardiness & Release Count ---
    ax3 = axes[2]
    ax3.plot(df['episode'], df['tardiness'], color='orange', label='Total Tardiness')
    ax3.set_ylabel("Tardiness (min)", color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    
    ax3_r = ax3.twinx()
    ax3_r.plot(df['episode'], df['release_count'], color='purple', linestyle='--', alpha=0.5, label='Release Count')
    ax3_r.set_ylabel("Avg Releases", color='purple')
    ax3_r.tick_params(axis='y', labelcolor='purple')
    
    ax3.set_title("Customer Satisfaction & Agent Activity")
    ax3.set_xlabel("Training Batch (10 Episodes each)")
    
    # Merge legends
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    print(f"✅ DDQN 訓練分析圖已儲存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_ddqn_training()
