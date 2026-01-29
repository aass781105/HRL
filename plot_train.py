# plot_train.py
import json, ast, re, os
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from params import configs
from common_utils import strToSuffix

# ======= 設定 =======
OUTPUT_PLOT_DIR  = "train_log_plot"
LINE_WIDTH       = 1.5
# =========================

NUM = r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?"
PAIR_RE = re.compile(r"\[\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\]")
NUM_LIST_RE = re.compile(NUM)

def parse_content(text: str) -> Union[List[Tuple[float,float]], List[float]]:
    # ... (Keep existing parse logic) ...
    text = text.strip()
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
                return [(float(p[0]), float(p[1])) for p in data]
            if all(isinstance(v, (int, float, np.float64, np.float32)) for v in data):
                return [float(v) for v in data]
    except Exception:
        pass
    
    pairs = [(float(a), float(b)) for a, b in PAIR_RE.findall(text)]
    if pairs: return pairs
    nums = [float(n) for n in NUM_LIST_RE.findall(text)]
    if nums: return nums
    return []

def load_xy(txt_path: Path) -> List[Tuple[float,float]]:
    # ... (Keep existing load logic) ...
    if not txt_path.exists():
        return []
    text = txt_path.read_text(encoding="utf-8")
    parsed = parse_content(text)
    if not parsed: return []
    if isinstance(parsed[0], tuple):
        return parsed
    ys = parsed
    return list(enumerate(ys))

def plot_reward_components(detailed_path: Path, output_path: Path, core_name: str):
    """
    Reads detailed_reward_*.txt which contains: [ep, r, mk_mean, mk_std, td_mean, td_std]
    Plots Total Reward, Makespan Gain (with std), Tardiness Penalty (with std).
    """
    if not detailed_path.exists():
        print(f"Detailed reward log not found: {detailed_path}")
        return

    text = detailed_path.read_text(encoding="utf-8")
    
    # [FIX] Use ast.literal_eval directly as parse_content is designed for 2D plots
    try:
        data = ast.literal_eval(text)
    except Exception as e:
        print(f"Error parsing detailed log: {e}")
        return
    
    if not data or len(data) == 0 or len(data[0]) < 8:
        print("Detailed reward data format incorrect (need 8 columns).")
        return

    ep = [x[0] for x in data]
    r = [x[1] for x in data]
    mk_mean = [x[2] for x in data]
    mk_std = [x[3] for x in data]
    td_mean = [x[4] for x in data]
    td_std = [x[5] for x in data]
    raw_mk = [x[6] for x in data]
    raw_td = [x[7] for x in data]

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f"Training Analysis: {core_name}", fontsize=16)

    # 1. Raw Performance (Dual Axis)
    axes[0].set_title('Training Performance: Raw Makespan vs Tardiness')
    ln1 = axes[0].plot(ep, raw_mk, color='blue', label='Makespan (Raw)')
    axes[0].set_ylabel('Makespan', color='blue')
    axes[0].tick_params(axis='y', labelcolor='blue')
    axes[0].grid(True, alpha=0.3)
    
    ax0_r = axes[0].twinx()
    ln2 = ax0_r.plot(ep, raw_td, color='orange', label='Tardiness (Raw)')
    ax0_r.set_ylabel('Tardiness', color='orange')
    ax0_r.tick_params(axis='y', labelcolor='orange')
    
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc='upper center')

    # 2. Normalized Components (Single Axis, No Band)
    axes[1].plot(ep, mk_mean, color='green', label='Mk Gain (Norm)')
    axes[1].plot(ep, td_mean, color='red', label='Td Penalty (Norm)')
    axes[1].set_ylabel('Normalized Reward')
    axes[1].set_title('Reward Components (Learning Signal)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    # Add zero line for reference
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')

    # 3. Components Std (Stability)
    axes[2].plot(ep, mk_std, color='green', linestyle='--', label='Mk Std')
    axes[2].plot(ep, td_std, color='red', linestyle='--', label='Td Std')
    axes[2].set_ylabel('Standard Deviation')
    axes[2].set_xlabel('Episodes')
    axes[2].set_title('Reward Stability (Std Dev)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Reward 組成分析圖已輸出：{output_path}")

def plot_indist_validation(log_dir: Path, output_path: Path, log_file_suffix: str, core_name: str):
    """
    Scans for vali_indist_*_{suffix}.txt files.
    Plots one subplot per group: Solid Obj line, translucent MK and TD lines in the same plot.
    """
    # 1. Scan for files
    files = list(log_dir.glob(f"vali_indist_*_{log_file_suffix}.txt"))
    if not files:
        print("No in-distribution validation logs found.")
        return

    # Extract group names and sort
    name_to_info = {}
    for f in files:
        # Match "vali_indist_G1_nj10_k1.2_num20_..."
        match = re.search(r"vali_indist_(G\d+)_nj(\d+)_k([\d.]+)_num(\d+)_", f.name)
        if match:
            g_name, nj, k, num = match.groups()
            title_str = f'{{"name": "{g_name}", "n_j": {nj}, "k": {k}, "num": {num}}}'
            name_to_info[g_name] = (f, title_str)
    
    # Sort by group number (G1, G2...)
    sorted_names = sorted(name_to_info.keys(), key=lambda x: int(x[1:]))
    n_subplots = len(sorted_names)
    
    # 2. Setup Figure Grid
    cols = 2
    rows = (n_subplots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
    fig.suptitle(f"In-Distribution Validation: {core_name}", fontsize=16)
    
    axes_flat = axes.flatten()
    
    # 3. Plot each group
    for i, name in enumerate(sorted_names):
        ax = axes_flat[i]
        f_path, title_str = name_to_info[name]
        text = f_path.read_text(encoding="utf-8")
        try:
            data = ast.literal_eval(text)
        except:
            ax.set_title(f"Group {name} (Parse Error)")
            continue
            
        if not data: continue
        
        mk_vals = np.array([x[0] for x in data])
        td_vals = np.array([x[1] for x in data])
        obj_vals = 0.5 * mk_vals + 0.5 * td_vals
        
        steps = np.arange(len(obj_vals)) * 10
        
        ax.plot(steps, mk_vals, color='green', alpha=0.3, label='Raw Makespan')
        ax.plot(steps, td_vals, color='red', alpha=0.3, label='Raw Tardiness')
        ax.plot(steps, obj_vals, color='black', linewidth=2, label='Objective (0.5/0.5)')
        
        ax.set_title(title_str, fontsize=10) # Set title as dictionary string
        ax.set_ylabel("Absolute Time / Score")
        ax.set_xlabel("Update Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # 4. Cleanup and Save
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ In-Distribution 驗證分析圖已輸出：{output_path}")

def main():
    # 1. Construct dynamic log name
    model_name = configs.eval_model_name
    # Assuming initial n_j=10, n_m=5 from configs default or current
    # Note: train_curriculum uses 'self.initial_n_j' which defaults to 10 in Trainer.__init__
    init_nj = 10 
    n_m = configs.n_m
    suffix = strToSuffix(configs.data_suffix)
    
    # Full name used in log files
    log_file_suffix = f"{model_name}_{init_nj}x{n_m}{suffix}"
    
    # 2. Setup Paths
    log_dir = Path(f"train_log/{configs.data_source}/").expanduser().resolve()
    
    print(f"Looking for logs with suffix: {log_file_suffix} in {log_dir}")
    
    # 統一構建所有路徑
    reward_path = log_dir / f"reward_{log_file_suffix}.txt"
    detailed_path = log_dir / f"detailed_reward_{log_file_suffix}.txt" 
    ms_path     = log_dir / f"valiquality_{log_file_suffix}.txt"
    td_path     = log_dir / f"valitardiness_{log_file_suffix}.txt"
    loss_path   = log_dir / f"loss_{log_file_suffix}.txt"

    if not reward_path.exists():
        print(f"找不到 Reward 檔案：{reward_path}")
        return

    CORE_NAME = model_name # For plot title
    out_dir = Path(OUTPUT_PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [ADDED] Plot Detailed Reward Components
    if detailed_path.exists():
        comp_out_path = out_dir / f"components_{CORE_NAME}.png"
        plot_reward_components(detailed_path, comp_out_path, CORE_NAME)
        
    # [ADDED] Plot In-Distribution Validation Analysis
    indist_out_path = out_dir / f"indist_vali_{CORE_NAME}.png"
    plot_indist_validation(log_dir, indist_out_path, log_file_suffix, CORE_NAME)

    # 讀取數據
    reward_pairs = load_xy(reward_path)
    ms_pairs = load_xy(ms_path)
    td_pairs = load_xy(td_path)
    
    loss_data = []
    if loss_path.exists():
        try:
            text = loss_path.read_text(encoding="utf-8").strip()
            data = ast.literal_eval(text)
            if isinstance(data, list) and len(data) > 0:
                if len(data[0]) >= 3:
                    loss_data = data
        except:
            print(f"Error parsing loss file: {loss_path}")

    # 開始繪圖：4 個子圖
    fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=False)
    fig.suptitle(f"Training Analysis: {CORE_NAME}", fontsize=16, y=0.99)

    # ---------------------------------------------------------
    # 1. Plot Training Reward
    # ---------------------------------------------------------
    if reward_pairs:
        rx, ry = zip(*reward_pairs)
        axes[0].plot(rx, ry, color='#1f77b4', linewidth=LINE_WIDTH, label='Total Reward')
        axes[0].set_title("Training Reward (Policy Optimization Goal)")
        axes[0].set_ylabel("Reward")
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[0].set_title("Reward Data Not Found")

    # ---------------------------------------------------------
    # 2. Plot Training Loss (Dual Axis: Policy vs Value)
    # ---------------------------------------------------------
    if loss_data:
        steps = [x[0] for x in loss_data]
        v_loss = [x[2] for x in loss_data]
        
        # 判斷是否有記錄 p_loss
        if len(loss_data[0]) >= 4:
            p_loss = [x[3] for x in loss_data]
            p_label = 'Policy Loss (Real)'
        else:
            total_loss = [x[1] for x in loss_data]
            p_loss = [t - 0.5 * v for t, v in zip(total_loss, v_loss)]
            p_label = 'Policy Loss (Approx)'
        
        color_p = 'purple'
        color_v = 'orange'
        
        ln1 = axes[1].plot(steps, p_loss, color=color_p, linewidth=LINE_WIDTH, label=p_label, alpha=0.8)
        axes[1].set_ylabel("Policy Loss", color=color_p)
        axes[1].tick_params(axis='y', labelcolor=color_p)
        axes[1].set_title("Training Loss Analysis")
        
        ax2 = axes[1].twinx()
        ln2 = ax2.plot(steps, v_loss, color=color_v, linewidth=LINE_WIDTH, label='Value Loss', alpha=0.6, linestyle='--')
        ax2.set_ylabel("Value Loss", color=color_v)
        ax2.tick_params(axis='y', labelcolor=color_v)
        
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        axes[1].legend(lns, labs, loc='upper center')
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1].set_title("Loss Data Not Found")

    # ---------------------------------------------------------
    # 3. Validation: Makespan vs Tardiness (Dual Axis)
    # ---------------------------------------------------------
    if ms_pairs and td_pairs:
        # 假設兩者長度一致，或者取最短
        min_len = min(len(ms_pairs), len(td_pairs))
        vx = [p[0] for p in ms_pairs[:min_len]] # Validation Epochs (or steps)
        
        ms_vals = [p[1] for p in ms_pairs[:min_len]]
        td_vals = [p[1] for p in td_pairs[:min_len]]
        
        color_ms = '#2ca02c' # Green
        color_td = '#d62728' # Red
        
        ln3 = axes[2].plot(vx, ms_vals, color=color_ms, linewidth=LINE_WIDTH, marker='.', label='Makespan (Efficiency)')
        axes[2].set_ylabel("Makespan", color=color_ms)
        axes[2].tick_params(axis='y', labelcolor=color_ms)
        axes[2].set_title("Validation: Efficiency vs Punctuality")
        
        ax3 = axes[2].twinx()
        ln4 = ax3.plot(vx, td_vals, color=color_td, linewidth=LINE_WIDTH, marker='.', label='Tardiness (Punctuality)')
        ax3.set_ylabel("Tardiness", color=color_td)
        ax3.tick_params(axis='y', labelcolor=color_td)
        
        lns2 = ln3 + ln4
        labs2 = [l.get_label() for l in lns2]
        axes[2].legend(lns2, labs2, loc='upper center')
        axes[2].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[2].set_title("Validation Data Incomplete")

    # ---------------------------------------------------------
    # 4. Validation: Objective (0.5 * MS + 0.5 * TD)
    # ---------------------------------------------------------
    if ms_pairs and td_pairs:
        min_len = min(len(ms_pairs), len(td_pairs))
        vx = [p[0] for p in ms_pairs[:min_len]]
        ms_vals = [p[1] for p in ms_pairs[:min_len]]
        td_vals = [p[1] for p in td_pairs[:min_len]]
        
        # 計算 Objective
        obj_vals = [0.5 * m + 0.5 * t for m, t in zip(ms_vals, td_vals)]
        
        axes[3].plot(vx, obj_vals, color='#1f77b4', linewidth=LINE_WIDTH, label='Objective: 0.5*MS + 0.5*TD')
        axes[3].set_ylabel("Weighted Score")
        axes[3].set_xlabel("Validation Epochs")
        axes[3].set_title("Overall Objective Convergence")
        axes[3].legend()
        axes[3].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[3].set_title("Cannot Calculate Objective")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    out_dir = Path(OUTPUT_PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"analysis_{CORE_NAME}.png"
    
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ 進階分析圖已輸出：{out_path}")

if __name__ == "__main__":
    main()
