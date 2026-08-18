# plot_train.py
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, ast, re
from pathlib import Path
from typing import List, Tuple, Union

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
from params import configs
from common_utils import lower_level_log_dir, lower_level_plot_dir, strToSuffix

# ======= 設定 =======
OUTPUT_PLOT_DIR  = lower_level_plot_dir()
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
    """
    Loads XY data from a text file, handling numpy string pollution.
    """
    if not txt_path.exists():
        return []
    text = txt_path.read_text(encoding="utf-8")
    
    # [FIXED] Global cleaning for all txt log loads
    clean_text = re.sub(r"np\.float\d+\(([^)]+)\)", r"\1", text)
    
    parsed = parse_content(clean_text)
    if not parsed: return []
    if isinstance(parsed[0], tuple):
        return parsed
    ys = parsed
    return list(enumerate(ys))

def plot_reward_components(detailed_path: Path, output_path: Path, core_name: str):
    """
    Reads detailed_reward_*.txt which contains: [ep, r, mk_mean, mk_std, td_mean, td_std, raw_mk, raw_td]
    [FIXED] Uses cleaned text and Dual Y-axes.
    """
    if not detailed_path.exists():
        print(f"Detailed reward log not found: {detailed_path}")
        return

    text = detailed_path.read_text(encoding="utf-8")
    clean_text = re.sub(r"np\.float\d+\(([^)]+)\)", r"\1", text)
    
    try:
        data = ast.literal_eval(clean_text)
    except Exception as e:
        print(f"Error parsing detailed log: {e}")
        return
    
    if not data or len(data) == 0 or len(data[0]) < 8:
        print("Detailed reward data format incorrect (need 8 columns).")
        return

    ep = [x[0] for x in data]
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
    
    lns0 = ln1 + ln2
    labs0 = [l.get_label() for l in lns0]
    axes[0].legend(lns0, labs0, loc='upper center')

    # 2. Normalized Components (Dual Axis - FIXED SCALE)
    axes[1].set_title('Reward Components (Learning Signal Breakdown)')
    ln3 = axes[1].plot(ep, mk_mean, color='green', label='Mk Gain')
    axes[1].set_ylabel('Mk Gain Reward', color='green')
    axes[1].tick_params(axis='y', labelcolor='green')
    axes[1].grid(True, alpha=0.3)
    
    ax1_r = axes[1].twinx()
    ln4 = ax1_r.plot(ep, td_mean, color='red', label='Td Penalty')
    ax1_r.set_ylabel('Td Penalty Reward', color='red')
    ax1_r.tick_params(axis='y', labelcolor='red')
    
    lns1 = ln3 + ln4
    labs1 = [l.get_label() for l in lns1]
    axes[1].legend(lns1, labs1, loc='upper center')

    # 3. Components Std (Dual Axis)
    axes[2].set_title('Reward Stability (Std Dev)')
    ln5 = axes[2].plot(ep, mk_std, color='green', linestyle='--', label='Mk Std')
    axes[2].set_ylabel('Mk Std', color='green')
    axes[2].tick_params(axis='y', labelcolor='green')
    axes[2].grid(True, alpha=0.3)
    
    ax2_r = axes[2].twinx()
    ln6 = ax2_r.plot(ep, td_std, color='red', linestyle='--', label='Td Std')
    ax2_r.set_ylabel('Td Std', color='red')
    ax2_r.tick_params(axis='y', labelcolor='red')
    
    lns2 = ln5 + ln6
    labs2 = [l.get_label() for l in lns2]
    axes[2].legend(lns2, labs2, loc='upper center')
    axes[2].set_xlabel('Episodes/Updates')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Reward 組成分析圖已輸出：{output_path}")

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

def plot_comparison_analysis(log_dir: Path, output_path: Path, log_file_suffix: str, core_name: str):
    """
    Plots comparison of 3 validation sets: Mixed (Main), Uniform, and Test (Benchmark).
    3 Subplots: Makespan, Tardiness, Objective.
    """
    # 1. Define file paths
    # Mixed (Main)
    path_mix_mk = log_dir / f"valiquality_{log_file_suffix}.txt"
    path_mix_td = log_dir / f"valitardiness_{log_file_suffix}.txt"
    
    # Uniform
    path_uni_mk = log_dir / f"valiquality_uniform_{log_file_suffix}.txt"
    path_uni_td = log_dir / f"valitardiness_uniform_{log_file_suffix}.txt"
    
    # Test
    path_test_mk = log_dir / f"valiquality_test_{log_file_suffix}.txt"
    path_test_td = log_dir / f"valitardiness_test_{log_file_suffix}.txt"

    # 2. Load Data
    def load_metric(p):
        data = load_xy(p)
        if not data: return [], []
        return zip(*data) # Returns (x_list, y_list)

    # Mixed
    mx_x, mx_mk = load_metric(path_mix_mk)
    _, mx_td = load_metric(path_mix_td)
    
    # Uniform
    ux_x, ux_mk = load_metric(path_uni_mk)
    _, ux_td = load_metric(path_uni_td)
    
    # Test
    tx_x, tx_mk = load_metric(path_test_mk)
    _, tx_td = load_metric(path_test_td)

    # Check if we have data
    if not mx_x:
        print("No validation data found for comparison plot.")
        return

    # 3. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f"Validation Comparison: {core_name}", fontsize=16)
    
    # Helper to safe get objective
    def get_obj(mk, td):
        if len(mk) != len(td): return []
        return [0.5 * m + 0.5 * t for m, t in zip(mk, td)]

    mx_obj = get_obj(mx_mk, mx_td)
    ux_obj = get_obj(ux_mk, ux_td)
    tx_obj = get_obj(tx_mk, tx_td)

    # Subplot 1: Makespan
    ax1 = axes[0]
    ax1.set_title("Makespan Comparison")
    ax1.plot(mx_x, mx_mk, label='Mixed (Main)', color='blue', linewidth=2)
    if ux_mk: ax1.plot(ux_x, ux_mk, label='Uniform', color='green', linestyle='--')
    
    # For Test, use dual axis if range is very different, else same
    # Simple heuristic: if test mean is > 2x mixed mean, use dual
    use_dual_test = False
    if tx_mk and mx_mk and (np.mean(tx_mk) > 2 * np.mean(mx_mk) or np.mean(tx_mk) < 0.5 * np.mean(mx_mk)):
        use_dual_test = True
        
    if tx_mk:
        if use_dual_test:
            ax1_r = ax1.twinx()
            ax1_r.plot(tx_x, tx_mk, label='Test (Benchmark) [Right Axis]', color='red', linestyle=':')
            ax1_r.set_ylabel("Test Makespan", color='red')
            ax1_r.legend(loc='upper right')
        else:
            ax1.plot(tx_x, tx_mk, label='Test (Benchmark)', color='red', linestyle=':')
    
    ax1.set_ylabel("Makespan")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Tardiness
    ax2 = axes[1]
    ax2.set_title("Tardiness Comparison")
    ax2.plot(mx_x, mx_td, label='Mixed (Main)', color='blue', linewidth=2)
    if ux_td: ax2.plot(ux_x, ux_td, label='Uniform', color='green', linestyle='--')
    
    # Check dual for TD
    use_dual_test_td = False
    if tx_td and mx_td and (np.mean(tx_td) > 2 * np.mean(mx_td) + 10 or np.mean(tx_td) < 0.5 * np.mean(mx_td)):
        use_dual_test_td = True

    if tx_td:
        if use_dual_test_td:
            ax2_r = ax2.twinx()
            ax2_r.plot(tx_x, tx_td, label='Test (Benchmark) [Right Axis]', color='red', linestyle=':')
            ax2_r.set_ylabel("Test Tardiness", color='red')
            ax2_r.legend(loc='upper right')
        else:
            ax2.plot(tx_x, tx_td, label='Test (Benchmark)', color='red', linestyle=':')

    ax2.set_ylabel("Tardiness")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Objective
    ax3 = axes[2]
    ax3.set_title("Objective (0.5*MS + 0.5*TD)")
    if mx_obj: ax3.plot(mx_x, mx_obj, label='Mixed (Main)', color='blue', linewidth=2)
    if ux_obj: ax3.plot(ux_x, ux_obj, label='Uniform', color='green', linestyle='--')
    
    # Check dual for Obj
    use_dual_test_obj = False
    if tx_obj and mx_obj and (np.mean(tx_obj) > 2 * np.mean(mx_obj) or np.mean(tx_obj) < 0.5 * np.mean(mx_obj)):
        use_dual_test_obj = True

    if tx_obj:
        if use_dual_test_obj:
            ax3_r = ax3.twinx()
            ax3_r.plot(tx_x, tx_obj, label='Test (Benchmark) [Right Axis]', color='red', linestyle=':')
            ax3_r.set_ylabel("Test Objective", color='red')
            ax3_r.legend(loc='upper right')
        else:
            ax3.plot(tx_x, tx_obj, label='Test (Benchmark)', color='red', linestyle=':')
            
    ax3.set_ylabel("Weighted Score")
    ax3.set_xlabel("Update Step")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ 比較分析圖已輸出：{output_path}")

def plot_validation_breakdown(csv_path: Path, output_path: Path, core_name: str):
    """
    Reads valibreakdown_*.csv and plots 3 subplots for 10J, 20J, 30J.
    """
    if not csv_path.exists():
        print(f"Breakdown CSV not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading breakdown CSV: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Validation Breakdown by Scale: {core_name}", fontsize=16)
    
    sizes = [10, 20, 30]
    for i, n_j in enumerate(sizes):
        ax = axes[i]
        ms_col = f'ms_{n_j}j'
        td_col = f'td_{n_j}j'
        
        if ms_col not in df.columns or td_col not in df.columns:
            ax.set_title(f"{n_j} Jobs (Data Missing)")
            continue

        # Plot TD
        ax.plot(df['update'], df[td_col], color='red', label='Tardiness')
        ax.set_title(f'Scale: {n_j} Jobs')
        ax.set_xlabel('Updates')
        ax.set_ylabel('Mean Total Tardiness', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)
        
        # Plot MS
        ax2 = ax.twinx()
        ax2.plot(df['update'], df[ms_col], color='blue', linestyle='--', label='Makespan')
        ax2.set_ylabel('Mean Makespan', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ 驗證細分趨勢圖已輸出：{output_path}")


def plot_training_diagnostics(loss_data, output_path: Path, core_name: str):
    """
    Reads loss_*.txt rows:
    [update, loss, v_loss, p_loss, vshare, pshare, err_mean, err_std,
     delta, over_delta_ratio, entropy, clip_frac, adv_std]
    """
    if not loss_data or len(loss_data[0]) < 10:
        print("Diagnostic loss data not found.")
        return

    steps = [x[0] for x in loss_data]
    vshare = [100.0 * x[4] for x in loss_data] if len(loss_data[0]) >= 5 else []
    pshare = [100.0 * x[5] for x in loss_data] if len(loss_data[0]) >= 6 else []
    err_mean = [x[6] for x in loss_data] if len(loss_data[0]) >= 7 else []
    err_std = [x[7] for x in loss_data] if len(loss_data[0]) >= 8 else []
    delta_vals = [x[8] for x in loss_data] if len(loss_data[0]) >= 9 else []
    over_delta_pct = [100.0 * x[9] for x in loss_data]
    entropy = [x[10] for x in loss_data] if len(loss_data[0]) >= 11 else []
    clip_pct = [100.0 * x[11] for x in loss_data] if len(loss_data[0]) >= 12 else []
    adv_std = [x[12] for x in loss_data] if len(loss_data[0]) >= 13 else []

    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f"Training Diagnostics: {core_name}", fontsize=16)

    delta_label = f">{delta_vals[0]:g}" if delta_vals else ">d"
    axes[0].plot(steps, over_delta_pct, color="#8c564b", linewidth=LINE_WIDTH, label=f"|critic error| {delta_label} (%)")
    axes[0].set_title("Critic Large-Error Ratio")
    axes[0].set_ylabel("Percent")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    if vshare and pshare:
        axes[1].plot(steps, vshare, color="orange", linewidth=LINE_WIDTH, label="Vshare")
        axes[1].plot(steps, pshare, color="purple", linewidth=LINE_WIDTH, linestyle="--", label="Pshare")
        axes[1].set_title("Loss Contribution Share")
        axes[1].set_ylabel("Percent")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1].set_title("Vshare/Pshare Data Not Found")

    if err_mean and err_std:
        axes[2].plot(steps, err_mean, color="#1f77b4", linewidth=LINE_WIDTH, label="Critic Error Mean")
        axes[2].plot(steps, err_std, color="#d62728", linewidth=LINE_WIDTH, linestyle="--", label="Critic Error Std")
        axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[2].set_title("Critic Error Distribution")
        axes[2].set_ylabel("Error")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[2].set_title("Critic Error Mean/Std Data Not Found")

    if entropy or clip_pct or adv_std:
        if entropy:
            axes[3].plot(steps, entropy, color="#2ca02c", linewidth=LINE_WIDTH, label="Entropy")
        if adv_std:
            axes[3].plot(steps, adv_std, color="#17becf", linewidth=LINE_WIDTH, linestyle="-.", label="AdvStd")
        axes[3].set_title("Policy Update Health")
        axes[3].set_ylabel("Entropy / AdvStd")
        axes[3].grid(True, linestyle="--", alpha=0.5)
        if clip_pct:
            ax3_r = axes[3].twinx()
            ax3_r.plot(steps, clip_pct, color="#ff7f0e", linewidth=LINE_WIDTH, linestyle="--", label="ClipFrac (%)")
            ax3_r.set_ylabel("ClipFrac (%)", color="#ff7f0e")
            ax3_r.tick_params(axis="y", labelcolor="#ff7f0e")
            lines_l, labels_l = axes[3].get_legend_handles_labels()
            lines_r, labels_r = ax3_r.get_legend_handles_labels()
            axes[3].legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")
        else:
            axes[3].legend(loc="upper right")
    else:
        axes[3].set_title("Policy Diagnostics Data Not Found")

    axes[3].set_xlabel("Updates")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Diagnostics plot saved: {output_path}")

def legacy_main():
    # 1. Construct dynamic log name
    model_name = configs.eval_model_name
    # Use configured training size for log suffix (e.g., 30x5).
    init_nj = int(getattr(configs, "n_j", 10))
    n_m = configs.n_m
    suffix = strToSuffix(configs.data_suffix)
    
    # Full name used in log files
    log_file_suffix = f"{model_name}_{init_nj}x{n_m}{suffix}"
    
    # 2. Setup Paths
    log_dir = Path(lower_level_log_dir()).expanduser().resolve()
    
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

    # Comparison plot was removed because it duplicated validation content in analysis.
    comp_analysis_path = out_dir / f"comparison_{CORE_NAME}.png"
    if comp_analysis_path.exists():
        comp_analysis_path.unlink()

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

    if loss_data:
        diag_path = out_dir / f"diagnostics_{CORE_NAME}.png"
        plot_training_diagnostics(loss_data, diag_path, CORE_NAME)


def parse_stability_reward_log(reward_file: Path):
    """
    Parses reward_*.txt for stability fine-tuning runs.
    Extracts step-level loss, reward, shares, and train metrics.
    """
    if not reward_file.exists():
        return None
    
    updates = []
    rewards = []
    losses = []
    v_losses = []
    mk_shares = []
    td_shares = []
    od_shares = []
    flip_shares = []
    mchg_shares = []
    train_mks = []
    train_tds = []
    train_flips = []
    train_flip_rates = []
    train_mchs = []
    train_mch_rates = []

    pattern = re.compile(
        r"Update\s+(\d+)/\d+\s+\|\s+R:\s*([-\d.]+)\s+\|\s+Loss:\s*([-\d.]+)\s+\|\s+V-Loss:\s*([-\d.]+)\s+\|\s+"
        r"MK_r:\s*([-\d.]+)%\s+\|\s+TD_r:\s*([-\d.]+)%\s+\|\s+OD_r:\s*([-\d.]+)%\s+\|\s+"
        r"FlipR:\s*([-\d.]+)%\s+\|\s+MChgR:\s*([-\d.]+)%\s+\|\s+"
        r"Train MK:\s*([-\d.]+)\s+\|\s+Train TD:\s*([-\d.]+)\s+\|\s+"
        r"Flip:\s*([-\d.]+)\s*\(\s*([-\d.]+)%\)\s+\|\s+"
        r"MChg:\s*([-\d.]+)\s*\(\s*([-\d.]+)%\)"
    )

    with open(reward_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                u, r, l, vl, mkr, tdr, odr, fr, mr, t_mk, t_td, t_flip, t_frate, t_mch, t_mrate = m.groups()
                updates.append(int(u))
                rewards.append(float(r))
                losses.append(float(l))
                v_losses.append(float(vl))
                mk_shares.append(float(mkr))
                td_shares.append(float(tdr))
                od_shares.append(float(odr))
                flip_shares.append(float(fr))
                mchg_shares.append(float(mr))
                train_mks.append(float(t_mk))
                train_tds.append(float(t_td))
                train_flips.append(float(t_flip))
                train_flip_rates.append(float(t_frate))
                train_mchs.append(float(t_mch))
                train_mch_rates.append(float(t_mrate))

    if not updates:
        return None

    return {
        "updates": np.array(updates),
        "rewards": np.array(rewards),
        "losses": np.array(losses),
        "v_losses": np.array(v_losses),
        "mk_shares": np.array(mk_shares),
        "td_shares": np.array(td_shares),
        "od_shares": np.array(od_shares),
        "flip_shares": np.array(flip_shares),
        "mchg_shares": np.array(mchg_shares),
        "train_mks": np.array(train_mks),
        "train_tds": np.array(train_tds),
        "train_flips": np.array(train_flips),
        "train_flip_rates": np.array(train_flip_rates),
        "train_mchs": np.array(train_mchs),
        "train_mch_rates": np.array(train_mch_rates),
    }


def parse_stability_detailed_log(detailed_file: Path):
    """
    Parses detailed_reward_*.txt for stability fine-tuning runs.
    Extracts validation group data and computes per-update means.
    """
    if not detailed_file.exists():
        return None

    val_by_update = {}
    with open(detailed_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("validation_group="):
                parts = dict(p.split("=", 1) for p in line.split(",") if "=" in p)
                try:
                    u = int(parts.get("update", 0))
                    if u not in val_by_update:
                        val_by_update[u] = []
                    val_by_update[u].append({
                        "group": parts.get("validation_group"),
                        "mk": float(parts.get("mk", 0.0)),
                        "td": float(parts.get("td", 0.0)),
                        "obj": float(parts.get("obj", 0.0)),
                        "flip": float(parts.get("flip", 0.0)),
                        "flip_rate": float(parts.get("flip_rate", 0.0)) * 100.0,
                        "mch": float(parts.get("machine_change", 0.0)),
                        "mch_rate": float(parts.get("machine_change_rate", 0.0)) * 100.0,
                    })
                except Exception:
                    continue

    if not val_by_update:
        return None

    sorted_updates = sorted(val_by_update.keys())
    val_data = {
        "updates": np.array(sorted_updates),
        "val_mks": np.array([np.mean([x["mk"] for x in val_by_update[u]]) for u in sorted_updates]),
        "val_tds": np.array([np.mean([x["td"] for x in val_by_update[u]]) for u in sorted_updates]),
        "val_objs": np.array([np.mean([x["obj"] for x in val_by_update[u]]) for u in sorted_updates]),
        "val_flips": np.array([np.mean([x["flip"] for x in val_by_update[u]]) for u in sorted_updates]),
        "val_flip_rates": np.array([np.mean([x["flip_rate"] for x in val_by_update[u]]) for u in sorted_updates]),
        "val_mchs": np.array([np.mean([x["mch"] for x in val_by_update[u]]) for u in sorted_updates]),
        "val_mch_rates": np.array([np.mean([x["mch_rate"] for x in val_by_update[u]]) for u in sorted_updates]),
    }
    return val_data


def plot_stability_finetune_results(model_name: str = None, log_dir: Path = None, output_dir: Path = None):
    """
    Plots the full 8-subplot training & validation analysis for a stability fine-tuning run.
    """
    if log_dir is None:
        log_dir = Path(lower_level_log_dir())
    if output_dir is None:
        output_dir = Path(lower_level_plot_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = getattr(configs, "eval_model_name", "ll_stability_finetune_ptscale")

    reward_file = log_dir / f"reward_{model_name}.txt"
    detailed_file = log_dir / f"detailed_reward_{model_name}.txt"

    train_data = parse_stability_reward_log(reward_file)
    val_data = parse_stability_detailed_log(detailed_file)

    if train_data is None and val_data is None:
        print(f"[PLOT] ⚠️ 找不到 {model_name} 的訓練日誌或格式不符，跳過繪圖。")
        return None

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f"Lower-Level Stability Fine-Tuning Analysis: {model_name}", fontsize=18, fontweight="bold", y=0.99)

    lw = LINE_WIDTH

    # 1. Loss & Value Loss
    ax = axes[0, 0]
    ax.set_title("1. Training Loss (PPO Policy Loss & Value Loss)", fontsize=13, fontweight="bold")
    if train_data:
        u = train_data["updates"]
        l1 = ax.plot(u, train_data["losses"], color="#1f77b4", linewidth=lw, label="Policy Loss")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Policy Loss", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_r = ax.twinx()
        l2 = ax_r.plot(u, train_data["v_losses"], color="#d62728", linewidth=lw, linestyle="--", label="Value Loss (V-Loss)")
        ax_r.set_ylabel("Value Loss", color="#d62728")
        ax_r.tick_params(axis="y", labelcolor="#d62728")

        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Train Data", ha="center", va="center")

    # 2. Total Reward
    ax = axes[0, 1]
    ax.set_title("2. Total Reward Convergence", fontsize=13, fontweight="bold")
    if train_data:
        ax.plot(train_data["updates"], train_data["rewards"], color="#2ca02c", linewidth=lw, label="Total Reward")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Total Reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "No Train Data", ha="center", va="center")

    # 3. Reward Components Breakdown (%)
    ax = axes[1, 0]
    ax.set_title("3. Individual Reward Shares (%)", fontsize=13, fontweight="bold")
    if train_data:
        u = train_data["updates"]
        ax.plot(u, train_data["mk_shares"], label="MK Gain Share %", color="#1f77b4", linewidth=lw)
        ax.plot(u, train_data["td_shares"], label="TD Penalty Share %", color="#d62728", linewidth=lw)
        ax.plot(u, train_data["od_shares"], label="OD Progress Share %", color="#ff7f0e", linewidth=lw)
        ax.plot(u, train_data["flip_shares"], label="Flip Penalty Share %", color="#9467bd", linewidth=lw)
        ax.plot(u, train_data["mchg_shares"], label="MChg Penalty Share %", color="#8c564b", linewidth=lw)
        ax.set_xlabel("Updates")
        ax.set_ylabel("Reward Component Share (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No Train Data", ha="center", va="center")

    # 4. Train MK vs Train TD
    ax = axes[1, 1]
    ax.set_title("4. Train Makespan vs. Tardiness", fontsize=13, fontweight="bold")
    if train_data:
        u = train_data["updates"]
        l1 = ax.plot(u, train_data["train_mks"], color="#1f77b4", linewidth=lw, label="Train Makespan")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Makespan", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_r = ax.twinx()
        l2 = ax_r.plot(u, train_data["train_tds"], color="#ff7f0e", linewidth=lw, linestyle="--", label="Train Tardiness")
        ax_r.set_ylabel("Tardiness", color="#ff7f0e")
        ax_r.tick_params(axis="y", labelcolor="#ff7f0e")

        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Train Data", ha="center", va="center")

    # 5. Train Flip & MChg
    ax = axes[2, 0]
    ax.set_title("5. Train Stability: Flips & Machine Changes", fontsize=13, fontweight="bold")
    if train_data:
        u = train_data["updates"]
        l1 = ax.plot(u, train_data["train_flips"], color="#9467bd", linewidth=lw, label="Train Flip Count")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Flip Count", color="#9467bd")
        ax.tick_params(axis="y", labelcolor="#9467bd")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_r = ax.twinx()
        l2 = ax_r.plot(u, train_data["train_mchs"], color="#8c564b", linewidth=lw, linestyle="--", label="Train MChg Count")
        ax_r.set_ylabel("Machine Change Count", color="#8c564b")
        ax_r.tick_params(axis="y", labelcolor="#8c564b")

        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Train Data", ha="center", va="center")

    # 6. Validation MK & TD & Objective
    ax = axes[2, 1]
    ax.set_title("6. Validation: MK, TD & Objective (0.5MK+0.5TD)", fontsize=13, fontweight="bold")
    if val_data:
        u = val_data["updates"]
        l1 = ax.plot(u, val_data["val_mks"], color="#2ca02c", marker="o", markersize=3, linewidth=lw, label="Val Makespan")
        l2 = ax.plot(u, val_data["val_objs"], color="#1f77b4", marker="s", markersize=3, linewidth=lw, linestyle=":", label="Val Obj (0.5MK+0.5TD)")
        ax.set_xlabel("Validation Updates")
        ax.set_ylabel("MK / Objective", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_r = ax.twinx()
        l3 = ax_r.plot(u, val_data["val_tds"], color="#d62728", marker="^", markersize=3, linewidth=lw, linestyle="--", label="Val Tardiness")
        ax_r.set_ylabel("Tardiness", color="#d62728")
        ax_r.tick_params(axis="y", labelcolor="#d62728")

        lns = l1 + l2 + l3
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Validation Data", ha="center", va="center")

    # 7. Validation Flip Count & Rate (%)
    ax = axes[3, 0]
    ax.set_title("7. Validation: Sequence Flip Count & Rate (%)", fontsize=13, fontweight="bold")
    if val_data:
        u = val_data["updates"]
        l1 = ax.plot(u, val_data["val_flips"], color="#9467bd", marker="o", markersize=3, linewidth=lw, label="Val Flip Count")
        ax.set_xlabel("Validation Updates")
        ax.set_ylabel("Flip Count", color="#9467bd")
        ax.tick_params(axis="y", labelcolor="#9467bd")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_r = ax.twinx()
        l2 = ax_r.plot(u, val_data["val_flip_rates"], color="#9467bd", marker="x", markersize=3, linewidth=lw, linestyle=":", label="Val Flip Rate (%)")
        ax_r.set_ylabel("Flip Rate (%)", color="#9467bd")
        ax_r.tick_params(axis="y", labelcolor="#9467bd")

        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Validation Data", ha="center", va="center")

    # 8. Validation Machine Change Count & Rate (%)
    ax = axes[3, 1]
    ax.set_title("8. Validation: Machine Change Count & Rate (%)", fontsize=13, fontweight="bold")
    if val_data:
        u = val_data["updates"]
        l1 = ax.plot(u, val_data["val_mchs"], color="#8c564b", marker="o", markersize=3, linewidth=lw, label="Val MChg Count")
        ax.set_xlabel("Validation Updates")
        ax.set_ylabel("Machine Change Count", color="#8c564b")
        ax.tick_params(axis="y", labelcolor="#8c564b")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_r = ax.twinx()
        l2 = ax_r.plot(u, val_data["val_mch_rates"], color="#8c564b", marker="x", markersize=3, linewidth=lw, linestyle=":", label="Val MChg Rate (%)")
        ax_r.set_ylabel("MChg Rate (%)", color="#8c564b")
        ax_r.tick_params(axis="y", labelcolor="#8c564b")

        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Validation Data", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = output_dir / f"{model_name}_training_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] 穩定性訓練視覺化圖表已輸出：{out_path}")
    return out_path


def plot_stability_models_comparison(models: list = None, log_dir: Path = None, output_dir: Path = None):
    """
    Plots a multi-model validation comparison chart (MK, TD, Objective, Flip, MChg).
    """
    if log_dir is None:
        log_dir = Path(lower_level_log_dir())
    if output_dir is None:
        output_dir = Path(lower_level_plot_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = [
            ("flip1_mch3 (ptscale)", "ll_stability_finetune_ptscale", "#1f77b4"),
            ("flip2_mch3", "ll_stability_finetune_flip2_mch3_ptscale", "#ff7f0e"),
            ("flip2_mch6", "ll_stability_finetune_flip2_mch6_ptscale", "#2ca02c"),
            ("flip4_mch6", "ll_stability_finetune_flip4_mch6_ptscale", "#d62728"),
        ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle("Lower-Level Stability Fine-Tuning: 4 Models Comparison", fontsize=18, fontweight="bold", y=0.99)

    valid_count = 0
    for label, mname, color in models:
        detailed_file = log_dir / f"detailed_reward_{mname}.txt"
        vdata = parse_stability_detailed_log(detailed_file)
        if not vdata:
            continue
        valid_count += 1
        u = vdata["updates"]
        lw = 1.8

        # 1. Val Makespan
        axes[0, 0].plot(u, vdata["val_mks"], label=label, color=color, linewidth=lw)
        # 2. Val Tardiness
        axes[0, 1].plot(u, vdata["val_tds"], label=label, color=color, linewidth=lw)
        # 3. Val Objective
        axes[1, 0].plot(u, vdata["val_objs"], label=label, color=color, linewidth=lw)
        # 4. Val Flip Count
        axes[1, 1].plot(u, vdata["val_flips"], label=label, color=color, linewidth=lw)
        # 5. Val Machine Change Count
        axes[2, 0].plot(u, vdata["val_mchs"], label=label, color=color, linewidth=lw)
        # 6. Val Flip Rate vs MChg Rate (Scatter / End Point)
        axes[2, 1].plot(u, vdata["val_mch_rates"], label=label, color=color, linewidth=lw)

    if valid_count == 0:
        plt.close(fig)
        return None

    axes[0, 0].set_title("Validation Makespan (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Updates")
    axes[0, 0].set_ylabel("Makespan")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].set_title("Validation Tardiness (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Updates")
    axes[0, 1].set_ylabel("Tardiness")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].set_title("Validation Objective (0.5*MK + 0.5*TD)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Updates")
    axes[1, 0].set_ylabel("Objective")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    axes[1, 1].set_title("Validation Sequence Flip Count (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Updates")
    axes[1, 1].set_ylabel("Flip Count")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    axes[2, 0].set_title("Validation Machine Change Count (Lower is Better)", fontsize=12, fontweight="bold")
    axes[2, 0].set_xlabel("Updates")
    axes[2, 0].set_ylabel("Machine Change Count")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 0].legend()

    axes[2, 1].set_title("Validation Machine Change Rate (%)", fontsize=12, fontweight="bold")
    axes[2, 1].set_xlabel("Updates")
    axes[2, 1].set_ylabel("MChg Rate (%)")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = output_dir / "stability_models_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] 4模型橫向對比圖已輸出：{out_path}")
    return out_path


def main():
    log_dir = Path(lower_level_log_dir())
    eval_model_name = getattr(configs, "eval_model_name", "")
    
    stability_models = [
        "ll_stability_finetune_ptscale",
        "ll_stability_finetune_flip2_mch3_ptscale",
        "ll_stability_finetune_flip2_mch6_ptscale",
        "ll_stability_finetune_flip4_mch6_ptscale",
    ]

    # If specific stability model configured via --config
    if eval_model_name and any(m in eval_model_name for m in stability_models):
        plot_stability_finetune_results(eval_model_name)
        plot_stability_models_comparison()
        return

    # Check if any stability logs exist
    has_stability = any((log_dir / f"detailed_reward_{m}.txt").exists() for m in stability_models)
    if has_stability:
        for m in stability_models:
            plot_stability_finetune_results(m)
        plot_stability_models_comparison()
        return

    # Default legacy plot
    legacy_main()


if __name__ == "__main__":
    main()
