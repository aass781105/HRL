# plot_train.py
import json, ast, re, os
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

# ======= 在這裡設定 =======
LOG_BASE_DIR = "train_log/SD2/"
CORE_NAME    = "mix_1_div_njob_10x5+mix"  # 請確認這與您的 Log 檔名一致
OUTPUT_PLOT_DIR  = "train_log_plot"
LINE_WIDTH       = 1.5
# =========================

NUM = r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?"
PAIR_RE = re.compile(r"\[\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\]")
NUM_LIST_RE = re.compile(NUM)

def parse_content(text: str) -> Union[List[Tuple[float,float]], List[float]]:
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
    if not txt_path.exists():
        return []
    text = txt_path.read_text(encoding="utf-8")
    parsed = parse_content(text)
    if not parsed: return []
    # 如果只有 y 值，自動補上 x (index)
    if isinstance(parsed[0], tuple):
        return parsed
    ys = parsed
    return list(enumerate(ys))

def main():
    log_dir = Path(LOG_BASE_DIR).expanduser().resolve()
    
    # 統一構建所有路徑
    reward_path = log_dir / f"reward_{CORE_NAME}.txt"
    ms_path     = log_dir / f"valiquality_{CORE_NAME}.txt"
    td_path     = log_dir / f"valitardiness_{CORE_NAME}.txt"
    loss_path   = log_dir / f"loss_{CORE_NAME}.txt"

    if not reward_path.exists():
        print(f"找不到 Reward 檔案：{reward_path}")
        print(f"請確認 CORE_NAME 是否正確：{CORE_NAME}")
        return

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
