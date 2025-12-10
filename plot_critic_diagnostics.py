import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- 設定檔案路徑 ---
# 請將您的日誌內容複製到這個檔案中
LOG_FILE_PATH = "critic_log.txt"
# --- 設定結束 ---


def parse_log_file(log_file_path):
    """解析日誌檔案，提取Critic診斷數據。"""
    steps = []
    actual_returns = []
    predicted_values = []
    value_losses = []

    # 用正則表達式捕捉診斷日誌中的四個數值
    log_pattern = re.compile(
        r"\[CRITIC DIAGNOSTIC\] Steps:\s*(\d+)\s*\|\s*Mean Actual Return \(G_t\):\s*(-?[\d\.]+)\s*\|\s*Mean Predicted Value V\(s\):\s*(-?[\d\.]+)\s*\|\s*Mean Value Loss:\s*(-?[\d\.]+)"
    )

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                actual_returns.append(float(match.group(2)))
                predicted_values.append(float(match.group(3)))
                value_losses.append(float(match.group(4)))
    
    return steps, actual_returns, predicted_values, value_losses

def plot_diagnostics(steps, actual_returns, predicted_values, value_losses, output_filename="critic_diagnostics.png"):
    """建立並儲存診斷圖表。"""
    if not steps:
        print("錯誤：在日誌檔案中未找到任何診斷數據，無法生成圖表。")
        return

    num_updates = len(steps)
    x_axis = np.arange(num_updates)

    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('PPO Critic 診斷分析 (PPO Critic Diagnostic Analysis)', fontsize=16)

    # --- 圖 1: 預測價值 V(s) vs. 真實回報 (G_t) ---
    axs[0].plot(x_axis, actual_returns, 'o-', label='平均真實回報 (G_t) - "客觀現實"', markersize=4, alpha=0.8, color='royalblue')
    axs[0].plot(x_axis, predicted_values, 's-', label='平均預測價值 V(s) - "主觀預測"', markersize=4, alpha=0.7, color='darkorange')
    axs[0].set_ylabel('價值 (Value)')
    axs[0].set_title('Critic 的預測 vs. 客觀現實')
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- 圖 2: 價值損失 ---
    ax2_twin = axs[1].twinx()
    axs[1].plot(x_axis, value_losses, '.-', label='平均價值損失 (MSE)', color='crimson')
    axs[1].set_ylabel('均方誤差 (MSE Loss)')
    axs[1].set_title("Critic 的預測誤差 (Mean Value Loss)")
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_yscale('log')  # 使用對數尺度，以便觀察損失的細微變化

    # 在圖2上疊加子問題步數，觀察關聯性
    ax2_twin.bar(x_axis, steps, label='子問題步數 (Batch Steps)', color='gray', alpha=0.2)
    ax2_twin.set_ylabel('步數 (Steps)')
    ax2_twin.legend(loc='upper right')


    # --- 圖 3: 子問題的時間尺度 (步數) ---
    axs[2].bar(x_axis, steps, label='子問題步數 (Batch Steps)', color='gray')
    axs[2].set_xlabel('PPO 更新次數 (PPO Update Number)')
    axs[2].set_ylabel('步數 (Number of Steps)')
    axs[2].set_title('每個子問題的長度 (時間尺度)')
    axs[2].legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=150)
    print(f"診斷圖表已儲存至: {output_filename}")

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE_PATH):
        print(f"錯誤: 找不到日誌檔案 '{LOG_FILE_PATH}'")
        print("請確認您已經創建了該檔案，並將終端機的日誌內容複製了進去。")
        sys.exit(1)
        
    try:
        s, g_t, v_s, v_loss = parse_log_file(LOG_FILE_PATH)
        # 從日誌檔名產生圖片檔名
        base_name = os.path.splitext(os.path.basename(LOG_FILE_PATH))[0]
        output_image_name = f"{base_name}_critic_diagnostics.png"
        plot_diagnostics(s, g_t, v_s, v_loss, output_filename=output_image_name)
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        sys.exit(1)

