# Gemini Agent Project Guide for FJSP-DRL (NO_GNN)

## 1. Project Overview (專案概觀)
本專案 (FJSP-DRL-main_NO_GNN) 旨在利用 **Deep Reinforcement Learning (PPO)** 解決 **Flexible Job Shop Scheduling Problem (FJSP)**。
系統採用「雙層架構」：
1.  **調度層 (Scheduler Agent)**: 使用 PPO + DAN (Dynamic Attention Network) 決定工序 (Operation) 的機器分配與排序。
2.  **門控層 (Gate Agent)**: (選用) 使用 DDQN 決定何時將動態到達的工單從緩衝區釋放到調度層。

核心目標是同時優化 **Makespan (完工時間)** 與 **Tardiness (延遲)**。

## 2. Key Architecture & Files (關鍵架構與檔案)

### Environment Layer (環境層)
-   **`FJSPEnvForVariousOpNums.py` (核心)**:
    -   處理靜態與動態 FJSP 模擬。
    -   特徵工程：14 維操作特徵 (含 `is_tardy`, `slack`, `wait_time` 等) + 8 維機器特徵。
    -   **最新修改**: `step` 函數中的 Tardiness 正規化公式已更新。
-   `dynamic_fjsp_env.py`: 用於純動態場景的包裝器。
-   `event_gate_env.py`: 用於訓練 Gate Policy 的上層環境。

### Model Layer (模型層)
-   `model/PPO.py`: PPO 演算法實作，包含 Actor-Critic 網絡。
-   `model/sub_layers.py`: DAN (Dynamic Attention Network) 的注意力機制層。
-   `model/ddqn_model.py`: Gate Policy 的 DDQN 網絡。

### Training & Execution (訓練與執行)
-   **`train_curriculum.py` (主要訓練)**: 實作課程學習 (Curriculum Learning)，動態調整工單數量 ($N_j$) 與交期緊湊度 ($k$)。
-   `train_ddqn.py`: 訓練 DDQN Gate Policy。
-   `main.py`: 整合測試與推論入口。

### Utilities (工具)
-   `params.py`: 全域參數配置 (使用 `argparse`)。
-   `data_utils.py`: SD2 / BenchData 實例生成器。
-   `plot_train.py`: 訓練曲線繪製與驗證分析 (含 In-Distribution G1-G4 分析)。

## 3. Critical Logic & Formulas (關鍵邏輯與公式)

### Tardiness Reward Normalization (更新於 2026-01-28)
在 `FJSPEnvForVariousOpNums.py` 的 `step` 函數中，Tardiness 的 Reward 計算方式如下：

```python
# k_val 來自外部傳入的 tightness (例如: 1.2, 1.5)，預設為 1.5
base_scale = mk - (mean_op_pt * 5 * k_val)
base_scale = max(base_scale, 1e-8)
reward_td = - alpha * (tardiness / base_scale)
```
*   **目的**: 動態調整懲罰力度，使其隨 Makespan ($mk$) 縮短而保持相對穩定，並考量交期設定 ($k$)。
*   **實作細節**: `set_initial_data` 現已接受 `tightness` 參數並傳遞給環境。

### Due Date Tightness (k-value) Scaling Rule
當工單數量 ($N_j$) 增加時，交期係數 $k$ 採用 **0.8 次方縮放準則**，而非線性縮放：
*   **公式**: $k_{new} = k_{base} \times (N_{new} / N_{base})^{0.8}$
*   **範例 ($10 \to 20$ jobs)**: $1.2 \times (2)^{0.8} \approx 2.1$ (傳統線性縮放為 2.4)
*   **設計動機**: 線性縮放會導致大規模問題下的交期過於寬鬆（因為並行處理紅利使得 Makespan 增長慢於工單數增長）。採用 0.8 次方能維持更穩定的訓練壓力。

### Feature Engineering
-   **Operation Features (14 dim)**: 包含狀態位元 (Scheduled/Completed)、時間資訊 (LB, Wait, Remain)、交期資訊 (Due Date, Slack, CR, Is_Tardy)。
-   **Machine Features (8 dim)**: 包含負載資訊 (Available Ops/Jobs, Wait Time, Remain Work)。

## 4. Development Conventions (開發規範)

-   **State Management**: 環境狀態使用 `EnvState` dataclass 管理，並大量使用 PyTorch Tensor 進行批次運算。修改特徵時需同步更新 `EnvState.update` 和 `config.fea_j_input_dim`。
-   **Config Management**: 優先修改 `configs/*.yml`，或透過 `params.py` 的 CLI 參數覆蓋。
-   **Logging**: 訓練日誌存於 `train_log/SD2/`，格式包含 `reward_*.txt`, `detailed_reward_*.txt`, `vali_indist_*.txt`。

## 5. Common Commands (常用指令)

### 啟動課程學習訓練 (Curriculum Training)
```bash
python train_curriculum.py --schedule_type deep_dive --device cuda:0
```
*   `--schedule_type`: 可選 `standard`, `deep_dive`, `alt`。

### 繪製訓練分析圖表
```bash
python plot_train.py --eval_model_name <MODEL_NAME>
```
*   輸出目錄: `train_log_plot/`
*   包含：Reward 曲線、Loss 曲線、G1-G4 分組驗證趨勢圖。

## 6. Current Context (當前狀態)
-   **最近修改**:
    1.  修復 `plot_train.py` 無法輸出 In-Distribution 驗證圖的問題。
    2.  更新 `FJSPEnvForVariousOpNums.py` 與 `train_curriculum.py`，支援傳入 `tightness` (k) 並用於 Tardiness Reward 正規化。
-   **待辦事項**:
    -   觀察新 Reward 公式對 Tardiness 收斂的影響。
    -   (選用) 檢查 `event_gate_env.py` 是否需要同步類似的正規化邏輯。
