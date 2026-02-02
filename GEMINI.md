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
-   **MLP Encoder**: 當前架構使用純 MLP 作為 Encoder (14 -> 128 -> 128 -> 64)。

### Training & Execution (訓練與執行)
-   **`train_curriculum.py` (主要訓練)**: 實作課程學習 (Curriculum Learning)，動態調整工單數量 ($N_j$) 與交期緊湊度 ($k$)。
    -   **資料增強 (Data Augmentation)**: 為了提升泛化能力 (對抗 Hurink vdata 等)，採樣時現在採用混合模式：
        -   **50% Uniform**: 傳統均勻分佈 ($U(1, 20)$ PT, 隨機靈活度)。
        -   **50% Realistic**: 仿 `vdata` 分佈。
            -   **PT**: 雙層生成 ($\mu_{op}$ 隨工序變動，$\sigma_{mch}$ 隨機台波動)。
            -   **Flexibility**: 混合靈活度 (20% 專用機, 50% 部分靈活, 30% 高靈活)。
-   `train_ddqn.py`: 訓練 DDQN Gate Policy。
-   `main.py`: 整合測試與推論入口。

### Utilities (工具)
-   `params.py`: 全域參數配置 (使用 `argparse`)。
-   `data_utils.py`: SD2 / BenchData 實例生成器。已更新支援 `mode='realistic'`。
-   `plot_train.py`: 訓練曲線繪製與驗證分析 (含 In-Distribution G1-G4 分析)。

## 3. Critical Logic & Formulas (關鍵邏輯與公式)

### Tardiness Reward Normalization (更新於 2026-01-28)
在 `FJSPEnvForVariousOpNums.py` 的 `step` 函數中，Tardiness 的 Reward 計算方式已簡化：

```python
# 使用預估完工時間作為穩定分母
base_scale = self.mean_op_pt * self.number_of_jobs
base_scale = max(base_scale, 1e-8)
reward_td = - alpha * (tardiness / base_scale)
```

### Feature Engineering
-   **Operation Features (14 dim)**: 包含狀態位元、時間資訊、交期資訊。
-   **Normalization (更新於 2026-01-28)**: 
    *   前 12 維特徵執行 **Instance-wise Z-Score Normalization**。
    *   **第 13-14 維 (CR, Is_Tardy)**: **保留原始數值 (Raw Values)**，不參與標準化。

#### 詳細特徵列表與處理
| Index | 特徵名稱 | 描述 | 正規化 |
| :--- | :--- | :--- | :--- |
| 0 | Scheduled Flag | 是否已排程 (0/1) | Z-Score |
| 1 | CT Lower Bound | 預估完工時間下界 | Z-Score |
| 2 | Min Process Time | 最短加工時間 (歸一化至 [0,1]) | Z-Score |
| 3 | PT Span | 加工時間跨度 (Max - Min) | Z-Score |
| 4 | Mean Process Time | 平均加工時間 | Z-Score |
| 5 | Waiting Time | 相對於 Next Schedule Time 的等待時間 | Z-Score |
| 6 | Remain Work | 工序剩餘加工時間 | Z-Score |
| 7 | Job Left Ops | 工單剩餘工序數 | Z-Score |
| 8 | Job Remain Work | 工單剩餘總工時 | Z-Score |
| 9 | Avail Mch Nums | 可用機器數比例 (`count / n_m`) | Z-Score |
| 10 | Remain Time (Due) | 距離交期剩餘時間 (`Due - CurrentTime`) | Z-Score |
| 11 | Slack | 寬裕時間 (`RemainTime - JobRemainWork`) | Z-Score |
| 12 | CR (Log) | 臨界比率 (`Log(1 + |RemainTime / JobRemainWork|)`) | **RAW** |
| 13 | Is Tardy | 是否已遲交 (0/1) | **RAW** |

-   **Machine Features (8 dim)**: 全部 8 維執行 **Z-Score Normalization**。
    - 包含：Avail Job/Op Nums, Min/Mean PT, Waiting/Remain Work, Free Time, Working Flag。

-   **Pair Features (8 dim)**: 加工時間比率與配對等待時間，不進行額外 Z-Score。

### 5. Common Commands (常用指令)

### 啟動課程學習訓練 (Curriculum Training)
```bash
python train_curriculum.py --schedule_type deep_dive --device cuda:0
```
*   `--schedule_type`: 可選 `standard`, `deep_dive`, `alt`。

### 繪製訓練分析圖表
```bash
python plot_train.py --eval_model_name <MODEL_NAME>
```

## 6. Current Context (當前狀態)
-   **最近修改 (2026-02-01)**:
    1.  **資料增強實作**: `data_utils.py` 新增 `realistic` 模式，模擬 `vdata` 的加工時間變異與靈活度分佈。
    2.  **訓練混合**: `train_curriculum.py` 現在以 50/50 比例混合 Uniform/Realistic 實例進行訓練，以解決泛化能力不足的問題。
-   **待辦事項**:
    -   觀察新的資料分佈對訓練收斂速度的影響 (可能變慢，但 Test Score 應提升)。
    -   (下一步) 考慮擴充 Machine Features，加入預估完工時間 (Estimated Finish Time) 以強化對 Tardiness 的感知。