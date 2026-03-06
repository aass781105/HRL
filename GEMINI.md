# FJSP-DRL (NO_GNN) 深度強化學習框架：極致細節技術手冊

## 1. 系統架構：雙層層次化強化學習 (HRL)
本框架採用兩層決策機制，解決動態生產環境中的資源分配與工作流控制問題。

### 1.1 高階門控層 (High-Level: Gate Agent)
- **模型**: DDQN (Double Deep Q-Network)
- **任務**: 流量控制。在任務到達時，判斷緩衝區 (Buffer) 內的工單是否應立即釋放入車間。
- **動作空間**: {0: HOLD (留在緩衝區), 1: RELEASE (釋放並重排)}

### 1.2 低階調度層 (Low-Level: Scheduler Agent)
- **模型**: PPO (Proximal Policy Optimization) + DANIEL (MLP-based DAN)
- **任務**: 指派與排序。決定已釋放工序的機器分配與開工時間。
- **動作空間**: 離散空間 [0, J * M - 1]，代表選擇一個 (工單, 機器) 組合。

---

## 2. 低階 PPO 特徵工程 (State Representation - 14維操作特徵)
所有的時間類特徵均在每個實例內部進行 **Instance-wise Z-Score** 正規化（第 12, 13 維除外）。

| Index | 特徵名稱 | 物理意義 | 處理邏輯 / 公式 |
| :--- | :--- | :--- | :--- |
| **0** | **Scheduled Flag** | 該工序的狀態位元 | 已排程為 1，否則為 0。 |
| **1** | **CT Lower Bound** | 完工時間下界 | 基於當前 MFT 與後續 PT 估算的該工序最早完工時間。 |
| **2** | **Min PT** | 最短加工時間 | 該工序在所有相容機器中的最小 PT (經 Instance Max-Min 縮放至 0~1)。 |
| **3** | **PT Span** | 加工時間跨度 | 該工序相容機器的 (Max PT - Min PT)。 |
| **4** | **Mean PT** | 平均加工時間 | 該工序在所有相容機器中的平均 PT。 |
| **5** | **Waiting Time** | 等待時長 | `max(0, 當前系統時間 - 工序就緒時間)`。 |
| **6** | **Remain Work** | 工序剩餘工時 | 該工序在其所在工單中，剩餘未做工序的平均總工時。 |
| **7** | **Job Left Ops** | 剩餘工序數 | 該工單中尚未排程的工序總量。 |
| **8** | **Job Remain Work** | 工單總剩餘負荷 | 該工單剩餘所有工序的平均加工時間總和。 |
| **9** | **Avail Mch Ratio** | 機器靈活性比例 | `該工序可用機器數 / 總機器數`。 |
| **10** | **Rem Due Time** | 距離交期剩餘時間 | `(絕對交期 - 當前系統時間) / mean_pt`。 |
| **11** | **Slack** | 寬裕時間 | `Rem Due Time - Job Remain Work`。反映逾期風險。 |
| **12** | **CR (Critical Ratio)** | **臨界比率** | `Log1p(max(0, RemDueTime) / max(1, JobRemainWork))`。**保留原始值**。 |
| **13** | **Is Tardy** | **遲交標記** | 若 `當前時間 > 絕對交期` 則為 1，否則為 0。**保留原始值**。 |

### 2.4 低階 PPO 網路架構 (DANIEL NO-GNN)
低階 Agent 採用改良後的 DANIEL 模型，移除圖卷積 (GNN) 以提升運算速度，轉而強化 MLP 編碼與注意力機制。

#### **2.4.1 MLP Encoder (特徵提煉)**
針對工序 (Operation) 與機器 (Machine) 分別設有獨立的編碼器：
- **結構**: `Input -> Linear(128) -> ReLU -> LayerNorm -> Linear(128) -> ReLU -> LayerNorm -> Linear(64)`。
- **潛在空間 (Latent Space)**: 最終將每個實體壓縮為 **64 維** 的特徵向量 (Embedding)。
- **物理意義**: 這 64 維向量封裝了工序的緊迫性、關鍵路徑位置以及機器未來的可用性。

#### **2.4.2 DAN (Dynamic Attention Network) 機制**
- **動態匹配**: 透過 Multi-Head Attention 計算工序與機器間的「親和力」。
- **全域池化 (Global Pooling)**: 對所有工序的 64 維 Embedding 進行 **Mean Pooling**，生成一個代表全系統計畫完整度的 **64 維全域特徵向量**。

#### **2.4.3 決策輸出 (Heads)**
- **Actor (Policy)**: 接收 [局部工序, 局部機器, 全域特徵] 拼接向量，輸出 $J \times M$ 的行動機率分佈。
- **Critic (Value)**: 基於全域 64 維特徵，輸出單一數值 $V(s)$ 用於計算優勢函數 (Advantage)。

---

## 3. 高階 DDQN 特徵工程 (State Representation - 16維系統特徵)
所有時間類特徵均除以 **mean_pt = (Low + High) / 2.0** 進行縮放。

| Index | 特徵名稱 | 物理意義與計算公式 |
| :--- | :--- | :--- |
| **o0** | **Buffer Size (Log)** | `Log1p(Buffer工單數)`。壓制大數值，增強對 1-5 個工單的靈敏度。 |
| **o1** | **Avg System Load** | `Mean(MFT_j - t_now)`。車間內所有機器平均還剩多少分鐘的工作。 |
| **o2** | **Min System Load** | `Min(MFT_j - t_now)`。最快空閒的那台機器還要多久才能接活。 |
| **o3** | **Weighted Idle** | `Integral(1 - (t - t0)/H) dt`。捕捉排程中的碎片閒置量。 |
| **o4** | **Buf NegSlack Ratio** | `Buffer 內理論 Slack < 0 的工單數 / Buffer 總工單數`。反映即將遲交的急迫廣度。 |
| **o5** | **Buf Min Slack** | `Min(Buffer_Job_Slack)`。緩衝區中最危險的那張單還剩幾分鐘。 |
| **o6** | **Buf Avg Slack** | `Mean(Buffer_Job_Slack)`。緩衝區整體訂單的寬裕度。 |
| **o7** | **WIP Min Slack** | `Min(In_Floor_Job_Slack)`。車間內正在做的單子中最緊急的 Slack。 |
| **o8** | **Load Std** | `Std(MFT_j)`。機器負荷的不平衡度。數值越高代表負載越不均。 |
| **o9** | **WIP Tardy Ratio** | `車間內已遲交工單數 (t_now > Due) / 車間內總工單數`。反映生產線已產生的實質延誤度。 |
| **o10** | **WIP Avg Slack** | `Mean(In_Floor_Job_Slack)`。車間內在製品的整體交付安全邊際。 |
| **o11** | **Congestion Log** | **(核心：擁塞監控)** `Log1p(o11 - 1.0)`。反映 PPO 造成的排隊延遲相對於理想極限的百分比。 |
| **o12** | **Buf Slack Std** | **(新加入)** 緩衝區工單 Slack 的標準差。反映待辦任務的緊急程度差異。 |
| **o13** | **WIP Slack Std** | **(新加入)** 車間內工單 Slack 的標準差。反映在製品壓力的不均勻度。 |
| **o14** | **WIP Job Count** | **(核心：擁塞度)** 目前車間內在製品的工單總數。區分「大單少件」與「小單多件」的結構差異。 |
| **o15** | **Load Span** | **(核心：不平衡預警)** `(Max_Load - Min_Load) / mean_pt`。偵測機器之間的嚴重過載與閒置差距。 |
| **o16** | **Slack Density** | **(核心：規模感應)** `WIP_Avg_Slack / (WIP_Count + 1)`。反映高工單密度下，Slack 的實際容錯價值縮減。 |
| **o17** | **Unweighted Idle** | **(核心：碎片總量)** 視窗 `[t_now, t_now + Max_Load]` 內機台的總閒置長度（無時間加權）。反映系統整體的空洞比例。 |


---

## 4. 獎勵函數設計 (Reward Shaping)

### 4.1 低階 PPO 獎勵
採用密集增量獎勵，總獎勵 = `(Makespan增量 + Tardiness懲罰) / N_jobs`。
- **Makespan增量**: `舊LB - 新LB`。
- **Tardiness懲罰**: 僅在最後一道工序觸發，`-(延遲量 * tardiness_alpha)`。

### 4.3 獎勵處理 (Reward Scaling)
為了記錄原始物理表現並建立基準線，全系統目前採用 **線性獎勵 (Linear/Raw)**：
- **公式**: `Reward = sum(r_raw)`
- **處理**: 
  - 所有分項均除以 `mean_pt` 進行歸一化。
  - 只有 **Reward_Release** 的正向收益（延遲減少）會進行 **50% 折減**，以確保決策保守性。
  - 已移除對數壓縮 (Log-scaling)，以保留原始的延遲跳變信號。

1. **Reward_Idle (區間閒置獎勵)**:
   - **計算範圍**: 採「雙決策點區間積分」。計算自上一次獎勵結算 `t_prev_reward` 到當前時刻 `t_now` 之間，全系統 $M$ 台機器的總空轉分鐘數。
   - **物理意義**: 懲罰排程中的「時間碎片」。透過 `compute_interval_metrics` 函數，精確計算機器在該特定時間視窗內的閒置 Gap（包含歷史遺留與當前排程的交集）。
   - **公式**: `-( 區間總閒置分鐘數 * idle_penalty_coef ) / mean_pt`。

2. **Reward_Buffer (臨界 Slack 延遲)**: 
   - **觸發條件**: 僅針對 Buffer 中滿足 `t_now + total_proc_time > Due_Date` (即理論 Slack < 0) 的工單。
   - **計算方式**: 計算在 `[t_prev_reward, t_now]` 區間內，這些工單產生的理論延遲 (Theoretical TD) 增量。
   - **物理意義**: 採用 **平均加工時間 (Mean PT)** 作為基準。只要工單進入「按照平均速度也註定遲交」的狀態，就開始懲罰。這提供了更早的預警空間與更平滑的獎勵梯度。
3. **Reward_Release (邊際增量法)**: `-( (新計畫全系統TD預估 - 舊計畫全系統TD預估) * release_penalty_coef ) / mean_pt`。
   - **修正**: 已取消「除以本次釋放工單數」的稀釋邏輯。
   - **保守優化**: 若重排導致 TD 減少 (即 Reward 為正)，則 **該正向獎勵減半 (50%)**。這能防止 Agent 為了「刷分」而進行頻繁且無意義的重排，強迫其追求真正的流量控制。
4. **Reward_Stab**: `-( Action * stability_scale )`。
5. **Reward_Flush (終局結算)**: `-( 最終Makespan * 0.5 )` (權重減半)。
   - **修正**: 採用 **線性原始值 (Raw)** 計算，不單獨進行對數壓縮。這確保了最終 Makespan 在總體獎勵中具有足夠的影響力，引導 Agent 追求全系統資源利用率。

---

## 5. 環境運行與數據規範

### 5.1 數據對齊原則 (Authoritative Consistency)
- **交期同源**: 全系統共用 `Master_Due_Date_Dictionary`，徹底消除隨機偏移。
- **物理判定**: WIP 狀態透過 `total_ops` 標籤與物理時鐘對齊，確保觀測真實性。
- **評價信心指標**: 透過對比 **o11 (已排定)** 與 **o12 (理論極限)**，Agent 具備了評價 PPO 排程品質的能力。

### 5.2 數據分佈抽樣
- **加工時間**: 50% 機率採 Uniform [1, 99]，50% 採 Realistic (仿 vdata 兩階段生成)。
- **交期抽樣**: 個別交期係數 $k \sim U(1.2, 2.0)$。每張工單獨立抽樣。

---

## 6. 模型訓練與管理

### 6.1 低階 PPO 訓練 (Curriculum Learning)
- **多環境並行 (Vectorized Envs)**: 採用 `num_envs=100` 進行數據收集，同時模擬 100 個獨立的 FJSP 實例，極大化樣本多樣性與 GPU 利用率。
- **課程學習階梯 (Schedules)**: 透過 `train_curriculum.py` 執行 5 階段難度提升。
  - **s1 (Baseline)**: 固定規模 ($10\times5$), 固定緊湊度 ($M=0.5$)。
  - **s2 (Size)**: 動態規模提升 ($10 \to 15 \to 20$), 固定緊湊度。
  - **s3 (Size+M)**: 動態規模提升 + 動態緊湊度提升 ($M: 0.5 \to 0.4 \to 0.3$)。
  - **s4/s5 (Noise)**: 在規模提升的同時，引入結構性噪聲變化以增強模型魯棒性。
- **優化器策略 (Sawtooth LR)**: 階段間學習率衰減，搭配階段內 Cosine Annealing (模擬熱重啟) 避免局部最優。

### 6.2 高階 DDQN 訓練 (Gate Training)
- **穩定化策略**: 每 10 個 Episode 更換隨機種子，確保 Agent 有充足機會在同一環境中優化。
- **大批量學習**: 採用 `batch_size=1024`，確保 DDQN 從 Replay Buffer 提取的經驗具備足夠的統計穩定性。

### 6.3 檔案命名與路徑規範
- **權重**: 
  - DDQN: `ddqn_ckpt/[ddqn_name].pth`
  - PPO: `trained_network/due_date/`
- **日誌**: `plots/global/DDQN_[ddqn_name]/` (含甘特圖與 CSV)。
- **分析工具**: `print_test_result_full.py` 生成包含每一 Instance 明細的 Excel。
