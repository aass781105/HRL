# FJSP 雙層門控策略：動態分叉仿真分析計畫 (Branching Simulation Plan)

本紀錄檔詳細定義了用於評估高階門控（Gate Agent）「釋放時機點」優劣的動態分叉仿真控制實驗（Branching Simulation Experiment）計畫。

---

## 1. 核心實驗設計：三路分叉仿真 (Three-Way Forking)
為了在相同起點下客觀對比「早釋放」與「耐性等待」，以及「早釋放後的不同救補策略」對系統造成的物理差異，我們在分歧點 $A$ 將模擬器進行複製，並分裂出**三條平行推進的路徑**，一直運行到事件 $B$（晚釋放點）結束。

```mermaid
graph TD
    Start[模擬開始 (Seed S)] --> Base[使用 cadence 5 運行]
    Base --> ForkPoint[到達 Event 40 決策點]
    ForkPoint --> Switch[切換為各自策略決策]
    Switch --> Div{偵測到第一個分歧點 Event A}
    
    Div -->|路徑一: PPO分支| PathPPO[Event A: HOLD]
    Div -->|路徑二: Slack0分支| PathSlack[Event A: RELEASE]
    
    PathPPO --> PPO_Wait[A+1 到 B-1 期間: 持續 HOLD]
    PPO_Wait --> PPO_Release[Event B: RELEASE]
    
    PathSlack -->|子分支 Early_Release_Then_Hold| S0_Hold[A+1 到 B-1 期間: 持續 HOLD]
    PathSlack -->|子分支 Early_Release_Always| S1_Cadence[A+1 到 B-1 期間: 每步均 RELEASE]
    
    S0_Hold --> S0_Release[Event B: RELEASE]
    S1_Cadence --> S1_Release[Event B: RELEASE]
    
    PPO_Release --> Alignment[到達 B 時點後立即終止模擬]
    S0_Release --> Alignment
    S1_Release --> Alignment
    
    Alignment --> Compare[在 B 時點對齊比較三路物理狀態]
```

### 實驗步驟：
1. **共同前段 (Common Base Path)**：
   * 模擬從指定隨機種子（Seed）開始，使用 `cadence 5` 策略推進到 Event 40 前一瞬間，以建立相同的初始負載。
2. **複製分叉 (Forking at A)**：
   * 從 Event 40 開始切換為各自策略決策，並定位出第一個決策不同的**分歧事件 A**（一個釋放，另一個選擇 HOLD）。
   * 在 $A$ 時點，系統分裂為以下三條路徑，均推進到事件 $B$（晚釋放點）：
     *   **分支一（Late路徑 - 耐性等待）**：
         在 $A$ 選擇 HOLD，且在 $[A+1, B-1]$ 期間持續 HOLD，直到 $B$ 時點執行 RELEASE。
     *   **分支二（Early_Release_Then_Hold - 早釋放 + 一路 HOLD）**：
         在 $A$ 執行 RELEASE，但在 $[A+1, B-1]$ 期間持續 HOLD，直到 $B$ 時點執行 RELEASE。
     *   **分支三（Early_Release_Always - 早釋放 + 積極補救）**：
         在 $A$ 執行 RELEASE，且在 $[A+1, B]$ 期間每一步都執行 RELEASE（使用 `cadence1` 模擬最積極的插單與重排補救）。
3. **交會點 $B$ 對齊比對與終止 (Stop at B)**：
   * 當三條分支均推進到事件 $B$ 時，**立即暫停並終止模擬**。
   * 對齊比較這三條路徑在時點 $B$ 的 WIP 數量、實際延誤與預估 Makespan。

---

## 2. 直向輸出 CSV 欄位與資料列定義：`release_timing_bridge.csv`

為避免橫向欄位過寬，本實驗採用**直向呈現法（Vertical Trace Format）**，每一行代表一個策略分支在「RELEASE 釋放當下」的各項物理指標。

### 📋 欄位定義 (22 Columns)
1.  **`Seed`**：隨機種子（如 2）。
2.  **`Branch`**：決策分支名稱，有以下 4 種值：
    *   `Late_Policy (slack0)` 或 `Late_Policy (ppo)`
    *   `Early_Policy (ppo)` 或 `Early_Policy (slack0)`
    *   `Early_Release_Then_Hold` (早釋放，隨後 HOLD)
    *   `Early_Release_Always` (早釋放，隨後每步放行)
3.  **`Event`**：觸發 RELEASE 的事件 ID（如 47, 52 等）。
4.  **`Time`**：該釋放事件發生時的仿真時間點。
5.  **`PPO_Prob`**：該狀態下高階 PPO 門控決策放行的機率（0.0 ~ 1.0）。
6.  **`Buf_Size`**：釋放前，待辦緩衝區內的工單數量。
7.  **`Buf_Slack_Min`** / **`Avg`** / **`Q25`**：釋放前，緩衝區寬裕時間的 Min, Average 以及第 25 分位數統計值。
8.  **`Buf_Urgent`** / **`Normal`**：釋放前，緩衝區內的急單與普通單數量。
9.  **`WIP_Bf`** / **`WIP_Af`**：釋放前/後車間在制品工單數。
10. **`MK_Bf`** / **`MK_Af`**：釋放前/後車間預估完工時間（Makespan）。
11. **`Conf_TD`**：歷史已確定完工出廠工單的總延誤（釋放前後不變）。
12. **`WIP_TD_Bf`** / **`WIP_TD_Af`**：釋放前/後在製品預估延誤。
13. **`Total_TD_Bf`** / **`Total_TD_Af`**：釋放前/後系統總預估延誤。
14. **`WIP_Slack_Min_Bf`** / **`Af`**：釋放前/後車間內最緊迫 WIP 工單的寬裕時間。
15. **`WIP_Slack_Avg_Bf`** / **`Af`**：釋放前/後車間內 WIP 工單的平均寬裕時間。
16. **`WIP_Slack_Q25_Bf`** / **`Af`**：釋放前/後車間內 WIP 工單的第 25 分位數寬裕時間。
17. **`Mch_Load_Avg_Bf`** / **`Af`**：釋放前/後平均機台剩餘負載。
18. **`Mch_Load_Std_Bf`** / **`Af`**：釋放前/後機台負載標準差。
19. **`Mch_Load_Span_Bf`** / **`Af`**：釋放前/後最忙與最閒機台的工時跨度。

### 📊 資料列定義 (4 Rows)
對於每個測試實例（例如 Seed 2, Event A=47, Event B=52），CSV 檔案中**只輸出以下 4 行核心對照資料**：

1.  **第一列**：`Late_Policy` 在 **Event B (52)** 釋放時的各項指標。
2.  **第二列**：`Early_Policy` 在 **Event A (47)** 釋放時的各項指標。
3.  **第三列**：`Early_Release_Then_Hold` 在 **Event B (52)** 釋放時的各項指標。
4.  **第四列**：`Early_Release_Always` 在 **Event B (52)** 釋放時的各項指標。

---

## 3. 三大延誤成分之精確計算公式
1. **`Confirmed_TD` (歷史已確定延誤)**：
   * **定義**：已經完全完工且出廠的工單之延誤。
   * **公式**：`Confirmed_TD = sum(max(0, _job_history_finishes[jid] - due) for jid in finished_jids)`
2. **`WIP_TD_Before` (WIP當下TD)**：
   * **定義**：已開工但未完工的在製品工單，在釋放前一瞬間以舊計畫運作時的預計延誤。
   * **公式**：`WIP_TD_Before = sum(max(0, p_finish[jid] - due) for jid in wip_jids)`
3. **`Rescheduled_TD_After` (release後的TD)**：
   * **定義**：釋放 Action=1 且低階排程器重新規劃排程後，新計畫下車間所有活動中工單的預估延誤總和。
