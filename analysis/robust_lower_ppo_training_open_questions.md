# 下層 PPO 穩定性訓練：未決事項

本文件只記錄目前實作中刻意跳過、需要之後再確認的事項。已經定案的
state、reward 係數、`/10` 放置位置、Fresh/Reschedule 比例與 reference
policy 共存方式，記錄在
`analysis/robust_lower_ppo_training_plan.md`，不在此重複討論。

## 已先實作的範圍

- 新增獨立腳本 `train_ll_stability_finetune.py`。
- 舊 lower-level checkpoint 以 frozen reference policy 使用；trainable policy
  由該 checkpoint 初始化，兩者分開存在。
- 新訓練資料使用 Fresh 30% 與 virtual-cut Reschedule 70%。
- Reschedule 子問題以 reference policy 先完成一個靜態排程，再選取 virtual cut，
  將相對於 cut 的 future old operations 與新 jobs 組成新的靜態子問題。
- 穩定性 reward 只計算目前 append action 新造成的 increment：
  `-(1 * flip_increment + 3 * machine_change_increment)`，再與既有 reward
  一起除以 `ll_reward_divisor`，預設為 10。
- 新 YAML 的 operation/pair state 使用 22/12 維，critic 使用已定義的 6 維
  stability summary；舊 YAML 的預設仍維持 stability disabled。
- `LLMemory` 的 done-aware GAE 只包在新腳本使用的 memory wrapper 中，沒有改動
  舊 curriculum trainer 的 GAE 行為。

## 刻意跳過的事項

### 1. 外部動態環境歷史的精確重建

目前腳本沒有直接讀取 `hrl_main.py` 或已儲存 dynamic instance 的每次歷史
release log 來建立 training sample。原因是不同輸出檔的 job ID、絕對時間、
machine calendar 與 operation reference 可能不是同一個 schema。

目前只使用「reference lower-level policy + virtual cut」建立可控的靜態
reschedule sample。之後若要直接用 dynamic instance history，需要先確認固定
的輸入 schema：

- cut 當下仍在加工的 operation
- 每台 machine 的 busy interval
- 每個 retained operation 的原始 machine 與 machine sequence rank
- job 的原始 operation ID 到子問題 operation ID 對應

### 2. 歷史 machine calendar 的完整區間特徵

目前 virtual-cut sample 只把 cut 當下的 machine remaining-free time 與 job ready
time 放回 lower-level environment。沒有把 cut 前所有已排程 operation 的完整
busy intervals 當成不可插入的歷史 calendar 傳入 state。

這不影響目前 append-only、`enable_gap_insertion: false` 的第一版流程；如果之後
要研究 gap insertion 或需要完整還原 cut 當下的 machine calendar，必須另外定義
歷史區間是否屬於 state 與如何不重複計算 stability violation。

### 3. stability validation 的 checkpoint 選擇規則

Validation 仍固定輸出 `0.5 * MK + 0.5 * TD`、flip 與 machine-change count，
但目前不再用 validation objective 選擇 checkpoint。訓練完成後會無條件儲存
最後一個 update 的 policy，因此不會因為 best-objective 選擇而回到較早的權重。

### 4. reward calibration 的最終數值

舊 detailed reward log 曾使用過不同的診斷尺度；目前已修正新 log 使用實際 `/10`
後的 MK/TD components，但舊 log 不能可靠地反推每一筆 transition 的 exact
reward。`flip=1`、`machine_change=3` 先照已定案的初版執行，最終量級要等新訓練
log 與 stability audit 對齊後再評估。

### 5. 舊 checkpoint 對新輸入維度的遷移效果

目前沿用 `LLMLPNet.load_state_dict` 的 partial input adaptation：舊 checkpoint
已有的 input columns 複製到新 layer，新增 state columns 保留新初始化值。
尚未把新增 state columns 的初始化、warm-up 長度或獨立 learning rate 視為
正式訓練策略；這需要實驗後再決定。

## 目前不應混入本版的變更

- 不加入 high-level release policy 或 dynamic cadence reward。
- 不把 stability count 改成 rate。
- 不把 machine change 再次計入 pair flip。
- 不把 cumulative stability count 每一步重複扣除。
- 不因為 validation 尚未定案而改寫既有 curriculum trainer。
