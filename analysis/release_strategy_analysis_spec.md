# Release Strategy Analysis Specification

## 目的

比較 PPO、Cadence、Slack 等上層 release 策略在相同環境與 seed 下的決策狀態與結果。

分析 checkpoint 預計設在：

```text
event_id = 0, 40, 80, 120, 160
```

checkpoint 應在該 event 的實際 release/hold 決策前擷取。這些欄位描述的是策略實際走到該 checkpoint 時的狀態，不是反事實的 release 結果。

## Checkpoint 欄位

完整 checkpoint 欄位如下，原始欄位與補充欄位全部保留：

```text
strategy
seed
event_id
sim_time

current_makespan
current_tardiness
current_objective

buffer_job_count
buffer_total_work
buffer_slack_min
buffer_slack_mean
buffer_slack_q25
buffer_slack_std
buffer_negative_slack_count
buffer_negative_slack_ratio

wip_job_count
wip_op_count
wip_total_work
wip_tardiness
wip_slack_min
wip_slack_mean
wip_slack_q25
wip_slack_std

machine_load_mean
machine_load_std
machine_load_min
machine_load_max
machine_load_imbalance
```

### 基本資訊

```text
strategy
seed
event_id
sim_time
```

### 目前排程結果

```text
current_makespan
current_tardiness
current_objective
```

其中：

```text
current_objective = 0.5 * current_makespan + 0.5 * current_tardiness
```

### Buffer 狀態

```text
buffer_job_count
buffer_op_count
buffer_total_work
buffer_slack_min
buffer_slack_mean
buffer_slack_q25
```

### WIP 狀態

```text
wip_job_count
wip_op_count
wip_total_work
wip_tardiness
```

### Machine 負載

```text
machine_load_mean
machine_load_std
```

## 完整 Checkpoint 欄位

以下欄位全部納入 checkpoint 記錄，且不與反事實 release 結果混用。

```text
buffer_negative_slack_count
buffer_negative_slack_ratio
wip_slack_min
wip_slack_mean
wip_slack_q25
machine_load_min
machine_load_max
machine_load_imbalance
decision_step
release_count_so_far
events_since_last_release
active_job_count
```

這些欄位用來補足 WIP slack 分布、瓶頸機台負載，以及策略在 checkpoint 前的 release 歷史。

## 欄位語意界線

### Checkpoint 狀態

```text
current_*
buffer_*
wip_*
machine_*
```

表示策略實際軌跡在該 event checkpoint 當下的狀態。

### 反事實 release 結果

後續另行記錄：

```text
release_now_makespan
release_now_tardiness
release_now_objective
```

這些欄位表示「如果在 checkpoint 當下立即 release」的模擬結果，不應覆蓋 `current_*` 欄位。

### 區間摘要

後續可另外建立每個 40-event 區間一列的摘要，例如：

```text
arrived_job_count
urgent_job_count
initial_slack_min
initial_slack_q25
initial_slack_mean
release_count
```

`segment_td_end`、`segment_buffer_jobs_end`、`segment_wip_jobs_end` 不需要另外保存，因為它們等同於下一個 checkpoint 的 `current_tardiness`、`buffer_job_count`、`wip_job_count`。

## CSV 短欄名

MD 中的完整名稱只作為語意文件；實際 CSV 使用以下短欄名。

| 語意名稱 | CSV 欄名 |
|---|---|
| strategy | strat |
| seed | seed |
| event_id | event |
| sim_time | time |
| current_makespan | mk |
| current_tardiness | td |
| current_objective | obj |
| buffer_job_count | b_jobs |
| buffer_total_work | b_work |
| buffer_slack_min | b_smin |
| buffer_slack_mean | b_smean |
| buffer_slack_q25 | b_sq25 |
| buffer_slack_std | b_sstd |
| buffer_negative_slack_count | b_neg_n |
| buffer_negative_slack_ratio | b_neg_r |
| wip_job_count | w_jobs |
| wip_op_count | w_ops |
| wip_total_work | w_work |
| wip_tardiness | w_td |
| wip_slack_min | w_smin |
| wip_slack_mean | w_smean |
| wip_slack_q25 | w_sq25 |
| wip_slack_std | w_sstd |
| machine_load_mean | m_lmean |
| machine_load_std | m_lstd |
| machine_load_min | m_lmin |
| machine_load_max | m_lmax |
| machine_load_imbalance | m_imb |
| release_now_makespan | rn_mk |
| release_now_tardiness | rn_td |
| release_now_objective | rn_obj |
| arrived_job_count | arr_jobs |
| urgent_job_count | urg_jobs |
| initial_slack_min | in_smin |
| initial_slack_q25 | in_sq25 |
| initial_slack_mean | in_smean |
| release_count | rel_n |

## CSV 數值格式

```text
seed、event、各類 job/op/count 欄位：整數
time、mk、td、obj、work、slack、machine load、release-now 數值：整數
b_sstd、w_sstd、m_lstd：小數點後 2 位
ratio 欄位，例如 b_neg_r：小數點後 3 位
```

時間相關整數欄位採四捨五入，不直接截斷小數。

空白或不適用的欄位使用空值，不以 `0` 代替，避免和真正的零值混淆。
