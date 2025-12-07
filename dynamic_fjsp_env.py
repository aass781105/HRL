
import numpy as np
import random
from copy import deepcopy
from typing import List, Dict, Optional, Tuple # Added List, Dict, Optional, Tuple

from params import configs
from data_utils import SD2_instance_generator
from fjsp_env_same_op_nums_online import FJSPEnvForSameOpNums, EnvState
from global_env import EventBurstGenerator, split_matrix_to_jobs, JobSpec, OperationSpec # Added JobSpec, OperationSpec

# Helper function to ensure a fixed number of jobs per arrival event
def fixed_k_sampler(K: int):
    """固定一次釋放 K 筆的 sampler。"""
    def _fn(rng: np.random.Generator) -> int:
        return int(K)
    return _fn

class DynamicFJSPEnv:
    """
    一個用於 PPO 訓練的動態 FJSP 環境。

    這個環境模擬了工單隨時間動態到達的場景。一個 Episode 對應一次完整的動態模擬。
    PPO Agent 的角色是在固定的時間間隔（每個 episode 隨機抽取）被觸發，
    對從緩衝區釋放的一批工單進行排程。
    """
    def __init__(self, config):
        """
        初始化動態環境。
        """
        print("Initializing Dynamic FJSP Environment for PPO training...")
        self.config = config
        
        # 動態模擬相關參數
        self.n_j = config.n_j
        self.n_m = config.n_m
        self.event_horizon = config.event_horizon
        
        # 從 config 中讀取 release_interval 的上下限
        #將這些值加入 params.py 中
        self.release_interval_min = float(getattr(config, "release_interval_min", 25.0))
        self.release_interval_max = float(getattr(config, "release_interval_max", 75.0))
        self.burst_size = int(getattr(config, "burst_size", 1))
        self.initial_jobs = int(getattr(config, "init_jobs", 10))

        # 模擬狀態變數
        self.t_now = 0.0
        self.machine_free_time = np.zeros(self.n_m, dtype=np.float32)
        self.buffer = []
        self.release_count = 0
        self.is_flushing = False
        self.release_interval = 0.0
        self.next_release_time = 0.0

        # 用於處理切分工單的狀態
        self._last_scheduled_rows: List[Dict] = []
        self._last_scheduled_jobs_snapshot: List[JobSpec] = []

        # 內部工具
        rng = np.random.default_rng(int(getattr(config, "event_seed", 42)))
        interarrival_mean = float(getattr(config, "interarrival_mean", 50.0))
        
        self.job_generator = EventBurstGenerator(
            sd2_fn=SD2_instance_generator,
            base_config=deepcopy(config),
            n_machines=config.n_m,
            interarrival_mean=interarrival_mean,
            k_sampler=fixed_k_sampler(self.burst_size),
            rng=rng,
        )
        
        # 複用靜態環境來解決子問題
        # 這個內部環境的 num_envs 始終為 1
        static_env_config = deepcopy(config)
        static_env_config.num_envs = 1
        self.static_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        
        print("Dynamic Environment Initialized.")

    def reset(self):
        """
        重置整個動態模擬 Episode。
        - 清空 buffer, machine_free_time, t_now。
        - 隨機抽取本次 episode 的 release_interval。
        - 產生初始工單並放入 buffer。
        - 準備並返回第一個 state。
        """
        print("Resetting Dynamic Environment...")
        # 重置模擬狀態
        self.t_now = 0.0
        self.machine_free_time = np.zeros(self.n_m, dtype=np.float32)
        self.buffer = []
        self.release_count = 0
        self.is_flushing = False

        # 重置切分工單相關狀態
        self._last_scheduled_rows = []
        self._last_scheduled_jobs_snapshot = []

        # 隨機化發布間隔
        self.release_interval = random.uniform(self.release_interval_min, self.release_interval_max)
        self.next_release_time = self.t_now + self.release_interval
        
        # 重置 job_generator 並生成初始工單
        self.job_generator.reset()
        if self.initial_jobs > 0:
            init_cfg = deepcopy(self.config)
            setattr(init_cfg, "n_j", self.initial_jobs)
            job_length, op_pt, _ = SD2_instance_generator(init_cfg)
            initial_jobs_list = split_matrix_to_jobs(job_length, op_pt, base_job_id=0, t_arrive=0.0)
            self.buffer.extend(initial_jobs_list)
            max_id = max((j.job_id for j in self.buffer), default=-1)
            self.job_generator.bump_next_id(max_id + 1)
        
        print(f"New episode started. Release interval set to: {self.release_interval:.2f}")

        # 準備並返回第一個狀態 (佔位符)
        # 在實際的訓練循環中，這一步會由 train.py 來調用
        return None

    def _slice_jobs_at_t_event(self, t_event: float,
                               last_full_rows: List[Dict],
                               last_jobs_snapshot: List[JobSpec]) -> List[JobSpec]:
        """
        根據 t_event 切分上一個子問題的排程結果，找出尚未完成的工單部分。
        適應自 global_env.py 中的 event_release_and_reschedule 邏輯。
        """
        remain_jobs: List[JobSpec] = []
        by_job: Dict[int, List[Dict]] = {}
        if last_full_rows:
            for r in last_full_rows:
                by_job.setdefault(int(r["job"]), []).append(r)

        if last_jobs_snapshot:
            for js in last_jobs_snapshot:
                jid = int(js.job_id)
                seq = sorted(by_job.get(jid, []), key=lambda x: x["op"]) if jid in by_job else []
                
                # Count ops that started before t_event
                k_done = sum(1 for r in seq if float(r["start"]) < t_event) 
                # Ops in progress at t_event
                inprog = [r for r in seq if (float(r["start"]) < t_event < float(r["end"]))]

                if inprog:
                    ready_at = float(inprog[0]["end"])  # Non-preemptive: next op waits for this in-progress op to finish
                    # Slice from the next op after in-progress
                    slice_from = int([r["op"] for r in seq].index(inprog[0]["op"]) + 1) 
                else:
                    ready_at = t_event # If no op in progress, ready at current event time
                    slice_from = int(k_done) # Slice from the next op after completed ones

                base_offset = int(js.meta.get("op_offset", 0))
                new_offset = base_offset + slice_from
                ops_left = js.operations[slice_from:] # The remaining operations for this job
                
                if ops_left:
                    js2 = JobSpec(job_id=jid, operations=list(ops_left), meta=dict(js.meta))
                    js2.meta["op_offset"] = new_offset
                    js2.meta["ready_at"] = ready_at  # Store absolute time
                    remain_jobs.append(js2)
        return remain_jobs

    def advance_simulation(self, last_scheduled_rows: Optional[List[Dict]], last_scheduled_jobs_snapshot: Optional[List[JobSpec]]):
        """
        在子問題被解決後，推進模擬時間並更新環境狀態。
        這個方法由外部的訓練循環 (train.py) 調用。
        """
        # 1. 從 static_env 獲取排程結果，更新全局時間線
        # 只有在 static_env 已經被初始化過數據後，才嘗試獲取 true_mch_free_time
        if hasattr(self.static_env, 'true_mch_free_time'):
            self.machine_free_time = self.static_env.true_mch_free_time[0, :].copy()

        # 捕獲上一個子問題的詳細排程結果和 JobSpec 快照
        self._last_scheduled_rows = last_scheduled_rows if last_scheduled_rows is not None else []
        self._last_scheduled_jobs_snapshot = last_scheduled_jobs_snapshot if last_scheduled_jobs_snapshot is not None else []

        # 如果不在 flush 模式，正常推進時間
        if not self.is_flushing:
            self.t_now = self.next_release_time
            self.release_count += 1

            # 檢查是否達到 event_horizon，如果達到則進入 flush 模式
            if self.release_count >= self.event_horizon:
                print(f"Event horizon of {self.event_horizon} reached. Entering flush mode.")
                self.is_flushing = True
        
        # 處理被切分的工單 (remain_jobs)
        # 這部分邏輯需要從 global_env.py 的 Orchestrator 中適配過來
        if self._last_scheduled_rows and self._last_scheduled_jobs_snapshot:
            remain_jobs = self._slice_jobs_at_t_event(self.t_now, 
                                                      self._last_scheduled_rows, 
                                                      self._last_scheduled_jobs_snapshot)
            if remain_jobs:
                self.buffer.extend(remain_jobs)

        # 2. 從 job_generator 獲取新到達的工單
        # 注意：在 flush 模式下，我們不再生成新工單，只處理緩衝區中的剩餘工單
        if not self.is_flushing:
            # 在 main.py 中，generate_burst 是在 t_now 時生成，所以這裡也一樣
            new_jobs = self.job_generator.generate_burst(self.t_now)
            if new_jobs:
                self.buffer.extend(new_jobs)
            
            # 更新下一個發布時間
            self.next_release_time += self.release_interval

        # 3. 判斷整個動態 episode 是否結束
        # 結束條件：處於 flush 模式，並且緩衝區已經沒有待處理的工單了
        done = self.is_flushing and not self.buffer
        if done:
            print(f"Episode finished at t={self.t_now:.2f}. Final makespan: {np.max(self.machine_free_time):.2f}")

        return done



if __name__ == '__main__':
    # 一個簡單的測試，展示如何使用這個環境
    # 這部分代碼只是示意，並不能直接運行
    
    # 1. 初始化環境
    env = DynamicFJSPEnv(configs)
    
    # 2. 開始一個新的 episode
    state = env.reset()
    
    print(f"Initial buffer size: {len(env.buffer)}")
    
    # 3. 從 buffer 準備第一個子問題
    # sub_problem_state = env._get_state_from_buffer()
    # print(f"Buffer size after getting state: {len(env.buffer)}")
    # if sub_problem_state:
    #      print("Successfully created a subproblem state for the PPO agent.")

    pass
