# debug_vector_info.py
import gymnasium as gym
import numpy as np
from event_gate_env import EventGateEnv
from params import configs

def debug_info():
    num_envs = 2
    def make_env():
        return EventGateEnv(
            n_machines=configs.n_m,
            heuristic=configs.scheduler_type,
            interarrival_mean=configs.interarrival_mean,
            burst_K=configs.burst_size,
            event_horizon=5, 
            init_jobs=int(configs.init_jobs)
        )

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    envs.reset(seed=42)
    
    print("--- Starting Simulation ---")
    done_count = 0
    while done_count < num_envs:
        actions = np.zeros(num_envs, dtype=int)
        _, _, dones, truncs, infos = envs.step(actions)
        
        for i in range(num_envs):
            if dones[i] or truncs[i]:
                done_count += 1
                print(f"\n[Env {i} Finished]")
                print(f"Infos Type: {type(infos)}")
                if isinstance(infos, dict):
                    print(f"Infos Keys: {list(infos.keys())}")
                    for k, v in infos.items():
                        if k == "final_info":
                            print(f"  - final_info[{i}]: {v[i]}")
                        elif isinstance(v, (list, np.ndarray)) and len(v) > i:
                            print(f"  - {k}[{i}]: {v[i]}")
                else:
                    print(f"Infos: {infos}")

    envs.close()

if __name__ == "__main__":
    debug_info()
