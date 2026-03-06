import os
import pandas as pd
from params import configs
from main import run_event_driven_until_nevents

def run_sensitivity_analysis():
    # 1. 設置實驗範圍
    cadence_range = range(1, 21) # 1 到 20
    results = []
    
    # 建立主輸出目錄
    base_plot_dir = "plots/cadence_study"
    os.makedirs(base_plot_dir, exist_ok=True)
    
    print("-" * 30)
    print("Starting Cadence Sensitivity Analysis (1 to 20)...")
    print("-" * 30)

    # 2. 執行循環測試
    master_seed = int(getattr(configs, "event_seed", 42))
    
    for c in cadence_range:
        print(f"\n[Testing] Cadence = {c}")
        
        # [CRITICAL] 確保每一輪的種子都完全一致，達成「同題競技」
        configs.event_seed = master_seed
        configs.gate_policy = "cadence"
        configs.gate_cadence = c
        configs.eval_model_name = f"cadence_{c:02d}"
        
        # 定義存圖與 CSV 的子資料夾
        sub_dir = os.path.join(base_plot_dir, f"cadence_{c:02d}")
        
        # 執行模擬
        mk, stats = run_event_driven_until_nevents(
            max_events=int(configs.event_horizon),
            interarrival_mean=configs.interarrival_mean,
            burst_K=configs.burst_size,
            plot_global_dir=sub_dir
        )
        
        # 3. 收集結果
        results.append({
            "Cadence": c,
            "Final_Makespan": mk,
            "Final_Tardiness": stats["total_tardiness"],
            "Release_Count": stats["release_count"]
        })
        
        print(f"Result: MK={mk:.2f}, TD={stats['total_tardiness']:.2f}, Releases={stats['release_count']}")

    # 4. 生成統計總表
    df = pd.DataFrame(results)
    summary_path = os.path.join(base_plot_dir, "cadence_sensitivity_summary.csv")
    df.to_csv(summary_path, index=False)
    
    # 同時印出表格方便複製
    print("\n" + "="*40)
    print("   CADENCE SENSITIVITY SUMMARY")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
