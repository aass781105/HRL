import argparse
import yaml

def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description='Arguments for DANIEL_FJSP')
parser.add_argument('--config', type=str, default='', help='Path to a YAML config file')

# ============================
# System & Hardware
# ============================
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')


# ============================
# File Naming & Paths
# ============================
parser.add_argument('--path_name', type=str, default='test', help='path name for saving network')
parser.add_argument('--model_suffix', type=str, default='', help='Suffix of the model')
parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')
parser.add_argument('--model_source', type=str, default='static_ppo', help='Suffix of the data that model trained on')
parser.add_argument('--data_source', type=str, default='BenchData', help='Suffix of test data')


# ============================
# Experiment Control
# ============================
parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')
parser.add_argument('--cover_heu_flag', type=str2bool, default=False, help='Whether covering test results of heuristics')
parser.add_argument('--cover_train_flag', type=str2bool, default=True, help='Whether covering the trained model')
parser.add_argument('--sort_flag', type=str2bool, default=True, help='Whether sorting the printed results by the makespan')


# ============================
# Data Generation & Instance Settings
# ============================
# Seeds
parser.add_argument('--seed_datagen', type=int, default=200, help='Seed for data generation')
parser.add_argument('--seed_train_vali_datagen', type=int, default=100, help='Seed for generate validation data')

# Instance Parameters
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of the instance')
parser.add_argument('--n_op', type=int, default=50, help='Number of operations of the instance')
parser.add_argument('--low', type=int, default=1, help='Lower Bound of processing time(PT)')
parser.add_argument('--high', type=int, default=99, help='Upper Bound of processing time')

# SD2 Generation Specifics
parser.add_argument('--op_per_job', type=float, default=0, help='Number of operations per job, default 0, means the number equals m')
parser.add_argument('--op_per_mch_min', type=int, default=1, help='Minimum number of compatible machines for each operation')
parser.add_argument('--op_per_mch_max', type=int, default=5, help='Maximum number of compatible machines for each operation')
parser.add_argument('--data_size', type=int, default=100, help='The number of instances for data generation')
parser.add_argument('--data_type', type=str, default="test", help='Generated data type (test/vali)')


# ============================
# PPO Network Architecture
# ============================
parser.add_argument('--fea_j_input_dim', type=int, default=10, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_heads_OAB', nargs='+', type=int, default=[4, 4], help='Number of attention head of operation message attention block')
parser.add_argument('--num_heads_MAB', nargs='+', type=int, default=[4, 4], help='Number of attention head of machine message attention block')
parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[32, 8], help='Output dimension of the DAN layers')

# Actor-Critic Details
parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=64, help='Hidden dimension of Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=64, help='Hidden dimension of Critic network')


# ============================
# PPO Training Algorithm
# ============================
parser.add_argument('--seed_train', type=int, default=300, help='Seed for training')
parser.add_argument('--num_envs', type=int, default=20, help='Batch size for training environments')
parser.add_argument('--max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='Entropy loss coefficient')
parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')
parser.add_argument('--train_size', type=str, default="10x5", help='Size of training instances')
parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
parser.add_argument('--reset_env_timestep', type=int, default=20, help='Interval for reseting the environment')
parser.add_argument('--minibatch_size', type=int, default=512, help='Batch size for computing the gradient')


# ============================
# Testing & Evaluation (Static)
# ============================
parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
parser.add_argument('--eval_seed', type=int, default=42, help='Seed for dynamic evaluation')
parser.add_argument('--eval_runs_per_instance', type=int, default=1, help='Number of runs per test instance')
parser.add_argument('--eval_num_instances', type=int, default=10, help='Number of test instances to evaluate')
parser.add_argument('--test_data', nargs='+', default=['Hurink_vdata'], help='List of data for testing')
parser.add_argument('--test_mode', type=str2bool, default=False, help='Whether using the sampling strategy in testing')
parser.add_argument('--sample_times', type=int, default=100, help='Sampling times for the sampling strategy')
parser.add_argument('--test_model', nargs='+', default=['curriculum_train_10x5+mix','curriculum_train_40x5+mix'], help='List of model for testing')
parser.add_argument('--test_method', nargs='+', default=["MWKR"], help='List of heuristic methods for testing')
parser.add_argument('--eval_model_name', type=str, default="original", help='用於儲存CSV的檔名')


# ============================
# Dynamic Simulation (Event-Driven)
# ============================
parser.add_argument('--event_horizon', type=float, default=100.0, help='事件驅動模式的模擬事件上限')
parser.add_argument('--interarrival_mean', type=float, default=25, help='Poisson interarrival_mean')
parser.add_argument('--init_jobs', type=int, default= 10, help='初始工單數')
parser.add_argument('--burst_size', type=int, default=1, help='每次生成工單數')
parser.add_argument('--event_seed', type=int, default=42, help='事件驅動到達過程的亂數種子（Exponential 間隔）')
parser.add_argument('--episode_seed_base', type=int, default=12345, help='episode 級別的基種子；每個 episode 以此為基準派生子亂數流')

# Cadence Sampling
parser.add_argument('--cadence_min', type=int, default=8, help='固定事件次數（cadence）抽樣下限（含）')
parser.add_argument('--cadence_max', type=int, default=12, help='固定事件次數（cadence）抽樣上限（含）')
parser.add_argument('--cadence_choices', type=int, nargs='*', default=None, help='固定事件次數（cadence）的離散候選集合（例如：--cadence_choices 1 2 4 8）')


# ============================
# PPO Orchestrator Integration
# ============================
parser.add_argument('--use_ppo', type=str2bool, default=True, help='是否改用 PPO 進行靜態批次的一次排到底')
parser.add_argument('--ppo_model_path', type=str, default=r'trained_network\SD2\sample_num_job.pth', help='PPO 權重檔 .pth 路徑（use_ppo=True 時必填）')
parser.add_argument('--ppo_sample', type=str2bool, default=False, help='PPO 推論是否採用抽樣；False=貪婪/取最大機率')


# ============================
# DDQN Gate Policy & Training
# ============================
parser.add_argument('--gate_policy', type=str, default='ddqn',
                    choices=['ddqn', 'always', 'threshold', 'time'],
                    help='Gate 策略：ddqn=用模型；always=每次到達都放行；threshold=緩衝超門檻才放；time=事件間隔累積超過門檻才放')
parser.add_argument('--eval_action_selection', type=str, default='greedy',
                    choices=['sample', 'greedy'],
                    help='sample or greedy')
parser.add_argument('--ddqn_model_path', type=str, default=r"yaml\55_100_30.yml", help='DDQN 推論權重路徑（.pth）')

# Gate Thresholds & Obs Params
parser.add_argument('--gate_time_threshold', type=float, default=250.0, help='當 gate_policy=time 時，兩次釋放之間所需的最小事件間隔（可累加）')
parser.add_argument('--gate_buffer_threshold', type=int, default=0, help='當 gate_policy=threshold 時的緩衝門檻（>=此值才 release）')
parser.add_argument('--gate_obs_buffer_cap', type=int, default=0, help='Gate 狀態中 buffer 正規化的分母（0=自動估計）')
parser.add_argument('--gate_time_scale', type=float, default=0.0, help='Gate 狀態中時間正規化尺度（0=以 interarrival_mean 近似）')

# DDQN Training Hyperparameters
parser.add_argument('--ddqn_train', type=str2bool, default=False, help='是否執行 DDQN 訓練流程（否則僅推論）')
parser.add_argument('--ddqn_reward_mode', type=str, default='original',
                    choices=['original', 'stability'],
                    help='Reward mode: original (idle + makespan) or stability (original + action penalty)')
parser.add_argument('--ddqn_episodes', type=int, default=100, help='DDQN 訓練集 episode 數')
parser.add_argument('--ddqn_lr', type=float, default=5e-5, help='DDQN 學習率')
parser.add_argument('--ddqn_gamma', type=float, default=0.995, help='DDQN 折扣因子 γ')
parser.add_argument('--ddqn_eps_start', type=float, default=0.6, help='ε-greedy 初始 ε')
parser.add_argument('--ddqn_eps_end', type=float, default=0.05, help='ε-greedy 最小 ε')
parser.add_argument('--ddqn_eps_decay_episodes', type=int, default=90, help='ε 從起始到終值的衰減 episode 數')
parser.add_argument('--ddqn_batch_size', type=int, default=128, help='DDQN 更新時的 minibatch 大小')
parser.add_argument('--ddqn_buffer_capacity', type=int, default=100_000, help='Replay buffer 容量')
parser.add_argument('--ddqn_target_tau', type=float, default=0.005, help='目標網路軟更新係數 τ')
parser.add_argument('--ddqn_seed', type=int, default=42, help='DDQN 訓練隨機種子（與事件種子獨立）')
parser.add_argument('--ddqn_validate_every', type=int, default=10, help='每多少個 episodes 做一次驗證（greedy）')
parser.add_argument('--ddqn_val_episodes', type=int, default=5, help='驗證時計算平均回報的 episodes 數')
parser.add_argument('--ddqn_out_dir', type=str, default='ddqn_ckpt', help='DDQN 訓練權重輸出資料夾')


# ============================
# Reward, Penalty & Scales
# ============================
parser.add_argument('--norm_scale', type=float, default=100, help='norm /scale')
parser.add_argument('--reward_scale', type=float, default=50.0, help='Scale factor for reward normalization')
parser.add_argument('--reward_alpha', type=float, default=0.3, help='Weight for Makespan in reward (alpha). Idle weight will be (1-alpha). Default 0.3 matches previous 0.3/0.7 split.')
parser.add_argument('--stability_scale', type=float, default=0.1, help='Scale factor for stability penalty in reward calculation')
parser.add_argument('--enable_full_idle_penalty', type=str2bool, default=False, help='If True, penalize HOLD when all machines idle but buffer has jobs')
parser.add_argument('--full_idle_penalty', type=float, default=100.0, help='Penalty value when full-idle HOLD happens')
parser.add_argument('--enable_final_flush_penalty', type=str2bool, default=True, help='If True, add final penalty when episode ends with leftover buffer')


# ============================
# Plotting & Visualization
# ============================
parser.add_argument('--plot_global_dir', type=str, default='plots/global', help='全局甘特圖輸出資料夾（每次重排立即輸出）')
parser.add_argument('--plot_batch_dir', type=str, default='plots/batch', help='批次甘特圖輸出資料夾（重排後該批 finalize 時輸出）')


# ============================
# External Solvers
# ============================
parser.add_argument('--max_solve_time', type=int, default=1800, help='The maximum solving time of OR-Tools')


# ============================
# Argument Parsing
# ============================
# 兩段式解析（最小化寫法）：
# 第一次：只拿 --config
_tmp, _ = parser.parse_known_args()

# 若有 YAML，就把 YAML 內容設為新的 defaults
if _tmp.config:
    with open(_tmp.config, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f) or {}
    parser.set_defaults(**file_cfg)

# 第二次：正式解析（優先序：CLI > YAML > 預設）
configs = parser.parse_args()