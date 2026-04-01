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
parser.add_argument('--data_source', type=str, default='SD2', help='Suffix of test data')


# ============================
# Experiment Control
# ============================
parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')
parser.add_argument('--cover_heu_flag', type=str2bool, default=False, help='Whether covering test results of heuristics')
parser.add_argument('--cover_train_flag', type=str2bool, default=True, help='Whether covering the trained model')
parser.add_argument('--sort_flag', type=str2bool, default=True, help='Whether sorting the printed results by the makespan')
parser.add_argument('--debug_reward_trace', type=str2bool, default=False, help='Whether printing detailed gate reward traces')
parser.add_argument('--debug_reward_positive_threshold', type=float, default=0.5, help='Print shaping traces when reward_shaping_penalty exceeds this threshold')


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
parser.add_argument('--fea_j_input_dim', type=int, default=14, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=9, help='Dimension of machine raw feature vectors')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[64, 64, 64], help='List of output dimensions for each layer of the Feature Encoder')

# Actor-Critic Details
parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=512, help='Hidden dimension of Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=256, help='Hidden dimension of Critic network')


# ============================
# PPO Training Algorithm
# ============================
parser.add_argument('--seed_train', type=int, default=3, help='Seed for training')
parser.add_argument('--num_envs', type=int, default=100, help='Batch size for training environments')
parser.add_argument('--max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--lr_decay', type=str2bool, default=True, help='Whether to decay learning rate linearly')
parser.add_argument('--lr_end', type=float, default=1e-4, help='Final learning rate at end of training')
parser.add_argument('--gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.03, help='Entropy loss coefficient')
parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')
parser.add_argument('--train_size', type=str, default="10x5", help='Size of training instances')
parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
parser.add_argument('--reset_env_timestep', type=int, default=40, help='Interval for reseting the environment')
parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')


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
parser.add_argument('--eval_model_name', type=str, default="new", help='用於儲存檔案的檔名')


# ============================
# Dynamic Simulation (Event-Driven)
# ============================
parser.add_argument('--event_horizon', type=float, default=20.0, help='事件驅動模式的模擬事件上限')
parser.add_argument('--interarrival_mean', type=float, default=25, help='Poisson interarrival_mean')
parser.add_argument('--init_jobs', type=int, default= 5, help='初始工單數')
parser.add_argument('--burst_size', type=int, default=1, help='每次生成工單數')
parser.add_argument('--event_seed', type=int, default=42, help='事件驅動到達過程的亂數種子（Exponential 間隔）')
parser.add_argument('--episode_seed_base', type=int, default=12345, help='episode 級別的基種子；每個 episode 以此為基準派生子亂數流')
parser.add_argument('--fast_mode', type=str2bool, default=False, help='是否開啟高速模式（跳過甘特圖與詳細 CSV 生成）')

# ============================
# Curriculum Learning Specifics
# ============================
parser.add_argument('--curriculum_cycle', type=int, default=250, help='Updates per curriculum stage')
parser.add_argument('--tardiness_dilution_power', type=float, default=1, help='Beta factor for tardiness dilution')
parser.add_argument('--schedule_type', type=str, default='same', choices=['s2', 's3', 'same'], help='Type of curriculum schedule to use')
parser.add_argument('--due_date_mode', type=str, default='k', choices=['k', 'M'], help='Due date generation mode: k (Individual) or M (Common). Note: M is used primarily in static curriculum.')
parser.add_argument('--m_value', type=float, default=0.6, help='M-value used for Common Due Date (Static Curriculum Only)')
parser.add_argument('--due_date_tightness', type=float, default=1.2, help='Tightness base factor (k). Current logic uses U[1.2, 2.0] for individual due dates.')
parser.add_argument('--due_date_noise', type=float, default=0.0, help='Multiplicative noise level for due dates')

# ============================
# Unified Scheduling Controller
# ============================
parser.add_argument('--scheduler_type', type=str, default='PPO', 
                    choices=['PPO', 'SPT', 'MWKR', 'FIFO', 'OR-Tools'],
                    help='Unified scheduling method used across all stages (Init, Dynamic, Flush)')
parser.add_argument('--ppo_model_path', type=str, default=r'trained_network\SD2\same.pth', help='PPO 權重檔 .pth 路徑')
parser.add_argument('--ppo_sample', type=str2bool, default=False, help='PPO 推論是否採用抽樣；False=貪婪/取最大機率')


parser.add_argument('--gate_policy', type=str, default='ppo',
                    choices=['ppo', 'cadence', 'slack_threshold'],
                    help='High-level gate policy: ppo=actor-critic, cadence=fixed event release cadence, slack_threshold=release when buffer min slack is below threshold')
parser.add_argument('--gate_cadence', type=int, default=5, help='當 gate_policy=cadence 時，每隔幾個到達事件釋放一次緩衝區')
parser.add_argument('--buffer_slack_release_threshold', type=float, default=0.0, help='When gate_policy=slack_threshold, release if current buffer min slack is below this threshold')
parser.add_argument('--eval_action_selection', type=str, default='greedy',
                    choices=['sample', 'greedy'],
                    help='sample or greedy')
parser.add_argument('--ppo_gate_model_path', type=str, default=r"ppo_ckpt\ppo_gate_latest.pth", help='PPO gate 推論權重路徑（.pth）')
parser.add_argument('--ppo_gate_name', type=str, default='test', help='PPO gate 訓練存檔名稱 (不含 .pth)')


# ============================
# Reward, Penalty & Weights
# ============================
parser.add_argument('--reward_alpha', type=float, default=0.3, help='Weight for Makespan in reward (alpha). Idle weight will be (1-alpha). Default 0.3 matches previous 0.3/0.7 split.')
parser.add_argument('--tardiness_alpha', type=float, default=1.0, help='Weight for Tardiness in PPO reward calculation.')
parser.add_argument('--stability_scale', type=float, default=0.0, help='決策穩定性懲罰 (Action 1 的額外扣分)。設為 0 代表純效能模式。')
parser.add_argument('--buffer_penalty_coef', type=float, default=0.0, help='Coefficient for buffer tardiness penalty')
parser.add_argument('--stability_mode', type=str, default='immediate_all', choices=['immediate_all', 'free_threshold'], help='Stability reward mode: immediate_all = current per-release penalty; free_threshold = releases are free until stability_free_releases, then penalize via terminal or redistribution.')
parser.add_argument('--stability_terminal_only', type=str2bool, default=False, help='If true, apply stability penalty only once at episode end using agent-chosen release count.')
parser.add_argument('--stability_free_releases', type=int, default=0, help='Number of agent-chosen releases that are free before stability penalty starts.')
parser.add_argument('--release_penalty_coef', type=float, default=0.1, help='Deprecated legacy parameter. No longer used by gate reward logic.')
parser.add_argument('--td_signal_source', type=str, default='', choices=['', 'none', 'agent_only', 'baseline_gap_final', 'baseline_gap_release_interval'], help='High-level TD reward source selector. Empty string keeps legacy compatibility.')
parser.add_argument('--td_credit_mode', type=str, default='', choices=['', 'step_only', 'redistribute', 'terminal_only'], help='High-level TD credit assignment mode. Empty string keeps legacy compatibility.')
parser.add_argument('--stability_mode_v2', type=str, default='', choices=['', 'off', 'immediate_all', 'immediate_all_terminal', 'free_threshold_terminal', 'free_threshold_distributed'], help='High-level stability mode selector. Empty string keeps legacy compatibility.')
parser.add_argument('--shaping_reward_coef', type=float, default=0.0, help='Coefficient for shaping reward term')
parser.add_argument('--td_reward_coef', type=float, default=0.1, help='Coefficient for final TD reward term')
parser.add_argument('--mk_reward_coef', type=float, default=0.0, help='Weight for final MK reward term at simulation end')


# ============================
# Plotting & Visualization
# ============================
parser.add_argument('--plot_global_dir', type=str, default='plots/global', help='全局甘特圖輸出資料夾（每次重排立即輸出）')
parser.add_argument('--plot_batch_dir', type=str, default='plots/batch', help='批次甘特圖輸出資料夾（重排後該批 finalize 時輸出）')
parser.add_argument('--plot_run_name', type=str, default='', help='Optional run-name override used in exported folder naming')


# ============================
# External Solvers
# ============================
parser.add_argument('--max_solve_time', type=int, default=1800, help='The maximum solving time of OR-Tools')

# PPO Gate Training Hyperparameters
parser.add_argument('--ppo_gate_num_layers', type=int, default=3, help='Number of hidden layers in PPO gate')
parser.add_argument('--ppo_gate_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate network')
parser.add_argument('--ppo_gate_separate_trunks', type=str2bool, default=False, help='Use separate actor/critic trunks for PPO gate')
parser.add_argument('--ppo_gate_actor_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate actor trunk')
parser.add_argument('--ppo_gate_actor_num_layers', type=int, default=3, help='Number of hidden layers in PPO gate actor trunk')
parser.add_argument('--ppo_gate_critic_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate critic trunk')
parser.add_argument('--ppo_gate_critic_num_layers', type=int, default=3, help='Number of hidden layers in PPO gate critic trunk')
parser.add_argument('--ppo_gate_value_num_layers', type=int, default=3, help='Number of hidden layers used by PPO gate value head (1 means linear head)')
parser.add_argument('--ppo_gate_value_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate value head MLP')
parser.add_argument('--ppo_gate_updates', type=int, default=1, help='Number of PPO gate updates')
parser.add_argument('--ppo_gate_num_envs', type=int, default=4, help='Number of PPO gate training environments collected per update')
parser.add_argument('--ppo_gate_use_async_envs', type=str2bool, default=True, help='Use AsyncVectorEnv for PPO gate collection; falls back to SyncVectorEnv on failure')
parser.add_argument('--ppo_gate_train_seed', type=int, default=142, help='Base seed used by PPO gate training instances (separate from main/event seed)')
parser.add_argument('--ppo_gate_lr', type=float, default=1e-4, help='PPO gate learning rate')
parser.add_argument('--ppo_gate_lr_decay', type=str2bool, default=True, help='Whether to linearly decay PPO gate learning rate')
parser.add_argument('--ppo_gate_lr_end', type=float, default=5e-5, help='Final PPO gate learning rate when decay is enabled')
parser.add_argument('--ppo_gate_gamma', type=float, default=0.99, help='Discount factor for PPO gate')
parser.add_argument('--ppo_gate_gae_lambda', type=float, default=0.95, help='GAE lambda for PPO gate')
parser.add_argument('--ppo_gate_clip', type=float, default=0.2, help='PPO clipping ratio')
parser.add_argument('--ppo_gate_entropy_coef', type=float, default=0.01, help='Entropy coefficient for PPO gate')
parser.add_argument('--ppo_gate_entropy_decay', type=str2bool, default=False, help='Whether to linearly decay PPO gate entropy coefficient')
parser.add_argument('--ppo_gate_entropy_end', type=float, default=0.005, help='Final PPO gate entropy coefficient when decay is enabled')
parser.add_argument('--ppo_gate_value_coef', type=float, default=0.5, help='Value loss coefficient for PPO gate')
parser.add_argument('--ppo_gate_update_epochs', type=int, default=5, help='Epochs per PPO gate update')
parser.add_argument('--ppo_gate_minibatch_size', type=int, default=64, help='Minibatch size for PPO gate updates')
parser.add_argument('--ppo_gate_max_grad_norm', type=float, default=1.0, help='Gradient clipping norm for PPO gate')
parser.add_argument('--ppo_gate_same_problem_eval_every', type=int, default=0, help='Same-problem greedy eval interval for PPO gate; 0 disables it')
parser.add_argument('--ppo_gate_validate_every', type=int, default=10, help='Validation interval for PPO gate; 0 disables it')
parser.add_argument('--ppo_gate_instance_episodes', type=int, default=10, help='Episodes to reuse the same event instance before switching (PPO gate)')


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
