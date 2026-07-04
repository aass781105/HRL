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
parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')
parser.add_argument('--data_source', type=str, default='SD2', help='Suffix of test data')


# ============================
# Experiment Control
# ============================
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')
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
parser.add_argument('--low', type=int, default=1, help='Lower Bound of processing time(PT)')
parser.add_argument('--high', type=int, default=99, help='Upper Bound of processing time')

# SD2 Generation Specifics
parser.add_argument('--op_per_job', type=float, default=0, help='Number of operations per job, default 0, means the number equals m')
parser.add_argument('--enable_op_mixture', type=str2bool, default=False, help='Enable mixed per-job operation counts (used by low-level PPO training only).')
parser.add_argument('--op_per_mch_min', type=int, default=1, help='Minimum number of compatible machines for each operation')
parser.add_argument('--op_per_mch_max', type=int, default=5, help='Maximum number of compatible machines for each operation')
parser.add_argument('--data_size', type=int, default=100, help='The number of instances for data generation')
parser.add_argument('--data_type', type=str, default="test", help='Generated data type (test/vali)')


# ============================
# PPO Network Architecture
# ============================
parser.add_argument('--fea_j_input_dim', type=int, default=20, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=9, help='Dimension of machine raw feature vectors')
parser.add_argument('--fea_pair_input_dim', type=int, default=8, help='Dimension of pair raw feature vectors')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[64, 64, 64], help='List of output dimensions for each layer of the Feature Encoder')
parser.add_argument('--separate_actor_critic_encoder', type=str2bool, default=False, help='Use separate feature encoders for actor and critic in the low-level PPO.')

# Actor-Critic Details
parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=512, help='Hidden dimension of Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=256, help='Hidden dimension of Critic network')
parser.add_argument('--critic_size_context_max_n_j', type=float, default=30.0, help='Training max job count used to scale critic-only size context n_j/max_n_j')
# ============================
# PPO Training Algorithm
# ============================
parser.add_argument('--seed_train', type=int, default=3, help='Seed for training')
parser.add_argument('--ll_num_envs', type=int, default=100, help='Batch size for training environments')
parser.add_argument('--ll_max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--ll_lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--ll_gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--ll_k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--ll_eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--ll_vloss_coef', type=float, default=0.1, help='Critic loss coefficient')
parser.add_argument('--ll_due_vloss_coef', type=str2bool, default=False, help='Use due-setting-specific low-level critic loss coefficients when due_date_mode=range3_hold.')
parser.add_argument('--ll_vloss_coef_loose', type=float, default=0.1, help='Low-level vloss_coef used for range3_loose when ll_due_vloss_coef is enabled.')
parser.add_argument('--ll_vloss_coef_mixed', type=float, default=0.075, help='Low-level vloss_coef used for range3_mixed when ll_due_vloss_coef is enabled.')
parser.add_argument('--ll_vloss_coef_tight', type=float, default=0.05, help='Low-level vloss_coef used for range3_tight when ll_due_vloss_coef is enabled.')
parser.add_argument('--ll_ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--ll_entloss_coef', type=float, default=0.03, help='Entropy loss coefficient')
parser.add_argument('--ll_tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--ll_gae_lambda', type=float, default=0.98, help='GAE parameter')
parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
parser.add_argument('--reset_env_timestep', type=int, default=40, help='Interval for reseting the environment')
parser.add_argument('--ll_minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')


# ============================
# Testing & Evaluation (Static)
# ============================
parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
parser.add_argument('--eval_seed', type=int, default=42, help='Seed for dynamic evaluation')
parser.add_argument('--instance_json', type=str, default='', help='Fixed dynamic instance JSON path for replay/evaluation')
parser.add_argument('--dynamic_instance_dir', type=str, default=r'instances\dynamic', help='Directory for exported fixed dynamic instance JSON files')
parser.add_argument('--eval_runs_per_instance', type=int, default=10, help='Number of runs per test instance')
parser.add_argument('--main_sample_runs', type=int, default=-1, help='Number of sample runs for main.py dynamic evaluation. If <=0, use eval_runs_per_instance.')
parser.add_argument('--eval_model_name', type=str, default="llmk1000", help='用於儲存檔案的檔名')


# ============================
# Dynamic Simulation (Event-Driven)
# ============================
parser.add_argument('--event_horizon', type=float, default=160.0, help='事件驅動模式的模擬事件上限')
parser.add_argument('--interarrival_mean', type=float, default=25, help='Poisson interarrival_mean')
parser.add_argument('--arrival_mode', type=str, default='uniform', choices=['exponential', 'uniform'], help='Arrival interval sampling mode for dynamic jobs.')
parser.add_argument('--interarrival_uniform_low', type=float, default=20.0, help='Lower bound of uniform inter-arrival interval when arrival_mode=uniform.')
parser.add_argument('--interarrival_uniform_high', type=float, default=43.0, help='Upper bound of uniform inter-arrival interval when arrival_mode=uniform.')
parser.add_argument('--init_jobs', type=int, default=50, help='初始工單數')
parser.add_argument('--burst_size', type=int, default=1, help='每次生成工單數')
parser.add_argument('--hl_env_scenario', type=str, default='baseline',
                    choices=['baseline', 'custom', 'burst_cluster', 'bottleneck_order', 'mixed3'],
                    help='High-level dynamic environment scenario preset')
parser.add_argument('--hl_burst_size_mode', type=str, default='fixed', choices=['fixed', 'uniform', 'inverse'],
                    help='Dynamic burst size sampler mode used by scenario presets')
parser.add_argument('--hl_burst_size_low', type=int, default=1, help='Minimum burst size when hl_burst_size_mode=uniform')
parser.add_argument('--hl_burst_size_high', type=int, default=5, help='Maximum burst size when hl_burst_size_mode=uniform')
parser.add_argument('--hl_bottleneck_order_prob', type=float, default=0.3, help='Probability that an order is converted into a bottleneck order in bottleneck_order scenario')
parser.add_argument('--hl_bottleneck_order_machine_count', type=int, default=2, help='Number of machines kept for bottleneck orders')
parser.add_argument('--hl_bottleneck_exclude_urgent', type=str2bool, default=False, help='If True, exclude urgent orders from being bottlenecked in bottleneck_order scenario')
parser.add_argument('--hl_bottleneck_group_sampling', type=str, default='random', choices=['random', 'avoid_prev', 'rolling_freq'], help='How bottleneck machine groups are sampled')
parser.add_argument('--hl_bottleneck_avoid_prev_machines', type=str2bool, default=False, help='If True, sample bottleneck machines from machines not used by the previous bottleneck order when possible')
parser.add_argument('--event_seed', type=int, default=42, help='事件驅動到達過程的亂數種子（Exponential 間隔）')
parser.add_argument('--episode_seed_base', type=int, default=12345, help='episode 級別的基種子；每個 episode 以此為基準派生子亂數流')
parser.add_argument('--fast_mode', type=str2bool, default=False, help='是否開啟高速模式（跳過甘特圖與詳細 CSV 生成）')
parser.add_argument('--disable_main_baseline', type=str2bool, default=True, help='Skip main.py cadence baseline simulation when baseline-gap rewards/logs are not needed.')

# ============================
# Curriculum Learning Specifics
# ============================
parser.add_argument('--ll_curriculum_cycle', type=int, default=250, help='Updates per curriculum stage')
parser.add_argument('--ll_schedule_type', type=str, default='same', choices=['s2', 's3', 'same', 'u10_30', 'u10_50', 'u30_50'], help='Type of curriculum schedule to use')
parser.add_argument('--ll_mixed_size_hold_updates', type=int, default=5, help='Updates to keep one sampled mixed-size n_j before resampling for u10_30/u10_50 schedules')
parser.add_argument('--ll_due_setting_hold_updates', type=int, default=10, help='Updates to keep one due setting when ll_due_date_mode=range3_hold.')
parser.add_argument('--ll_due_date_mode', type=str, default='range', choices=['k', 'range', 'range3', 'range3_hold', 'range3_loose', 'range3_mixed', 'range3_tight'], help='Due date generation mode: k (Individual), range (uniform symmetric range), range3 (randomly choose tight/mixed/loose range per instance), range3_hold (training code cycles one due setting for several updates), range3_loose/mixed/tight (fixed range3 sub-mode).')
parser.add_argument('--ll_val_due_date_mode', type=str, default='range', choices=['', 'k', 'range', 'range3', 'range3_loose', 'range3_mixed', 'range3_tight'], help='Validation due date mode override. Empty string means using ll_due_date_mode.')
parser.add_argument('--ll_due_range_scale', type=float, default=0.7, help='Scale factor for range/range3 due-date span a: a = ll_due_range_scale * n_j * mean_pt.')
parser.add_argument('--ll_range3_overdue_prob', type=float, default=0.0, help='Probability to inject severe overdue jobs into each range3 low-level training instance.')
parser.add_argument('--ll_range3_overdue_factor_low', type=float, default=-1.5, help='Lower factor for injected overdue due-date range.')
parser.add_argument('--ll_range3_overdue_factor_high', type=float, default=-0.7, help='Upper factor for injected overdue due-date range.')
parser.add_argument('--ll_range3_overdue_job_frac_low', type=float, default=0.01, help='Lower fraction of jobs injected as overdue when an instance is selected.')
parser.add_argument('--ll_range3_overdue_job_frac_high', type=float, default=0.20, help='Upper fraction of jobs injected as overdue when an instance is selected.')
parser.add_argument('--ll_r60_case_path', type=str, default=r'debug_tools\dynamic_r60_candidate_features\r060_event118_t03691_fixed_ll_case.npz', help='Fixed residual subproblem case used for low-level r60 diagnostic validation.')
parser.add_argument('--hl_due_date_tightness', type=float, default=1.2, help='Tightness base factor for high-level dynamic environment.')
parser.add_argument('--ll_due_date_k_constant', type=float, default=1.2, help='Constant tightness factor k for low-level static training in k-mode.')
parser.add_argument('--ll_due_date_noise', type=float, default=0.0, help='Multiplicative noise level for low-level due dates')
parser.add_argument('--hl_due_date_urgent_prob', type=float, default=0.3, help='Probability of a job being an urgent order in high-level dynamic environment.')
parser.add_argument('--hl_due_date_k_urgent_low', type=float, default=1.5, help='Lower bound of workload multiplier k for urgent due dates.')
parser.add_argument('--hl_due_date_k_urgent_high', type=float, default=2.5, help='Upper bound of workload multiplier k for urgent due dates.')
parser.add_argument('--hl_due_date_k_normal_low', type=float, default=6.5, help='Lower bound of workload multiplier k for normal due dates.')
parser.add_argument('--hl_due_date_k_normal_high', type=float, default=8.5, help='Upper bound of workload multiplier k for normal due dates.')

# ============================
# Unified Scheduling Controller
# ============================
parser.add_argument('--scheduler_type', type=str, default='PPO', 
                    choices=['PPO', 'SPT', 'MWKR', 'FIFO', 'OR-Tools'],
                    help='Unified scheduling method used across all stages (Init, Dynamic, Flush)')
parser.add_argument('--ll_ppo_model_path', type=str, default=r'trained_weights\lower_level\ll_u1030_esttd_odprog.pth', help='PPO 權重檔 .pth 路徑')
parser.add_argument('--ll_ppo_sample', type=str2bool, default=False, help='PPO 推論是否採用抽樣；False=貪婪/取最大機率')
parser.add_argument('--enable_gap_insertion', type=str2bool, default=False, help='Allow low-level scheduler to insert operations into machine idle gaps instead of always appending after machine free time.')


parser.add_argument('--hl_gate_policy', type=str, default='ppo',
                    choices=['ppo', 'cadence', 'slack_threshold', 'random'],
                    help='High-level gate policy: ppo=actor-critic, cadence=fixed event release cadence, slack_threshold=release when buffer min slack is below threshold, random=random probability release')
parser.add_argument('--hl_gate_cadence', type=int, default=5, help='當 gate_policy=cadence 時，每隔幾個到達事件釋放一次緩衝區')
parser.add_argument('--hl_gate_decision_interval', type=int, default=1, help='高階 Agent 的決策步長/間隔 (Arrive 事件數)')
parser.add_argument('--hl_buffer_slack_release_threshold', type=float, default=0.0, help='When gate_policy=slack_threshold, release if current buffer min slack is below this threshold')
parser.add_argument('--hl_eval_action_selection', type=str, default='sample',
                    choices=['sample', 'greedy'],
                    help='Action selection mode for high-level gate during evaluation: sample or greedy')
parser.add_argument('--ll_eval_action_selection', type=str, default='sample',
                    choices=['sample', 'greedy'],
                    help='Action selection mode for low-level scheduler during evaluation: sample or greedy')
parser.add_argument('--ll_rollout_k', type=int, default=10,
                    help='Number of parallel candidate schedules generated for the low-level subproblem (when > 1, forces sampling mode).')
parser.add_argument('--hl_ppo_model_path', type=str, default=r"trained_weights\high_level\hl_ppo_gate_formal_stab_none.pth", help='PPO gate 推論權重路徑（.pth）')
parser.add_argument('--hl_ppo_name', type=str, default='test', help='PPO gate 訓練存檔名稱 (不含 .pth)')


# ============================
# Reward, Penalty & Weights
# ============================
parser.add_argument('--reward_alpha', type=float, default=0.3, help='Deprecated. Unused in current low-level PPO reward path.')
parser.add_argument('--tardiness_alpha', type=float, default=1.0, help='Deprecated. TD weighting is now controlled by ll_td_coef.')
parser.add_argument('--ll_mk_coef', type=float, default=1.0, help='Low-level PPO reward coefficient for MK component.')
parser.add_argument('--ll_td_coef', type=float, default=1.0, help='Low-level PPO reward coefficient for TD component.')
parser.add_argument('--ll_td_mode', type=str, default='mean_pt',
                    choices=['mean_pt', 'workload', 'slack_delta_mean_pt', 'tardiness_delta_mean_pt',
                             'mean_pt_split_ops', 'td_minus_workload_relu',
                             'terminal_split_ops_equal', 'terminal_split_ops_pt',
                             'terminal_split_ops_exp', 'terminal_split_ops_job_ct_delta',
                             'system_slack_delta_mean',
                             'system_neg_slack_delta_mean', 'chosen_neg_slack_delta_mean',
                             'chosen_partial_tardiness_delta', 'chosen_est_tardiness_delta'],
                    help='Low-level TD reward mode: mean_pt=terminal TD/mean_pt, workload=terminal TD/job_workload, slack_delta_mean_pt=negative slack-drop / mean_pt (only at job completion), tardiness_delta_mean_pt=per-op marginal tardiness increase / mean_pt, td_minus_workload_relu=-max(0, tardiness-workload). terminal_split_ops_equal redistributes final job TD equally over the selected job ops; terminal_split_ops_pt redistributes final job TD proportional to selected op processing times; terminal_split_ops_exp redistributes final job TD with exponentially larger weights on later ops; terminal_split_ops_job_ct_delta redistributes final job TD by each selected op completion frontier delta within that job; system_slack_delta_mean replaces TD with active-system total slack delta based on mean remaining work; system_neg_slack_delta_mean uses active-system negative slack proxy delta; chosen_neg_slack_delta_mean uses only the selected job negative slack proxy delta; chosen_partial_tardiness_delta penalizes increases in selected job current tardiness after each scheduled op; chosen_est_tardiness_delta uses the paper-style accuracy-weighted estimated tardiness increase of the selected job.')
parser.add_argument('--ll_td_split_exp_decay', type=float, default=0.8, help='Decay used by terminal_split_ops_exp. Later ops get larger weights: decay^(n-1), ..., decay, 1.')
parser.add_argument('--ll_td_split_max_share', type=float, default=0.5, help='Max reward share per selected op for terminal_split_ops_job_ct_delta. Set <=0 or >=1 to disable.')
parser.add_argument('--ll_system_slack_beta_mode', type=str, default='fixed', choices=['fixed', 'by_n_j'], help='Scaling mode for system slack reward modes: fixed uses ll_system_slack_beta; by_n_j divides the system slack delta by the instance job count.')
parser.add_argument('--ll_system_slack_beta', type=float, default=0.05, help='Fixed beta for system slack reward modes when ll_system_slack_beta_mode=fixed.')
parser.add_argument('--hl_stability_scale', type=float, default=0.0, help='決策穩定性懲罰 (Action 1 的額外扣分)。設為 0 代表純效能模式。')
parser.add_argument('--ll_overdue_progress_coef', type=float, default=0.0, help='Extra low-level reward coefficient for overdue weighted progress: -coef * log1p(max(0,-due)/mean_pt) * delta_job_progress_ct/mean_pt.')
parser.add_argument('--ll_ready_overdue_wait_coef', type=float, default=0.0, help='Extra low-level reward coefficient that penalizes not choosing the most overdue ready job at each decision.')
parser.add_argument('--ll_ready_overdue_wait_threshold', type=float, default=0.0, help='Minimum ready overdue weight required before applying ready-overdue waiting penalty.')
parser.add_argument('--ll_vtarget_norm', type=str2bool, default=False, help='Normalize low-level PPO value targets per-env trajectory before critic loss.')
parser.add_argument('--ll_critic_loss', type=str, default='mse', choices=['mse', 'huber'], help='Critic loss type for low-level PPO.')
parser.add_argument('--ll_reward_norm_by_size', type=str2bool, default=False, help='Normalize low-level PPO step rewards by running mean/std tracked separately for each n_j.')
parser.add_argument('--hl_buffer_penalty_coef', type=float, default=0.0, help='Coefficient for buffer tardiness penalty')
parser.add_argument('--hl_stability_mode', type=str, default='immediate_all', choices=['immediate_all', 'free_threshold'], help='Stability reward mode: immediate_all = current per-release penalty; free_threshold = releases are free until stability_free_releases, then penalize via terminal or redistribution.')
parser.add_argument('--hl_stability_terminal_only', type=str2bool, default=False, help='If true, apply stability penalty only once at episode end using agent-chosen release count.')
parser.add_argument('--hl_stability_free_releases', type=int, default=0, help='Number of agent-chosen releases that are free before stability penalty starts.')
parser.add_argument('--hl_stability_power', type=float, default=0.0, help='Optional power for free-threshold stability penalty. Set >0 to use scale * excess^power; 0 keeps legacy triangular penalty.')
parser.add_argument('--release_penalty_coef', type=float, default=0.1, help='Deprecated legacy parameter. No longer used by gate reward logic.')
parser.add_argument('--hl_release_reward_decay', type=float, default=0.9, help='Decay rate for high-level release reward redistribution.')
parser.add_argument('--hl_td_signal_source', type=str, default='agent_only', choices=['', 'none', 'agent_only', 'baseline_gap_final', 'baseline_gap_release_interval'], help='High-level TD reward source selector. Empty string keeps legacy compatibility.')
parser.add_argument('--hl_td_credit_mode', type=str, default='', choices=['', 'step_only', 'redistribute', 'terminal_only'], help='High-level TD credit assignment mode. Empty string keeps legacy compatibility.')
parser.add_argument('--hl_stability_mode_v2', type=str, default='', choices=['', 'off', 'immediate_all', 'immediate_all_terminal', 'free_threshold_terminal', 'free_threshold_distributed'], help='High-level stability mode selector. Empty string keeps legacy compatibility.')
parser.add_argument('--hl_shaping_reward_coef', type=float, default=0.0, help='Coefficient for shaping reward term')
parser.add_argument('--hl_td_reward_coef', type=float, default=0.1, help='Coefficient for final TD reward term')
parser.add_argument('--hl_mk_reward_coef', type=float, default=0.0, help='Weight for final MK reward term at simulation end')


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
parser.add_argument('--ortools_subproblem_time_limit', type=float, default=30.0, help='OR-Tools time limit per dynamic subproblem/release, in seconds')
parser.add_argument('--ortools_total_solve_time_budget', type=float, default=0.0, help='Total OR-Tools solve-time budget. <=0 means use per-subproblem limit directly')
parser.add_argument('--ortools_time_scale', type=int, default=1, help='Scale factor used to convert continuous times to CP-SAT integer times')

# PPO Gate Training Hyperparameters
parser.add_argument('--hl_ppo_num_layers', type=int, default=3, help='Number of hidden layers in PPO gate')
parser.add_argument('--hl_ppo_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate network')
parser.add_argument('--hl_ppo_separate_trunks', type=str2bool, default=False, help='Use separate actor/critic trunks for PPO gate')
parser.add_argument('--hl_ppo_actor_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate actor trunk')
parser.add_argument('--hl_ppo_actor_num_layers', type=int, default=3, help='Number of hidden layers in PPO gate actor trunk')
parser.add_argument('--hl_ppo_critic_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate critic trunk')
parser.add_argument('--hl_ppo_critic_num_layers', type=int, default=3, help='Number of hidden layers in PPO gate critic trunk')
parser.add_argument('--hl_ppo_value_num_layers', type=int, default=3, help='Number of hidden layers used by PPO gate value head (1 means linear head)')
parser.add_argument('--hl_ppo_value_hidden_dim', type=int, default=512, help='Hidden dimension of PPO gate value head MLP')
parser.add_argument('--hl_use_ll_buffer_embedding', type=str2bool, default=False, help='Append frozen low-level PPO encoder global buffer embedding to high-level gate state.')
parser.add_argument('--hl_ll_buffer_embedding_dim', type=int, default=128, help='Raw LL buffer embedding dimension appended to the high-level state.')
parser.add_argument('--hl_ll_buffer_projection_dim', type=int, default=16, help='Projection dimension for LL buffer embedding inside PPO gate model.')
parser.add_argument('--hl_initial_release_prob', type=float, default=-1.0, help='Optional initial PPO gate release probability prior. Set <=0 to disable.')
parser.add_argument('--hl_ppo_updates', type=int, default=200, help='Number of PPO gate updates')
parser.add_argument('--hl_ppo_num_envs', type=int, default=4, help='Number of PPO gate training environments collected per update')
parser.add_argument('--hl_ppo_use_async_envs', type=str2bool, default=True, help='Use AsyncVectorEnv for PPO gate collection; falls back to SyncVectorEnv on failure')


parser.add_argument('--hl_ppo_train_seed', type=int, default=142, help='Base seed used by PPO gate training instances (separate from main/event seed)')
parser.add_argument('--hl_ppo_lr', type=float, default=1e-4, help='PPO gate learning rate')
parser.add_argument('--hl_ppo_lr_decay', type=str2bool, default=True, help='Whether to linearly decay PPO gate learning rate')
parser.add_argument('--hl_ppo_lr_end', type=float, default=5e-5, help='Final PPO gate learning rate when decay is enabled')
parser.add_argument('--hl_ppo_gamma', type=float, default=0.995, help='Discount factor for PPO gate')
parser.add_argument('--hl_ppo_gae_lambda', type=float, default=0.98, help='GAE lambda for PPO gate')
parser.add_argument('--hl_ppo_clip', type=float, default=0.2, help='PPO clipping ratio')
parser.add_argument('--hl_ppo_entropy_coef', type=float, default=0.01, help='Entropy coefficient for PPO gate')
parser.add_argument('--hl_ppo_entropy_decay', type=str2bool, default=False, help='Whether to linearly decay PPO gate entropy coefficient')
parser.add_argument('--hl_ppo_entropy_end', type=float, default=0.005, help='Final PPO gate entropy coefficient when decay is enabled')
parser.add_argument('--hl_ppo_value_coef', type=float, default=0.5, help='Value loss coefficient for PPO gate')
parser.add_argument('--hl_ppo_update_epochs', type=int, default=5, help='Epochs per PPO gate update')
parser.add_argument('--hl_ppo_minibatch_size', type=int, default=64, help='Minibatch size for PPO gate updates')
parser.add_argument('--hl_ppo_max_grad_norm', type=float, default=1.0, help='Gradient clipping norm for PPO gate')
parser.add_argument('--hl_ppo_same_problem_eval_every', type=int, default=0, help='Same-problem greedy eval interval for PPO gate; 0 disables it')
parser.add_argument('--hl_ppo_validate_every', type=int, default=10, help='Validation interval for PPO gate; 0 disables it')
parser.add_argument('--hl_ppo_instance_episodes', type=int, default=10, help='Episodes to reuse the same event instance before switching (PPO gate)')


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
