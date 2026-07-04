# Project Entry Points

This file lists the main scripts after the cleanup. Keep new one-off debug or analysis scripts under `debug_tools/` or `experiments/` instead of adding more root-level scripts.

## Lower-Level PPO

- Train lower-level PPO:
  `python train_ll_curriculum.py --config yaml_config\<config>.yml`
- Generate static range-due test instances:
  `python generate_range_due_test_instances.py`
- Evaluate static range-due test instances:
  `python evaluate_range_due_test_batch.py --config yaml_config\<config>.yml`
- Plot training logs:
  `python plot_train.py --config yaml_config\<config>.yml`

## Dynamic PPO / HRL

- Export a fixed dynamic instance:
  `python export_dynamic_instance.py --config yaml_config\<config>.yml`
- Replay PPO on a fixed dynamic instance:
  `python run_dynamic_ppo_cadence_export.py --config yaml_config\<config>.yml`
- Train high-level PPO gate:
  `python train_hl_ppo_gate.py --config yaml_config\<config>.yml`
- Run dynamic PPO solver utility:
  `python solve_dynamic_ppo.py --config yaml_config\<config>.yml`

## OR-Tools

- Dynamic OR-Tools cadence scheduling:
  `python ortools_tools\dynamic\run_dynamic_ortools_cadence.py --config yaml_config\<config>.yml`
- Dynamic OR-Tools fixed-instance solving:
  `python ortools_tools\dynamic\solve_dynamic_ortools.py --config yaml_config\<config>.yml`
- Static OR-Tools benchmark:
  `python ortools_tools\static\benchmark_ortools.py`

## Support Folders

- `debug_tools/`: targeted debug scripts and fixed debug cases.
- `experiments/analysis/`: analysis/export scripts that are not part of the main training path.
- `experiments/cadence/`: cadence comparison scripts.
- `legacy_archive/`: old experimental policies retained for reference.
- `ortools_tools/`: OR-Tools related static/dynamic utilities.

## Core Modules

- `ll_fjsp_env.py`: static lower-level scheduling environment.
- `dynamic_job_stream.py`: dynamic job generation and release registration.
- `hl_gate_env.py`: high-level gate environment.
- `hl_env_scenarios.py`: high-level environment scenario generation.
- `hrl_orchestrator.py`: dynamic orchestration and low-level scheduler integration.
- `params.py`: shared configuration parser and defaults.

## Model Weights

- New lower-level PPO checkpoints are saved under:
  `trained_weights/lower_level/<model_name>.pth`
- New high-level PPO checkpoints are saved under:
  `trained_weights/high_level/<model_name>.pth`
- Existing checkpoints have been migrated into `trained_weights/`.
- YAML files that still contain old checkpoint paths are resolved by filename into the new folders.

## Logs, Plots, And Test Results

- Lower-level training logs:
  `train_log/lower_level/`
- High-level training logs:
  `train_log/high_level/`
- Lower-level training plots:
  `train_log_plot/lower_level/`
- High-level training plots:
  `train_log_plot/high_level/`
- Lower-level test/evaluation results:
  `test_results/lower_level/`
- High-level test/evaluation results:
  `test_results/high_level/`

## Instances

- All instance/data folders are grouped under:
  `instances/`
- Dynamic replay/export instances:
  `instances/dynamic/`
- Static uniform instances:
  `instances/or_instances_uniform/`
- Static range-due test instances:
  `instances/or_instances_uniform_test_30_50_due_scaled/`
- Legacy data folders such as `data/` and `TestDataToExcel/` are now under `instances/`.
