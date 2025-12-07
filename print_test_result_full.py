from __future__ import annotations

from params import configs
import os
import re
import numpy as np
import pandas as pd
import time as time
import argparse
from typing import Optional, List, Tuple

null_val = np.nan
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
non_exist_data: List[List[str]] = []
winner = []


def load_result(source: str, model_name: str, data_name: str) -> Tuple[np.ndarray, float, float]:
    """
    讀模型測試結果：
    return: (per-instance makespan array, avg makespan, avg time)
    """
    file_path = f'./test_results/{source}/{data_name}/Result_{model_name}_{data_name}.npy'
    # file_path =r"test_results\BenchData\Hurink_vdata\Result_DANIELG+ppo_static_final_Hurink_vdata.npy"
    if os.path.exists(file_path):
        test_result = np.load(file_path)
        make_span = test_result[:, 0]
        make_span_mean = float(np.mean(make_span)) if len(make_span) else null_val
        time_mean = float(np.mean(test_result[:, 1])) if test_result.shape[1] > 1 else null_val
        return make_span, make_span_mean, time_mean
    else:
        non_exist_data.append([source, f'Result_{model_name}_{data_name}.npy'])
        return np.array([]), null_val, null_val


def load_solution_by_or(source: str, data_name: str) -> Tuple[np.ndarray, float, float, float]:
    """
    讀 OR-Tools 參考解：
    return: (per-instance makespan, avg makespan, avg time, success ratio)
    """
    file_path = f'./or_solution/{source}/solution_{data_name}.npy'
    if os.path.exists(file_path):
        solution = np.load(file_path)
        or_make_span = solution[:, 0]
        or_make_span_mean = float(np.mean(or_make_span)) if len(or_make_span) else null_val
        or_time_mean = float(np.mean(solution[:, 1])) if solution.shape[1] > 1 else null_val
        or_percentage = (
            np.where(solution[:, 1] < configs.max_solve_time)[0].shape[0] / solution[:, 1].shape[0]
            if solution.shape[1] > 1 else null_val
        )
        return or_make_span, or_make_span_mean, or_time_mean, or_percentage
    else:
        non_exist_data.append([source, f'solution_{data_name}.npy'])
        return np.array([]), null_val, null_val, null_val


def load_benchmark_solution(data_name: str) -> Tuple[np.ndarray, float]:
    """
    讀 BenchData 的最佳已知解 (ub)
    """
    file_path = f'./data/BenchData/BenchDataSolution.csv'  # 你的新版路徑
    bench_data = pd.read_csv(file_path)
    make_span = bench_data.loc[bench_data['benchname'] == data_name, 'ub'].values
    make_span_mean = float(np.mean(make_span)) if len(make_span) else null_val
    return make_span.astype(float), make_span_mean


def _pad_to(vec: Optional[np.ndarray], L: int) -> np.ndarray:
    """將向量補到長度 L（不足以 NaN 填充）"""
    if vec is None or len(vec) == 0:
        return np.full(L, np.nan, dtype=float)
    if len(vec) == L:
        return vec.astype(float)
    out = np.full(L, np.nan, dtype=float)
    out[:min(L, len(vec))] = vec[:min(L, len(vec))].astype(float)
    return out


def print_test_results_to_excel(source: str, data_list: List[str], model_list: List[str], file_name: str = f"test_{str_time}"):
    """
    產出一份 Excel，包含：
      - makespan (平均)
      - gap（百分比；BenchData 以 BenchDataSolution.csv 的 ub 為基準；其他以 OR 為基準）
      - time (平均)
      - or_percentage
      - winner
      - nonExistData
      - 逐 instance 明細：每個 data_name 一張 sheet，含 Best、Ortools（若有）、各模型的 Makespan / Gap(%) / Time
    """
    out_dir = f'./TestDataToExcel/{source}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = np.append('Ortools', model_list)
    columns = data_list

    make_span_form: List[List[float]] = []
    gap_form: List[List[float]] = []
    time_form: List[List[float]] = []

    optimal_make_span_list: List[np.ndarray] = []  # 作為 gap 的 per-instance 基準；BenchData=bench ub；其他=OR

    or_make_span_list: List[float] = []
    or_time_list: List[float] = []
    or_gap_list: List[float] = []
    or_percentage_list: List[float] = []

    # === 基準與 OR 匯總 ===
    for data_name in data_list:
        or_ms, or_ms_mean, or_time_mean, or_pct = load_solution_by_or(source, data_name)
        or_make_span_list.append(or_ms_mean)
        or_time_list.append(or_time_mean)
        or_percentage_list.append(or_pct)

        if source == 'BenchData':
            bench_ms, bench_ms_mean = load_benchmark_solution(data_name)
            optimal_make_span_list.append(bench_ms)  # per-instance
            if len(or_ms) == 0 or len(bench_ms) == 0:
                gap = null_val
            else:
                gap = float(np.mean((or_ms / bench_ms - 1.0) * 100.0))
            or_gap_list.append(gap)
        else:
            # 非 BenchData 以 OR per-instance 作為基準
            optimal_make_span_list.append(or_ms)

    # === 各模型的平均表 ===
    for model_name in model_list:
        make_span_row = []
        gap_row = []
        time_row = []
        for i, data_name in enumerate(data_list):
            ms, ms_mean, t_mean = load_result(source, model_name, data_name)
            base = optimal_make_span_list[i]
            if len(base) == 0 or len(ms) == 0:
                gap = null_val
            else:
                gap = float(np.mean((ms / base - 1.0) * 100.0))
            make_span_row.append(ms_mean)
            gap_row.append(gap)
            time_row.append(t_mean)

        make_span_form.append(make_span_row)
        gap_form.append(gap_row)
        time_form.append(time_row)

    # === 建立 writer ===
    writer = pd.ExcelWriter(f'{out_dir}/{file_name}.xlsx')

    # 插入 Ortools 匯總列
    make_span_form.insert(0, or_make_span_list)
    time_form.insert(0, or_time_list)

    # console message
    print('=' * 25 + 'DataToExcelMessage' + '=' * 25)
    print(f'source:{source}')
    print(f'model_list:{np.array(model_list)}')
    print(f'data_list:{np.array(data_list)}')

    # Sheet1: makespan（平均）
    make_span_file = pd.DataFrame(make_span_form, columns=columns, index=idx)

    # Sheet2: gap（平均）— 先建數值版，再轉字串百分比
    if source == 'BenchData':
        gap_form.insert(0, or_gap_list)
        gap_num = pd.DataFrame(gap_form, columns=columns, index=idx)
    else:
        gap_num = pd.DataFrame(gap_form, columns=columns, index=idx[1:])

    # Sheet3: time（平均）
    time_file = pd.DataFrame(time_form, columns=columns, index=idx)

    # Sheet4: optimal percentage（OR 在時限內成功比例）
    or_percentage_file = pd.DataFrame([or_percentage_list], columns=columns, index=['percentage'])

    # Sheet5: winner（不含 Ortools 列）
    winner_pd = pd.DataFrame(make_span_file.iloc[1:].idxmin(), columns=['winner'])

    # Sheet6: non-existing data summary
    non_exist_data_pd = pd.DataFrame(non_exist_data, columns=['source', 'filename'])

    # 排序（若只一個資料集且要求排序）
    if len(data_list) == 1 and getattr(configs, "sort_flag", False):
        make_span_file = make_span_file.sort_values(by=data_list[0])
        time_file = time_file.sort_values(by=data_list[0])
        gap_num = gap_num.sort_values(by=data_list[0])

    # 將 gap 轉成字串百分比（避免 dtype 警告）
    gap_str = gap_num.copy()
    gap_str = gap_str.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}%")

    # 寫入總表
    make_span_file.to_excel(writer, sheet_name='makespan', index=True)
    gap_str.to_excel(writer, sheet_name='gap', index=True)
    time_file.to_excel(writer, sheet_name='time', index=True)
    or_percentage_file.to_excel(writer, sheet_name='or_percentage', index=True)
    winner_pd.to_excel(writer, sheet_name='winner', index=True)
    non_exist_data_pd.to_excel(writer, sheet_name='nonExistData', index=True)

    # === 逐 instance 明細（每個 data_name 一張 sheet）===
    for data_name in data_list:
        # 基準（Best）
        best = None
        best_from = None
        if source == 'BenchData':
            bench_ms, _ = load_benchmark_solution(data_name)  # per-instance
            if len(bench_ms) > 0:
                best = bench_ms.astype(float)
                best_from = 'Bench'
        if best is None:
            or_ms, _, _, _ = load_solution_by_or(source, data_name)
            if len(or_ms) > 0:
                best = or_ms.astype(float)
                best_from = 'Ortools'

        # OR per-instance（僅展示；per-instance time 不在原檔，預設 NaN）
        or_ms, _, _, _ = load_solution_by_or(source, data_name)
        or_t = np.full_like(or_ms, np.nan, dtype=float) if len(or_ms) > 0 else np.array([])

        # 各模型 per-instance
        per_model = {}  # {model_name: (ms_vec, t_vec)}
        max_len = 0
        for model_name in model_list:
            ms, _, t_mean = load_result(source, model_name, data_name)
            if len(ms) > 0:
                t_vec = np.full_like(ms, np.nan, dtype=float)  # 若未儲存 per-instance time，暫以 NaN
                per_model[model_name] = (ms.astype(float), t_vec)
                max_len = max(max_len, len(ms))
        max_len = max(max_len, len(or_ms), len(best) if best is not None else 0)

        if max_len == 0:
            continue  # 無資料跳過

        # 對齊長度
        best_vec = _pad_to(best, max_len)
        or_ms_vec = _pad_to(or_ms, max_len)
        or_t_vec = _pad_to(or_t, max_len)

        # 數值表
        arrays = []
        cols = []

        # Best
        arrays.append(best_vec.reshape(-1, 1))
        cols.append(("Best", best_from if best_from is not None else "N/A"))

        # Ortools（若有）
        if len(or_ms) > 0:
            arrays.append(or_ms_vec.reshape(-1, 1))
            cols.append(("Ortools", "Makespan"))
            if best is not None and len(best_vec) == max_len:
                or_gap = (or_ms_vec / best_vec - 1.0) * 100.0
                arrays.append(or_gap.reshape(-1, 1))
                cols.append(("Ortools", "Gap(%)"))
            arrays.append(or_t_vec.reshape(-1, 1))
            cols.append(("Ortools", "Time"))

        # 模型
        for model_name, (ms_vec, t_vec) in per_model.items():
            ms_pad = _pad_to(ms_vec, max_len)
            t_pad = _pad_to(t_vec, max_len)
            arrays.append(ms_pad.reshape(-1, 1))
            cols.append((model_name, "Makespan"))
            if best is not None and len(best_vec) == max_len:
                gap = (ms_pad / best_vec - 1.0) * 100.0
                arrays.append(gap.reshape(-1, 1))
                cols.append((model_name, "Gap(%)"))
            arrays.append(t_pad.reshape(-1, 1))
            cols.append((model_name, "Time"))

        table = np.concatenate(arrays, axis=1)
        col_index = pd.MultiIndex.from_tuples(cols, names=["Model", "Metric"])
        inst_index = pd.Index(range(max_len), name="instance_id")
        detail_df_num = pd.DataFrame(table, index=inst_index, columns=col_index)

        # 將 Gap(%) 欄轉成字串百分比
        detail_df_str = detail_df_num.copy()
        for c in detail_df_str.columns:
            if c[1] == "Gap(%)":
                detail_df_str[c] = detail_df_str[c].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}%")

        sheet_name = f"instances_{data_name[:28]}"  # Excel 名稱上限 31
        detail_df_str.to_excel(writer, sheet_name=sheet_name, index=True)

    writer.close()

    # 主控台
    print(pd.DataFrame(make_span_form, columns=columns, index=idx))
    print('\n Successfully print data to excel! \n')
    print('=' * 50)
    print('winner:\n')
    print(pd.DataFrame(make_span_file.iloc[1:].idxmin(), columns=['winner']))
    print('\nnon_exist_data:\n\n')
    print(pd.DataFrame(non_exist_data, columns=['source', 'filename']))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_source", help="e.g., SD2 or BenchData")
    ap.add_argument("--test_data", nargs="+", help="one or more data names, e.g., Hurink_vdata")
    return ap.parse_args()


def main():
    """
    print the testing results (Obj./Gap/Time) to files (excel)
    data: {configs.test_data}
    source: {configs.data_source}
    """
    args = parse_args()
    if args.data_source:
        configs.data_source = args.data_source
    if args.test_data:
        configs.test_data = args.test_data

    # 從檔名 robust 解析 model_name：抓 'Result_' 與 '_{data_name}.npy' 之間的字串
    model_list: List[str] = []
    for data_name in configs.test_data:
        dir_path = f'./test_results/{configs.data_source}/{data_name}'
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            # 範例：Result_DANIELG+10x5+mix_Hurink_vdata.npy
            m = re.match(rf'^Result_(.+)_({re.escape(data_name)})\.npy$', fname)
            if m:
                model_list.append(m.group(1))
    model_list = sorted(list(set(model_list)))

    file_name = f"test_{str_time}_{configs.data_source}_{configs.test_data[0]}"
    print_test_results_to_excel(configs.data_source, configs.test_data, model_list, file_name)


if __name__ == '__main__':
    main()
