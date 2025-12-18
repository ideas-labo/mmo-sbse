import pandas as pd
import os
import re
import warnings
import numpy as np
import sys
from typing import List, Dict, Set, Tuple, Any
import multiprocessing
import csv
import concurrent.futures
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
from Code.WSC.Feature.multi_feature import main_wsc_multi
from Code.WSC.Feature.single_feature import main_wsc_single

warnings.filterwarnings("ignore", category=FutureWarning)

MODE_PRIORITY_ORDER = {
    'gaussian': 2,
    'reciprocal': 1,
    'g1': 4,
    'age': 3,
    'novelty': 7,
    'diversity': 6,
    'penalty': 5
}

SAMPLING_METHODS = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
DATASET_NAMES = ["workflow_1", "workflow_2", "workflow_3", "workflow_4", "workflow_5",
                 "workflow_6", "workflow_7", "workflow_8", "workflow_9", "workflow_10"]
RESULT_DIR = './Results/real_data/'
WORKFLOW_DIR = '../Datasets/Original_data/'
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 50


def check_wsc_sampling_data_exists(selected_datasets: List[str], sampling_methods: List[str],
                                   num_samples: int, random_seeds: range) -> bool:
    base_dir = "./Results/Samples_multi/"

    for dataset in selected_datasets:
        for sampling_method in sampling_methods:
            for seed in random_seeds:
                fig1_file = f"sampled_data_{dataset}_g1_{sampling_method}_{num_samples}_{seed}_figure1.csv"
                fig2_file = f"sampled_data_{dataset}_g1_{sampling_method}_{num_samples}_{seed}_figure2.csv"
                fig1_path = os.path.join(base_dir, fig1_file)
                fig2_path = os.path.join(base_dir, fig2_file)

                fig1_reverse_file = f"sampled_data_{dataset}_g1_{sampling_method}_{num_samples}_{seed}_figure1_reverse.csv"
                fig2_reverse_file = f"sampled_data_{dataset}_g1_{sampling_method}_{num_samples}_{seed}_figure2_reverse.csv"
                fig1_reverse_path = os.path.join(base_dir, fig1_reverse_file)
                fig2_reverse_path = os.path.join(base_dir, fig2_reverse_file)

                if not (os.path.exists(fig1_path) and os.path.exists(fig2_path)):
                    print(f"Missing WSC forward sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

                if not (os.path.exists(fig1_reverse_path) and os.path.exists(fig2_reverse_path)):
                    print(f"Missing WSC reverse sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

    print("All WSC sampled data (both forward and reverse) exists")
    return True


def check_wsc_nsga2_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                random_seeds: range) -> bool:
    base_dir = "../../../Results/RQ1-raw-data/WSC"

    if not os.path.exists(base_dir):
        print(f"WSC NSGA2 directory does not exist: {base_dir}")
        return False

    missing_files = []

    for dataset in selected_datasets:
        for mode in selected_modes:
            for seed in random_seeds:
                forward_files = [
                    f"{dataset}_{seed}_{mode}.csv",
                    f"{dataset}_{seed}_{mode}_fa.csv",
                    f"{dataset}_{seed}_{mode}_maximization_fa.csv",
                    f"{dataset}_{seed}_{mode}_g2.csv",
                ]

                reverse_files = [
                    f"{dataset}_{seed}_{mode}_reverse.csv",
                    f"{dataset}_{seed}_{mode}_fa_reverse.csv",
                    f"{dataset}_{seed}_{mode}_maximization_fa_reverse.csv",
                    f"{dataset}_{seed}_{mode}_g2_reverse.csv",
                ]

                found_forward = False
                for filename in forward_files:
                    file_path = os.path.join(base_dir, filename)
                    if os.path.exists(file_path):
                        found_forward = True
                        break

                if not found_forward:
                    missing_files.append(f"Forward: {dataset}, {mode}, seed {seed}")
                    print(f"Missing WSC forward NSGA2 data: {dataset}, {mode}, seed {seed}")

                found_reverse = False
                for filename in reverse_files:
                    file_path = os.path.join(base_dir, filename)
                    if os.path.exists(file_path):
                        found_reverse = True
                        break

                if not found_reverse:
                    missing_files.append(f"Reverse: {dataset}, {mode}, seed {seed}")
                    print(f"Missing WSC reverse NSGA2 data: {dataset}, {mode}, seed {seed}")

    if missing_files:
        print(f"\nTotal missing WSC NSGA2 files: {len(missing_files)}")
        return False
    else:
        print("All WSC NSGA2 data (both forward and reverse) exists")
        return True


def check_wsc_multi_feature_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                        sampling_methods: List[str]) -> bool:
    base_dir = "./Results/Output-draw/"

    for mode in selected_modes:
        csv_file = f"{mode}_statistics.csv"
        csv_path = os.path.join(base_dir, csv_file)

        csv_reverse_file = f"{mode}_statistics_reverse.csv"
        csv_reverse_path = os.path.join(base_dir, csv_reverse_file)

        if not os.path.exists(csv_path):
            print(f"Missing WSC forward multi-objective feature data: {csv_file}")
            return False

        if not os.path.exists(csv_reverse_path):
            print(f"Missing WSC reverse multi-objective feature data: {csv_reverse_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"WSC forward multi-feature data missing required columns {missing_cols}: {csv_file}")
                return False

            df_reverse = pd.read_csv(csv_reverse_path)
            missing_cols_reverse = [col for col in required_cols if col not in df_reverse.columns]
            if missing_cols_reverse:
                print(f"WSC reverse multi-feature data missing required columns {missing_cols_reverse}: {csv_reverse_file}")
                return False

            for dataset in selected_datasets:
                if dataset not in df['Dataset Name'].values:
                    print(f"WSC forward multi-feature data missing dataset: {dataset}")
                    return False
                if dataset not in df_reverse['Dataset Name'].values:
                    print(f"WSC reverse multi-feature data missing dataset: {dataset}")
                    return False

            for sampling_method in sampling_methods:
                if sampling_method not in df['Sampling Method'].values:
                    print(f"WSC forward multi-feature data missing sampling method: {sampling_method}")
                    return False
                if sampling_method not in df_reverse['Sampling Method'].values:
                    print(f"WSC reverse multi-feature data missing sampling method: {sampling_method}")
                    return False

        except Exception as e:
            print(f"Error checking WSC multi-objective feature data: {e}")
            return False

    print("All WSC multi-objective feature data (both forward and reverse) exists")
    return True


def check_wsc_landscape_feature_data_exists(selected_datasets: List[str]) -> bool:
    base_dir = "./Results/real_data/"

    for dataset in selected_datasets:
        csv_file = f"{dataset}.csv"
        csv_path = os.path.join(base_dir, csv_file)

        csv_reverse_file = f"{dataset}_reverse.csv"
        csv_reverse_path = os.path.join(base_dir, csv_reverse_file)

        if not os.path.exists(csv_path):
            print(f"Missing WSC forward landscape feature data: {csv_file}")
            return False

        if not os.path.exists(csv_reverse_path):
            print(f"Missing WSC reverse landscape feature data: {csv_reverse_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"WSC forward landscape feature data file is empty: {csv_file}")
                return False

            df_reverse = pd.read_csv(csv_reverse_path)
            if df_reverse.empty:
                print(f"WSC reverse landscape feature data file is empty: {csv_reverse_file}")
                return False

            required_cols = ['Name', 'Sampling Method', 'Sample Size', 'Random Seed']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"WSC forward landscape data missing required columns {missing_cols}: {csv_file}")
                return False

            missing_cols_reverse = [col for col in required_cols if col not in df_reverse.columns]
            if missing_cols_reverse:
                print(f"WSC reverse landscape data missing required columns {missing_cols_reverse}: {csv_reverse_file}")
                return False

        except Exception as e:
            print(f"Error checking WSC landscape feature data: {e}")
            return False

    print("All WSC landscape feature data (both forward and reverse) exists")
    return True


def extract_info_from_filename(file_name):
    is_reverse = file_name.endswith("_reverse.csv")
    if is_reverse:
        file_name = file_name[:-13]
    elif file_name.endswith(".csv"):
        file_name = file_name[:-4]

    parts = file_name.split('_')
    mode = parts[3]
    dataset_name = parts[0] + '_' + parts[1]
    seed = int(parts[2])

    return dataset_name, seed, mode, is_reverse


def get_pareto_ratios(csv_path, mode_type='original'):
    try:
        with open(csv_path, 'r') as file:
            lines = file.readlines()
            is_reverse = csv_path.endswith("_reverse.csv")
            if mode_type == 'original':
                budget_line = lines[-5].strip()
                budget_match = re.search(r'budget_used:(\d+)', budget_line)
                budget = int(budget_match.group(1)) if budget_match else None

                time_line = lines[-4].strip()
                time_match = re.search(r'Running time: (\d+\.?\d*) seconds', time_line)
                time = float(time_match.group(1)) if time_match else None

                best_solution_line = lines[-2].strip()
                if is_reverse:
                    ft_match = re.search(r"throughput=(-?\d+\.?\d*)", best_solution_line)
                else:
                    ft_match = re.search(r"latency=(-?\d+\.?\d*)", best_solution_line)
                ft = float(ft_match.group(1)) if ft_match else None

                best_solution_text = next((line for line in lines if "Best Solution" in line), None)
                if not best_solution_text:
                    print(f"No 'Best Solution' line found in file {csv_path}")
                    return None, None, ft, budget, time

                p_match = re.search(r'p: (\d+\.\d+)', best_solution_text)
                best_p = float(p_match.group(1)) if p_match else None

                p_values_text = next((line for line in lines if "p values until best solution" in line), None)
                if not p_values_text:
                    print(f"No 'p values until best solution' line found in file {csv_path}")
                    return best_p, None, ft, budget, time

                p_values_str_list = p_values_text.split(": ")[1].strip().split(",")
                p_values = []
                for p_str in p_values_str_list:
                    p_str = p_str.strip('"')
                    try:
                        p_values.append(float(p_str))
                    except ValueError:
                        print(f"Cannot convert p value '{p_str}' to float in file {csv_path}")
                        return best_p, None, ft, budget, time

                p_values_mean = sum(p_values) / len(p_values) if p_values else None

            elif mode_type == 'adaptive':
                budget_line = lines[-4].strip()
                budget_match = re.search(r'budget_used:(\d+)', budget_line)
                budget = int(budget_match.group(1)) if budget_match else None

                time_line = lines[-3].strip()
                time_match = re.search(r'Running time: (\d+\.?\d*) seconds', time_line)
                time = float(time_match.group(1)) if time_match else None

                best_solution_line = lines[-1].strip()
                ft_match = re.search(r"'ft': (-?\d+\.?\d*)", best_solution_line)
                ft = float(ft_match.group(1)) if ft_match else None

                p_values = []
                for line in lines:
                    if "Generation" in line and "p-value" in line:
                        p_match = re.search(r'p-value: (\d+\.\d+)', line)
                        if p_match:
                            p_values.append(float(p_match.group(1)))

                if not p_values:
                    print(f"No p-values found in file {csv_path}")
                    return None, None, ft, budget, time

                best_p = p_values[-1] if p_values else None
                p_values_mean = sum(p_values) / len(p_values) if p_values else None

            return best_p, p_values_mean, ft, budget, time

    except Exception as e:
        print(f"Exception processing file {csv_path}: {e}")
        return None, None, None, None, None


def read_landscape_data(landscape_csv_dir, selected_datasets, start_seed, end_seed):
    landscape_dfs = []

    for file in os.listdir(landscape_csv_dir):
        if file.endswith('.csv') and '_significance' not in file:
            base_name = file.split('.')[0]
            is_reverse = False
            if base_name.endswith('_reverse'):
                base_name = base_name[:-8]
                is_reverse = True

            if any(base_name.startswith(ds) for ds in selected_datasets):
                df = pd.read_csv(os.path.join(landscape_csv_dir, file))
                df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]

                dataset_name = base_name + ('_reverse' if is_reverse else '')
                df['Dataset Name'] = dataset_name
                df['is_reverse'] = is_reverse

                landscape_dfs.append(df)

    if not landscape_dfs:
        return pd.DataFrame()

    landscape_df = pd.concat(landscape_dfs, ignore_index=True)

    landscape_df = landscape_df.loc[:, ~((landscape_df.isna() | (landscape_df == 0)).all(axis=0))]

    group_cols = ['Dataset Name', 'is_reverse', 'Sampling Method']

    numeric_cols = [col for col in landscape_df.select_dtypes(include=[np.number]).columns
                    if col not in group_cols]

    median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()

    print(f"Landscape median data shape: {median_df.shape}")
    return median_df


def read_nsga2_data(nsga2_csv_dir, selected_datasets, start_seed, end_seed, selected_modes, mode_type='original'):
    p_data = []
    total_files = 0
    valid_files = 0

    for file in os.listdir(nsga2_csv_dir):
        if file.endswith('.csv'):
            total_files += 1
            file_name = os.path.basename(file)
            dataset_name, seed, mode, is_reverse = extract_info_from_filename(file_name)

            if dataset_name in selected_datasets:
                dataset_name = dataset_name + ('_reverse' if is_reverse else '')
                if start_seed <= seed <= end_seed and mode in selected_modes:
                    csv_path = os.path.join(nsga2_csv_dir, file)
                    best_p, p_values_mean, ft, budget, time = get_pareto_ratios(csv_path, mode_type)

                    if best_p is not None and p_values_mean is not None and budget is not None and time is not None:
                        valid_files += 1
                        p_data.append({
                            'Random Seed': seed,
                            'Dataset Name': dataset_name,
                            'Best_Pareto_Ratio': best_p,
                            'Pareto_Ratios_Mean': p_values_mean,
                            'mode': mode,
                            'ft': ft,
                            'budget': budget,
                            'time': time,
                            'is_reverse': is_reverse
                        })

    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_files}")

    if not p_data:
        return pd.DataFrame()

    p_df = pd.DataFrame(p_data)

    group_cols = ['Dataset Name', 'mode', 'is_reverse']
    numeric_cols = ['Best_Pareto_Ratio', 'Pareto_Ratios_Mean', 'ft', 'budget', 'time', 'Random Seed']
    numeric_cols = [col for col in numeric_cols if col in p_df.columns]

    median_df = p_df.groupby(group_cols)[numeric_cols].median().reset_index()
    return median_df


def read_sampling_data(sampling_csv_dir, selected_datasets, start_seed, end_seed, selected_modes, pic_types):
    pic_id_mapping = {1: 'PMO', 2: 'MMO'}

    all_sampling_dfs = []
    for pic_type in pic_types:
        sampling_dfs = []
        for file in os.listdir(sampling_csv_dir):
            if file.endswith('.csv'):
                parts = file.split('_')
                mode = parts[0]
                is_reverse = '_reverse' in file

                if mode in selected_modes:
                    df = pd.read_csv(os.path.join(sampling_csv_dir, file))
                    df['Dataset Name'] = df['Dataset Name'].str.replace('_reverse', '')
                    df = df[df['Dataset Name'].isin(selected_datasets)]
                    df['Dataset Name'] = df['Dataset Name'] + ('_reverse' if is_reverse else '')

                    df['Figure Number'] = df['Figure Number'].map(pic_id_mapping)
                    df = df[df['Figure Number'] == pic_type]
                    df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]
                    df['mode'] = mode
                    df['is_reverse'] = is_reverse

                    sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                        'covering_array']
                    df = df[df['Sampling Method'].isin(sampling_methods)]

                    for col in df.columns:
                        if df[col].dtype == 'object' and df[col].str.contains('%').any():
                            df[col] = df[col].str.rstrip('%').astype(float) / 100

                    non_merge_cols = [col for col in df.columns if
                                      col not in ['Random Seed', 'Dataset Name', 'mode',
                                                  'Sample Size', 'Sampling Method',
                                                  'Figure Number', 'is_reverse']]
                    df = df.rename(columns={col: f"{col}_{pic_type}" for col in non_merge_cols})

                    sampling_dfs.append(df)

        if not sampling_dfs:
            continue

        sampling_df = pd.concat(sampling_dfs, ignore_index=True)

        group_cols = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method', 'is_reverse']
        numeric_cols = sampling_df.select_dtypes(include=[np.number]).columns.tolist()

        group_cols = [col for col in group_cols if col in sampling_df.columns]
        numeric_cols = [col for col in numeric_cols if col not in group_cols]

        median_df = sampling_df.groupby(group_cols)[numeric_cols].median().reset_index()
        all_sampling_dfs.append(median_df)

        print(f"{pic_type} sampling median data shape: {median_df.shape}")

    if not all_sampling_dfs:
        return pd.DataFrame()

    merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method', 'is_reverse', 'Random Seed']
    combined_sampling_df = all_sampling_dfs[0]
    for df in all_sampling_dfs[1:]:
        combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='inner')

    print(f"Combined sampling median data shape: {combined_sampling_df.shape}")
    return combined_sampling_df


def add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only'):
    ranked_df = p_df.copy()

    for (dataset, seed, is_reverse), group in ranked_df.groupby(['Dataset Name', 'Random Seed', 'is_reverse']):
        base_dataset = dataset.replace("_reverse", "") if is_reverse else dataset
        if is_reverse:
            should_maximize = base_dataset in reverse_maximize_datasets
        else:
            should_maximize = base_dataset in maximize_datasets

        if ranking_mode == 'ft_only':
            if should_maximize:
                ranked_df.loc[group.index, 'ft_rank'] = group['ft'].rank(ascending=False, method='min')
            else:
                ranked_df.loc[group.index, 'ft_rank'] = group['ft'].rank(ascending=True, method='min')

        elif ranking_mode == 'ft_time':
            if should_maximize:
                ranked_df.loc[group.index, 'ft_rank'] = group.sort_values(
                    by=['ft', 'time'], ascending=[False, True]
                ).groupby('ft', sort=False).ngroup() + 1
            else:
                ranked_df.loc[group.index, 'ft_rank'] = group.sort_values(
                    by=['ft', 'time'], ascending=[True, True]
                ).groupby('ft', sort=False).ngroup() + 1

        elif ranking_mode == 'ft_mode':
            group_with_prio = group.copy()
            group_with_prio['mode_priority'] = group_with_prio['mode'].map(MODE_PRIORITY_ORDER)

            if should_maximize:
                sorted_group = group_with_prio.sort_values(
                    by=['ft', 'mode_priority'], ascending=[False, True]
                )
            else:
                sorted_group = group_with_prio.sort_values(
                    by=['ft', 'mode_priority'], ascending=[True, True]
                )

            sorted_group['ft_rank'] = range(1, len(sorted_group) + 1)

            ranked_df.loc[sorted_group.index, 'ft_rank'] = sorted_group['ft_rank']

        ranked_df.loc[group.index, 'time_rank'] = group['time'].rank(ascending=True, method='min')

        ranked_df.loc[group.index, 'budget_rank'] = group['budget'].rank(ascending=True, method='min')

        if ranking_mode == 'ft_only':
            if should_maximize:
                optimal_ft = group['ft'].max()
                optimal_rows = group[group['ft'] == optimal_ft]
            else:
                optimal_ft = group['ft'].min()
                optimal_rows = group[group['ft'] == optimal_ft]

            optimal_best_p = optimal_rows['Best_Pareto_Ratio'].mean()
            optimal_p_mean = optimal_rows['Pareto_Ratios_Mean'].mean()

        elif ranking_mode == 'ft_time':
            if should_maximize:
                sorted_group = group.sort_values(by=['ft', 'time'], ascending=[False, True])
                optimal_ft = sorted_group['ft'].max()
                optimal_rows = sorted_group[sorted_group['ft'] == optimal_ft].head(1)
            else:
                sorted_group = group.sort_values(by=['ft', 'time'], ascending=[True, True])
                optimal_ft = sorted_group['ft'].min()
                optimal_rows = sorted_group[sorted_group['ft'] == optimal_ft].head(1)

            optimal_best_p = optimal_rows['Best_Pareto_Ratio'].mean()
            optimal_p_mean = optimal_rows['Pareto_Ratios_Mean'].mean()

        elif ranking_mode == 'ft_mode':
            group_with_prio = group.copy()
            group_with_prio['mode_priority'] = group_with_prio['mode'].map(MODE_PRIORITY_ORDER)

            if should_maximize:
                sorted_group = group_with_prio.sort_values(by=['ft', 'mode_priority'], ascending=[False, True])
                optimal_ft = sorted_group['ft'].max()
                optimal_rows = sorted_group[sorted_group['ft'] == optimal_ft].head(1)
            else:
                sorted_group = group_with_prio.sort_values(by=['ft', 'mode_priority'], ascending=[True, True])
                optimal_ft = sorted_group['ft'].min()
                optimal_rows = sorted_group[sorted_group['ft'] == optimal_ft].head(1)

            optimal_best_p = optimal_rows['Best_Pareto_Ratio'].mean()
            optimal_p_mean = optimal_rows['Pareto_Ratios_Mean'].mean()

        ranked_df.loc[group.index, 'Optimal_Best_Pareto_Ratio'] = optimal_best_p
        ranked_df.loc[group.index, 'Optimal_Pareto_Ratios_Mean'] = optimal_p_mean

        ranked_df.loc[group.index, 'Percent_Diff_Best_P'] = abs(group['Best_Pareto_Ratio'] - optimal_best_p)
        ranked_df.loc[group.index, 'Percent_Diff_P_Mean'] = abs(group['Pareto_Ratios_Mean'] - optimal_p_mean)

    return ranked_df


def process_datasets(landscape_df, p_df, combined_sampling_df, selected_datasets, selected_modes,
                           maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only'):
    p_df = add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode)

    if 'Sampling Method' in landscape_df.columns and 'Sampling Method' in combined_sampling_df.columns and 'Sample Size' in landscape_df.columns and 'Sample Size' in combined_sampling_df.columns:
        sampling_methods = pd.concat(
            [landscape_df['Sampling Method'], combined_sampling_df['Sampling Method']]).unique()
        sampling_sizes = combined_sampling_df['Sample Size'].unique()

        combined_dfs = []
        all_selected_datasets = [ds + '_reverse' for ds in selected_datasets] + selected_datasets
        for dataset_name in all_selected_datasets:
            is_reverse = dataset_name.endswith("_reverse")
            base_dataset = dataset_name.replace("_reverse", "")

            for sampling_method in sampling_methods:
                for sampling_size in sampling_sizes:
                    landscape_filtered = landscape_df[(landscape_df['Sampling Method'] == sampling_method) & (
                            landscape_df['Sample Size'] == sampling_size) & (
                                                              landscape_df['Dataset Name'] == dataset_name)].copy()

                    for mode in selected_modes:
                        if mode == 'reciprocal':
                            if not is_reverse and base_dataset in ['dnn_adiac', 'dnn_dsr', 'dnn_sa']:
                                continue
                            if is_reverse and base_dataset == 'x264':
                                continue

                        sampling_filtered = combined_sampling_df[
                            (combined_sampling_df['Sampling Method'] == sampling_method) & (
                                    combined_sampling_df['Sample Size'] == sampling_size) & (
                                    combined_sampling_df['Dataset Name'] == dataset_name) & (
                                    combined_sampling_df['mode'] == mode)].copy()

                        column_mapping = {
                            'Random Seed': 'Random Seed',
                            'Sample Size': 'Sample Size'
                        }
                        for old_col, new_col in column_mapping.items():
                            if old_col in landscape_filtered.columns and new_col in sampling_filtered.columns:
                                landscape_filtered.rename(columns={old_col: new_col}, inplace=True)

                        landscape_filtered = landscape_filtered.sort_values(
                            by=['Dataset Name', 'Random Seed', 'Sample Size'])
                        sampling_filtered = sampling_filtered.sort_values(
                            by=['Dataset Name', 'mode', 'Random Seed', 'Sample Size'])

                        landscape_filtered = landscape_filtered.reset_index(drop=True)
                        sampling_filtered = sampling_filtered.reset_index(drop=True)

                        combined_df = pd.merge(
                            landscape_filtered,
                            sampling_filtered,
                            on=['Dataset Name', 'Random Seed', 'Sample Size', 'Sampling Method'],
                            how='inner'
                        )

                        p_df_filtered = p_df[(p_df['Dataset Name'] == dataset_name) & (p_df['mode'] == mode)].copy()
                        p_df_filtered = p_df_filtered.sort_values(by=['Random Seed'])
                        p_df_filtered = p_df_filtered.reset_index(drop=True)

                        p_columns_to_keep = [
                            'Random Seed', 'Dataset Name', 'mode',
                            'Best_Pareto_Ratio', 'Pareto_Ratios_Mean',
                            'ft', 'budget', 'time',
                            'ft_rank', 'time_rank', 'budget_rank',
                            'Optimal_Best_Pareto_Ratio', 'Optimal_Pareto_Ratios_Mean',
                            'Percent_Diff_Best_P', 'Percent_Diff_P_Mean'
                        ]
                        p_df_filtered = p_df_filtered[p_columns_to_keep]

                        combined_df = pd.merge(
                            combined_df,
                            p_df_filtered,
                            on=['Dataset Name', 'Random Seed', 'mode'],
                            how='left'
                        )

                        combined_dfs.append(combined_df)

        all_combined_df = pd.concat(combined_dfs, ignore_index=True)
        print(f"All combined data shape: {all_combined_df.shape}")

        all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

        columns_to_keep = [
            'Random Seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method',
            'Best_Pareto_Ratio', 'Pareto_Ratios_Mean',
            'ft', 'budget', 'time',
            'ft_rank', 'time_rank', 'budget_rank',
            'Optimal_Best_Pareto_Ratio', 'Optimal_Pareto_Ratios_Mean',
            'Percent_Diff_Best_P', 'Percent_Diff_P_Mean'
        ]

        numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
        columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

        X_numeric = all_combined_df[columns_to_keep].dropna(axis=1, how='all')
        X_numeric = X_numeric.reset_index(drop=True)
        print(f"X_numeric data shape: {X_numeric.shape}")

        X_numeric = X_numeric[
            X_numeric['Random Seed'].isin(p_df['Random Seed']) &
            X_numeric['mode'].isin(p_df['mode']) &
            X_numeric['Dataset Name'].isin(p_df['Dataset Name'])
            ].reset_index(drop=True)

        print(f"Final data shape: {X_numeric.shape}")
        return X_numeric
    print("Sampling Method and Sample Size columns do not match, cannot continue processing.")
    return None


def filter_columns_by_nan(df):
    column_nan_counts = df.isna().sum()
    columns_to_drop = []

    for col in df.columns:
        current_col_nan = column_nan_counts[col]

        if current_col_nan > 10:
            nan_row_indices = df[df[col].isna()].index

            row_nan_counts = df.loc[nan_row_indices].isna().sum(axis=1)

            if (row_nan_counts <= 10).all():
                columns_to_drop.append(col)

    if columns_to_drop:
        print(f"\nColumns meeting NaN criteria will be dropped (count={len(columns_to_drop)}):")
        for col in columns_to_drop:
            print(f"- {col} (NaN count: {column_nan_counts[col]})")
        return df.drop(columns=columns_to_drop)
    else:
        print("\nNo columns met the NaN criteria")
        return df


def process_data(start_seed, end_seed, selected_modes, selected_datasets, pic_types, data_mode,
                 maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only'):
    landscape_df = read_landscape_data('./Results/real_data/', selected_datasets, start_seed,
                                       end_seed)
    p_df = read_nsga2_data('../../../Results/RQ1-raw-data/WSC', selected_datasets,
                           start_seed, end_seed, selected_modes)

    if data_mode == 'three_datasets':
        combined_sampling_df = read_sampling_data('./Results/Output-draw/', selected_datasets,
                                                  start_seed, end_seed, selected_modes, pic_types)
        final_df = process_datasets(landscape_df, p_df, combined_sampling_df, selected_datasets, selected_modes,
                                          maximize_datasets, reverse_maximize_datasets, ranking_mode)
    else:
        print("Invalid data extraction mode, choose 'three_datasets'.")
        final_df = None

    if final_df is not None and not final_df.empty:
        print("Starting NaN-based column filtering...")
        final_df = filter_columns_by_nan(final_df)

    return final_df


def coordinated_pipeline_wsc(
        selected_datasets=None,
        selected_modes=None,
        sampling_methods=None,
        random_seeds=None,
        num_samples=1000,
        fa_construction=None,
        use_multiprocessing=True,
        max_workers=None,
        debug=False,
        start_seed=None,
        end_seed=None,
        pic_types=None,
        data_mode='three_datasets',
        maximize_datasets=None,
        reverse_maximize_datasets=None,
        ranking_mode='ft_mode',
        workflow_base_path='../Datasets/Original_data/'
):
    if selected_datasets is None:
        selected_datasets = ["workflow_1", "workflow_2", "workflow_3", "workflow_4", "workflow_5",
                             "workflow_6", "workflow_7", "workflow_8", "workflow_9", "workflow_10"]
    if selected_modes is None:
        selected_modes = ['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
    if random_seeds is None:
        random_seeds = range(0, 10)
    if fa_construction is None:
        fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if start_seed is None:
        start_seed = min(random_seeds)
    if end_seed is None:
        end_seed = max(random_seeds)
    if pic_types is None:
        pic_types = ['PMO', 'MMO']
    if maximize_datasets is None:
        maximize_datasets = []
    if reverse_maximize_datasets is None:
        reverse_maximize_datasets = selected_datasets

    print("=" * 60)
    print("Starting WSC Coordinated Data Processing Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Processing both forward and reverse data")
    print(f"  Number of datasets: {len(selected_datasets)}")
    print("=" * 60)

    print("\nStage 1: Check WSC sampled data (both forward and reverse)")
    sampling_data_exists = check_wsc_sampling_data_exists(
        selected_datasets, sampling_methods, num_samples, random_seeds
    )

    if not sampling_data_exists:
        print("Starting to generate WSC sampled data...")

        print("Generating forward sampled data...")
        main_wsc_multi(
            dataset_names=selected_datasets,
            fa_construction=['g1'],
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=False,
            first_sample=True,
            file_base_path=workflow_base_path,
            debug=debug
        )

        print("Generating reverse sampled data...")
        main_wsc_multi(
            dataset_names=selected_datasets,
            fa_construction=['g1'],
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=True,
            first_sample=True,
            file_base_path=workflow_base_path,
            debug=debug
        )
        print("WSC sampled data generation completed")
    else:
        print("WSC sampled data exists, skipping sampling stage")

    print("\nStage 2: Check WSC multi-objective feature data (both forward and reverse)")
    multi_feature_data_exists = check_wsc_multi_feature_data_exists(
        selected_datasets, selected_modes, sampling_methods
    )

    if not multi_feature_data_exists:
        print("Starting WSC multi-objective feature computation...")

        print("Computing forward multi-objective features...")
        main_wsc_multi(
            dataset_names=selected_datasets,
            fa_construction=fa_construction,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=False,
            first_sample=False,
            file_base_path=workflow_base_path,
            debug=debug
        )

        print("Computing reverse multi-objective features...")
        main_wsc_multi(
            dataset_names=selected_datasets,
            fa_construction=fa_construction,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=True,
            first_sample=False,
            file_base_path=workflow_base_path,
            debug=debug
        )
        print("WSC multi-objective feature computation completed")
    else:
        print("WSC multi-objective feature data exists, skipping computation stage")

    print("\nStage 3: Check WSC landscape feature data (both forward and reverse)")
    landscape_feature_data_exists = check_wsc_landscape_feature_data_exists(selected_datasets)

    if not landscape_feature_data_exists:
        print("Starting WSC landscape feature computation...")

        print("Computing forward landscape features...")
        main_wsc_single(
            dataset_names=selected_datasets,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=False,
            debug=debug,
        )

        print("Computing reverse landscape features...")
        main_wsc_single(
            dataset_names=selected_datasets,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=True,
            debug=debug,
        )
        print("WSC landscape feature computation completed")
    else:
        print("WSC landscape feature data exists, skipping computation stage")

    print("\nStage 4: Check WSC NSGA2 data (both forward and reverse)")
    nsga2_data_exists = check_wsc_nsga2_data_exists(selected_datasets, selected_modes, random_seeds)

    if not nsga2_data_exists:
        print("Warning: Some WSC NSGA2 data is missing. Ensure NSGA2 has been run and produced results")
        print("Proceeding with available data...")

    print("\nStage 5: WSC Data merging and processing (both forward and reverse)")

    processed_data = process_data(
        start_seed=start_seed,
        end_seed=end_seed,
        selected_modes=selected_modes,
        selected_datasets=selected_datasets,
        pic_types=pic_types,
        data_mode=data_mode,
        maximize_datasets=maximize_datasets,
        reverse_maximize_datasets=reverse_maximize_datasets,
        ranking_mode=ranking_mode,
    )

    if processed_data is not None:
        output_folder = '../../../Results/Predict-raw-data/ProcessedData'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_filename = 'processed_data_wsc.csv'
        output_path = os.path.join(output_folder, output_filename)
        processed_data.to_csv(output_path, index=False)
        print(f"WSC Final processed data (combined forward and reverse) saved to: {output_path}")

        print("\nWSC Data summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        if 'Dataset Name' in processed_data.columns:
            dataset_counts = processed_data['Dataset Name'].value_counts()
            print(f"\nDataset distribution:")
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count} rows")

        forward_count = len([ds for ds in processed_data['Dataset Name'] if not ds.endswith('_reverse')])
        reverse_count = len([ds for ds in processed_data['Dataset Name'] if ds.endswith('_reverse')])
        print(f"\nForward data rows: {forward_count}")
        print(f"Reverse data rows: {reverse_count}")

        print("\n" + "=" * 60)
        print("WSC Data processing pipeline completed")
        print("=" * 60)

        return processed_data
    else:
        print("No valid WSC data combinations generated")
        return None


if __name__ == "__main__":
    processed_data = coordinated_pipeline_wsc(
        selected_datasets=["workflow_1", "workflow_2", "workflow_3", "workflow_4", "workflow_5",
                             "workflow_6", "workflow_7", "workflow_8", "workflow_9", "workflow_10"],
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'],
        random_seeds=range(0, 10),
        num_samples=1000,
        fa_construction=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=100,
        debug=True,
        pic_types=['PMO', 'MMO'],
        workflow_base_path='../Datasets/Original_data/',
        maximize_datasets=[],
        reverse_maximize_datasets=["workflow_1", "workflow_2"]
    )