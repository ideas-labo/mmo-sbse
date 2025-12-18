from collections import defaultdict
import pandas as pd
import os
import re
import warnings
import numpy as np
import multiprocessing
import concurrent.futures
import csv
import sys
from typing import List, Dict, Set, Tuple, Any

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

DATASET_NAMES = ["china-train", "desharnais-train", "finnish-train", "maxwell-train", "miyazaki-train"]
SAMPLING_METHODS = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array', 'halton',
                    'random_walk']
RESULT_DIR = "./Results/"
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 30

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

from Code.SEE.Feature.multi_feature import main_see_multi
from Code.SEE.Feature.single_feature import main_see_single


def check_sampling_data_exists(selected_datasets: List[str], sampling_methods: List[str],
                               num_samples: int, random_seeds: range) -> bool:
    base_dir = "./Results/Samples_multi/"

    for dataset in selected_datasets:
        for sampling_method in sampling_methods:
            for seed in random_seeds:
                fig1_file = f"sampled_data_{dataset}_g1_{sampling_method}_{num_samples}_{seed}_figure1.csv"
                fig2_file = f"sampled_data_{dataset}_g1_{sampling_method}_{num_samples}_{seed}_figure2.csv"

                fig1_path = os.path.join(base_dir, fig1_file)
                fig2_path = os.path.join(base_dir, fig2_file)

                if not (os.path.exists(fig1_path) and os.path.exists(fig2_path)):
                    print(f"Missing sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

    print("All sampled data exists")
    return True


def check_nsga2_data_exists(selected_datasets: List[str], selected_modes: List[str],
                            random_seeds: range) -> bool:
    base_dir = "../../../Results/RQ1-raw-data/SEE"

    if not os.path.exists(base_dir):
        print(f"NSGA2 directory does not exist: {base_dir}")
        return False

    missing_files = []

    for dataset in selected_datasets:
        dataset=dataset.split('-')[0] +'_' + dataset.split('_')[1]
        for mode in selected_modes:
            for seed in random_seeds:
                possible_files = [
                    f"{dataset}_{seed}_{mode}.csv",
                    f"{dataset}_{seed}_{mode}_fa.csv",
                    f"{dataset}_{seed}_{mode}_maximization_fa.csv",
                    f"{dataset}_{seed}_{mode}_g2.csv",
                    f"{dataset}_reverse_{seed}_{mode}.csv",
                    f"{dataset}_reverse_{seed}_{mode}_fa.csv",
                    f"{dataset}_reverse_{seed}_{mode}_maximization_fa.csv",
                    f"{dataset}_reverse_{seed}_{mode}_g2.csv"
                ]

                found = False
                for filename in possible_files:
                    file_path = os.path.join(base_dir, filename)
                    if os.path.exists(file_path):
                        found = True
                        break

                if not found:
                    missing_files.append(f"{dataset}, {mode}, seed {seed}")
                    print(f"Missing NSGA2 data: {dataset}, {mode}, seed {seed}")

    if missing_files:
        print(f"\nTotal missing NSGA2 files: {len(missing_files)}")
        return False
    else:
        print("All NSGA2 data exists")
        return True


def check_multi_feature_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                    sampling_methods: List[str]) -> bool:
    base_dir = "./Results/Output-draw/"

    for mode in selected_modes:
        csv_file = f"{mode}_statistics.csv"
        csv_path = os.path.join(base_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"Missing multi-objective feature data: {csv_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Multi-feature data missing required columns {missing_cols}: {csv_file}")
                return False

            for dataset in selected_datasets:
                if dataset not in df['Dataset Name'].values:
                    print(f"Multi-feature data missing dataset: {dataset}")
                    return False

            for sampling_method in sampling_methods:
                if sampling_method not in df['Sampling Method'].values:
                    print(f"Multi-feature data missing sampling method: {sampling_method}")
                    return False

        except Exception as e:
            print(f"Error checking multi-objective feature data: {e}")
            return False

    print("All multi-objective feature data exists")
    return True


def check_landscape_feature_data_exists(selected_datasets: List[str]) -> bool:
    base_dir = "./Results/real_data/"

    for dataset in selected_datasets:
        csv_file = f"{dataset}.csv"
        csv_path = os.path.join(base_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"Missing landscape feature data: {csv_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"Landscape feature data file is empty: {csv_file}")
                return False

            required_cols = ['Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Landscape data missing required columns {missing_cols}: {csv_file}")
                return False

        except Exception as e:
            print(f"Error checking landscape feature data: {e}")
            return False

    print("All landscape feature data exists")
    return True


def extract_info_from_filename(file_name, process_reverse=False):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]

    if process_reverse and file_name.endswith("_reverse"):
        file_name = file_name[:-8]
        is_reverse = True
    else:
        is_reverse = False

    parts = file_name.split('_')
    mode = parts[3]
    dataset_name = parts[0] + "-train_" + parts[1]
    seed = parts[2]
    return dataset_name, mode, is_reverse, seed


def get_pareto_ratios(csv_path):
    try:
        with open(csv_path, 'r') as file:
            lines = file.readlines()

            best_solution_text = next((line for line in lines if "Best Solution" in line), None)
            if best_solution_text is None:
                print(f"No 'Best Solution' line found in file {csv_path}")
                return None, None, None, None, None
            p_match = re.search(r'p: (\d+\.\d+)', best_solution_text)
            if p_match:
                best_p = float(p_match.group(1))
            else:
                print(f"Cannot extract Best Pareto ratio from file {csv_path}; check file format.")
                return None, None, None, None, None

            p_values_text = next((line for line in lines if "p values until best solution" in line), None)
            if p_values_text is None:
                print(f"No 'p values until best solution' line found in file {csv_path}")
                return None, None, None, None, None
            p_values_str_list = p_values_text.split(": ")[1].strip().split(",")

            p_values = []
            for p_str in p_values_str_list:
                p_str = p_str.strip('"')
                try:
                    p = float(p_str)
                    p_values.append(p)
                except ValueError:
                    print(f"Cannot convert p value '{p_str}' to float in file {csv_path}; check value format.")
                    return None, None, None, None, None

            p_values_mean = sum(p_values) / len(p_values)

            ft_line = lines[-2].strip()
            ft_match = re.search(r"'ft': (-?\d+\.?\d*)", ft_line)
            if ft_match:
                ft = float(ft_match.group(1))
            else:
                print(f"Cannot extract ft value from file {csv_path}; check file format.")
                ft = None

            budget_line = lines[-5].strip()
            budget_match = re.search(r'budget_used:(\d+)', budget_line)
            if budget_match:
                budget = int(budget_match.group(1))
            else:
                print(f"Cannot extract budget value from file {csv_path}; check file format.")
                budget = None

            time_line = lines[-4].strip()
            time_match = re.search(r'Running time: (\d+\.?\d*) seconds', time_line)
            if time_match:
                time = float(time_match.group(1))
            else:
                print(f"Cannot extract time value from file {csv_path}; check file format.")
                time = None

        return best_p, p_values_mean, ft, budget, time
    except Exception as e:
        print(f"Exception processing file {csv_path}: {e}")
        return None, None, None, None, None


def read_landscape_data(landscape_csv_dir, selected_datasets, start_seed, end_seed, process_reverse=False):
    landscape_dfs = []
    for file in os.listdir(landscape_csv_dir):
        if file.endswith('.csv') and '_significance' not in file:
            base_name = file.split('.')[0]
            is_reverse = False
            if process_reverse and base_name.endswith('_reverse'):
                base_name = base_name[:-8]
                is_reverse = True

            if base_name in selected_datasets:
                df = pd.read_csv(os.path.join(landscape_csv_dir, file))
                df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]
                sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                    'covering_array', 'halton', 'random_walk']
                df = df[df['Sampling Method'].isin(sampling_methods)]
                if process_reverse:
                    df['Dataset Name'] = base_name + ('_reverse' if is_reverse else '')
                    df['is_reverse'] = is_reverse
                else:
                    df['Dataset Name'] = base_name
                landscape_dfs.append(df)

    if landscape_dfs:
        landscape_df = pd.concat(landscape_dfs, ignore_index=True)

        cols_to_drop = ["main_Ang", "main_Gra", "aux_Ang", "aux_Gra"]
        landscape_df = landscape_df.drop(columns=[col for col in cols_to_drop if col in landscape_df.columns])

        landscape_df = landscape_df.loc[:, ~((landscape_df.isna() | (landscape_df == 0)).all(axis=0))]

        group_cols = ['Dataset Name', 'Sample Size', 'Sampling Method']
        if process_reverse:
            group_cols.append('is_reverse')
        numeric_cols = [col for col in landscape_df.select_dtypes(include=[np.number]).columns
                        if col not in group_cols]

        median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        print(f"Landscape median data shape: {median_df.shape}")
        print(f"Remaining columns after dropping target cols: {median_df.columns.tolist()}")
        return median_df
    else:
        print("No landscape data found")
        return pd.DataFrame()


def read_nsga2_data(nsga2_csv_dir, selected_datasets, selected_modes, process_reverse=False):
    all_data = []
    total_files = 0
    valid_files = 0

    for file in os.listdir(nsga2_csv_dir):
        if file.endswith('.csv'):
            total_files += 1
            dataset_name, mode, is_reverse, seed = extract_info_from_filename(file, process_reverse)

            if dataset_name in selected_datasets and mode in selected_modes:
                csv_path = os.path.join(nsga2_csv_dir, file)
                best_p, p_values_mean, ft, budget, time = get_pareto_ratios(csv_path)

                if best_p is not None and p_values_mean is not None:
                    valid_files += 1
                    all_data.append({
                        'Dataset Name': dataset_name,
                        'mode': mode,
                        'is_reverse': is_reverse,
                        'Random Seed': int(seed),
                        'Best_Pareto_Ratio': best_p,
                        'Pareto_Ratios_Mean': p_values_mean,
                        'ft': ft,
                        'budget': budget,
                        'time': time
                    })

    print(f"\nProcessing completed: total files {total_files}, valid files {valid_files}")

    if all_data:
        nsga2_df = pd.DataFrame(all_data)

        group_cols = ['Dataset Name', 'mode', 'is_reverse']
        numeric_cols = [col for col in nsga2_df.select_dtypes(include=[np.number]).columns
                        if col not in group_cols]

        median_df = nsga2_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        print(f"NSGA2 median data shape: {median_df.shape}")
        return median_df
    else:
        return pd.DataFrame(columns=['Dataset Name', 'mode', 'is_reverse', 'Random Seed',
                                     'Best_Pareto_Ratio', 'Pareto_Ratios_Mean', 'ft', 'budget', 'time'])


def read_sampling_data(sampling_csv_dir, selected_datasets, start_seed, end_seed, selected_modes, pic_types,
                       process_reverse=False):
    pic_id_mapping = {1: 'PMO', 2: 'MMO'}

    all_sampling_dfs = []
    for pic_type in pic_types:
        sampling_dfs = []
        for file in os.listdir(sampling_csv_dir):
            if file.endswith('.csv'):
                parts = file.split('_')
                mode = parts[0]

                if mode in selected_modes:
                    df = pd.read_csv(os.path.join(sampling_csv_dir, file))
                    if process_reverse:
                        df['Dataset Name'] = df['Dataset Name'].str.replace('_reverse', '')
                        df = df[df['Dataset Name'].isin(selected_datasets)]
                        df['Dataset Name'] = df['Dataset Name'] + ('_reverse' if '_reverse' in file else '')
                    else:
                        df = df[df['Dataset Name'].isin(selected_datasets)]

                    df['Figure Number'] = df['Figure Number'].map(pic_id_mapping)
                    df = df[df['Figure Number'] == pic_type]
                    df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]
                    df['mode'] = mode
                    if process_reverse:
                        df['is_reverse'] = '_reverse' in file

                    sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                        'covering_array', 'halton', 'random_walk']
                    df = df[df['Sampling Method'].isin(sampling_methods)]

                    for col in df.columns:
                        if df[col].dtype == 'object' and df[col].str.contains('%').any():
                            df[col] = df[col].str.rstrip('%').astype(float) / 100
                    df = df.rename(columns={col: f"{col.replace('Figure Number', pic_type)}" for col in df.columns if
                                            col not in ['Random Seed', 'Dataset Name', 'mode', 'Sample Size',
                                                        'Sampling Method', 'Figure Number']})
                    df = df.drop(columns=['Figure Number'])
                    sampling_dfs.append(df)

        if sampling_dfs:
            sampling_df = pd.concat(sampling_dfs, ignore_index=True)

            group_cols = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
            if process_reverse:
                group_cols.append('is_reverse')
            numeric_cols = [col for col in sampling_df.select_dtypes(include=[np.number]).columns
                            if col not in group_cols]

            median_df = sampling_df.groupby(group_cols, as_index=False)[numeric_cols].median()
            non_merge_cols = [col for col in median_df.columns if
                              col not in ['Random Seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method'] + (
                                  ['is_reverse'] if process_reverse else [])]
            median_df = median_df.rename(columns={col: f"{col}_{pic_type}" for col in non_merge_cols})
            all_sampling_dfs.append(median_df)
            print(f"{pic_type} sampling median data shape: {median_df.shape}")

    if all_sampling_dfs:
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method', 'Random Seed'] + (
            ['is_reverse'] if process_reverse else [])
        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='inner')
        print(f"Combined sampling median data shape: {combined_sampling_df.shape}")
        return combined_sampling_df
    else:
        print("No sampling data found")
        return pd.DataFrame()


def add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only'):
    ranked_df = p_df.copy()

    for (dataset, seed, is_reverse), group in ranked_df.groupby(['Dataset Name', 'Random Seed', 'is_reverse']):
        if is_reverse:
            should_maximize = dataset.replace("_reverse", "") in reverse_maximize_datasets
        else:
            should_maximize = dataset in maximize_datasets

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

    return ranked_df


def process_three_datasets(landscape_df, p_df, combined_sampling_df, selected_datasets, selected_modes,
                           maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only', process_reverse=False):
    p_df = add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode)

    if 'Sampling Method' in landscape_df.columns and 'Sampling Method' in combined_sampling_df.columns and 'Sample Size' in landscape_df.columns and 'Sample Size' in combined_sampling_df.columns:
        sampling_methods = pd.concat(
            [landscape_df['Sampling Method'], combined_sampling_df['Sampling Method']]).unique()
        sampling_sizes = combined_sampling_df['Sample Size'].unique()

        combined_dfs = []
        all_selected_datasets = [ds + '_reverse' for ds in
                                 selected_datasets] + selected_datasets if process_reverse else selected_datasets

        for dataset_name in all_selected_datasets:
            is_reverse = dataset_name.endswith("_reverse") if process_reverse else False
            base_dataset = dataset_name.replace("_reverse", "") if process_reverse else dataset_name

            for sampling_method in sampling_methods:
                for sampling_size in sampling_sizes:
                    landscape_filtered = landscape_df[
                        (landscape_df['Sampling Method'] == sampling_method) &
                        (landscape_df['Sample Size'] == sampling_size) &
                        (landscape_df['Dataset Name'] == dataset_name)
                        ].copy()

                    for mode in selected_modes:
                        sampling_filtered = combined_sampling_df[
                            (combined_sampling_df['Sampling Method'] == sampling_method) &
                            (combined_sampling_df['Sample Size'] == sampling_size) &
                            (combined_sampling_df['Dataset Name'] == dataset_name) &
                            (combined_sampling_df['mode'] == mode)
                            ].copy()

                        column_mapping = {'Random Seed': 'Random Seed', 'Sample Size': 'Sample Size'}
                        for old_col, new_col in column_mapping.items():
                            if old_col in landscape_filtered.columns and new_col in sampling_filtered.columns:
                                landscape_filtered.rename(columns={old_col: new_col}, inplace=True)

                        landscape_filtered = landscape_filtered.sort_values(
                            by=['Dataset Name', 'Random Seed', 'Sample Size']).reset_index(drop=True)
                        sampling_filtered = sampling_filtered.sort_values(
                            by=['Dataset Name', 'mode', 'Random Seed', 'Sample Size']).reset_index(drop=True)

                        combined_df = pd.concat(
                            [landscape_filtered.drop(columns=['Random Seed'], errors='ignore'), sampling_filtered],
                            axis=1
                        )

                        p_df_filtered = p_df[
                            (p_df['Dataset Name'] == dataset_name) &
                            (p_df['mode'] == mode)
                            ].sort_values(by=['Random Seed']).reset_index(drop=True)

                        final_combined = pd.concat([combined_df, p_df_filtered], axis=1)
                        combined_dfs.append(final_combined)

        if combined_dfs:
            all_combined_df = pd.concat(combined_dfs, ignore_index=True)
            all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

            columns_to_keep = [
                'Random Seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method',
                'Best_Pareto_Ratio', 'Pareto_Ratios_Mean',
                'ft', 'budget', 'time',
                'ft_rank', 'time_rank', 'budget_rank'
            ]
            if process_reverse:
                columns_to_keep.append('is_reverse')

            numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
            columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

            X_numeric = all_combined_df[columns_to_keep].dropna(axis=1, how='all')
            X_numeric = X_numeric.reset_index(drop=True)

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


def coordinated_pipeline_see(
        selected_datasets=None,
        selected_modes=None,
        sampling_methods=None,
        random_seeds=None,
        num_samples=1000,
        fa_construction=None,
        use_multiprocessing=True,
        max_workers=None,
        reverse=False,
        use_saved_data=False,
        debug=False,
        start_seed=None,
        end_seed=None,
        pic_types=None,
        data_mode='three_datasets',
        maximize_datasets=None,
        reverse_maximize_datasets=None,
        ranking_mode='ft_mode',
        process_reverse=False,
        workflow_base_path='../Datasets/'
):
    if selected_datasets is None:
        selected_datasets=["china-train", "desharnais-train", "finnish-train", "maxwell-train", "miyazaki-train"]
    if selected_modes is None:
        selected_modes = ['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if sampling_methods is None:
        sampling_methods = ['monte_carlo', 'latin_hypercube', 'sobol', 'stratified', 'halton', 'random_walk']
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
        maximize_datasets = selected_datasets
    if reverse_maximize_datasets is None:
        reverse_maximize_datasets = []

    print("=" * 60)
    print("Starting SEE Coordinated Data Processing Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Number of datasets: {len(selected_datasets)}")
    print("=" * 60)
    selected_datasets_fold = []
    folds = ["fold1", "fold2", "fold3"]
    for dataset in selected_datasets:
        for fold in folds:
            selected_datasets_fold.append(dataset + "_" + fold)
    print("\nStage 1: Check sampled data")
    sampling_data_exists = check_sampling_data_exists(selected_datasets_fold, sampling_methods, num_samples, random_seeds)

    if not sampling_data_exists:
        print("Starting to generate sampled data...")
        main_see_multi(
            dataset_names=selected_datasets,
            fa_construction=['g1'],
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            first_sample=True,
            data_base_path=workflow_base_path
        )
        print("Sampled data generation completed")
    else:
        print("Sampled data exists, skipping sampling stage")

    print("\nStage 2: Check multi-objective feature data")
    multi_feature_data_exists = check_multi_feature_data_exists(selected_datasets_fold, selected_modes, sampling_methods)

    if not multi_feature_data_exists:
        print("Starting multi-objective feature computation...")
        main_see_multi(
            dataset_names=selected_datasets,
            fa_construction=fa_construction,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            first_sample=False,
            data_base_path=workflow_base_path
        )
        print("Multi-objective feature computation completed")
    else:
        print("Multi-objective feature data exists, skipping computation stage")

    print("\nStage 3: Check landscape feature data")
    landscape_feature_data_exists = check_landscape_feature_data_exists(selected_datasets_fold)

    if not landscape_feature_data_exists:
        print("Starting landscape feature computation...")
        main_see_single(
            dataset_names=selected_datasets,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            debug=debug,
            use_saved_data=True
        )
        print("Landscape feature computation completed")
    else:
        print("Landscape feature data exists, skipping computation stage")

    print("\nStage 4: Check NSGA2 data")
    nsga2_data_exists = check_nsga2_data_exists(selected_datasets_fold, selected_modes, random_seeds)

    if not nsga2_data_exists:
        print("Warning: Some NSGA2 data is missing. Ensure NSGA2 has been run and produced results")
        print("Proceeding with available data...")

    print("\nStage 5: Data merging and processing")
    print("Starting data merging...")

    landscape_df = read_landscape_data('./Results/real_data/', selected_datasets_fold, start_seed,
                                       end_seed, process_reverse)
    p_df = read_nsga2_data('../../../Results/RQ1-raw-data/SEE/', selected_datasets_fold,
                           selected_modes, process_reverse)
    combined_sampling_df = read_sampling_data('./Results/Output-draw/', selected_datasets_fold,
                                              start_seed, end_seed, selected_modes, pic_types, process_reverse)

    print(f"landscape_df columns: {list(landscape_df.columns)}")
    print(f"combined_sampling_df columns: {list(combined_sampling_df.columns)}")
    print(f"p_df columns: {list(p_df.columns)}")

    if landscape_df.empty:
        print("Warning: Landscape data is empty")
    if combined_sampling_df.empty:
        print("Warning: Sampling data is empty")
    if p_df.empty:
        print("Warning: NSGA2 data is empty")

    if not p_df.empty:
        p_df = add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode)

    required_cols_landscape = ['Dataset Name', 'Sample Size', 'Sampling Method']
    required_cols_sampling = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

    landscape_missing = [col for col in required_cols_landscape if col not in landscape_df.columns]
    sampling_missing = [col for col in required_cols_sampling if col not in combined_sampling_df.columns]

    if landscape_missing:
        print(f"Landscape data missing required columns: {landscape_missing}")
        return None
    if sampling_missing:
        print(f"Sampling data missing required columns: {sampling_missing}")
        return None

    sampling_methods_available = pd.concat(
        [landscape_df['Sampling Method'], combined_sampling_df['Sampling Method']]).unique()
    sampling_sizes = combined_sampling_df['Sample Size'].unique()

    combined_dfs = []
    all_selected_datasets = [ds + '_reverse' for ds in
                             selected_datasets_fold] + selected_datasets_fold if process_reverse else selected_datasets_fold

    for dataset_name in all_selected_datasets:
        is_reverse = dataset_name.endswith("_reverse") if process_reverse else False
        base_dataset = dataset_name.replace("_reverse", "") if process_reverse else dataset_name

        for sampling_method in sampling_methods_available:
            for sampling_size in sampling_sizes:
                landscape_filtered = landscape_df[
                    (landscape_df['Sampling Method'] == sampling_method) &
                    (landscape_df['Sample Size'] == sampling_size) &
                    (landscape_df['Dataset Name'] == dataset_name)
                    ].copy()

                for mode in selected_modes:
                    if mode == 'reciprocal':
                        if not is_reverse and base_dataset in ['dnn_adiac', 'dnn_dsr', 'dnn_sa']:
                            continue
                        if is_reverse and base_dataset == 'x264':
                            continue

                    sampling_filtered = combined_sampling_df[
                        (combined_sampling_df['Sampling Method'] == sampling_method) &
                        (combined_sampling_df['Sample Size'] == sampling_size) &
                        (combined_sampling_df['Dataset Name'] == dataset_name) &
                        (combined_sampling_df['mode'] == mode)
                        ].copy()

                    column_mapping = {'Sample Size': 'Sample Size'}
                    for old_col, new_col in column_mapping.items():
                        if old_col in landscape_filtered.columns and new_col in sampling_filtered.columns:
                            landscape_filtered.rename(columns={old_col: new_col}, inplace=True)

                    sort_cols_landscape = [col for col in ['Dataset Name', 'Sample Size'] if
                                           col in landscape_filtered.columns]
                    sort_cols_sampling = [col for col in ['Dataset Name', 'mode', 'Sample Size'] if
                                          col in sampling_filtered.columns]

                    if sort_cols_landscape:
                        landscape_filtered = landscape_filtered.sort_values(by=sort_cols_landscape).reset_index(
                            drop=True)
                    if sort_cols_sampling:
                        sampling_filtered = sampling_filtered.sort_values(by=sort_cols_sampling).reset_index(drop=True)

                    combined_df = pd.concat([landscape_filtered, sampling_filtered], axis=1)

                    p_df_filtered = p_df[
                        (p_df['Dataset Name'] == dataset_name) &
                        (p_df['mode'] == mode)
                        ].sort_values(by=['Random Seed']).reset_index(drop=True)

                    final_combined = pd.concat([combined_df, p_df_filtered], axis=1)
                    combined_dfs.append(final_combined)

    if combined_dfs:
        all_combined_df = pd.concat(combined_dfs, ignore_index=True)
        all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

        columns_to_keep = [
            'Dataset Name', 'mode', 'Sample Size', 'Sampling Method',
            'Best_Pareto_Ratio', 'Pareto_Ratios_Mean',
            'ft', 'budget', 'time',
            'ft_rank', 'time_rank', 'budget_rank',
            'Optimal_Best_Pareto_Ratio', 'Optimal_Pareto_Ratios_Mean',
            'Percent_Diff_Best_P', 'Percent_Diff_P_Mean'
        ]
        if process_reverse:
            columns_to_keep.append('is_reverse')

        existing_columns = [col for col in columns_to_keep if col in all_combined_df.columns]
        numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
        existing_columns.extend([col for col in numeric_columns if col not in existing_columns])

        processed_data = all_combined_df[existing_columns].dropna(axis=1, how='all')
        processed_data = processed_data.reset_index(drop=True)

        if 'seed' in processed_data.columns:
            processed_data = processed_data.rename(columns={'seed': 'Random Seed'})

        print(f"Final data shape: {processed_data.shape}")

        processed_data = filter_columns_by_nan(processed_data)

        output_folder = '../../../Results/Predict-raw-data/ProcessedData'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = os.path.join(output_folder, 'processed_data_see.csv')
        processed_data.to_csv(output_path, index=False)
        print(f"Final processed data saved to: {output_path}")

        print("\nData summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        print("\n" + "=" * 60)
        print("SEE Data processing pipeline completed")
        print("=" * 60)

        return processed_data
    else:
        print("No valid data combinations generated")
        return None


if __name__ == "__main__":
    processed_data = coordinated_pipeline_see(
        selected_datasets=["china-train", "desharnais-train", "finnish-train", "maxwell-train", "miyazaki-train"],
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=[ 'monte_carlo', 'latin_hypercube', 'sobol', 'stratified', 'halton', 'random_walk'],
        random_seeds=range(0, 10),
        num_samples=1000,
        fa_construction=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=50,
        reverse=False,
        use_saved_data=False,
        debug=True,
        pic_types=['PMO', 'MMO'],
        process_reverse=False,
        workflow_base_path='../Datasets/'
    )