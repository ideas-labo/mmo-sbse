import pandas as pd
import os
import re
import warnings
import numpy as np
import sys
from typing import List, Dict, Set, Tuple, Any
import multiprocessing
import csv

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

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/')

from Code.SCT.Feature.multi_feature import main_sct_multi
from Code.SCT.Feature.single_feature import main_sct_single


def check_sct_sampling_data_exists(selected_datasets: List[str], sampling_methods: List[str],
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
                    print(f"Missing SCT forward sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

                if not (os.path.exists(fig1_reverse_path) and os.path.exists(fig2_reverse_path)):
                    print(f"Missing SCT reverse sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

    print("All SCT sampled data (both forward and reverse) exists")
    return True


def check_sct_nsga2_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                random_seeds: range) -> bool:
    base_dir = "../../../Results/RQ1-raw-data/SCT"

    if not os.path.exists(base_dir):
        print(f"SCT NSGA2 directory does not exist: {base_dir}")
        return False

    missing_files = []

    for dataset in selected_datasets:
        for mode in selected_modes:
            for seed in random_seeds:
                # reciprocal applicability rules:
                # - forward (non-reverse): reciprocal not applicable for ['dnn_adiac','dnn_dsr','dnn_sa']
                # - reverse: reciprocal not applicable for base dataset 'x264'
                skip_forward_reciprocal = False
                skip_reverse_reciprocal = False
                if mode == 'reciprocal':
                    if dataset in ['dnn_adiac', 'dnn_dsr', 'dnn_sa']:
                        skip_forward_reciprocal = True
                    if dataset == 'x264':
                        skip_reverse_reciprocal = True

                forward_files = [
                    f"{dataset}-{seed}_{mode}.csv",
                    f"{dataset}-{seed}_{mode}_fa.csv",
                    f"{dataset}-{seed}_{mode}_maximization_fa.csv",
                    f"{dataset}-{seed}_{mode}_g2.csv",
                ]

                reverse_files = [
                    f"{dataset}-{seed}_{mode}_reverse.csv",
                    f"{dataset}-{seed}_{mode}_fa_reverse.csv",
                    f"{dataset}-{seed}_{mode}_maximization_fa_reverse.csv",
                    f"{dataset}-{seed}_{mode}_g2_reverse.csv",
                ]

                # check forward files unless explicitly skipped by reciprocal rule
                found_forward = False
                if skip_forward_reciprocal:
                    found_forward = True
                else:
                    for filename in forward_files:
                        file_path = os.path.join(base_dir, filename)
                        if os.path.exists(file_path):
                            found_forward = True
                            break

                if not found_forward:
                    missing_files.append(f"Forward: {dataset}, {mode}, seed {seed}")
                    print(f"Missing SCT forward NSGA2 data: {dataset}, {mode}, seed {seed}")

                # check reverse files unless explicitly skipped by reciprocal rule
                found_reverse = False
                if skip_reverse_reciprocal:
                    found_reverse = True
                else:
                    for filename in reverse_files:
                        file_path = os.path.join(base_dir, filename)
                        if os.path.exists(file_path):
                            found_reverse = True
                            break

                if not found_reverse:
                    missing_files.append(f"Reverse: {dataset}, {mode}, seed {seed}")
                    print(f"Missing SCT reverse NSGA2 data: {dataset}, {mode}, seed {seed}")

    if missing_files:
        print(f"\nTotal missing SCT NSGA2 files: {len(missing_files)}")
        return False
    else:
        print("All SCT NSGA2 data (both forward and reverse) exists")
        return True


def check_sct_multi_feature_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                        sampling_methods: List[str]) -> bool:
    base_dir = "./Results/Output-draw/"

    for mode in selected_modes:
        csv_file = f"{mode}_statistics.csv"
        csv_path = os.path.join(base_dir, csv_file)

        csv_reverse_file = f"{mode}_statistics_reverse.csv"
        csv_reverse_path = os.path.join(base_dir, csv_reverse_file)

        if not os.path.exists(csv_path):
            print(f"Missing SCT forward multi-objective feature data: {csv_file}")
            return False

        if not os.path.exists(csv_reverse_path):
            print(f"Missing SCT reverse multi-objective feature data: {csv_reverse_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"SCT forward multi-feature data missing required columns {missing_cols}: {csv_file}")
                return False

            df_reverse = pd.read_csv(csv_reverse_path)
            missing_cols_reverse = [col for col in required_cols if col not in df_reverse.columns]
            if missing_cols_reverse:
                print(f"SCT reverse multi-feature data missing required columns {missing_cols_reverse}: {csv_reverse_file}")
                return False

            for dataset in selected_datasets:
                # forward: skip reciprocal for specific datasets
                if mode == 'reciprocal' and dataset in ['dnn_adiac', 'dnn_dsr', 'dnn_sa']:
                    pass
                else:
                    if dataset not in df['Dataset Name'].values:
                        print(f"SCT forward multi-feature data missing dataset: {dataset}")
                        return False

                # reverse: skip reciprocal for x264 in reverse direction
                if mode == 'reciprocal' and dataset == 'x264':
                    pass
                else:
                    if not (dataset in df_reverse['Dataset Name'].values or f"{dataset}_reverse" in df_reverse['Dataset Name'].values):
                        print(f"SCT reverse multi-feature data missing dataset: {dataset}")
                        return False

            for sampling_method in sampling_methods:
                if sampling_method not in df['Sampling Method'].values:
                    print(f"SCT forward multi-feature data missing sampling method: {sampling_method}")
                    return False
                if sampling_method not in df_reverse['Sampling Method'].values:
                    print(f"SCT reverse multi-feature data missing sampling method: {sampling_method}")
                    return False

        except Exception as e:
            print(f"Error checking SCT multi-objective feature data: {e}")
            return False

    print("All SCT multi-objective feature data (both forward and reverse) exists")
    return True


def check_sct_landscape_feature_data_exists(selected_datasets: List[str]) -> bool:
    base_dir = "./Results/real_data/"

    for dataset in selected_datasets:
        csv_file = f"{dataset}.csv"
        csv_path = os.path.join(base_dir, csv_file)

        csv_reverse_file = f"{dataset}_reverse.csv"
        csv_reverse_path = os.path.join(base_dir, csv_reverse_file)

        if not os.path.exists(csv_path):
            print(f"Missing SCT forward landscape feature data: {csv_file}")
            return False

        if not os.path.exists(csv_reverse_path):
            print(f"Missing SCT reverse landscape feature data: {csv_reverse_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"SCT forward landscape feature data file is empty: {csv_file}")
                return False

            df_reverse = pd.read_csv(csv_reverse_path)
            if df_reverse.empty:
                print(f"SCT reverse landscape feature data file is empty: {csv_reverse_file}")
                return False

            required_cols = ['Name', 'Sampling Method', 'Sample Size', 'Random Seed']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"SCT forward landscape data missing required columns {missing_cols}: {csv_file}")
                return False

            missing_cols_reverse = [col for col in required_cols if col not in df_reverse.columns]
            if missing_cols_reverse:
                print(f"SCT reverse landscape data missing required columns {missing_cols_reverse}: {csv_reverse_file}")
                return False

        except Exception as e:
            print(f"Error checking SCT landscape feature data: {e}")
            return False

    print("All SCT landscape feature data (both forward and reverse) exists")
    return True


def extract_info_from_filename_sct(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]

    is_reverse = False
    if "_reverse" in file_name:
        file_name = file_name.replace("_reverse", "")
        is_reverse = True

    parts = file_name.split('-')

    if len(parts) >= 2:
        dataset_name = parts[0]
        remaining_parts = parts[1].split('_')

        seed = remaining_parts[0]
        mode = remaining_parts[1]

        return dataset_name, mode, is_reverse, seed
    else:
        print(f"Unable to parse filename: {file_name}")
        return None, None, None, None


def get_pareto_ratios_sct(csv_path):
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
                print(f"Unable to extract best Pareto ratio from file {csv_path}; please check file format.")
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
                    print(f"Unable to convert p value '{p_str}' to float in file {csv_path}; please check value format.")
                    return None, None, None, None, None

            p_values_mean = sum(p_values) / len(p_values)

            ft_line = lines[-2].strip()
            ft_match = re.search(r"'ft': (-?\d+\.?\d*)", ft_line)
            if ft_match:
                ft = float(ft_match.group(1))
            else:
                print(f"Unable to extract ft value from file {csv_path}; please check file format.")
                ft = None

            budget_line = lines[-5].strip()
            budget_match = re.search(r'budget_used:(\d+)', budget_line)
            if budget_match:
                budget = int(budget_match.group(1))
            else:
                print(f"Unable to extract budget value from file {csv_path}; please check file format.")
                budget = None

            time_line = lines[-4].strip()
            time_match = re.search(r'Running time: (\d+\.?\d*) seconds', time_line)
            if time_match:
                time = float(time_match.group(1))
            else:
                print(f"Unable to extract time value from file {csv_path}; please check file format.")
                time = None

        return best_p, p_values_mean, ft, budget, time
    except Exception as e:
        print(f"Exception while processing file {csv_path}: {e}")
        return None, None, None, None, None


def read_sct_nsga2_data(nsga2_csv_dir, selected_datasets, selected_modes):
    p_data = []
    total_files = 0
    valid_files = 0

    for file in os.listdir(nsga2_csv_dir):
        if file.endswith('.csv'):
            total_files += 1
            file_name = os.path.basename(file)
            dataset_name, mode, is_reverse, seed = extract_info_from_filename_sct(file_name)

            if dataset_name and dataset_name in selected_datasets and mode in selected_modes:
                full_dataset_name = dataset_name
                if is_reverse:
                    full_dataset_name = f"{dataset_name}_reverse"

                csv_path = os.path.join(nsga2_csv_dir, file)
                best_p, p_values_mean, ft, budget, time = get_pareto_ratios_sct(csv_path)

                if best_p is not None and p_values_mean is not None and budget is not None and time is not None:
                    valid_files += 1
                    p_data.append({
                        'Random Seed': int(seed),
                        'Dataset Name': full_dataset_name,
                        'Best_Pareto_Ratio': best_p,
                        'Pareto_Ratios_Mean': p_values_mean,
                        'mode': mode,
                        'ft': ft,
                        'budget': budget,
                        'time': time,
                        'is_reverse': is_reverse
                    })

    print(f"SCT total files: {total_files}")
    print(f"SCT valid files: {valid_files}")

    if p_data:
        p_df = pd.DataFrame(p_data)
        group_cols = ['Dataset Name', 'mode']
        numeric_cols = ['Best_Pareto_Ratio', 'Pareto_Ratios_Mean', 'ft', 'budget', 'time', 'Random Seed']

        median_df = p_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        return median_df
    else:
        return pd.DataFrame()


def read_sct_landscape_data(landscape_csv_dir, selected_datasets, start_seed, end_seed):
    landscape_dfs = []
    for file in os.listdir(landscape_csv_dir):
        if file.endswith('.csv') and '_significance' not in file:
            base_name = file.split('.')[0]

            is_reverse = False
            dataset_name = base_name
            if base_name.endswith('_reverse'):
                dataset_name = base_name[:-8]
                is_reverse = True

            if dataset_name in selected_datasets:
                df = pd.read_csv(os.path.join(landscape_csv_dir, file))

                if 'Name' in df.columns and 'Dataset Name' not in df.columns:
                    df = df.rename(columns={'Name': 'Dataset Name'})

                required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: SCT Landscape data missing columns {missing_cols} in file {file}")
                    continue

                df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]

                sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                    'covering_array', 'halton']
                df = df[df['Sampling Method'].isin(sampling_methods)]

                full_dataset_name = dataset_name
                if is_reverse:
                    full_dataset_name = f"{dataset_name}_reverse"

                df['Dataset Name'] = full_dataset_name
                df['is_reverse'] = is_reverse

                landscape_dfs.append(df)

    if landscape_dfs:
        landscape_df = pd.concat(landscape_dfs, ignore_index=True)

        landscape_df = landscape_df.loc[:, ~((landscape_df.isna() | (landscape_df == 0)).all(axis=0))]

        group_cols = ['Dataset Name', 'Sample Size', 'Sampling Method']

        numeric_cols = [col for col in landscape_df.select_dtypes(include=[np.number]).columns
                        if col not in group_cols + ['Random Seed', 'is_reverse']]

        if numeric_cols:
            median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()
            print(f"SCT Landscape median data shape: {median_df.shape}")
            return median_df
        else:
            print("No numeric columns found in SCT landscape data")
            if all(col in landscape_df.columns for col in group_cols):
                return landscape_df[group_cols].drop_duplicates()
            else:
                print(f"Missing required columns: {group_cols}")
                return pd.DataFrame()
    else:
        print("No SCT landscape data found")
        return pd.DataFrame()


def read_sct_sampling_data(sampling_csv_dir, selected_datasets, start_seed, end_seed, selected_modes, pic_types):
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

                    if 'Mode' in df.columns and 'mode' not in df.columns:
                        df = df.rename(columns={'Mode': 'mode'})
                    if 'Dataset Name' not in df.columns and 'Name' in df.columns:
                        df = df.rename(columns={'Name': 'Dataset Name'})

                    original_datasets = df['Dataset Name'].unique()
                    for original_ds in original_datasets:
                        base_ds = original_ds.replace('_reverse', '')

                        if base_ds in selected_datasets:
                            is_reverse = '_reverse' in file

                            full_dataset_name = base_ds
                            if is_reverse:
                                full_dataset_name = f"{base_ds}_reverse"

                            df.loc[df['Dataset Name'] == original_ds, 'Dataset Name'] = full_dataset_name

                    target_datasets = selected_datasets + [f"{ds}_reverse" for ds in selected_datasets]
                    df = df[df['Dataset Name'].isin(target_datasets)]

                    if 'Figure Number' in df.columns:
                        df['Figure Number'] = df['Figure Number'].map(pic_id_mapping)
                        df = df[df['Figure Number'] == pic_type]
                    else:
                        continue

                    df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]
                    df['mode'] = mode

                    sampling_methods = ['sobol', 'orthogonal', 'halton', 'stratified', 'latin_hypercube', 'monte_carlo',
                                        'covering_array']
                    df = df[df['Sampling Method'].isin(sampling_methods)]

                    for col in df.columns:
                        if df[col].dtype == 'object' and df[col].str.contains('%').any():
                            df[col] = df[col].str.rstrip('%').astype(float) / 100

                    non_merge_cols = [col for col in df.columns if
                                      col not in ['Random Seed', 'Dataset Name', 'mode', 'Sample Size',
                                                  'Sampling Method', 'Figure Number']]

                    rename_dict = {col: f"{col}_{pic_type}" for col in non_merge_cols}
                    df = df.rename(columns=rename_dict)

                    df = df.drop(columns=['Figure Number'], errors='ignore')
                    sampling_dfs.append(df)

        if sampling_dfs:
            sampling_df = pd.concat(sampling_dfs, ignore_index=True)

            group_cols = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
            numeric_cols = [col for col in sampling_df.select_dtypes(include=[np.number]).columns
                            if col not in group_cols + ['Random Seed']]

            if numeric_cols:
                median_df = sampling_df.groupby(group_cols, as_index=False)[numeric_cols].median()
                all_sampling_dfs.append(median_df)
                print(f"SCT {pic_type} sampling median data shape: {median_df.shape}")
            else:
                print(f"No numeric columns found for {pic_type} sampling data")

    if all_sampling_dfs:
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='outer')

        print(f"SCT Combined sampling median data shape: {combined_sampling_df.shape}")
        return combined_sampling_df
    else:
        print("No SCT sampling data found")
        return pd.DataFrame()


def add_sct_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only'):
    ranked_df = p_df.copy()

    ranked_df['ft_rank'] = 1
    ranked_df['time_rank'] = 1
    ranked_df['budget_rank'] = 1

    group_columns = ['Dataset Name', 'Random Seed']

    for group_key, group in ranked_df.groupby(group_columns):
        dataset = group_key[0]
        seed = group_key[1]

        should_maximize = False
        if dataset.endswith('_reverse'):
            base_dataset = dataset.replace('_reverse', '')
            should_maximize = base_dataset in reverse_maximize_datasets
        else:
            should_maximize = dataset in maximize_datasets

        group = group.sort_values(by='time', ascending=True)
        group['time_rank'] = range(1, len(group) + 1)

        group = group.sort_values(by='budget', ascending=True)
        group['budget_rank'] = range(1, len(group) + 1)

        if ranking_mode == 'ft_only':
            if should_maximize:
                group = group.sort_values(by='ft', ascending=False)
            else:
                group = group.sort_values(by='ft', ascending=True)
        elif ranking_mode == 'ft_time':
            if should_maximize:
                group = group.sort_values(by=['ft', 'time'], ascending=[False, True])
            else:
                group = group.sort_values(by=['ft', 'time'], ascending=[True, True])
        elif ranking_mode == 'ft_mode':
            group['mode_priority'] = group['mode'].map(MODE_PRIORITY_ORDER)
            if should_maximize:
                group = group.sort_values(by=['ft', 'mode_priority'], ascending=[False, True])
            else:
                group = group.sort_values(by=['ft', 'mode_priority'], ascending=[True, True])

        group['ft_rank'] = range(1, len(group) + 1)

        ranked_df.loc[group.index, ['ft_rank', 'time_rank', 'budget_rank']] = group[
            ['ft_rank', 'time_rank', 'budget_rank']]

    return ranked_df


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
        print(f"\nColumns meeting NaN criteria will be dropped (total {len(columns_to_drop)} columns):")
        for col in columns_to_drop:
            print(f"- {col} (NaN count: {column_nan_counts[col]})")
        return df.drop(columns=columns_to_drop)
    else:
        print("\nNo columns met the NaN criteria (no columns dropped).")
        return df


def coordinated_pipeline_sct(
        selected_datasets=None,
        selected_modes=None,
        sampling_methods=None,
        random_seeds=None,
        num_samples=900,
        fa_construction=None,
        use_multiprocessing=True,
        max_workers=None,
        use_saved_data=False,
        debug=False,
        start_seed=None,
        end_seed=None,
        pic_types=None,
        data_mode='three_datasets',
        maximize_datasets=None,
        reverse_maximize_datasets=None,
        ranking_mode='ft_mode',
        workflow_base_path='../Datasets/'
):
    if selected_datasets is None:
        selected_datasets = ['dnn_adiac', 'dnn_coffee', 'dnn_dsr', 'dnn_sa',
                             'llvm', 'lrzip', 'mariadb', 'mongodb', 'vp9', 'x264',
                             'storm_rs', 'storm_wc', 'trimesh']
    if selected_modes is None:
        selected_modes = ['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'halton', 'stratified', 'latin_hypercube', 'monte_carlo',
                            'covering_array']
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
    print("Starting SCT Coordinated Data Processing Pipeline")
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

    print("\nStage 1: Check SCT sampled data (both forward and reverse)")
    sampling_data_exists = check_sct_sampling_data_exists(
        selected_datasets, sampling_methods, num_samples, random_seeds
    )

    if not sampling_data_exists:
        print("Starting to generate SCT sampled data...")

        print("Generating forward sampled data...")
        main_sct_multi(
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
            use_saved_data=use_saved_data,
            debug=debug
        )

        print("Generating reverse sampled data...")
        main_sct_multi(
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
            use_saved_data=use_saved_data,
            debug=debug
        )
        print("SCT sampled data generation completed")
    else:
        print("SCT sampled data exists, skipping sampling stage")

    print("\nStage 2: Check SCT multi-objective feature data (both forward and reverse)")
    multi_feature_data_exists = check_sct_multi_feature_data_exists(
        selected_datasets, selected_modes, sampling_methods
    )

    if not multi_feature_data_exists:
        print("Starting SCT multi-objective feature computation...")

        print("Computing forward multi-objective features...")
        main_sct_multi(
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
            use_saved_data=True,
            debug=debug
        )

        print("Computing reverse multi-objective features...")
        main_sct_multi(
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
            use_saved_data=True,
            debug=debug
        )
        print("SCT multi-objective feature computation completed")
    else:
        print("SCT multi-objective feature data exists, skipping computation stage")

    print("\nStage 3: Check SCT landscape feature data (both forward and reverse)")
    landscape_feature_data_exists = check_sct_landscape_feature_data_exists(selected_datasets)

    if not landscape_feature_data_exists:
        print("Starting SCT landscape feature computation...")

        print("Computing forward landscape features...")
        main_sct_single(
            dataset_names=selected_datasets,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=False,
            debug=debug,
            use_saved_data=True
        )

        print("Computing reverse landscape features...")
        main_sct_single(
            dataset_names=selected_datasets,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=True,
            debug=debug,
            use_saved_data=True
        )
        print("SCT landscape feature computation completed")
    else:
        print("SCT landscape feature data exists, skipping computation stage")

    print("\nStage 4: Check SCT NSGA2 data (both forward and reverse)")
    nsga2_data_exists = check_sct_nsga2_data_exists(selected_datasets, selected_modes, random_seeds)

    if not nsga2_data_exists:
        print("Warning: Some SCT NSGA2 data is missing. Ensure NSGA2 has been run and produced results")
        print("Proceeding with available data...")

    print("\nStage 5: SCT Data merging and processing (both forward and reverse)")

    print("Starting SCT data merging...")

    landscape_df = read_sct_landscape_data('./Results/real_data/', selected_datasets, start_seed, end_seed)
    p_df = read_sct_nsga2_data('../../../Results/RQ1-raw-data/SCT/', selected_datasets, selected_modes)
    combined_sampling_df = read_sct_sampling_data('./Results/Output-draw/', selected_datasets,
                                                  start_seed, end_seed, selected_modes, pic_types)

    print(f"SCT landscape_df shape: {landscape_df.shape}")
    print(f"SCT combined_sampling_df shape: {combined_sampling_df.shape}")
    print(f"SCT p_df shape: {p_df.shape}")

    if landscape_df.empty:
        print("Warning: SCT Landscape data is empty")
    if combined_sampling_df.empty:
        print("Warning: SCT Sampling data is empty")
    if p_df.empty:
        print("Warning: SCT NSGA2 data is empty")

    if not p_df.empty:
        p_df = add_sct_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode)

    required_cols_landscape = ['Dataset Name', 'Sample Size', 'Sampling Method']
    required_cols_sampling = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

    landscape_missing = [col for col in required_cols_landscape if col not in landscape_df.columns]
    sampling_missing = [col for col in required_cols_sampling if col not in combined_sampling_df.columns]

    if landscape_missing:
        print(f"SCT Landscape data missing required columns: {landscape_missing}")
        return None
    if sampling_missing:
        print(f"SCT Sampling data missing required columns: {sampling_missing}")
        return None

    all_selected_datasets = selected_datasets + [f"{ds}_reverse" for ds in selected_datasets]

    combined_dfs = []

    for dataset_name in all_selected_datasets:
        is_reverse = dataset_name.endswith("_reverse")
        base_dataset = dataset_name.replace("_reverse", "") if is_reverse else dataset_name

        landscape_filtered = landscape_df[landscape_df['Dataset Name'] == dataset_name].copy()

        if landscape_filtered.empty:
            continue

        landscape_combinations = landscape_filtered[['Sampling Method', 'Sample Size']].drop_duplicates()

        for _, combo in landscape_combinations.iterrows():
            sampling_method = combo['Sampling Method']
            sampling_size = combo['Sample Size']

            landscape_specific = landscape_filtered[
                (landscape_filtered['Sampling Method'] == sampling_method) &
                (landscape_filtered['Sample Size'] == sampling_size)
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

                if sampling_filtered.empty:
                    continue

                combined_df = pd.merge(
                    landscape_specific,
                    sampling_filtered,
                    on=['Dataset Name', 'Sample Size', 'Sampling Method'],
                    how='inner'
                )

                p_df_filtered = p_df[
                    (p_df['Dataset Name'] == dataset_name) &
                    (p_df['mode'] == mode)
                    ].copy()

                if not p_df_filtered.empty:
                    combined_df = pd.merge(
                        combined_df,
                        p_df_filtered,
                        on=['Dataset Name', 'mode'],
                        how='left'
                    )

                combined_dfs.append(combined_df)

    if combined_dfs:
        all_combined_df = pd.concat(combined_dfs, ignore_index=True)
        print(f"SCT All combined data shape: {all_combined_df.shape}")

        all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

        columns_to_keep = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

        numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
        columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

        existing_columns = [col for col in columns_to_keep if col in all_combined_df.columns]
        processed_data = all_combined_df[existing_columns].dropna(axis=1, how='all')
        processed_data = processed_data.reset_index(drop=True)

        print(f"SCT Final data shape after column selection: {processed_data.shape}")

        processed_data = filter_columns_by_nan(processed_data)

        output_folder = '../../../Results/Predict-raw-data/ProcessedData'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_filename = 'processed_data_sct.csv'
        output_path = os.path.join(output_folder, output_filename)
        processed_data.to_csv(output_path, index=False)
        print(f"SCT Final processed data (combined forward and reverse) saved to: {output_path}")

        print("\nSCT Data summary:")
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
        print("SCT Data processing pipeline completed")
        print("=" * 60)

        return processed_data
    else:
        print("No valid SCT data combinations generated")
        return None


if __name__ == "__main__":
    processed_data = coordinated_pipeline_sct(
        selected_datasets=['dnn_adiac', 'dnn_coffee', 'dnn_dsr', 'dnn_sa',
                             'llvm', 'lrzip', 'mariadb', 'mongodb', 'vp9', 'x264',
                             'storm_rs', 'storm_wc', 'trimesh'],
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['latin_hypercube', 'sobol', 'orthogonal', 'stratified', 'monte_carlo','covering_array'],
        random_seeds=range(0, 10),
        num_samples=900,
        fa_construction=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=50,
        use_saved_data=False,
        debug=True,
        pic_types=['PMO', 'MMO'],
        workflow_base_path='../Datasets/',
        maximize_datasets=['storm_wc', 'storm_rs', 'storm_sol', 'dnn_dsr', 'dnn_coffee', 'dnn_adiac', 'x264',
                           'trimesh', 'dnn_coffee', 'dnn_dsr']
    )