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


def load_external_ranking_info(ranking_csv_path):
    """
    Load ranking information from external ranking result CSV file

    Parameters:
        ranking_csv_path: Path to the ranking result CSV file

    Returns:
        ranking_dict: Dictionary with keys (dataset_name, mode, is_reverse) and ranking info
    """
    if not os.path.exists(ranking_csv_path):
        print(f"[ERROR] External ranking result file does not exist: {ranking_csv_path}")
        print("[ERROR] Please run the ranking analysis code first to generate ranking results")
        return None

    try:
        ranking_df = pd.read_csv(ranking_csv_path)

        # Check required columns
        required_cols = ['Dataset Name', 'mode', 'unique_rank']
        missing_cols = [col for col in required_cols if col not in ranking_df.columns]
        if missing_cols:
            print(f"[ERROR] Ranking result file missing required columns: {missing_cols}")
            print(f"Available columns: {ranking_df.columns.tolist()}")
            return None

        ranking_dict = {}
        for _, row in ranking_df.iterrows():
            dataset_name = row['Dataset Name']
            mode = row['mode']
            unique_rank = row['unique_rank']

            # Handle reverse datasets (check if dataset name contains _reverse)
            is_reverse = '_reverse' in dataset_name

            # Create unique key (dataset, mode, is_reverse)
            key = (dataset_name, mode, is_reverse)
            ranking_dict[key] = {
                'unique_rank': unique_rank,

            }

        print(
            f"[INFO] Loaded ranking information for {len(ranking_dict)} dataset-mode combinations from {ranking_csv_path}")
        return ranking_dict

    except Exception as e:
        print(f"[ERROR] Failed to read ranking result file: {e}")
        return None


def create_ranking_df_from_external_wsc(selected_datasets, selected_modes, ranking_dict, include_reverse=True):
    """
    Create WSC ranking DataFrame from external ranking information

    Parameters:
        selected_datasets: Selected dataset list
        selected_modes: Selected mode list
        ranking_dict: Ranking dictionary loaded from CSV
        include_reverse: Whether to include reverse datasets

    Returns:
        ranking_df: DataFrame containing ranking information
    """
    ranking_data = []

    # Create all datasets including reverse if needed
    all_selected_datasets = selected_datasets
    if include_reverse:
        all_selected_datasets = selected_datasets + [f"{ds}_reverse" for ds in selected_datasets]

    for dataset_name in all_selected_datasets:
        # Determine if this is a reverse dataset
        is_reverse = dataset_name.endswith("_reverse") if include_reverse else False

        # Extract base dataset name (without _reverse suffix for matching)
        base_dataset = dataset_name

        for mode in selected_modes:
            # Create key for lookup (using base dataset name)
            key = (base_dataset, mode, is_reverse)

            # Try to find ranking info
            if key in ranking_dict:
                rank_info = ranking_dict[key]
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': rank_info['unique_rank'],

                })
            else:
                # If not found, use default ranking
                print(
                    f"[WARNING] No ranking info found for dataset '{dataset_name}', mode '{mode}', reverse={is_reverse}, using default rank 1")
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': 1,

                })

    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        print(f"[INFO] Created ranking DataFrame with {len(ranking_df)} rows for WSC")
        return ranking_df
    else:
        print("[WARNING] Failed to create any WSC ranking information")
        return pd.DataFrame()


def process_data_with_external_ranking(start_seed, end_seed, selected_modes, selected_datasets, pic_types, data_mode,
                                       ranking_df, maximize_datasets, reverse_maximize_datasets):
    """
    Process WSC data using external ranking information instead of NSGA2 data

    Parameters:
        start_seed: Starting random seed
        end_seed: Ending random seed
        selected_modes: List of selected modes
        selected_datasets: List of selected datasets
        pic_types: List of picture types
        data_mode: Data extraction mode
        ranking_df: Ranking DataFrame from external ranking file
        maximize_datasets: List of datasets to maximize
        reverse_maximize_datasets: List of reverse datasets to maximize

    Returns:
        final_df: Processed DataFrame with ranking information
    """
    # Read landscape feature data and sampling data (these are still needed)
    landscape_df = read_landscape_data('./Results/real_data/', selected_datasets, start_seed, end_seed)

    if data_mode == 'three_datasets':
        combined_sampling_df = read_sampling_data('./Results/Output-draw/', selected_datasets,
                                                  start_seed, end_seed, selected_modes, pic_types)
    else:
        print("Invalid data extraction mode, choose 'three_datasets'.")
        return None

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

                        # Merge landscape and sampling data
                        if not landscape_filtered.empty and not sampling_filtered.empty:
                            combined_df = pd.merge(
                                landscape_filtered,
                                sampling_filtered,
                                on=['Dataset Name', 'Random Seed', 'Sample Size', 'Sampling Method'],
                                how='inner'
                            )

                            # Get ranking information for this dataset and mode
                            ranking_filtered = ranking_df[(ranking_df['Dataset Name'] == dataset_name) &
                                                          (ranking_df['mode'] == mode)].copy()

                            if not ranking_filtered.empty:
                                # Merge ranking information
                                combined_df = pd.merge(
                                    combined_df,
                                    ranking_filtered,
                                    on=['Dataset Name', 'mode'],
                                    how='left'
                                )

                                combined_dfs.append(combined_df)

        if combined_dfs:
            all_combined_df = pd.concat(combined_dfs, ignore_index=True)
            print(f"All combined data shape: {all_combined_df.shape}")

            all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

            # Define columns to keep (with ranking information)
            columns_to_keep = [
                'Random Seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method',
                'ft_rank'
            ]

            numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
            columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

            X_numeric = all_combined_df[columns_to_keep].dropna(axis=1, how='all')
            X_numeric = X_numeric.reset_index(drop=True)
            print(f"X_numeric data shape: {X_numeric.shape}")

            # Filter by available ranking data
            if 'ft_rank' in X_numeric.columns:
                X_numeric = X_numeric[X_numeric['ft_rank'].notna()].reset_index(drop=True)

            print(f"Final data shape: {X_numeric.shape}")
            return X_numeric

    print("Sampling Method and Sample Size columns do not match, cannot continue processing.")
    return None


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
        workflow_base_path='../Datasets/Original_data/',
        ranking_csv_path=None  # New parameter: ranking result CSV file path
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

    # Check if ranking_csv_path is provided
    if ranking_csv_path is None:
        print("[ERROR] Must provide ranking_csv_path parameter to specify ranking result CSV file path")
        print("[ERROR] Please run ranking analysis code first to generate ranking results, then specify file path")
        return None

    print("=" * 60)
    print("Starting WSC (Workflow Scheduling with Cost) Coordinated Data Processing Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  Processing both forward and reverse data")
    print(f"  Number of datasets: {len(selected_datasets)}")
    print(f"  External ranking file: {ranking_csv_path}")
    print("=" * 60)

    # Load external ranking information
    print(f"\n[INFO] Loading external ranking information from: {ranking_csv_path}")
    external_ranking_dict = load_external_ranking_info(ranking_csv_path)

    if external_ranking_dict is None:
        print(
            "[ERROR] Cannot load external ranking information, please ensure ranking result file exists and format is correct")
        print("[ERROR] Please run ranking analysis code first to generate ranking result file")
        return None

    # Data checking stages (keep these as they are needed for other data)
    print("\nStage 1: Check WSC sampled data (both forward and reverse)")
    sampling_data_exists = check_wsc_sampling_data_exists(
        selected_datasets, sampling_methods, num_samples, random_seeds
    )

    if not sampling_data_exists:
        print("Starting to generate WSC sampled data...")

        print("Generating forward sampled data...")
        # Import here to avoid circular imports
        from Code.WSC.Feature.multi_feature import main_wsc_multi
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
        # Import here to avoid circular imports
        from Code.WSC.Feature.multi_feature import main_wsc_multi
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
        # Import here to avoid circular imports
        from Code.WSC.Feature.single_feature import main_wsc_single
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

    print("\nStage 4: Using external ranking information")
    print("[INFO] Skipping NSGA2 data reading, using external ranking information only")

    print("\nStage 5: WSC Data merging and processing with external ranking (both forward and reverse)")

    # Create ranking DataFrame from external ranking information
    ranking_df = create_ranking_df_from_external_wsc(selected_datasets, selected_modes, external_ranking_dict,
                                                     include_reverse=True)

    if ranking_df.empty:
        print("[ERROR] Failed to create ranking DataFrame from external ranking information")
        return None

    # Process data using external ranking
    processed_data = process_data_with_external_ranking(
        start_seed=start_seed,
        end_seed=end_seed,
        selected_modes=selected_modes,
        selected_datasets=selected_datasets,
        pic_types=pic_types,
        data_mode=data_mode,
        ranking_df=ranking_df,
        maximize_datasets=maximize_datasets,
        reverse_maximize_datasets=reverse_maximize_datasets,
    )

    if processed_data is not None:
        print("Starting NaN-based column filtering...")
        processed_data = filter_columns_by_nan(processed_data)

        output_folder = '../../../Results/Predict-raw-data/ProcessedData'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Use external ranking output filename
        output_filename = 'processed_data_wsc.csv'
        output_path = os.path.join(output_folder, output_filename)
        processed_data.to_csv(output_path, index=False)
        print(f"WSC Final processed data (combined forward and reverse) saved to: {output_path}")

        print("\nWSC Data summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        # Check if ranking columns are included
        if 'ft_rank' in processed_data.columns:
            print(f"\nRanking statistics:")
            rank_stats = processed_data['ft_rank'].value_counts().sort_index()
            for rank, count in rank_stats.items():
                print(f"  Rank {rank}: {count} rows")

        if 'is_best_mode' in processed_data.columns:
            best_count = processed_data['is_best_mode'].sum()
            print(f"  Best mode count: {best_count}")

        forward_count = len([ds for ds in processed_data['Dataset Name'] if not ds.endswith('_reverse')])
        reverse_count = len([ds for ds in processed_data['Dataset Name'] if ds.endswith('_reverse')])
        print(f"\nForward data rows: {forward_count}")
        print(f"Reverse data rows: {reverse_count}")

        print("\n" + "=" * 60)
        print("WSC Data processing pipeline completed using external ranking information")
        print("=" * 60)

        return processed_data
    else:
        print("No valid WSC data combinations generated")
        return None


# 修改主调用部分
if __name__ == "__main__":
    # Must provide ranking result CSV file path
    ranking_csv_path = '../../../Results/Predict-raw-data/Ranking/non_ft_modes_ranking_wsc.csv'

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
        reverse_maximize_datasets=["workflow_1", "workflow_2"],
        ranking_csv_path=ranking_csv_path  # Required parameter
    )