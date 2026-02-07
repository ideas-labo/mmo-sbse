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

SAMPLING_METHODS = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array', 'halton', 'random_walk']
RESULT_DIR = "./Results/"
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 30

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

from Code.SPSP.Feature.multi_feature import main_spsp_multi
from Code.SPSP.Feature.single_feature import main_spsp_single

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
        file_name = file_name[4:]
    if process_reverse and file_name.endswith("_reverse"):
        file_name = file_name[:-8]
        is_reverse = True
    else:
        is_reverse = False

    parts = file_name.split('_')
    mode = parts[2]

    parts_name = parts[0].split('-')
    dataset_name = '-'.join(parts_name[1:])
    seed = parts[1]
    return dataset_name, mode, is_reverse, seed

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
        landscape_df = landscape_df.loc[:, ~((landscape_df.isna() | (landscape_df == 0)).all(axis=0))]

        group_cols = ['Dataset Name', 'Sample Size', 'Sampling Method']
        if process_reverse:
            group_cols.append('is_reverse')
        numeric_cols = [col for col in landscape_df.select_dtypes(include=[np.number]).columns
                        if col not in group_cols]

        median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        print(f"Landscape median data shape: {median_df.shape}")
        return median_df
    else:
        print("No landscape data found")
        return pd.DataFrame()

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
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method','Random Seed'] + (
            ['is_reverse'] if process_reverse else [])
        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='inner')
        print(f"Combined sampling median data shape: {combined_sampling_df.shape}")
        return combined_sampling_df
    else:
        print("No sampling data found")
        return pd.DataFrame()


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
                'unique_rank': unique_rank,}

        return ranking_dict

    except Exception as e:
        print(f"[ERROR] Failed to read ranking result file: {e}")
        return None


def create_ranking_df_from_external_spsp(selected_datasets, selected_modes, ranking_dict, process_reverse=False):
    """
    Create SPSP ranking DataFrame from external ranking information

    Parameters:
        selected_datasets: Selected dataset list
        selected_modes: Selected mode list
        ranking_dict: Ranking dictionary loaded from CSV
        process_reverse: Whether to process reverse datasets

    Returns:
        ranking_df: DataFrame containing ranking information
    """
    ranking_data = []

    # For each dataset (including reverse if needed)
    all_datasets_to_process = selected_datasets.copy()
    if process_reverse:
        all_datasets_to_process += [f"{ds}_reverse" for ds in selected_datasets]

    for dataset_name in all_datasets_to_process:
        # Determine if this is a reverse dataset
        is_reverse = dataset_name.endswith("_reverse") if process_reverse else False

        # For SPSP, dataset names in ranking file should match selected_datasets
        # No special processing needed for SPSP dataset names
        base_dataset = dataset_name.replace("_reverse", "") if is_reverse else dataset_name

        for mode in selected_modes:
            # Create key for lookup
            key = (base_dataset, mode, is_reverse)

            # Try to find ranking info
            if key in ranking_dict:
                rank_info = ranking_dict[key]
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'is_reverse': is_reverse,
                    'ft_rank': rank_info['unique_rank'],

                })
            else:
                # If not found, use default ranking
                print(
                    f"[WARNING] No ranking info found for dataset '{dataset_name}', mode '{mode}', reverse={is_reverse}, using default rank 1")
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'is_reverse': is_reverse,
                    'ft_rank': 1,

                })

    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        print(f"[INFO] Created ranking DataFrame with {len(ranking_df)} rows for SPSP")
        return ranking_df
    else:
        print("[WARNING] Failed to create any SPSP ranking information")
        return pd.DataFrame()


def coordinated_pipeline_spsp(
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
        workflow_base_path='../Datasets/',
        ranking_csv_path=None  # New parameter: path to ranking CSV file
):
    if selected_datasets is None:
        selected_datasets = [
            "10-5-skill-4-5", "10-5-skill-6-7",
            "10-10-skill-4-5", "10-10-skill-6-7",
            "10-15-skill-4-5", "10-15-skill-6-7",
            "20-5-skill-4-5", "20-5-skill-6-7",
            "20-10-skill-4-5", "20-10-skill-6-7",
            "20-15-skill-4-5", "20-15-skill-6-7",
            "30-5-skill-4-5", "30-5-skill-6-7",
            "30-10-skill-4-5", "30-10-skill-6-7",
            "30-15-skill-4-5", "30-15-skill-6-7",
        ]
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
        maximize_datasets = []
    if reverse_maximize_datasets is None:
        reverse_maximize_datasets = []

    # Check if ranking_csv_path is provided
    if ranking_csv_path is None:
        print("[ERROR] Must provide ranking_csv_path parameter to specify ranking result CSV file path")
        print("[ERROR] Please run ranking analysis code first to generate ranking results, then specify file path")
        return None

    print("=" * 60)
    print("Starting SPSP Coordinated Data Processing Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
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

    print("\nStage 1: Check sampled data")
    sampling_data_exists = check_sampling_data_exists(selected_datasets, sampling_methods, num_samples, random_seeds)

    if not sampling_data_exists:
        print("Starting to generate sampled data...")
        main_spsp_multi(
            dataset_names=selected_datasets,
            fa_construction=['g1'],
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            first_sample=True,
            workflow_base_path=workflow_base_path
        )
        print("Sampled data generation completed")
    else:
        print("Sampled data exists, skipping sampling stage")

    print("\nStage 2: Check multi-objective feature data")
    multi_feature_data_exists = check_multi_feature_data_exists(selected_datasets, selected_modes, sampling_methods)

    if not multi_feature_data_exists:
        print("Starting multi-objective feature computation...")
        main_spsp_multi(
            dataset_names=selected_datasets,
            fa_construction=fa_construction,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            first_sample=False,
            workflow_base_path=workflow_base_path
        )
        print("Multi-objective feature computation completed")
    else:
        print("Multi-objective feature data exists, skipping computation stage")

    print("\nStage 3: Check landscape feature data")
    landscape_feature_data_exists = check_landscape_feature_data_exists(selected_datasets)

    if not landscape_feature_data_exists:
        print("Starting landscape feature computation...")
        main_spsp_single(
            dataset_names=selected_datasets,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            debug=debug,
            use_saved_data=True,
            workflow_base_path=workflow_base_path
        )
        print("Landscape feature computation completed")
    else:
        print("Landscape feature data exists, skipping computation stage")

    print("\nStage 4: Using external ranking information")
    print("[INFO] Skipping NSGA2 data reading, using external ranking information only")

    print("\nStage 5: Data merging and processing")

    print("Starting data merging...")

    # Read landscape and sampling data (still needed)
    landscape_df = read_landscape_data('./Results/real_data/', selected_datasets, start_seed,
                                       end_seed, process_reverse)
    combined_sampling_df = read_sampling_data('./Results/Output-draw/', selected_datasets,
                                              start_seed, end_seed, selected_modes, pic_types, process_reverse)

    # Create ranking DataFrame from external ranking information
    ranking_df = create_ranking_df_from_external_spsp(selected_datasets, selected_modes,
                                                      external_ranking_dict, process_reverse)

    print(f"landscape_df shape: {landscape_df.shape}")
    print(f"combined_sampling_df shape: {combined_sampling_df.shape}")
    print(f"ranking_df shape: {ranking_df.shape}")

    if landscape_df.empty:
        print("Warning: Landscape data is empty")
    if combined_sampling_df.empty:
        print("Warning: Sampling data is empty")
    if ranking_df.empty:
        print("Warning: Ranking data is empty")

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
                             selected_datasets] + selected_datasets if process_reverse else selected_datasets

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
                    # Skip reciprocal mode for specific conditions (if still applicable)
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

                    # Get ranking information for this dataset and mode
                    ranking_filtered = ranking_df[
                        (ranking_df['Dataset Name'] == dataset_name) &
                        (ranking_df['mode'] == mode) &
                        (ranking_df['is_reverse'] == is_reverse)
                        ].copy()

                    if not ranking_filtered.empty:
                        # Merge ranking information
                        ranking_columns_to_add = ['ft_rank']
                        for col in ranking_columns_to_add:
                            if col in ranking_filtered.columns:
                                # Add ranking column to combined_df
                                combined_df[col] = ranking_filtered[col].iloc[0]

                    combined_dfs.append(combined_df)

    if combined_dfs:
        all_combined_df = pd.concat(combined_dfs, ignore_index=True)
        all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

        # Update columns to keep: remove NSGA2 original data columns, keep ranking columns
        columns_to_keep = [
            'Dataset Name', 'mode', 'Sample Size', 'Sampling Method',
            'ft_rank'
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

        # Use external ranking output filename
        output_filename = 'processed_data_spsp.csv'
        output_path = os.path.join(output_folder, output_filename)
        processed_data.to_csv(output_path, index=False)
        print(f"Final processed data saved to: {output_path}")

        print("\nData summary:")
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

        print("\n" + "=" * 60)
        print("SPSP Data processing pipeline completed using external ranking information")
        print("=" * 60)

        return processed_data
    else:
        print("No valid data combinations generated")
        return None


# 修改主调用部分
if __name__ == "__main__":
    # Must provide ranking result CSV file path
    ranking_csv_path = '../../../Results/Predict-raw-data/Ranking/non_ft_modes_ranking_spsp.csv'

    processed_data = coordinated_pipeline_spsp(
        selected_datasets=["10-5-skill-4-5", "10-5-skill-6-7",
                           "10-10-skill-4-5", "10-10-skill-6-7",
                           "10-15-skill-4-5", "10-15-skill-6-7",
                           "20-5-skill-4-5", "20-5-skill-6-7",
                           "20-10-skill-4-5", "20-10-skill-6-7",
                           "20-15-skill-4-5", "20-15-skill-6-7",
                           "30-5-skill-4-5", "30-5-skill-6-7",
                           "30-10-skill-4-5", "30-10-skill-6-7",
                           "30-15-skill-4-5", "30-15-skill-6-7", ],
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['monte_carlo', 'latin_hypercube', 'sobol', 'stratified', 'halton', 'random_walk'],
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
        workflow_base_path='../Datasets/',
        ranking_csv_path=ranking_csv_path  # Must provide this parameter
    )