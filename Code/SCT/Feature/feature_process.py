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


def load_external_ranking_info(ranking_csv_path):
    if not os.path.exists(ranking_csv_path):

        return None

    try:
        ranking_df = pd.read_csv(ranking_csv_path)


        required_cols = ['Dataset Name', 'mode', 'unique_rank']
        missing_cols = [col for col in required_cols if col not in ranking_df.columns]
        if missing_cols:

            return None

        ranking_dict = {}
        for _, row in ranking_df.iterrows():
            dataset_name = row['Dataset Name']
            mode = row['mode']
            unique_rank = row['unique_rank']

            key = (dataset_name, mode)
            ranking_dict[key] = {
                'unique_rank': unique_rank,
            }

        return ranking_dict

    except Exception as e:
        return None


def create_ranking_df_from_external(selected_datasets, selected_modes, ranking_dict):

    ranking_data = []

    all_selected_datasets = selected_datasets + [f"{ds}_reverse" for ds in selected_datasets]

    for dataset_name in all_selected_datasets:
        for mode in selected_modes:
            if mode == 'reciprocal':
                is_reverse = dataset_name.endswith('_reverse')
                base_dataset = dataset_name.replace('_reverse', '') if is_reverse else dataset_name

                if not is_reverse and base_dataset in ['dnn_adiac', 'dnn_dsr', 'dnn_sa']:
                    continue
                if is_reverse and base_dataset == 'x264':
                    continue

            key = (dataset_name, mode)

            if key not in ranking_dict and dataset_name.endswith('_reverse'):
                base_dataset = dataset_name[:-8]
                key_alt = (base_dataset, mode)
                if key_alt in ranking_dict:
                    key = key_alt

            if key in ranking_dict:
                rank_info = ranking_dict[key]
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': rank_info['unique_rank']
                })
            else:
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': 1,
                })

    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        return ranking_df
    else:
        return pd.DataFrame()

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
        workflow_base_path='../Datasets/',
        ranking_csv_path=None
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

    if ranking_csv_path is None:
        print("Error: ranking_csv_path parameter must be provided")
        print("Error: Please run the ranking analysis code first to generate the ranking results file and specify the file path")
        return None

    print("Starting SCT Coordinated Data Processing Pipeline")
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Processing both forward and reverse data")
    print(f"  Number of datasets: {len(selected_datasets)}")
    print(f"  External ranking file: {ranking_csv_path}")

    print("Loading external ranking information from file")
    external_ranking_dict = load_external_ranking_info(ranking_csv_path)

    if external_ranking_dict is None:
        print("Error: Unable to load external ranking information")
        print("Error: Please ensure the ranking results file exists and has correct format")
        return None

    print("Stage 1: Check SCT sampled data (both forward and reverse)")
    sampling_data_exists = check_sct_sampling_data_exists(
        selected_datasets, sampling_methods, num_samples, random_seeds
    )

    if not sampling_data_exists:
        print("Generating SCT sampled data...")

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

    print("Stage 2: Check SCT multi-objective feature data (both forward and reverse)")
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

    print("Stage 3: Check SCT landscape feature data (both forward and reverse)")
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

    print("Stage 4: Check SCT NSGA2 data (both forward and reverse)")
    print("Skipping NSGA2 data check as only ranking information is needed")

    print("Stage 5: SCT Data merging and processing (both forward and reverse)")
    print("Starting SCT data merging...")

    landscape_df = read_sct_landscape_data('./Results/real_data/', selected_datasets, start_seed, end_seed)
    combined_sampling_df = read_sct_sampling_data('./Results/Output-draw/', selected_datasets,
                                                  start_seed, end_seed, selected_modes, pic_types)

    ranking_df = create_ranking_df_from_external(selected_datasets, selected_modes, external_ranking_dict)

    print(f"SCT landscape_df shape: {landscape_df.shape}")
    print(f"SCT combined_sampling_df shape: {combined_sampling_df.shape}")
    print(f"SCT ranking_df shape: {ranking_df.shape}")

    if landscape_df.empty:
        print("Warning: SCT Landscape data is empty")
    if combined_sampling_df.empty:
        print("Warning: SCT Sampling data is empty")
    if ranking_df.empty:
        print("Warning: SCT Ranking data is empty")

    required_cols_landscape = ['Dataset Name', 'Sample Size', 'Sampling Method']
    required_cols_sampling = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

    landscape_missing = [col for col in required_cols_landscape if col not in landscape_df.columns]
    sampling_missing = [col for col in required_cols_sampling if col not in combined_sampling_df.columns]

    if landscape_missing:
        print(f"Error: SCT Landscape data missing required columns: {landscape_missing}")
        return None
    if sampling_missing:
        print(f"Error: SCT Sampling data missing required columns: {sampling_missing}")
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

                ranking_filtered = ranking_df[
                    (ranking_df['Dataset Name'] == dataset_name) &
                    (ranking_df['mode'] == mode)
                    ].copy()

                if not ranking_filtered.empty:
                    combined_df = pd.merge(
                        combined_df,
                        ranking_filtered,
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
        print(f"SCT Final processed data saved to: {output_path}")

        print("SCT Data processing pipeline completed")
        return processed_data
    else:
        print("No valid SCT data combinations generated")
        return None


if __name__ == "__main__":
    ranking_csv_path = '../../../Results/Predict-raw-data/Ranking/non_ft_modes_ranking_sct.csv'

    processed_data = coordinated_pipeline_sct(
        selected_datasets=['dnn_adiac', 'dnn_coffee', 'dnn_dsr', 'dnn_sa',
                           'llvm', 'lrzip', 'mariadb', 'mongodb', 'vp9', 'x264',
                           'storm_rs', 'storm_wc', 'trimesh'],
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['latin_hypercube', 'sobol', 'orthogonal', 'stratified', 'monte_carlo', 'covering_array'],
        random_seeds=range(0, 10),
        num_samples=900,
        fa_construction=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=50,
        use_saved_data=False,
        debug=True,
        pic_types=['PMO', 'MMO'],
        workflow_base_path='../Datasets/',
        ranking_csv_path=ranking_csv_path  # 必须提供的参数
    )