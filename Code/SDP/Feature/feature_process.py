import pandas as pd
import os
import re
import warnings
import numpy as np
import sys
from typing import List, Dict, Set, Tuple, Any
import multiprocessing
import csv
from collections import defaultdict
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/')
from Code.SDP.Feature.multi_feature import main_sdp_multi
from Code.SDP.Feature.single_feature import main_sdp_single

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


def check_sdp_sampling_data_exists(selected_datasets: List[str], sampling_methods: List[str],
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
                    print(f"Missing SDP sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

    print("All SDP sampled data exists")
    return True

def check_sdp_multi_feature_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                        sampling_methods: List[str]) -> bool:
    base_dir = "./Results/Output-draw/"

    for mode in selected_modes:
        csv_file = f"{mode}_statistics.csv"
        csv_path = os.path.join(base_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"Missing SDP multi-objective feature data: {csv_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"SDP multi-feature data missing required columns {missing_cols}: {csv_file}")
                return False

            for dataset in selected_datasets:
                if dataset not in df['Dataset Name'].values:
                    print(f"SDP multi-feature data missing dataset: {dataset}")
                    return False

            for sampling_method in sampling_methods:
                if sampling_method not in df['Sampling Method'].values:
                    print(f"SDP multi-feature data missing sampling method: {sampling_method}")
                    return False

        except Exception as e:
            print(f"Error checking SDP multi-objective feature data: {e}")
            return False

    print("All SDP multi-objective feature data exists")
    return True


def check_sdp_landscape_feature_data_exists(selected_datasets: List[str]) -> bool:
    base_dir = "./Results/real_data/"

    for dataset in selected_datasets:
        csv_file = f"{dataset}.csv"
        csv_path = os.path.join(base_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"Missing SDP landscape feature data: {csv_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"SDP landscape feature data file is empty: {csv_file}")
                return False

            required_cols = ['Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"SDP landscape data missing required columns {missing_cols}: {csv_file}")
                return False

        except Exception as e:
            print(f"Error checking SDP landscape feature data: {e}")
            return False

    print("All SDP landscape feature data exists")
    return True

def read_sdp_landscape_data(landscape_csv_dir, selected_datasets, start_seed, end_seed):
    landscape_dfs = []

    for dataset_name in selected_datasets:
        csv_file = f"{dataset_name}.csv"
        csv_path = os.path.join(landscape_csv_dir, csv_file)

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                if 'Name' in df.columns and 'Dataset Name' not in df.columns:
                    df = df.rename(columns={'Name': 'Dataset Name'})

                required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: SDP Landscape data missing columns {missing_cols} in file {csv_file}")
                    continue

                df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]

                sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                    'covering_array', 'halton']
                df = df[df['Sampling Method'].isin(sampling_methods)]

                df['Dataset Name'] = dataset_name

                landscape_dfs.append(df)

            except Exception as e:
                print(f"Error reading landscape file {csv_file}: {e}")
                continue

    if landscape_dfs:
        landscape_df = pd.concat(landscape_dfs, ignore_index=True)

        landscape_df = landscape_df.loc[:, ~((landscape_df.isna() | (landscape_df == 0)).all(axis=0))]

        group_cols = ['Dataset Name', 'Sample Size', 'Sampling Method']

        numeric_cols = [col for col in landscape_df.select_dtypes(include=[np.number]).columns
                        if col not in group_cols + ['Random Seed']]

        if numeric_cols:
            median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()
            print(f"SDP Landscape median data shape: {median_df.shape}")
            return median_df
        else:
            print("No numeric columns found in SDP landscape data")
            return pd.DataFrame()
    else:
        print("No SDP landscape data found")
        return pd.DataFrame()


def read_sdp_sampling_data(sampling_csv_dir, selected_datasets, start_seed, end_seed, selected_modes, pic_types):
    pic_id_mapping = {1: 'PMO', 2: 'MMO'}

    all_sampling_dfs = []

    for pic_type in pic_types:
        sampling_dfs = []
        for mode in selected_modes:
            csv_file = f"{mode}_statistics.csv"
            csv_path = os.path.join(sampling_csv_dir, csv_file)

            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)

                    if 'Mode' in df.columns and 'mode' not in df.columns:
                        df = df.rename(columns={'Mode': 'mode'})
                    if 'Dataset Name' not in df.columns and 'Name' in df.columns:
                        df = df.rename(columns={'Name': 'Dataset Name'})

                    df = df[df['Dataset Name'].isin(selected_datasets)]

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

                except Exception as e:
                    print(f"Error reading sampling file {csv_file}: {e}")
                    continue

        if sampling_dfs:
            sampling_df = pd.concat(sampling_dfs, ignore_index=True)

            group_cols = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
            numeric_cols = [col for col in sampling_df.select_dtypes(include=[np.number]).columns
                            if col not in group_cols + ['Random Seed']]

            if numeric_cols:
                median_df = sampling_df.groupby(group_cols, as_index=False)[numeric_cols].median()
                all_sampling_dfs.append(median_df)
                print(f"SDP {pic_type} sampling median data shape: {median_df.shape}")
            else:
                print(f"No numeric columns found for {pic_type} sampling data")

    if all_sampling_dfs:
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='outer')

        print(f"SDP Combined sampling median data shape: {combined_sampling_df.shape}")
        return combined_sampling_df
    else:
        print("No SDP sampling data found")
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
        print(f"Columns meeting NaN criteria will be dropped (count={len(columns_to_drop)}):")
        for col in columns_to_drop:
            print(f"- {col} (NaN count: {column_nan_counts[col]})")
        return df.drop(columns=columns_to_drop)
    else:
        print("No columns met the NaN criteria")
        return df


def load_external_ranking_info(ranking_csv_path):
    if not os.path.exists(ranking_csv_path):
        print(f"Error: External ranking file does not exist: {ranking_csv_path}")
        print("Error: Please run ranking analysis code to generate ranking result file first")
        return None

    try:
        ranking_df = pd.read_csv(ranking_csv_path)

        required_cols = ['Dataset Name', 'mode', 'unique_rank']
        missing_cols = [col for col in required_cols if col not in ranking_df.columns]
        if missing_cols:
            print(f"Error: Ranking result file missing required columns: {missing_cols}")
            print(f"Existing columns: {ranking_df.columns.tolist()}")
            return None

        ranking_dict = {}
        for _, row in ranking_df.iterrows():
            dataset_name = row['Dataset Name']
            mode = row['mode']
            unique_rank = row['unique_rank']

            key = (dataset_name, mode)
            ranking_dict[key] = {
                'unique_rank': unique_rank,
                'is_best': 'is_best' in row and row['is_best'],

            }

        print(f"Information: Loaded ranking information for {len(ranking_dict)} datasets from {ranking_csv_path}")
        return ranking_dict

    except Exception as e:
        print(f"Error: Failed to read ranking result file: {e}")
        return None


def create_ranking_df_from_external_sdp(selected_datasets_fold, selected_modes, ranking_dict):
    ranking_data = []

    for dataset_name in selected_datasets_fold:
        for mode in selected_modes:
            key = (dataset_name, mode)

            if key in ranking_dict:
                rank_info = ranking_dict[key]
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': rank_info['unique_rank'],
                    'is_best_mode': rank_info.get('is_best', False),

                })
            else:
                print(f"Warning: No ranking information found for dataset '{dataset_name}' mode '{mode}', using default rank 1")
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': 1,
                    'is_best_mode': False,

                })

    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        print(f"Information: Created ranking DataFrame with {len(ranking_df)} rows for SDP")
        return ranking_df
    else:
        print("Warning: Failed to create any SDP ranking information")
        return pd.DataFrame()


def coordinated_pipeline_sdp(
        selected_datasets=None,
        selected_modes=None,
        sampling_methods=None,
        random_seeds=None,
        num_samples=1000,
        fa_construction=None,
        use_multiprocessing=True,
        max_workers=None,
        reverse=False,
        debug=False,
        start_seed=None,
        end_seed=None,
        pic_types=None,
        data_mode='three_datasets',
        workflow_base_path='../Datasets/',
        classifiers=None,
        ranking_csv_path=None
):
    reverse = False

    if selected_datasets is None:
        selected_datasets = ['ant-1.7', 'camel-1.6', 'ivy-2.0', 'jedit-4.0', 'lucene-2.4',
                             'poi-3.0', 'synapse-1.2', 'velocity-1.6', 'xalan-2.6', 'xerces-1.4']
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

    if ranking_csv_path is None:
        print("Error: ranking_csv_path parameter is required")
        print("Error: Please run ranking analysis code to generate ranking result file first, then specify the file path")
        return None

    if classifiers is None:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB

        classifiers = {
            "J48": DecisionTreeClassifier(criterion="entropy", random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "LR": LogisticRegression(max_iter=1000, random_state=42),
            "NB": GaussianNB()
        }

    selected_datasets_fold = []
    input_folders = [key for key in classifiers.keys()]
    for rule in selected_datasets:
        for folder in input_folders:
            dataset = f"{rule}_{folder}"
            selected_datasets_fold.append(dataset)

    print(f"Information: Loading ranking information from external ranking result file: {ranking_csv_path}")
    external_ranking_dict = load_external_ranking_info(ranking_csv_path)

    if external_ranking_dict is None:
        print("Error: Unable to load external ranking information")
        print("Error: Please run ranking analysis code to generate ranking result file first")
        return None

    print("=" * 60)
    print("Starting SDP Coordinated Data Processing Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Datasets with classifiers: {len(selected_datasets_fold)}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Reverse: {reverse} (fixed to False)")
    print(f"  External ranking file: {ranking_csv_path}")
    print("=" * 60)

    print("Stage 1: Check SDP sampled data")
    sampling_data_exists = check_sdp_sampling_data_exists(
        selected_datasets_fold, sampling_methods, num_samples, random_seeds
    )

    if not sampling_data_exists:
        print("Starting to generate SDP sampled data...")
        dataset_paths = [f"{workflow_base_path}/{ds.split('_')[0]}.csv" for ds in selected_datasets]

        main_sdp_multi(
            dataset_paths=dataset_paths,
            fa_construction_list=['g1'],
            classifiers=classifiers,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=list(random_seeds),
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            first_sample=True,
            data_base_path=workflow_base_path
        )
        print("SDP sampled data generation completed")
    else:
        print("SDP sampled data exists, skipping sampling stage")

    print("Stage 2: Check SDP multi-objective feature data")
    multi_feature_data_exists = check_sdp_multi_feature_data_exists(
        selected_datasets_fold, selected_modes, sampling_methods
    )

    if not multi_feature_data_exists:
        print("Starting SDP multi-objective feature computation...")
        dataset_paths = [f"{workflow_base_path}/{ds.split('_')[0]}.csv" for ds in selected_datasets]

        main_sdp_multi(
            dataset_paths=dataset_paths,
            fa_construction_list=fa_construction,
            classifiers=classifiers,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=list(random_seeds),
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            first_sample=False,
            data_base_path=workflow_base_path
        )
        print("SDP multi-objective feature computation completed")
    else:
        print("SDP multi-objective feature data exists, skipping computation stage")

    print("Stage 3: Check SDP landscape feature data")
    landscape_feature_data_exists = check_sdp_landscape_feature_data_exists(selected_datasets_fold)

    if not landscape_feature_data_exists:
        print("Starting SDP landscape feature computation...")
        main_sdp_single(
            dataset_names=[ds.split('_')[0] for ds in selected_datasets],
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=list(random_seeds),
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            debug=debug,
            result_dir="./Results/real_data/"
        )
        print("SDP landscape feature computation completed")
    else:
        print("SDP landscape feature data exists, skipping computation stage")

    print("Stage 4: Check SDP NSGA2 data")
    print("Information: Skipping NSGA2 data check, only ranking information is needed")

    print("Stage 5: SDP Data merging and processing")

    print("Starting SDP data merging...")

    landscape_df = read_sdp_landscape_data('./Results/real_data/', selected_datasets_fold, start_seed, end_seed)
    combined_sampling_df = read_sdp_sampling_data('./Results/Output-draw/', selected_datasets_fold,
                                                  start_seed, end_seed, selected_modes, pic_types)

    ranking_df = create_ranking_df_from_external_sdp(selected_datasets_fold, selected_modes, external_ranking_dict)

    print(f"SDP landscape_df shape: {landscape_df.shape}")
    print(f"SDP combined_sampling_df shape: {combined_sampling_df.shape}")
    print(f"SDP ranking_df shape: {ranking_df.shape}")

    if landscape_df.empty:
        print("Warning: SDP Landscape data is empty")
    if combined_sampling_df.empty:
        print("Warning: SDP Sampling data is empty")
    if ranking_df.empty:
        print("Warning: SDP Ranking data is empty")

    combined_dfs = []

    for dataset_name in selected_datasets_fold:
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
        print(f"SDP All combined data shape: {all_combined_df.shape}")

        all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

        columns_to_keep = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

        numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
        columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

        existing_columns = [col for col in columns_to_keep if col in all_combined_df.columns]
        processed_data = all_combined_df[existing_columns].dropna(axis=1, how='all')
        processed_data = processed_data.reset_index(drop=True)

        print(f"SDP Final data shape after column selection: {processed_data.shape}")

        processed_data = filter_columns_by_nan(processed_data)

        output_folder = '../../../Results/Predict-raw-data/ProcessedData'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_filename = 'processed_data_sdp_external_ranking.csv'
        output_path = os.path.join(output_folder, output_filename)
        processed_data.to_csv(output_path, index=False)
        print(f"SDP Final processed data saved to: {output_path}")

        print("SDP Data summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        if 'Dataset Name' in processed_data.columns:
            dataset_counts = processed_data['Dataset Name'].value_counts()
            print("Dataset distribution:")
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count} rows")

        if 'ft_rank' in processed_data.columns:
            print("Ranking statistics:")
            rank_stats = processed_data['ft_rank'].value_counts().sort_index()
            for rank, count in rank_stats.items():
                print(f"  Rank {rank}: {count} rows")

        print("=" * 60)
        print("SDP Data processing pipeline completed")
        print("=" * 60)

        return processed_data
    else:
        print("No valid SDP data combinations generated")
        return None

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    CLASSIFIERS = {
        "J48": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "NB": GaussianNB()
    }

    selected_datasets = ['ant-1.7', 'camel-1.6', 'ivy-2.0', 'jedit-4.0', 'lucene-2.4',
                         'poi-3.0', 'synapse-1.2', 'velocity-1.6', 'xalan-2.6', 'xerces-1.4']

    ranking_csv_path = '../../../Results/Predict-raw-data/Ranking/non_ft_modes_ranking_sdp.csv'

    processed_data = coordinated_pipeline_sdp(
        selected_datasets=selected_datasets,
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'],
        random_seeds=range(0, 10),
        num_samples=1000,
        fa_construction=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=50,
        reverse=False,
        debug=True,
        pic_types=['PMO', 'MMO'],
        workflow_base_path='../Datasets/',
        classifiers=CLASSIFIERS,
        ranking_csv_path=ranking_csv_path
    )