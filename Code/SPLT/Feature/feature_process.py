from collections import defaultdict
import pandas as pd
import os
import re
import warnings
import numpy as np
import sys
import multiprocessing
import concurrent.futures
import csv
from typing import List, Dict, Set, Tuple, Any

warnings.filterwarnings("ignore", category=FutureWarning)

MODE_PRIORITY_ORDER = {
    'gaussian': 2,
    'reciprocal': 1,
    'g1_g2': 4,
    'age': 3,
    'novelty': 7,
    'diversity': 6,
    'penalty': 5
}

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

try:
    from Code.SPLT.Feature.multi_feature import main_splt_multi
    from Code.SPLT.Feature.single_feature import main_splt_single

    SPLT_IMPORT_SUCCESS = True
    print("Successfully imported SPLT feature computation modules")
except ImportError as e:
    SPLT_IMPORT_SUCCESS = False
    print(f"Warning: Could not import SPLT feature computation modules: {e}")
    print("Will use simulated functions in the pipeline")

WORKFLOW_DIR = "../Datasets/"
RESULT_DIR = "./Results/real_data/"
OUTPUT_DRAW_DIR = "./Results/Output-draw/"
SAMPLES_DIR = "./Results/Samples_multi/"
NSGA2_DIR = "../../../Results/RQ1-raw-data/SPLT/"
PROCESSED_DATA_DIR = "../../../Results/Predict-raw-data/ProcessedData"

def check_sampling_data_exists(selected_datasets: List[str], sampling_methods: List[str],
                               num_samples: int, random_seeds: range) -> bool:
    base_dir = SAMPLES_DIR

    for dataset in selected_datasets:
        for sampling_method in sampling_methods:
            for seed in random_seeds:
                fig1_file = f"sampled_data_{dataset}_g1_g2_{sampling_method}_{num_samples}_seed_{seed}_figure1.csv"
                fig2_file = f"sampled_data_{dataset}_g1_g2_{sampling_method}_{num_samples}_seed_{seed}_figure2.csv"

                fig1_path = os.path.join(base_dir, fig1_file)
                fig2_path = os.path.join(base_dir, fig2_file)

                if not (os.path.exists(fig1_path) and os.path.exists(fig2_path)):
                    print(f"Missing SPLT sampled data: {dataset}, {sampling_method}, seed {seed}")
                    return False

    print("All SPLT sampled data exists")
    return True

def check_multi_feature_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                    sampling_methods: List[str]) -> bool:
    base_dir = OUTPUT_DRAW_DIR

    for mode in selected_modes:
        csv_file = f"{mode}_statistics.csv"
        csv_path = os.path.join(base_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"Missing SPLT multi-objective feature data: {csv_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['Dataset Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"SPLT multi-feature data missing required columns {missing_cols}: {csv_file}")
                return False

            for dataset in selected_datasets:
                if dataset not in df['Dataset Name'].values:
                    print(f"SPLT multi-feature data missing dataset: {dataset}")
                    return False

            for sampling_method in sampling_methods:
                if sampling_method not in df['Sampling Method'].values:
                    print(f"SPLT multi-feature data missing sampling method: {sampling_method}")
                    return False

        except Exception as e:
            print(f"Error checking SPLT multi-objective feature data: {e}")
            return False

    print("All SPLT multi-objective feature data exists")
    return True

def check_landscape_feature_data_exists(selected_datasets: List[str]) -> bool:
    base_dir = RESULT_DIR

    for dataset in selected_datasets:
        csv_file = f"{dataset}.csv"
        csv_path = os.path.join(base_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"Missing SPLT landscape feature data: {csv_file}")
            return False

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"SPLT landscape feature data file is empty: {csv_file}")
                return False

            required_cols = ['Name', 'Sampling Method', 'Sample Size', 'Random Seed']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"SPLT landscape data missing required columns {missing_cols}: {csv_file}")
                return False

        except Exception as e:
            print(f"Error checking SPLT landscape feature data: {e}")
            return False

    print("All SPLT landscape feature data exists")
    return True

def read_landscape_data(landscape_csv_dir, selected_datasets, start_seed, end_seed, process_reverse=False):
    landscape_dfs = []
    for file in os.listdir(landscape_csv_dir):
        if file.endswith('.csv') and '_significance' not in file:
            base_name = file[:-4]
            is_reverse = False
            if process_reverse and base_name.endswith('_reverse'):
                base_name = base_name[:-8]
                is_reverse = True

            if base_name in selected_datasets:
                df = pd.read_csv(os.path.join(landscape_csv_dir, file))

                if 'Name' in df.columns and 'Dataset Name' not in df.columns:
                    df = df.rename(columns={'Name': 'Dataset Name'})

                df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]

                sampling_methods = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"]
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
                        if col not in group_cols + ['Random Seed']]

        if numeric_cols:
            median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()
            print(f"SPLT Landscape median data shape: {median_df.shape}")
            return median_df
        else:
            print("No numeric columns found in SPLT landscape data")
            return landscape_df[group_cols]
    else:
        print("No SPLT landscape data found")
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
                if mode=='g1':
                    mode='g1_g2'
                if mode in selected_modes:
                    df = pd.read_csv(os.path.join(sampling_csv_dir, file))

                    if 'Mode' in df.columns and 'mode' not in df.columns:
                        df = df.rename(columns={'Mode': 'mode'})

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

                    sampling_methods = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo",
                                        "covering_array"]
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
                            if col not in group_cols + ['Random Seed']]

            median_df = sampling_df.groupby(group_cols, as_index=False)[numeric_cols].median()

            non_merge_cols = [col for col in median_df.columns if
                              col not in ['Random Seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method'] + (
                                  ['is_reverse'] if process_reverse else [])]
            median_df = median_df.rename(columns={col: f"{col}_{pic_type}" for col in non_merge_cols})
            all_sampling_dfs.append(median_df)
            print(f"SPLT {pic_type} sampling median data shape: {median_df.shape}")

    if all_sampling_dfs:
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
        if process_reverse:
            merge_keys.append('is_reverse')

        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='inner')
        print(f"SPLT Combined sampling median data shape: {combined_sampling_df.shape}")
        return combined_sampling_df
    else:
        print("No SPLT sampling data found")
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
    Load ranking information from external CSV file

    Parameters:
        ranking_csv_path: Path to ranking result CSV file

    Returns:
        ranking_dict: Dictionary, key is (dataset_name, mode), value is ranking info dictionary
    """
    if not os.path.exists(ranking_csv_path):
        print(f"[ERROR] External ranking result file does not exist: {ranking_csv_path}")
        print("[ERROR] Please run ranking analysis code first to generate ranking result file")
        return None

    try:
        ranking_df = pd.read_csv(ranking_csv_path)

        # Check required columns
        required_cols = ['Dataset Name', 'mode', 'unique_rank']
        missing_cols = [col for col in required_cols if col not in ranking_df.columns]
        if missing_cols:
            print(f"[ERROR] Ranking result file missing required columns: {missing_cols}")
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

            }

        print(
            f"[INFO] Loaded ranking information for {len(ranking_dict)} dataset-mode combinations from {ranking_csv_path}")
        return ranking_dict

    except Exception as e:
        print(f"[ERROR] Failed to read ranking result file: {e}")
        return None


def create_ranking_df_from_external_splt(selected_datasets, selected_modes, ranking_dict, process_reverse=False):
    """
    Create SPLT ranking DataFrame from external ranking information

    Parameters:
        selected_datasets: List of selected datasets
        selected_modes: List of selected modes
        ranking_dict: Dictionary loaded from CSV
        process_reverse: Whether to process reverse datasets

    Returns:
        ranking_df: DataFrame containing ranking information
    """
    ranking_data = []

    # Handle regular and reverse datasets
    all_dataset_variants = []
    for dataset in selected_datasets:
        all_dataset_variants.append(dataset)
        if process_reverse:
            all_dataset_variants.append(f"{dataset}_reverse")

    for dataset_name in all_dataset_variants:
        is_reverse = dataset_name.endswith("_reverse") and process_reverse
        base_dataset = dataset_name.replace("_reverse", "") if is_reverse else dataset_name

        for mode in selected_modes:
            # Handle mode mapping: 'g1_g2' in SPLT might be 'g1' in ranking CSV
            # We need to check both possibilities
            ranking_mode = mode
            if mode == 'g1_g2':
                # Try both 'g1_g2' and 'g1'
                possible_modes = ['g1_g2', 'g1']
                found = False
                for possible_mode in possible_modes:
                    key = (base_dataset, possible_mode)
                    if key in ranking_dict:
                        ranking_mode = possible_mode
                        found = True
                        break
                if not found:
                    print(
                        f"[WARNING] No ranking information found for dataset '{dataset_name}' mode '{mode}', using default rank 1")
                    ranking_data.append({
                        'Dataset Name': dataset_name,
                        'mode': mode,
                        'ft_rank': 1,

                    })
                    continue
            else:
                key = (base_dataset, ranking_mode)

            # Get ranking information
            if key in ranking_dict:
                rank_info = ranking_dict[key]
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': rank_info['unique_rank'],
                    'is_best_mode': rank_info.get('is_best', False),

                })
            else:
                # No ranking information found, use default rank 1
                print(
                    f"[WARNING] No ranking information found for dataset '{dataset_name}' mode '{mode}', using default rank 1")
                ranking_data.append({
                    'Dataset Name': dataset_name,
                    'mode': mode,
                    'ft_rank': 1,

                })

    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        print(f"[INFO] Created ranking DataFrame with {len(ranking_df)} rows")
        return ranking_df
    else:
        print("[WARNING] Could not create any ranking information")
        return pd.DataFrame()


def validate_ranking_coverage(ranking_df, selected_datasets, selected_modes, process_reverse=False):
    """
    Validate the coverage of ranking information

    Parameters:
        ranking_df: Ranking DataFrame
        selected_datasets: List of selected datasets
        selected_modes: List of selected modes
        process_reverse: Whether reverse datasets are processed

    Returns:
        coverage_rate: Coverage rate of ranking information
    """
    # Calculate total combinations
    all_dataset_variants = []
    for dataset in selected_datasets:
        all_dataset_variants.append(dataset)
        if process_reverse:
            all_dataset_variants.append(f"{dataset}_reverse")

    total_combinations = len(all_dataset_variants) * len(selected_modes)

    ranked_combinations = len(ranking_df[ranking_df['ft_rank'] > 0])

    coverage_rate = ranked_combinations / total_combinations if total_combinations > 0 else 0

    print(f"[INFO] Ranking coverage rate: {coverage_rate:.2%} ({ranked_combinations}/{total_combinations})")

    if coverage_rate < 0.5:
        print("[WARNING] Low ranking coverage rate, may need to regenerate ranking data")

    return coverage_rate


def coordinated_pipeline_splt(
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
        ranking_csv_path=None  # New parameter: path to ranking result CSV file
):
    if selected_datasets is None:
        selected_datasets = [
            "7z", "Amazon", "BerkeleyDBC", "CocheEcologico", "CounterStrikeSimpleFeatureModel",
            "DSSample", "Dune", "ElectronicDrum", "HiPAcc", "Drupal",
            "JavaGC", "JHipster", "lrzip", "ModelTransformation",
            "SmartHomev2.2", "SPLSSimuelESPnP", "VideoPlayer",
            "VP9", "WebPortal", "x264", 'Polly'
        ]

    if selected_modes is None:
        selected_modes = ['penalty', 'g1_g2', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']

    if sampling_methods is None:
        sampling_methods = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"]

    if random_seeds is None:
        random_seeds = range(0, 10)

    if fa_construction is None:
        fa_construction = ['g1_g2', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']

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

    # Check if ranking CSV path is provided
    if ranking_csv_path is None:
        print("[ERROR] Must provide ranking_csv_path parameter, specify the path to ranking result CSV file")
        print(
            "[ERROR] Please run ranking analysis code first to generate ranking result file, then specify the file path")
        return None

    print("=" * 60)
    print("Starting SPLT Coordinated Data Processing Pipeline with External Ranking")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Process reverse: {process_reverse}")
    print(f"  External ranking file: {ranking_csv_path}")
    print("=" * 60)

    # Load external ranking information
    print(f"\n[INFO] Loading ranking information from external ranking result file: {ranking_csv_path}")
    external_ranking_dict = load_external_ranking_info(ranking_csv_path)

    if external_ranking_dict is None:
        print(
            "[ERROR] Unable to load external ranking information, please ensure ranking result file exists and format is correct")
        print("[ERROR] Please run ranking analysis code first to generate ranking result file")
        return None

    # Create ranking DataFrame from external ranking information
    ranking_df = create_ranking_df_from_external_splt(
        selected_datasets, selected_modes, external_ranking_dict, process_reverse
    )

    if ranking_df.empty:
        print("[ERROR] Ranking DataFrame is empty, cannot proceed")
        return None

    # Validate ranking coverage
    validate_ranking_coverage(ranking_df, selected_datasets, selected_modes, process_reverse)

    if not SPLT_IMPORT_SUCCESS:
        print(
            "Warning: SPLT feature computation modules failed to import; automatic feature computation is unavailable")
        print("Ensure the following paths contain the SPLT modules:")
        print("  /home/ccj/code/mmo")
        print("  /mnt/sdaDisk/ccj/code/mmo")
        print("Continuing with data checks and merging...")

    print("\nStage 1: Check SPLT sampled data")
    sampling_data_exists = check_sampling_data_exists(selected_datasets, sampling_methods, num_samples, random_seeds)

    if not sampling_data_exists:
        print("Warning: SPLT sampled data does not exist")
        print("Note: SPLT has no sampling implementation and cannot auto-generate sampling data")
        print("Ensure sampled data is generated and stored in the following directory:")
        print(f"  {SAMPLES_DIR}")
        print("Sampled data filename format:")
        print(f"  sampled_data_{{dataset}}_g1_g2_{{sampling_method}}_{{num_samples}}_seed_{{seed}}_figure1.csv")
        print(f"  sampled_data_{{dataset}}_g1_g2_{{sampling_method}}_{{num_samples}}_seed_{{seed}}_figure2.csv")
        print("Continuing with existing data...")
    else:
        print("SPLT Sampled data exists")

    print("\nStage 2: Check SPLT multi-objective feature data")
    multi_feature_data_exists = check_multi_feature_data_exists(selected_datasets, selected_modes, sampling_methods)

    if not multi_feature_data_exists and SPLT_IMPORT_SUCCESS:
        print("Starting SPLT multi-objective feature computation...")
        try:
            main_splt_multi(
                dataset_names=selected_datasets,
                fa_construction=fa_construction,
                minimize=True,
                fixed_sample_sizes=[num_samples],
                sampling_methods=sampling_methods,
                random_seeds=random_seeds,
                use_multiprocessing=use_multiprocessing,
                max_workers=max_workers,
                reverse=reverse,
                first_sample=False,
                workflow_base_path=workflow_base_path,
                use_saved_data=True,
                debug=debug
            )
            print("SPLT Multi-objective feature computation completed")
        except Exception as e:
            print(f"SPLT multi-objective feature computation failed: {e}")
            print("Continuing with existing data...")
    elif not multi_feature_data_exists:
        print("Unable to compute SPLT multi-objective features: feature modules not imported")
    else:
        print("SPLT Multi-objective feature data exists, skipping computation stage")

    print("\nStage 3: Check SPLT landscape feature data")
    landscape_feature_data_exists = check_landscape_feature_data_exists(selected_datasets)

    if not landscape_feature_data_exists and SPLT_IMPORT_SUCCESS:
        print("Starting SPLT landscape feature computation...")
        try:
            main_splt_single(
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
            print("SPLT Landscape feature computation completed")
        except Exception as e:
            print(f"SPLT landscape feature computation failed: {e}")
            print("Continuing with existing data...")
    elif not landscape_feature_data_exists:
        print("Unable to compute SPLT landscape features: feature modules not imported")
    else:
        print("SPLT Landscape feature data exists, skipping computation stage")

    print("\nStage 4: SPLT Data merging and processing (skipping NSGA2 data reading)")

    try:
        print("Starting SPLT data merging...")

        # Read landscape feature data and sampling data (these are still needed)
        landscape_df = read_landscape_data(RESULT_DIR, selected_datasets, start_seed,
                                           end_seed, process_reverse)
        combined_sampling_df = read_sampling_data(OUTPUT_DRAW_DIR, selected_datasets,
                                                  start_seed, end_seed, selected_modes, pic_types, process_reverse)

        print(f"SPLT landscape_df shape: {landscape_df.shape}")
        print(f"SPLT combined_sampling_df shape: {combined_sampling_df.shape}")
        print(f"SPLT ranking_df shape: {ranking_df.shape}")

        if landscape_df.empty:
            print("Warning: SPLT Landscape data is empty")
        if combined_sampling_df.empty:
            print("Warning: SPLT Sampling data is empty")
        if ranking_df.empty:
            print("Warning: SPLT Ranking data is empty")

        required_cols_landscape = ['Dataset Name', 'Sample Size', 'Sampling Method']
        required_cols_sampling = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

        landscape_missing = [col for col in required_cols_landscape if col not in landscape_df.columns]
        sampling_missing = [col for col in required_cols_sampling if col not in combined_sampling_df.columns]

        if landscape_missing:
            print(f"SPLT Landscape data missing required columns: {landscape_missing}")
            return None
        if sampling_missing:
            print(f"SPLT Sampling data missing required columns: {sampling_missing}")
            return None

        # Determine all dataset variants to process
        all_dataset_variants = []
        for dataset in selected_datasets:
            all_dataset_variants.append(dataset)
            if process_reverse:
                all_dataset_variants.append(f"{dataset}_reverse")

        combined_dfs = []

        for dataset_name in all_dataset_variants:
            landscape_filtered = landscape_df[landscape_df['Dataset Name'] == dataset_name].copy()

            if landscape_filtered.empty:
                continue

            # Get unique combinations of sampling method and sample size
            landscape_combinations = landscape_filtered[['Sampling Method', 'Sample Size']].drop_duplicates()

            for _, combo in landscape_combinations.iterrows():
                sampling_method = combo['Sampling Method']
                sampling_size = combo['Sample Size']

                landscape_specific = landscape_filtered[
                    (landscape_filtered['Sampling Method'] == sampling_method) &
                    (landscape_filtered['Sample Size'] == sampling_size)
                    ].copy()

                for mode in selected_modes:
                    # Filter sampling data
                    sampling_filtered = combined_sampling_df[
                        (combined_sampling_df['Sampling Method'] == sampling_method) &
                        (combined_sampling_df['Sample Size'] == sampling_size) &
                        (combined_sampling_df['Dataset Name'] == dataset_name) &
                        (combined_sampling_df['mode'] == mode)
                        ].copy()

                    if sampling_filtered.empty:
                        continue

                    # Merge landscape and sampling data
                    combined_df = pd.merge(
                        landscape_specific,
                        sampling_filtered,
                        on=['Dataset Name', 'Sample Size', 'Sampling Method'],
                        how='inner'
                    )

                    # Add ranking information
                    ranking_filtered = ranking_df[
                        (ranking_df['Dataset Name'] == dataset_name) &
                        (ranking_df['mode'] == mode)
                        ].copy()

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
            print(f"SPLT All combined data shape: {all_combined_df.shape}")

            # Remove duplicate columns
            all_combined_df = all_combined_df.loc[:, ~all_combined_df.columns.duplicated()]

            # Select columns to keep
            columns_to_keep = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
            if process_reverse:
                columns_to_keep.append('is_reverse')

            # Add numeric columns
            numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
            columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

            # Keep only existing columns
            existing_columns = [col for col in columns_to_keep if col in all_combined_df.columns]
            processed_data = all_combined_df[existing_columns].dropna(axis=1, how='all')
            processed_data = processed_data.reset_index(drop=True)

            print(f"SPLT Final data shape after column selection: {processed_data.shape}")

            # Filter columns with too many NaN values
            processed_data = filter_columns_by_nan(processed_data)

            # Save processed data
            if not os.path.exists(PROCESSED_DATA_DIR):
                os.makedirs(PROCESSED_DATA_DIR)

            # Use filename indicating external ranking
            output_filename = 'processed_data_splt.csv'
            output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
            processed_data.to_csv(output_path, index=False)
            print(f"SPLT Final processed data saved to: {output_path}")

            print("\nSPLT Data summary:")
            print(f"Total rows: {len(processed_data)}")
            print(f"Total columns: {len(processed_data.columns)}")
            print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
            print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

            # Print dataset distribution
            if 'Dataset Name' in processed_data.columns:
                dataset_counts = processed_data['Dataset Name'].value_counts()
                print(f"\nDataset distribution:")
                for dataset, count in dataset_counts.items():
                    print(f"  {dataset}: {count} rows")

            # Print ranking statistics if available
            if 'ft_rank' in processed_data.columns:
                print(f"\nRanking statistics:")
                rank_stats = processed_data['ft_rank'].value_counts().sort_index()
                for rank, count in rank_stats.items():
                    print(f"  Rank {rank}: {count} rows")

            print("\n" + "=" * 60)
            print("SPLT Data processing pipeline with external ranking completed")
            print("=" * 60)

            return processed_data
        else:
            print("No valid SPLT data combinations generated")
            return None

    except Exception as e:
        print(f"SPLT data processing error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Modified main call
if __name__ == "__main__":
    # Specify the path to ranking result CSV file
    ranking_csv_path = '../../../Results/Predict-raw-data/Ranking/non_ft_modes_ranking_splt.csv'

    processed_data = coordinated_pipeline_splt(
        selected_datasets=["7z", "Amazon", "BerkeleyDBC", "CocheEcologico", "CounterStrikeSimpleFeatureModel",
                           "DSSample", "Dune", "ElectronicDrum", "HiPAcc", "Drupal",
                           "JavaGC", "JHipster", "lrzip", "ModelTransformation",
                           "SmartHomev2.2", "SPLSSimuelESPnP", "VideoPlayer",
                           "VP9", "WebPortal", "x264", 'Polly'],
        selected_modes=['penalty', 'g1_g2', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"],
        random_seeds=range(0, 10),
        num_samples=1000,
        fa_construction=['penalty', 'g1_g2', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
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