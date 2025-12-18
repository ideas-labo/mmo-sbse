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


def check_sdp_nsga2_data_exists(selected_datasets: List[str], selected_modes: List[str],
                                random_seeds: range) -> bool:
    base_dir = "../../../Results/RQ1-raw-data/SDP"

    if not os.path.exists(base_dir):
        print(f"SDP NSGA2 directory does not exist: {base_dir}")
        return False

    missing_files = []

    for dataset in selected_datasets:
        for mode in selected_modes:
            for seed in random_seeds:
                possible_files = [
                    f"{dataset}_{seed}_{mode}_g2.csv",
                    f"{dataset}_{seed}_{mode}_fa.csv",
                    f"{dataset}_{seed}_{mode}_maximization_fa.csv",
                ]

                found = False
                for filename in possible_files:
                    file_path = os.path.join(base_dir, filename)
                    if os.path.exists(file_path):
                        found = True
                        break

                if not found:
                    missing_files.append(f"{dataset}, {mode}, seed {seed}")
                    print(f"Missing SDP NSGA2 data: {dataset}, {mode}, seed {seed}")

    if missing_files:
        print(f"\nTotal missing SDP NSGA2 files: {len(missing_files)}")
        return False
    else:
        print("All SDP NSGA2 data exists")
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


def extract_info_from_filename_sdp(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]
    part = file_name.split('_')
    if part:
        dataset_name = part[0] + '_' + part[1]
        seed = int(part[2])
        mode = part[3]
        return dataset_name, seed, mode

    return None, None, None


def get_pareto_ratios_sdp(csv_path):
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
                    print(f"Cannot convert p value '{p_str}' to float in file {csv_path}; check format.")
                    return None, None, None, None, None

            p_values_mean = sum(p_values) / len(p_values)

            ft_line = lines[-3].strip()
            ft_match = re.search(r"'test_auc': (-?\d+\.?\d*)", ft_line)
            if ft_match:
                ft = float(ft_match.group(1))
            else:
                print(f"Cannot extract ft value from file {csv_path}; check file format.")
                ft = None

            budget_line = lines[-6].strip()
            budget_match = re.search(r'budget_used:(\d+)', budget_line)
            if budget_match:
                budget = int(budget_match.group(1))
            else:
                print(f"Cannot extract budget value from file {csv_path}; check file format.")
                budget = None

            time_line = lines[-5].strip()
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


def read_sdp_nsga2_data(nsga2_csv_dir, selected_datasets, selected_modes):
    p_data = []
    total_files = 0
    valid_files = 0

    for file in os.listdir(nsga2_csv_dir):
        if file.endswith('.csv'):
            total_files += 1
            file_name = os.path.basename(file)
            dataset_name, seed, mode = extract_info_from_filename_sdp(file_name)

            if dataset_name and dataset_name in selected_datasets and mode in selected_modes:
                csv_path = os.path.join(nsga2_csv_dir, file)
                best_p, p_values_mean, ft, budget, time = get_pareto_ratios_sdp(csv_path)

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
                        'time': time
                    })

    print(f"SDP total files: {total_files}")
    print(f"SDP valid files: {valid_files}")

    if p_data:
        p_df = pd.DataFrame(p_data)
        group_cols = ['Dataset Name', 'mode']
        numeric_cols = ['Best_Pareto_Ratio', 'Pareto_Ratios_Mean', 'ft', 'budget', 'time', 'Random Seed']

        median_df = p_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        return median_df
    else:
        return pd.DataFrame()


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


def add_sdp_ranks(p_df, maximize_datasets, ranking_mode='ft_only'):
    ranked_df = p_df.copy()

    ranked_df['ft_rank'] = 1
    ranked_df['time_rank'] = 1
    ranked_df['budget_rank'] = 1

    group_columns = ['Dataset Name', 'Random Seed']

    for group_key, group in ranked_df.groupby(group_columns):
        dataset = group_key[0]
        seed = group_key[1]

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
        print(f"\nColumns meeting NaN criteria will be dropped (count={len(columns_to_drop)}):")
        for col in columns_to_drop:
            print(f"- {col} (NaN count: {column_nan_counts[col]})")
        return df.drop(columns=columns_to_drop)
    else:
        print("\nNo columns met the NaN criteria")
        return df


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
        maximize_datasets=None,
        ranking_mode='ft_mode',
        workflow_base_path='../Datasets/',
        classifiers=None
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
    if maximize_datasets is None:
        maximize_datasets = selected_datasets

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
    selected_datasets_fold=[]
    input_folders = [key for key in classifiers.keys()]
    for rule in selected_datasets:
        for folder in input_folders:
            dataset = f"{rule}_{folder}"
            selected_datasets_fold.append(dataset)
    print("=" * 60)
    print("Starting SDP Coordinated Data Processing Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Reverse: {reverse} (fixed to False)")
    print(f"  Number of datasets: {len(selected_datasets)}")
    print("=" * 60)

    print("\nStage 1: Check SDP sampled data")
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

    print("\nStage 2: Check SDP multi-objective feature data")
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

    print("\nStage 3: Check SDP landscape feature data")
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

    print("\nStage 4: Check SDP NSGA2 data")
    nsga2_data_exists = check_sdp_nsga2_data_exists(selected_datasets_fold, selected_modes, random_seeds)

    if not nsga2_data_exists:
        print("Warning: Some SDP NSGA2 data is missing. Ensure NSGA2 has been run and produced results")
        print("Proceeding with available data...")

    print("\nStage 5: SDP Data merging and processing")

    print("Starting SDP data merging...")

    landscape_df = read_sdp_landscape_data('./Results/real_data/', selected_datasets_fold, start_seed, end_seed)
    p_df = read_sdp_nsga2_data('../../../Results/RQ1-raw-data/SDP/', selected_datasets_fold, selected_modes)
    combined_sampling_df = read_sdp_sampling_data('./Results/Output-draw/', selected_datasets_fold,
                                                  start_seed, end_seed, selected_modes, pic_types)

    print(f"SDP landscape_df shape: {landscape_df.shape}")
    print(f"SDP combined_sampling_df shape: {combined_sampling_df.shape}")
    print(f"SDP p_df shape: {p_df.shape}")

    if landscape_df.empty:
        print("Warning: SDP Landscape data is empty")
    if combined_sampling_df.empty:
        print("Warning: SDP Sampling data is empty")
    if p_df.empty:
        print("Warning: SDP NSGA2 data is empty")

    if not p_df.empty:
        p_df = add_sdp_ranks(p_df, maximize_datasets, ranking_mode)

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

        output_filename = 'processed_data_sdp.csv'
        output_path = os.path.join(output_folder, output_filename)
        processed_data.to_csv(output_path, index=False)
        print(f"SDP Final processed data saved to: {output_path}")

        print("\nSDP Data summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        if 'Dataset Name' in processed_data.columns:
            dataset_counts = processed_data['Dataset Name'].value_counts()
            print(f"\nDataset distribution:")
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count} rows")

        print("\n" + "=" * 60)
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
        maximize_datasets=selected_datasets,
        classifiers=CLASSIFIERS
    )