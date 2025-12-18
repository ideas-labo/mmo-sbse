import sys
from collections import defaultdict
import pandas as pd
import os
import re
import warnings
import numpy as np
from typing import List, Dict, Set, Tuple, Any
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/')
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

from Code.NAS.mmo_nas import c10_search_space_configs, cityseg_search_space_configs, in1k_search_space_configs
from Code.NAS.Feature.multi_feature import main_evox_multi, EvoXProblemManager, EVOXBENCH_PROBLEMS
from Code.NAS.Feature.single_feature import main_evox_single


def extract_info_from_filename(file_name, process_reverse=False):
    is_reverse = "_reverse" in file_name
    file_core = file_name[:-4] if file_name.endswith(".csv") else file_name
    parts = file_core.split('_')

    if is_reverse:
        if len(parts) >= 4:
            dataset_name = parts[0] + '_' + parts[1]
            try:
                seed = int(parts[2])
                mode_parts = parts[3:]
                if mode_parts and mode_parts[-1] in ['fa', 'g2']:
                    mode_parts = mode_parts[:-1]
                if mode_parts and mode_parts[-1] == 'maximization' and len(mode_parts) > 1:
                    mode_parts = mode_parts[:-2]
                mode = '_'.join(mode_parts) if mode_parts else parts[3]
                return dataset_name, mode, is_reverse, seed
            except (ValueError, IndexError):
                return None, None, None, None
    else:
        if len(parts) >= 3:
            dataset_name = parts[0] + '_' + parts[1]
            try:
                seed = int(parts[2])
                mode_parts = parts[3:]
                if mode_parts and mode_parts[-1] in ['fa', 'g2']:
                    mode_parts = mode_parts[:-1]
                if mode_parts and mode_parts[-1] == 'maximization' and len(mode_parts) > 1:
                    mode_parts = mode_parts[:-2]
                mode = '_'.join(mode_parts) if mode_parts else parts[3]
                return dataset_name, mode, is_reverse, seed
            except (ValueError, IndexError):
                return None, None, None, None

    print(f"[Warning] Filename format does not match: {file_name}")
    return None, None, None, None


def get_pareto_ratios(csv_path):
    try:
        with open(csv_path, 'r') as file:
            lines = file.readlines()

            best_solution_text = next((line for line in lines if "Best Solution" in line), None)
            if best_solution_text is None:
                print(f"Best Solution line not found in file {csv_path}")
                return None, None, None, None, None
            p_match = re.search(r'p: (\d+\.\d+)', best_solution_text)
            if p_match:
                best_p = float(p_match.group(1))
            else:
                print(f"Unable to extract Best Pareto ratio from file {csv_path}")
                return None, None, None, None, None

            p_values_text = next((line for line in lines if "p values until best solution" in line), None)
            if p_values_text is None:
                print(f"'p values' line not found in file {csv_path}")
                return None, None, None, None, None
            p_values_str_list = p_values_text.split(": ")[1].strip().split(",")

            p_values = []
            for p_str in p_values_str_list:
                p_str = p_str.strip('"')
                try:
                    p = float(p_str)
                    p_values.append(p)
                except ValueError:
                    print(f"Cannot convert p value '{p_str}' to float in file {csv_path}")
                    return None, None, None, None, None

            p_values_mean = sum(p_values) / len(p_values)

            ft_line = lines[-2].strip()
            ft_match = re.search(r"'ft': (-?\d+\.?\d*)", ft_line)
            if ft_match:
                ft = float(ft_match.group(1))
            else:
                print(f"Unable to extract ft value from file {csv_path}")
                ft = None

            budget_line = lines[-6].strip()
            budget_match = re.search(r'budget_used:(\d+)', budget_line)
            if budget_match:
                budget = int(budget_match.group(1))
            else:
                print(f"Unable to extract budget value from file {csv_path}")
                budget = None

            time_line = lines[-5].strip()
            time_match = re.search(r'Running time: (\d+\.?\d*) seconds', time_line)
            if time_match:
                time = float(time_match.group(1))
            else:
                print(f"Unable to extract time value from file {csv_path}")
                time = None

        return best_p, p_values_mean, ft, budget, time
    except Exception as e:
        print(f"Exception while processing file {csv_path}: {e}")
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
                if 'Random Seed' in df.columns:
                    df = df.rename(columns={'Random Seed': 'seed'})
                df = df[(df['seed'] >= start_seed) & (df['seed'] <= end_seed)]
                sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                    'covering_array']
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
                        if col not in group_cols + ['seed']]

        median_df = landscape_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        print(f"Landscape median data shape: {median_df.shape}")
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

            if dataset_name and mode and dataset_name in selected_datasets and mode in selected_modes:
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

    print(f"\nProcessing completed: {total_files} files processed, {valid_files} valid")

    if all_data:
        temp_df = pd.DataFrame(all_data)
        temp_df = temp_df.dropna(subset=['ft', 'Best_Pareto_Ratio', 'Pareto_Ratios_Mean'])

        group_keys = ['Dataset Name', 'mode', 'is_reverse']
        numeric_cols = ['Best_Pareto_Ratio', 'Pareto_Ratios_Mean', 'ft', 'budget', 'time', 'Random Seed']

        median_df = temp_df.groupby(group_keys, as_index=False)[numeric_cols].median()
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
                    if 'Random Seed' in df.columns:
                        df = df.rename(columns={'Random Seed': 'seed'})

                    if process_reverse:
                        df['Dataset Name'] = df['Dataset Name'].str.replace('_reverse', '')
                        df = df[df['Dataset Name'].isin(selected_datasets)]
                        df['Dataset Name'] = df['Dataset Name'] + ('_reverse' if '_reverse' in file else '')
                    else:
                        df = df[df['Dataset Name'].isin(selected_datasets)]

                    df['Figure Number'] = df['Figure Number'].map(pic_id_mapping)
                    df = df[df['Figure Number'] == pic_type]
                    df = df[(df['seed'] >= start_seed) & (df['seed'] <= end_seed)]
                    df['mode'] = mode
                    if process_reverse:
                        df['is_reverse'] = '_reverse' in file

                    sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                        'covering_array']
                    df = df[df['Sampling Method'].isin(sampling_methods)]

                    for col in df.columns:
                        if df[col].dtype == 'object' and df[col].str.contains('%').any():
                            df[col] = df[col].str.rstrip('%').astype(float) / 100

                    exclude_cols = ['seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method', 'Figure Number']
                    if process_reverse:
                        exclude_cols.append('is_reverse')

                    df = df.rename(columns={col: f"{col.replace('Figure Number', pic_type)}" for col in df.columns if
                                            col not in exclude_cols})
                    df = df.drop(columns=['Figure Number'])
                    sampling_dfs.append(df)

        if sampling_dfs:
            sampling_df = pd.concat(sampling_dfs, ignore_index=True)

            group_cols = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
            if process_reverse:
                group_cols.append('is_reverse')
            numeric_cols = [col for col in sampling_df.select_dtypes(include=[np.number]).columns
                            if col not in group_cols + ['seed']]

            median_df = sampling_df.groupby(group_cols, as_index=False)[numeric_cols].median()
            non_merge_cols = [col for col in median_df.columns if
                              col not in ['seed', 'Dataset Name', 'mode', 'Sample Size', 'Sampling Method'] + (
                                  ['is_reverse'] if process_reverse else [])]
            median_df = median_df.rename(columns={col: f"{col}_{pic_type}" for col in non_merge_cols})
            all_sampling_dfs.append(median_df)
            print(f"{pic_type} sampling median data shape: {median_df.shape}")

    if all_sampling_dfs:
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

        has_seed = all('seed' in df.columns for df in all_sampling_dfs)
        if has_seed:
            merge_keys.append('seed')

        if process_reverse:
            has_reverse = all('is_reverse' in df.columns for df in all_sampling_dfs)
            if has_reverse:
                merge_keys.append('is_reverse')

        print(f"Merge keys: {merge_keys}")

        for i, df in enumerate(all_sampling_dfs):
            print(f"DataFrame {i} columns: {list(df.columns)}")
            missing_cols = [col for col in merge_keys if col not in df.columns]
            if missing_cols:
                print(f"DataFrame {i} missing columns: {missing_cols}")

        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            common_merge_keys = [col for col in merge_keys if col in combined_sampling_df.columns and col in df.columns]
            if common_merge_keys:
                combined_sampling_df = combined_sampling_df.merge(df, on=common_merge_keys, how='inner')
            else:
                print(f"Warning: No common merge keys between DataFrames")

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
        print(f"\nColumns to drop due to NaN condition (total {len(columns_to_drop)}):")
        for col in columns_to_drop:
            print(f"- {col} (NaN count: {column_nan_counts[col]})")
        return df.drop(columns=columns_to_drop)
    else:
        print("\nNo columns met the NaN condition")
        return df


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
    base_dir = "../../../Results/RQ1-raw-data/NAS"

    if not os.path.exists(base_dir):
        print(f"NSGA2 directory does not exist: {base_dir}")
        return False

    missing_files = []

    for dataset in selected_datasets:
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
            for dataset in selected_datasets:
                if dataset not in df['Dataset Name'].values:
                    print(f"Sampling data missing dataset: {dataset}")
                    return False

            for sampling_method in sampling_methods:
                if sampling_method not in df['Sampling Method'].values:
                    print(f"Sampling data missing sampling method: {sampling_method}")
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

            required_cols = ['Random Seed', 'Sampling Method', 'Sample Size']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Landscape data missing required columns {missing_cols}: {csv_file}")
                return False

        except Exception as e:
            print(f"Error checking landscape feature data: {e}")
            return False

    print("All landscape feature data exists")
    return True


def generate_selected_datasets(problem_types: List[str], selected_modes: List[str],
                               sampling_methods: List[str], random_seeds: range, num_samples: int) -> List[str]:
    selected_datasets = []

    for problem_type in problem_types:
        if problem_type == 'c10mop':
            problem_ids = [1, 3, 5, 8, 10, 11, 12, 13]
            search_space_configs = c10_search_space_configs
        elif problem_type == 'citysegmop':
            problem_ids = [3]
            search_space_configs = cityseg_search_space_configs
        elif problem_type == 'in1kmop':
            problem_ids = [1, 4, 7]
            search_space_configs = in1k_search_space_configs
        else:
            continue

        for pid in problem_ids:
            pid_configs = search_space_configs.get(pid, [])
            valid_configs = [cfg for cfg in pid_configs if cfg]

            if not valid_configs:
                print(f"Warning: {problem_type}{pid} has no valid configuration, skipping")
                continue

            for config_idx in range(len(valid_configs)):
                dataset_name = f"{problem_type}{pid}_{config_idx}"
                selected_datasets.append(dataset_name)

    print(f"Generated {len(selected_datasets)} datasets based on configuration")
    print(f"Problem types: {problem_types}")
    print(f"Modes: {selected_modes}")
    print(f"Sampling methods: {sampling_methods}")
    print(f"Random seeds: {list(random_seeds)}")
    print(f"Sample size: {num_samples}")

    return selected_datasets


def coordinated_pipeline(
        problem_types=None,
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
        process_reverse=False
):
    if problem_types is None:
        problem_types = ['c10mop', 'citysegmop', 'in1kmop']
    if selected_modes is None:
        selected_modes = ['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if sampling_methods is None:
        sampling_methods = ['sobol','orthogonal','stratified','latin_hypercube','monte_carlo','covering_array']
    if random_seeds is None:
        random_seeds = range(0, 10)
    if fa_construction is None:
        fa_construction = ['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
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

    selected_datasets = generate_selected_datasets(
        problem_types=problem_types,
        selected_modes=selected_modes,
        sampling_methods=sampling_methods,
        random_seeds=random_seeds,
        num_samples=num_samples
    )

    print("=" * 60)
    print("Starting coordinated data processing pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Problem types: {problem_types}")
    print(f"  Modes: {selected_modes}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {num_samples}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Number of datasets: {len(selected_datasets)}")
    print("=" * 60)

    print("\nStage 1: Check sampled data")
    sampling_data_exists = check_sampling_data_exists(selected_datasets, sampling_methods, num_samples, random_seeds)

    if not sampling_data_exists:
        print("Starting to generate sampled data...")
        main_evox_multi(
            problem_types=problem_types,
            fa_construction=['g1'],
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            use_saved_data=use_saved_data,
            debug=debug,
            first_sample=True
        )
        print("Sampled data generation completed")
    else:
        print("Sampled data exists, skipping sampling stage")

    print("\nStage 2: Check multi-objective feature data")
    multi_feature_data_exists = check_multi_feature_data_exists(selected_datasets, selected_modes, sampling_methods)

    if not multi_feature_data_exists:
        print("Starting multi-objective feature computation...")
        main_evox_multi(
            problem_types=problem_types,
            fa_construction=fa_construction,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            use_saved_data=True,
            debug=debug,
            first_sample=False
        )
        print("Multi-objective feature computation completed")
    else:
        print("Multi-objective feature data exists, skipping computation stage")

    print("\nStage 3: Check landscape feature data")
    landscape_feature_data_exists = check_landscape_feature_data_exists(selected_datasets)

    if not landscape_feature_data_exists:
        print("Starting landscape feature computation...")
        main_evox_single(
            problem_types=problem_types,
            modes=selected_modes,
            sampling_methods=sampling_methods,
            sample_size=num_samples,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            debug=debug
        )
        print("Landscape feature computation completed")
    else:
        print("Landscape feature data exists, skipping computation stage")

    print("\nStage 4: Check NSGA2 data")
    nsga2_data_exists = check_nsga2_data_exists(selected_datasets, selected_modes, random_seeds)

    if not nsga2_data_exists:
        print("Warning: Some NSGA2 data is missing. Ensure NSGA2 has been run and produced results")
        print("Proceeding with available data...")

    print("\nStage 5: Data merging and processing")

    print("Starting data merging...")

    landscape_df = read_landscape_data('./Results/real_data/', selected_datasets, start_seed,
                                       end_seed, process_reverse)
    p_df = read_nsga2_data('../../../Results/RQ1-raw-data/NAS/', selected_datasets,
                           selected_modes, process_reverse)
    combined_sampling_df = read_sampling_data('./Results/Output-draw/', selected_datasets,
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
            'ft_rank', 'time_rank', 'budget_rank'
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

        output_path = os.path.join(output_folder, 'processed_data_nas.csv')
        processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

        print("\nData summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        print("\n" + "=" * 60)
        print("Data processing pipeline completed")
        print("=" * 60)

        return processed_data
    else:
        print("No valid data combinations generated")
        return None


if __name__ == "__main__":
    processed_data = coordinated_pipeline(
        problem_types=['c10mop', 'citysegmop', 'in1kmop'],
        selected_modes=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['sobol','orthogonal','stratified','latin_hypercube','monte_carlo','covering_array'],
        random_seeds=range(0, 10),
        num_samples=1000,
        fa_construction=['penalty', 'g1', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=50,
        reverse=False,
        use_saved_data=False,
        debug=True,
        pic_types=['PMO', 'MMO'],
        process_reverse=False
    )