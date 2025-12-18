import pandas as pd
import os
import re
import warnings
import numpy as np
from typing import List, Dict, Set, Tuple, Any
import sys

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

from Code.NRP.Feature.multi_feature import main_nrp_multi
from Code.NRP.Feature.single_feature import main_nrp_single


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
    base_dir = "../../../Results/RQ1-raw-data/NRP"

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
    mode = parts[2]
    dataset_name = parts[0]
    seed = parts[1]
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

            budget_line = lines[-6].strip()
            budget_match = re.search(r'budget_used:(\d+)', budget_line)
            if budget_match:
                budget = int(budget_match.group(1))
            else:
                print(f"Unable to extract budget value from file {csv_path}; please check file format.")
                budget = None

            time_line = lines[-5].strip()
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

                if 'Name' in df.columns and 'Dataset Name' not in df.columns:
                    df = df.rename(columns={'Name': 'Dataset Name'})

                df = df[(df['Random Seed'] >= start_seed) & (df['Random Seed'] <= end_seed)]

                sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo',
                                    'covering_array', 'halton']
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
            print(f"Landscape median data shape: {median_df.shape}")
            return median_df
        else:
            print("No numeric columns found in landscape data")
            return landscape_df[group_cols]
    else:
        print("No landscape data found")
        return pd.DataFrame()


def read_nsga2_data(nsga2_csv_dir, selected_datasets, selected_modes, process_reverse=False):
    p_data = []
    total_files = 0
    valid_files = 0
    for file in os.listdir(nsga2_csv_dir):
        if file.endswith('.csv'):
            total_files += 1
            file_name = os.path.basename(file)
            dataset_name, mode, is_reverse, seed = extract_info_from_filename(file_name, process_reverse)

            if dataset_name in selected_datasets and mode in selected_modes:
                if process_reverse:
                    dataset_name = dataset_name + ('_reverse' if is_reverse else '')

                csv_path = os.path.join(nsga2_csv_dir, file)
                best_p, p_values_mean, ft, budget, time = get_pareto_ratios(csv_path)
                if best_p is not None and p_values_mean is not None and budget is not None and time is not None:
                    valid_files += 1
                    p_data.append({
                        'Random Seed': int(seed),
                        'Dataset Name': dataset_name,
                        'Best_Pareto_Ratio': best_p,
                        'Pareto_Ratios_Mean': p_values_mean,
                        'mode': mode,
                        'ft': ft,
                        'budget': budget,
                        'time': time,
                        'is_reverse': is_reverse if process_reverse else False
                    })

    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_files}")

    if p_data:
        p_df = pd.DataFrame(p_data)
        group_cols = ['Dataset Name', 'mode']
        if process_reverse:
            group_cols.append('is_reverse')
        numeric_cols = ['Best_Pareto_Ratio', 'Pareto_Ratios_Mean', 'ft', 'budget', 'time', 'Random Seed']

        median_df = p_df.groupby(group_cols, as_index=False)[numeric_cols].median()
        return median_df
    else:
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

                    sampling_methods = ['sobol', 'orthogonal', 'halton', 'stratified', 'latin_hypercube', 'monte_carlo',
                                        'covering_array']
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
            print(f"{pic_type} sampling median data shape: {median_df.shape}")

    if all_sampling_dfs:
        merge_keys = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
        if process_reverse:
            merge_keys.append('is_reverse')

        combined_sampling_df = all_sampling_dfs[0]
        for df in all_sampling_dfs[1:]:
            combined_sampling_df = combined_sampling_df.merge(df, on=merge_keys, how='inner')
        print(f"Combined sampling median data shape: {combined_sampling_df.shape}")
        return combined_sampling_df
    else:
        print("No sampling data found")
        return pd.DataFrame()


def add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only', process_reverse=False):
    ranked_df = p_df.copy()

    ranked_df['ft_rank'] = 1
    ranked_df['time_rank'] = 1
    ranked_df['budget_rank'] = 1

    group_columns = ['Dataset Name', 'Random Seed']
    if process_reverse:
        group_columns.append('is_reverse')

    for group_key, group in ranked_df.groupby(group_columns):
        dataset = group_key[0]
        seed = group_key[1]
        is_reverse = group_key[2] if process_reverse else False

        if process_reverse and is_reverse:
            should_maximize = dataset.replace('_reverse', '') in reverse_maximize_datasets
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


def process_three_datasets(landscape_df, p_df, combined_sampling_df, selected_datasets, selected_modes,
                           maximize_datasets, reverse_maximize_datasets, ranking_mode='ft_only', process_reverse=False):
    p_df = add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode, process_reverse)

    required_landscape_cols = ['Dataset Name', 'Sample Size', 'Sampling Method']
    required_sampling_cols = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']

    landscape_missing = [col for col in required_landscape_cols if col not in landscape_df.columns]
    sampling_missing = [col for col in required_sampling_cols if col not in combined_sampling_df.columns]

    if landscape_missing:
        print(f"Landscape data missing required columns: {landscape_missing}")
        return None
    if sampling_missing:
        print(f"Sampling data missing required columns: {sampling_missing}")
        return None

    if 'Sampling Method' in landscape_df.columns and 'Sampling Method' in combined_sampling_df.columns and 'Sample Size' in landscape_df.columns and 'Sample Size' in combined_sampling_df.columns:
        sampling_methods = pd.concat(
            [landscape_df['Sampling Method'], combined_sampling_df['Sampling Method']]).unique()
        sampling_sizes = combined_sampling_df['Sample Size'].unique()

        combined_dfs = []
        all_selected_datasets = [ds + '_reverse' for ds in
                                 selected_datasets] + selected_datasets if process_reverse else selected_datasets

        for dataset_name in all_selected_datasets:
            for sampling_method in sampling_methods:
                for sampling_size in sampling_sizes:
                    landscape_filtered = landscape_df[(landscape_df['Sampling Method'] == sampling_method) & (
                            landscape_df['Sample Size'] == sampling_size) & (
                                                              landscape_df['Dataset Name'] == dataset_name)].copy()
                    for mode in selected_modes:
                        sampling_filtered = combined_sampling_df[
                            (combined_sampling_df['Sampling Method'] == sampling_method) & (
                                    combined_sampling_df['Sample Size'] == sampling_size) & (
                                    combined_sampling_df['Dataset Name'] == dataset_name) & (
                                    combined_sampling_df['mode'] == mode)].copy()

                        column_mapping = {'Sample Size': 'Sample Size'}
                        for old_col, new_col in column_mapping.items():
                            if old_col in landscape_filtered.columns and new_col in sampling_filtered.columns:
                                landscape_filtered.rename(columns={old_col: new_col}, inplace=True)

                        landscape_filtered = landscape_filtered.sort_values(
                            by=['Dataset Name', 'Sample Size'])
                        sampling_filtered = sampling_filtered.sort_values(
                            by=['Dataset Name', 'mode', 'Sample Size'])

                        landscape_filtered = landscape_filtered.reset_index(drop=True)
                        sampling_filtered = sampling_filtered.reset_index(drop=True)

                        combined_df = pd.concat([landscape_filtered, sampling_filtered], axis=1)

                        p_df_filtered = p_df[(p_df['Dataset Name'] == dataset_name) & (p_df['mode'] == mode)].copy()
                        p_df_filtered = p_df_filtered.sort_values(by=['Random Seed'])
                        p_df_filtered = p_df_filtered.reset_index(drop=True)

                        rank_p_cols = ['ft_rank', 'time_rank', 'budget_rank', 'Best_Pareto_Ratio', 'Pareto_Ratios_Mean']
                        combined_df = pd.concat([combined_df, p_df_filtered.drop(columns=rank_p_cols, errors='ignore')],
                                                axis=1)

                        combined_dfs.append(combined_df)

        if combined_dfs:
            all_combined_df = pd.concat(combined_dfs, ignore_index=True)
            print(f"All combined data shape: {all_combined_df.shape}")

            columns_to_keep = ['Dataset Name', 'mode', 'Sample Size', 'Sampling Method']
            if process_reverse:
                columns_to_keep.append('is_reverse')

            numeric_columns = all_combined_df.select_dtypes(include=['number']).columns
            columns_to_keep.extend([col for col in numeric_columns if col not in columns_to_keep])

            X_numeric = all_combined_df[columns_to_keep].dropna(axis=1, how='all')

            X_numeric = X_numeric.loc[:, ~X_numeric.columns.duplicated()]
            X_numeric = X_numeric.reset_index(drop=True)
            print(f"X_numeric data shape: {X_numeric.shape}")

            X_numeric = X_numeric[
                X_numeric['Dataset Name'].isin(p_df['Dataset Name']) &
                X_numeric['mode'].isin(p_df['mode'])].reset_index(drop=True)

            p_df_filtered = p_df[
                p_df['Dataset Name'].isin(X_numeric['Dataset Name']) &
                p_df['mode'].isin(X_numeric['mode'])].reset_index(drop=True)

            print(f"Filtered X_numeric data shape: {X_numeric.shape}")
            print(f"Filtered p_df data shape: {p_df_filtered.shape}")

            optimal_p_data = []
            for dataset in all_selected_datasets:
                dataset_df = p_df_filtered[p_df_filtered['Dataset Name'] == dataset]
                for seed in dataset_df['Random Seed'].unique():
                    seed_df = dataset_df[dataset_df['Random Seed'] == seed]

                    is_reverse = dataset.endswith("_reverse") if process_reverse else False
                    base_dataset = dataset.replace("_reverse", "") if process_reverse else dataset

                    if is_reverse:
                        should_maximize = base_dataset in reverse_maximize_datasets
                    else:
                        should_maximize = base_dataset in maximize_datasets

                    if ranking_mode == 'ft_only':
                        if should_maximize:
                            optimal_ft = seed_df['ft'].max()
                            optimal_rows = seed_df[seed_df['ft'] == optimal_ft]
                        else:
                            optimal_ft = seed_df['ft'].min()
                            optimal_rows = seed_df[seed_df['ft'] == optimal_ft]

                        for _, optimal_row in optimal_rows.iterrows():
                            optimal_best_p = optimal_row['Best_Pareto_Ratio']
                            optimal_p_mean = optimal_row['Pareto_Ratios_Mean']

                            for _, row in seed_df.iterrows():
                                mode = row['mode']
                                best_p = row['Best_Pareto_Ratio']
                                p_mean = row['Pareto_Ratios_Mean']
                                ft_rank = row['ft_rank']
                                time_rank = row['time_rank']
                                budget_rank = row['budget_rank']

                                percent_diff_best_p = abs(best_p - optimal_best_p)
                                percent_diff_p_mean = abs(p_mean - optimal_p_mean)

                                optimal_p_data.append({
                                    'Random Seed': seed,
                                    'Dataset Name': dataset,
                                    'mode': mode,
                                    'Optimal_Best_Pareto_Ratio': optimal_best_p,
                                    'Optimal_Pareto_Ratios_Mean': optimal_p_mean,
                                    'Best_Pareto_Ratio': best_p,
                                    'Pareto_Ratios_Mean': p_mean,
                                    'Percent_Diff_Best_P': percent_diff_best_p,
                                    'Percent_Diff_P_Mean': percent_diff_p_mean,
                                    'ft_rank': ft_rank,
                                    'time_rank': time_rank,
                                    'budget_rank': budget_rank
                                })

            optimal_p_df = pd.DataFrame(optimal_p_data)

            merge_columns = ['Dataset Name', 'mode', 'Random Seed']
            if process_reverse:
                merge_columns.append('is_reverse')

            final_df = pd.merge(X_numeric, optimal_p_df, on=merge_columns, how='left')

            if 'is_reverse' in final_df.columns and not process_reverse:
                final_df = final_df.drop(columns=['is_reverse'])

            print(f"Final data shape: {final_df.shape}")
            return final_df

    print("Sampling Method and Sample Size columns do not match, cannot proceed.")
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
        print(f"\nColumns meeting NaN criteria will be dropped (total {len(columns_to_drop)} columns):")
        for col in columns_to_drop:
            print(f"- {col} (NaN count: {column_nan_counts[col]})")
        return df.drop(columns=columns_to_drop)
    else:
        print("\nNo columns met the NaN criteria (no columns dropped).")
        return df


def coordinated_pipeline_nrp(
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
        selected_datasets = ['nrp1', 'nrp2', 'nrp3', 'nrp4', 'nrp5',
                             'nrp-e1', 'nrp-e2', 'nrp-e3', 'nrp-e4',
                             'nrp-g1', 'nrp-g2', 'nrp-g3', 'nrp-g4',
                             'nrp-m1', 'nrp-m2', 'nrp-m3', 'nrp-m4']
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
    print("Starting NRP Coordinated Data Processing Pipeline")
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

    print("\nStage 1: Check sampled data")
    sampling_data_exists = check_sampling_data_exists(selected_datasets, sampling_methods, num_samples, random_seeds)

    if not sampling_data_exists:
        print("Starting to generate sampled data...")
        main_nrp_multi(
            dataset_names=selected_datasets,
            fa_construction=['g1'],
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            workflow_base_path=workflow_base_path,
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
        main_nrp_multi(
            dataset_names=selected_datasets,
            fa_construction=fa_construction,
            minimize=True,
            fixed_sample_sizes=[num_samples],
            sampling_methods=sampling_methods,
            random_seeds=random_seeds,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
            reverse=reverse,
            workflow_base_path=workflow_base_path,
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
        main_nrp_single(
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
    nsga2_data_exists = check_nsga2_data_exists(selected_datasets, selected_modes, random_seeds)

    if not nsga2_data_exists:
        print("Warning: Some NSGA2 data is missing. Ensure NSGA2 has been run and produced results")
        print("Proceeding with available data...")

    print("\nStage 5: Data merging and processing")

    print("Starting data merging...")

    landscape_df = read_landscape_data('./Results/real_data/', selected_datasets, start_seed,
                                       end_seed, process_reverse)
    p_df = read_nsga2_data('../../../Results/RQ1-raw-data/NRP/', selected_datasets,
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
        p_df = add_ranks(p_df, maximize_datasets, reverse_maximize_datasets, ranking_mode, process_reverse)

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

        output_path = os.path.join(output_folder, 'processed_data_nrp.csv')
        processed_data.to_csv(output_path, index=False)
        print(f"Final processed data saved to: {output_path}")

        print("\nData summary:")
        print(f"Total rows: {len(processed_data)}")
        print(f"Total columns: {len(processed_data.columns)}")
        print(f"Numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical columns: {len(processed_data.select_dtypes(include=['object']).columns)}")

        print("\n" + "=" * 60)
        print("NRP Data processing pipeline completed")
        print("=" * 60)

        return processed_data
    else:
        print("No valid data combinations generated")
        return None




if __name__ == "__main__":
    processed_data = coordinated_pipeline_nrp(
        selected_datasets=['nrp1', 'nrp2', 'nrp3', 'nrp4', 'nrp5',
                         'nrp-e1', 'nrp-e2', 'nrp-e3', 'nrp-e4',
                         'nrp-g1', 'nrp-g2', 'nrp-g3', 'nrp-g4',
                         'nrp-m1', 'nrp-m2', 'nrp-m3', 'nrp-m4'],
        selected_modes=['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        sampling_methods=['sobol', 'orthogonal', 'halton', 'latin_hypercube', 'monte_carlo', 'covering_array'],
        random_seeds=range(0, 10),
        num_samples=1000,
        fa_construction=['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'],
        use_multiprocessing=True,
        max_workers=50,
        reverse=False,
        use_saved_data=False,
        debug=True,
        pic_types=['PMO', 'MMO'],
        process_reverse=False,
        workflow_base_path='../Datasets/'
    )