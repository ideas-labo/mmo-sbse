import os
import re
import warnings
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MAXIMIZATION_DATASETS_FT = [
    "7z", "Amazon", "BerkeleyDBC", "CocheEcologico", "CounterStrikeSimpleFeatureModel",
    "DSSample", "Dune", "ElectronicDrum", "HiPAcc", "Drupal",
    "JavaGC", "JHipster", "lrzip", "ModelTransformation",
    "SmartHomev2.2", "SPLSSimuelESPnP", "VideoPlayer",
    "VP9", "WebPortal", "x264", 'Polly'
]

ALGORITHM_DIR_MAP = {
    'NSGA2': os.path.abspath('../../../../Results/RQ1-raw-data/SPLT')
}

try:
    sk = importr('ScottKnottESD')
except Exception:
    raise SystemExit(1)

ALL_ALGORITHM_PRIORITY = {
    'NSGA2_non_ft_best': 1,
    'PromiseTune': 2,
    'RS': 3,
    'LINE': 4,
    'LITE': 5,
    'SWAY': 6,
}

def get_ft_from_nsga2(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            return None
        target_line = lines[-2] if len(lines) >= 2 else lines[-1]
        ft_match = re.search(r"'ft': (-?\d+\.?\d*)", target_line)
        if ft_match:
            return float(ft_match.group(1))
        return None
    except Exception:
        return None

def unify_dataset_name(dataset_name):
    if dataset_name is None:
        return None
    return dataset_name.lower()

def read_non_ft_best_nsga2_data(selected_datasets, start_seed, end_seed, output_path=None):
    nsga2_dir = ALGORITHM_DIR_MAP.get('NSGA2')
    if not os.path.exists(nsga2_dir):
        return pd.DataFrame(), pd.DataFrame()

    non_ft_modes = ['penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity', 'g1']
    all_data_rows = []

    MODE_PRIORITY_ORDER = {
        'gaussian': 2,
        'reciprocal': 1,
        'g1': 4,
        'age': 3,
        'novelty': 7,
        'diversity': 6,
        'penalty': 5
    }

    unified_selected_datasets = [unify_dataset_name(ds) for ds in selected_datasets]

    dataset_name_mapping = {}
    for dataset in selected_datasets:
        unified_name = unify_dataset_name(dataset)
        dataset_name_mapping[unified_name] = dataset

    for file in os.listdir(nsga2_dir):
        if not file.endswith('.csv'):
            continue

        file_core = file[:-4] if file.endswith(".csv") else file
        is_reverse = file_core.endswith("_reverse")
        if is_reverse:
            core_no_suffix = file_core[:-8]
        else:
            core_no_suffix = file_core

        nsga2_pattern = core_no_suffix.split('-')
        if nsga2_pattern:
            raw_dataset_name = nsga2_pattern[0]
            unified_dataset_name = unify_dataset_name(raw_dataset_name)
            next_pt = nsga2_pattern[1].split('_')
            seed = int(next_pt[0])
            mode = next_pt[1]

        if (unified_dataset_name not in unified_selected_datasets or
                not (start_seed <= seed <= end_seed) or
                mode not in non_ft_modes):
            continue

        full_path = os.path.join(nsga2_dir, file)
        raw_ft = get_ft_from_nsga2(full_path)
        if raw_ft is None:
            continue

        if unified_dataset_name in dataset_name_mapping:
            original_selected_name = dataset_name_mapping[unified_dataset_name]
        else:
            original_selected_name = raw_dataset_name

        if unified_dataset_name in [unify_dataset_name(ds) for ds in MAXIMIZATION_DATASETS_FT] and not is_reverse:
            processed_ft = -raw_ft
        else:
            processed_ft = raw_ft

        if is_reverse:
            dataset_display_name = f"{original_selected_name}_reverse"
        else:
            dataset_display_name = original_selected_name

        all_data_rows.append({
            'Dataset Name': dataset_display_name,
            'Unified Dataset Name': f"{unified_dataset_name}_reverse" if is_reverse else unified_dataset_name,
            'Original Dataset': original_selected_name,
            'Raw Dataset Name': raw_dataset_name,
            'Random Seed': seed,
            'mode': mode,
            'ft': processed_ft,
            'is_reverse': is_reverse
        })

    if not all_data_rows:
        return pd.DataFrame(), pd.DataFrame()

    all_data_df = pd.DataFrame(all_data_rows)

    best_modes_by_dataset = {}
    ranking_results_by_dataset = {}
    skipped_datasets = []

    for unified_dataset_name in all_data_df['Unified Dataset Name'].unique():
        dataset_data = all_data_df[all_data_df['Unified Dataset Name'] == unified_dataset_name]

        if dataset_data.empty:
            continue
        dataset_display_name = dataset_data.iloc[0]['Dataset Name']

        available_modes = dataset_data['mode'].unique()
        if len(available_modes) < 2:
            skipped_datasets.append(dataset_display_name)
            continue

        available_seeds = dataset_data['Random Seed'].unique()
        if len(available_seeds) < 2:
            skipped_datasets.append(dataset_display_name)
            continue

        pivot_df = dataset_data.pivot_table(
            index='Random Seed',
            columns='mode',
            values='ft',
            aggfunc='first'
        )

        for mode in non_ft_modes:
            if mode not in pivot_df.columns:
                pivot_df[mode] = np.nan

        pivot_df = pivot_df[non_ft_modes]
        pivot_df = pivot_df.dropna(axis=1, how='all')

        if pivot_df.empty or len(pivot_df.columns) < 2:
            skipped_datasets.append(dataset_display_name)
            continue

        pivot_df_clean = pivot_df.dropna()

        if pivot_df_clean.empty or len(pivot_df_clean) < 2:
            skipped_datasets.append(dataset_display_name)
            continue

        min_val = pivot_df_clean.min().min()
        max_val = pivot_df_clean.max().max()

        if max_val - min_val < 1e-12:
            normalized_df = pivot_df_clean.copy()
            normalized_df[:] = 0.5
        else:
            normalized_df = (pivot_df_clean - min_val) / (max_val - min_val)

        try:
            with localconverter(default_converter + pandas2ri.converter):
                r_wide = ro.conversion.py2rpy(normalized_df)

            r_sk_result = sk.sk_esd(r_wide, version="np")
            groups = list(ro.r('as.integer')(r_sk_result[1]))
            mode_names = list(normalized_df.columns)

            group_stats = {}
            for mode, group in zip(mode_names, groups):
                if group not in group_stats:
                    group_stats[group] = []
                group_stats[group].append(mode)

            group_means = {}
            for group, modes in group_stats.items():
                group_mean = 0
                for mode in modes:
                    group_mean += normalized_df[mode].mean()
                group_mean /= len(modes)
                group_means[group] = group_mean

            sorted_groups = sorted(group_means.keys(), key=lambda x: group_means[x])
            group_rank = {g: i + 1 for i, g in enumerate(sorted_groups)}

            all_modes_ranking = []

            for group_rank_num, group in enumerate(sorted_groups, 1):
                modes_in_group = group_stats[group]

                mode_medians = {}
                for mode in modes_in_group:
                    mode_data = dataset_data[dataset_data['mode'] == mode]['ft']
                    mode_medians[mode] = mode_data.median()

                if len(modes_in_group) == 1:
                    mode = modes_in_group[0]
                    all_modes_ranking.append({
                        'mode': mode,
                        'group': group,
                        'group_rank': group_rank_num,
                        'median': mode_medians.get(mode, np.nan),
                        'priority': MODE_PRIORITY_ORDER.get(mode, float('inf'))
                    })
                else:
                    modes_with_stats = []
                    for mode in modes_in_group:
                        modes_with_stats.append({
                            'mode': mode,
                            'median': mode_medians.get(mode, np.nan),
                            'priority': MODE_PRIORITY_ORDER.get(mode, float('inf'))
                        })

                    modes_with_stats.sort(key=lambda x: (x['median'], x['priority']))

                    for mode_info in modes_with_stats:
                        all_modes_ranking.append({
                            'mode': mode_info['mode'],
                            'group': group,
                            'group_rank': group_rank_num,
                            'median': mode_info['median'],
                            'priority': mode_info['priority']
                        })

            dataset_ranking_results = []
            for i, mode_info in enumerate(all_modes_ranking, 1):
                mode = mode_info['mode']

                dataset_ranking_results.append({
                    'Dataset Name': dataset_display_name,
                    'mode': mode,
                    'unique_rank': i,
                    'group_rank': mode_info['group_rank'],
                    'is_best': (i == 1)
                })

            ranking_results_by_dataset[dataset_display_name] = pd.DataFrame(dataset_ranking_results)
            best_modes_by_dataset[dataset_display_name] = all_modes_ranking[0]['mode']

        except Exception:
            skipped_datasets.append(dataset_display_name)
            continue

    if not best_modes_by_dataset:
        return pd.DataFrame(), pd.DataFrame()

    all_ranking_results = []
    for dataset_name, ranking_df in ranking_results_by_dataset.items():
        all_ranking_results.append(ranking_df)

    complete_ranking_df = pd.concat(all_ranking_results, ignore_index=True)

    best_rows = []

    for dataset_display_name in best_modes_by_dataset.keys():
        dataset_data = all_data_df[all_data_df['Dataset Name'] == dataset_display_name]
        best_mode = best_modes_by_dataset[dataset_display_name]

        best_mode_data = dataset_data[dataset_data['mode'] == best_mode]

        for seed in dataset_data['Random Seed'].unique():
            seed_data = best_mode_data[best_mode_data['Random Seed'] == seed]

            if not seed_data.empty:
                best_row = seed_data.iloc[0].copy()
                best_row['mode'] = 'NSGA2_non_ft_best'
                best_rows.append(best_row)
            else:
                median_val = dataset_data[dataset_data['mode'] == best_mode]['ft'].median()

                if not pd.isna(median_val):
                    default_row = {
                        'Dataset Name': dataset_display_name,
                        'Unified Dataset Name': dataset_data['Unified Dataset Name'].iloc[
                            0] if not dataset_data.empty else '',
                        'Original Dataset': dataset_data['Original Dataset'].iloc[
                            0] if not dataset_data.empty else dataset_display_name.replace('_reverse', ''),
                        'Random Seed': seed,
                        'mode': 'NSGA2_non_ft_best',
                        'ft': median_val,
                        'is_reverse': '_reverse' in dataset_display_name
                    }
                    best_rows.append(default_row)

    if output_path and not complete_ranking_df.empty:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            complete_ranking_df[['Dataset Name', 'mode', 'unique_rank', 'group_rank', 'is_best']].to_csv(
                output_path, index=False, encoding='utf-8'
            )
        except Exception:
            pass

    if best_rows:
        best_modes_data = pd.DataFrame(best_rows)
        columns_to_keep = ['Dataset Name', 'Random Seed', 'mode', 'ft', 'is_reverse']
        best_modes_data = best_modes_data[columns_to_keep]
    else:
        best_modes_data = pd.DataFrame()

    return best_modes_data, complete_ranking_df

def main():
    config = {
        'start_seed': 0,
        'end_seed': 9,
        'selected_datasets': ["7z", "Amazon", "BerkeleyDBC", "CocheEcologico", "CounterStrikeSimpleFeatureModel",
                              "DSSample", "Dune", "ElectronicDrum", "HiPAcc", "Drupal",
                              "JavaGC", "JHipster", "lrzip", "ModelTransformation",
                              "SmartHomev2.2", "SPLSSimuelESPnP", "VideoPlayer",
                              "VP9", "WebPortal", "x264", 'Polly'],
        'output_folder': os.path.abspath('../../../../Results/Predict-raw-data/Ranking'),
        'ranking_filename': 'non_ft_modes_ranking_splt.csv',
    }

    os.makedirs(config['output_folder'], exist_ok=True)

    ranking_output_path = os.path.join(config['output_folder'], config['ranking_filename'])

    non_ft_best_data, non_ft_ranking_data = read_non_ft_best_nsga2_data(
        selected_datasets=config['selected_datasets'],
        start_seed=config['start_seed'],
        end_seed=config['end_seed'],
        output_path=ranking_output_path
    )

    if non_ft_best_data.empty:
        return

if __name__ == "__main__":
    main()