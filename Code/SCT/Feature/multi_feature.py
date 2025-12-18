import os
import warnings

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import qmc
import sys

sys.path.append('../')
sys.path.append('../..')
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
from Code.SCT.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
from Code.SCT.Utils.Construct_secondary_objective import generate_fa, update_novelty_archive
import multiprocessing
from Code.SCT.Utils.get_objectives import get_objective_score_with_similarity
from itertools import combinations, product

import numpy as np

def read_csv(file_path, minimize=True, reverse=False):
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(',')
        data = np.loadtxt(file_path, delimiter=',', dtype=float, skiprows=1)

        target_indices = [i for i, col in enumerate(header) if '<$' in col]
        if len(target_indices) < 1:
            raise ValueError("CSV file does not contain columns with '<$'. Please check the file.")

        independent_vars = np.delete(data, target_indices, axis=1)

        ft = data[:, target_indices[0]]
        fa = data[:, target_indices[1]] if len(target_indices) > 1 else None

        maximize_sets_ft = {'storm_wc', 'storm_rs', 'storm_sol', 'dnn_dsr', 'dnn_coffee', 'dnn_adiac', 'dnn_sa',
                            'trimesh', 'dnn_coffee', 'dnn_dsr','x264'}
        maximize_sets_fa = {}
        if any(name in file_path for name in maximize_sets_fa) and fa is not None:
            fa = -fa
        if any(name in file_path for name in maximize_sets_ft):
            ft = -ft

        if reverse and len(target_indices) > 1:
            ft, fa = fa, ft
            original_ft = data[:, target_indices[1]]
            original_fa = data[:, target_indices[0]]
        else:
            original_ft = data[:, target_indices[0]]
            original_fa = data[:, target_indices[1]] if len(target_indices) > 1 else None

        unique_elements_per_column = [list(set(col)) for col in independent_vars.T]
        return independent_vars.tolist(), ft.tolist(), ft.tolist(), fa.tolist() if fa is not None else None, header, unique_elements_per_column
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2

import csv
import numpy as np

def load_sampled_data_from_csv(dataset_name, mode, sampling_method, num_samples, random_seed, figure_type, reverse):
    if reverse:
        file_path = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        file_path = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"

    try:
        sampled_data = []
        original_ft = []
        original_fa = []
        normalized_ft = []
        normalized_fa = []

        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            feature_headers = header[:-4]

            for row in reader:
                features = list(map(float, row[:-4]))
                sampled_data.append(features)
                original_ft.append(float(row[-4]))
                original_fa_val = float(row[-3]) if row[-3] else None
                original_fa.append(original_fa_val)
                normalized_ft.append(float(row[-2]))
                normalized_fa.append(float(row[-1]))

        return (sampled_data, original_ft, original_fa,
                normalized_ft, normalized_fa, feature_headers)

    except FileNotFoundError:
        raise FileNotFoundError(f"Saved CSV file not found: {file_path}\nPlease set first_sample=True to generate data")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

import warnings
import numpy as np
import random
from itertools import combinations, product
from scipy.stats import qmc


def generate_samples(data, unique_elements_per_column, sampling_method, num_samples, random_seed, sample_type):
    np.random.seed(random_seed)
    random.seed(random_seed)

    if sample_type == 'percentage':
        target_num = int(len(data) * num_samples / 100)
    else:
        target_num = num_samples
    target_num = min(target_num, len(data))
    if target_num <= 0:
        raise ValueError(f"Invalid target sample size ({target_num}). Please check the original data or sampling parameters")

    num_features = len(data[0]) if data else 0
    is_discrete = all(isinstance(unique, (list, np.ndarray)) for unique in unique_elements_per_column)

    unique_sampled_set = set()
    final_sampled_data = []

    while len(unique_sampled_set) < target_num:
        remaining = target_num - len(unique_sampled_set)
        batch_size = max(1, int(remaining * 1.2))

        if sampling_method == 'sobol':
            try:
                warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")
                sampler = qmc.Sobol(d=num_features, scramble=True, seed=random_seed)
                if batch_size & (batch_size - 1) == 0:
                    sample = sampler.random_base2(m=int(np.log2(batch_size)))
                else:
                    sample = sampler.random(n=batch_size)

                sampled_features = []
                for i in range(num_features):
                    unique_vals = unique_elements_per_column[i]
                    if is_discrete:
                        indices = (sample[:, i] * len(unique_vals)).astype(int)
                        indices = np.clip(indices, 0, len(unique_vals) - 1)
                        sampled_features.append([unique_vals[idx] for idx in indices])
                    else:
                        min_val = min(row[i] for row in data)
                        max_val = max(row[i] for row in data)
                        sampled_features.append(sample[:, i] * (max_val - min_val) + min_val)
                batch_sampled = [list(feat) for feat in zip(*sampled_features)]

            except Exception as e:
                print(f"Sobol sampling failed, falling back to Latin Hypercube: {str(e)}")
                sampling_method = 'latin_hypercube'
                continue

        elif sampling_method == 'orthogonal':
            try:
                sampler = qmc.LatinHypercube(d=num_features, optimization="random-cd", seed=random_seed)
                sample = sampler.random(n=batch_size)
                for i in range(num_features):
                    sample[:, i] = np.argsort(np.argsort(sample[:, i])) / (batch_size - 1)

                sampled_features = []
                for i in range(num_features):
                    unique_vals = unique_elements_per_column[i]
                    if is_discrete:
                        indices = (sample[:, i] * len(unique_vals)).astype(int)
                        indices = np.clip(indices, 0, len(unique_vals) - 1)
                        sampled_features.append([unique_vals[idx] for idx in indices])
                    else:
                        min_val = min(row[i] for row in data)
                        max_val = max(row[i] for row in data)
                        sampled_features.append(sample[:, i] * (max_val - min_val) + min_val)
                batch_sampled = [list(feat) for feat in zip(*sampled_features)]

            except Exception as e:
                print(f"Orthogonal sampling failed, falling back to Latin Hypercube: {str(e)}")
                sampling_method = 'latin_hypercube'
                continue

        elif sampling_method == 'stratified':
            try:
                max_strata_per_dim = 20
                strata_per_dim = []
                for i in range(num_features):
                    unique_vals = unique_elements_per_column[i]
                    if is_discrete:
                        strata_per_dim.append(len(unique_vals) if len(unique_vals) <= max_strata_per_dim else 1)
                    else:
                        strata_per_dim.append(max_strata_per_dim)

                grid = []
                for i in range(num_features):
                    unique_vals = unique_elements_per_column[i]
                    if is_discrete:
                        grid.append(unique_vals if strata_per_dim[i] > 1 else None)
                    else:
                        min_val = min(row[i] for row in data)
                        max_val = max(row[i] for row in data)
                        grid.append(np.linspace(min_val, max_val, strata_per_dim[i]))

                dims_to_stratify = [i for i in range(num_features) if strata_per_dim[i] > 1]
                strata_combinations = list(product(*[range(strata_per_dim[dim]) for dim in dims_to_stratify]))

                batch_sampled = []
                samples_per_stratum = max(1, int(np.ceil(batch_size / len(strata_combinations))))
                for stratum in strata_combinations:
                    for _ in range(samples_per_stratum):
                        sample = []
                        for dim in range(num_features):
                            if strata_per_dim[dim] == 1:
                                if is_discrete:
                                    sample.append(random.choice(unique_elements_per_column[dim]))
                                else:
                                    min_val = min(row[dim] for row in data)
                                    max_val = max(row[dim] for row in data)
                                    sample.append(random.uniform(min_val, max_val))
                            else:
                                dim_pos = dims_to_stratify.index(dim) if dim in dims_to_stratify else -1
                                if dim_pos >= 0:
                                    stratum_idx = stratum[dim_pos]
                                    layer = grid[dim]
                                    if is_discrete:
                                        sample.append(layer[stratum_idx])
                                    else:
                                        lower = layer[stratum_idx]
                                        upper = layer[stratum_idx + 1] if stratum_idx + 1 < len(layer) else layer[stratum_idx]
                                        sample.append(random.uniform(lower, upper))
                                else:
                                    sample.append(random.choice(data)[dim])
                        batch_sampled.append(sample)
                batch_sampled = batch_sampled[:batch_size]

            except Exception as e:
                print(f"Stratified sampling failed, falling back to Latin Hypercube: {str(e)}")
                sampling_method = 'latin_hypercube'
                continue

        elif sampling_method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=num_features)
            sample = sampler.random(n=batch_size)

            sampled_features = []
            for i in range(num_features):
                unique_vals = unique_elements_per_column[i]
                if is_discrete:
                    indices = (sample[:, i] * len(unique_vals)).astype(int)
                    indices = np.clip(indices, 0, len(unique_vals) - 1)
                    sampled_features.append([unique_vals[idx] for idx in indices])
                else:
                    min_val = min(row[i] for row in data)
                    max_val = max(row[i] for row in data)
                    sampled_features.append(sample[:, i] * (max_val - min_val) + min_val)
            batch_sampled = [list(feat) for feat in zip(*sampled_features)]

        elif sampling_method == 'monte_carlo':
            if len(data) < batch_size:
                sampled_indices = np.random.choice(len(data), len(data), replace=False)
                batch_sampled = [data[i] for i in sampled_indices]
                remaining_batch = batch_size - len(batch_sampled)
                if remaining_batch > 0:
                   add_indices = np.random.choice(len(data), remaining_batch, replace=True)
                   batch_sampled.extend([data[i] for i in add_indices])
            else:
                sampled_indices = np.random.choice(len(data), batch_size, replace=False)
                batch_sampled = [data[i] for i in sampled_indices]

        elif sampling_method == 'covering_array' and is_discrete:
            def generate_t2_covering(features, unique_vals_list, n_samples):
                dim_pairs = list(combinations(range(len(unique_vals_list)), 2))
                required_pairs = set()
                for dim1, dim2 in dim_pairs:
                    required_pairs.update(product(unique_vals_list[dim1], unique_vals_list[dim2]))

                selected = []
                remaining_pairs = required_pairs.copy()
                while len(selected) < n_samples and remaining_pairs:
                    target_pair = random.choice(list(remaining_pairs))
                    target_dims = random.choice(dim_pairs)
                    candidate = list(random.choice(features))
                    candidate[target_dims[0]] = target_pair[0]
                    candidate[target_dims[1]] = target_pair[1]
                    selected.append(candidate)
                    new_covered = {(candidate[target_dims[0]], candidate[target_dims[1]])}
                    remaining_pairs -= new_covered
                while len(selected) < n_samples:
                    selected.append(random.choice(features))
                return selected

            try:
                batch_sampled = generate_t2_covering(data, unique_elements_per_column, batch_size)
            except Exception as e:
                raise ValueError(f"Covering sampling failed: {str(e)}")

        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method} (ensure features are discrete for covering_array)")

        for sample in batch_sampled:
            sample_tuple = tuple(sample)
            if sample_tuple not in unique_sampled_set:
                unique_sampled_set.add(sample_tuple)
                final_sampled_data.append(sample)
                if len(unique_sampled_set) == target_num:
                    break
            if len(unique_sampled_set) == target_num:
                break

    return final_sampled_data

def process_g1_mode(data_orig, dict_search, header, unique_elements_per_column,
                       minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, mode,
                       unique_id, reverse, first_sample=False):
    if first_sample:
        sample_and_save(data_orig, dict_search, header, unique_elements_per_column,
                        minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse)
        return

    try:
        sampled_data, sampled_ft, sampled_fa, r0_ft, r0_fa, feature_headers = load_sampled_data_from_csv(
            dataset_name=dataset_name, mode='g1', sampling_method=sampling_method,
            num_samples=num_samples, random_seed=random_seed, figure_type='figure1', reverse=reverse
        )
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Please run this program with first_sample=True to generate sampled data before using this mode.")
        return
    except Exception as e:
        print(f"[g1] Error reading sampled data: {e}")
        return

    r0_points = np.column_stack((r0_ft, r0_fa))
    sampled_dict_fig1 = {tuple(sample): (ft_val, fa_val) for sample, ft_val, fa_val in
                         zip(sampled_data, r0_points[:, 0], r0_points[:, 1])}

    plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict_fig1, reverse)

    try:
        _, _, _, g1, g2, _ = load_sampled_data_from_csv(
            dataset_name=dataset_name, mode='g1', sampling_method=sampling_method,
            num_samples=num_samples, random_seed=random_seed, figure_type='figure2', reverse=reverse
        )
    except FileNotFoundError:
        g1, g2 = transform_points_for_figure2(r0_points)
    except Exception as e:
        print(f"[g1] Error reading figure2 data: {e}")
        g1, g2 = transform_points_for_figure2(r0_points)

    sampled_dict_fig2 = {tuple(sample): (ft_val, fa_val) for sample, ft_val, fa_val in
                         zip(sampled_data, g1, g2)}

    plot_figure2(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict_fig2, reverse)

def sample_and_save(data_orig, dict_search, header, unique_elements_per_column,
                    minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse=False):
    if sample_type == 'percentage':
        percentage = num_samples
        num_samples = int(len(data_orig) * num_samples / 100)

    sampled_data = generate_samples(data_orig, unique_elements_per_column, sampling_method,
                                    num_samples, random_seed, sample_type)
    print(f"📊 Sampling completed, unique samples: {len(sampled_data)}")

    sampled_ft = []
    sampled_fa = []
    for sample in sampled_data:
        dep_value, _ = get_objective_score_with_similarity(dict_search, sample)
        sampled_ft.append(dep_value[0])
        sampled_fa.append(dep_value[1])

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(np.column_stack((sampled_ft, sampled_fa)))
    r0_points = normalized
    g1, g2 = transform_points_for_figure2(r0_points)

    if sample_type == 'percentage':
        num_samples = percentage

    save_sampled_data_to_csv(sampled_data, sampled_ft, sampled_fa, r0_points[:, 0], r0_points[:, 1],
                             header, dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1',
                             reverse)

    save_sampled_data_to_csv(sampled_data, sampled_ft, sampled_fa, g1, g2,
                             header, dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure2',
                             reverse)

    print(f"[sample_and_save] Sampling and saving completed: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")

def process_fa_construction_mode(data_orig, dict_search, header, unique_elements_per_column,
                                 minimize, num_samples, random_seed, fa_construction, sampling_method,
                                 sample_type, dataset_name, mode, unique_id, reverse, first_sample=False):
    if first_sample:
        sample_and_save(data_orig, dict_search, header, unique_elements_per_column,
                        minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse)
        return

    try:
        sampled_data, original_ft, original_fa, _, _, _ = load_sampled_data_from_csv(
            dataset_name=dataset_name, mode='g1', sampling_method=sampling_method,
            num_samples=num_samples, random_seed=random_seed, figure_type='figure1', reverse=reverse
        )
        print(f"📥 Loaded g1 base data: {dataset_name} (seed={random_seed})")
    except FileNotFoundError:
        print(f"[FA mode] g1 base sampled data not found: {dataset_name}, {sampling_method}, {random_seed}")
        print("Please run this program with first_sample=True to generate sampled data before using this mode.")
        return
    except Exception as e:
        print(f"[FA mode] Error reading g1 data: {e}")
        return

    all_solutions = []
    for i, sample in enumerate(sampled_data):
        ft_val = original_ft[i]
        fa_val = original_fa[i] if original_fa[i] is not None else None
        solution = type('Solution', (), {
            'variables': sample,
            'objectives': [ft_val, fa_val] if fa_val is not None else [ft_val]
        })()
        all_solutions.append(solution)
    all_solutions.sort(key=lambda x: x.objectives[0])

    dataset_t_max = {
        'mariadb': 40, 'storm_sol': 24, 'storm_wc': 16, 'vp9': 46, 'storm_rs': 28,
        'lrzip': 40, 'mongodb': 50, 'llvm': 60, 'dnn_sa': 40, 'dnn_adiac': 40,
        'x264': 100, 'trimesh': 100, 'dnn_coffee': 36, 'dnn_dsr': 26
    }
    t_max = dataset_t_max.get(dataset_name, 100)
    batch_size = 20
    num_batches = (len(all_solutions) + batch_size - 1) // batch_size
    batch_indices = get_batch_indices(list(range(len(all_solutions))), batch_size, num_batches)
    t, novelty_archive, all_ft_normalized, all_fa_normalized = 1, [], [], []
    age_info = None

    for batch_num in range(num_batches):
        batch_solutions = [all_solutions[i] for i in batch_indices[batch_num]]
        batch_vars = [s.variables for s in batch_solutions]
        batch_ft = [s.objectives[0] for s in batch_solutions]

        if fa_construction == 'age':
            age_info = [batch_size + i + 1 for i in range(len(batch_solutions))] if batch_num == 0 else [batch_size + t - 1] * len(batch_solutions)
        elif fa_construction == 'novelty':
            update_novelty_archive(batch_solutions, novelty_archive)

        batch_ft_norm, batch_fa_norm = generate_fa(
            batch_vars, batch_ft, fa_construction, minimize,
            file_path=f'./Datasets/Original_data1/{dataset_name}.csv',
            unique_elements_per_column=unique_elements_per_column,
            t=t, t_max=t_max, random_seed=random_seed,
            age_info=age_info if fa_construction == 'age' else None,
            novelty_archive=novelty_archive if fa_construction in ['novelty'] else None,
            k=min(10, len(batch_solutions) // 2)
        )
        all_ft_normalized.extend(batch_ft_norm)
        all_fa_normalized.extend(batch_fa_norm)
        t += 1

    save_num_samples = num_samples if sample_type != 'percentage' else num_samples
    save_sampled_data_to_csv(
        [s.variables for s in all_solutions],
        [s.objectives[0] for s in all_solutions],
        [s.objectives[1] for s in all_solutions],
        all_ft_normalized, all_fa_normalized,
        header, dataset_name, mode, sampling_method, save_num_samples, random_seed, 'figure1', reverse
    )

    sampled_dict = {tuple(s.variables): (ft, fa) for s, ft, fa in
                   zip(all_solutions, all_ft_normalized, all_fa_normalized)}
    plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict, reverse)

    r0_points = np.column_stack((all_ft_normalized, all_fa_normalized))
    g1, g2 = transform_points_for_figure2(r0_points)
    save_sampled_data_to_csv(
        [s.variables for s in all_solutions], [s.objectives[0] for s in all_solutions],
        [s.objectives[1] for s in all_solutions], g1, g2,
        header, dataset_name, mode, sampling_method, save_num_samples, random_seed, 'figure2', reverse
    )

    sampled_dict_fig2 = {tuple(s.variables): (g1_val, g2_val) for s, g1_val, g2_val in zip(all_solutions, g1, g2)}
    plot_figure2(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict_fig2, reverse)

    del sampled_dict, sampled_dict_fig2, all_solutions, sampled_data

def save_sampled_data_to_csv(sampled_data, original_ft, original_fa, normalized_ft, normalized_fa,
                             header, dataset_name, mode, sampling_method, num_samples, random_seed, figure_type,reverse):
    if mode == 'g1':
        if reverse:
            output_filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
        else:
            output_filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    else:
        if reverse:
            output_filename = f"./Results/Samples_multi_fa/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
        else:
            output_filename = f"./Results/Samples_multi_fa/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    feature_headers = [h for h in header if '<$' not in h]
    output_headers = feature_headers + ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    rows = []
    for i, sample in enumerate(sampled_data):
        row = list(sample) + [original_ft[i], original_fa[i] if original_fa else None, normalized_ft[i],
                              normalized_fa[i]]
        rows.append(row)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(output_headers)
        writer.writerows(rows)

    print(f"Sampled data saved to: {output_filename}")

import random

def get_batch_indices(sorted_indices, batch_size, num_batches, reverse_prob=0.8):
    batch_indices = [[] for _ in range(num_batches)]

    for idx in sorted_indices:
        assigned = False
        attempts = 0

        while not assigned and attempts < 3:
            if random.random() < reverse_prob:
                preferred_batch = min(num_batches - 1, int((1 - (idx / len(sorted_indices))) * num_batches))
            else:
                preferred_batch = random.randint(0, num_batches - 1)

            if len(batch_indices[preferred_batch]) < batch_size:
                batch_indices[preferred_batch].append(idx)
                assigned = True
            attempts += 1

        if not assigned:
            for batch_idx in range(num_batches):
                if len(batch_indices[batch_idx]) < batch_size:
                    batch_indices[batch_idx].append(idx)
                    assigned = True
                    break

    return batch_indices

from typing import List, Dict, Any

class ProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.dataset_resources = {}
        return cls._instance

    @classmethod
    def preload_datasets(cls, dataset_names: List[str], file_base_path: str,
                        minimize: bool, reverse_list: List[bool]):
        instance = cls()
        for dataset in dataset_names:
            for reverse in reverse_list:
                key = (dataset, reverse)
                if key not in instance.dataset_resources:
                    try:
                        file_path = f"{file_base_path}{dataset}.csv"
                        data_orig, ft_orig, _, original_fa_all, header, unique_elements_per_column = read_csv(
                            file_path, minimize, reverse
                        )
                        dict_search = {tuple(row): (ft_orig[i], original_fa_all[i])
                                      for i, row in enumerate(data_orig)}
                        instance.dataset_resources[key] = {
                            'data_orig': data_orig,
                            'dict_search': dict_search,
                            'header': header,
                            'unique_elements_per_column': unique_elements_per_column
                        }
                        print(f"✅ Preloaded: {dataset} (reverse={reverse})")
                    except Exception as e:
                        print(f"❌ Failed to load {dataset} (reverse={reverse}): {str(e)}")
        return instance.dataset_resources

    @classmethod
    def get_dataset_resource(cls, dataset_name: str, reverse: bool) -> Dict[str, Any]:
        instance = cls()
        key = (dataset_name, reverse)
        if key not in instance.dataset_resources:
            raise ValueError(f"Preloaded resource not found: {dataset_name} (reverse={reverse})")
        return instance.dataset_resources[key]

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_in_batches(all_tasks: List[Dict[str, Any]], max_workers: int = 80, batch_size: int = 80):
    max_workers = min(multiprocessing.cpu_count(), max_workers)
    total_tasks = len(all_tasks)
    print(f"\n🚀 Starting batched processing. Total tasks: {total_tasks}, batch size: {batch_size}, max workers: {max_workers}")

    for i in range(0, total_tasks, batch_size):
        current_batch = all_tasks[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"\n=== Processing batch {batch_num} (tasks in batch: {len(current_batch)}) ===")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_task, task) for task in current_batch]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"✅ {result}")
                except Exception as e:
                    print(f"❌ Task failed: {str(e)}")

def process_single_task(task: Dict[str, Any]) -> str:
    try:
        mode = task['mode']
        dataset_name = task['dataset_name']
        sampling_method = task['sampling_method']
        num_samples = task['num_samples']
        sample_type = task['sample_type']
        minimize = task['minimize']
        random_seed = task['random_seed']
        fa_construction = task['fa_construction']
        unique_id = task['unique_id']
        reverse = task['reverse']
        first_sample = task.get('first_sample', False)

        dataset_resource = ProblemManager.get_dataset_resource(dataset_name, reverse)
        data_orig = dataset_resource['data_orig']
        dict_search = dataset_resource['dict_search']
        header = dataset_resource['header']
        unique_elements_per_column = dataset_resource['unique_elements_per_column']

        if mode == 'g1':
            process_g1_mode(
                data_orig=data_orig,
                dict_search=dict_search,
                header=header,
                unique_elements_per_column=unique_elements_per_column,
                minimize=minimize,
                num_samples=num_samples,
                random_seed=random_seed,
                sampling_method=sampling_method,
                sample_type=sample_type,
                dataset_name=dataset_name,
                mode=mode,
                unique_id=unique_id,
                reverse=reverse,
                first_sample=first_sample
            )
        elif mode in fa_construction:
            process_fa_construction_mode(
                data_orig=data_orig,
                dict_search=dict_search,
                header=header,
                unique_elements_per_column=unique_elements_per_column,
                minimize=minimize,
                num_samples=num_samples,
                random_seed=random_seed,
                fa_construction=mode,
                sampling_method=sampling_method,
                sample_type=sample_type,
                dataset_name=dataset_name,
                mode=mode,
                unique_id=unique_id,
                reverse=reverse,
                first_sample=first_sample
            )

        return f"Task completed: {mode} | {dataset_name} | seed={random_seed} | first_sample={first_sample}"
    except Exception as e:
        return f"Task failed: {mode} | {dataset_name} | seed={task.get('random_seed','?')} | Error: {str(e)}"


def main_sct_multi(dataset_names, fa_construction, minimize=True,
                   fixed_sample_sizes=[900],
                   sampling_methods=['latin_hypercube', 'sobol', 'orthogonal', 'stratified', 'monte_carlo','covering_array'],
                   random_seeds=range(0, 10),
                   use_multiprocessing=True,
                   max_workers=None,
                   reverse=False,
                   first_sample=False,
                   file_base_path='./Datasets/',
                   use_saved_data=False,
                   debug=False,
                   workflow_base_path=None):
    if workflow_base_path:
        file_base_path = workflow_base_path

    print("🔍 Starting to preload dataset resources...")
    ProblemManager.preload_datasets(dataset_names, file_base_path, minimize, reverse_list=[reverse])

    if first_sample:
        fa_construction = ["g1"]

    all_tasks = []
    for mode in fa_construction:
        for dataset in dataset_names:
            for sampling_method in sampling_methods:
                for num_sample in fixed_sample_sizes:
                    for random_seed in random_seeds:
                        if mode == 'reciprocal':
                            if not reverse and dataset in ['dnn_adiac', 'dnn_dsr', 'dnn_sa']:
                                continue
                            if reverse and dataset == 'x264':
                                continue
                        unique_id = f"{dataset}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}"
                        task = {
                            'mode': mode,
                            'dataset_name': dataset,
                            'sampling_method': sampling_method,
                            'num_samples': num_sample,
                            'sample_type': 'fixed',
                            'minimize': minimize,
                            'random_seed': random_seed,
                            'fa_construction': fa_construction,
                            'unique_id': unique_id,
                            'reverse': reverse,
                            'first_sample': first_sample
                        }
                        all_tasks.append(task)

    print(f"📋 Tasks generated: total {len(all_tasks)}")

    if use_multiprocessing:
        if max_workers is None:
            max_workers = 50
        process_in_batches(all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        print("🔄 Processing tasks in single-process mode...")
        for task in all_tasks:
            result = process_single_task(task)
            print(result)

if __name__ == "__main__":
    fa_construction = [ 'g1','penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    dataset_names = ['dnn_adiac', 'dnn_coffee', 'dnn_dsr', 'dnn_sa',
             'llvm', 'lrzip', 'mariadb', 'mongodb', 'vp9', 'x264',
             'storm_rs', 'storm_wc', 'trimesh']
    file_base_path = '../Datasets/'

    main_sct_multi(
        dataset_names=dataset_names,
        fa_construction=fa_construction,
        use_multiprocessing=True,
        reverse=True,
        first_sample=False,
        file_base_path=file_base_path
    )