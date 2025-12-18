import multiprocessing
import os
import csv
import warnings
from itertools import combinations, product

import numpy as np
from numpy import genfromtxt
from scipy.stats import qmc
import random
import concurrent.futures
import pandas as pd
from scipy.stats import ttest_ind
import sys
import os
sys.path.append('../')
sys.path.append('../..')
from Code.SCT.Feature.multi_feature import load_sampled_data_from_csv
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.classical_ela_features import *


SAMPLING_METHODS = [
    'sobol',
    'orthogonal',
    'halton',
    'latin_hypercube',
    'monte_carlo',
    'covering_array'
]


def generate_samples(data, unique_elements_per_column, sampling_method, num_samples, random_seed, sample_type,
                     reverse=False):
    np.random.seed(random_seed)
    random.seed(random_seed)

    if sample_type == 'percentage':
        num_samples = int(len(data) * num_samples / 100)
    num_samples = min(num_samples, len(data))

    target_col = -1 if reverse else -2
    feature_data = [row[:-2] for row in data]
    target_data = [row[target_col] for row in data]
    num_features = len(feature_data[0])

    if sampling_method == 'sobol':
        warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")
        sampler = qmc.Sobol(d=num_features, scramble=True, seed=random_seed)
        sample = sampler.random_base2(m=int(np.log2(num_samples))) if num_samples & (num_samples - 1) == 0 \
            else sampler.random(n=num_samples)

        sampled_features = []
        for i in range(num_features):
            unique_values = unique_elements_per_column[i]
            if isinstance(unique_values, (list, np.ndarray)):
                indices = (sample[:, i] * len(unique_values)).astype(int)
                indices = np.clip(indices, 0, len(unique_values) - 1)
                sampled_features.append([unique_values[idx] for idx in indices])
            else:
                min_val, max_val = min([row[i] for row in feature_data]), max([row[i] for row in feature_data])
                sampled_features.append(sample[:, i] * (max_val - min_val) + min_val)

        sampled_features = list(zip(*sampled_features))
        target_indices = np.random.choice(len(target_data), num_samples)
        return [list(features) + [target_data[idx]] for features, idx in zip(sampled_features, target_indices)]

    elif sampling_method == 'orthogonal':
        sampler = qmc.LatinHypercube(d=num_features, optimization="random-cd", seed=random_seed)
        sample = sampler.random(n=num_samples)
        for i in range(num_features):
            sample[:, i] = np.argsort(np.argsort(sample[:, i])) / (num_samples - 1)

        sampled_features = []
        for i in range(num_features):
            unique_values = unique_elements_per_column[i]
            if isinstance(unique_values, (list, np.ndarray)):
                indices = (sample[:, i] * len(unique_values)).astype(int)
                indices = np.clip(indices, 0, len(unique_values) - 1)
                sampled_features.append([unique_values[idx] for idx in indices])
            else:
                min_val, max_val = min([row[i] for row in feature_data]), max([row[i] for row in feature_data])
                sampled_features.append(sample[:, i] * (max_val - min_val) + min_val)

        sampled_features = list(zip(*sampled_features))
        target_indices = np.random.choice(len(target_data), num_samples)
        return [list(features) + [target_data[idx]] for features, idx in zip(sampled_features, target_indices)]

    if sampling_method == 'stratified':
        try:
            max_strata_per_dim = 20
            strata_per_dim = []
            for i in range(num_features):
                unique_values = unique_elements_per_column[i]
                if isinstance(unique_values, (list, np.ndarray)):
                    if len(unique_values) > max_strata_per_dim:
                        strata_per_dim.append(1)
                    else:
                        strata_per_dim.append(len(unique_values))
                else:
                    strata_per_dim.append(max_strata_per_dim)

            grid = []
            for i in range(num_features):
                unique_values = unique_elements_per_column[i]
                if isinstance(unique_values, (list, np.ndarray)):
                    if strata_per_dim[i] == 1:
                        grid.append(None)
                    else:
                        grid.append(unique_values)
                else:
                    min_val = min([row[i] for row in feature_data])
                    max_val = max([row[i] for row in feature_data])
                    grid.append(np.linspace(min_val, max_val, strata_per_dim[i]))

            dims_to_stratify = [i for i in range(num_features) if strata_per_dim[i] > 1]
            strata_combinations = list(product(*[range(strata_per_dim[dim]) for dim in dims_to_stratify]))
            samples_per_stratum = max(1, int(np.ceil(num_samples / len(strata_combinations))))
            sampled_features = []

            for stratum in strata_combinations:
                for _ in range(samples_per_stratum):
                    sample = []
                    for dim in range(num_features):
                        if strata_per_dim[dim] == 1:
                            if isinstance(unique_elements_per_column[dim], (list, np.ndarray)):
                                sample.append(random.choice(unique_elements_per_column[dim]))
                            else:
                                min_val = min([row[dim] for row in feature_data])
                                max_val = max([row[dim] for row in feature_data])
                                sample.append(random.uniform(min_val, max_val))
                        else:
                            dim_pos = dims_to_stratify.index(dim) if dim in dims_to_stratify else -1
                            if dim_pos >= 0:
                                stratum_idx = stratum[dim_pos]
                                layer = grid[dim]
                                if isinstance(layer, (list, np.ndarray)):
                                    sample.append(layer[stratum_idx])
                                else:
                                    lower = layer[stratum_idx]
                                    upper = layer[stratum_idx + 1] if stratum_idx + 1 < len(layer) else layer[stratum_idx]
                                    sample.append(random.uniform(lower, upper))
                            else:
                                sample.append(random.choice(feature_data)[dim])

                    sampled_features.append(sample)

            sampled_features = random.sample(sampled_features, min(num_samples, len(sampled_features)))
            target_indices = np.random.choice(len(target_data), len(sampled_features))
            return [list(features) + [target_data[idx]] for features, idx in zip(sampled_features, target_indices)]

        except Exception as e:
            print(f"Stratified sampling failed, falling back to Latin Hypercube: {str(e)}")
            sampling_method = 'latin_hypercube'

    elif sampling_method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(d=num_features, seed=random_seed)
        sample = sampler.random(n=num_samples)

        sampled_features = []
        for i in range(num_features):
            unique_values = unique_elements_per_column[i]
            if isinstance(unique_values, (list, np.ndarray)):
                indices = (sample[:, i] * len(unique_values)).astype(int)
                indices = np.clip(indices, 0, len(unique_values) - 1)
                sampled_features.append([unique_values[idx] for idx in indices])
            else:
                min_val, max_val = min([row[i] for row in feature_data]), max([row[i] for row in feature_data])
                sampled_features.append(sample[:, i] * (max_val - min_val) + min_val)

        sampled_features = list(zip(*sampled_features))
        target_indices = np.random.choice(len(target_data), num_samples)
        return [list(features) + [target_data[idx]] for features, idx in zip(sampled_features, target_indices)]

    elif sampling_method == 'monte_carlo':
        indices = np.random.choice(len(data), num_samples, replace=False)
        if reverse:
            sampled_data = [list(data[i][:-2]) + [data[i][-1]] for i in indices]
        else:
            sampled_data = [list(data[i][:-2]) + [data[i][-2]] for i in indices]
        return sampled_data

    elif sampling_method == 'covering_array':
        def generate_t2_covering(features, unique_values, n_samples):
            dim_pairs = list(combinations(range(len(unique_values)), 2))
            required_pairs = set()
            for dim1, dim2 in dim_pairs:
                required_pairs.update(product(unique_values[dim1], unique_values[dim2]))

            selected = []
            remaining_pairs = required_pairs.copy()

            while len(selected) < n_samples and remaining_pairs:
                target_val1, target_val2 = random.choice(list(remaining_pairs))
                target_dims = random.choice([d for d in dim_pairs])
                candidate = list(random.choice(features))
                candidate[target_dims[0]] = target_val1
                candidate[target_dims[1]] = target_val2

                new_covered = set()
                for (v1, v2) in remaining_pairs:
                    if (candidate[target_dims[0]] == v1 and
                            candidate[target_dims[1]] == v2):
                        new_covered.add((v1, v2))

                selected.append(candidate)
                remaining_pairs -= new_covered

            while len(selected) < n_samples:
                selected.append(random.choice(features))

            return selected[:n_samples]

        try:
            sampled_features = generate_t2_covering(
                feature_data,
                unique_elements_per_column,
                num_samples
            )
            target_indices = np.random.choice(len(target_data), len(sampled_features))
            return [list(sampled_features[i]) + [target_data[idx]] for i, idx in enumerate(target_indices)]

        except Exception as e:
            raise ValueError(f"覆盖采样失败: {str(e)} (建议检查输入数据维度或唯一值数量)")

    else:
        raise ValueError(f"不支持的采样方法: {sampling_method}")


def auto_correlation_at(x, N, lag):
    s = sum(x)
    mean = s / N
    start = max(0, -lag)
    limit = min(N, len(x))
    limit = min(limit, len(x) - lag)
    sum_corr = 0.0
    for i in range(start, limit):
        sum_corr += (x[i] - mean) * (x[i + lag] - mean)
    return sum_corr / (N - lag)


def euclidean_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")
    sum_squared_diff = sum((a - b) ** 2 for a, b in zip(list1, list2))
    distance = np.sqrt(sum_squared_diff)
    return distance


def calculate_sd(num_array):
    sum_num = sum(num_array)
    mean = sum_num / len(num_array)
    variance = sum((num - mean) ** 2 for num in num_array) / len(num_array)
    return np.sqrt(variance)


def hamming_dist(str1, str2):
    return sum(a != b for a, b in zip(str1, str2))


def find_value_indices(lst, Min=True):
    if Min:
        min_value = min(lst)
    else:
        min_value = max(lst)
    min_indices = [index for index, value in enumerate(lst) if value == min_value]
    return min_indices


class landscape():
    def __init__(self, map, Min, df, last_column_values, lower_bound, upper_bound) -> None:
        self.map = map
        self.populations = list(map.keys())
        self.preds = list(map.values())
        self.Min = Min
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.df = df
        self.last_column_values = last_column_values
        self.best = find_value_indices(self.preds, Min=self.Min)
        self.best_populations = [self.populations[i] for i in self.best]
        self.ela_distr = calculate_ela_distribution(df, last_column_values)
        self.ic = None

    def _get_auto_correlation(self):
        list_keys = self.populations[:]
        base = self.best_populations

        def _min_distance(x, base):
            dis = []
            for single_base in base:
                x_best = single_base
                if hamming_dist(x_best, x) != 0:
                    dis.append(hamming_dist(x_best, x))
            if dis:
                return min(dis)
            else:
                return 1e8

        list_keys.sort(key=lambda x: _min_distance(x, base))
        total = 0
        size = 50
        for _ in range(size):
            sub_list = base[:]
            first = random.choice(base)
            p = random.randint(0, len(list_keys) - 1)
            for _ in range(50):
                k = 1
                temp = []
                while len(temp) == 0 or len(temp) == 1:
                    temp = [list_keys[i] for i in range(len(list_keys)) if hamming_dist(list_keys[i], first) <= k]
                    k += 1
                p = random.randint(0, len(temp) - 1)
                sub_list.append(temp[p])
                first = temp[p]
            data = [self.map[key] for key in sub_list]
            r = auto_correlation_at(data, len(data), 1)
            std = calculate_sd(data)
            if std == 0:
                size = size - 1
                continue
            total += r / (std * std)
        if total == 0 and size == 0:
            result = 0
        else:
            result = total / size
        return result

    def calculate_correlation_length(self):
        d = self._get_auto_correlation()
        if d == 0 or abs(d) == 1:
            return "nan"
        return (1 / np.log(abs(d))) * -1.0

    def calculate_best_distance(self, real_bests):
        base = self.best_populations
        dis = []
        for real_best in real_bests:
            for single_list in base:
                dis.append(hamming_dist(single_list, real_best))
        return min(dis)

    def calculate_Proportion_of_local_optimal(self, unique_elements_per_column):
        def _is_better(x_value, y_value, minimization):
            if minimization:
                return x_value <= y_value
            else:
                return x_value >= y_value

        def _find_k_neighbors(target, configs, k=5):
            distances = []
            for config in configs:
                if config == target:
                    continue
                distance = sum(1 for x, y in zip(target, config) if x != y)
                distances.append((config, distance))
            distances.sort(key=lambda x: x[1])
            return [config for config, _ in distances[:k]]

        dim = len(self.populations[0]) if self.populations else 0
        k = min(max(3, dim), 10)
        local_optima_count = 0
        configs = list(self.map.keys())

        for config in configs:
            neighbors = _find_k_neighbors(config, configs, k)
            is_local_optima = True
            for neighbor in neighbors:
                if not _is_better(self.map[config], self.map[neighbor], self.Min):
                    is_local_optima = False
                    break
            if is_local_optima:
                local_optima_count += 1

        return local_optima_count / len(configs) if configs else 0

    def calculate_FDC(self):
        fdc = calculate_fitness_distance_correlation(self.df, self.last_column_values, minimize=self.Min)
        return fdc['fitness_distance.fd_correlation']

    def calculate_skewness(self):
        return self.ela_distr['ela_distr.skewness']

    def calculate_kurtosis(self):
        return self.ela_distr['ela_distr.kurtosis']

    def calculate_h_max(self):
        ic = calculate_information_content(self.df, self.last_column_values, seed=42)
        self.ic = ic
        return ic['ic.h_max']

    def calculate_eps_s(self):
        if self.ic:
            return self.ic['ic.eps_s']
        else:
            return calculate_information_content(self.df, self.last_column_values, seed=42)['ic.eps_s']

    def calculate_NBC(self):
        nbc = calculate_nbc(self.df, self.last_column_values)
        return nbc['nbc.nn_nb.mean_ratio']


def deduplicate_samples(sampled_data):
    seen = set()
    deduplicated = []
    for sample in sampled_data:
        features = tuple(sample[:-1])
        if features not in seen:
            seen.add(features)
            deduplicated.append(sample)

    remainder = len(deduplicated) % 10
    if remainder != 0:
        indices_to_keep = np.random.choice(len(deduplicated), len(deduplicated) - remainder, replace=False)
        deduplicated = [deduplicated[i] for i in indices_to_keep]

    return deduplicated


def run_main(figure1_data_path, found_file, name, lower_bound, upper_bound, real_best, unique_elements_per_column, min_,
             sampling_method, sample_size, sample_type, random_seed, result_queue, reverse=False):
    try:
        print(f"Loading data: {figure1_data_path} (seed={random_seed})")

        sampled_data_tuple = load_sampled_data_from_csv(
            dataset_name=name,
            mode='g1',
            sampling_method=sampling_method,
            num_samples=sample_size,
            random_seed=random_seed,
            figure_type='figure1',
            reverse=reverse
        )

        sampled_features, original_ft, original_fa, _, _, _ = sampled_data_tuple

        sampled_data = [
            list(features) + [ft, fa]
            for features, ft, fa in zip(sampled_features, original_ft, original_fa)
            if fa is not None
        ]

        sampled_data = deduplicate_samples(sampled_data)
        print(f"Deduplicated sample size: {len(sampled_data)}")

        if not sampled_data:
            raise ValueError("No valid sample data")

        numpy_array = np.array([row[:-2] for row in sampled_data])
        df_features = pd.DataFrame(numpy_array)
        df_features.index = pd.RangeIndex(start=0, stop=len(df_features), step=1)

        sampled_lower_bound = [np.min(numpy_array[:, i]) for i in range(numpy_array.shape[1])]
        sampled_upper_bound = [np.max(numpy_array[:, i]) for i in range(numpy_array.shape[1])]
        for k, (i, j) in enumerate(zip(sampled_lower_bound, sampled_upper_bound)):
            if i == j:
                sampled_lower_bound[k] -= 1e-8
                sampled_upper_bound[k] += 1e-8

        main_values = pd.Series([row[-2] for row in sampled_data])
        main_map_real = {tuple(row[:-2]): float(row[-2]) for row in sampled_data}
        performance_column = main_values.values
        if min_:
            min_performance = np.min(performance_column)
        else:
            min_performance = np.max(performance_column)
        sampled_real_best = [tuple(row[:-2]) for row in sampled_data if row[-2] == min_performance]

        landscape_main = landscape(
            map=main_map_real,
            Min=min_,
            df=df_features,
            last_column_values=main_values,
            lower_bound=sampled_lower_bound,
            upper_bound=sampled_upper_bound
        )

        h_max = landscape_main.calculate_h_max()
        kur = landscape_main.calculate_kurtosis()
        nbc = landscape_main.calculate_NBC()
        ske = landscape_main.calculate_skewness()

        aux_values = pd.Series([row[-1] for row in sampled_data])
        aux_map_real = {tuple(row[:-2]): float(row[-1]) for row in sampled_data}
        aux_performance_column = aux_values.values
        aux_min_performance = np.min(aux_performance_column) if min_ else np.max(aux_performance_column)
        aux_sampled_real_best = [tuple(row[:-2]) for row in sampled_data if row[-1] == aux_min_performance]

        landscape_aux = landscape(
            map=aux_map_real,
            Min=min_,
            df=df_features,
            last_column_values=aux_values,
            lower_bound=sampled_lower_bound,
            upper_bound=sampled_upper_bound
        )

        aux_h_max = landscape_aux.calculate_h_max()
        aux_kur = landscape_aux.calculate_kurtosis()
        aux_nbc = landscape_aux.calculate_NBC()
        aux_ske = landscape_aux.calculate_skewness()

        result = [
            found_file, sampling_method, sample_size, sample_type, random_seed,
            h_max, kur, nbc, ske,
            aux_h_max, aux_kur, aux_nbc, aux_ske
        ]
        result_queue.put(result)
        print(f"|{found_file} done|")

        return [h_max, kur, nbc, ske, aux_h_max, aux_kur, aux_nbc, aux_ske]

    except FileNotFoundError as e:
        error_msg = f"Data read failed: file not found {e.filename}"
        print(error_msg)
        result_queue.put([found_file, sampling_method, sample_size, sample_type, random_seed] + [float('nan')] * 8)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error processing {found_file}: {str(e)}"
        print(error_msg)
        result_queue.put([found_file, sampling_method, sample_size, sample_type, random_seed] + [float('nan')] * 8)
        return [float('nan')] * 8


def main_sct_single(dataset_names=None,
                    sampling_methods=None,
                    sample_size=900,
                    random_seeds=None,
                    use_multiprocessing=False,
                    max_workers=None,
                    reverse=False,
                    debug=False,
                    use_saved_data=True):
    if dataset_names is None:
        dataset_names = ['dnn_adiac', 'dnn_coffee', 'dnn_dsr', 'dnn_sa',
                         'llvm', 'lrzip', 'mariadb', 'mongodb', 'vp9', 'x264',
                         'storm_rs', 'storm_wc', 'trimesh']

    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']

    if random_seeds is None:
        random_seeds = range(0, 10)

    if max_workers is None:
        max_workers = 50

    print(f"Max concurrent workers set to: {max_workers}")

    mode = 'g1'
    fixed_sample_sizes = [sample_size]

    figure1_data_dir = "./Results/Samples_multi"
    result_dir = './Results/real_data/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for name in dataset_names:
        dataset = os.path.join('../Datasets', f'{name}.csv')
        min_ = True

        print(f'Processing dataset: {name}')
        try:
            df_original = pd.read_csv(dataset, header=0)
            whole_data = df_original.values
            unique_elements_per_column = [np.unique(whole_data[:, i]) for i in range(whole_data.shape[1] - 2)]
        except Exception as e:
            print(f"Failed to read {dataset}: {str(e)}")
            continue

        if reverse:
            result_file = os.path.join(result_dir, f'{name}_reverse.csv')
        else:
            result_file = os.path.join(result_dir, f'{name}.csv')

        if not os.path.exists(result_file):
            with open(result_file, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    'Name', 'Sampling Method', 'Sample Size', 'Sample Type', 'Random Seed',
                    'h_max', 'kur', 'nbc', 'ske',
                    'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
                ])

        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        writer_process = multiprocessing.Process(target=result_writer, args=(result_file, result_queue))
        writer_process.start()

        tasks = []
        for sampling_method in sampling_methods:
            for sample_size_val in fixed_sample_sizes:
                for random_seed in random_seeds:
                    if reverse:
                        fig1_filename = f"sampled_data_{name}_{mode}_{sampling_method}_{sample_size_val}_{random_seed}_figure1_reverse.csv"
                    else:
                        fig1_filename = f"sampled_data_{name}_{mode}_{sampling_method}_{sample_size_val}_{random_seed}_figure1.csv"
                    fig1_path = os.path.join(figure1_data_dir, fig1_filename)

                    if not os.path.exists(fig1_path):
                        print(f"Warning: data file not found {fig1_path}, skipping")
                        continue

                    key = f'{sampling_method}-{sample_size_val}-fixed-{random_seed}'
                    task = (fig1_path, dataset, name, None, None, None,
                            unique_elements_per_column, min_, sampling_method, sample_size_val, 'fixed', random_seed,
                            result_queue, reverse)
                    tasks.append((key, task))

        if use_multiprocessing:
            pool = multiprocessing.Pool(processes=max_workers)
            futures = []
            for key, task in tasks:
                future = pool.apply_async(run_main, args=task)
                futures.append(future)
            pool.close()
            for future in futures:
                future.wait()
            pool.join()
        else:
            for key, task in tasks:
                try:
                    run_main(*task)
                except Exception as exc:
                    print(f'{key} task error: {exc}')

        result_queue.put("DONE")
        writer_process.join()

    print("All datasets processed")


def result_writer(result_file, result_queue):
    with open(result_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            csv_writer.writerow(result)
            f.flush()


if __name__ == "__main__":
    use_multiprocessing = True
    main_sct_single(reverse=True)
    main_sct_single(reverse=False)