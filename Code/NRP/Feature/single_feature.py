import concurrent
import sys
import warnings
from itertools import combinations, product
from scipy.stats import qmc
from tqdm import tqdm
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
from Code.NRP.Feature.multi_feature import load_sampled_data_from_csv
from Code.NRP.mmo_nrp import parse
import multiprocessing
import os
import csv
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.classical_ela_features import *
import faiss
import pandas as pd
import math

SAMPLING_METHODS = [
    'sobol',
    'orthogonal',
    'halton',
    'latin_hypercube',
    'monte_carlo',
    'covering_array'
]

WORKFLOW_DIR = "../Datasets/"
RESULT_DIR = './Results/real_data/'
DATASET_NAMES = [
'nrp1','nrp2','nrp3','nrp4','nrp5',
'nrp-e1','nrp-e2','nrp-e3','nrp-e4',
'nrp-g1','nrp-g2','nrp-g3','nrp-g4',
'nrp-m1','nrp-m2','nrp-m3','nrp-m4',
]
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 50

class ExactHammingIndex:

    def __init__(self, dimension):
        self.dimension = dimension
        self.data = []
        self._position_index = defaultdict(list)
        self._distance_cache = {}

    def add(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.int32)

        start_idx = len(self.data)
        self.data.extend(vectors.tolist())

        for i in range(start_idx, len(self.data)):
            for pos in range(self.dimension):
                val = self.data[i][pos]
                self._position_index[(pos, val)].append(i)

    def search(self, query, k, max_candidates=1000):
        query_tuple = tuple(query)
        cache_key = (query_tuple, k)

        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        if not self.data:
            return [], []

        query = np.array(query, dtype=np.int32).flatten()

        candidate_counts = defaultdict(int)
        for pos in range(self.dimension):
            val = query[pos]
            for idx in self._position_index.get((pos, val), []):
                candidate_counts[idx] += 1

        sorted_candidates = sorted(candidate_counts.items(),
                                   key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in sorted_candidates[:max_candidates]]

        distances = []
        query_arr = np.array(query)
        for idx in candidate_indices:
            target = np.array(self.data[idx])
            distances.append(np.sum(query_arr != target))

        sorted_indices = np.argsort(distances)[:k]
        nearest_indices = [candidate_indices[i] for i in sorted_indices]
        nearest_distances = [distances[i] for i in sorted_indices]

        result = ([tuple(self.data[i]) for i in nearest_indices], nearest_distances)
        self._distance_cache[cache_key] = result
        return result

    def clear_cache(self):
        self._distance_cache.clear()

    def __len__(self):
        return len(self.data)

class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_makespan': float('inf'), 'original_cost': float('inf')}

def generate_qmc_samples(dimensions: int, upper_bound: int, num_samples: int,
                         sampling_method: str, random_seed: int) -> List[List[int]]:
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")

    if sampling_method == 'sobol':
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_seed)
    elif sampling_method == 'orthogonal':
        sampler = qmc.LatinHypercube(d=dimensions, optimization="random-cd", seed=random_seed)
    elif sampling_method == 'halton':
        sampler = qmc.Halton(d=dimensions, scramble=True, seed=random_seed)
    elif sampling_method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(d=dimensions, seed=random_seed)
    else:
        raise ValueError(f"Unsupported QMC sampling method: {sampling_method}")

    if num_samples & (num_samples - 1) == 0:
        sample = sampler.random_base2(m=int(np.log2(num_samples)))
    else:
        sample = sampler.random(n=num_samples)

    samples = (sample * upper_bound).astype(int)
    samples = np.clip(samples, 0, upper_bound - 1)

    return samples.tolist()

def generate_covering_array_samples(dimensions: int, upper_bound: int,
                                    num_samples: int, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)

    dim_pairs = list(combinations(range(dimensions), 2))
    required_pairs = set()
    for dim1, dim2 in dim_pairs:
        required_pairs.update(product(range(upper_bound), range(upper_bound)))

    selected = []
    remaining_pairs = required_pairs.copy()

    while len(selected) < num_samples and remaining_pairs:
        target_val1, target_val2 = random.choice(list(remaining_pairs))
        target_dims = random.choice(dim_pairs)

        candidate = [random.randint(0, upper_bound - 1) for _ in range(dimensions)]
        candidate[target_dims[0]] = target_val1
        candidate[target_dims[1]] = target_val2

        new_covered = set()
        for (v1, v2) in remaining_pairs:
            if (candidate[target_dims[0]] == v1 and
                    candidate[target_dims[1]] == v2):
                new_covered.add((v1, v2))

        selected.append(candidate)
        remaining_pairs -= new_covered

    while len(selected) < num_samples:
        selected.append([random.randint(0, upper_bound - 1) for _ in range(dimensions)])

    return selected[:num_samples]

def calculate_real_best(sampled_data: List[List[Any]], min_: bool = True) -> List[Tuple]:
    if not sampled_data:
        return []

    objectives = [row[-1] for row in sampled_data]
    optimal_value = min(objectives) if min_ else max(objectives)
    best_indices = [i for i, val in enumerate(objectives) if val == optimal_value]
    return [tuple(sampled_data[i][:-1]) for i in best_indices]

def calculate_unique_elements(sampled_data: List[List[Any]]) -> List[np.ndarray]:
    if not sampled_data:
        return []
    feature_data = np.array([row[:-1] for row in sampled_data])
    unique_elements = [np.unique(feature_data[:, i]) for i in range(feature_data.shape[1])]
    return unique_elements

def calculate_bounds(sampled_data: List[List[Any]]) -> Tuple[List[float], List[float]]:
    if not sampled_data:
        return [], []
    feature_data = np.array([row[:-1] for row in sampled_data])
    lower_bound = np.min(feature_data, axis=0).tolist()
    upper_bound = np.max(feature_data, axis=0).tolist()
    for i in range(len(lower_bound)):
        if lower_bound[i] == upper_bound[i]:
            lower_bound[i] -= 1e-8
            upper_bound[i] += 1e-8
    return lower_bound, upper_bound

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
    distance = math.sqrt(sum_squared_diff)
    return distance

def calculate_sd(num_array):
    sum_num = sum(num_array)
    mean = sum_num / len(num_array)
    variance = sum((num - mean) ** 2 for num in num_array) / len(num_array)
    return math.sqrt(variance)

def hamming_dist(str1, str2):
    return sum(a != b for a, b in zip(str1, str2))

def rank_dif(y_test, y_pred):
    y_test_return = []
    y_pred_retuen = []
    y_test_sort = sorted(y_test)
    y_pred_sort = sorted(y_pred)
    for i, val in enumerate(y_test):
        y_test_return.append(y_test_sort.index(val) + 1)
    for i, val in enumerate(y_pred):
        y_pred_retuen.append(y_pred_sort.index(val) + 1)
    dist = np.sum(np.abs(np.array(y_test_return) - np.array(y_pred_retuen))) / len(y_test_return) / len(y_test_return)
    return dist

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    errors = np.abs((actual - predicted) / actual)
    mape = np.mean(errors) * 100
    return mape

def find_value_indices(lst, Min=True):
    if Min:
        min_value = min(lst)
    else:
        min_value = max(lst)
    min_indices = [index for index, value in enumerate(lst) if value == min_value]
    return min_indices

class landscape():
    def __init__(self, map, Min, df, last_column_values, lower_bound, upper_bound, dag=None,
                 num_valid_tasks=None, random_seed=None, debug=False):
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
        self.random_seed = random_seed
        self.debug = debug
        self._init_direct_hamming_index()

    def _get_auto_correlation(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        list_keys = self.populations[:]
        base = self.best_populations

        def _min_distance(x, base):
            min_dist = float('inf')
            for single_base in base:
                _, distances = self._find_nearest_neighbors_direct(x, k=1)
                if distances and distances[0] < min_dist:
                    min_dist = distances[0]
            return min_dist if min_dist != float('inf') else 1e8

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
                    temp = [list_keys[i] for i in range(len(list_keys))
                            if (self._find_nearest_neighbors_direct(list_keys[i], k=1)[1][0]
                                if self._find_nearest_neighbors_direct(list_keys[i], k=1)[1]
                                else float('inf')) <= k]
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
        return total / size

    def calculate_correlation_length(self):
        d = self._get_auto_correlation()
        if d == 0 or abs(d) == 1:
            return "nan"
        return (1 / math.log(abs(d))) * -1.0

    def _init_direct_hamming_index(self):
        if not self.populations:
            self.index = None
            return

        sample_solution = next(iter(self.populations))
        self.dimension = len(sample_solution)

        if self.debug:
            print("Data conversion in progress...")
        self.index_data = np.array(
            [np.array(sol, dtype=np.int32) for sol in tqdm(self.populations, desc="Data conversion", disable=not self.debug)])

        self.index = ExactHammingIndex(self.dimension)
        self.index.add(self.index_data)

    def _get_adaptive_k(self):
        if self.dimension <= 100:
            return 10
        elif self.dimension <= 200:
            return 8
        else:
            return 5

    def _find_nearest_neighbors_direct(self, query, k=None):
        if k is None:
            k = self._get_adaptive_k()

        if self.index is None or len(self.index) == 0:
            return [], []

        query = np.array(query, dtype=np.int32)
        neighbors, distances = self.index.search(query, k)
        return neighbors, distances

    def calculate_Proportion_of_local_optimal(self, unique_elements_per_column=None):
        if not hasattr(self, 'index') or self.index is None or len(self.index) == 0:
            if self.debug:
                print("[PLO] Warning: index not initialized or empty")
            return float('nan')

        k_neighbors = self._get_adaptive_k()
        if self.debug:
            print(f"\n[PLO] Start computing proportion of local optima (PLO), adaptive K={k_neighbors}")

        batch_size = min(1000, len(self.populations))
        num_batches = (len(self.populations) + batch_size - 1) // batch_size
        local_optima_count = 0

        for batch_idx in tqdm(range(num_batches), desc="PLO batch progress", disable=not self.debug):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(self.populations))
            batch_solutions = self.populations[batch_start:batch_end]

            for solution in batch_solutions:
                try:
                    solution_value = self.map[solution]
                    neighbors, _ = self._find_nearest_neighbors_direct(solution, k_neighbors)

                    is_local_optimum = all(
                        self.map[neighbor] >= solution_value if self.Min else self.map[neighbor] <= solution_value
                        for neighbor in neighbors if neighbor in self.map
                    )

                    if is_local_optimum:
                        local_optima_count += 1

                except Exception as e:
                    if self.debug:
                        print(f"\n[PLO Error] Error processing solution {solution}: {str(e)}")
                    continue

        plo_value = local_optima_count / len(self.populations)
        if self.debug:
            print(f"[PLO Result] Proportion of local optima: {plo_value:.2%}")
        return plo_value

    def calculate_FDC(self):
        try:
            if self.debug:
                print("\n[FDC] Objective statistics:")
                print(f"Count: {len(self.last_column_values)}")
                print(f"Range: [{self.last_column_values.min()}, {self.last_column_values.max()}]")
                print(f"Std: {self.last_column_values.std()}")

            fdc = calculate_fitness_distance_correlation(self.df, self.last_column_values, minimize=self.Min)

            if self.debug:
                print(f"[FDC] Result: {fdc['fitness_distance.fd_correlation']}")

            return fdc['fitness_distance.fd_correlation']
        except Exception as e:
            if self.debug:
                print(f"[FDC] Error: {str(e)}")
            return float('nan')

    def calculate_best_distance(self, real_bests):
        if not hasattr(self, 'index') or self.index is None:
            return float('nan')

        min_distance = float('inf')
        for real_best in real_bests:
            _, distances = self._find_nearest_neighbors_direct(real_best, k=1)
            if distances and distances[0] < min_distance:
                min_distance = distances[0]
        return min_distance

    def calculate_skewness(self):
        return self.ela_distr['ela_distr.skewness']

    def calculate_kurtosis(self):
        return self.ela_distr['ela_distr.kurtosis']

    def calculate_h_max(self):
        ic = calculate_information_content(self.df, self.last_column_values, seed=self.random_seed)
        self.ic = ic
        return ic['ic.h_max']

    def calculate_eps_s(self):
        if self.ic:
            return self.ic['ic.eps_s']
        else:
            return calculate_information_content(self.df, self.last_column_values, seed=self.random_seed)['ic.eps_s']

    def calculate_NBC(self):
        nbc = calculate_nbc(self.df, self.last_column_values)
        return nbc['nbc.nn_nb.mean_ratio']

    def calculate_Gradient_Homogeneity(self):
        cm_grad = calculate_cm_grad(self.df, self.last_column_values, lower_bound=self.lower_bound,
                                    upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
        return cm_grad['cm_grad.mean']

    def calculate_Angle(self):
        cm_angle = calculate_cm_angle(self.df, self.last_column_values, lower_bound=self.lower_bound,
                                      upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
        return cm_angle["cm_angle.angle_mean"]

def run_main(workflow_file: str, name: str, min_: bool,
             sample_size: int, random_seed: int, result_queue: multiprocessing.Queue,
             sampling_method: str = 'random', debug=False,
             use_saved_data: bool = True):
    try:
        if debug:
            print(f'Starting processing {name} (seed={random_seed}, sampling={sampling_method})')

        if use_saved_data:
            try:
                sampled_solutions = load_sampled_data_from_csv(
                    dataset_name=name,
                    mode='g1',
                    sampling_method=sampling_method,
                    num_samples=sample_size,
                    random_seed=random_seed,
                    figure_type='figure1',
                    reverse=False
                )
                if debug:
                    print(f'Loaded {len(sampled_solutions)} samples')
            except FileNotFoundError as e:
                if debug:
                    print(f'CSV file not found: {e}')
                result_queue.put([name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)
                return

            sampled_data = []
            for sol in sampled_solutions:
                var_values = [1 if x else 0 for x in sol.variables[0]]
                main_obj = sol.attributes.get('ft', float('inf'))
                aux_obj = sol.attributes.get('fa', float('inf'))
                sampled_data.append(var_values + [main_obj, aux_obj])

            seen = set()
            deduplicated_data = []
            for data in sampled_data:
                features = tuple(data[:-2])
                if features not in seen:
                    seen.add(features)
                    deduplicated_data.append(data)
            sampled_data = deduplicated_data

            if not sampled_data:
                if debug:
                    print('No valid samples, skipping')
                result_queue.put([name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)
                return

        main_sampled_data = [row[:-2] + [row[-2]] for row in sampled_data]
        aux_sampled_data = [row[:-2] + [row[-1]] for row in sampled_data]

        num_dimensions = len(main_sampled_data[0]) - 1
        main_last_column = pd.Series([row[-1] for row in main_sampled_data], name='fitness')
        main_numpy_array = np.array([row[:-1] for row in main_sampled_data])
        main_df = pd.DataFrame(main_numpy_array, columns=[f'var_{i}' for i in range(num_dimensions)])

        main_lower, main_upper = calculate_bounds(main_sampled_data)
        main_landscape = landscape(
            map={tuple(row[:-1]): float(row[-1]) for row in main_sampled_data},
            Min=min_,
            df=main_df,
            last_column_values=main_last_column,
            lower_bound=main_lower,
            upper_bound=main_upper,
            random_seed=random_seed,
            debug=debug
        )

        h_max = main_landscape.calculate_h_max()
        kur = main_landscape.calculate_kurtosis()
        nbc = main_landscape.calculate_NBC()
        ske = main_landscape.calculate_skewness()

        aux_last_column = pd.Series([row[-1] for row in aux_sampled_data], name='fitness')
        aux_numpy_array = np.array([row[:-1] for row in aux_sampled_data])
        aux_df = pd.DataFrame(aux_numpy_array, columns=[f'var_{i}' for i in range(num_dimensions)])
        aux_lower, aux_upper = calculate_bounds(aux_sampled_data)
        aux_landscape = landscape(
            map={tuple(row[:-1]): float(row[-1]) for row in aux_sampled_data},
            Min=True,
            df=aux_df,
            last_column_values=aux_last_column,
            lower_bound=aux_lower,
            upper_bound=aux_upper,
            random_seed=random_seed,
            debug=debug
        )

        aux_h_max = aux_landscape.calculate_h_max()
        aux_kur = aux_landscape.calculate_kurtosis()
        aux_nbc = aux_landscape.calculate_NBC()
        aux_ske = aux_landscape.calculate_skewness()

        result = [
            name, sampling_method, sample_size, 'fixed', random_seed,
            h_max, kur, nbc, ske,
            aux_h_max, aux_kur, aux_nbc, aux_ske
        ]

        result_queue.put(result)
        if debug:
            print(f'Result queued: {name} (seed={random_seed})')

    except Exception as e:
        if debug:
            print(f"Error processing (dataset: {name}, seed: {random_seed}): {str(e)}")
        result_queue.put([name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)

def main_nrp_single(
        dataset_names=None,
        sampling_methods=None,
        sample_size=1000,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=50,
        debug=False,
        use_saved_data=True
):
    if dataset_names is None:
        dataset_names = DATASET_NAMES
    if sampling_methods is None:
        sampling_methods = SAMPLING_METHODS
    if random_seeds is None:
        random_seeds = range(0, 10)

    manager = multiprocessing.Manager()
    result_queues = {name: manager.Queue() for name in dataset_names}

    writers = []
    for name in dataset_names:
        result_file = os.path.join(RESULT_DIR, name + ".csv")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        with open(result_file, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                'Name', 'Sampling Method', 'Sample Size', 'Sample Type', 'Random Seed',
                'h_max', 'kur', 'nbc', 'ske',
                'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
            ])

        p = multiprocessing.Process(
            target=result_writer,
            args=(result_file, result_queues[name])
        )
        p.start()
        writers.append(p)

    if use_multiprocessing:
        if debug:
            print(f"[main_nrp_single] Using multiprocessing mode, max workers: {max_workers}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for name in dataset_names:
                workflow_file = os.path.join(WORKFLOW_DIR, name + ".txt")
                min_ = True

                for sampling_method in sampling_methods:
                    for random_seed in random_seeds:
                        future = executor.submit(
                            run_main,
                            workflow_file, name, min_,
                            sample_size, random_seed,
                            result_queues[name],
                            sampling_method, debug, use_saved_data
                        )
                        futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing task: {e}")
    else:
        if debug:
            print("[main_nrp_single] Using single-process mode")
        for name in dataset_names:
            workflow_file = os.path.join(WORKFLOW_DIR, name + ".txt")
            min_ = True

            if debug:
                print(f"\n{'=' * 50}")
                print(f"Processing dataset: {name} (Single Process)")
                print(f"{'=' * 50}")

            for sampling_method in sampling_methods:
                if debug:
                    print(f"\nProcessing sampling method: {sampling_method}")

                for random_seed in random_seeds:
                    try:
                        run_main(
                            workflow_file, name, min_,
                            sample_size, random_seed,
                            result_queues[name],
                            sampling_method, debug, use_saved_data
                        )
                    except Exception as e:
                        print(f"Error processing {name} with {sampling_method} (seed {random_seed}): {str(e)}")
                        result_queues[name].put(
                            [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)

    for name in dataset_names:
        result_queues[name].put("DONE")

    for writer in writers:
        writer.join()

    if debug:
        print("\nAll tasks completed!")

def result_writer(result_file: str, result_queue: multiprocessing.Queue):
    with open(result_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            csv_writer.writerow(result)
            f.flush()

if __name__ == "__main__":
    main_nrp_single(debug=True)