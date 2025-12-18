import concurrent
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
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
import re

SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'
]

WORKFLOW_DIR = "../Datasets/"
RESULT_DIR = './Results/real_data/'
os.makedirs(RESULT_DIR, exist_ok=True)

MIGRATION_RULES = ["migrationRule1", "migrationRule2","migrationRule3", "migrationRule4", "migrationRule5", "migrationRule7", "migrationRule8",
                       "migrationRule10", "migrationRule18"]
INPUT_FOLDERS = ['input_ALL', 'input_CO', 'input_MS', 'input_DS']

DATASET_FULL_PATHS = []
for rule in MIGRATION_RULES:
    for folder in INPUT_FOLDERS:
        for run_idx in range(0, 10):
            full_path = f"{rule}/pop_size_250/input/{folder}/run_{run_idx}/knapsack_file"
            DATASET_FULL_PATHS.append(full_path)

SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = False
MAX_WORKERS = 70

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
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
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

    def __init__(self, number_of_items: int):
        self.variables = [[0] * number_of_items]
        self.objectives = [float('inf'), float('inf')]
        self.constraints = [0.0]
        self.attributes = {
            'original_profit': float('-inf'),
            'original_count': float('inf'),
            'total_weight': float('inf'),
            'normalized_ft': 0.0,
            'normalized_fa': 0.0
        }


class KnapsackProblem:

    def __init__(self, profits: List[float], weights: List[int], capacity: int,
                 random_seed=None, debug=False):
        self.profits = profits.copy()
        self.weights = weights.copy()
        self.capacity = capacity
        self.number_of_bits = len(profits)
        self.random_seed = random_seed
        self.debug = debug
        self.lower_bound = [0] * self.number_of_bits
        self.upper_bound = [1] * self.number_of_bits
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if debug:
            print(f"[KnapsackProblem] Initialization complete, items: {self.number_of_bits}, capacity: {self.capacity}")

    def evaluate(self, solution: SolutionWrapper) -> None:
        try:
            candidate = solution.variables[0]
            total_weight = sum(w for w, val in zip(self.weights, candidate) if val == 1)
            total_profit = sum(p for p, val in zip(self.profits, candidate) if val == 1)
            total_count = sum(candidate)
            constraint_violation = max(0.0, total_weight - self.capacity)
            solution.attributes.update({
                'original_profit': total_profit,
                'original_count': total_count,
                'total_weight': total_weight
            })
            solution.objectives = [-total_profit, total_count]
            solution.constraints[0] = constraint_violation
        except Exception as e:
            if self.debug:
                print(f"Evaluation error: {str(e)}")
            solution.objectives = [float('inf'), float('inf')]
            solution.constraints[0] = float('inf')


def load_knapsack_data(file_path: str) -> Tuple[List[float], List[int], int]:
    profits = []
    weights = []
    capacity = 0
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knapsack data file not found: {file_path}")
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    first_line = lines[0]
    n_match = re.search(r'(\d+)\s+items', first_line)
    if not n_match:
        raise ValueError(f"Could not find item count in first line: {first_line}")
    n = int(n_match.group(1))
    capacity_line = next((line for line in lines if 'capacity:' in line), None)
    if not capacity_line:
        raise ValueError("Capacity information not found")
    c_match = re.search(r'capacity:\s*\+?(\d+)', capacity_line)
    if not c_match:
        raise ValueError(f"Could not parse capacity: {capacity_line}")
    capacity = int(c_match.group(1))
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('item:'):
            if i + 2 >= len(lines):
                i += 1
                continue
            weight_line = lines[i + 1]
            w_match = re.search(r'weight:\s*\+?(\d+)', weight_line)
            weights.append(int(w_match.group(1)) if w_match else 0)
            profit_line = lines[i + 2]
            p_match = re.search(r'profit:\s*\+?([\d.]+)', profit_line)
            profits.append(float(p_match.group(1)) if p_match else 0.0)
            i += 3
        else:
            i += 1
    if len(weights) != n or len(profits) != n:
        raise ValueError(f"Item count mismatch: parsed {len(weights)}, expected {n}")
    return profits, weights, capacity


def load_sampled_data_from_csv(dataset_name: str, mode: str, sampling_method: str,
                               num_samples: int, random_seed: int, figure_type: str,
                               number_of_items: int, reverse: bool = False) -> List[SolutionWrapper]:
    simplified_name = dataset_name.split('/run_')[0].replace('/pop_size_250/input/input_', '_')
    if reverse:
        filename = f"./Results/Samples_multi/sampled_data_{simplified_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        filename = f"./Results/Samples_multi/sampled_data_{simplified_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Saved sampled data not found: {filename}")
    sampled_solutions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            variables = list(map(int, row[:-4]))
            original_profit = float(row[-4])
            original_count = float(row[-3])
            normalized_ft = float(row[-2])
            normalized_fa = float(row[-1])
            sol = SolutionWrapper(number_of_items)
            sol.variables[0] = variables
            sol.objectives = [-original_profit, original_count]
            sol.attributes['original_profit'] = original_profit
            sol.attributes['original_count'] = original_count
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa
            sampled_solutions.append(sol)
    return sampled_solutions


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
    return [np.unique(feature_data[:, i]) for i in range(feature_data.shape[1])]


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


def calculate_sd(num_array):
    sum_num = sum(num_array)
    mean = sum_num / len(num_array)
    variance = sum((num - mean) ** 2 for num in num_array) / len(num_array)
    return math.sqrt(variance)


def find_value_indices(lst, Min=True):
    if Min:
        min_value = min(lst)
    else:
        min_value = max(lst)
    return [index for index, value in enumerate(lst) if value == min_value]


class landscape():

    def __init__(self, map, Min, df, last_column_values, lower_bound, upper_bound,
                 random_seed=None, debug=False):
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

    def _init_direct_hamming_index(self):
        if not self.populations:
            self.index = None
            return
        sample_solution = next(iter(self.populations))
        self.dimension = len(sample_solution)
        if self.debug:
            print("Converting data...")
        self.index_data = np.array([np.array(sol, dtype=np.int32) for sol in tqdm(self.populations, desc="Converting data", disable=not self.debug)])
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

    def calculate_skewness(self):
        return self.ela_distr['ela_distr.skewness']

    def calculate_kurtosis(self):
        return self.ela_distr['ela_distr.kurtosis']

    def calculate_h_max(self):
        ic = calculate_information_content(self.df, self.last_column_values, seed=self.random_seed)
        self.ic = ic
        return ic['ic.h_max']

    def calculate_NBC(self):
        nbc = calculate_nbc(self.df, self.last_column_values)
        return nbc['nbc.nn_nb.mean_ratio']


def run_main(knapsack_file: str, full_path: str, min_: bool,
             sample_size: int, random_seed: int, result_queue: multiprocessing.Queue,
             sampling_method: str = 'random', debug=False,
             use_saved_data: bool = True):
    try:
        simplified_name = full_path.split('/run_')[0].replace('/pop_size_250/input/input_', '_')
        print(f'Processing... {simplified_name} (sampling: {sampling_method}, seed: {random_seed})')

        if not os.path.exists(knapsack_file):
            print(f"Error: data file not found {knapsack_file}")
            result_queue.put([simplified_name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)
            return

        profits, weights, capacity = load_knapsack_data(knapsack_file)
        problem = KnapsackProblem(profits=profits, weights=weights, capacity=capacity, random_seed=random_seed, debug=debug)
        number_of_items = problem.number_of_bits

        sampled_solutions = load_sampled_data_from_csv(
            dataset_name=full_path,
            mode='g1',
            sampling_method=sampling_method,
            num_samples=sample_size,
            random_seed=random_seed,
            figure_type='figure1',
            number_of_items=number_of_items,
            reverse=False
        )

        sampled_data = []
        for sol in sampled_solutions:
            var_values = sol.variables[0]
            main_obj = sol.objectives[0]
            aux_obj = sol.objectives[1]
            sampled_data.append(var_values + [main_obj, aux_obj])

        seen = set()
        deduplicated_data = []
        for data in sampled_data:
            features = tuple(data[:-2])
            if features not in seen:
                seen.add(features)
                deduplicated_data.append(data)
        sampled_data = deduplicated_data

        if debug:
            print(f'After deduplication sample count: {len(sampled_data)}')

        if not sampled_data:
            result_queue.put([simplified_name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)
            return

        main_sampled_data = [row[:-2] + [row[-2]] for row in sampled_data]
        aux_sampled_data = [row[:-2] + [row[-1]] for row in sampled_data]

        main_np = np.array([row[:-1] for row in main_sampled_data])
        main_df = pd.DataFrame(main_np)
        main_last_col = pd.Series([row[-1] for row in main_sampled_data])
        main_lower, main_upper = calculate_bounds(main_sampled_data)

        main_landscape = landscape(
            map={tuple(row[:-1]): float(row[-1]) for row in main_sampled_data},
            Min=min_,
            df=main_df,
            last_column_values=main_last_col,
            lower_bound=main_lower,
            upper_bound=main_upper,
            random_seed=random_seed,
            debug=debug
        )

        h_max = main_landscape.calculate_h_max()
        kur = main_landscape.calculate_kurtosis()
        nbc = main_landscape.calculate_NBC()
        ske = main_landscape.calculate_skewness()

        aux_df = pd.DataFrame(main_np)
        aux_last_col = pd.Series([row[-1] for row in aux_sampled_data])
        aux_lower, aux_upper = calculate_bounds(aux_sampled_data)

        aux_landscape = landscape(
            map={tuple(row[:-1]): float(row[-1]) for row in aux_sampled_data},
            Min=min_,
            df=aux_df,
            last_column_values=aux_last_col,
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
            simplified_name, sampling_method, sample_size, 'fixed', random_seed,
            h_max, kur, nbc, ske,
            aux_h_max, aux_kur, aux_nbc, aux_ske
        ]
        result_queue.put(result)
        print(f'Finished processing... {simplified_name} (sampling: {sampling_method}, seed: {random_seed})')

    except Exception as e:
        simplified_name = full_path.split('/run_')[0].replace('/pop_size_250/input/input_', '_')
        if debug:
            print(f"Processing error {simplified_name} (seed {random_seed}): {str(e)}")
        error_result = [simplified_name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
        result_queue.put(error_result)


def result_writer(result_file: str, result_queue: multiprocessing.Queue):
    with open(result_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            csv_writer.writerow(result)
            f.flush()


def main_tplm_single(
        dataset_names=None,
        sampling_methods=None,
        sample_size=1000,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=None,
        debug=False,
        use_saved_data=True,
        workflow_base_path='../Datasets/',
        result_dir=None
):
    if dataset_names is None:
        dataset_names = []
        migration_rules = ["migrationRule1", "migrationRule2", "migrationRule3", "migrationRule4", "migrationRule5",
                           "migrationRule7", "migrationRule8", "migrationRule10", "migrationRule18"]
        input_folders = ['ALL', 'CO', 'MS', 'DS']
        for data in migration_rules:
            for input_folder in input_folders:
                if data == "migrationRule3" and input_folder == "DS":
                    continue
                dataset = data + '_' + input_folder
                dataset_names.append(dataset)

    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
    if random_seeds is None:
        random_seeds = range(0, 10)
    if result_dir is None:
        result_dir = "./Results/real_data/"

    print(f"TPLM single-objective feature configuration:")
    print(f"  Number of datasets: {len(dataset_names)}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample size: {sample_size}")
    print(f"  Workflow base path: {workflow_base_path}")
    print(f"  Result directory: {result_dir}")

    dataset_full_paths = []
    for dataset_name in dataset_names:
        try:
            parts = dataset_name.split('_')
            if len(parts) >= 2:
                rule = '_'.join(parts[:-1])
                category = parts[-1]

                for run_idx in random_seeds:
                    full_path = f"{rule}/pop_size_250/input/input_{category}/run_{run_idx}/knapsack_file"
                    dataset_full_paths.append(full_path)
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue

    print(f"Generated {len(dataset_full_paths)} dataset instance paths")

    simplified_names = []
    name_mapping = {}
    for full_path in dataset_full_paths:
        try:
            parts = full_path.split('/')
            rule = parts[0]
            category = parts[3].split('_')[1]
            simplified_name = f"{rule}_{category}"

            if simplified_name not in simplified_names:
                simplified_names.append(simplified_name)
            name_mapping[simplified_name] = f"{rule}_{category}"
        except Exception as e:
            print(f"Error processing path {full_path}: {e}")
            continue

    print(f"Generated {len(simplified_names)} simplified names")

    manager = multiprocessing.Manager() if use_multiprocessing else None
    result_queues = {name: manager.Queue() for name in simplified_names} if use_multiprocessing else None
    writers = []

    for name in simplified_names:
        result_file = os.path.join(result_dir, f"{name}.csv")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                'Name', 'Sampling Method', 'Sample Size', 'Sample Type', 'Random Seed',
                'h_max', 'kur', 'nbc', 'ske',
                'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
            ])
        if use_multiprocessing:
            p = multiprocessing.Process(target=result_writer, args=(result_file, result_queues[name]))
            p.start()
            writers.append(p)

    if use_multiprocessing:
        print(f"Starting multiprocessing with {max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for full_path in dataset_full_paths:
                knapsack_file = os.path.join(workflow_base_path, full_path)
                try:
                    parts = full_path.split('/')
                    rule = parts[0]
                    category = parts[3].split('_')[1]
                    simplified_name = f"{rule}_{category}"

                    min_ = True
                    run_match = re.search(r'run_(\d+)', full_path)
                    random_seed = int(run_match.group(1)) if run_match else 42

                    for sampling_method in sampling_methods:
                        future = executor.submit(
                            run_main,
                            knapsack_file, full_path, min_,
                            sample_size, random_seed,
                            result_queues[simplified_name],
                            sampling_method, debug, use_saved_data
                        )
                        futures.append(future)
                except Exception as e:
                    print(f"Failed to create task {full_path}: {e}")
                    continue

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{len(futures)} tasks")
                except Exception as e:
                    print(f"Task processing error: {e}")
    else:
        print("Using single-process mode...")
        processed = 0
        for full_path in dataset_full_paths:
            try:
                knapsack_file = os.path.join(workflow_base_path, full_path)
                parts = full_path.split('/')
                rule = parts[0]
                category = parts[3].split('_')[1]
                simplified_name = f"{rule}_{category}"

                min_ = True
                run_match = re.search(r'run_(\d+)', full_path)
                random_seed = int(run_match.group(1)) if run_match else 42

                for sampling_method in sampling_methods:
                    run_main(
                        knapsack_file, full_path, min_,
                        sample_size, random_seed,
                        result_queues[simplified_name] if use_multiprocessing else None,
                        sampling_method, debug, use_saved_data
                    )
                    processed += 1
                    if processed % 10 == 0:
                        print(f"Processed {processed}/{len(dataset_full_paths) * len(sampling_methods)} tasks")
            except Exception as e:
                print(f"Processing error {full_path}: {str(e)}")
                if use_multiprocessing and simplified_name in result_queues:
                    result_queues[simplified_name].put(
                        [simplified_name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8)

    if use_multiprocessing:
        for name in simplified_names:
            result_queues[name].put("DONE")
        for p in writers:
            p.join()

    print(f"\nAll tasks completed! Results saved to {result_dir}")


if __name__ == "__main__":
    main_tplm_single(debug=True)