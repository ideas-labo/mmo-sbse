import concurrent
import sys
import warnings
from itertools import combinations, product

from scipy.stats import qmc
from tqdm import tqdm
sys.path.insert(0, '/home/ccj/mmo')
sys.path.append('../')
sys.path.append('../..')

from Code.WS.Feature.multi_feature import load_sampled_data_from_csv, VmsProblem

import multiprocessing
import os
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.classical_ela_features import *
import pandas as pd
import math


SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'stratified',
    'latin_hypercube', 'monte_carlo', 'covering_array'
]
WORKFLOW_DIR = "../Datasets/"
RESULT_DIR = 'Results/real_data/'
DATASET_NAMES = [
        "CyberShake_30", "CyberShake_50", "CyberShake_100",
        "Epigenomics_24", "Epigenomics_46", "Epigenomics_100",
        "Inspiral_30", "Inspiral_50", "Inspiral_100",
        "Montage_25", "Montage_50", "Montage_100",
        "Sipht_30", "Sipht_60", "Sipht_100", "Sipht_1000",
        "CyberShake_1000","Inspiral_1000","Montage_1000", "Epigenomics_997"
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


class DependencyAwareSampler:
    def __init__(self, dag, random_seed, debug=False):
        self.dag = dag
        self.debug = debug
        if self.debug:
            print(f"\n[Init] Building dependency graph...")
        self._build_dependency_graph()
        if self.debug:
            print(f"[Init] Dependency graph built")
            print(f"[Init] Computing topological levels...")
        self._compute_topological_levels()
        if self.debug:
            print(f"[Init] Topological levels computed")
            print(f"[Init] Sampling parameters initialized")
        self._init_sampling_parameters()

    def _build_dependency_graph(self):
        self.children = defaultdict(set)
        self.parents = defaultdict(set)
        self.roots = set()

        for task in self.dag.tasks:
            if task < 0:
                continue

            preds = set(p for p in self.dag.requiring.get(task, []) if p >= 0)
            if not preds:
                self.roots.add(task)

            for p in preds:
                self.parents[task].add(p)
                self.children[p].add(task)

        if self.debug:
            print(f"[Dependency graph] roots: {len(self.roots)}")
            print(f"[Dependency graph] total parent keys: {len(self.parents)}")

    def _compute_topological_levels(self):
        self.levels = defaultdict(list)
        remaining = set(self.parents.keys()).union(self.roots)
        current_level = 0

        while remaining:
            current_tasks = [t for t in remaining
                             if all(p not in remaining for p in self.parents[t])]

            if not current_tasks:
                raise ValueError("DAG contains cycles")

            for t in current_tasks:
                self.levels[current_level].append(t)

            remaining -= set(current_tasks)
            current_level += 1

        self.max_level = current_level - 1
        self.num_tasks = sum(len(tasks) for tasks in self.levels.values())

        if self.debug:
            print(f"[Topological levels] levels: {self.max_level + 1}")
            for level, tasks in self.levels.items():
                print(f"[Topological levels] level {level}: {len(tasks)} tasks")

        self.level_tasks = [np.array(tasks) for tasks in self.levels.values()]

    def _init_sampling_parameters(self):
        if self.num_tasks <= 50:
            self.order_samples_per_batch = 100
        elif self.num_tasks <= 200:
            self.order_samples_per_batch = 50
        else:
            self.order_samples_per_batch = 20

    def sample_orders(self, n_samples):
        samples = []

        for _ in range(n_samples):
            sample = np.zeros(self.num_tasks, dtype=np.int32)

            current_pos = 0
            for level in range(self.max_level + 1):
                tasks = self.level_tasks[level]
                if len(tasks) > 1:
                    sample[current_pos:current_pos + len(tasks)] = np.random.permutation(tasks)
                else:
                    sample[current_pos] = tasks[0]
                current_pos += len(tasks)

            samples.append(sample.tolist())

        return samples


class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_makespan': float('inf'), 'original_cost': float('inf')}


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
        self.dag = dag
        self.num_valid_tasks = num_valid_tasks
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
        if not hasattr(self, 'dimension'):
            return 20
        if self.dimension <= 200:
            return 20
        elif self.dimension <= 500:
            return 10
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


def run_main(workflow_file: str, name: str, min_: bool,
             sample_size: int, random_seed: int,
             result_queue: multiprocessing.Queue = None,
             sampling_method: str = 'random', debug=False,
             use_saved_data: bool = False):
    try:
        problem = VmsProblem(workflow_file=workflow_file, max_simultaneous_ins=10,
                             random_seed=random_seed)

        dag = problem.dag
        num_valid_tasks = problem.num_valid_tasks
        mode = 'g1'

        try:
            sampled_solutions = load_sampled_data_from_csv(
                dataset_name=name,
                mode=mode,
                sampling_method=sampling_method,
                num_samples=sample_size,
                random_seed=random_seed,
                figure_type='figure1',
                reverse=False
            )
        except FileNotFoundError as e:
            msg = (f"Missing pre-generated sampled data (dataset={name}, method={sampling_method}, "
                   f"size={sample_size}, seed={random_seed}, figure_type=figure1).\n"
                   "Please run the sampling script to generate figure1 data files before retrying."
                   f"\nMissing file info: {e}")
            print(msg)
            error_result = [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
            if result_queue is not None:
                result_queue.put(error_result)
            else:
                return error_result
            return
        except Exception as e:
            msg = (f"Error loading sampled data (dataset={name}, method={sampling_method}, seed={random_seed}): {e}\n"
                   "Please check sampling files or regenerate sampled data.")
            print(msg)
            error_result = [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
            if result_queue is not None:
                result_queue.put(error_result)
            else:
                return error_result
            return

        if not sampled_solutions:
            msg = (f"Loaded sampled data is empty or invalid (dataset={name}, method={sampling_method}, "
                   f"size={sample_size}, seed={random_seed}, figure_type=figure1).\n"
                   "Please ensure the sampling script successfully produced the data file (figure1) and that the file format matches load_sampled_data_from_csv expectations.")
            print(msg)
            error_result = [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
            if result_queue is not None:
                result_queue.put(error_result)
            else:
                return error_result
            return

        sampled_data = []
        for sol in sampled_solutions:
            ft = sol.attributes.get('original_makespan', float('nan'))
            fa = sol.attributes.get('original_cost', float('nan'))
            sampled_data.append(sol.variables + [ft, fa])

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
            msg = (f"Sampled data (dataset={name}, seed={random_seed}) is empty after deduplication. Please check the sampling output and regenerate.")
            print(msg)
            error_result = [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
            if result_queue is not None:
                result_queue.put(error_result)
            else:
                return error_result
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
            dag=dag,
            num_valid_tasks=num_valid_tasks,
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
            dag=dag,
            num_valid_tasks=num_valid_tasks,
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

        if result_queue is not None:
            result_queue.put(result)
        else:
            return result

    except Exception as e:
        if debug:
            print(f"Error processing {name} (seed: {random_seed}): {str(e)}")
        error_result = [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
        if result_queue is not None:
            result_queue.put(error_result)
        else:
            return error_result


def result_writer(result_file: str, result_queue: multiprocessing.Queue):
    with open(result_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            csv_writer.writerow(result)
            f.flush()


def main_ws_single(
        dataset_names=None,
        sampling_methods=None,
        sample_size=1000,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=None,
        debug=False,
        use_saved_data=True,
        workflow_base_path='../Datasets/'
):
    if dataset_names is None:
        dataset_names = [
            "CyberShake_30", "CyberShake_50", "CyberShake_100",
            "Epigenomics_24", "Epigenomics_46", "Epigenomics_100",
            "Inspiral_30", "Inspiral_50", "Inspiral_100",
            "Montage_25", "Montage_50", "Montage_100",
            "Sipht_30", "Sipht_60", "Sipht_100", "Sipht_1000",
            "CyberShake_1000", "Inspiral_1000", "Montage_1000", "Epigenomics_997"
        ]

    if sampling_methods is None:
        sampling_methods = [
            'sobol', 'orthogonal', 'stratified',
            'latin_hypercube', 'monte_carlo', 'covering_array'
        ]

    if random_seeds is None:
        random_seeds = range(5)

    if max_workers is None:
        max_workers = 50

    WORKFLOW_DIR = workflow_base_path
    RESULT_DIR = 'Results/real_data/'

    SAMPLE_SIZE = sample_size
    USE_MULTIPROCESSING = use_multiprocessing
    MAX_WORKERS = max_workers

    manager = multiprocessing.Manager()
    result_queues = {(name, seed): manager.Queue() for name in dataset_names for seed in random_seeds}

    writers = []
    for name in dataset_names:
        result_file = os.path.join(RESULT_DIR, f"{name}.csv")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                'Name', 'Sampling Method', 'Sample Size', 'Sample Type', 'Random Seed',
                'h_max', 'kur', 'nbc', 'ske',
                'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
            ])
        for seed in random_seeds:
            p = multiprocessing.Process(
                target=result_writer,
                args=(result_file, result_queues[(name, seed)])
            )
            p.start()
            writers.append(p)

    if USE_MULTIPROCESSING:
        if debug:
            print(f"Using multiprocessing mode, max workers: {MAX_WORKERS}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for name in dataset_names:
                workflow_file = os.path.join(WORKFLOW_DIR, f"{name}.xml")
                min_ = True
                for seed in random_seeds:
                    for sampling_method in sampling_methods:
                        futures.append(executor.submit(
                            run_main,
                            workflow_file, name, min_,
                            SAMPLE_SIZE, seed,
                            result_queues[(name, seed)],
                            sampling_method, debug,
                            use_saved_data
                        ))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing task: {e}")
    else:
        for name in dataset_names:
            workflow_file = os.path.join(WORKFLOW_DIR, f"{name}.xml")
            min_ = True
            for seed in random_seeds:
                for sampling_method in sampling_methods:
                    res = run_main(workflow_file, name, min_, SAMPLE_SIZE, seed, None, sampling_method, debug,
                                   use_saved_data)
                    result_file = os.path.join(RESULT_DIR, f"{name}.csv")
                    with open(result_file, 'a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(res)

    for name in dataset_names:
        for seed in random_seeds:
            result_queues[(name, seed)].put("DONE")
    for p in writers:
        p.join()


if __name__ == "__main__":
    main_ws_single(debug=False)