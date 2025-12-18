import concurrent
import sys
import warnings
from itertools import combinations, product
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../../')

from Code.SDP.Feature.multi_feature import (
    DefectDataManager,
    load_sampled_data_from_csv,
    deduplicate_samples,
    generate_samples
)

from sklearn.base import clone
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
from jmetal.core.solution import BinarySolution

def defect_data_manager_is_loaded(cls, dataset_path: str) -> bool:
    instance = cls()
    return dataset_path in instance.data_store

if not hasattr(DefectDataManager, 'is_loaded'):
    DefectDataManager.is_loaded = classmethod(defect_data_manager_is_loaded)

SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'
]

DATASET_NAMES = [
    'ant-1.7', 'camel-1.6', 'ivy-2.0', 'jedit-4.0', 'lucene-2.4',
    'poi-3.0', 'synapse-1.2', 'velocity-1.6', 'xalan-2.6', 'xerces-1.4'
]
DATASET_PATHS = [f'../Datasets/{name}.csv' for name in DATASET_NAMES]
RESULT_DIR = './Results/real_data/'
os.makedirs(RESULT_DIR, exist_ok=True)

CLASSIFIERS = {
    "J48": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LR": LogisticRegression(max_iter=1000, random_state=42),
    "NB": GaussianNB()
}

SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = min(multiprocessing.cpu_count(), 50)
RANDOM_SEEDS = range(10)

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
        valid_pairs = []
        query_arr = np.array(query)
        for idx in candidate_indices:
            target_vec = self.data[idx]
            if tuple(target_vec) == query_tuple:
                continue
            distance = np.sum(query_arr != np.array(target_vec))
            valid_pairs.append((idx, distance))
        valid_pairs.sort(key=lambda x: x[1])
        selected_pairs = valid_pairs[:k]
        nearest_indices = [idx for idx, _ in selected_pairs]
        nearest_distances = [dist for _, dist in selected_pairs]
        nearest_vectors = [tuple(self.data[i]) for i in nearest_indices]
        result = (nearest_vectors, nearest_distances)
        self._distance_cache[cache_key] = result
        return result

    def clear_cache(self):
        self._distance_cache.clear()

    def __len__(self):
        return len(self.data)

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
    opt_value = min(lst) if Min else max(lst)
    return [index for index, value in enumerate(lst) if value == opt_value]

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
        self.best_populations = [self.populations[i] for i in self.best] if self.best else []
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
            print("Data conversion in progress...")
        self.index_data = np.array([np.array(sol, dtype=np.int32) for sol in tqdm(self.populations, desc="Data conversion", disable=not self.debug)])
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

    def calculate_Proportion_of_local_optimal(self):
        if not hasattr(self, 'index') or self.index is None or len(self.index) == 0:
            if self.debug:
                print("[PLO] index not initialized")
            return float('nan')
        k_neighbors = self._get_adaptive_k()
        batch_size = min(1000, len(self.populations))
        num_batches = (len(self.populations) + batch_size - 1) // batch_size
        local_optima_count = 0
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(self.populations))
            batch_solutions = self.populations[batch_start:batch_end]
            for solution in batch_solutions:
                try:
                    solution_value = self.map[solution]
                    neighbors, _ = self._find_nearest_neighbors_direct(solution, k_neighbors)
                    is_local_optimum = all(self.map[neighbor] >= solution_value if self.Min else self.map[neighbor] <= solution_value for neighbor in neighbors if neighbor in self.map)
                    if is_local_optimum:
                        local_optima_count += 1
                except Exception:
                    continue
        return local_optima_count / len(self.populations) if self.populations else float('nan')

    def calculate_FDC(self):
        try:
            fdc = calculate_fitness_distance_correlation(self.df, self.last_column_values, minimize=self.Min)
            return fdc['fitness_distance.fd_correlation']
        except Exception:
            return float('nan')

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

    def calculate_Gradient_Homogeneity(self):
        cm_grad = calculate_cm_grad(self.df, self.last_column_values, lower_bound=self.lower_bound,
                                    upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
        return cm_grad['cm_grad.mean']

    def calculate_Angle(self):
        cm_angle = calculate_cm_angle(self.df, self.last_column_values, lower_bound=self.lower_bound,
                                      upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
        return cm_angle["cm_angle.angle_mean"]

def run_main(dataset_path: str, clf_name: str, sampling_method: str,
             sample_size: int, random_seed: int, result_queue: multiprocessing.Queue,
             debug=False, use_saved_data: bool = True):
    try:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        task_id = f"{dataset_name}_{clf_name}_{sampling_method}_seed{random_seed}"
        print(f"Starting processing: {task_id}")

        if not DefectDataManager.is_loaded(dataset_path):
            print(f"[Initialization error] {task_id}: Dataset not preloaded")
            result_queue.put([dataset_name, clf_name, sampling_method, 'fixed', random_seed] + [float('nan')] * 8)
            return

        try:
            X_train, X_test, y_train, y_test, feature_names = DefectDataManager.get_eval_data(dataset_path)
            problem = DefectDataManager.get_problem(dataset_path, clf_name)
        except Exception as e:
            print(f"[Initialization error] {task_id}: {str(e)[:100]}")
            result_queue.put([dataset_name, clf_name, sampling_method, 'fixed', random_seed] + [float('nan')] * 8)
            return

        if not use_saved_data:
            print(f"[Error] {task_id}: require use_saved_data=True")
            result_queue.put([dataset_name, clf_name, sampling_method, 'fixed', random_seed] + [float('nan')] * 8)
            return

        try:
            sampled_solutions = load_sampled_data_from_csv(
                dataset_name=dataset_name,
                clf_name=clf_name,
                mode='g1',
                sampling_method=sampling_method,
                num_samples=sample_size,
                random_seed=random_seed,
                figure_type='figure1',
                reverse=False
            )
            sampled_solutions = deduplicate_samples(sampled_solutions)
            if len(sampled_solutions) == 0:
                raise ValueError("No valid samples after deduplication")
        except Exception as e:
            print(f"[Load sampled data error] {task_id}: {str(e)[:120]}")
            result_queue.put([dataset_name, clf_name, sampling_method, 'fixed', random_seed] + [float('nan')] * 8)
            return

        sampled_data = []
        for sol in sampled_solutions:
            feature_vec = sol.variables
            ft = sol.attributes['original_auc']
            fa = sol.attributes['original_featcount']
            sampled_data.append(feature_vec + [ft, fa])

        if len(sampled_data) == 0:
            print(f"[Empty sample error] {task_id}")
            result_queue.put([dataset_name, clf_name, sampling_method, 'fixed', random_seed] + [float('nan')] * 8)
            return

        numpy_array = np.array([row[:-2] for row in sampled_data])
        df_features = pd.DataFrame(numpy_array)
        df_features.index = pd.RangeIndex(start=0, stop=len(df_features), step=1)

        sampled_lower_bound, sampled_upper_bound = calculate_bounds([[*row[:-2], row[-2]] for row in sampled_data])
        main_values = pd.Series([row[-2] for row in sampled_data])
        main_map = {tuple(row[:-2]): float(row[-2]) for row in sampled_data}

        landscape_main = landscape(
            map=main_map,
            Min=True,
            df=df_features,
            last_column_values=main_values,
            lower_bound=sampled_lower_bound,
            upper_bound=sampled_upper_bound,
            random_seed=random_seed,
            debug=debug
        )

        h_max = landscape_main.calculate_h_max()
        kur = landscape_main.calculate_kurtosis()
        nbc = landscape_main.calculate_NBC()
        ske = landscape_main.calculate_skewness()

        aux_values = pd.Series([row[-1] for row in sampled_data])
        aux_map = {tuple(row[:-2]): float(row[-1]) for row in sampled_data}

        landscape_aux = landscape(
            map=aux_map,
            Min=True,
            df=df_features,
            last_column_values=aux_values,
            lower_bound=sampled_lower_bound,
            upper_bound=sampled_upper_bound,
            random_seed=random_seed,
            debug=debug
        )

        aux_h_max = landscape_aux.calculate_h_max()
        aux_kur = landscape_aux.calculate_kurtosis()
        aux_nbc = landscape_aux.calculate_NBC()
        aux_ske = landscape_aux.calculate_skewness()

        result = [
            dataset_name + '_' + clf_name, sampling_method, SAMPLE_SIZE, 'fixed', random_seed,
            h_max, kur, nbc, ske,
            aux_h_max, aux_kur, aux_nbc, aux_ske
        ]
        result_queue.put(result)
        print(f"Completed: {task_id}")

    except Exception as e:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"[Task failed] {dataset_name}_{clf_name}_seed{random_seed}: {str(e)[:200]}")
        result_queue.put([dataset_name + '_' + clf_name, sampling_method, SAMPLE_SIZE, 'fixed', random_seed] + [float('nan')] * 8)

def result_writer(result_file: str, result_queue: multiprocessing.Queue):
    with open(result_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            writer.writerow(result)
            f.flush()

def main_sdp_single(
        dataset_names: List[str] = None,
        sampling_methods: List[str] = None,
        sample_size: int = 1000,
        random_seeds: List[int] = None,
        use_multiprocessing: bool = True,
        max_workers: int = None,
        reverse: bool = False,
        debug: bool = False,
        use_saved_data: bool = True,
        result_dir: str = "./Results/real_data/"
):
    if dataset_names is None:
        dataset_names = [
            'ant-1.7', 'camel-1.6', 'ivy-2.0', 'jedit-4.0', 'lucene-2.4',
            'poi-3.0', 'synapse-1.2', 'velocity-1.6', 'xalan-2.6', 'xerces-1.4'
        ]
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
    if random_seeds is None:
        random_seeds = list(range(10))
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 5)

    print("=" * 80)
    print("Starting preload of defect prediction datasets (using DefectDataManager)")

    dataset_paths = [f'../Datasets/{name}.csv' for name in dataset_names]
    classifiers = {
        "J48": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "NB": GaussianNB()
    }

    DefectDataManager.preload_data(dataset_paths, classifiers, random_seed=42)

    loaded_dataset_paths = [p for p in dataset_paths if DefectDataManager.is_loaded(p)]
    if not loaded_dataset_paths:
        print("ERROR: No datasets were successfully preloaded")
        return
    print(f"Number of successfully preloaded datasets: {len(loaded_dataset_paths)}")

    manager = multiprocessing.Manager()
    result_keys = [f"{os.path.splitext(os.path.basename(p))[0]}_{clf}" for p in loaded_dataset_paths for clf in
                   classifiers.keys()]
    result_queues = {key: manager.Queue() for key in result_keys}
    writers = []

    for key in result_keys:
        dataset_name, clf_name = key.split('_', 1)
        result_file = os.path.join(result_dir, f"{dataset_name}_{clf_name}.csv")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            with open(result_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Name', 'Sampling Method', 'Sample Size', 'Sample Type', 'Random Seed',
                    'h_max', 'kur', 'nbc', 'ske',
                    'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
                ])
        p = multiprocessing.Process(target=result_writer, args=(result_file, result_queues[key]))
        p.start()
        writers.append(p)

    all_tasks = []
    for dataset_path in loaded_dataset_paths:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        for clf_name in classifiers.keys():
            result_key = f"{dataset_name}_{clf_name}"
            for sampling_method in sampling_methods:
                for random_seed in random_seeds:
                    all_tasks.append({
                        'dataset_path': dataset_path,
                        'clf_name': clf_name,
                        'sampling_method': sampling_method,
                        'random_seed': random_seed,
                        'result_key': result_key
                    })

    print(f"Total tasks: {len(all_tasks)}")

    if use_multiprocessing:
        print(f"Starting multiprocessing ({max_workers} workers)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in all_tasks:
                future = executor.submit(
                    run_main,
                    dataset_path=task['dataset_path'],
                    clf_name=task['clf_name'],
                    sampling_method=task['sampling_method'],
                    sample_size=sample_size,
                    random_seed=task['random_seed'],
                    result_queue=result_queues[task['result_key']],
                    debug=debug,
                    use_saved_data=use_saved_data
                )
                futures.append(future)
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    future.result()
                    print(f"Progress: {idx + 1}/{len(futures)}")
                except Exception as e:
                    print(f"Task exception: {str(e)[:120]}")
    else:
        for idx, task in enumerate(all_tasks):
            run_main(
                dataset_path=task['dataset_path'],
                clf_name=task['clf_name'],
                sampling_method=task['sampling_method'],
                sample_size=sample_size,
                random_seed=task['random_seed'],
                result_queue=result_queues[task['result_key']],
                debug=debug,
                use_saved_data=use_saved_data
            )
            print(f"Progress: {idx + 1}/{len(all_tasks)}")

    for key in result_queues:
        result_queues[key].put("DONE")
    for writer in writers:
        writer.join()

    print("All defect landscape feature calculation tasks completed, results saved to:", result_dir)

if __name__ == "__main__":
    main_sdp_single(debug=True)