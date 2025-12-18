import concurrent
import sys
import warnings
import multiprocessing
import os
import csv
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_information_content,
    calculate_nbc,
    calculate_cm_grad,
    calculate_cm_angle
)
import pandas as pd
import math
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.insert(0, 'home/ccj/code/mmo')
from Code.SPLT.Feature.utils.multi_feature_compute import calculate_java_hamming_distance
from Code.SPLT.Feature.multi_feature import get_java_sample_csv_path, load_java_individuals_from_csv

RESULT_DIR = 'Results/real_data/'
DATASET_NAMES = ["7z","Amazon","BerkeleyDBC","CocheEcologico","CounterStrikeSimpleFeatureModel",
                "DSSample","Dune","ElectronicDrum","HiPAcc","Drupal",
                "JavaGC","JHipster","lrzip","ModelTransformation",
                "SmartHomev2.2","SPLSSimuelESPnP","VideoPlayer",
                "VP9","WebPortal","x264",'Polly']
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 50

class ExactHammingIndex:
    def __init__(self):
        self.data = []
        self._position_index = defaultdict(list)
        self._distance_cache = {}
        self.product_count = None
        self.feature_count = None

    def add(self, vectors: List[List[List[int]]]):
        if not vectors:
            return

        if self.product_count is None or self.feature_count is None:
            self.product_count = max(len(ind) for ind in vectors) if vectors else 0
            self.feature_count = max(len(prod) for ind in vectors for prod in ind) if vectors else 0
            if self.product_count == 0 or self.feature_count == 0:
                raise ValueError("Unable to infer 2D data dimensions: empty or invalid structure")

        unified_vectors = []
        for ind in vectors:
            unified_prods = list(ind)
            if len(unified_prods) < self.product_count:
                fill_prod = [[0] * self.feature_count for _ in range(self.product_count - len(unified_prods))]
                unified_prods.extend(fill_prod)
            for i in range(len(unified_prods)):
                prod = unified_prods[i]
                if len(prod) < self.feature_count:
                    unified_prods[i] = prod + [0] * (self.feature_count - len(prod))
                elif len(prod) > self.feature_count:
                    unified_prods[i] = prod[:self.feature_count]
            unified_ind = tuple(tuple(prod) for prod in unified_prods)
            unified_vectors.append(unified_ind)

        start_idx = len(self.data)
        self.data.extend(unified_vectors)

        for idx in range(start_idx, len(self.data)):
            ind = self.data[idx]
            for p in range(self.product_count):
                prod = ind[p]
                for f in range(self.feature_count):
                    v = prod[f]
                    self._position_index[(p, f, v)].append(idx)

    def search(self, query: List[List[int]], k: int, max_candidates: int = 1000) -> Tuple[List[Tuple], List[int]]:
        if self.product_count is None or self.feature_count is None:
            raise ValueError("Index has no data; cannot infer dimensions")

        unified_query = list(query)
        if len(unified_query) < self.product_count:
            fill_prod = [[0] * self.feature_count for _ in range(self.product_count - len(unified_query))]
            unified_query.extend(fill_prod)
        for i in range(len(unified_query)):
            prod = unified_query[i]
            if len(prod) < self.feature_count:
                unified_query[i] = prod + [0] * (self.feature_count - len(prod))
            elif len(prod) > self.feature_count:
                unified_query[i] = prod[:self.feature_count]
        query_tuple = tuple(tuple(prod) for prod in unified_query)
        cache_key = (query_tuple, k)

        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        candidate_counts = defaultdict(int)
        for p in range(self.product_count):
            prod = query_tuple[p]
            for f in range(self.feature_count):
                v = prod[f]
                for idx in self._position_index.get((p, f, v), []):
                    candidate_counts[idx] += 1

        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in sorted_candidates[:max_candidates]]
        if not candidate_indices:
            return [], []

        distances = []
        for idx in candidate_indices:
            target_ind = self.data[idx]
            dist = calculate_java_hamming_distance(query_tuple, target_ind)
            distances.append(dist)

        sorted_indices = np.argsort(distances)[:k]
        nearest_indices = [candidate_indices[i] for i in sorted_indices]
        nearest_neighbors = [self.data[i] for i in nearest_indices]
        nearest_distances = [distances[i] for i in sorted_indices]

        result = (nearest_neighbors, nearest_distances)
        self._distance_cache[cache_key] = result
        return result

    def clear_cache(self):
        self._distance_cache.clear()

    def __len__(self):
        return len(self.data)

def calculate_real_best(sampled_map: Dict[Tuple, float], min_: bool = True) -> List[Tuple]:
    if not sampled_map:
        return []
    optimal_value = min(sampled_map.values()) if min_ else max(sampled_map.values())
    return [var for var, val in sampled_map.items() if val == optimal_value]

def calculate_unique_elements(sampled_vectors: List[List[int]]) -> List[np.ndarray]:
    if not sampled_vectors:
        return []
    feature_data = np.array(sampled_vectors)
    return [np.unique(feature_data[:, i]) for i in range(feature_data.shape[1])]

def calculate_bounds(sampled_vectors: List[List[int]]) -> Tuple[List[float], List[float]]:
    if not sampled_vectors:
        return [], []
    feature_data = np.array(sampled_vectors)
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
    def __init__(self, map: Dict[Tuple, float], Min: bool, df: pd.DataFrame, last_column_values: pd.Series,
                 lower_bound: List[float], upper_bound: List[float], random_seed: int = None, debug: bool = False):
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
        vectors = [list(ind) for ind in self.populations]
        self.index = ExactHammingIndex()
        if self.debug:
            print(f"[Index init] adding {len(vectors)} 2D decision variables")
        self.index.add(vectors)
        if self.debug:
            print(f"[Index init] done, dims: {self.index.product_count}x{self.index.feature_count}")

    def _get_adaptive_k(self):
        if not hasattr(self.index, 'product_count') or not hasattr(self.index, 'feature_count'):
            return 20
        total_dim = self.index.product_count * self.index.feature_count
        if total_dim <= 200:
            return 20
        elif total_dim <= 500:
            return 10
        else:
            return 5

    def _find_nearest_neighbors_direct(self, query: Tuple, k: int = None) -> Tuple[List[Tuple], List[int]]:
        if k is None:
            k = self._get_adaptive_k()
        if self.index is None or len(self.index) == 0:
            return [], []
        query_list = list(query)
        return self.index.search(query_list, k)

    def calculate_Proportion_of_local_optimal(self, unique_elements_per_column: List[np.ndarray] = None) -> float:
        if self.index is None or len(self.index) == 0:
            if self.debug:
                print("[PLO] index not initialized")
            return float('nan')
        k_neighbors = self._get_adaptive_k()
        local_optima_count = 0
        batch_size = min(1000, len(self.populations))
        num_batches = (len(self.populations) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="PLO batches", disable=not self.debug):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(self.populations))
            batch_solutions = self.populations[batch_start:batch_end]
            for solution in batch_solutions:
                try:
                    solution_val = self.map[solution]
                    neighbors, _ = self._find_nearest_neighbors_direct(solution, k_neighbors)
                    is_local_opt = all(
                        self.map[neigh] >= solution_val if self.Min else self.map[neigh] <= solution_val
                        for neigh in neighbors if neigh in self.map
                    )
                    if is_local_opt:
                        local_optima_count += 1
                except Exception as e:
                    if self.debug:
                        print(f"[PLO error] {str(e)}")
                    continue
        plo = local_optima_count / len(self.populations) if len(self.populations) > 0 else 0.0
        return plo

    def calculate_FDC(self) -> float:
        try:
            fdc = calculate_fitness_distance_correlation(self.df, self.last_column_values, minimize=self.Min)
            return fdc['fitness_distance.fd_correlation']
        except Exception:
            return float('nan')

    def calculate_best_distance(self, real_bests: List[Tuple]) -> float:
        if self.index is None or not real_bests:
            return float('nan')
        min_dist = float('inf')
        for real_best in real_bests:
            _, distances = self._find_nearest_neighbors_direct(real_best, k=1)
            if distances and distances[0] < min_dist:
                min_dist = distances[0]
        return min_dist if min_dist != float('inf') else float('nan')

    def calculate_skewness(self) -> float:
        return self.ela_distr.get('ela_distr.skewness', float('nan'))

    def calculate_kurtosis(self) -> float:
        return self.ela_distr.get('ela_distr.kurtosis', float('nan'))

    def calculate_h_max(self) -> float:
        if self.ic is None:
            self.ic = calculate_information_content(self.df, self.last_column_values, seed=self.random_seed)
        return self.ic.get('ic.h_max', float('nan'))

    def calculate_NBC(self) -> float:
        try:
            nbc = calculate_nbc(self.df, self.last_column_values)
            return nbc.get('nbc.nn_nb.mean_ratio', float('nan'))
        except Exception:
            return float('nan')

    def calculate_Gradient_Homogeneity(self) -> float:
        try:
            total_dim = self.df.shape[1]
            if total_dim > 10:
                return float('nan')
            cm_grad = calculate_cm_grad(self.df, self.last_column_values, lower_bound=self.lower_bound,
                                        upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
            return cm_grad.get('cm_grad.mean', float('nan'))
        except Exception:
            return float('nan')

    def calculate_Angle(self) -> float:
        try:
            total_dim = self.df.shape[1]
            if total_dim > 10:
                return float('nan')
            cm_angle = calculate_cm_angle(self.df, self.last_column_values, lower_bound=self.lower_bound,
                                          upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
            return cm_angle.get('cm_angle.angle_mean', float('nan'))
        except Exception:
            return float('nan')

def run_main(dataset_name: str, sampling_method: str, sample_size: int, random_seed: int,
             result_queue: multiprocessing.Queue = None, debug: bool = False) -> List[Any]:
    try:
        mode = "g1_g2"
        fig1_csv_path = get_java_sample_csv_path(
            dataset_name=dataset_name,
            mode=mode,
            sampling_method=sampling_method,
            sample_size=sample_size,
            seed=random_seed,
            figure_type="figure1"
        )

        fig1_individuals = load_java_individuals_from_csv(fig1_csv_path, fill_to_max=True)
        if len(fig1_individuals) == 0:
            raise ValueError(f"No valid individuals read from figure1 CSV: {fig1_csv_path}")

        var_list = []
        flattened_list = []
        main_map = {}
        aux_map = {}

        for ind in fig1_individuals:
            decision_var = ind.products
            var_tuple = tuple(tuple(prod) for prod in decision_var)
            ft = ind.originalObjectives[0] if ind.originalObjectives[0] != float('inf') else float('nan')
            fa = ind.originalObjectives[1] if ind.originalObjectives[1] != float('inf') else float('nan')
            if not np.isnan(ft) and not np.isnan(fa):
                if var_tuple not in main_map:
                    var_list.append(var_tuple)
                    flattened = sum(decision_var, [])
                    flattened_list.append(flattened)
                main_map[var_tuple] = ft
                aux_map[var_tuple] = fa

        if len(var_list) == 0:
            raise ValueError("No valid samples (both objectives valid)")

        main_df = pd.DataFrame(flattened_list)
        main_last_col = pd.Series([main_map[var] for var in var_list])
        aux_last_col = pd.Series([aux_map[var] for var in var_list])
        main_real_best = calculate_real_best({var: main_map[var] for var in var_list}, min_=True)
        aux_real_best = calculate_real_best({var: aux_map[var] for var in var_list}, min_=True)
        main_lower, main_upper = calculate_bounds(flattened_list)

        main_land = landscape(
            map={var: main_map[var] for var in var_list},
            Min=True,
            df=main_df,
            last_column_values=main_last_col,
            lower_bound=main_lower,
            upper_bound=main_upper,
            random_seed=random_seed,
            debug=debug
        )

        h_max = main_land.calculate_h_max()
        kur = main_land.calculate_kurtosis()
        nbc = main_land.calculate_NBC()
        ske = main_land.calculate_skewness()

        aux_land = landscape(
            map={var: aux_map[var] for var in var_list},
            Min=True,
            df=main_df,
            last_column_values=aux_last_col,
            lower_bound=main_lower,
            upper_bound=main_upper,
            random_seed=random_seed,
            debug=debug
        )

        aux_h_max = aux_land.calculate_h_max()
        aux_kur = aux_land.calculate_kurtosis()
        aux_nbc = aux_land.calculate_NBC()
        aux_ske = aux_land.calculate_skewness()

        result = [
            dataset_name, sampling_method, sample_size, 'fixed', random_seed,
            h_max, kur, nbc, ske,
            aux_h_max, aux_kur, aux_nbc, aux_ske
        ]

        if result_queue is not None:
            result_queue.put(result)
        else:
            return result

    except Exception as e:
        if debug:
            print(f"[run_main error] dataset: {dataset_name}, seed: {random_seed}, error: {str(e)}")
        error_result = [dataset_name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
        if result_queue is not None:
            result_queue.put(error_result)
        else:
            return error_result

def result_writer(result_file: str, result_queue: multiprocessing.Queue):
    with open(result_file, 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        while True:
            item = result_queue.get()
            if item == "DONE":
                break
            csv_writer.writerow(item)
            f.flush()

def main_splt_single(
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
        dataset_names = ["7z", "Amazon", "BerkeleyDBC", "CocheEcologico", "CounterStrikeSimpleFeatureModel",
                         "DSSample", "Dune", "ElectronicDrum", "HiPAcc", "Drupal",
                         "JavaGC", "JHipster", "lrzip", "ModelTransformation",
                         "SmartHomev2.2", "SPLSSimuelESPnP", "VideoPlayer",
                         "VP9", "WebPortal", "x264", 'Polly']

    if sampling_methods is None:
        sampling_methods = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"]

    if random_seeds is None:
        random_seeds = range(10)

    if max_workers is None:
        max_workers = 50

    RESULT_DIR = 'Results/real_data/'
    DATASET_NAMES = dataset_names
    SAMPLE_SIZE = sample_size
    USE_MULTIPROCESSING = use_multiprocessing
    MAX_WORKERS = max_workers

    manager = multiprocessing.Manager() if USE_MULTIPROCESSING else None
    result_queues = {(name, seed): manager.Queue() for name in DATASET_NAMES for seed in
                     random_seeds} if USE_MULTIPROCESSING else None

    writers = []
    for name in DATASET_NAMES:
        for seed in random_seeds:
            result_file = os.path.join(RESULT_DIR, f"{name}.csv")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            if not os.path.exists(result_file):
                with open(result_file, 'w', newline='', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        'Name', 'Sampling Method', 'Sample Size', 'Sample Type', 'Random Seed',
                        'h_max', 'kur', 'nbc', 'ske',
                        'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
                    ])
            if USE_MULTIPROCESSING:
                p = multiprocessing.Process(target=result_writer, args=(result_file, result_queues[(name, seed)]))
                p.start()
                writers.append(p)

    if USE_MULTIPROCESSING:
        if debug:
            print(f"Using multiprocessing mode, max workers: {MAX_WORKERS}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for name in DATASET_NAMES:
                for seed in random_seeds:
                    for sampling_method in sampling_methods:
                        futures.append(executor.submit(
                            run_main,
                            dataset_name=name,
                            sampling_method=sampling_method,
                            sample_size=SAMPLE_SIZE,
                            random_seed=seed,
                            result_queue=result_queues[(name, seed)],
                            debug=debug
                        ))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[task error] {str(e)}")
        for name in DATASET_NAMES:
            for seed in random_seeds:
                result_queues[(name, seed)].put("DONE")
        for p in writers:
            p.join()
    else:
        if debug:
            print("Using single-process mode")
        for name in DATASET_NAMES:
            for seed in random_seeds:
                for sampling_method in sampling_methods:
                    res = run_main(dataset_name=name, sampling_method=sampling_method, sample_size=SAMPLE_SIZE,
                                   random_seed=seed, result_queue=None, debug=debug)
                    result_file = os.path.join(RESULT_DIR, f"{name}.csv")
                    with open(result_file, 'a', newline='', encoding='utf-8') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(res)

if __name__ == "__main__":
    warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')
    main_splt_single(debug=False)