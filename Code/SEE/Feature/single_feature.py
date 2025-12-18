import concurrent
import sys
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
from Code.SEE.Feature.multi_feature import load_sampled_data_from_csv
import multiprocessing
import os
import csv
import random
from typing import List, Tuple, Dict, Any
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.classical_ela_features import *
import faiss
import pandas as pd
import math

SAMPLING_METHODS = [
    'monte_carlo', 'latin_hypercube',
    'sobol', 'stratified', 'halton', 'random_walk'
]

WORKFLOW_DIR = "../Datasets/"
RESULT_DIR = './Results/real_data/'
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 50

from scipy.spatial import KDTree
import numpy as np


class ExactEuclideanNeighborFinder:
    def __init__(self):
        self.precision = 6
        self.kd_tree = None
        self.rounded_data = []
        self.original_to_rounded = {}
        self._needs_rebuild = True

    def _round_vector(self, vector):
        return tuple(round(float(x), self.precision) for x in vector)

    def add(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float64)
        self.rounded_data = [self._round_vector(v) for v in vectors]
        self.original_to_rounded = {
            tuple(v): rounded for v, rounded in zip(vectors, self.rounded_data)
        }
        self._needs_rebuild = True

    def _rebuild_tree(self):
        if self._needs_rebuild and len(self.rounded_data) > 0:
            self.kd_tree = KDTree(np.array(self.rounded_data, dtype=np.float64))
            self._needs_rebuild = False

    def search(self, query, k):
        if len(self.rounded_data) == 0:
            return [], []
        self._rebuild_tree()
        rounded_query = self._round_vector(query)
        distances, indices = self.kd_tree.query(
            np.array(rounded_query).reshape(1, -1),
            k=k
        )
        if k == 1:
            result = [self.rounded_data[indices[0]]], [float(distances[0])]
        else:
            result = (
                [self.rounded_data[i] for i in indices[0]],
                distances[0].tolist()
            )
        return result

    def __len__(self):
        return len(self.rounded_data)


class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_sae': float('inf'), 'original_ci': float('inf')}


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
    def __init__(self, map, Min, df, last_column_values, lower_bound, upper_bound,
                 dag=None, num_valid_tasks=None, random_seed=None, debug=False):
        self.debug = debug
        self.random_seed = random_seed
        self.map = map
        self.populations = list(self.map.keys())
        self.preds = list(self.map.values())
        self.Min = Min
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.df = df
        self.last_column_values = last_column_values
        self._init_index()
        self.ela_distr = calculate_ela_distribution(df, last_column_values)
        self.best = find_value_indices(self.preds, Min=self.Min)
        self.best_populations = [self.populations[i] for i in self.best]

    def _init_index(self):
        if not self.populations:
            self.index = None
            if self.debug:
                print("[Index] Empty dataset, skipping index build")
            return
        self.index = ExactEuclideanNeighborFinder()
        self.index.add(np.array([np.array(k) for k in self.populations]))
        if self.debug:
            print(f"[Index] Index built, contains {len(self.index)} solutions")

    def calculate_Proportion_of_local_optimal(self, unique_elements_per_column=None):
        if not hasattr(self, 'index') or self.index is None:
            if self.debug:
                print("[PLO] Error: index not initialized")
            return float('nan')
        k_neighbors = min(20, max(5, int(len(self.populations) ** 0.5)))
        local_optima_count = 0
        for i, solution in enumerate(self.populations):
            solution_value = self.map[solution]
            neighbors, distances = self.index.search(solution, k_neighbors)
            if not neighbors and self.debug:
                print(f"[PLO] Warning: solution {i} has no neighbors")
                continue
            is_local_optimum = True
            for neighbor in neighbors:
                neighbor_value = self.map.get(neighbor, float('inf'))
                if (self.Min and neighbor_value < solution_value) or \
                        (not self.Min and neighbor_value > solution_value):
                    is_local_optimum = False
                    break
            if is_local_optimum:
                local_optima_count += 1
        result = local_optima_count / len(self.populations) if self.populations else 0
        return result

    def calculate_best_distance(self, real_bests):
        if not hasattr(self, 'index') or self.index is None:
            if self.debug:
                print("[FBD] Error: index not initialized")
            return float('nan')
        min_distance = float('inf')
        found_count = 0
        for i, real_best in enumerate(real_bests):
            _, distances = self.index.search(real_best, k=1)
            if distances:
                current_dist = distances[0]
                if current_dist < min_distance:
                    min_distance = current_dist
                    found_count += 1
        if found_count == 0 and self.debug:
            print("[FBD] Warning: No valid distances found")
        return min_distance if min_distance != float('inf') else float('nan')

    def _get_auto_correlation(self):
        if self.debug:
            print("[CL] Starting autocorrelation...")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        base = self.best_populations
        total = 0
        valid_runs = 0
        num_sublists = 50
        sublist_length = 10
        for _ in range(num_sublists):
            sublist = []
            current = random.choice(base)
            sublist.append(current)
            for _ in range(sublist_length - 1):
                neighbors, _ = self.index.search(current, k=5)
                if not neighbors:
                    break
                current = random.choice(neighbors)
                sublist.append(current)
            data = [self.map.get(sol, float('inf')) for sol in sublist]
            if len(data) < 2:
                continue
            r = auto_correlation_at(data, len(data), 1)
            std = calculate_sd(data)
            if std > 1e-6:
                total += r / (std ** 2)
                valid_runs += 1
        return total / valid_runs if valid_runs > 0 else 0

    def calculate_correlation_length(self):
        d = self._get_auto_correlation()
        if d == 0 or abs(d) == 1:
            return "nan"
        return (1 / math.log(abs(d))) * -1.0

    def _init_exact_index(self):
        if not self.populations:
            self.index = None
            return
        self.index = ExactEuclideanNeighborFinder()
        self.index.add(np.array([np.array(sol) for sol in self.populations]))

    def _find_nearest_neighbors(self, query, k=None):
        if self.index is None or len(self.index) == 0:
            if self.debug:
                print("[NN] Warning: empty index, cannot search neighbors")
            return [], []
        if k is None:
            k = min(20, max(5, int(len(self.populations) ** 0.5)))
            if self.debug:
                print(f"[NN] Adaptive k: {k}")
        try:
            neighbors, distances = self.index.search(query, k)
            if self.debug and len(neighbors) > 0:
                print(f"[NN] Found {len(neighbors)} neighbors, nearest distance: {distances[0]:.6f}")
            return neighbors, distances
        except Exception as e:
            if self.debug:
                print(f"[NN] Search error: {str(e)}")
            return [], []

    def calculate_FDC(self):
        try:
            if self.debug:
                print("\n[FDC] Target statistics:")
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
             use_saved_data: bool = True, fold_name: str = 'fold1'):
    try:
        print(f'Starting processing... (dataset: {name}, fold: {fold_name}, seed: {random_seed})')
        mode = 'g1'
        figure_type = 'figure1'
        if use_saved_data:
            if debug:
                print(f'Loading samples from CSV (dataset: {name}, fold: {fold_name}, seed: {random_seed})...')
            try:
                sampled_solutions = load_sampled_data_from_csv(
                    dataset_name=name,
                    fold_name=fold_name,
                    mode=mode,
                    sampling_method=sampling_method,
                    num_samples=sample_size,
                    random_seed=random_seed,
                    figure_type=figure_type,
                    reverse=False
                )
            except FileNotFoundError as e:
                if debug:
                    print(f'Sampled data not found: {e}')
                raise
            sampled_data = []
            for sol in sampled_solutions:
                sae = sol.attributes['original_sae']
                ci = sol.attributes['original_ci']
                sampled_data.append(sol.variables + [sae, ci])
            if debug:
                print(f'Loaded from CSV, total samples: {len(sampled_data)}')
        else:
            raise NotImplementedError("Sampling generation for SEEProblem is not implemented here")
        seen = set()
        deduplicated_data = []
        for data in sampled_data:
            features = tuple(data[:-2])
            if features not in seen:
                seen.add(features)
                deduplicated_data.append(data)
        sampled_data = deduplicated_data
        if debug:
            print(f'Sample count after deduplication: {len(sampled_data)}')
        if not sampled_data:
            raise ValueError("No samples after deduplication")
        numpy_array = np.array([row[:-2] for row in sampled_data])
        df_features = pd.DataFrame(numpy_array)
        df_features.index = pd.RangeIndex(len(df_features))
        main_sampled_data = [row[:-2] + [row[-2]] for row in sampled_data]
        main_map = {tuple(row[:-1]): float(row[-1]) for row in main_sampled_data}
        main_last_col = pd.Series([row[-1] for row in main_sampled_data])
        main_lower, main_upper = calculate_bounds(main_sampled_data)
        main_landscape = landscape(
            map=main_map,
            Min=min_,
            df=df_features,
            last_column_values=main_last_col,
            lower_bound=main_lower,
            upper_bound=main_upper,
            random_seed=random_seed,
            debug=debug
        )
        main_h_max = main_landscape.calculate_h_max()
        main_kur = main_landscape.calculate_kurtosis()
        main_nbc = main_landscape.calculate_NBC()
        main_ske = main_landscape.calculate_skewness()
        aux_sampled_data = [row[:-2] + [row[-1]] for row in sampled_data]
        aux_map = {tuple(row[:-1]): float(row[-1]) for row in aux_sampled_data}
        aux_last_col = pd.Series([row[-1] for row in aux_sampled_data])
        aux_lower, aux_upper = calculate_bounds(aux_sampled_data)
        aux_landscape = landscape(
            map=aux_map,
            Min=min_,
            df=df_features,
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
            f"{name}_{fold_name}", sampling_method, sample_size, 'fixed', random_seed,
            main_h_max, main_kur, main_nbc, main_ske,
            aux_h_max, aux_kur, aux_nbc, aux_ske
        ]
        result_queue.put(result)
    except Exception as e:
        if debug:
            print(f"Error processing {name}_{fold_name} (seed: {random_seed}): {str(e)}")
        error_result = [f"{name}_{fold_name}", sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
        result_queue.put(error_result)


def main_see_single(
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
        dataset_names = ["china-train", "desharnais-train", "finnish-train", "maxwell-train", "miyazaki-train"]
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array',
                            'halton', 'random_walk']
    if random_seeds is None:
        random_seeds = range(0, 10)
    SELECTED_FOLDS = ['fold1', 'fold2', 'fold3']
    manager = multiprocessing.Manager()
    result_queues = {}
    for name in dataset_names:
        for fold_name in SELECTED_FOLDS:
            queue_key = f"{name}_{fold_name}"
            result_queues[queue_key] = manager.Queue()
    writers = []
    for name in dataset_names:
        for fold_name in SELECTED_FOLDS:
            result_file = os.path.join(RESULT_DIR, f"{name}_{fold_name}.csv")
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
                args=(result_file, result_queues[f"{name}_{fold_name}"])
            )
            p.start()
            writers.append(p)
    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for name in dataset_names:
                min_ = True
                for fold_name in SELECTED_FOLDS:
                    current_queue = result_queues[f"{name}_{fold_name}"]
                    for sampling_method in sampling_methods:
                        for random_seed in random_seeds:
                            future = executor.submit(
                                run_main,
                                workflow_file=name,
                                name=name,
                                min_=min_,
                                sample_size=sample_size,
                                random_seed=random_seed,
                                result_queue=current_queue,
                                sampling_method=sampling_method,
                                debug=debug,
                                use_saved_data=use_saved_data,
                                fold_name=fold_name,
                            )
                            futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task error: {str(e)}")
    else:
        for name in dataset_names:
            min_ = True
            for fold_name in SELECTED_FOLDS:
                current_queue = result_queues[f"{name}_{fold_name}"]
                for sampling_method in sampling_methods:
                    for random_seed in random_seeds:
                        try:
                            run_main(
                                workflow_file=name,
                                name=name,
                                min_=min_,
                                sample_size=sample_size,
                                random_seed=random_seed,
                                result_queue=current_queue,
                                sampling_method=sampling_method,
                                debug=debug,
                                use_saved_data=use_saved_data,
                                fold_name=fold_name,
                            )
                        except Exception as e:
                            print(f"Error processing {name}_{fold_name}: {str(e)}")
                            error_result = [f"{name}_{fold_name}", sampling_method, sample_size, 'fixed',
                                            random_seed] + [float('nan')] * 8
                            current_queue.put(error_result)
    for queue_key in result_queues.keys():
        result_queues[queue_key].put("DONE")
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
    main_see_single(debug=True)