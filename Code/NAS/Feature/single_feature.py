import concurrent
import sys
import warnings
from itertools import combinations, product
import multiprocessing
import os
import csv
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any

from evoxbench.benchmarks import NASBench201Benchmark, NATSBenchmark
from evoxbench.test_suites import c10mop, citysegmop, in1kmop
from jmetal.core.solution import IntegerSolution
from pflacco.classical_ela_features import calculate_ela_distribution, calculate_information_content, calculate_nbc, \
    calculate_cm_grad, calculate_cm_angle
from pflacco.misc_features import calculate_fitness_distance_correlation
from scipy.stats import qmc
from tqdm import tqdm
import faiss
import pandas as pd
import math

sys.path.append('../')
sys.path.append('../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
from Code.NAS.mmo_nas import C10MOPProblem, CitySegMOPProblem, In1KMOPProblem
from Code.NAS.Feature.multi_feature import load_sampled_data_from_csv

RESULT_DIR = 'Results/real_data/'
SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 50

PROBLEM_TYPES = ['c10mop', 'citysegmop', 'in1kmop']
PROBLEM_IDS = {
    'c10mop': [1, 3, 5, 8, 10, 11, 12, 13],
    'citysegmop': [3],
    'in1kmop': [1, 4, 7]
}

MODES = ['g1']
SAMPLING_METHODS = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']

c10_search_space_configs = {
    1: [[0, 1]],
    2: [],
    3: [[0, 1]],
    4: [],
    5: [[0, 1]],
    6: [],
    8: [[0, 1]],
    9: [],
    10: [[0, 1]],
    11: [[0, 1]],
    12: [[0, 1]],
    13: [[0, 1]],
}

cityseg_search_space_configs = {
    1: [],
    2: [],
    3: [[0, 2]],
    4: [],
    6: [],
    9: [],
}

in1k_search_space_configs = {
    1: [[0, 1]],
    2: [],
    4: [[0, 1]],
    5: [],
    7: [[0, 1]],
    8: [],
    9: []
}


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


class SolutionWrapper(IntegerSolution):
    def __init__(self, variables, problem):
        super().__init__(
            lower_bound=problem.lower_bound,
            upper_bound=problem.upper_bound,
            number_of_objectives=problem.number_of_objectives()
        )
        self.variables = variables
        self.attributes = {}


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
            print("Data conversion in progress...")
        self.index_data = np.array(
            [np.array(sol, dtype=np.int32) for sol in tqdm(self.populations, desc="Data conversion", disable=not self.debug)])

        self.index = ExactHammingIndex(self.dimension)
        self.index.add(self.index_data)

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


def get_configs(problem_type, pid):
    if problem_type == 'c10mop':
        raw_configs = c10_search_space_configs.get(pid, [])
    elif problem_type == 'citysegmop':
        raw_configs = cityseg_search_space_configs.get(pid, [])
    elif problem_type == 'in1kmop':
        raw_configs = in1k_search_space_configs.get(pid, [])
    else:
        print(f"Warning: Unknown problem type {problem_type}, returning empty configuration")
        return []

    valid_configs = [cfg for cfg in raw_configs if cfg]
    if not valid_configs:
        print(f"Warning: Problem {problem_type}{pid} has no valid target combination configurations")
    return valid_configs


def create_problem(problem_type, pid, selected_objs, mode, seed):
    if problem_type == 'c10mop':
        if pid == 10:
            benchmark = NASBench201Benchmark(
                200, objs='err&params&flops&edgegpu_latency&edgegpu_energy', dataset='cifar100',
                normalized_objectives=True)
        elif pid == 11:
            benchmark = NASBench201Benchmark(
                200, objs='err&params&flops&edgegpu_latency&edgegpu_energy', dataset='ImageNet16-120',
                normalized_objectives=True)
        elif pid == 12:
            benchmark = NATSBenchmark(
                90, objs='err&params&flops&latency', dataset='cifar100', normalized_objectives=True)
        elif pid == 13:
            benchmark = NATSBenchmark(
                90, objs='err&params&flops&latency', dataset='ImageNet16-120', normalized_objectives=True)
        else:
            benchmark = c10mop(pid)
        return C10MOPProblem(benchmark, mode=mode, selected_objs=selected_objs, random_seed=seed)
    elif problem_type == 'citysegmop':
        benchmark = citysegmop(pid)
        return CitySegMOPProblem(benchmark, mode=mode, selected_objs=selected_objs, random_seed=seed)
    elif problem_type == 'in1kmop':
        benchmark = in1kmop(pid)
        return In1KMOPProblem(benchmark, mode=mode, selected_objs=selected_objs, random_seed=seed)
    raise ValueError(f"不支持的问题类型: {problem_type}")


def run_main(problem_type: str, pid: int, config_idx: int, mode: str,
             sample_size: int, random_seed: int,
             result_queue: multiprocessing.Queue = None,
             sampling_method: str = 'random', debug=False):
    name = f"{problem_type}{pid}_{config_idx}"

    try:
        configs = get_configs(problem_type, pid)
        if not configs:
            raise ValueError(f"问题 {problem_type}{pid} 无有效目标组合配置")
        if config_idx < 0 or config_idx >= len(configs):
            raise ValueError(f"config_idx {config_idx} 超出范围！问题 {problem_type}{pid} 仅支持 0~{len(configs) - 1}")
        selected_objs = configs[config_idx]

        problem = create_problem(problem_type, pid, selected_objs, mode, random_seed)

        dataset_name = name
        if debug:
            print(f'[run_main] Processing dataset: {dataset_name} (mode: {mode}, sampling method: {sampling_method}, seed: {random_seed})')

        try:
            sampled_solutions = load_sampled_data_from_csv(
                dataset_name=dataset_name,
                mode=mode,
                sampling_method=sampling_method,
                num_samples=sample_size,
                random_seed=random_seed,
                figure_type='figure1',
                problem=problem,
                reverse=False
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"未找到采样数据文件！请先执行采样脚本生成以下路径的文件：\n"
                f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{sample_size}_{random_seed}_figure1.csv"
            )

        if len(sampled_solutions) == 0:
            raise ValueError(f"加载的样本为空！请检查采样数据文件：{dataset_name}")
        if debug:
            print(f'[run_main] Loading complete, total {len(sampled_solutions)} valid samples')

        sampled_data = []
        for sol in sampled_solutions:
            sampled_data.append(
                sol.variables + [
                    sol.attributes['selected_objectives'][0],
                    sol.attributes['selected_objectives'][1]
                ]
            )

        seen = set()
        deduplicated_data = []
        for data in sampled_data:
            features = tuple(data[:-2])
            if features not in seen:
                seen.add(features)
                deduplicated_data.append(data)
        if debug:
            print(f'[run_main] After deduplication sample count: {len(deduplicated_data)} (original: {len(sampled_data)})')
        sampled_data = deduplicated_data

        main_sampled_data = [row[:-2] + [row[-2]] for row in sampled_data]
        aux_sampled_data = [row[:-2] + [row[-1]] for row in sampled_data]

        main_real_best = calculate_real_best(main_sampled_data, min_=True)
        main_lower_bound, main_upper_bound = calculate_bounds(main_sampled_data)
        num_dimensions = len(sampled_data[0]) - 2 if sampled_data else 0

        main_last_column = pd.Series([row[-1] for row in main_sampled_data], name='fitness')
        main_numpy_array = np.array([row[:-1] for row in main_sampled_data])
        main_df = pd.DataFrame(main_numpy_array, columns=[f'var_{i}' for i in range(num_dimensions)])

        main_landscape = landscape(
            map={tuple(row[:-1]): float(row[-1]) for row in main_sampled_data},
            Min=True,
            df=main_df,
            last_column_values=main_last_column,
            lower_bound=main_lower_bound,
            upper_bound=main_upper_bound,
            random_seed=random_seed,
            debug=debug
        )

        h_max = main_landscape.calculate_h_max()
        Kur = main_landscape.calculate_kurtosis()
        NBC = main_landscape.calculate_NBC()
        Ske = main_landscape.calculate_skewness()

        aux_lower_bound, aux_upper_bound = calculate_bounds(aux_sampled_data)
        aux_last_column = pd.Series([row[-1] for row in aux_sampled_data], name='fitness')
        aux_numpy_array = np.array([row[:-1] for row in aux_sampled_data])
        aux_df = pd.DataFrame(aux_numpy_array, columns=[f'var_{i}' for i in range(num_dimensions)])

        aux_landscape = landscape(
            map={tuple(row[:-1]): float(row[-1]) for row in aux_sampled_data},
            Min=True,
            df=aux_df,
            last_column_values=aux_last_column,
            lower_bound=aux_lower_bound,
            upper_bound=aux_upper_bound,
            random_seed=random_seed,
            debug=debug
        )

        aux_h_max = aux_landscape.calculate_h_max()
        aux_Kur = aux_landscape.calculate_kurtosis()
        aux_NBC = aux_landscape.calculate_NBC()
        aux_Ske = aux_landscape.calculate_skewness()

        result = [
            name, sampling_method, sample_size,
            random_seed, mode,
            h_max, Kur, NBC, Ske,
            aux_h_max, aux_Kur, aux_NBC, aux_Ske
        ]

        if result_queue is not None:
            result_queue.put(result)
        else:
            return result

    except Exception as e:
        error_msg = f"Error processing {name} (seed: {random_seed}, sampling_method: {sampling_method}): {str(e)}"
        if debug:
            print(f"[Error] {error_msg}")

        error_result = [
            name, sampling_method, sample_size,
            random_seed, mode,
            float('nan'), float('nan'), float('nan'), float('nan'),
            float('nan'), float('nan'), float('nan'), float('nan')
        ]

        if result_queue is not None:
            result_queue.put(error_result)
        else:
            return error_result


def main_evox_single(
        problem_types=None,
        problem_ids=None,
        modes=None,
        sampling_methods=None,
        sample_size=1000,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=100,
        debug=False
):
    if problem_types is None:
        problem_types = PROBLEM_TYPES
    if problem_ids is None:
        problem_ids = PROBLEM_IDS
    if modes is None:
        modes = MODES
    if sampling_methods is None:
        sampling_methods = SAMPLING_METHODS
    if random_seeds is None:
        random_seeds = range(0, 10)

    manager = multiprocessing.Manager()
    result_queues = {}

    for problem_type in problem_types:
        for pid in problem_ids.get(problem_type, []):
            configs = get_configs(problem_type, pid)
            for config_idx in range(len(configs)):
                queue_key = (problem_type, pid, config_idx)
                result_queues[queue_key] = manager.Queue()

    writers = []
    for problem_type in problem_types:
        for pid in problem_ids.get(problem_type, []):
            configs = get_configs(problem_type, pid)
            for config_idx in range(len(configs)):
                result_file = os.path.join(
                    RESULT_DIR,
                    f"{problem_type}{pid}_{config_idx}.csv"
                )
                os.makedirs(os.path.dirname(result_file), exist_ok=True)

                if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
                    with open(result_file, 'w', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([
                            'Name', 'Sampling Method', 'Sample Size',
                            'Random Seed', 'Mode',
                            'h_max', 'kur', 'nbc', 'ske',
                            'h_max_auxiliary', 'kur_auxiliary', 'nbc_auxiliary', 'ske_auxiliary'
                        ])

                p = multiprocessing.Process(
                    target=result_writer,
                    args=(result_file, result_queues[(problem_type, pid, config_idx)])
                )
                p.start()
                writers.append(p)

    if use_multiprocessing:
        if debug:
            print(f"[main] Using multiprocessing mode, max workers: {max_workers}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for problem_type in problem_types:
                for pid in problem_ids.get(problem_type, []):
                    configs = get_configs(problem_type, pid)
                    for config_idx in range(len(configs)):
                        queue_key = (problem_type, pid, config_idx)
                        current_queue = result_queues[queue_key]

                        for seed in random_seeds:
                            for mode in modes:
                                for sampling_method in sampling_methods:
                                    future = executor.submit(
                                        run_main,
                                        problem_type=problem_type,
                                        pid=pid,
                                        config_idx=config_idx,
                                        mode=mode,
                                        sample_size=sample_size,
                                        random_seed=seed,
                                        result_queue=current_queue,
                                        sampling_method=sampling_method,
                                        debug=debug
                                    )
                                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[main] Multiprocessing task execution error: {str(e)}")
    else:
        if debug:
            print("[main] Using single-process mode")
        for problem_type in problem_types:
            for pid in problem_ids.get(problem_type, []):
                configs = get_configs(problem_type, pid)
                for config_idx in range(len(configs)):
                    result_file = os.path.join(
                        RESULT_DIR,
                        f"{problem_type}{pid}_{config_idx}.csv"
                    )

                    for seed in random_seeds:
                        for mode in modes:
                            for sampling_method in sampling_methods:
                                if debug:
                                    print(
                                        f"\n[main] Processing: {problem_type}{pid}_{config_idx} (mode: {mode}, "
                                        f"sampling method: {sampling_method}, seed: {seed})"
                                    )

                                result = run_main(
                                    problem_type=problem_type,
                                    pid=pid,
                                    config_idx=config_idx,
                                    mode=mode,
                                    sample_size=sample_size,
                                    random_seed=seed,
                                    result_queue=None,
                                    sampling_method=sampling_method,
                                    debug=debug
                                )

                                with open(result_file, 'a', newline='') as f:
                                    csv_writer = csv.writer(f)
                                    csv_writer.writerow(result)

    for queue in result_queues.values():
        queue.put("DONE")

    for writer in writers:
        writer.join()


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
    main_evox_single(debug=True)