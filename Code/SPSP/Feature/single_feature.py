import concurrent
import sys
import warnings
from itertools import combinations, product

import hnswlib
import networkx as nx
from jmetal.core.solution import FloatSolution
from scipy.stats import qmc
from tqdm import tqdm

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/mydrive/ccj/code/mmo/')
sys.path.append('../')
sys.path.append('../..')

from Code.SPSP.Feature.multi_feature import load_sampled_data_from_csv

from Code.SPSP.mmo_spsp import SPSProblem
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

DATASET_NAMES = [
    "10-5-skill-4-5", "10-5-skill-6-7",
    "10-10-skill-4-5", "10-10-skill-6-7",
    "10-15-skill-4-5", "10-15-skill-6-7",
    "20-5-skill-4-5", "20-5-skill-6-7",
    "20-10-skill-4-5", "20-10-skill-6-7",
    "20-15-skill-4-5", "20-15-skill-6-7",
    "30-5-skill-4-5", "30-5-skill-6-7",
    "30-10-skill-4-5", "30-10-skill-6-7",
    "30-15-skill-4-5", "30-15-skill-6-7",
]

SAMPLE_SIZE = 1000
USE_MULTIPROCESSING = True
MAX_WORKERS = 50

from scipy.spatial import KDTree
import numpy as np
from collections import defaultdict

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
        self.attributes = {'original_makespan': float('inf'), 'original_cost': float('inf')}


def monte_carlo_sampling(num_samples, num_dimensions, random_seed=None):
    np.random.seed(random_seed)
    return np.random.random((num_samples, num_dimensions))


def latin_hypercube_sampling(num_samples, num_dimensions, random_seed=None):
    sampler = qmc.LatinHypercube(d=num_dimensions, seed=random_seed)
    return sampler.random(n=num_samples)


def sobol_sampling(num_samples, num_dimensions, random_seed=None):
    sampler = qmc.Sobol(d=num_dimensions, scramble=True, seed=random_seed)
    return sampler.random(n=num_samples)


def random_walk_sampling(num_samples, num_dimensions, random_seed=None):
    np.random.seed(random_seed)
    samples = np.zeros((num_samples, num_dimensions))
    current_point = np.random.random(num_dimensions)
    for i in range(num_samples):
        step_size = 0.1
        step = np.random.normal(0, step_size, num_dimensions)
        current_point = np.clip(current_point + step, 0, 1)
        samples[i] = current_point
    return samples


def stratified_sampling(num_samples, num_dimensions, random_seed=None):
    np.random.seed(random_seed)
    samples = np.zeros((num_samples, num_dimensions))
    if num_samples <= 32:
        stratum_size = 5
    else:
        stratum_size = 10
    for i in range(num_samples):
        lower = i * stratum_size
        upper = (i + 1) * stratum_size
        samples[i, :] = np.random.uniform(lower, upper, num_dimensions)
    return samples


def halton_sampling(num_samples, num_dimensions, random_seed=None):
    sampler = qmc.Halton(d=num_dimensions, scramble=True, seed=random_seed)
    return sampler.random(n=num_samples)


def generate_samples(problem: SPSProblem, num_samples: int, random_seed: int,
                     sampling_method: str = 'random', debug=False) -> List[List[float]]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    samples = []
    task_ids = sorted(problem.task_skills.keys())
    emp_ids = sorted(problem.employee_skills.keys())
    num_tasks = len(task_ids)
    num_employees = len(emp_ids)
    dimension = num_tasks * num_employees
    precision = 6

    if debug:
        print(f"\n[SPS sampling start] target samples: {num_samples}, total dim: {dimension}, per-employee subdim: {num_tasks}")
        print(f"[sampling method] {sampling_method} (employee-skill matched per-employee subsampling)")

    emp_eligible_tasks = {}
    for emp_idx, emp_id in enumerate(emp_ids):
        emp_skills = set(problem.employee_skills[emp_id])
        eligible_tasks = []
        for task_idx, task_id in enumerate(task_ids):
            task_skills = set(problem.task_skills[task_id])
            if not emp_skills.isdisjoint(task_skills):
                eligible_tasks.append(task_idx)
        emp_eligible_tasks[emp_idx] = eligible_tasks

    sampling_functions = {
        'monte_carlo': monte_carlo_sampling,
        'latin_hypercube': latin_hypercube_sampling,
        'sobol': sobol_sampling,
        'stratified': stratified_sampling,
        'halton': halton_sampling,
        'random_walk': random_walk_sampling
    }

    sampler = sampling_functions.get(sampling_method, monte_carlo_sampling)

    valid_samples = 0
    attempts = 0
    max_attempts = num_samples * 10

    while valid_samples < num_samples and attempts < max_attempts:
        sample = np.zeros(dimension)
        for emp_idx in range(num_employees):
            eligible_tasks = emp_eligible_tasks[emp_idx]
            num_emp_tasks = len(eligible_tasks)
            if num_emp_tasks == 0:
                continue
            if sampling_method == 'random_walk':
                emp_dedications = np.random.random(num_emp_tasks)
            else:
                emp_subsample = sampler(num_emp_tasks, 1, 500 * random_seed + 10 * emp_idx + valid_samples)
                emp_dedications = emp_subsample.flatten()
            scaling_factor = np.random.uniform(0, 1)
            if np.sum(emp_dedications) > 0:
                emp_dedications = emp_dedications * (scaling_factor / np.sum(emp_dedications))
            for i, task_idx in enumerate(eligible_tasks):
                var_idx = emp_idx * num_tasks + task_idx
                sample[var_idx] = round(emp_dedications[i], precision)

        solution = FloatSolution(
            lower_bound=problem.lower_bound,
            upper_bound=problem.upper_bound,
            number_of_objectives=problem.number_of_objectives,
            number_of_constraints=problem.number_of_constraints
        )
        solution.variables = sample.tolist()
        rounded_vars = [round(float(x), precision) for x in solution.variables]
        solution.variables = rounded_vars
        problem.evaluate(solution)
        if problem.is_feasible(solution):
            samples.append(solution.variables + [solution.objectives[0]])
            valid_samples += 1
        attempts += 1

    unique_samples = []
    seen = set()
    for sample in samples:
        sample_tuple = tuple(sample[:-1])
        if sample_tuple not in seen:
            seen.add(sample_tuple)
            unique_samples.append(sample)

    if len(unique_samples) % 10 != 0:
        remainder = len(unique_samples) % 10
        if remainder > 0:
            indices_to_keep = np.random.choice(len(unique_samples), len(unique_samples) - remainder, replace=False)
            unique_samples = [unique_samples[i] for i in indices_to_keep]

    return unique_samples[:num_samples]


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
    def __init__(self, map, Min, df, last_column_values, lower_bound, upper_bound,
                 dag=None, num_valid_tasks=None, random_seed=None, debug=False):
        self.debug = debug
        self.random_seed = random_seed
        self.map = map
        self.populations = list(map.keys())
        self.preds = list(map.values())
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
                print(f"[PLO] Warning: solution {i} had no neighbors")
                continue
            is_local_optimum = True
            for neighbor in neighbors:
                neighbor_value = self.map.get(neighbor, float('inf'))
                if (self.Min and neighbor_value < solution_value) or (not self.Min and neighbor_value > solution_value):
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
        for i, real_best in enumerate(real_bests):
            _, distances = self.index.search(real_best, k=1)
            if distances:
                current_dist = distances[0]
                if current_dist < min_distance:
                    min_distance = current_dist
        return min_distance if min_distance != float('inf') else float('nan')

    def _get_auto_correlation(self):
        if self.debug:
            print("[CL] Starting autocorrelation computation...")
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


def run_main(workflow_file: str, name: str, min_: bool,
             sample_size: int, random_seed: int, result_queue: multiprocessing.Queue,
             sampling_method: str = 'random', debug=False,
             use_saved_data: bool = True):
    try:
        if debug:
            print(f"[run_main] {name} seed={random_seed} sampling={sampling_method}")

        instance_file = f"{WORKFLOW_DIR}inst-{workflow_file}.conf"
        problem = SPSProblem(instance_file=instance_file, random_seed=random_seed)

        mode = 'g1'
        figure_type = 'figure1'

        sampled_solutions = load_sampled_data_from_csv(
            dataset_name=name,
            mode=mode,
            sampling_method=sampling_method,
            num_samples=sample_size,
            random_seed=random_seed,
            figure_type=figure_type,
            reverse=False
        )

        sampled_data = []
        for sol in sampled_solutions:
            ft = sol.objectives[0]
            fa = sol.objectives[1]
            sampled_data.append(sol.variables + [ft, fa])

        seen = set()
        deduplicated = []
        for row in sampled_data:
            features = tuple(row[:-2])
            if features not in seen:
                seen.add(features)
                deduplicated.append(row)
        sampled_data = deduplicated

        if debug:
            print(f"[run_main] Loaded {len(sampled_data)} unique samples")

        if not sampled_data:
            raise ValueError("No samples after deduplication")

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
            print(f"[run_main error] {name} seed={random_seed}: {str(e)}")
        error_result = [name, sampling_method, sample_size, 'fixed', random_seed] + [float('nan')] * 8
        if result_queue is not None:
            result_queue.put(error_result)
        else:
            return error_result


def result_writer(result_file: str, result_queue: multiprocessing.Queue):
    with open(result_file, 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            csv_writer.writerow(result)
            f.flush()


def main_spsp_single(
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
            "10-5-skill-4-5", "10-5-skill-6-7",
            "10-10-skill-4-5", "10-10-skill-6-7",
            "10-15-skill-4-5", "10-15-skill-6-7",
            "20-5-skill-4-5", "20-5-skill-6-7",
            "20-10-skill-4-5", "20-10-skill-6-7",
            "20-15-skill-4-5", "20-15-skill-6-7",
            "30-5-skill-4-5", "30-5-skill-6-7",
            "30-10-skill-4-5", "30-10-skill-6-7",
            "30-15-skill-4-5", "30-15-skill-6-7",
        ]
    if sampling_methods is None:
        sampling_methods = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"]
    if random_seeds is None:
        random_seeds = range(0, 10)

    manager = multiprocessing.Manager() if use_multiprocessing else None
    result_queues = {name: manager.Queue() for name in dataset_names} if use_multiprocessing else None
    writers = []

    for name in dataset_names:
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
        if use_multiprocessing:
            p = multiprocessing.Process(target=result_writer, args=(result_file, result_queues[name]))
            p.start()
            writers.append(p)

    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for name in dataset_names:
                workflow_file = name
                min_ = True
                for sampling_method in sampling_methods:
                    for random_seed in random_seeds:
                        futures.append(executor.submit(
                            run_main,
                            workflow_file, name, min_,
                            sample_size, random_seed,
                            result_queues[name],
                            sampling_method, debug,
                            use_saved_data,
                        ))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[task error] {str(e)}")

        for name in dataset_names:
            result_queues[name].put("DONE")
        for p in writers:
            p.join()
    else:
        for name in dataset_names:
            workflow_file = name
            min_ = True
            for sampling_method in sampling_methods:
                for random_seed in random_seeds:
                    res = run_main(workflow_file, name, min_, sample_size, random_seed, None, sampling_method, debug,
                                   use_saved_data)
                    result_file = os.path.join(RESULT_DIR, f"{name}.csv")
                    with open(result_file, 'a', newline='', encoding='utf-8') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(res)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')
    main_spsp_single(debug=True)