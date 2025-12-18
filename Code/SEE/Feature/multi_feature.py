from typing import List, Any, Dict
import sys
from scipy.stats import qmc

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')

from Code.SEE.mmo_see import SEEProblem, ExcelReader
from Code.SEE.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
from Code.SEE.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
import random
import multiprocessing
import sys

sys.path.append('../')
sys.path.append('../..')

SAMPLING_METHODS = [
    'monte_carlo', 'latin_hypercube', 'sobol', 'stratified', 'halton', 'random_walk'
]


class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_sae': float('inf'), 'original_ci': float('inf')}


import csv
import os
from typing import List
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def save_sampled_data_to_csv(sampled_solutions: List[SolutionWrapper],
                             header: List[str], dataset_name: str, fold_name: str, mode: str,
                             sampling_method: str, num_samples: int, random_seed: int,
                             figure_type: str, reverse: bool = False) -> None:
    if mode == 'g1':
        if reverse:
            filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{fold_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
        else:
            filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{fold_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    else:
        if reverse:
            filename = f"./Results/Samples_multi_fa/sampled_data_{dataset_name}_{fold_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
        else:
            filename = f"./Results/Samples_multi_fa/sampled_data_{dataset_name}_{fold_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    rows = []
    for sol in sampled_solutions:
        row = sol.variables + [
            sol.objectives[0],
            sol.objectives[1],
            sol.attributes.get('normalized_ft', 0),
            sol.attributes.get('normalized_fa', 0)
        ]
        rows.append(row)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Data saved to: {filename}")


def load_sampled_data_from_csv(dataset_name: str, fold_name: str, mode: str, sampling_method: str,
                               num_samples: int, random_seed: int, figure_type: str,
                               reverse: bool = False) -> List[SolutionWrapper]:
    if reverse:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{fold_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{fold_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Saved sampled data not found: {filename}")

    sampled_solutions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            variables = list(map(float, row[:-4]))
            original_ft = float(row[-4])
            original_fa = float(row[-3])
            normalized_ft = float(row[-2])
            normalized_fa = float(row[-1])

            sol = SolutionWrapper(variables)
            sol.objectives = [original_ft, original_fa]
            sol.attributes['original_sae'] = original_ft
            sol.attributes['original_ci'] = original_fa
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa

            sampled_solutions.append(sol)

    return sampled_solutions


def generate_samples(problem: SEEProblem, num_samples: int, random_seed: int,
                     sampling_method: str = 'random', debug=False) -> List[SolutionWrapper]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    samples = []
    num_dimensions = problem._number_of_variables
    precision = 6
    if debug:
        print(f"\n[SEE sampling start] target samples: {num_samples}, dimensions: {num_dimensions}")
        print(f"[sampling method] {sampling_method}")

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
    max_attempts = num_samples * 100

    lower_bounds = np.array(problem.lower_bound)
    upper_bounds = np.array(problem.upper_bound)
    ranges = upper_bounds - lower_bounds

    while valid_samples < num_samples and attempts < max_attempts:
        if sampling_method == 'random_walk':
            sample_uniform = random_walk_sampling(1, num_dimensions, random_seed + attempts)[0]
        else:
            sample_uniform = sampler(1, num_dimensions, random_seed + attempts).flatten()

        sample = lower_bounds + sample_uniform * ranges

        solution = SolutionWrapper(sample.tolist())
        problem.evaluate(solution)

        if not any(np.isinf(obj) for obj in solution.objectives):
            samples.append(solution)
            valid_samples += 1
        attempts += 1

    if debug:
        print(f"SEE sampling completed, valid samples: {valid_samples}, attempts: {attempts}")

    if len(samples) % 10 != 0:
        remainder = len(samples) % 10
        if remainder > 0:
            indices_to_keep = np.random.choice(len(samples), len(samples) - remainder, replace=False)
            samples = [samples[i] for i in indices_to_keep]

    return samples[:num_samples]


def sample_and_save(problem: SEEProblem, dataset_name: str, fold_name: str, minimize, num_samples, random_seed,
                    sampling_method, sample_type, reverse: bool = False, debug: bool = False):
    sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method)
    if not sampled_solutions:
        print(f"[sample_and_save] Warning: no valid samples generated: {dataset_name}_{fold_name}")
        return

    ft = [s.objectives[0] for s in sampled_solutions]
    fa = [s.objectives[1] for s in sampled_solutions]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(np.column_stack((ft, fa)))

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = normalized[i, 0]
        sol.attributes['normalized_fa'] = normalized[i, 1]

    header = [f'var_{i}' for i in range(problem._number_of_variables)] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, fold_name, 'g1', sampling_method,
        num_samples, random_seed, 'figure1', reverse
    )

    g1, g2 = transform_points_for_figure2(normalized)
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, fold_name, 'g1', sampling_method,
        num_samples, random_seed, 'figure2', reverse
    )

    if debug:
        print(f"[sample_and_save] Sampling and saving completed: {dataset_name}_{fold_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")


def process_g1_mode(problem: SEEProblem, dataset_name: str, fold_name: str, minimize, num_samples, random_seed,
                       sampling_method,
                       sample_type, mode, unique_id, reverse: bool = False, first_sample: bool = False):
    dimension = problem._number_of_variables
    header = [f'var_{i}' for i in range(dimension)] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    if first_sample:
        sample_and_save(problem, dataset_name, fold_name, minimize, num_samples, random_seed, sampling_method,
                        sample_type, reverse, debug=True)
        return

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, fold_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse
        )
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}_{fold_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Run this program with first_sample=True to generate sampled CSV first.")
        return

    normalized = np.array([[s.attributes['normalized_ft'], s.attributes['normalized_fa']] for s in sampled_solutions])
    r0_points = normalized
    g1, g2 = transform_points_for_figure2(r0_points)

    sampled_data = [s.variables for s in sampled_solutions]

    sampled_dict = {tuple(sample): (ft_val, fa_val)
                    for sample, ft_val, fa_val in zip(sampled_data, r0_points[:, 0], r0_points[:, 1])}
    plot_figure1(random_seed, 'mean', sampling_method, num_samples, f"{dataset_name}_{fold_name}", mode, unique_id,
                 sample_type,
                 sampled_dict, reverse)

    sampled_dict_fig2 = {tuple(sample): (g1_val, g2_val)
                         for sample, g1_val, g2_val in zip(sampled_data, g1, g2)}
    plot_figure2(random_seed, 'mean', sampling_method, num_samples, f"{dataset_name}_{fold_name}", mode, unique_id,
                 sample_type,
                 sampled_dict_fig2, reverse)


def process_fa_construction_mode(problem: SEEProblem, dataset_name: str, fold_name: str, minimize, num_samples,
                                 random_seed, fa_construction,
                                 sampling_method, sample_type, mode, unique_id, reverse: bool = False,
                                 first_sample: bool = False):
    if first_sample:
        sample_and_save(problem, dataset_name, fold_name, minimize, num_samples, random_seed, sampling_method,
                        sample_type, reverse, debug=True)
        return

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, fold_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Please run g1 mode to generate base data first: {dataset_name}_{fold_name}")

    sorted_solutions = sorted(sampled_solutions, key=lambda x: sum(x.objectives))
    if len(sorted_solutions) % 10 != 0:
        target_size = (len(sorted_solutions) // 10) * 10
        sorted_solutions = sorted_solutions[:target_size]
        print(f"Adjusted sample count to multiple of 10: {len(sorted_solutions)}")
    sampled_solutions = sorted_solutions

    batch_size = 20
    num_batches = (len(sampled_solutions) + batch_size - 1) // batch_size
    sorted_indices = list(range(len(sampled_solutions)))

    def get_batch_indices_local(sorted_indices, batch_size, num_batches, reverse_prob=0.8):
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

    batch_indices = get_batch_indices_local(sorted_indices, batch_size, num_batches)

    t = 1
    t_max = 1000
    novelty_archive = []
    all_ft_normalized = []
    all_fa_normalized = []
    age_info = None

    for batch_num in range(num_batches):
        batch_solutions = [sampled_solutions[i] for i in batch_indices[batch_num]]
        batch_ft = [s.objectives[0] for s in batch_solutions]
        batch_vars = [s.variables for s in batch_solutions]

        num_cols = len(batch_vars[0]) if batch_vars else 0
        unique_elements_per_column = []
        for col in range(num_cols):
            unique_elements = set(row[col] for row in batch_vars)
            unique_elements_per_column.append(sorted(unique_elements))

        if fa_construction == 'age':
            age_info = [i + 1 if batch_num == 0 else batch_num * batch_size + 1
                        for i in range(len(batch_solutions))]
        elif fa_construction == 'novelty':
            update_novelty_archive(batch_solutions, novelty_archive)

        batch_ft_norm, batch_fa_norm = generate_fa(
            batch_vars, batch_ft, fa_construction, minimize, unique_elements_per_column,
            t, t_max, random_seed,
            age_info=age_info if fa_construction == 'age' else None,
            novelty_archive=novelty_archive if fa_construction in ['novelty', 'age'] else None,
            k=min(10, len(batch_solutions) // 2)
        )

        all_ft_normalized.extend(batch_ft_norm)
        all_fa_normalized.extend(batch_fa_norm)
        t += 1

    dimension = problem._number_of_variables
    header = [f'var_{i}' for i in range(dimension)] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = all_ft_normalized[i]
        sol.attributes['normalized_fa'] = all_fa_normalized[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, fold_name, mode, sampling_method,
        num_samples, random_seed, 'figure1', reverse
    )

    r0_points = np.column_stack((all_fa_normalized, all_ft_normalized))
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, fold_name, mode, sampling_method,
        num_samples, random_seed, 'figure2', reverse
    )

    sampled_data = [s.variables for s in sampled_solutions]
    sampled_dict = {tuple(s.variables): (ft_val, fa_val)
                    for s, ft_val, fa_val in zip(sampled_solutions, all_ft_normalized, all_fa_normalized)}
    plot_figure1(random_seed, 'mean', sampling_method, num_samples, f"{dataset_name}_{fold_name}",
                 mode, unique_id, sample_type, sampled_dict, reverse)

    sampled_dict_fig2 = {tuple(s.variables): (g1_val, g2_val)
                         for s, g1_val, g2_val in zip(sampled_solutions, g1, g2)}
    plot_figure2(random_seed, 'mean', sampling_method, num_samples,
                 f"{dataset_name}_{fold_name}", mode, unique_id, sample_type, sampled_dict_fig2)


def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2


import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed


class ProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problems(cls, dataset_names: List[str], data_base_path: str, random_seed: int = 42) -> Dict[
        str, SEEProblem]:
        instance = cls()
        for dataset in dataset_names:
            data_path = os.path.join(data_base_path, f"{dataset}.xls")
            if not os.path.exists(data_path):
                print(f"Warning: dataset file not found {data_path}, skipping preload")
                continue

            try:
                dataset_obj = ExcelReader.read(data_path)
                for fold in dataset_obj.get_folds():
                    fold_name = fold.name
                    key = (dataset, fold_name)
                    if key not in instance.problems:
                        instance.problems[key] = SEEProblem(fold, random_seed)
                        print(f"Successfully preloaded SEE instance: {dataset}_{fold_name}")
            except Exception as e:
                print(f"Failed to load SEE instance {dataset}: {str(e)}")
        return instance.problems

    @classmethod
    def get_problem(cls, dataset_name: str, fold_name: str) -> SEEProblem:
        instance = cls()
        return instance.problems.get((dataset_name, fold_name))


def init_worker():
    pass


def process_single_task(mode, dataset_name, fold_name, sampling_method, num_samples, sample_type,
                        minimize, random_seed, fa_construction, unique_id, reverse,
                        first_sample: bool = False, **kwargs):
    try:
        problem = ProblemManager.get_problem(dataset_name, fold_name)
        if problem is None:
            raise ValueError(f"Preloaded SEE problem instance not found: {dataset_name}_{fold_name}")

        np.random.seed(random_seed)
        random.seed(random_seed)

        if mode == 'g1':
            process_g1_mode(
                problem, dataset_name, fold_name, minimize, num_samples, random_seed,
                sampling_method, sample_type, mode, unique_id, reverse, first_sample=first_sample
            )
        elif mode in fa_construction:
            process_fa_construction_mode(
                problem, dataset_name, fold_name, minimize, num_samples, random_seed,
                mode, sampling_method, sample_type, mode, unique_id, reverse, first_sample=first_sample
            )

        return f"SEE task completed: {unique_id}"
    except Exception as e:
        return f"SEE task failed: {unique_id}, Error: {str(e)}"


def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)
    total_tasks = len(all_tasks)
    total_batches = (total_tasks + batch_size - 1) // batch_size

    print(f"Starting batched processing: total tasks={total_tasks}, batches={total_batches}, batch size={batch_size}, max_workers={max_workers}")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_tasks)
        current_batch = all_tasks[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}, tasks in batch: {len(current_batch)}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_task, **task): task for task in current_batch}

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    print(f"Task result: {result}")
                except Exception as e:
                    print(f"Task exception ({task['unique_id']}): {str(e)}")


def main_see_multi(dataset_names, fa_construction, minimize=True,
                   fixed_sample_sizes=[1000],
                   percentage_sample_sizes=[10, 20, 30, 40, 50],
                   sampling_methods=None,
                   use_multiprocessing=True,
                   max_workers=None,
                   reverse=False,
                   first_sample: bool = False,
                   data_base_path='./Datasets/',
                   random_seeds=None):
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array',
                            'halton', 'random_walk']
    if random_seeds is None:
        random_seeds = range(0, 10)

    print("Preloading all SEE instances (including folds)...")
    for seed in random_seeds:
        ProblemManager.preload_problems(dataset_names, data_base_path, seed)

    all_problems = ProblemManager().problems
    print(f"Total preloaded SEE instances: {len(all_problems)}")
    if first_sample:
        fa_construction = ["g1"]

    all_tasks = []
    for dataset in dataset_names:
        fold_names = []
        for (ds_name, fold_name), _ in all_problems.items():
            if ds_name == dataset:
                fold_names.append(fold_name)

        if not fold_names:
            fold_names = [f"fold{i + 1}" for i in range(3)]
            print(f"Dataset {dataset} has no preloaded folds, using default folds: {fold_names}")
        else:
            fold_names = list(dict.fromkeys(fold_names))
            print(f"Dataset {dataset} contains folds: {fold_names}")

        for fold_name in fold_names:
            for mode in fa_construction:
                for sampling_method in sampling_methods:
                    for num_sample in fixed_sample_sizes:
                        for random_seed in random_seeds:
                            unique_id = f"{dataset}_{fold_name}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}_reverse_{reverse}" if reverse else f"{dataset}_{fold_name}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}"
                            task = {
                                'mode': mode,
                                'dataset_name': dataset,
                                'fold_name': fold_name,
                                'sampling_method': sampling_method,
                                'num_samples': num_sample,
                                'sample_type': 'fixed',
                                'minimize': minimize,
                                'random_seed': random_seed,
                                'fa_construction': fa_construction,
                                'unique_id': unique_id,
                                'reverse': reverse,
                                'first_sample': first_sample,
                            }
                            all_tasks.append(task)

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 50)

    if use_multiprocessing:
        print(f"Starting multiprocessing for SEE tasks, total {len(all_tasks)} tasks...")
        batch_size = max_workers
        process_in_batches(
            all_tasks=all_tasks,
            max_workers=max_workers,
            batch_size=batch_size
        )
    else:
        print("Processing tasks in single-process mode...")
        for task in all_tasks:
            print(process_single_task(**task))


if __name__ == "__main__":
    fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    dataset_names = ["china-train", "desharnais-train", "finnish-train", "maxwell-train",
                     "miyazaki-train"]

    main_see_multi(dataset_names, fa_construction, use_multiprocessing=True, reverse=False, first_sample=True)
    main_see_multi(dataset_names, fa_construction, use_multiprocessing=True, reverse=False, first_sample=False)