import concurrent
import csv
import os
import sys
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Any, Dict
import numpy as np
from itertools import combinations, product
import multiprocessing
import warnings

sys.path.append('../')
sys.path.append('../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.insert(0, 'home/ccj/code/mmo')

from Code.WSC.mmo_wsc import ServiceCompositionProblem
from sklearn.preprocessing import MinMaxScaler
from Code.WSC.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from Code.WSC.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2

import scipy.stats._qmc as qmc

warnings.filterwarnings("ignore", category=Warning)


class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {
            'latency': float('inf'),
            'throughput': float('inf'),
            'normalized_ft': 0.0,
            'normalized_fa': 0.0
        }


def deduplicate_samples(sampled_data: List[SolutionWrapper]) -> List[SolutionWrapper]:
    seen = set()
    deduplicated = []

    for sample in sampled_data:
        features = tuple(sample.variables)
        if features not in seen:
            seen.add(features)
            deduplicated.append(sample)

    return deduplicated


SAMPLING_METHODS = [
    'sobol', 'covering_array', 'stratified', 'orthogonal', 'latin_hypercube', 'monte_carlo'
]


def generate_stratified_samples(dimensions: int, upper_bound: int,
                                num_samples: int, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    if num_samples <= 20:
        strat_dims = 10
    else:
        strat_dims = min(20, dimensions)
    samples = []

    strata = int(np.ceil(num_samples ** (1 / strat_dims)))
    samples_per_stratum = int(np.ceil(num_samples / (strata ** strat_dims)))

    strata_combinations = list(product(*[range(strata) for _ in range(strat_dims)]))
    for stratum in strata_combinations:
        for _ in range(samples_per_stratum):
            sample = []
            for dim in range(strat_dims):
                lower = stratum[dim] * upper_bound / strata
                upper = (stratum[dim] + 1) * upper_bound / strata
                sample.append(random.randint(int(lower), int(upper) - 1))
            for dim in range(strat_dims, dimensions):
                sample.append(random.randint(0, upper_bound - 1))
            samples.append(sample)

    return random.sample(samples, min(num_samples, len(samples)))


def generate_samples(problem: ServiceCompositionProblem, num_samples: int, random_seed: int,
                     sampling_method: str = 'random', reverse: bool = False, debug=False) -> List[SolutionWrapper]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    sampled_solutions = []
    num_abstract_services = problem.number_of_variables

    if debug:
        print(f"\n[web sampling start] target samples: {num_samples}, abstract services: {num_abstract_services}")
        print(f"[sampling method] {sampling_method}, [reverse] {reverse}")

    if sampling_method == 'stratified':
        strat_dims = 10 if num_samples <= 20 else min(20, num_abstract_services)
        samples = []

        strata = int(np.ceil(num_samples ** (1 / strat_dims)))
        samples_per_stratum = int(np.ceil(num_samples / (strata ** strat_dims)))

        strata_combinations = list(product(*[range(strata) for _ in range(strat_dims)]))
        for stratum in strata_combinations:
            for _ in range(samples_per_stratum):
                sample = []
                for dim in range(strat_dims):
                    max_idx = len(problem.abstract_to_concrete[problem.abstract_services[dim]]) - 1
                    lower = stratum[dim] * max_idx / strata
                    upper = (stratum[dim] + 1) * max_idx / strata
                    sample_val = random.randint(int(lower), int(upper) - 1) if int(upper) - 1 >= int(lower) else 0
                    sample.append(sample_val)
                for dim in range(strat_dims, num_abstract_services):
                    max_idx = len(problem.abstract_to_concrete[problem.abstract_services[dim]]) - 1
                    sample.append(random.randint(0, max_idx))
                samples.append(sample)

        samples = random.sample(samples, min(num_samples, len(samples)))

    elif sampling_method == 'monte_carlo':
        samples = []
        for _ in range(num_samples):
            variables = []
            for i in range(num_abstract_services):
                max_idx = len(problem.abstract_to_concrete[problem.abstract_services[i]]) - 1
                variables.append(random.randint(0, max_idx))
            samples.append(variables)

    elif sampling_method in ['sobol', 'orthogonal', 'latin_hypercube']:
        samples = []
        if sampling_method == 'sobol':
            sampler = qmc.Sobol(d=num_abstract_services, scramble=True, seed=random_seed)
        elif sampling_method == 'orthogonal':
            sampler = qmc.LatinHypercube(d=num_abstract_services, optimization="random-cd", seed=random_seed)
        else:
            sampler = qmc.LatinHypercube(d=num_abstract_services, seed=random_seed)

        if num_samples & (num_samples - 1) == 0:
            base_sample = sampler.random_base2(m=int(np.log2(num_samples)))
        else:
            base_sample = sampler.random(n=num_samples)

        for row in base_sample:
            variables = []
            for i in range(num_abstract_services):
                max_idx = len(problem.abstract_to_concrete[problem.abstract_services[i]]) - 1
                var = int(round(row[i] * max_idx))
                variables.append(min(var, max_idx))
            samples.append(variables)

    elif sampling_method == 'covering_array':
        samples = generate_covering_array_samples(
            dimensions=num_abstract_services,
            upper_bounds=[len(problem.abstract_to_concrete[abs_svc])
                          for abs_svc in problem.abstract_services],
            num_samples=num_samples,
            random_seed=random_seed
        )

    else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}")

    for variables in samples:
        solution = SolutionWrapper(variables)
        problem.evaluate(solution)
        original_latency = solution.objectives[0]
        original_throughput = solution.objectives[1]

        solution.attributes['latency'] = original_latency
        solution.attributes['throughput'] = original_throughput

        if reverse:
            solution.objectives[0] = original_throughput
            solution.objectives[1] = original_latency
        else:
            solution.objectives[0] = original_latency
            solution.objectives[1] = original_throughput

        sampled_solutions.append(solution)

    seen = set()
    unique_solutions = []
    for sol in sampled_solutions:
        var_tuple = tuple(sol.variables)
        if var_tuple not in seen:
            seen.add(var_tuple)
            unique_solutions.append(sol)

    if len(unique_solutions) % 10 != 0:
        remainder = len(unique_solutions) % 10
        indices = np.random.choice(len(unique_solutions), len(unique_solutions) - remainder, replace=False)
        unique_solutions = [unique_solutions[i] for i in indices]

    final_solutions = unique_solutions[:num_samples]

    if debug:
        print(f"[web sampling completed] deduped samples: {len(final_solutions)}")
        print(f"[definition] ft={'throughput (original)' if reverse else 'latency'}, fa={'latency' if reverse else 'throughput (original)'}")

    return final_solutions


def generate_covering_array_samples(dimensions: int, upper_bounds: List[int],
                                    num_samples: int, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    dim_pairs = list(combinations(range(dimensions), 2))
    required_pairs = set()
    for dim1, dim2 in dim_pairs:
        pairs = product(range(upper_bounds[dim1]), range(upper_bounds[dim2]))
        required_pairs.update({(dim1, dim2, v1, v2) for v1, v2 in pairs})

    selected = []
    remaining_pairs = required_pairs.copy()

    while len(selected) < num_samples and remaining_pairs:
        target = random.choice(list(remaining_pairs))
        dim1, dim2, val1, val2 = target

        candidate = [random.randint(0, upper_bounds[i] - 1) for i in range(dimensions)]
        candidate[dim1] = val1
        candidate[dim2] = val2

        new_covered = set()
        for (d1, d2, v1, v2) in remaining_pairs:
            if candidate[d1] == v1 and candidate[d2] == v2:
                new_covered.add((d1, d2, v1, v2))

        selected.append(candidate)
        remaining_pairs -= new_covered

    while len(selected) < num_samples:
        candidate = [random.randint(0, upper_bounds[i] - 1) for i in range(dimensions)]
        selected.append(candidate)

    return selected[:num_samples]


def generate_monte_carlo_batch(problem: ServiceCompositionProblem, n_samples: int, dim: int, is_vm_mapping: bool,
                               debug=False) -> List[List[int]]:
    if debug:
        print(f"Monte Carlo batch generation, dim: {dim}, VM mapping: {is_vm_mapping}")
    if is_vm_mapping:
        samples = np.random.randint(0, problem.max_simultaneous_ins, size=(n_samples, dim))
    else:
        samples = np.random.randint(0, len(problem.vm_configs), size=(n_samples, dim))

    return samples.tolist()


def get_batch_indices(sorted_indices, batch_size, num_batches, reverse_prob=0.8):
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


def save_sampled_data_to_csv(sampled_solutions: List[SolutionWrapper],
                             header: List[str], dataset_name: str, mode: str,
                             sampling_method: str, num_samples: int, random_seed: int,
                             figure_type: str, reverse: bool = False) -> None:
    if reverse:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"

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

    print(f"Sampled data saved to: {filename}")


def load_sampled_data_from_csv(dataset_name: str, mode: str, sampling_method: str,
                               num_samples: int, random_seed: int, figure_type: str,
                               reverse: bool = False) -> List[SolutionWrapper]:
    if reverse:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Saved sampled data not found: {filename}")

    sampled_solutions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            variables = list(map(float, row[:-4]))
            ft = float(row[-4])
            fa = float(row[-3])
            normalized_ft = float(row[-2])
            normalized_fa = float(row[-1])

            sol = SolutionWrapper(variables)
            sol.objectives = [ft, fa]
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa

            if reverse:
                sol.attributes['throughput'] = ft
                sol.attributes['latency'] = fa
            else:
                sol.attributes['latency'] = ft
                sol.attributes['throughput'] = fa

            sampled_solutions.append(sol)

    return sampled_solutions


def sample_and_save(problem: ServiceCompositionProblem, workflow_file: str, minimize: bool, num_samples: int,
                    random_seed: int, sampling_method: str, sample_type: str, dataset_name: str,
                    reverse: bool = False, debug: bool = False):
    sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method, reverse=reverse, debug=debug)
    if not sampled_solutions:
        print(f"[sample_and_save] Warning: no valid samples generated: {dataset_name}, seed={random_seed}, reverse={reverse}")
        return

    ft = [s.objectives[0] for s in sampled_solutions]
    fa = [s.objectives[1] for s in sampled_solutions]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(np.column_stack((ft, fa)))

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = normalized[i, 0]
        sol.attributes['normalized_fa'] = normalized[i, 1]

    header = [f'var_{i}' for i in range(problem.number_of_variables)] + ['ft', 'fa', 'normalized_ft', 'normalized_fa']
    save_sampled_data_to_csv(sampled_solutions, header, dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse)

    r0_points = normalized
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]
    save_sampled_data_to_csv(sampled_solutions, header, dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure2', reverse)

    if debug:
        print(f"[sample_and_save] Sampling and saving completed: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}, reverse={reverse}")


def process_g1_mode(problem, workflow_file, minimize, num_samples, random_seed, sampling_method,
                       sample_type, dataset_name, mode, unique_id, reverse=False, first_sample: bool = False):
    header = [f'var_{i}' for i in range(problem.number_of_variables)] + \
             ['ft', 'fa', 'normalized_ft', 'normalized_fa']

    if first_sample:
        sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse, debug=True)
        return

    try:
        sampled_solutions = load_sampled_data_from_csv(dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse)
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}, reverse={reverse}")
        print("Run this program with first_sample=True to generate sampled CSV first.")
        return

    try:
        sampled_solutions_fig2 = load_sampled_data_from_csv(dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure2', reverse)
    except FileNotFoundError:
        sampled_solutions_fig2 = None

    sampled_dict = {
        tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
        for s in sampled_solutions
    }
    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples, dataset_name,
        mode, unique_id, sample_type, sampled_dict, reverse,
        None, None
    )

    if sampled_solutions_fig2 is not None:
        sampled_dict_fig2 = {
            tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions_fig2
        }
    else:
        r0_points = np.array([[s.attributes['normalized_ft'], s.attributes['normalized_fa']] for s in sampled_solutions])
        g1, g2 = transform_points_for_figure2(r0_points)
        for i, s in enumerate(sampled_solutions):
            s.attributes['normalized_ft'] = g1[i]
            s.attributes['normalized_fa'] = g2[i]
        sampled_dict_fig2 = {
            tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions
        }

    plot_figure2(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type, sampled_dict_fig2, reverse,
        None, None
    )


def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2


def process_fa_construction_mode(problem, workflow_file, minimize, num_samples, random_seed, fa_construction,
                                 sampling_method, sample_type, dataset_name, mode, unique_id, reverse=False, first_sample: bool = False):
    if first_sample:
        sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse, debug=True)
        return

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse
        )
        print(f"Loaded data from g1: {dataset_name}, {sampling_method}, seed={random_seed}, reverse={reverse}")
    except FileNotFoundError:
        print(f"[FA mode] g1 base sampled data not found: {dataset_name}, {sampling_method}, seed={random_seed}, reverse={reverse}")
        print("Run this program with first_sample=True to generate sampled CSV first.")
        return

    if len(sampled_solutions) % 10 != 0:
        sorted_solutions = sorted(sampled_solutions, key=lambda x: x.objectives[0])
        target_size = (len(sorted_solutions) // 10) * 10
        sampled_solutions = sorted_solutions[:target_size]
        print(f"Adjusted sample count to multiple of 10: {len(sampled_solutions)}")

    sorted_solutions = sorted(sampled_solutions, key=lambda x: x.objectives[0])
    sorted_indices = list(range(len(sorted_solutions)))

    batch_size = 20
    num_batches = (len(sorted_solutions) + batch_size - 1) // batch_size
    batch_indices = get_batch_indices(sorted_indices, batch_size, num_batches)

    t = 1
    t_max = 50
    novelty_archive = []
    all_ft_normalized = []
    all_fa_normalized = []
    age_info = None

    for batch_num in range(num_batches):
        batch_solutions = [sorted_solutions[i] for i in batch_indices[batch_num]]
        batch_ft = [s.objectives[0] for s in batch_solutions]
        batch_vars = [s.variables for s in batch_solutions]

        if batch_vars:
            num_cols = len(batch_vars[0])
            unique_elements_per_column = []
            for col in range(num_cols):
                unique_elements = set()
                for row in batch_vars:
                    unique_elements.add(row[col])
                unique_elements_per_column.append(sorted(unique_elements))
        else:
            unique_elements_per_column = []

        if fa_construction == 'age':
            if batch_num == 0:
                age_info = [i + 1 for i in range(len(batch_solutions))]
            else:
                base_age = batch_size + t - 1
                age_info = [base_age] * len(batch_solutions)
        elif fa_construction == 'novelty':
            update_novelty_archive(batch_solutions, novelty_archive)

        batch_ft_normalized, batch_fa_normalized = generate_fa(
            batch_vars,
            batch_ft,
            fa_construction,
            minimize,
            workflow_file,
            unique_elements_per_column,
            t,
            t_max,
            random_seed,
            age_info=age_info if fa_construction == 'age' else None,
            novelty_archive=novelty_archive if fa_construction in ['novelty'] else None,
            k=min(10, len(batch_solutions) // 2)
        )

        all_ft_normalized.extend(batch_ft_normalized)
        all_fa_normalized.extend(batch_fa_normalized)
        t += 1

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = all_ft_normalized[i]
        sol.attributes['normalized_fa'] = all_fa_normalized[i]

    header = [f'var_{i}' for i in range(problem.number_of_variables)] + \
             ['ft', 'fa', 'normalized_ft', 'normalized_fa']
    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, mode, sampling_method,
        num_samples, random_seed, 'figure1', reverse
    )

    sampled_dict = {
        tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
        for s in sampled_solutions
    }
    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type, sampled_dict, reverse,
        None, None
    )

    r0_points = np.column_stack((all_ft_normalized, all_fa_normalized))
    g1, g2 = transform_points_for_figure2(r0_points)

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, mode, sampling_method,
        num_samples, random_seed, 'figure2', reverse
    )

    sampled_dict_fig2 = {
        tuple(s.variables): (g1[i], g2[i])
        for i, s in enumerate(sampled_solutions)
    }
    plot_figure2(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type, sampled_dict_fig2, reverse,
        None, None
    )


def init_worker():
    pass


class ProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problems(cls, dataset_names: List[str], workflow_base_path: str,
                         random_seed: int = 42) -> Dict[str, ServiceCompositionProblem]:
        instance = cls()
        for dataset in dataset_names:
            if dataset not in instance.problems:
                try:
                    workflow_file = f'{workflow_base_path}{dataset}.csv'
                    instance.problems[dataset] = ServiceCompositionProblem(
                        workflow_file=workflow_file,
                        seed=random_seed,
                        mode='ft_fa'
                    )
                    print(f"Successfully preloaded web service problem: {dataset}")
                except Exception as e:
                    print(f"Failed to load web service problem ({dataset}): {str(e)}")
        return instance.problems

    @classmethod
    def get_problem(cls, dataset_name: str) -> ServiceCompositionProblem:
        instance = cls()
        return instance.problems.get(dataset_name)


def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)

    total_tasks = len(all_tasks)
    for i in range(0, total_tasks, batch_size):
        batch = all_tasks[i:i + batch_size]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_task, **task) for task in batch]

            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f"Task error: {str(e)}")


def process_single_task(mode, dataset_name, sampling_method, num_samples, sample_type,
                        minimize, random_seed, fa_construction, unique_id, reverse,
                        workflow_file, first_sample: bool = False):
    try:
        problem = ProblemManager.get_problem(dataset_name)
        if problem is None:
            raise ValueError(f"Preloaded problem instance not found: {dataset_name}")

        np.random.seed(random_seed)
        random.seed(random_seed)

        if first_sample:
            sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse, debug=True)
            return f"[sampling done] {unique_id}, reverse={reverse}"

        if mode == 'g1':
            process_g1_mode(
                problem, workflow_file, minimize, num_samples, random_seed,
                sampling_method, sample_type, dataset_name, mode, unique_id,
                reverse, first_sample=False
            )
        elif mode in fa_construction:
            process_fa_construction_mode(
                problem, workflow_file, minimize, num_samples, random_seed,
                mode, sampling_method, sample_type, dataset_name, mode,
                unique_id, reverse, first_sample=False
            )

        return f"Task completed: {unique_id}, reverse={reverse}"
    except Exception as e:
        return f"Task failed: {unique_id}, reverse={reverse}, Error: {str(e)}"


def main_wsc_multi(dataset_names, fa_construction, minimize=True,
                   fixed_sample_sizes=[1000],
                   sampling_methods=None,
                   random_seeds=None,
                   use_multiprocessing=True,
                   max_workers=None,
                   reverse=False,
                   first_sample=False,
                   file_base_path='./Datasets/Original_data/',

                   debug=False):
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
    if random_seeds is None:
        random_seeds = range(10)
    if max_workers is None:
        max_workers = 5

    print("Preloading all web service problem instances...")
    for seed in random_seeds:
        ProblemManager.preload_problems(dataset_names, file_base_path, seed)

    if first_sample:
        fa_construction = ["g1"]

    all_tasks = []
    for dataset in dataset_names:
        for mode in fa_construction:
            for sampling_method in sampling_methods:
                for num_sample in fixed_sample_sizes:
                    for random_seed in random_seeds:
                        unique_id = f"{dataset}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}"
                        task = {
                            'mode': mode,
                            'dataset_name': dataset,
                            'sampling_method': sampling_method,
                            'num_samples': num_sample,
                            'sample_type': 'fixed',
                            'minimize': minimize,
                            'random_seed': random_seed,
                            'fa_construction': fa_construction,
                            'unique_id': unique_id,
                            'reverse': reverse,
                            'workflow_file': f'{file_base_path}{dataset}.csv',
                            'first_sample': first_sample,
                        }
                        all_tasks.append(task)

    if use_multiprocessing:
        print(f"Starting multiprocessing for web service problems, total {len(all_tasks)} tasks, reverse={reverse}...")
        process_in_batches(all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        print(f"Starting single-process handling for web service problems, total {len(all_tasks)} tasks, reverse={reverse}...")
        for task in all_tasks:
            print(process_single_task(**task))


if __name__ == "__main__":
    fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    dataset_names = ["workflow_1","workflow_2","workflow_3","workflow_4","workflow_5","workflow_6","workflow_7","workflow_8","workflow_9","workflow_10"]
    main_wsc_multi(
        dataset_names,
        fa_construction,
        use_multiprocessing=True,
        reverse=False,
        first_sample=False,
    )