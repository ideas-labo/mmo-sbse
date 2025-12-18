import csv
from typing import List, Any, Dict, Tuple
import numpy as np
import random
import warnings
import scipy.stats._qmc as qmc
import os
from itertools import combinations, product

from evoxbench.benchmarks import NASBench201Benchmark, NATSBenchmark
from evoxbench.test_suites import c10mop, citysegmop, in1kmop
from jmetal.core.solution import IntegerSolution
import sys

sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/mydrive/ccj/code/mmo')

from Code.NAS.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from Code.NAS.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
from Code.NAS.mmo_nas import C10MOPProblem, CitySegMOPProblem, In1KMOPProblem

SAMPLING_METHODS = [
    'sobol','orthogonal','stratified','latin_hypercube','monte_carlo','covering_array'
]

EVOXBENCH_PROBLEMS = {
    'c10mop': {
        'class': C10MOPProblem,
        'ids': [1, 3, 5, 8, 10, 11, 12, 13],
        'search_space_configs': {
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
    },
    'citysegmop': {
        'class': CitySegMOPProblem,
        'ids': [3],
        'search_space_configs': {
            1: [],
            2: [],
            3: [[0, 2]],
            4: [],
            6: [],
            9: [],
        }
    },
    'in1kmop': {
        'class': In1KMOPProblem,
        'ids': [1, 4, 7],
        'search_space_configs': {
            1: [[0, 1]],
            2: [],
            4: [[0, 1]],
            5: [],
            7: [[0, 1]],
            8: [],
            9: []
        }
    }
}

class EvoXSolutionWrapper:
    def __init__(self, variables, problem):
        self.variables = variables
        self.objectives = [float('inf')] * problem.number_of_objectives()
        self.attributes = {
            'original_objectives': [float('inf')] * problem.benchmark.evaluator.n_objs,
            'selected_objectives': [float('inf')] * 2,
            'normalized_ft': float('inf'),
            'normalized_fa': float('inf')
        }

class EvoXProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problems(cls, problem_types, problem_ids, random_seed=42):
        instance = cls()
        for problem_type in problem_types:
            config = EVOXBENCH_PROBLEMS[problem_type]
            for pid in problem_ids[problem_type]:
                pid_configs = config['search_space_configs'].get(pid, [])
                valid_configs = [cfg for cfg in pid_configs if cfg]
                if not valid_configs:
                    print(f"Warning: {problem_type}_{pid} has no valid configuration, skipping")
                    continue
                for config_idx, selected_objs in enumerate(valid_configs):
                    key = f"{problem_type}{pid}_{config_idx}"
                    if key not in instance.problems:
                        try:
                            if problem_type=='c10mop':
                                if pid == 10:
                                    benchmark = NASBench201Benchmark(
                                        200, objs='err&params&flops&edgegpu_latency&edgegpu_energy', dataset='cifar100',
                                        normalized_objectives=True)
                                elif pid == 11:
                                    benchmark = NASBench201Benchmark(
                                        200, objs='err&params&flops&edgegpu_latency&edgegpu_energy',
                                        dataset='ImageNet16-120',
                                        normalized_objectives=True)
                                elif pid == 12:
                                    benchmark = NATSBenchmark(
                                        90, objs='err&params&flops&latency', dataset='cifar100',
                                        normalized_objectives=True)
                                elif pid == 13:
                                    benchmark = NATSBenchmark(
                                        90, objs='err&params&flops&latency', dataset='ImageNet16-120',
                                        normalized_objectives=True)
                                else:
                                    benchmark = globals()[problem_type](pid)
                            else:
                                benchmark = globals()[problem_type](pid)
                            problem = config['class'](
                                benchmark=benchmark,
                                mode='ft_fa',
                                selected_objs=selected_objs,
                                random_seed=random_seed
                            )
                            instance.problems[key] = {
                                'problem': problem,
                                'benchmark': benchmark,
                                'config': selected_objs,
                                'config_idx': config_idx,
                                'selected_objs': selected_objs
                            }
                            print(f"Successfully preloaded: {key}")
                        except Exception as e:
                            print(f"Failed to load {key}: {str(e)}")
        return instance.problems

    @classmethod
    def get_problem(cls, problem_type, pid, config_idx):
        instance = cls()
        key = f"{problem_type}{pid}_{config_idx}"
        return instance.problems.get(key)

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

def generate_stratified_samples(dimensions: int, upper_bound: int,
                                num_samples: int, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)

    if num_samples <= 20:
        strat_dims = 10
    else:
        strat_dims = min(20, dimensions)
    samples = []
    if dimensions<strat_dims:
        strat_dims=dimensions-2
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

def generate_qmc_samples(dimensions: int, upper_bound: int, num_samples: int,
                         sampling_method: str, random_seed: int) -> List[List[int]]:
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")

    if sampling_method == 'sobol':
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_seed)
        if num_samples & (num_samples - 1) == 0:
            sample = sampler.random_base2(m=int(np.log2(num_samples)))
        else:
            sample = sampler.random(n=num_samples)
    elif sampling_method == 'orthogonal':
        sampler = qmc.LatinHypercube(d=dimensions, optimization="random-cd", seed=random_seed)
        sample = sampler.random(n=num_samples)
    elif sampling_method == 'stratified':
        return generate_stratified_samples(dimensions, upper_bound, num_samples, random_seed)
    elif sampling_method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(d=dimensions, seed=random_seed)
        sample = sampler.random(n=num_samples)
    else:
        raise ValueError(f"Unsupported QMC sampling method: {sampling_method}")

    samples = (sample * upper_bound).astype(int)
    samples = np.clip(samples, 0, upper_bound - 1)

    return samples.tolist()

def generate_evox_samples(problem_data, num_samples, random_seed, sampling_method='random', debug=False):
    problem = problem_data['problem']
    problem.true_eval = True
    benchmark = problem_data['benchmark']
    search_space = benchmark.search_space
    valid_solutions = []

    np.random.seed(random_seed)
    random.seed(random_seed)

    if debug:
        print(f"\n[Sampling start] Target samples: {num_samples}, Problem: {problem.name()}, Eval mode: test set (true_eval=True)")

    dimensions = problem.number_of_variables()
    lb = search_space.lb
    ub = search_space.ub

    max_attempts = num_samples * 100
    attempts = 0

    while len(valid_solutions) < num_samples and attempts < max_attempts:
        batch_size = min(num_samples - len(valid_solutions), 100)
        samples = []

        if sampling_method in ['sobol', 'orthogonal', 'stratified', 'latin_hypercube']:
            upper_bounds = [ub[i] - lb[i] + 1 for i in range(dimensions)]
            batch_samples = generate_qmc_samples(
                dimensions,
                max(upper_bounds),
                batch_size,
                sampling_method,
                random_seed + attempts
            )

            for i in range(len(batch_samples)):
                for j in range(dimensions):
                    batch_samples[i][j] = max(lb[j], min(batch_samples[i][j], ub[j]))
            samples.extend(batch_samples)

        elif sampling_method == 'covering_array':
            max_range = max([ub[i] - lb[i] for i in range(dimensions)]) + 1
            batch_samples = generate_covering_array_samples(
                dimensions,
                max_range,
                batch_size,
                random_seed + attempts
            )
            for i in range(len(batch_samples)):
                for j in range(dimensions):
                    batch_samples[i][j] = max(lb[j], min(batch_samples[i][j], ub[j]))
            samples.extend(batch_samples)

        else:
            for _ in range(batch_size):
                variables = [
                    np.random.randint(lb[i], ub[i] + 1)
                    for i in range(dimensions)
                ]
                samples.append(variables)

        for variables in samples:
            if len(valid_solutions) >= num_samples:
                break

            solution_wrapper = EvoXSolutionWrapper(variables, problem)

            jmetal_solution = IntegerSolution(
                lower_bound=lb,
                upper_bound=ub,
                number_of_objectives=problem.number_of_objectives()
            )
            jmetal_solution.variables = variables

            try:
                evaluated_solution = problem.evaluate(jmetal_solution)

                solution_wrapper.objectives = evaluated_solution.attributes[
                    'selected_objectives'].copy()
                solution_wrapper.attributes['original_objectives'] = evaluated_solution.attributes[
                    'original_objectives'].copy()
                solution_wrapper.attributes['selected_objectives'] = evaluated_solution.attributes[
                    'selected_objectives'].copy()

                if not np.isinf(solution_wrapper.attributes['selected_objectives']).any():
                    var_tuple = tuple(solution_wrapper.variables)
                    if not any(tuple(sol.variables) == var_tuple for sol in valid_solutions):
                        valid_solutions.append(solution_wrapper)

            except Exception as e:
                if debug:
                    print(f"Evaluation error: {str(e)}")

            attempts += 1

    if len(valid_solutions) % 10 != 0:
        remainder = len(valid_solutions) % 10
        if remainder > 0:
            valid_solutions = valid_solutions[:-remainder]

    if debug:
        print(f"Sampling completed, valid samples: {len(valid_solutions)}/{num_samples}, total attempts: {attempts}, eval mode: test set (true_eval=True)")

    return valid_solutions[:num_samples]

def save_sampled_data_to_csv(sampled_solutions: List[EvoXSolutionWrapper],
                             header: List[str], dataset_name: str, mode: str,
                             sampling_method: str, num_samples: int, random_seed: int,
                             figure_type: str, reverse: bool = False) -> None:
    if mode=='g1':
        if reverse:
            filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
        else:
            filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    else:
        if reverse:
            filename = f"./Results/Samples_multi_fa/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
        else:
            filename = f"./Results/Samples_multi_fa/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    rows = []
    for sol in sampled_solutions:
        row = sol.variables + [
            sol.attributes['selected_objectives'][0],
            sol.attributes['selected_objectives'][1],
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
                               problem, reverse: bool = False) -> List[EvoXSolutionWrapper]:
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
            variables = list(map(int, row[:-4]))
            original_ft = float(row[-4])
            original_fa = float(row[-3])
            normalized_ft = float(row[-2])
            normalized_fa = float(row[-1])

            sol = EvoXSolutionWrapper(variables, problem)
            sol.objectives = [original_ft, original_fa]
            sol.attributes['original_objectives'] = [float('inf')] * problem.benchmark.evaluator.n_objs
            sol.attributes['selected_objectives'] = [original_ft, original_fa]
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa

            sampled_solutions.append(sol)

    return sampled_solutions

def transform_points_for_figure2(r0_points) -> Tuple[np.ndarray, np.ndarray]:
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2

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

def sample_and_save(problem_data, num_samples, random_seed, sampling_method,
                    dataset_name, reverse=False, debug=False):
    problem = problem_data['problem']

    header = [f'var_{i}' for i in range(problem.number_of_variables())] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    sampled_solutions = generate_evox_samples(
        problem_data, num_samples, random_seed, sampling_method, debug
    )

    if not sampled_solutions:
        print(f"Warning: No valid solutions generated - sample_and_save for {dataset_name}")
        return

    ft = [s.attributes['selected_objectives'][0] for s in sampled_solutions]
    fa = [s.attributes['selected_objectives'][1] for s in sampled_solutions]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(np.column_stack((ft, fa)))

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = normalized[i, 0]
        sol.attributes['normalized_fa'] = normalized[i, 1]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, 'g1', sampling_method,
        num_samples, random_seed, 'figure1', reverse
    )

    r0_points = normalized
    g1, g2 = transform_points_for_figure2(r0_points)

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, 'g1', sampling_method,
        num_samples, random_seed, 'figure2', reverse
    )

    if debug:
        print(f"[sample_and_save] Sampling and saving completed: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")

def process_g1_mode_evox(problem_data, num_samples, random_seed, sampling_method,
                            dataset_name, mode, unique_id, reverse=False, debug=False):
    problem = problem_data['problem']

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1',
            problem, reverse
        )
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Please run sample_and_save (first_sample=True) to generate sampled data before running this mode.")
        return

    sampled_dict = {
        tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
        for s in sampled_solutions
    }
    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples, dataset_name,
        mode, unique_id, 'fixed', sampled_dict, reverse
    )

    try:
        sampled_solutions_fig2 = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure2',
            problem, reverse
        )
        sampled_dict_fig2 = {
            tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions_fig2
        }
        plot_figure2(
            random_seed, 'mean', sampling_method, num_samples,
            dataset_name, mode, unique_id, 'fixed', sampled_dict_fig2, reverse
        )
    except FileNotFoundError:
        r0_points = np.array([[s.attributes['normalized_ft'], s.attributes['normalized_fa']]
                              for s in sampled_solutions])
        g1, g2 = transform_points_for_figure2(r0_points)
        for i, sol in enumerate(sampled_solutions):
            sol.attributes['normalized_ft'] = g1[i]
            sol.attributes['normalized_fa'] = g2[i]
        sampled_dict_fig2 = {
            tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions
        }
        plot_figure2(
            random_seed, 'mean', sampling_method, num_samples,
            dataset_name, mode, unique_id, 'fixed', sampled_dict_fig2, reverse
        )

    if debug:
        print(f"[g1 mode] Processing completed: {unique_id}, valid samples: {len(sampled_solutions)}")

def process_fa_construction_mode_evox(problem_data, num_samples, random_seed, fa_construction,
                                      sampling_method, dataset_name, mode, unique_id, reverse=False,
                                      use_saved_data=False, debug=False):
    try:
        problem = problem_data['problem']
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1',
            problem, reverse
        )
        print(f"Loaded data from g1 mode: {dataset_name}, {sampling_method}, {random_seed}")
    except FileNotFoundError:
        print(f"g1 base sampled data not found: {dataset_name}, {sampling_method}, {random_seed}")
        print("Please run sample_and_save (first_sample=True) to generate sampled data before running this mode.")
        return

    if not sampled_solutions:
        print(f"Warning: Loaded samples are empty - {unique_id}")
        return

    problem = problem_data['problem']
    benchmark = problem_data['benchmark']

    if len(sampled_solutions) % 10 != 0:
        sorted_solutions = sorted(sampled_solutions, key=lambda x: sum(x.attributes['selected_objectives']))
        target_size = (len(sorted_solutions) // 10) * 10
        sampled_solutions = sorted_solutions[:target_size]
        if debug:
            print(f"Initial adjustment to multiple of 10: {len(sampled_solutions)}")

    batch_size = 20
    num_batches = (len(sampled_solutions) + batch_size - 1) // batch_size
    sorted_with_indices = sorted(enumerate(sampled_solutions), key=lambda x: x[1].attributes['selected_objectives'][0])
    sorted_indices = [idx for idx, _ in sorted_with_indices]
    sorted_solutions = [sol for _, sol in sorted_with_indices]
    batch_indices = get_batch_indices(sorted_indices, batch_size, num_batches)

    t = 1
    t_max = 100
    novelty_archive = []
    all_ft_normalized = []
    all_fa_normalized = []
    all_original_indices = []
    age_info = None

    for batch_num in range(num_batches):
        batch_data = [sorted_with_indices[i] for i in batch_indices[batch_num]]
        batch_original_indices = [idx for idx, _ in batch_data]
        batch_solutions = [sol for _, sol in batch_data]
        batch_ft = [s.attributes['selected_objectives'][0] for s in batch_solutions]
        batch_vars = [s.variables for s in batch_solutions]

        if debug:
            print(f"Processing batch {batch_num + 1}/{num_batches}, samples: {len(batch_solutions)}")

        unique_elements_per_column = []
        if batch_vars:
            num_cols = len(batch_vars[0])
            for col in range(num_cols):
                unique_elements = set(row[col] for row in batch_vars)
                unique_elements_per_column.append(sorted(unique_elements))

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
            minimize=True,
            file_path='',
            unique_elements_per_column=unique_elements_per_column,
            t=t,
            t_max=t_max,
            random_seed=random_seed,
            age_info=age_info if fa_construction == 'age' else None,
            novelty_archive=novelty_archive if fa_construction in ['novelty', 'age'] else None,
            k=min(10, len(batch_solutions) // 2)
        )

        all_ft_normalized.extend(batch_ft_normalized)
        all_fa_normalized.extend(batch_fa_normalized)
        all_original_indices.extend(batch_original_indices)
        t += 1

    if fa_construction == 'reciprocal':
        ft_arr = np.array(all_ft_normalized, dtype=np.float64)
        fa_arr = np.array(all_fa_normalized, dtype=np.float64)
        indices_arr = np.array(all_original_indices, dtype=np.int64)

        valid_mask = ~(
            np.isnan(ft_arr) | np.isinf(ft_arr) |
            np.isnan(fa_arr) | np.isinf(fa_arr)
        )

        before_clean = len(valid_mask)
        after_clean = np.sum(valid_mask)
        if debug:
            print(f"[reciprocal clean] Before: {before_clean}, NaN/Inf removed: {before_clean - after_clean}, After: {after_clean}")

        if after_clean == 0:
            print(f"Error: reciprocal mode cleaned all samples - {unique_id}")
            return
        ft_arr_clean = ft_arr[valid_mask]
        fa_arr_clean = fa_arr[valid_mask]
        indices_clean = indices_arr[valid_mask]

        valid_original_indices = list(set(indices_clean))
        valid_original_indices.sort()
        sampled_solutions_clean = [sampled_solutions[idx] for idx in valid_original_indices]
        all_ft_normalized = [ft_arr_clean[np.where(indices_clean == idx)[0][0]] for idx in valid_original_indices]
        all_fa_normalized = [fa_arr_clean[np.where(indices_clean == idx)[0][0]] for idx in valid_original_indices]

        sampled_solutions = sampled_solutions_clean
        if debug:
            print(f"[reciprocal clean completed] Final valid samples: {len(sampled_solutions)}")

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = float(all_ft_normalized[i])
        sol.attributes['normalized_fa'] = float(all_fa_normalized[i])

    header = [f'var_{i}' for i in range(problem.number_of_variables())] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']
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
        dataset_name, mode, unique_id, 'fixed', sampled_dict, reverse
    )

    r0_points = np.column_stack((all_ft_normalized, all_fa_normalized))
    g1, g2 = transform_points_for_figure2(r0_points)

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = float(g1[i])
        sol.attributes['normalized_fa'] = float(g2[i])

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
        dataset_name, mode, unique_id, 'fixed', sampled_dict_fig2, reverse
    )

    if debug:
        print(f"[fa construction mode] Processing completed: {unique_id}, final valid samples: {len(sampled_solutions)}")

def init_worker():
    pass

def process_single_task_evox(mode, problem_type, pid, config_idx, sampling_method, num_samples, sample_type,
                             minimize, random_seed, fa_construction, unique_id, reverse,
                             use_saved_data=False, debug=False, first_sample=False):
    try:
        problem_data = EvoXProblemManager.get_problem(problem_type, pid, config_idx)
        if problem_data is None:
            raise ValueError(f"Preloaded problem instance not found: {problem_type}{pid}_{config_idx}")

        np.random.seed(random_seed)
        random.seed(random_seed)

        dataset_name = f"{problem_type}{pid}_{config_idx}"

        if first_sample:
            sample_and_save(problem_data, num_samples, random_seed, sampling_method, dataset_name, reverse, debug)
            return f"Sampling and saving completed: {unique_id}"

        if mode == 'g1':
            process_g1_mode_evox(
                problem_data, num_samples, random_seed, sampling_method,
                dataset_name, mode, unique_id, reverse, debug
            )
        elif mode in fa_construction:
            process_fa_construction_mode_evox(
                problem_data, num_samples, random_seed, mode, sampling_method,
                dataset_name, mode, unique_id, reverse, use_saved_data, debug
            )

        return f"Task completed: {unique_id}"
    except Exception as e:
        return f"Task failed: {unique_id}, Error: {str(e)}"

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)

    total_tasks = len(all_tasks)
    for i in range(0, total_tasks, batch_size):
        batch = all_tasks[i:i + batch_size]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_task_evox, **task) for task in batch]

            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f"Task error: {str(e)}")

def main_evox_multi(
        problem_types=None,
        problem_ids=None,
        fa_construction=None,
        minimize=True,
        fixed_sample_sizes=None,
        sampling_methods=None,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=None,
        reverse=False,
        use_saved_data=False,
        debug=False,
        first_sample=False
):
    if problem_types is None:
        problem_types = ['c10mop', 'citysegmop', 'in1kmop']
    if problem_ids is None:
        problem_ids = {
            'c10mop': EVOXBENCH_PROBLEMS['c10mop']['ids'],
            'citysegmop': EVOXBENCH_PROBLEMS['citysegmop']['ids'],
            'in1kmop': EVOXBENCH_PROBLEMS['in1kmop']['ids']
        }
    if fa_construction is None:
        fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if fixed_sample_sizes is None:
        fixed_sample_sizes = [1000]
    if sampling_methods is None:
        sampling_methods = SAMPLING_METHODS
    if random_seeds is None:
        random_seeds = range(0, 10)

    if first_sample:
        fa_construction = ["g1"]

    print("Preloading all evoxbench problems (including config indices)...")
    for seed in random_seeds:
        EvoXProblemManager.preload_problems(problem_types, problem_ids, seed)

    all_tasks = []
    for problem_type in problem_types:
        current_problem_cfg = EVOXBENCH_PROBLEMS[problem_type]
        for pid in problem_ids.get(problem_type, []):
            pid_configs = current_problem_cfg['search_space_configs'].get(pid, [])
            valid_configs = [cfg for cfg in pid_configs if cfg]
            if not valid_configs:
                print(f"Warning: {problem_type}{pid} has no valid configuration, skipping PID")
                continue

            for config_idx in range(len(valid_configs)):
                for mode in fa_construction:
                    for sampling_method in sampling_methods:
                        for num_sample in fixed_sample_sizes:
                            for random_seed in random_seeds:
                                unique_id = f"{problem_type}{pid}_{config_idx}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}_reverse_{reverse}"
                                task = {
                                    'mode': mode,
                                    'problem_type': problem_type,
                                    'pid': pid,
                                    'config_idx': config_idx,
                                    'sampling_method': sampling_method,
                                    'num_samples': num_sample,
                                    'sample_type': 'fixed',
                                    'minimize': minimize,
                                    'random_seed': random_seed,
                                    'fa_construction': fa_construction,
                                    'unique_id': unique_id,
                                    'reverse': reverse,
                                    'use_saved_data': use_saved_data,
                                    'debug': debug,
                                    'first_sample': first_sample
                                }
                                all_tasks.append(task)

    print(f"Generated {len(all_tasks)} tasks (including config indices)")

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 50)

    if use_multiprocessing:
        print(f"Starting multiprocessing for {len(all_tasks)} tasks using {max_workers} workers...")
        process_in_batches(all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        for task in all_tasks:
            print(process_single_task_evox(**task))

if __name__ == "__main__":
    main_evox_multi()