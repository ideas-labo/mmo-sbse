import csv
import os
from typing import List, Any, Dict
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import warnings
import scipy.stats._qmc as qmc
from itertools import combinations, product
sys.path.insert(0, '/home/ccj/code/mmo')
from Code.NRP.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from Code.NRP.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from Code.NRP.mmo_nrp import NRP_Problem, NRP_MOO, parse

SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'halton', 'latin_hypercube', 'monte_carlo', 'covering_array'
]


class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_score': float('inf'), 'original_cost': float('inf')}


class NRPProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problem(cls, dataset_name: str, data_file_path: str, budget_constraint: float = 0.7,
                        random_seed: int = 42):
        instance = cls()
        problem_key = (dataset_name, random_seed)

        if problem_key not in instance.problems:
            requirements, clients, dependencies = parse(data_file_path)

            problem = NRPProblem(
                requirements=requirements,
                clients=clients,
                dependencies=dependencies,
                budget_constraint=budget_constraint,
                random_seed=random_seed,
                debug=False
            )
            instance.problems[problem_key] = problem
            print(f"Preloaded NRP problem: key={problem_key}, number_of_bits={problem.number_of_bits}")

        return instance.problems[problem_key]

    @classmethod
    def get_problem(cls, dataset_name: str, random_seed: int):
        instance = cls()
        problem_key = (dataset_name, random_seed)
        return instance.problems.get(problem_key, None)


class NRPProblem:
    def __init__(self, requirements, clients, dependencies, budget_constraint, random_seed=None, debug=False):
        self.requirements = requirements.copy()
        self.clients = clients.copy()
        self.dependencies = dependencies.copy()
        self.budget_constraint = budget_constraint
        self.max_budget = self.get_max_budget(self.budget_constraint)
        self.number_of_bits = len(self.clients)

        self.random_seed = random_seed
        self.debug = debug

        self.lower_bound = [0] * self.number_of_bits
        self.upper_bound = [1] * self.number_of_bits

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if debug:
            print(
                f"[NRPProblem] Initialization complete | number_of_bits: {self.number_of_bits} | requirements: {len(self.requirements)} | max_budget: {self.max_budget}"
            )

    def get_selected_requirements(self, customer_candidate):
        return NRP_Problem.get_selected_requirements(self, customer_candidate)

    def simplify_dependencies(self, requirement_candidate):
        return NRP_Problem.simplify_dependencies(self, requirement_candidate)

    def get_score(self, customer_candidate):
        return NRP_Problem.get_score(self, customer_candidate)

    def get_cost(self, customer_candidate):
        return NRP_Problem.get_cost(self, customer_candidate)

    def get_max_budget(self, budget_constraint):
        total_cost = sum(self.requirements)
        return round(total_cost * budget_constraint)

    def ensure_budget_constraint(self, customer_candidate):
        moo_instance = NRP_MOO(
            requirements=self.requirements,
            clients=self.clients,
            dependencies=self.dependencies,
            budget_constraint=self.budget_constraint,
            population_size=100
        )
        return moo_instance._ensure_budget_constraint(customer_candidate)

    def evaluate(self, solution: SolutionWrapper) -> None:
        try:
            customer_candidate = solution.variables[0]

            customer_candidate = self.ensure_budget_constraint(customer_candidate)
            solution.variables[0] = customer_candidate

            original_score = -self.get_score(customer_candidate)
            original_cost = self.get_cost(customer_candidate)

            solution.attributes.update({
                'original_score': original_score,
                'original_cost': original_cost,
                'max_budget': self.max_budget,
            })
            solution.objectives = [original_score, original_cost]

        except Exception as e:
            if self.debug:
                print(f"[Evaluation error] {str(e)}")
            solution.objectives = [float('inf'), float('inf')]


def generate_samples(problem: NRPProblem, num_samples: int, random_seed: int,
                     sampling_method: str = 'random', debug=True):
    np.random.seed(random_seed)
    random.seed(random_seed)

    samples = []
    dimension = problem.number_of_bits
    max_budget = problem.max_budget

    if debug:
        print(
            f"\n[Sampling start] binary customer encoding | target_samples: {num_samples} | dimension: {dimension} | method: {sampling_method}"
        )

    BATCH_THRESHOLD = 50
    BATCH_SIZE = 50
    if dimension <= BATCH_THRESHOLD:
        batches = [(0, dimension)]
    else:
        num_batches = (dimension + BATCH_SIZE - 1) // BATCH_SIZE
        batches = [(i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, dimension)) for i in range(num_batches)]
        if debug:
            print(f"[High-dim handling] dimension {dimension} split into {len(batches)} batches")

    if sampling_method == 'monte_carlo':
        if dimension <= BATCH_THRESHOLD:
            samples = [[random.random() < 0.5 for _ in range(dimension)] for _ in range(num_samples)]
        else:
            batch_samples = []
            for batch_start, batch_end in batches:
                batch_dim = batch_end - batch_start
                batch_samples.append([[random.random() < 0.5 for _ in range(batch_dim)] for _ in range(num_samples)])
            samples = [sum([batch[i] for batch in batch_samples], []) for i in range(num_samples)]

    elif sampling_method in ['sobol', 'orthogonal', 'halton', 'latin_hypercube']:
        samples = generate_qmc_samples(dimension, 2, num_samples, sampling_method, random_seed, batches)

    elif sampling_method == 'covering_array':
        samples = generate_covering_array_samples(dimension, 2, num_samples, random_seed, batches)

    else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}")

    valid_samples = []
    discarded_samples = 0
    for sample in samples:
        customer_candidate = [int(x > 0.5) if sampling_method in ['sobol', 'orthogonal', 'halton', 'latin_hypercube']
                              else int(x) for x in sample]

        customer_candidate = problem.ensure_budget_constraint(customer_candidate)
        current_cost = problem.get_cost(customer_candidate)

        if current_cost <= max_budget:
            valid_samples.append(customer_candidate)
        else:
            discarded_samples += 1
            if debug:
                print(f"[Sample discarded] budget exceeded (cost: {current_cost} > budget: {max_budget})")

    if debug:
        print(f"[Sampling filter] valid_samples: {len(valid_samples)} | discarded: {discarded_samples}")

    sampled_solutions = []
    for sample in valid_samples:
        solution = SolutionWrapper([sample])
        problem.evaluate(solution)
        if solution.attributes['original_score'] != 0:
            sampled_solutions.append(solution)
        if len(sampled_solutions) % 10 == 0 and debug:
            print(f"\rcurrent valid samples: {len(sampled_solutions)}", end="", flush=True)

    seen = set()
    deduplicated_solutions = []
    for sol in sampled_solutions:
        var_tuple = tuple(sol.variables[0])
        if var_tuple not in seen:
            seen.add(var_tuple)
            deduplicated_solutions.append(sol)

    remainder = len(deduplicated_solutions) % 10
    if remainder != 0:
        deduplicated_solutions = deduplicated_solutions[:-remainder]

    if debug:
        print(f"\n[Sampling completed] final samples: {len(deduplicated_solutions)} (multiple of 10)")
    return deduplicated_solutions


def generate_qmc_samples(dimensions: int, upper_bound: int, num_samples: int,
                         sampling_method: str, random_seed: int, batches: List[tuple]) -> List[List[int]]:
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")

    if sampling_method == 'sobol':
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_seed)
    elif sampling_method == 'orthogonal':
        sampler = qmc.LatinHypercube(d=dimensions, optimization="random-cd", seed=random_seed)
    elif sampling_method == 'halton':
        sampler = qmc.Halton(d=dimensions, scramble=True, seed=random_seed)
    elif sampling_method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(d=dimensions, seed=random_seed)
    else:
        raise ValueError(f"Unsupported QMC method: {sampling_method}")

    if num_samples & (num_samples - 1) == 0:
        sample = sampler.random_base2(m=int(np.log2(num_samples)))
    else:
        sample = sampler.random(n=num_samples)

    samples = (sample * upper_bound).astype(int)
    samples = np.clip(samples, 0, upper_bound - 1)
    return samples.tolist()


def generate_covering_array_samples(dimensions: int, upper_bound: int,
                                    num_samples: int, random_seed: int, batches: List[tuple]) -> List[List[int]]:
    np.random.seed(random_seed)
    dim_pairs = list(combinations(range(dimensions), 2))
    required_pairs = set(product(range(upper_bound), range(upper_bound)))

    selected = []
    remaining_pairs = required_pairs.copy()
    while len(selected) < num_samples and remaining_pairs:
        target_val1, target_val2 = random.choice(list(remaining_pairs))
        target_dims = random.choice(dim_pairs)
        candidate = [random.randint(0, upper_bound - 1) for _ in range(dimensions)]
        candidate[target_dims[0]] = target_val1
        candidate[target_dims[1]] = target_val2
        new_covered = {(v1, v2) for (v1, v2) in remaining_pairs
                       if candidate[target_dims[0]] == v1 and candidate[target_dims[1]] == v2}
        selected.append(candidate)
        remaining_pairs -= new_covered

    while len(selected) < num_samples:
        selected.append([random.randint(0, upper_bound - 1) for _ in range(dimensions)])
    return selected[:num_samples]


def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2


def save_sampled_data_to_csv(sampled_solutions: List[SolutionWrapper],
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
        var_values = [1 if x else 0 for x in sol.variables[0]]
        original_ft = sol.attributes.get('ft', sol.objectives[0])
        original_fa = sol.attributes.get('fa', sol.objectives[1])
        normalized_ft = sol.attributes.get('normalized_ft', 0.0)
        normalized_fa = sol.attributes.get('normalized_fa', 0.0)

        if np.isinf(original_ft) or np.isinf(original_fa):
            continue

        row = var_values + [
            round(original_ft, 6),
            round(original_fa, 6),
            round(normalized_ft, 6),
            round(normalized_fa, 6)
        ]
        rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"[Save error] No valid data to write CSV (dataset={dataset_name}, figure_type={figure_type})")

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[CSV saved] {len(rows)} rows -> {filename}")


def load_sampled_data_from_csv(dataset_name: str, mode: str, sampling_method: str,
                               num_samples: int, random_seed: int, figure_type: str,
                               reverse: bool = False) -> List[SolutionWrapper]:
    if reverse:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        filename = f"./Results/Samples_multi/sampled_data_{dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    sampled_solutions = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        var_count = len(header) - 4
        if var_count <= 0:
            raise ValueError(f"[Load error] CSV header invalid, no client columns (file: {filename})")

        for row_idx, row in enumerate(reader, 1):
            if len(row) != len(header):
                print(f"[Warning] Row {row_idx} column count mismatch, skipping (file: {filename})")
                continue

            try:
                bool_vars = [bool(int(x)) for x in row[:var_count]
                             ]
            except ValueError:
                print(f"[Warning] Row {row_idx} invalid client encoding, skipping (file: {filename})")
                continue

            try:
                original_ft = float(row[-4])
                original_fa = float(row[-3])
                normalized_ft = float(row[-2])
                normalized_fa = float(row[-1])
            except ValueError:
                print(f"[Warning] Row {row_idx} invalid numeric values, skipping (file: {filename})")
                continue

            sol = SolutionWrapper([bool_vars])
            sol.objectives = [original_ft, original_fa]
            sol.attributes.update({
                'ft': original_ft,
                'fa': original_fa,
                'normalized_ft': normalized_ft,
                'normalized_fa': normalized_fa
            })
            sampled_solutions.append(sol)

    if len(sampled_solutions) == 0:
        raise ValueError(f"[Load error] No valid data loaded (file: {filename})")

    print(f"[CSV loaded] {len(sampled_solutions)} rows <- {filename}")
    return sampled_solutions


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


def process_g1_mode(problem: NRPProblem, workflow_file, minimize, num_samples, random_seed, sampling_method,
                       sample_type, dataset_name, mode, unique_id, reverse=False, use_saved_data=False):
    header = [f'client_{i}' for i in range(problem.number_of_bits)] + \
             ['ft', 'fa', 'normalized_ft', 'normalized_fa']

    if not use_saved_data:
        sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method)
        if len(sampled_solutions) == 0:
            raise ValueError(f"[Normalize error] No valid samples for normalization (dataset={dataset_name}, seed={random_seed})")

        ft = [sol.objectives[0] for sol in sampled_solutions]
        fa = [sol.objectives[1] for sol in sampled_solutions]

        ft = np.array(ft, dtype=np.float64)
        fa = np.array(fa, dtype=np.float64)
        valid_mask = ~(np.isinf(ft) | np.isinf(fa))
        ft_valid = ft[valid_mask]
        fa_valid = fa[valid_mask]
        sampled_solutions_valid = [sampled_solutions[i] for i in range(len(sampled_solutions)) if valid_mask[i]]

        if len(sampled_solutions_valid) == 0:
            raise ValueError(f"[Normalize error] All sample values invalid (all inf), cannot normalize (dataset={dataset_name})")

        scaler = MinMaxScaler()
        combined_data = np.column_stack((ft_valid, fa_valid))

        normalized_data = scaler.fit_transform(combined_data)

        for i, sol in enumerate(sampled_solutions_valid):
            sol.attributes['normalized_ft'] = normalized_data[i, 0]
            sol.attributes['normalized_fa'] = normalized_data[i, 1]
            sol.attributes['ft'] = ft_valid[i]
            sol.attributes['fa'] = fa_valid[i]

        save_sampled_data_to_csv(
            sampled_solutions_valid, header, dataset_name, mode, sampling_method,
            num_samples, random_seed, 'figure1', reverse
        )

        r0_points = normalized_data
        g1, g2 = transform_points_for_figure2(r0_points)

        for i, sol in enumerate(sampled_solutions_valid):
            sol.attributes['normalized_ft'] = g1[i]
            sol.attributes['normalized_fa'] = g2[i]

        save_sampled_data_to_csv(
            sampled_solutions_valid, header, dataset_name, mode, sampling_method,
            num_samples, random_seed, 'figure2', reverse
        )

        sampled_data = [[1 if x else 0 for x in sol.variables[0]] for sol in sampled_solutions_valid]
        sampled_dict_fig1 = {tuple(sampled_data[i]): (sol.attributes['normalized_ft'], sol.attributes['normalized_fa'])
                             for i, sol in enumerate(sampled_solutions_valid)}
        sampled_dict_fig2 = {tuple(sampled_data[i]): (g1[i], g2[i]) for i in range(len(sampled_data))}

    else:
        sampled_solutions_valid = load_sampled_data_from_csv(
            dataset_name, mode, sampling_method, num_samples, random_seed, 'figure1', reverse
        )
        sampled_data = [[1 if x else 0 for x in sol.variables[0]] for sol in sampled_solutions_valid]
        sampled_dict_fig1 = {tuple(sampled_data[i]): (sol.attributes['normalized_ft'], sol.attributes['normalized_fa'])
                             for i, sol in enumerate(sampled_solutions_valid)}

        sampled_solutions_fig2 = load_sampled_data_from_csv(
            dataset_name, mode, sampling_method, num_samples, random_seed, 'figure2', reverse
        )
        g1 = [sol.attributes['normalized_ft'] for sol in sampled_solutions_fig2]
        g2 = [sol.attributes['normalized_fa'] for sol in sampled_solutions_fig2]
        sampled_dict_fig2 = {tuple(sampled_data[i]): (g1[i], g2[i]) for i in range(len(sampled_data))}

    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type,
        sampled_dict_fig1, reverse, None, None
    )
    plot_figure2(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type,
        sampled_dict_fig2, reverse, None, None
    )


def process_fa_construction_mode(problem: NRPProblem, workflow_file, minimize, num_samples, random_seed,
                                 fa_construction, sampling_method, sample_type, dataset_name, mode, unique_id,
                                 reverse=False, use_saved_data=False):
    header = [f'client_{i}' for i in range(problem.number_of_bits)] + \
             ['ft', 'fa', 'normalized_ft', 'normalized_fa']

    try:
        g1_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse
        )
    except Exception as e:
        raise FileNotFoundError(f"[FA mode error] g1 base data not found: {str(e)} -> run g1 first")

    valid_solutions = [sol for sol in g1_solutions
                       if not (sol.attributes['normalized_ft'] == 0 and sol.attributes['normalized_fa'] == 0)]
    if len(valid_solutions) < 10:
        raise ValueError(f"[FA mode error] insufficient valid samples ({len(valid_solutions)} < 10), cannot batch (dataset={dataset_name})")

    target_count = (len(valid_solutions) // 10) * 10
    g1_solutions = valid_solutions[:target_count]
    sorted_indices = list(range(len(g1_solutions)))

    batch_size = 20
    num_batches = (len(g1_solutions) + batch_size - 1) // batch_size
    batch_indices = get_batch_indices(sorted_indices, batch_size, num_batches)

    t, t_max = 1, 1000
    novelty_archive = []
    all_ft_norm, all_fa_norm = [], []
    age_info = None

    for batch_num in range(num_batches):
        batch_sols = [g1_solutions[i] for i in batch_indices[batch_num]]
        if len(batch_sols) == 0:
            continue

        batch_ft = [sol.attributes['ft'] for sol in batch_sols]
        batch_vars = [sol.variables[0] for sol in batch_sols]

        unique_elements_per_column = []
        if batch_vars:
            num_cols = len(batch_vars[0])
            unique_elements_per_column = [sorted(set(row[col] for row in batch_vars))
                                          for col in range(num_cols)]

        if fa_construction == 'age':
            age_info = [i + 1 for i in range(len(batch_sols))] if batch_num == 0 else [batch_size + t - 1] * len(
                batch_sols)
        elif fa_construction == 'novelty':
            update_novelty_archive(batch_sols, novelty_archive)

        ft_norm, fa_norm = generate_fa(
            batch_vars, batch_ft, fa_construction, minimize, workflow_file,
            unique_elements_per_column, t, t_max, random_seed,
            age_info=age_info if fa_construction == 'age' else None,
            novelty_archive=novelty_archive if fa_construction == 'novelty' else None,
            k=min(10, len(batch_sols) // 2)
        )

        ft_norm = [0.5 if x == 0 else x for x in ft_norm]
        fa_norm = [0.5 if x == 0 else x for x in fa_norm]

        all_ft_norm.extend(ft_norm)
        all_fa_norm.extend(fa_norm)
        t += 1

    if not use_saved_data:
        for i, sol in enumerate(g1_solutions):
            sol.attributes['normalized_ft'] = all_ft_norm[i]
            sol.attributes['normalized_fa'] = all_fa_norm[i]
            sol.attributes['fa'] = sol.objectives[1]

        save_sampled_data_to_csv(
            g1_solutions, header, dataset_name, mode, sampling_method,
            num_samples, random_seed, 'figure1', reverse
        )

        r0_points = np.column_stack((all_ft_norm, all_fa_norm))
        g1, g2 = transform_points_for_figure2(r0_points)

        for i, sol in enumerate(g1_solutions):
            sol.attributes['normalized_ft'] = g1[i]
            sol.attributes['normalized_fa'] = g2[i]

        save_sampled_data_to_csv(
            g1_solutions, header, dataset_name, mode, sampling_method,
            num_samples, random_seed, 'figure2', reverse
        )

    else:
        g1_solutions = load_sampled_data_from_csv(
            dataset_name, mode, sampling_method, num_samples, random_seed, 'figure1', reverse
        )
        all_ft_norm = [sol.attributes['normalized_ft'] for sol in g1_solutions]
        all_fa_norm = [sol.attributes['normalized_fa'] for sol in g1_solutions]

        fa_solutions_fig2 = load_sampled_data_from_csv(
            dataset_name, mode, sampling_method, num_samples, random_seed, 'figure2', reverse
        )
        g1 = [sol.attributes['normalized_ft'] for sol in fa_solutions_fig2]
        g2 = [sol.attributes['normalized_fa'] for sol in fa_solutions_fig2]

    sampled_data = [[1 if x else 0 for x in sol.variables[0]] for sol in g1_solutions]
    sampled_dict_fig1 = {tuple(sampled_data[i]): (all_ft_norm[i], all_fa_norm[i]) for i, s in enumerate(sampled_data)}
    sampled_dict_fig2 = {tuple(sampled_data[i]): (g1[i], g2[i]) for i, s in enumerate(sampled_data)}

    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type,
        sampled_dict_fig1, reverse, None, None
    )
    plot_figure2(
        random_seed, 'mean', sampling_method, num_samples,
        dataset_name, mode, unique_id, sample_type,
        sampled_dict_fig2, reverse, None, None
    )


def load_nrp_problem(dataset_name: str, data_file_path: str, budget_constraint: float = 0.7, random_seed: int = 42):
    NRPProblemManager.preload_problem(
        dataset_name=dataset_name,
        data_file_path=data_file_path,
        budget_constraint=budget_constraint,
        random_seed=random_seed
    )
    problem = NRPProblemManager.get_problem(dataset_name=dataset_name, random_seed=random_seed)
    if problem is None:
        raise ValueError(f"Failed to get NRP problem: dataset={dataset_name}, seed={random_seed}")
    return problem


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


def sample_and_save(problem: NRPProblem, dataset_name: str, num_samples: int, random_seed: int,
                    sampling_method: str, reverse: bool = False, debug: bool = True):
    header = [f'client_{i}' for i in range(problem.number_of_bits)] + \
             ['ft', 'fa', 'normalized_ft', 'normalized_fa']

    sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method)
    if len(sampled_solutions) == 0:
        raise ValueError(f"[Normalize error] No valid samples for normalization (dataset={dataset_name}, seed={random_seed})")

    ft = [sol.objectives[0] for sol in sampled_solutions]
    fa = [sol.objectives[1] for sol in sampled_solutions]

    ft = np.array(ft, dtype=np.float64)
    fa = np.array(fa, dtype=np.float64)
    valid_mask = ~(np.isinf(ft) | np.isinf(fa))
    ft_valid = ft[valid_mask]
    fa_valid = fa[valid_mask]
    sampled_solutions_valid = [sampled_solutions[i] for i in range(len(sampled_solutions)) if valid_mask[i]]

    if len(sampled_solutions_valid) == 0:
        raise ValueError(f"[Normalize error] All sample values invalid (all inf), cannot normalize (dataset={dataset_name})")

    scaler = MinMaxScaler()
    combined_data = np.column_stack((ft_valid, fa_valid))
    normalized_data = scaler.fit_transform(combined_data)

    for i, sol in enumerate(sampled_solutions_valid):
        sol.attributes['normalized_ft'] = normalized_data[i, 0]
        sol.attributes['normalized_fa'] = normalized_data[i, 1]
        sol.attributes['ft'] = ft_valid[i]
        sol.attributes['fa'] = fa_valid[i]

    save_sampled_data_to_csv(
        sampled_solutions_valid, header, dataset_name, 'g1', sampling_method,
        num_samples, random_seed, 'figure1', reverse
    )

    r0_points = normalized_data
    g1, g2 = transform_points_for_figure2(r0_points)

    for i, sol in enumerate(sampled_solutions_valid):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(
        sampled_solutions_valid, header, dataset_name, 'g1', sampling_method,
        num_samples, random_seed, 'figure2', reverse
    )

    if debug:
        print(f"[sample_and_save] Sampling and saving completed: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")


def process_single_task(mode, dataset_name, sampling_method, num_samples, sample_type,
                        minimize, random_seed, fa_construction, unique_id, reverse,
                        data_file_path, use_saved_data=False, first_sample=False):
    try:
        problem = NRPProblemManager.get_problem(dataset_name=dataset_name, random_seed=random_seed)
        if problem is None:
            problem = load_nrp_problem(
                dataset_name=dataset_name,
                data_file_path=data_file_path,
                random_seed=random_seed
            )

        random.seed(random_seed)
        np.random.seed(random_seed)

        if first_sample:
            sample_and_save(problem, dataset_name, num_samples, random_seed, sampling_method, reverse, debug=True)
            return f"[sampling done] {unique_id}"

        if mode == 'g1':
            try:
                process_g1_mode(
                    problem=problem, workflow_file=data_file_path, minimize=minimize,
                    num_samples=num_samples, random_seed=random_seed, sampling_method=sampling_method,
                    sample_type=sample_type, dataset_name=dataset_name, mode=mode, unique_id=unique_id,
                    reverse=reverse, use_saved_data=True
                )
            except FileNotFoundError as e:
                print(f"[g1] Sampled data not found (dataset={dataset_name}, seed={random_seed}). Run first_sample=True to generate samples. Error: {e}")
                return f"[g1 skipped] {unique_id} - missing sampled data"
        elif mode in fa_construction:
            try:
                process_fa_construction_mode(
                    problem=problem, workflow_file=data_file_path, minimize=minimize,
                    num_samples=num_samples, random_seed=random_seed, fa_construction=mode,
                    sampling_method=sampling_method, sample_type=sample_type, dataset_name=dataset_name,
                    mode=mode, unique_id=unique_id, reverse=reverse, use_saved_data=False
                )
            except FileNotFoundError as e:
                print(f"[FA mode] g1 sampled data not found (dataset={dataset_name}, seed={random_seed}). Run first_sample=True to generate samples. Error: {e}")
                return f"[FA skipped] {unique_id} - missing sampled data"
        else:
            print(f"[Warning] Unknown mode: {mode}, skipping {unique_id}")
            return f"[skipped] {unique_id} - unknown mode"

        return f"[task completed] {unique_id}"
    except Exception as e:
        return f"[task failed] {unique_id} | Error: {str(e)}"


def main_nrp_multi(
        dataset_names=None,
        fa_construction=None,
        minimize=True,
        fixed_sample_sizes=None,
        percentage_sample_sizes=None,
        sampling_methods=None,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=None,
        reverse=False,
        workflow_base_path='../Datasets/',
        use_saved_data=False,
        debug=False,
        first_sample=False
):
    if dataset_names is None:
        dataset_names = ['nrp1', 'nrp2', 'nrp3', 'nrp4', 'nrp5',
                         'nrp-e1', 'nrp-e2', 'nrp-e3', 'nrp-e4',
                         'nrp-g1', 'nrp-g2', 'nrp-g3', 'nrp-g4',
                         'nrp-m1', 'nrp-m2', 'nrp-m3', 'nrp-m4']
    if fa_construction is None:
        fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    if fixed_sample_sizes is None:
        fixed_sample_sizes = [1000]
    if percentage_sample_sizes is None:
        percentage_sample_sizes = [10, 20, 30, 40, 50]
    if sampling_methods is None:
        sampling_methods = SAMPLING_METHODS
    if random_seeds is None:
        random_seeds = range(10)

    all_tasks = []

    print("=" * 50)
    print("Starting preload of all NRP problems...")
    for dataset in dataset_names:
        data_file_path = f'{workflow_base_path}{dataset}.txt'
        if not os.path.exists(data_file_path):
            print(f"[Warning] Dataset missing: {data_file_path} -> skipping")
            continue
        for random_seed in random_seeds:
            try:
                load_nrp_problem(
                    dataset_name=dataset,
                    data_file_path=data_file_path,
                    random_seed=random_seed
                )
            except Exception as e:
                print(f"[Preload failed] dataset={dataset}, seed={random_seed}: {str(e)}")
    print("=" * 50)

    if first_sample:
        fa_construction = ["g1"]

    print("\nGenerating all tasks...")
    for dataset in dataset_names:
        data_file_path = f'{workflow_base_path}{dataset}.txt'
        if not os.path.exists(data_file_path):
            continue
        for mode in fa_construction:
            for sampling_method in sampling_methods:
                for num_sample in fixed_sample_sizes:
                    for random_seed in random_seeds:
                        unique_id = f"{dataset}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}_reverse_{reverse}"
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
                            'data_file_path': data_file_path,
                            'use_saved_data': use_saved_data,
                            'first_sample': first_sample
                        }
                        all_tasks.append(task)

    print(f"Task generation complete: total {len(all_tasks)} tasks")
    if len(all_tasks) == 0:
        print("No tasks to process, exiting")
        return

    if use_multiprocessing:
        max_workers = max_workers or min(multiprocessing.cpu_count(), 50)
        if debug:
            print(f"[main_nrp_multi] multiprocessing mode, max_workers: {max_workers}, batch_size: {max_workers}")
        process_in_batches(
            all_tasks=all_tasks,
            max_workers=max_workers,
            batch_size=max_workers
        )
    else:
        if debug:
            print("[main_nrp_multi] single-process mode")
        for idx, task in enumerate(all_tasks, 1):
            if debug:
                print(f"\nProcessing {idx}/{len(all_tasks)} tasks ({task['unique_id']})")
            result = process_single_task(**task)
            if debug:
                print(f"Task result: {result}")

    print("\n" + "=" * 50)
    print("All tasks completed!")


if __name__ == "__main__":
    fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    dataset_names = ['nrp1', 'nrp2', 'nrp3', 'nrp4', 'nrp5',
                     'nrp-e1', 'nrp-e2', 'nrp-e3', 'nrp-e4',
                     'nrp-g1', 'nrp-g2', 'nrp-g3', 'nrp-g4',
                     'nrp-m1', 'nrp-m2', 'nrp-m3', 'nrp-m4']
    main_nrp_multi(
        dataset_names=dataset_names,
        fa_construction=fa_construction,
        use_multiprocessing=True,
        reverse=False,
        use_saved_data=False,
        workflow_base_path='../Datasets/',
        first_sample=False
    )