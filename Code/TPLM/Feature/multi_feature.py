import concurrent
import csv
import re
from typing import List, Any, Dict
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
from Code.TPLM.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
from Code.TPLM.mmo_tplm import MultiObjectiveKnapsack
from Code.TPLM.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import multiprocessing
from itertools import combinations, product
import scipy.stats._qmc as qmc
import os
from jmetal.core.solution import BinarySolution
from concurrent.futures import ProcessPoolExecutor, as_completed

SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'
]


class SolutionWrapper(BinarySolution):

    def __init__(self, number_of_items: int):
        super().__init__(number_of_variables=1, number_of_objectives=2)
        self.variables[0] = [0] * number_of_items
        self.objectives = [float('inf'), float('inf')]
        self.constraints = [0.0]
        self.attributes = {
            'original_profit': float('-inf'),
            'original_count': float('inf'),
            'total_weight': float('inf'),
            'normalized_ft': 0.0,
            'normalized_fa': 0.0
        }


class KnapsackSampler:

    def __init__(self, number_of_items: int, random_seed=None, debug=False):
        self.number_of_items = number_of_items
        self.debug = debug
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        if self.debug:
            print(f"\n[Init] Knapsack sampler initialized, items: {number_of_items}")

    def sample_solutions(self, n_samples: int) -> List[SolutionWrapper]:
        samples = []
        for _ in range(n_samples):
            solution = SolutionWrapper(self.number_of_items)
            solution.variables[0] = [random.randint(0, 1) for _ in range(self.number_of_items)]
            samples.append(solution)

        if self.debug:
            print(f"[Sampling] Generated {n_samples} initial solutions")
        return samples


class KnapsackProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problems(cls, dataset_names: List[str], data_base_path: str,
                         random_seed: int = 42) -> Dict[str, MultiObjectiveKnapsack]:
        instance = cls()
        for dataset in dataset_names:
            if dataset not in instance.problems:
                try:
                    profits, weights, capacity = cls._load_knapsack_data(f"{data_base_path}{dataset}")
                    number_of_items = len(profits)
                    problem = MultiObjectiveKnapsack(
                        number_of_items=number_of_items,
                        capacity=capacity,
                        profits=profits,
                        weights=weights,
                        mode='ft_fa',
                        t_max=250
                    )
                    instance.problems[dataset] = (problem, profits, weights)
                    print(f"Successfully preloaded knapsack problem: {dataset}, items: {number_of_items}, capacity: {capacity}")
                except Exception as e:
                    print(f"Failed to load {dataset}: {str(e)}")
        return instance.problems

    @classmethod
    def get_problem(cls, dataset_name: str) -> MultiObjectiveKnapsack:
        instance = cls()
        problem_data = instance.problems.get(dataset_name)
        return problem_data[0] if problem_data else None

    @classmethod
    def _load_knapsack_data(cls, file_path: str) -> (List[float], List[int], int):
        profits = []
        weights = []
        capacity = 0

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        first_line = lines[0]
        n_match = re.search(r'(\d+)\s+items', first_line)
        if not n_match:
            raise ValueError(f"First line missing item count: {first_line}")
        n = int(n_match.group(1))

        capacity_line = next((line for line in lines if 'capacity:' in line), None)
        if not capacity_line:
            raise ValueError("Capacity information not found")
        c_match = re.search(r'capacity:\s*\+?(\d+)', capacity_line)
        if not c_match:
            raise ValueError(f"Could not parse capacity: {capacity_line}")
        capacity = int(c_match.group(1))

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('item:'):
                if i + 2 >= len(lines):
                    i += 1
                    continue
                weight_line = lines[i + 1]
                w_match = re.search(r'weight:\s*\+?(\d+)', weight_line)
                weights.append(int(w_match.group(1)) if w_match else 0)
                profit_line = lines[i + 2]
                p_match = re.search(r'profit:\s*\+?([\d.]+)', profit_line)
                profits.append(float(p_match.group(1)) if p_match else 0.0)
                i += 3
            else:
                i += 1

        if len(weights) != n or len(profits) != n:
            raise ValueError(f"Item count mismatch: parsed {len(weights)}, expected {n}")

        return profits, weights, capacity


def deduplicate_samples(sampled_data: List[SolutionWrapper]) -> List[SolutionWrapper]:
    seen = set()
    deduplicated = []

    for sample in sampled_data:
        features = tuple(sample.variables[0])
        if features not in seen:
            seen.add(features)
            deduplicated.append(sample)

    return deduplicated


def generate_binary_samples(dimensions: int, num_samples: int,
                            sampling_method: str, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    samples = []

    if sampling_method == 'monte_carlo':
        for _ in range(num_samples):
            sample = [random.randint(0, 1) for _ in range(dimensions)]
            samples.append(sample)

    elif sampling_method in ['sobol', 'orthogonal', 'latin_hypercube']:
        if sampling_method == 'sobol':
            sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_seed)
        elif sampling_method == 'orthogonal':
            sampler = qmc.LatinHypercube(d=dimensions, optimization="random-cd", seed=random_seed)
        else:
            sampler = qmc.LatinHypercube(d=dimensions, seed=random_seed)

        if num_samples & (num_samples - 1) == 0:
            sample = sampler.random_base2(m=int(np.log2(num_samples)))
        else:
            sample = sampler.random(n=num_samples)

        binary_samples = (sample >= 0.5).astype(int)
        samples = binary_samples.tolist()

    elif sampling_method == 'stratified':
        strata = int(np.ceil(num_samples ** (1 / dimensions))) if dimensions > 0 else 1
        for _ in range(num_samples):
            sample = []
            for dim in range(dimensions):
                stratum = random.randint(0, strata - 1)
                sample.append(0 if stratum < strata / 2 else 1)
            samples.append(sample)

    elif sampling_method == 'covering_array':
        dim_pairs = list(combinations(range(dimensions), 2)) if dimensions > 1 else []
        required_pairs = set(product([0, 1], [0, 1]))

        selected = []
        remaining_pairs = required_pairs.copy()

        while len(selected) < num_samples and remaining_pairs:
            if dim_pairs and remaining_pairs:
                target_val1, target_val2 = random.choice(list(remaining_pairs))
                target_dims = random.choice(dim_pairs)

                candidate = [random.randint(0, 1) for _ in range(dimensions)]
                candidate[target_dims[0]] = target_val1
                candidate[target_dims[1]] = target_val2

                if (target_val1, target_val2) in remaining_pairs:
                    remaining_pairs.remove((target_val1, target_val2))
            else:
                candidate = [random.randint(0, 1) for _ in range(dimensions)]

            selected.append(candidate)

        while len(selected) < num_samples:
            selected.append([random.randint(0, 1) for _ in range(dimensions)])

        samples = selected[:num_samples]

    return samples


def generate_samples(problem: MultiObjectiveKnapsack, num_samples: int, random_seed: int,
                     sampling_method: str = 'random', debug=False) -> List[SolutionWrapper]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    sampled_solutions = []
    number_of_items = problem.number_of_bits
    capacity = problem.capacity
    weights = problem.weights

    if number_of_items <= 0:
        raise ValueError(f"Invalid number of items: {number_of_items}, must be positive")

    if debug:
        print(f"\n[Sampling start] target samples: {num_samples}, items: {number_of_items}")
        print(f"[sampling method] {sampling_method}")
        print(f"[capacity constraint] capacity: {capacity}")

    binary_samples = generate_binary_samples(
        dimensions=number_of_items,
        num_samples=num_samples * 2,
        sampling_method=sampling_method,
        random_seed=random_seed
    )

    valid_samples = []
    discarded_samples = 0
    constraint_error_count = 0

    for sample in binary_samples:
        if len(sample) != number_of_items:
            print(f"Skipping invalid sample (length mismatch: {len(sample)} vs {number_of_items})")
            discarded_samples += 1
            continue

        solution = SolutionWrapper(number_of_items)
        try:
            if not hasattr(solution, 'constraints') or len(solution.constraints) == 0:
                solution.constraints = [0.0]
                if debug:
                    print("Initialized empty constraints list")

            solution.variables[0] = sample

            problem.evaluate(solution)

            if len(solution.constraints) == 0 or solution.constraints[0] is None:
                raise ValueError("Constraint value not correctly assigned")

            total_weight = solution.attributes['total_weight']

            if total_weight <= capacity:
                valid_samples.append(solution)
            else:
                adjusted = False
                candidate = sample.copy()
                selected_indices = [i for i, val in enumerate(candidate) if val == 1]
                original_selected_count = len(selected_indices)

                max_removal = original_selected_count - 1 if original_selected_count > 0 else 0
                if max_removal <= 0:
                    discarded_samples += 1
                    continue

                removal_attempts = 0
                while total_weight > capacity and selected_indices and removal_attempts < 100:
                    removal_attempts += 1

                    remaining_removal = max_removal - (original_selected_count - len(selected_indices))
                    if remaining_removal <= 0:
                        break

                    sample_remove = min(random.randint(1, 5), remaining_removal)
                    if len(selected_indices) < sample_remove:
                        sample_remove = len(selected_indices)

                    to_remove = random.sample(selected_indices, sample_remove)
                    for idx in to_remove:
                        if 0 <= idx < len(candidate):
                            candidate[idx] = 0
                        else:
                            if debug:
                                print(f"Skipping invalid index {idx} (sample length {len(candidate)})")

                    total_weight = sum(weights[i] for i, val in enumerate(candidate) if val == 1)
                    selected_indices = [i for i, val in enumerate(candidate) if val == 1]

                if total_weight <= capacity:
                    try:
                        solution_adjusted = SolutionWrapper(number_of_items)
                        solution_adjusted.variables[0] = candidate
                        if not hasattr(solution_adjusted, 'constraints') or len(solution_adjusted.constraints) == 0:
                            solution_adjusted.constraints = [0.0]
                        problem.evaluate(solution_adjusted)
                        valid_samples.append(solution_adjusted)
                    except Exception as e:
                        print(f"Adjusted sample evaluation failed: {str(e)}")
                        discarded_samples += 1
                else:
                    discarded_samples += 1
                    if debug and discarded_samples % 100 == 0:
                        print(f"Discarded {discarded_samples} unfixable over-capacity samples")

        except IndexError as e:
            print(f"Index error: {str(e)}")
            print(f"Sample length: {len(sample)}, constraints length: {len(solution.constraints) if hasattr(solution, 'constraints') else 'undefined'}")
            discarded_samples += 1
            constraint_error_count += 1
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            discarded_samples += 1

        if len(valid_samples) >= num_samples:
            break

    seen = set()
    unique_solutions = []
    for sol in valid_samples:
        var_tuple = tuple(sol.variables[0])
        if var_tuple not in seen:
            seen.add(var_tuple)
            unique_solutions.append(sol)

    final_count = min(num_samples, len(unique_solutions))
    if final_count % 10 != 0 and final_count > 0:
        final_count = (final_count // 10) * 10

    final_solutions = unique_solutions[:final_count]

    if debug:
        print(f"\n[Sampling completed] valid samples: {len(valid_samples)}, deduped: {len(unique_solutions)}")
        print(f"Final samples: {len(final_solutions)}, discarded: {discarded_samples}")
        if constraint_error_count > 0:
            print(f"Constraint-related errors: {constraint_error_count} times")

    return final_solutions


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
        variables = sol.variables[0]
        row = variables + [
            -sol.attributes['original_profit'],
            sol.attributes['original_count'],
            sol.attributes['normalized_ft'],
            sol.attributes['normalized_fa']
        ]
        rows.append(row)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Sampled data saved to: {filename}")


def load_sampled_data_from_csv(dataset_name: str, mode: str, sampling_method: str,
                               num_samples: int, random_seed: int, figure_type: str,
                               number_of_items: int, reverse: bool = False) -> List[SolutionWrapper]:
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
            original_profit = float(row[-4])
            original_count = float(row[-3])
            normalized_ft = float(row[-2])
            normalized_fa = float(row[-1])

            sol = SolutionWrapper(number_of_items)
            sol.variables[0] = variables
            sol.objectives = [original_profit, original_count]
            sol.attributes['original_profit'] = original_profit
            sol.attributes['original_count'] = original_count
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa

            sampled_solutions.append(sol)

    return sampled_solutions


def sample_and_save(problem: MultiObjectiveKnapsack, dataset_name: str, num_samples: int,
                    random_seed: int, sampling_method: str, reverse: bool = False, debug: bool = False):
    number_of_items = problem.number_of_bits

    header = [f'item_{i}' for i in range(number_of_items)] + \
             ['original_profit', 'original_count', 'normalized_ft', 'normalized_fa']

    sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method)

    ft = [s.objectives[0] for s in sampled_solutions]
    fa = [s.objectives[1] for s in sampled_solutions]

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


def process_g1_mode(problem: MultiObjectiveKnapsack, data_file: str, minimize, num_samples, random_seed,
                       sampling_method, sample_type, dataset_name, mode, unique_id, reverse=False):
    number_of_items = problem.number_of_bits

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1',
            number_of_items, reverse
        )
        sampled_dict = {
            tuple(s.variables[0]): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions
        }
        plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name,
                     mode, unique_id, sample_type, sampled_dict, reverse,
                     None, number_of_items)

        r0_points = np.array([[s.attributes['normalized_ft'], s.attributes['normalized_fa']]
                              for s in sampled_solutions])
        g1, g2 = transform_points_for_figure2(r0_points)

        try:
            sampled_solutions_fig2 = load_sampled_data_from_csv(
                dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure2',
                number_of_items, reverse
            )
            sampled_dict_fig2 = {
                tuple(s.variables[0]): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
                for s in sampled_solutions_fig2
            }
            plot_figure2(random_seed, 'mean', sampling_method, num_samples,
                         dataset_name, mode, unique_id, sample_type, sampled_dict_fig2, reverse,
                         None, number_of_items)
        except FileNotFoundError:
            for i, sol in enumerate(sampled_solutions):
                sol.attributes['normalized_ft'] = g1[i]
                sol.attributes['normalized_fa'] = g2[i]

            sampled_dict_fig2 = {
                tuple(s.variables[0]): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
                for s in sampled_solutions
            }
            plot_figure2(random_seed, 'mean', sampling_method, num_samples,
                         dataset_name, mode, unique_id, sample_type, sampled_dict_fig2, reverse,
                         None, number_of_items)

    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Please run sample_and_save (first_sample=True) to generate sampled data, then run this mode.")
        return


def process_fa_construction_mode(problem: MultiObjectiveKnapsack, data_file: str, minimize, num_samples, random_seed,
                                 fa_construction, sampling_method, sample_type, dataset_name, mode, unique_id,
                                 reverse=False):
    number_of_items = problem.number_of_bits

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1',
            number_of_items, reverse
        )
        print(f"Loaded sampling data from g1: {dataset_name}, {sampling_method}, {random_seed}")
    except FileNotFoundError:
        print(f"g1 base sampled data not found: {dataset_name}, {sampling_method}, {random_seed}")
        print("Please run sample_and_save (first_sample=True) to generate sampled data, then run this mode.")
        return

    if len(sampled_solutions) % 10 != 0:
        sorted_solutions = sorted(
            sampled_solutions,
            key=lambda x: (x.attributes['original_profit'], x.attributes['original_count'])
        )
        target_size = (len(sorted_solutions) // 10) * 10
        sampled_solutions = sorted_solutions[:target_size]
        print(f"Adjusted samples to multiple of 10: {len(sampled_solutions)}")

    sorted_solutions = sorted(
        sampled_solutions,
        key=lambda x: (x.attributes['original_profit'], x.attributes['original_count'])
    )
    sorted_indices = list(range(len(sorted_solutions)))

    batch_size = 20
    num_batches = (len(sorted_solutions) + batch_size - 1) // batch_size
    batch_indices = get_batch_indices(sorted_indices, batch_size, num_batches)

    t = 1
    t_max = 250
    novelty_archive = []
    all_ft_normalized = []
    all_fa_normalized = []
    age_info = None

    for batch_num in range(num_batches):
        batch_solutions = [sorted_solutions[i] for i in batch_indices[batch_num]]
        batch_ft = [s.objectives[0] for s in batch_solutions]
        batch_vars = [s.variables[0] for s in batch_solutions]

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
            data_file,
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

    sampled_dict = {
        tuple(s.variables[0]): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
        for s in sampled_solutions
    }
    plot_figure1(random_seed, 'mean', sampling_method, num_samples,
                 dataset_name, mode, unique_id, sample_type, sampled_dict, reverse,
                 None, number_of_items)

    header = [f'item_{i}' for i in range(number_of_items)] + \
             ['original_profit', 'original_count', 'normalized_ft', 'normalized_fa']

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, mode, sampling_method,
        num_samples, random_seed, 'figure1', reverse
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
        tuple(s.variables[0]): (g1[i], g2[i])
        for i, s in enumerate(sampled_solutions)
    }
    plot_figure2(random_seed, 'mean', sampling_method, num_samples,
                 dataset_name, mode, unique_id, sample_type, sampled_dict_fig2, reverse,
                 None, number_of_items)


def init_worker():
    pass


def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2


def process_single_task(mode, dataset_name, full_dataset_path, sampling_method, num_samples, sample_type,
                        minimize, random_seed, fa_construction, unique_id, reverse,
                        data_file, first_sample=False):
    try:
        problem = KnapsackProblemManager.get_problem(full_dataset_path)
        if problem is None:
            raise ValueError(f"Preloaded problem instance not found: {full_dataset_path}")

        np.random.seed(random_seed)
        random.seed(random_seed)

        if first_sample:
            sample_and_save(problem, dataset_name, num_samples, random_seed, sampling_method, reverse, debug=False)
            return f"Sampling and saved: {unique_id}"

        if mode == 'g1':
            process_g1_mode(
                problem, data_file, minimize, num_samples, random_seed,
                sampling_method, sample_type, dataset_name,
                mode, unique_id, reverse
            )
        else:
            process_fa_construction_mode(
                problem, data_file, minimize, num_samples, random_seed,
                mode, sampling_method, sample_type, dataset_name,
                mode, unique_id, reverse
            )

        return f"Task completed: {unique_id}"
    except Exception as e:
        return f"Task failed: {unique_id}, Error: {str(e)}"


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


def main_tplm_multi(dataset_names, fa_construction, minimize=True,
                    fixed_sample_sizes=[1000],
                    percentage_sample_sizes=[10, 20, 30, 40, 50],
                    sampling_methods=None,
                    use_multiprocessing=True,
                    max_workers=None,
                    reverse=False,
                    first_sample=False,
                    data_base_path="../Datasets/",
                    random_seeds=None):
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
    if random_seeds is None:
        random_seeds = range(0, 10)

    print(f"TPLM multi-objective feature configuration:")
    print(f"  Number of datasets: {len(dataset_names)}")
    print(f"  Sampling methods: {sampling_methods}")
    print(f"  Random seeds: {list(random_seeds)}")
    print(f"  Sample sizes: {fixed_sample_sizes}")
    print(f"  FA constructions: {fa_construction}")
    print(f"  Only sampling mode: {first_sample}")

    dataset_info = []
    for dataset_name in dataset_names:
        try:
            parts = dataset_name.split('_')
            if len(parts) >= 2:
                rule = '_'.join(parts[:-1])
                category = 'input_' + parts[-1]

                for run_idx in random_seeds:
                    full_path = f"{rule}/pop_size_250/input/{category}/run_{run_idx}/knapsack_file"
                    simplified_name = dataset_name
                    dataset_info.append((full_path, simplified_name, run_idx))
            else:
                print(f"Warning: dataset name format incorrect: {dataset_name}")
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue

    print(f"Generated {len(dataset_info)} dataset instances")

    print("Preloading all knapsack problems...")
    for full_path, simplified_name, seed in dataset_info:
        if full_path not in KnapsackProblemManager().problems:
            try:
                KnapsackProblemManager.preload_problems([full_path], data_base_path, seed)
                print(f"Preloaded: {full_path}")
            except Exception as e:
                print(f"Preload failed {full_path}: {e}")

    if first_sample:
        fa_construction = ["g1"]
        print("Only sampling mode enabled; FA construction will be skipped")

    all_tasks = []
    for full_path, simplified_name, random_seed in dataset_info:
        for mode in fa_construction:
            for sampling_method in sampling_methods:
                for num_sample in fixed_sample_sizes:
                    unique_id = f"{simplified_name}_{sampling_method}_{num_sample}_fixed_seed_{random_seed}_{mode}"
                    task = {
                        'mode': mode,
                        'dataset_name': simplified_name,
                        'full_dataset_path': full_path,
                        'sampling_method': sampling_method,
                        'num_samples': num_sample,
                        'sample_type': 'fixed',
                        'minimize': minimize,
                        'random_seed': random_seed,
                        'fa_construction': fa_construction,
                        'unique_id': unique_id,
                        'reverse': reverse,
                        'data_file': os.path.join(data_base_path, full_path),
                        'first_sample': first_sample
                    }
                    all_tasks.append(task)

    print(f"Generated {len(all_tasks)} tasks")

    if max_workers is None:
        max_workers = 5

    if use_multiprocessing:
        print(f"Starting multiprocessing for tasks, total {len(all_tasks)} tasks, reverse={reverse}...")
        process_in_batches(all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        print(f"Starting single-process handling for tasks, total {len(all_tasks)} tasks, reverse={reverse}...")
        for task in all_tasks:
            print(process_single_task(**task))


if __name__ == "__main__":
    migration_rules = ["migrationRule1", "migrationRule2", "migrationRule3", "migrationRule4", "migrationRule5",
                       "migrationRule7", "migrationRule8",
                       "migrationRule10", "migrationRule18"]
    input_folders = ['input_ALL', 'input_CO', 'input_MS', 'input_DS']

    dataset_names = []
    for rule in migration_rules:
        for folder in input_folders:
            dataset_names.append(f"{rule}_{folder}")

    fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']

    main_tplm_multi(
        dataset_names=dataset_names,
        fa_construction=fa_construction,
        use_multiprocessing=True,
        reverse=False,
        first_sample=False,
        data_base_path="../Datasets/"
    )