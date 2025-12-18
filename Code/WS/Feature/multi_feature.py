from typing import List, Any, Dict
import sys

from scipy.stats import qmc

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
from Code.WS.Utils.DAGCentralSimulator_class import DAGCentralSimulator
from Code.WS.Utils.Load_workflow_class import PegasusWorkflowLoader
from Code.WS.Utils.MOHEFT_class import INFRA_CONFIG, MOHEFT
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import multiprocessing
import sys
from Code.WS.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
sys.path.append('../')
sys.path.append('../..')
from Code.WS.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from itertools import combinations, product
import warnings
import scipy.stats._qmc as qmc
import csv
import os
from collections import defaultdict
from functools import partial
from typing import Dict, Any
import concurrent.futures

class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_makespan': float('inf'), 'original_cost': float('inf')}

class DependencyAwareSampler:
    def __init__(self, dag, random_seed=None, debug=False):
        self.dag = dag
        self.debug = debug
        if self.debug:
            print(f"\n[Init] Building dependency graph...")
        self._build_dependency_graph()
        if self.debug:
            print(f"[Init] Dependency graph built")
            print(f"[Init] Computing topological levels...")
        self._compute_topological_levels()
        if self.debug:
            print(f"[Init] Topological levels computed")
            print(f"[Init] Initializing sampling parameters")
        self._init_sampling_parameters()

    def _build_dependency_graph(self):
        self.children = defaultdict(set)
        self.parents = defaultdict(set)
        self.roots = set()

        for task in self.dag.tasks:
            if task < 0:
                continue

            preds = set(p for p in self.dag.requiring.get(task, []) if p >= 0)
            if not preds:
                self.roots.add(task)

            for p in preds:
                self.parents[task].add(p)
                self.children[p].add(task)

    def _compute_topological_levels(self):
        self.levels = defaultdict(list)
        remaining = set(self.parents.keys()).union(self.roots)
        current_level = 0

        while remaining:
            current_tasks = [t for t in remaining
                             if all(p not in remaining for p in self.parents[t])]

            if not current_tasks:
                raise ValueError("DAG contains cycles")

            for t in current_tasks:
                self.levels[current_level].append(t)

            remaining -= set(current_tasks)
            current_level += 1

        self.max_level = current_level - 1
        self.num_tasks = sum(len(tasks) for tasks in self.levels.values())
        self.level_tasks = [np.array(tasks) for tasks in self.levels.values()]

    def _init_sampling_parameters(self):
        if self.num_tasks <= 50:
            self.order_samples_per_batch = 100
        elif self.num_tasks <= 200:
            self.order_samples_per_batch = 50
        else:
            self.order_samples_per_batch = 20

    def sample_orders(self, n_samples):
        samples = []

        for _ in range(n_samples):
            sample = np.zeros(self.num_tasks, dtype=np.int32)
            current_pos = 0
            for level in range(self.max_level + 1):
                tasks = self.level_tasks[level]
                if len(tasks) > 1:
                    sample[current_pos:current_pos + len(tasks)] = np.random.permutation(tasks)
                else:
                    sample[current_pos] = tasks[0]
                current_pos += len(tasks)
            samples.append(sample.tolist())

        return samples

class VmsProblem:
    def __init__(self, workflow_file: str, max_simultaneous_ins: int = 10, random_seed: int = 42):
        self.workflow_file = workflow_file
        self.max_simultaneous_ins = max_simultaneous_ins
        self.random_seed = random_seed
        self._load_workflow()
        self._init_infra()
        self.num_valid_tasks = len([t for t in self.tasks if t.id >= 0])
        self.number_of_variables = 2 * self.num_valid_tasks + max_simultaneous_ins
        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = (
                [self.num_valid_tasks - 1] * self.num_valid_tasks +
                [self.max_simultaneous_ins - 1] * self.num_valid_tasks +
                [7] * self.max_simultaneous_ins
        )
        self.sampler = DependencyAwareSampler(self.dag, random_seed=random_seed)
        self._init_sampling_parameters()
        self._generate_heft_seeds()

    def _init_sampling_parameters(self):
        total_dims = 2 * self.num_valid_tasks + self.max_simultaneous_ins

        if total_dims <= 50:
            self.order_samples_per_batch = 100
            self.vm_samples_per_batch = 100
        elif total_dims <= 200:
            self.order_samples_per_batch = 50
            self.vm_samples_per_batch = 50
        else:
            self.order_samples_per_batch = 20
            self.vm_samples_per_batch = 20

        self.vm_type_categories = [list(range(len(self.vm_configs)))] * self.max_simultaneous_ins
        self.vm_mapping_categories = [list(range(self.max_simultaneous_ins))] * self.num_valid_tasks

    def _sample_vm_mapping(self, n_samples, dims, n_vms):
        samples = np.zeros((n_samples, dims), dtype=int)
        if dims <= 50:
            block_size = dims
        elif dims <= 200:
            block_size = 50
        else:
            block_size = 25

        for i in range(0, dims, block_size):
            block_dims = min(block_size, dims - i)
            samples[:, i:i + block_dims] = np.random.randint(0, n_vms, size=(n_samples, block_dims))

        return samples

    def _sample_vm_types(self, n_samples, dims, n_types):
        adjusted_samples = ((n_samples // n_types) + 1) * n_types
        samples = np.zeros((adjusted_samples, dims), dtype=int)

        for d in range(dims):
            base = np.tile(np.arange(n_types), adjusted_samples // n_types)
            samples[:, d] = np.random.permutation(base)

        return samples[:n_samples]

    def create_solution(self) -> SolutionWrapper:
        task_order = self.sampler.sample_orders(1)[0]
        vm_mapping = self._sample_vm_mapping(1, self.num_valid_tasks, self.max_simultaneous_ins)[0].tolist()
        vm_types = self._sample_vm_types(1, self.max_simultaneous_ins, len(self.vm_configs))[0].tolist()
        variables = task_order + vm_mapping + vm_types
        solution = SolutionWrapper(variables)
        self.evaluate(solution)
        return solution

    def _init_infra(self):
        self.vm_configs = INFRA_CONFIG["vm_types"]
        if not any(vm['name'] == 'm3.medium' for vm in self.vm_configs):
            raise ValueError("必须包含m3.medium VM类型")

    def _load_workflow(self):
        loader = PegasusWorkflowLoader(self.workflow_file)
        self.tasks, self.dag = loader.load()
        self.dag.totalCloudletNum = len([t for t in self.tasks if t.id >= 0])

    def _generate_heft_seeds(self):
        self.heft_seeds = MOHEFT.generate_initial_solutions(self.dag, self.max_simultaneous_ins)

    def _is_valid_order(self, order: List[int]) -> bool:
        seen = set()
        for task_id in order:
            for pred in self.dag.requiring.get(task_id, []):
                if pred >= 0 and pred not in seen:
                    return False
            seen.add(task_id)
        return True

    def evaluate(self, solution: SolutionWrapper) -> None:
        try:
            n = self.num_valid_tasks
            real_task_order = solution.variables[:n]

            if len(real_task_order) != n or not self._is_valid_order(real_task_order):
                return

            task2ins = solution.variables[n:2 * n]
            ins2type = solution.variables[2 * n:2 * n + self.max_simultaneous_ins]
            ins2type = [min(max(vm_type, 0), len(self.vm_configs) - 1) for vm_type in ins2type]

            vmlist = self._create_vm_list(task2ins, ins2type)
            self.dag.calc_file_transfer_times(task2ins, vmlist)

            simulator = DAGCentralSimulator()
            simulator.set_cloudlet_dag(self.dag)
            simulator.set_vm_list(vmlist)

            for task_id in real_task_order:
                task = self.dag.tasks[task_id]
                ins_id = task2ins[task_id]
                task.vm_id = ins_id
                transfer_time = self.dag.file_transfer_time.get(task.id, 0.0)
                simulator.task_submit(task, transfer_time)

            if simulator.boot():
                solution.attributes['original_makespan'] = simulator._calculate_makespan()
                solution.attributes['original_cost'] = self._calculate_cost(simulator, vmlist)
                solution.objectives = [
                    solution.attributes['original_makespan'],
                    solution.attributes['original_cost']
                ]
        except Exception as e:
            print(f"Evaluation error: {str(e)}")

    def _create_vm_list(self, task2ins, ins2type):
        vmlist = []
        for vm_id in range(self.max_simultaneous_ins):
            vm_type_idx = ins2type[vm_id] if vm_id < len(ins2type) else 0
            vm_type_idx = max(0, min(vm_type_idx, len(self.vm_configs) - 1))
            vm_config = self.vm_configs[vm_type_idx]
            vmlist.append({
                'id': vm_id,
                'mips': vm_config['mips'],
                'bw': vm_config['bw'],
                'price': vm_config['price']
            })
        return vmlist

    def _calculate_cost(self, simulator, vmlist):
        vm_usage = defaultdict(lambda: {'start': float('inf'), 'end': 0.0})
        for task in simulator.finished_tasks:
            if task.id < 0: continue
            vm_id = task.vm_id
            vm_usage[vm_id]['start'] = min(vm_usage[vm_id]['start'], task.exec_start_time)
            vm_usage[vm_id]['end'] = max(vm_usage[vm_id]['end'], task.finish_time)

        total_cost = 0.0
        for vm_id, usage in vm_usage.items():
            if usage['end'] > usage['start']:
                duration_hours = (usage['end'] - usage['start']) / 3600
                vm_price = next(vm['price'] for vm in vmlist if vm['id'] == vm_id)
                total_cost += vm_price * duration_hours
        return total_cost

def deduplicate_samples(sampled_data: List[List[Any]]) -> List[List[Any]]:
    seen = set()
    deduplicated = []

    for sample in sampled_data:
        features = tuple(sample.variables)

        if features not in seen:
            seen.add(features)
            deduplicated.append(sample)

    return deduplicated

SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'
]

def generate_qmc_samples(dimensions: int, upper_bound: int, num_samples: int,
                         sampling_method: str, random_seed: int) -> List[List[int]]:
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")

    if sampling_method == 'sobol':
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_seed)
    elif sampling_method == 'orthogonal':
        sampler = qmc.LatinHypercube(d=dimensions, optimization="random-cd", seed=random_seed)
    elif sampling_method == 'stratified':
        return generate_stratified_samples(dimensions, upper_bound, num_samples, random_seed)
    elif sampling_method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(d=dimensions, seed=random_seed)
    else:
        raise ValueError(f"Unsupported QMC sampling method: {sampling_method}")

    if num_samples & (num_samples - 1) == 0:
        sample = sampler.random_base2(m=int(np.log2(num_samples)))
    else:
        sample = sampler.random(n=num_samples)

    samples = (sample * upper_bound).astype(int)
    samples = np.clip(samples, 0, upper_bound - 1)

    return samples.tolist()

def generate_stratified_samples(dimensions: int, upper_bound: int,
                                num_samples: int, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)

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

def generate_monte_carlo_batch(problem: VmsProblem, n_samples: int, dim: int, is_vm_mapping: bool, debug=False) -> List[List[int]]:
    if debug:
        print(f"Monte Carlo generation, dim: {dim}, is_vm_mapping: {is_vm_mapping}")
    if is_vm_mapping:
        samples = np.random.randint(0, problem.max_simultaneous_ins, size=(n_samples, dim))
    else:
        samples = np.random.randint(0, len(problem.vm_configs), size=(n_samples, dim))

    return samples.tolist()

def generate_samples(problem: VmsProblem, num_samples: int, random_seed: int, sampling_method: str = 'random',
                     debug=False) -> List[SolutionWrapper]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    sampled_solutions = []

    if debug:
        print(f"\n[Sampling start] target samples: {num_samples}")
        print(f"[sampling method] {sampling_method}")

    sampler = DependencyAwareSampler(problem.dag, random_seed, debug=debug)
    task_orders = sampler.sample_orders(num_samples)

    if sampling_method == 'monte_carlo':
        if problem.num_valid_tasks > 900:
            batch_size = 20
            vm_mappings = []
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                vm_mappings.extend(
                    np.random.randint(0, problem.max_simultaneous_ins,
                                      size=(current_batch, problem.num_valid_tasks)).tolist()
                )
        else:
            vm_mappings = np.random.randint(0, problem.max_simultaneous_ins,
                                            size=(num_samples, problem.num_valid_tasks)).tolist()
        vm_types = np.random.randint(0, len(problem.vm_configs),
                                     size=(num_samples, problem.max_simultaneous_ins)).tolist()

    elif sampling_method in ['sobol', 'orthogonal', 'stratified', 'latin_hypercube']:
        if problem.num_valid_tasks > 900:
            batch_size = 20
            vm_mappings = []
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                vm_mappings.extend(generate_qmc_samples(
                    problem.num_valid_tasks,
                    problem.max_simultaneous_ins,
                    current_batch,
                    sampling_method,
                    random_seed + i
                ))
        else:
            vm_mappings = generate_qmc_samples(
                problem.num_valid_tasks,
                problem.max_simultaneous_ins,
                num_samples,
                sampling_method,
                random_seed
            )
        vm_types = generate_qmc_samples(
            problem.max_simultaneous_ins,
            len(problem.vm_configs),
            num_samples,
            sampling_method,
            random_seed
        )

    elif sampling_method == 'covering_array':
        if problem.num_valid_tasks > 900:
            batch_size = 20
            vm_mappings = []
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                vm_mappings.extend(generate_covering_array_samples(
                    problem.num_valid_tasks,
                    problem.max_simultaneous_ins,
                    current_batch,
                    random_seed + i
                ))
        else:
            vm_mappings = generate_covering_array_samples(
                problem.num_valid_tasks,
                problem.max_simultaneous_ins,
                num_samples,
                random_seed
            )
        vm_types = generate_covering_array_samples(
            problem.max_simultaneous_ins,
            len(problem.vm_configs),
            num_samples,
            random_seed
        )
    else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}")

    for i in range(num_samples):
        task_order = task_orders[i]
        vm_mapping = vm_mappings[i] if vm_mappings else []
        vm_type = vm_types[i] if vm_types else []

        variables = task_order + vm_mapping + vm_type
        solution = SolutionWrapper(variables)
        problem.evaluate(solution)
        sampled_solutions.append(solution)

        if len(sampled_solutions) % 10 == 0 and debug:
            print(f"\rCurrent sampled: {len(sampled_solutions)}", end="", flush=True)

    seen = set()
    unique_solutions = []
    for sol in sampled_solutions:
        var_tuple = tuple(sol.variables)
        if var_tuple not in seen:
            seen.add(var_tuple)
            unique_solutions.append(sol)

    if len(unique_solutions) % 10 != 0:
        remainder = len(unique_solutions) % 10
        if remainder > 0:
            indices = np.random.choice(len(unique_solutions), len(unique_solutions) - remainder, replace=False)
            unique_solutions = [unique_solutions[i] for i in indices]

    final_solutions = unique_solutions[:num_samples]

    if debug:
        print(f"\nSampling completed, deduped and adjusted samples: {len(final_solutions)}")

    return final_solutions

def get_batch_indices(sorted_indices, dataset_name, reverse_prob=0.8):
    batch_size = 50
    max_complete_batches = len(sorted_indices) // batch_size
    if max_complete_batches == 0:
        return []
    batch_indices = [[] for _ in range(max_complete_batches)]
    for idx in sorted_indices:
        if all(len(batch) >= batch_size for batch in batch_indices):
            break

        assigned = False
        attempts = 0

        while not assigned and attempts < 3:
            if random.random() < reverse_prob:
                preferred_batch = min(
                    max_complete_batches - 1,
                    int((1 - (idx / len(sorted_indices))) * max_complete_batches)
                )
            else:
                preferred_batch = random.randint(0, max_complete_batches - 1)

            if len(batch_indices[preferred_batch]) < batch_size:
                batch_indices[preferred_batch].append(idx)
                assigned = True
            attempts += 1

        if not assigned:
            for batch_idx in range(max_complete_batches):
                if len(batch_indices[batch_idx]) < batch_size:
                    batch_indices[batch_idx].append(idx)
                    assigned = True
                    break

    return [batch for batch in batch_indices if len(batch) == batch_size]

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
            original_ft = float(row[-4])
            original_fa = float(row[-3])
            normalized_ft = float(row[-2])
            normalized_fa = float(row[-1])

            sol = SolutionWrapper(variables)
            sol.objectives = [original_ft, original_fa]
            sol.attributes['original_makespan'] = original_ft
            sol.attributes['original_cost'] = original_fa
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa

            sampled_solutions.append(sol)

    return sampled_solutions

def sample_and_save(problem: VmsProblem, workflow_file: str, minimize: bool, num_samples: int,
                    random_seed: int, sampling_method: str, sample_type: str, dataset_name: str,
                    reverse: bool = False, debug: bool = False):
    sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method, debug=debug)
    if not sampled_solutions:
        print(f"[sample_and_save] Warning: no valid samples generated: {dataset_name}, seed={random_seed}")
        return

    ft = [s.objectives[0] for s in sampled_solutions]
    fa = [s.objectives[1] for s in sampled_solutions]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(np.column_stack((ft, fa)))

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = normalized[i, 0]
        sol.attributes['normalized_fa'] = normalized[i, 1]

    header = [f'var_{i}' for i in range(problem.number_of_variables)] + \
             ['original_makespan', 'original_cost', 'normalized_ft', 'normalized_fa']

    save_sampled_data_to_csv(sampled_solutions, header, dataset_name, 'g1', sampling_method,
                             num_samples, random_seed, 'figure1', reverse)

    r0_points = normalized
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(sampled_solutions, header, dataset_name, 'g1', sampling_method,
                             num_samples, random_seed, 'figure2', reverse)

    if debug:
        print(f"[sample_and_save] Sampling and saving completed: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")

def process_g1_mode(problem, workflow_file, minimize, num_samples, random_seed, sampling_method,
                       sample_type, dataset_name, mode, unique_id, reverse=False, first_sample: bool = False):
    header = [f'var_{i}' for i in range(problem.number_of_variables)] + \
             ['original_makespan', 'original_cost', 'normalized_ft', 'normalized_fa']

    if first_sample:
        sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse, debug=True)
        return

    try:
        sampled_solutions_fig1 = load_sampled_data_from_csv(dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse)
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Please run this program with first_sample=True to generate sampled CSV before running this mode.")
        return

    try:
        sampled_solutions_fig2 = load_sampled_data_from_csv(dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure2', reverse)
    except FileNotFoundError:
        sampled_solutions_fig2 = None

    sampled_data = [s.variables for s in sampled_solutions_fig1]
    sampled_dict_fig1 = {tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa']) for s in sampled_solutions_fig1}

    plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict_fig1, reverse, problem.dag, problem.num_valid_tasks)

    if sampled_solutions_fig2 is not None:
        sampled_dict_fig2 = {tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa']) for s in sampled_solutions_fig2}
        plot_figure2(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                     sampled_dict_fig2, reverse, problem.dag, problem.num_valid_tasks)
    else:
        r0_points = np.array([[s.attributes['normalized_ft'], s.attributes['normalized_fa']] for s in sampled_solutions_fig1])
        g1, g2 = transform_points_for_figure2(r0_points)
        for i, s in enumerate(sampled_solutions_fig1):
            s.attributes['normalized_ft'] = g1[i]
            s.attributes['normalized_fa'] = g2[i]
        sampled_dict_fig2 = {tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa']) for s in sampled_solutions_fig1}
        plot_figure2(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                     sampled_dict_fig2, reverse, problem.dag, problem.num_valid_tasks)

def process_fa_construction_mode(problem, workflow_file, minimize, num_samples, random_seed, fa_construction,
                                 sampling_method, sample_type, dataset_name, mode, unique_id, reverse=False, first_sample: bool = False):
    if first_sample:
        sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse, debug=True)
        return

    try:
        g1_solutions = load_sampled_data_from_csv(dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse)
        print(f"Loaded data from g1 mode: {dataset_name}, {sampling_method}, {random_seed}")
    except FileNotFoundError:
        print(f"[FA mode] g1 base sampled data not found: {dataset_name}, {sampling_method}, seed={random_seed}")
        print("Please run this program with first_sample=True to generate sampled CSV before running this mode.")
        return

    target_count = (len(g1_solutions) // 10) * 10
    if len(g1_solutions) > target_count:
        g1_solutions = g1_solutions[:target_count]

    batch_size = 50
    num_batches = (len(g1_solutions) + batch_size - 1) // batch_size
    batch_indices = [range(i * batch_size, min((i + 1) * batch_size, len(g1_solutions))) for i in range(num_batches)]

    t = 1
    t_max = 1000
    novelty_archive = []
    all_ft_norm = []
    all_fa_norm = []
    age_info = None

    for batch_num in range(num_batches):
        batch_sols = [g1_solutions[i] for i in batch_indices[batch_num]]
        batch_ft = [s.objectives[0] for s in batch_sols]
        batch_vars = [s.variables for s in batch_sols]

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
                age_info = [i + 1 for i in range(len(batch_sols))]
            else:
                base_age = batch_size + t - 1
                age_info = [base_age] * len(batch_sols)

        if fa_construction == 'novelty':
            update_novelty_archive(batch_sols, novelty_archive)

        ft_norm, fa_norm = generate_fa(
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
            k=min(10, len(batch_sols) // 2)
        )

        for i, sol in enumerate(batch_sols):
            sol.attributes['normalized_ft'] = ft_norm[i]
            sol.attributes['normalized_fa'] = fa_norm[i]
            all_ft_norm.append(ft_norm[i])
            all_fa_norm.append(fa_norm[i])

        t += 1

    header = [f'var_{i}' for i in range(problem.number_of_variables)] + ['original_makespan', 'original_cost', 'normalized_ft', 'normalized_fa']
    save_sampled_data_to_csv(g1_solutions, header, dataset_name, mode, sampling_method, num_samples, random_seed, 'figure1', reverse)

    r0_points = np.column_stack((all_ft_norm, all_fa_norm))
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(g1_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(g1_solutions, header, dataset_name, mode, sampling_method, num_samples, random_seed, 'figure2', reverse)

    sampled_dict_fig1 = {tuple(s.variables): (all_ft_norm[i], all_fa_norm[i]) for i, s in enumerate(g1_solutions)}
    plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type, sampled_dict_fig1, reverse, problem.dag, problem.num_valid_tasks)

    sampled_dict_fig2 = {tuple(s.variables): (g1[i], g2[i]) for i, s in enumerate(g1_solutions)}
    plot_figure2(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type, sampled_dict_fig2, reverse, problem.dag, problem.num_valid_tasks)

def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2

class ProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problems(cls, dataset_names: List[str], workflow_base_path: str, random_seed: int = 42) -> Dict[str, VmsProblem]:
        instance = cls()
        for dataset in dataset_names:
            if dataset not in instance.problems:
                try:
                    workflow_file = f'{workflow_base_path}{dataset}.xml'
                    instance.problems[dataset] = VmsProblem(workflow_file=workflow_file, random_seed=random_seed)
                    print(f"Successfully preloaded: {dataset}")
                except Exception as e:
                    print(f"Failed to load {dataset}: {str(e)}")
        return instance.problems

    @classmethod
    def get_problem(cls, dataset_name: str) -> VmsProblem:
        instance = cls()
        return instance.problems.get(dataset_name)

def init_worker():
    pass

def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)
    total_tasks = len(all_tasks)

    for i in range(0, total_tasks, batch_size):
        current_batch = all_tasks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_tasks + batch_size - 1) // batch_size
        print(f"=== Processing batch {batch_num}/{total_batches} | Tasks in batch: {len(current_batch)} | Workers: {max_workers} ===")

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker
        ) as executor:
            futures = [executor.submit(process_single_task, **task) for task in current_batch]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Batch task exception: {str(e)}")

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
            return f"[Sampling done] {unique_id}"

        if mode == 'g1':
            process_g1_mode(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, mode, unique_id, reverse, first_sample=False)
        elif mode in fa_construction:
            process_fa_construction_mode(problem, workflow_file, minimize, num_samples, random_seed, mode, sampling_method, sample_type, dataset_name, mode, unique_id, reverse, first_sample=False)
        else:
            return f"[Skipped] Unknown mode: {mode}"

        return f"Task completed: {unique_id}"
    except Exception as e:
        import traceback
        return f"Task failed: {unique_id}, Error: {str(e)}\n{traceback.format_exc()}"

def main_ws_multi(
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
        first_sample: bool = False,
        workflow_base_path='../Datasets/',
        use_saved_data=False,
        debug=False
):
    if dataset_names is None:
        dataset_names = ["CyberShake_30", "CyberShake_50", "CyberShake_100",
                         "Epigenomics_24", "Epigenomics_46", "Epigenomics_100",
                         "Inspiral_30", "Inspiral_50", "Inspiral_100",
                         "Montage_25", "Montage_50", "Montage_100",
                         "Sipht_30", "Sipht_60", "Sipht_100", "Sipht_1000",
                         "CyberShake_1000", "Inspiral_1000", "Montage_1000", "Epigenomics_997"]

    if fa_construction is None:
        fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']

    if fixed_sample_sizes is None:
        fixed_sample_sizes = [1000]

    if percentage_sample_sizes is None:
        percentage_sample_sizes = [10, 20, 30, 40, 50]

    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']

    if random_seeds is None:
        random_seeds = range(10)

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 50)

    print("=== Preloading all DAG instances ===")
    for seed in random_seeds:
        ProblemManager.preload_problems(dataset_names, workflow_base_path, seed)
    print("=== DAG preloading completed ===")

    if first_sample:
        fa_construction = ["g1"]

    all_tasks = []
    for dataset in dataset_names:
        for mode in fa_construction:
            for sampling_method in sampling_methods:
                for num_sample in fixed_sample_sizes:
                    for random_seed in random_seeds:
                        unique_id = f"{dataset}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}_reverse_{reverse}" if reverse else f"{dataset}_{sampling_method}_{num_sample}_fixed_{random_seed}_{mode}"
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
                            'workflow_file': f'{workflow_base_path}{dataset}.xml',
                            'first_sample': first_sample,
                        }
                        all_tasks.append(task)

    print(f"\n=== Task summary ===\nTotal tasks: {len(all_tasks)} | Max workers: {max_workers}")

    if use_multiprocessing:
        process_in_batches(all_tasks=all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        print("=== Single-process mode ===")
        for task in all_tasks:
            print(process_single_task(**task))

if __name__ == "__main__":
    fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    dataset_names = ["CyberShake_30", "CyberShake_50", "CyberShake_100",
        "Epigenomics_24", "Epigenomics_46", "Epigenomics_100",
        "Inspiral_30", "Inspiral_50", "Inspiral_100",
        "Montage_25", "Montage_50", "Montage_100",
        "Sipht_30", "Sipht_60", "Sipht_100", "Sipht_1000",
        "CyberShake_1000","Inspiral_1000","Montage_1000", "Epigenomics_997"]

    main_ws_multi(dataset_names, fa_construction, use_multiprocessing=True, reverse=False, first_sample=False, workflow_base_path='../Datasets/')