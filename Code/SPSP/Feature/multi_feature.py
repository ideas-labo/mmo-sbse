from typing import List, Any, Dict
import sys
from scipy.stats import qmc
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/mydrive/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code')
from Code.SPSP.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from Code.SPSP.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2
from Code.SPSP.mmo_spsp import SPSProblem
import networkx as nx
import random
import multiprocessing
import sys
import csv
import os
from typing import List
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../')
sys.path.append('../..')

SAMPLING_METHODS = [
    'monte_carlo', 'latin_hypercube','sobol', 'stratified', 'halton', 'random_walk'
]

class SolutionWrapper:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = [float('inf'), float('inf')]
        self.attributes = {'original_makespan': float('inf'), 'original_cost': float('inf')}
        self.constraints = []

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

    print(f"Data saved to: {filename}")

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

def generate_samples(problem, num_samples: int, random_seed: int,
                     sampling_method: str = 'random', debug=False) -> List[SolutionWrapper]:
    np.random.seed(random_seed)
    random.seed(random_seed)

    samples = []
    task_ids = sorted(problem.task_skills.keys())
    emp_ids = sorted(problem.employee_skills.keys())
    num_tasks = len(task_ids)
    num_employees = len(emp_ids)
    dimension = num_tasks * num_employees
    minded = problem.minded

    if debug:
        print(f"\n[Improved sampling] target samples: {num_samples}, total dim: {dimension}, tasks: {num_tasks}, employees: {num_employees}")
        print(f"[dedication constraint] minded={minded:.6f}")

    task_layers = {}
    dag = problem.dag
    for task_id in nx.topological_sort(dag):
        task_idx = task_ids.index(task_id)
        predecessors = list(dag.predecessors(task_id))
        if not predecessors:
            task_layers[task_idx] = 0
        else:
            pred_layers = [task_layers[task_ids.index(p)] for p in predecessors]
            task_layers[task_idx] = max(pred_layers) + 1

    max_layer = max(task_layers.values())
    if debug:
        print(f"[topological layers] total {max_layer + 1} layers")

    emp_eligible_tasks = {}
    task_eligible_emps = {}
    for emp_idx, emp_id in enumerate(emp_ids):
        emp_skills = set(problem.employee_skills[emp_id])
        eligible_tasks = []
        for task_idx, task_id in enumerate(task_ids):
            task_skills = set(problem.task_skills[task_id])
            if not emp_skills.isdisjoint(task_skills):
                eligible_tasks.append(task_idx)
                if task_idx not in task_eligible_emps:
                    task_eligible_emps[task_idx] = []
                task_eligible_emps[task_idx].append(emp_idx)
        emp_eligible_tasks[emp_idx] = eligible_tasks

    for task_idx in range(num_tasks):
        if task_idx not in task_eligible_emps or len(task_eligible_emps[task_idx]) == 0:
            raise ValueError(f"Task {task_idx} has no matching employees, cannot generate valid samples!")

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
    max_attempts = num_samples * 20

    while valid_samples < num_samples and attempts < max_attempts:
        dedication = np.zeros((num_employees, num_tasks))

        for layer in range(max_layer + 1):
            layer_tasks = [t for t, l in task_layers.items() if l == layer]

            for task_idx in layer_tasks:
                required_skills = set(problem.task_skills[task_ids[task_idx]])

                emp_layer_dedication = {}
                for emp_idx in task_eligible_emps[task_idx]:
                    layer_ded = sum(dedication[emp_idx, t] for t in layer_tasks)
                    emp_layer_dedication[emp_idx] = layer_ded

                available_emps = [e for e in task_eligible_emps[task_idx]
                                  if emp_layer_dedication.get(e, 0) < 0.7]

                if not available_emps:
                    available_emps = task_eligible_emps[task_idx]

                selected_emps = []
                covered_skills = set()

                sorted_emps = sorted(available_emps,
                                     key=lambda e: emp_layer_dedication.get(e, 0))

                for emp_idx in sorted_emps:
                    emp_id = emp_ids[emp_idx]
                    emp_skills = set(problem.employee_skills[emp_id])
                    new_skills = emp_skills & required_skills - covered_skills

                    if new_skills:
                        selected_emps.append(emp_idx)
                        covered_skills.update(emp_skills)

                        if required_skills.issubset(covered_skills):
                            break

                if not required_skills.issubset(covered_skills):
                    for emp_idx in sorted_emps:
                        if emp_idx not in selected_emps:
                            emp_id = emp_ids[emp_idx]
                            emp_skills = set(problem.employee_skills[emp_id])
                            selected_emps.append(emp_idx)
                            covered_skills.update(emp_skills)
                            if required_skills.issubset(covered_skills):
                                break

                if not selected_emps:
                    selected_emps = [random.choice(available_emps)]

                for emp_idx in selected_emps:
                    layer_ded = sum(dedication[emp_idx, t] for t in layer_tasks)
                    available = max(0, 0.8 - layer_ded)

                    if available < minded:
                        continue

                    max_multipliers = int(available / minded)
                    if max_multipliers > 0:
                        num_multipliers = random.randint(1, min(max_multipliers, 10))
                        dedication[emp_idx, task_idx] = num_multipliers * minded

        for task_idx in range(num_tasks):
            task_has_participant = np.any(dedication[:, task_idx] > 1e-9)

            if not task_has_participant:
                eligible_emps = task_eligible_emps[task_idx]
                layer = task_layers[task_idx]
                layer_tasks = [t for t, l in task_layers.items() if l == layer]

                emp_loads = [(e, sum(dedication[e, t] for t in layer_tasks))
                             for e in eligible_emps]
                emp_loads.sort(key=lambda x: x[1])
                selected_emp = emp_loads[0][0]

                layer_ded = sum(dedication[selected_emp, t] for t in layer_tasks)
                available = max(0, 0.8 - layer_ded)

                if available >= minded:
                    num_mult = min(int(available / minded), 3)
                    dedication[selected_emp, task_idx] = num_mult * minded
                else:
                    other_tasks = [t for t in layer_tasks if t != task_idx
                                   and dedication[selected_emp, t] > 1e-9]
                    if other_tasks:
                        reduce_task = random.choice(other_tasks)
                        reduce_amount = min(dedication[selected_emp, reduce_task], minded * 2)
                        dedication[selected_emp, reduce_task] -= reduce_amount
                        dedication[selected_emp, task_idx] = reduce_amount

        sample = dedication.flatten()

        solution = SolutionWrapper(sample.tolist())
        problem.evaluate(solution)

        if problem.is_feasible(solution):
            samples.append(solution)
            valid_samples += 1
            if debug and valid_samples % 100 == 0:
                print(f"Generated {valid_samples}/{num_samples} valid samples (attempts {attempts})")

        attempts += 1

    if debug:
        print(f"Sampling completed, valid samples: {valid_samples}, attempts: {attempts}")
        if valid_samples < num_samples:
            print(f"Warning: only produced {valid_samples}/{num_samples} valid samples")

    if len(samples) % 10 != 0:
        remainder = len(samples) % 10
        if remainder > 0:
            indices_to_keep = np.random.choice(len(samples), len(samples) - remainder, replace=False)
            samples = [samples[i] for i in indices_to_keep]

    if not samples:
        raise RuntimeError(f"Unable to produce valid samples! Reached max attempts {max_attempts}.")

    return samples[:num_samples]

def process_g1_mode(problem, workflow_file, minimize, num_samples, random_seed, sampling_method,
                       sample_type,
                       dataset_name, mode, unique_id, reverse=False, first_sample: bool = False):
    num_tasks = len(problem.task_skills)
    num_employees = len(problem.employee_skills)
    dimension = num_tasks * num_employees

    header = [f'var_{i}' for i in range(dimension)] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    if first_sample:
        sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse)
        return

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse
        )
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Please run this program with first_sample=True to generate sampled CSV before running this mode.")
        return

    normalized = np.array([[s.attributes['normalized_ft'], s.attributes['normalized_fa']]
                           for s in sampled_solutions])

    sampled_data = [s.variables for s in sampled_solutions]
    r0_points = normalized
    g1, g2 = transform_points_for_figure2(r0_points)

    sampled_dict = {tuple(sample): (ft_val, fa_val)
                    for sample, ft_val, fa_val in zip(sampled_data, r0_points[:, 0], r0_points[:, 1])}
    plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict, reverse)

    sampled_dict_fig2 = {tuple(sample): (g1_val, g2_val)
                         for sample, g1_val, g2_val in zip(sampled_data, g1, g2)}
    plot_figure2(random_seed, 'mean', sampling_method, num_samples, dataset_name, mode, unique_id, sample_type,
                 sampled_dict_fig2, reverse)

def sample_and_save(problem, workflow_file, minimize, num_samples,
                    random_seed, sampling_method, sample_type, dataset_name, reverse=False):
    sampled_solutions = generate_samples(problem, num_samples, random_seed, sampling_method, debug=True)
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

    header = [f'var_{i}' for i in range(problem.task_count * problem.employee_count)] if hasattr(problem, 'task_count') else [f'var_{i}' for i in range(len(sampled_solutions[0].variables))]
    header = header + ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']
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

    print(f"[sample_and_save] Sampling and saving completed: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")

def process_fa_construction_mode(problem, workflow_file, minimize, num_samples, random_seed, fa_construction,
                                 sampling_method, sample_type, dataset_name, mode, unique_id, reverse=False, first_sample: bool = False):
    if first_sample:
        sample_and_save(problem, workflow_file, minimize, num_samples, random_seed, sampling_method, sample_type, dataset_name, reverse)
        return

    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name, 'g1', sampling_method, num_samples, random_seed, 'figure1', reverse
        )
    except FileNotFoundError:
        print(f"Please run the g1 pre-sampling (first_sample=True) to generate base data: {dataset_name}")
        return

    sorted_solutions = sorted(sampled_solutions, key=lambda x: sum(x.objectives))

    if len(sorted_solutions) % 10 != 0:
        target_size = (len(sorted_solutions) // 10) * 10
        sorted_solutions = sorted_solutions[:target_size]
        print(f"Adjusted samples to multiple of 10: {len(sorted_solutions)}")

    sampled_solutions = sorted_solutions

    batch_size = 20
    num_batches = (len(sampled_solutions) + batch_size - 1) // batch_size

    sorted_indices = list(range(len(sampled_solutions)))

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

    batch_indices = get_batch_indices(sorted_indices, batch_size, num_batches)

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
            age_info = [i + 1 if batch_num == 0 else batch_size + t - 1
                        for i in range(len(batch_solutions))]
        elif fa_construction == 'novelty':
            update_novelty_archive(batch_solutions, novelty_archive)

        batch_ft_norm, batch_fa_norm = generate_fa(
            batch_vars, batch_ft, fa_construction, minimize, unique_elements_per_column,
            t, t_max, random_seed,
            age_info=age_info if fa_construction == 'age' else None,
            novelty_archive=novelty_archive if fa_construction in ['novelty'] else None,
            k=min(10, len(batch_solutions) // 2)
        )

        all_ft_normalized.extend(batch_ft_norm)
        all_fa_normalized.extend(batch_fa_norm)
        t += 1

    num_tasks = len(problem.task_skills)
    num_employees = len(problem.employee_skills)
    dimension = num_tasks * num_employees

    header = [f'var_{i}' for i in range(dimension)] + \
             ['original_ft', 'original_fa', 'normalized_ft', 'normalized_fa']

    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = all_ft_normalized[i]
        sol.attributes['normalized_fa'] = all_fa_normalized[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, mode, sampling_method,
        num_samples, random_seed, 'figure1', reverse
    )

    r0_points = np.column_stack((all_fa_normalized, all_ft_normalized))
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    save_sampled_data_to_csv(
        sampled_solutions, header, dataset_name, mode, sampling_method,
        num_samples, random_seed, 'figure2', reverse
    )

    sampled_data = [s.variables for s in sampled_solutions]
    sampled_dict = {tuple(s.variables): (ft_val, fa_val)
                    for s, ft_val, fa_val in zip(sampled_solutions, all_ft_normalized, all_fa_normalized)}
    plot_figure1(random_seed, 'mean', sampling_method, num_samples, dataset_name,
                 mode, unique_id, sample_type, sampled_dict, reverse)

    sampled_dict_fig2 = {tuple(s.variables): (g1_val, g2_val)
                         for s, g1_val, g2_val in zip(sampled_solutions, g1, g2)}
    plot_figure2(random_seed, 'mean', sampling_method, num_samples,
                 dataset_name, mode, unique_id, sample_type, sampled_dict_fig2)

def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2

import concurrent.futures

class ProblemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.problems = {}
        return cls._instance

    @classmethod
    def preload_problems(cls, dataset_names: List[str], workflow_base_path: str,
                         random_seed: int = 42) -> Dict[str, Any]:
        instance = cls()
        for dataset in dataset_names:
            if dataset not in instance.problems:
                try:
                    instance_file = f"{workflow_base_path}inst-{dataset}.conf"

                    if not os.path.exists(instance_file):
                        print(f"Warning: instance file not found: {instance_file}")
                        continue

                    instance.problems[dataset] = SPSProblem(
                        instance_file=instance_file,
                        random_seed=random_seed
                    )
                    print(f"Successfully preloaded PSP instance: {dataset} (ver={instance.problems[dataset].ver}, minded={instance.problems[dataset].minded:.6f})")
                except Exception as e:
                    print(f"Failed to load PSP instance {dataset}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        return instance.problems

    @classmethod
    def get_problem(cls, dataset_name: str):
        instance = cls()
        return instance.problems.get(dataset_name)

def init_worker():
    pass

def process_single_task(mode, dataset_name, sampling_method, num_samples, sample_type,
                        minimize, random_seed, fa_construction, unique_id, reverse,
                        workflow_file, first_sample: bool = False):
    try:
        problem = ProblemManager.get_problem(dataset_name)
        if problem is None:
            raise ValueError(f"Preloaded PSP instance not found: {dataset_name}")

        np.random.seed(random_seed)
        random.seed(random_seed)

        if mode == 'g1':
            process_g1_mode(
                problem, workflow_file, minimize, num_samples, random_seed,
                sampling_method, sample_type, dataset_name, mode, unique_id,
                reverse, first_sample=first_sample
            )
        elif mode in fa_construction:
            process_fa_construction_mode(
                problem, workflow_file, minimize, num_samples, random_seed,
                mode, sampling_method, sample_type, dataset_name, mode, unique_id, reverse, first_sample=first_sample
            )

        return f"PSP task completed: {unique_id}"
    except Exception as e:
        import traceback
        return f"PSP task failed: {unique_id}, Error: {str(e)}\n{traceback.format_exc()}"

def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)

    total_tasks = len(all_tasks)
    for i in range(0, total_tasks, batch_size):
        batch = all_tasks[i:i + batch_size]

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker
        ) as executor:
            futures = [executor.submit(
                process_single_task,
                mode=task['mode'],
                dataset_name=task['dataset_name'],
                sampling_method=task['sampling_method'],
                num_samples=task['num_samples'],
                sample_type=task['sample_type'],
                minimize=task['minimize'],
                random_seed=task['random_seed'],
                fa_construction=task['fa_construction'],
                unique_id=task['unique_id'],
                reverse=task['reverse'],
                workflow_file=task['workflow_file'],
                first_sample=task.get('first_sample', False)
            ) for task in batch]

            for future in concurrent.futures.as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f"Task error: {str(e)}")

def main_spsp_multi(dataset_names, fa_construction, minimize=True,
                    fixed_sample_sizes=[1000],
                    percentage_sample_sizes=[10, 20, 30, 40, 50],
                    sampling_methods=None,
                    use_multiprocessing=True,
                    max_workers=None,
                    reverse=False,
                    first_sample: bool = False,
                    workflow_base_path='../Datasets/',
                    random_seeds=None):
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array',
                            'halton', 'random_walk']
    if random_seeds is None:
        random_seeds = range(0, 10)

    if first_sample:
        fa_construction = ["g1"]

    print("Preloading all PSP instances (new dataset format, fixed max dedication=1.0)...")
    print(f"Instance list: {dataset_names}")
    for seed in random_seeds:
        ProblemManager.preload_problems(dataset_names, workflow_base_path, seed)

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
                            'workflow_file': f'{workflow_base_path}inst-{dataset}.conf',
                            'first_sample': first_sample
                        }
                        all_tasks.append(task)

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 50)

    if use_multiprocessing:
        print(f"Starting multiprocessing for PSP problems, total {len(all_tasks)} tasks, using {max_workers} workers...")
        process_in_batches(all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        for task in all_tasks:
            print(process_single_task(**task))

if __name__ == "__main__":
    fa_construction = ['g1', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
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

    main_spsp_multi(dataset_names, fa_construction, use_multiprocessing=True, reverse=False, first_sample=False, workflow_base_path='../Datasets/')