import csv
import math
import os
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import List
from evoxbench.benchmarks import NASBench201Benchmark, NATSBenchmark
from evoxbench.test_suites import c10mop, citysegmop, in1kmop
from jmetal.core.problem import Problem
from jmetal.core.solution import IntegerSolution
from jmetal.operator import BinaryTournamentSelection, IntegerSBXCrossover, IntegerPolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
import sys

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../../..')

from Code.Utils.remove_duplicates import partial_duplicate_replacement
from Utils.Construct_secondary_objective import generate_fa, update_novelty_archive

_DEBUG = False  # run in debug mode
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.util.solution import get_non_dominated_solutions
import numpy as np

# Configuration
USE_PARALLEL = True  # Set to False to disable parallel processing
CPU_CORES = 50
MAX_RUNTIME = 24 * 3600  # 24 hours in seconds
MODES = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa','age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']
SEEDS = range(0, 10)  # 5 random seeds (0-4)
PROBLEM_IDS = [1, 3, 5, 8, 10, 11, 12, 13]  # C10MOP problem IDs,
CITYSEG_PROBLEM_IDS = [3]  # CitySegMOP problem IDs
IN1K_PROBLEM_IDS = [1, 4, 7]  # IN1KMOP problem IDs

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


class RepairedIntegerSBXCrossover(IntegerSBXCrossover):
    """Crossover operator with repair mechanism for integer variables"""

    def __init__(self, probability: float, distribution_index: float):
        super().__init__(probability, distribution_index)

    def execute(self, parents: list[IntegerSolution]) -> list[IntegerSolution]:
        offspring = super().execute(parents)

        for solution in offspring:
            for i in range(len(solution.variables)):
                var = round(solution.variables[i])
                var = max(solution.lower_bound[i], min(var, solution.upper_bound[i]))
                solution.variables[i] = var

        return offspring


class RepairedIntegerPolynomialMutation(IntegerPolynomialMutation):
    """Mutation operator with repair mechanism for integer variables"""

    def __init__(self, probability: float, distribution_index: float):
        super().__init__(probability, distribution_index)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        mutated_solution = super().execute(solution)

        for i in range(len(mutated_solution.variables)):
            var = round(mutated_solution.variables[i])
            var = max(mutated_solution.lower_bound[i], min(var, mutated_solution.upper_bound[i]))
            mutated_solution.variables[i] = var

        return mutated_solution


class C10MOPProblem(Problem[IntegerSolution]):
    def __init__(self, benchmark, mode='ft_fa', selected_objs=None, random_seed=0):
        super().__init__()
        self.benchmark = benchmark
        self.search_space = benchmark.search_space
        self.mode = mode
        self.selected_objs = selected_objs if selected_objs is not None else [0, 1]

        if self.benchmark.evaluator.n_objs > 2 and self.selected_objs is None:
            self.selected_objs = [0, 1]

        self.directions = [self.MINIMIZE] * 2
        self.labels = [f"Objective {i + 1}" for i in range(2)]
        self.current_raw_F = None
        self.max_generations = 100
        self.random_seed = random_seed
        self.true_eval = False

    def normalize_population(self, population: List[IntegerSolution], age_info, novelty_archive,
                             current_generation) -> None:
        self.current_raw_F = np.array([s.attributes['original_objectives'] for s in population])
        selected_F = np.array([s.attributes['selected_objectives'] for s in population])
        n_obj = selected_F.shape[1]
        assert n_obj == 2, "Must select two objectives for optimization"

        # Filter valid solutions (without inf)
        valid_mask = ~np.isinf(selected_F).any(axis=1)
        valid_F = selected_F[valid_mask]
        valid_pop = [pop for pop, mask in zip(population, valid_mask) if mask]

        # Return if no valid solutions
        if not np.any(valid_mask):
            return

        f1 = valid_F[:, 0]
        f2 = valid_F[:, 1]

        if self.mode == 'ft_fa':
            for i, sol in enumerate(valid_pop):
                sol.objectives = valid_F[i].tolist()

        elif self.mode == 'g1_g2':
            f1_min, f1_max = f1.min(), f1.max()
            f2_min, f2_max = f2.min(), f2.max()

            for i, sol in enumerate(valid_pop):
                norm_f1 = (f1[i] - f1_min) / (f1_max - f1_min) if f1_max != f1_min else 0.5
                norm_f2 = (f2[i] - f2_min) / (f2_max - f2_min) if f2_max != f2_min else 0.5
                sol.objectives = [norm_f1 + norm_f2, norm_f1 - norm_f2]

        elif self.mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa',
                           'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_pop]
            unique_elements = [sorted({sol.variables[i] for sol in valid_pop})
                               for i in range(len(valid_pop[0].variables))]

            mode_prefix = self.mode.split('_')[0]
            ft, fa = generate_fa(
                configurations=configurations,
                ft=f1,
                fa_construction=mode_prefix,
                minimize=True,
                file_path='',
                unique_elements_per_column=unique_elements,
                t=current_generation,
                t_max=self.max_generations,
                random_seed=self.random_seed,
                age_info=age_info,
                novelty_archive=novelty_archive,
                k=len(valid_pop) // 2
            )

            for i, sol in enumerate(valid_pop):
                sol.objectives = [ft[i] + fa[i], ft[i] - fa[i]]

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        x = np.array(solution.variables).reshape(1, -1)
        raw_F = self.benchmark.evaluate(x, true_eval=self.true_eval).flatten()

        solution.attributes['original_objectives'] = raw_F.copy()
        # Get selected objective values
        selected_obj_values = raw_F[self.selected_objs].copy()

        # Check if selected objectives contain inf, if yes set all objectives to inf, otherwise assign normally
        if np.isinf(selected_obj_values).any():
            solution.objectives = [np.inf, np.inf]
            solution.attributes['selected_objectives'] = [np.inf, np.inf]
        else:
            solution.objectives = selected_obj_values.tolist()
            solution.attributes['selected_objectives'] = selected_obj_values
        return solution

    def number_of_variables(self) -> int:
        return self.search_space.n_var

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def create_solution(self) -> IntegerSolution:
        lb = self.search_space.lb
        ub = self.search_space.ub
        variables = [
            np.random.randint(lb[i], ub[i] + 1)
            for i in range(self.number_of_variables())
        ]
        solution = IntegerSolution(
            lower_bound=lb,
            upper_bound=ub,
            number_of_objectives=self.number_of_objectives()
        )
        solution.variables = variables
        return solution

    def name(self) -> str:
        return f"C10MOPProblem ({self.mode})"


class CitySegMOPProblem(Problem[IntegerSolution]):
    def __init__(self, benchmark, mode='ft_fa', selected_objs=None, random_seed=0):
        super().__init__()
        self.benchmark = benchmark
        self.search_space = benchmark.search_space
        self.mode = mode
        self.selected_objs = selected_objs if selected_objs is not None else [0, 1]
        if self.benchmark.evaluator.n_objs > 2 and self.selected_objs is None:
            self.selected_objs = [0, 1]

        self.directions = [self.MINIMIZE] * 2
        self.labels = [f"Objective {i + 1}" for i in range(2)]
        self.current_raw_F = None
        self.max_generations = 100
        self.random_seed = random_seed
        self.true_eval = False

    def normalize_population(self, population: List[IntegerSolution], age_info, novelty_archive,
                             current_generation) -> None:
        self.current_raw_F = np.array([s.attributes['original_objectives'] for s in population])
        selected_F = np.array([s.attributes['selected_objectives'] for s in population])
        n_obj = selected_F.shape[1]
        assert n_obj == 2, "Must select two objectives for optimization"

        # Filter valid solutions (without inf)
        valid_mask = ~np.isinf(selected_F).any(axis=1)
        valid_F = selected_F[valid_mask]
        valid_pop = [pop for pop, mask in zip(population, valid_mask) if mask]

        # Return if no valid solutions
        if not np.any(valid_mask):
            return

        f1 = valid_F[:, 0]
        f2 = valid_F[:, 1]

        if self.mode == 'ft_fa':
            for i, sol in enumerate(valid_pop):
                sol.objectives = valid_F[i].tolist()
        elif self.mode == 'g1_g2':
            f1_min, f1_max = f1.min(), f1.max()
            f2_min, f2_max = f2.min(), f2.max()
            for i, sol in enumerate(valid_pop):
                norm_f1 = (f1[i] - f1_min) / (f1_max - f1_min) if f1_max != f1_min else 0.5
                norm_f2 = (f2[i] - f2_min) / (f2_max - f2_min) if f2_max != f2_min else 0.5
                sol.objectives = [norm_f1 + norm_f2, norm_f1 - norm_f2]
        elif self.mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa',
                           'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_pop]
            unique_elements = [sorted({sol.variables[i] for sol in valid_pop})
                               for i in range(len(valid_pop[0].variables))]
            mode_prefix = self.mode.split('_')[0]
            ft, fa = generate_fa(
                configurations=configurations,
                ft=f1,
                fa_construction=mode_prefix,
                minimize=True,
                file_path='',
                unique_elements_per_column=unique_elements,
                t=current_generation,
                t_max=self.max_generations,
                random_seed=self.random_seed,
                age_info=age_info,
                novelty_archive=novelty_archive,
                k=len(valid_pop) // 2
            )
            for i, sol in enumerate(valid_pop):
                sol.objectives = [ft[i] + fa[i], ft[i] - fa[i]]

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        x = np.array(solution.variables).reshape(1, -1)
        raw_F = self.benchmark.evaluate(x, true_eval=self.true_eval).flatten()
        solution.attributes['original_objectives'] = raw_F.copy()
        # Get selected objective values
        selected_obj_values = raw_F[self.selected_objs].copy()

        # Check if selected objectives contain inf, if yes set all objectives to inf, otherwise assign normally
        if np.isinf(selected_obj_values).any():
            solution.objectives = [np.inf, np.inf]
            solution.attributes['selected_objectives'] = [np.inf, np.inf]
        else:
            solution.objectives = selected_obj_values.tolist()
            solution.attributes['selected_objectives'] = selected_obj_values
        return solution

    def number_of_variables(self) -> int:
        return self.search_space.n_var

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def create_solution(self) -> IntegerSolution:
        lb = self.search_space.lb
        ub = self.search_space.ub
        variables = [
            np.random.randint(lb[i], ub[i] + 1)
            for i in range(self.number_of_variables())
        ]
        solution = IntegerSolution(
            lower_bound=lb,
            upper_bound=ub,
            number_of_objectives=self.number_of_objectives()
        )
        solution.variables = variables
        return solution

    def name(self) -> str:
        return f"CitySegMOPProblem ({self.mode})"


class In1KMOPProblem(Problem[IntegerSolution]):
    def __init__(self, benchmark, mode='ft_fa', selected_objs=None, random_seed=0):
        super().__init__()
        self.benchmark = benchmark
        self.search_space = benchmark.search_space
        self.mode = mode
        self.selected_objs = selected_objs if selected_objs is not None else [0, 1]

        if self.benchmark.evaluator.n_objs > 2 and self.selected_objs is None:
            self.selected_objs = [0, 1]

        self.directions = [self.MINIMIZE] * 2
        self.labels = [f"Objective {i + 1}" for i in range(2)]
        self.current_raw_F = None
        self.max_generations = 100
        self.random_seed = random_seed

        self.true_eval = False

    def normalize_population(self, population: List[IntegerSolution], age_info, novelty_archive,
                             current_generation) -> None:
        self.current_raw_F = np.array([s.attributes['original_objectives'] for s in population])
        selected_F = np.array([s.attributes['selected_objectives'] for s in population])
        n_obj = selected_F.shape[1]
        assert n_obj == 2, "Must select two objectives for optimization"

        # Filter valid solutions (without inf)
        valid_mask = ~np.isinf(selected_F).any(axis=1)
        valid_F = selected_F[valid_mask]
        valid_pop = [pop for pop, mask in zip(population, valid_mask) if mask]

        # Return if no valid solutions
        if not np.any(valid_mask):
            return

        f1 = valid_F[:, 0]
        f2 = valid_F[:, 1]

        if self.mode == 'ft_fa':
            for i, sol in enumerate(valid_pop):
                sol.objectives = valid_F[i].tolist()
        elif self.mode == 'g1_g2':
            f1_min, f1_max = f1.min(), f1.max()
            f2_min, f2_max = f2.min(), f2.max()
            for i, sol in enumerate(valid_pop):
                norm_f1 = (f1[i] - f1_min) / (f1_max - f1_min) if f1_max != f1_min else 0.5
                norm_f2 = (f2[i] - f2_min) / (f2_max - f2_min) if f2_max != f2_min else 0.5
                sol.objectives = [norm_f1 + norm_f2, norm_f1 - norm_f2]
        elif self.mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa',
                           'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_pop]
            unique_elements = [sorted({sol.variables[i] for sol in valid_pop})
                               for i in range(len(valid_pop[0].variables))]
            mode_prefix = self.mode.split('_')[0]
            ft, fa = generate_fa(
                configurations=configurations,
                ft=f1,
                fa_construction=mode_prefix,
                minimize=True,
                file_path='',
                unique_elements_per_column=unique_elements,
                t=current_generation,
                t_max=self.max_generations,
                random_seed=self.random_seed,
                age_info=age_info,
                novelty_archive=novelty_archive,
                k=len(valid_pop) // 2
            )
            for i, sol in enumerate(valid_pop):
                sol.objectives = [ft[i] + fa[i], ft[i] - fa[i]]

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        """Evaluation function: compute original objectives and store selected two objectives"""
        x = np.array(solution.variables).reshape(1, -1)
        raw_F = self.benchmark.evaluate(x, true_eval=self.true_eval).flatten()
        solution.attributes['original_objectives'] = raw_F.copy()
        # Get selected objective values
        selected_obj_values = raw_F[self.selected_objs].copy()
        # Check if selected objectives contain inf, if yes set all objectives to inf, otherwise assign normally
        if np.isinf(selected_obj_values).any():
            solution.objectives = [np.inf, np.inf]
            solution.attributes['selected_objectives'] = [np.inf, np.inf]
        else:
            solution.objectives = selected_obj_values.tolist()
            solution.attributes['selected_objectives'] = selected_obj_values
        return solution

    def number_of_variables(self) -> int:
        return self.search_space.n_var

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def create_solution(self) -> IntegerSolution:
        """Create initial solution (integer random sampling)"""
        lb = self.search_space.lb
        ub = self.search_space.ub
        variables = [
            np.random.randint(lb[i], ub[i] + 1)
            for i in range(self.number_of_variables())
        ]
        solution = IntegerSolution(
            lower_bound=lb,
            upper_bound=ub,
            number_of_objectives=self.number_of_objectives()
        )
        solution.variables = variables
        return solution

    def name(self) -> str:
        return f"In1KMOPProblem ({self.mode})"


def get_genetic_operator(crx_prob=1.0,
                         crx_eta=30.0,
                         mut_prob=0.9,
                         mut_eta=20.0,
                         problem=None):
    crossover = RepairedIntegerSBXCrossover(
        probability=crx_prob,
        distribution_index=crx_eta
    )
    mutation = RepairedIntegerPolynomialMutation(
        probability=mut_prob if mut_prob is not None else 1.0 / problem.number_of_variables(),
        distribution_index=mut_eta
    )
    return crossover, mutation


def get_benchmark_settings(n_obj):
    n_evaluations = 10000
    pop_size = 100
    ref_dirs = np.array([[i / 99, 1 - i / 99] for i in range(100)])
    return pop_size, n_evaluations, ref_dirs


def nsga2(pop_size, problem, **kwargs):
    crossover, mutation = get_genetic_operator(
        crx_prob=kwargs.get('crx_prob', 1.0),
        crx_eta=kwargs.get('crx_eta', 30.0),
        mut_prob=kwargs.get('mut_prob', 0.9),
        mut_eta=kwargs.get('mut_eta', 20.0),
        problem=problem
    )

    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=pop_size,
        crossover=crossover,
        mutation=mutation,
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=kwargs.get('n_evaluations', 10000))
    )

    return algorithm


def run_single_experiment(args):
    """Run a single experiment with given parameters"""
    # Parse task parameters
    task_type, pid, combo_idx, mode, seed, run = args
    output_dir = "../Results/RQ1-raw-data/NAS/"
    os.makedirs(output_dir, exist_ok=True)

    # Select configuration based on task type
    if task_type == 'c10mop':
        selected_objs = c10_search_space_configs[pid][combo_idx]
        output_file = os.path.join(output_dir, f"c10mop{pid}_{combo_idx}_{seed}_{mode}.csv")
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
        problem_class = C10MOPProblem
    elif task_type == 'citysegmop':
        selected_objs = cityseg_search_space_configs[pid][combo_idx]
        output_file = os.path.join(output_dir, f"citysegmop{pid}_{combo_idx}_{seed}_{mode}.csv")
        benchmark = citysegmop(pid)
        problem_class = CitySegMOPProblem
    else:  # in1kmop
        selected_objs = in1k_search_space_configs[pid][combo_idx]
        output_file = os.path.join(output_dir, f"in1kmop{pid}_{combo_idx}_{seed}_{mode}.csv")
        benchmark = in1kmop(pid)
        problem_class = In1KMOPProblem

    start_time = time.time()
    MAX_RUNTIME = 24 * 3600

    # Create file and write initial information
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([f"NSGA-II Run for {task_type}{pid} combo{combo_idx} with seed {seed} and mode {mode}"])
        writer.writerow(["Budget: 10000"])
        writer.writerow(["评估模式：优化过程使用验证集（true_eval=False），最终测试集评估（true_eval=True）"])
        writer.writerow([])

    try:
        random.seed(seed)
        np.random.seed(seed)
        problem = problem_class(
            benchmark,
            mode=mode,
            selected_objs=selected_objs,
            random_seed=seed
        )

        # Configure NSGA-II algorithm
        pop_size, n_evaluations, _ = get_benchmark_settings(2)
        algorithm = nsga2(pop_size, problem, n_evaluations=n_evaluations)

        # Initialize special variables
        age_info = None
        if mode == 'age_maximization_fa':
            age_info = list(range(1, pop_size + 1))

        novelty_archive = [] if mode == 'novelty_maximization_fa' else None

        # Create initial population and assign uuid
        population = algorithm.create_initial_solutions()
        for solution in population:
            solution.uuid = uuid.uuid4()

        # Evaluate initial population (true_eval=False, using validation set)
        evaluated_population = []
        for solution in population:
            if (time.time() - start_time) >= MAX_RUNTIME:
                break
            problem.evaluate(solution)
            evaluated_population.append(solution)
        population = evaluated_population
        evaluation_count = len(population)

        current_generation = 1

        # Initial novelty archive update
        if novelty_archive is not None:
            update_novelty_archive(population, novelty_archive)

        # Normalize initial population
        problem.normalize_population(
            population,
            age_info=age_info,
            novelty_archive=novelty_archive,
            current_generation=current_generation
        )

        # Initialize best solution tracking
        best_objectives = [float('inf'), float('inf')]
        best_variables = None
        best_generation = 0
        p_values_history = []

        # Main optimization loop (using validation set throughout, true_eval remains False)
        while evaluation_count < n_evaluations and (time.time() - start_time) < MAX_RUNTIME:
            # Selection and reproduction
            mating_population = algorithm.selection(population)
            offspring_population = algorithm.reproduction(mating_population)

            # Assign uuid to offspring
            for solution in offspring_population:
                solution.uuid = uuid.uuid4()

            # Evaluate offspring (true_eval=False, validation set)
            evaluated_offspring = []
            for solution in offspring_population:
                if evaluation_count >= n_evaluations or (time.time() - start_time) >= MAX_RUNTIME:
                    break
                problem.evaluate(solution)
                evaluated_offspring.append(solution)
                evaluation_count += 1

            if (time.time() - start_time) >= MAX_RUNTIME:
                break

            # Merge parent and offspring populations
            combined_population = population + evaluated_offspring

            # Generate combined age information
            combined_age_info = None
            if mode == 'age_maximization_fa':
                offspring_age = [pop_size + current_generation] * len(evaluated_offspring)
                combined_age_info = age_info + offspring_age

            # Normalize combined population
            problem.normalize_population(
                combined_population,
                age_info=combined_age_info,
                novelty_archive=novelty_archive,
                current_generation=current_generation
            )

            # Update novelty archive
            if novelty_archive is not None:
                update_novelty_archive(evaluated_offspring, novelty_archive)

            # Remove duplicates
            unique_combined_population = []
            unique_variables = set()
            for sol in combined_population:
                var_tuple = tuple(sol.variables)
                if var_tuple not in unique_variables:
                    unique_combined_population.append(sol)
                    unique_variables.add(var_tuple)

            # Calculate p value
            non_dominated = get_non_dominated_solutions(unique_combined_population)
            p_value = len(non_dominated) / len(unique_combined_population) if unique_combined_population else 0
            p_values_history.append(p_value)

            # Select next generation population
            if mode == 'ft_fa':
                new_population = algorithm.replacement(population, evaluated_offspring)
            else:
                new_population = partial_duplicate_replacement(combined_population, pop_size)

            # Update age_info
            if mode == 'age_maximization_fa':
                uuid_to_index = {sol.uuid: idx for idx, sol in enumerate(combined_population)}
                selected_indices = [uuid_to_index[sol.uuid] for sol in new_population]
                age_info = [combined_age_info[idx] for idx in selected_indices]

            population = new_population

            # Update best solution
            for solution in population:
                if not any(math.isinf(obj) for obj in solution.objectives):
                    current_makespan = solution.attributes['selected_objectives'][0]
                    current_cost = solution.attributes['selected_objectives'][1]
                    if current_makespan < best_objectives[0]:
                        best_objectives = [current_makespan, current_cost]
                        best_variables = solution.variables.copy()
                        best_generation = current_generation

            print(f"{task_type}{pid} combo{combo_idx} {mode} | Gen {current_generation} | "
                  f"Evaluations: {evaluation_count}/{n_evaluations} | "
                  f"Best: {best_objectives[0]:.10f}")
            current_generation += 1
            # Write generation information
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(
                    [
                        f"Generation {current_generation},p vaue: {p_value:.4f}, best value (验证集): {best_objectives[0]:.8f}"])

        test_set_objectives = None
        if best_variables is not None:
            # Create solution object for test set evaluation
            test_solution = IntegerSolution(
                lower_bound=problem.search_space.lb,
                upper_bound=problem.search_space.ub,
                number_of_objectives=2
            )
            test_solution.variables = best_variables

            # Switch to test set evaluation mode
            original_true_eval = problem.true_eval
            problem.true_eval = True

            # Evaluate on test set
            problem.evaluate(test_solution)
            test_set_objectives = test_solution.attributes['selected_objectives'].copy()

            # Restore original mode (validation set)
            problem.true_eval = original_true_eval

        # Write final results
        runtime = time.time() - start_time
        budget_used = evaluation_count
        p_values_until_best = p_values_history[:best_generation] if best_generation > 0 else []

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([])
            writer.writerow([f"budget_used:{budget_used}"])
            writer.writerow([f"Running time: {runtime:.2f} seconds"])
            writer.writerow([])
            best_p = p_values_history[best_generation - 1] if best_generation > 0 and p_values_history else 0
            writer.writerow([
                f"Best Solution (Validation Set): 'ft': {best_objectives[0]:.10f}, 'fa': {best_objectives[1]:.10f} "
                f"appeared in Generation {best_generation}, p: {best_p:.4f}"
            ])

            # Write test set results
            if test_set_objectives is not None and not any(np.isinf(test_set_objectives)):
                writer.writerow([
                    f"Best Solution (Test Set): 'ft': {test_set_objectives[0]:.10f}, 'fa': {test_set_objectives[1]:.10f}"
                ])
            else:
                writer.writerow(["No valid best solution found to evaluate on test set"])

            if p_values_until_best:
                p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
                writer.writerow([f"p values until best solution: {p_values_str}"])

        return (task_type, pid, combo_idx, mode, seed, run, runtime, best_objectives[0])

    except Exception as e:
        print(
            f"Error in experiment ({task_type} PID:{pid}, Combo:{combo_idx}, Mode:{mode}, Seed:{seed}, Run:{run}): {str(e)}")
        with open(output_file, 'a') as f:
            f.write(f"Error occurred during processing (Seed {seed}): {str(e)}\n")
        return (task_type, pid, combo_idx, mode, seed, run, -1, float('inf'))


def main(argv=None):
    """
    Main entry point with argparse-style arguments.

    Supported arguments (defaults follow original script behavior):
    --use-parallel / --no-parallel   : enable/disable ProcessPoolExecutor (default: enabled)
    --cpu-cores N                     : number of workers for ProcessPoolExecutor (default: CPU_CORES global)
    --mode MODE                       : one of MODES or 'all' (default: 'all')
    --seeds SEEDS                     : seeds specification: single '5', csv '0,1,2' or range '0-9' (default: original SEEDS -> 0..9)

    argv: list of arguments (like sys.argv[1:]) or None to read from sys.argv.
    Returns the list of completed_tasks.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run NAS/MMO experiments with argument configuration")
    # Use the module-level defaults if available
    try:
        default_use_parallel = USE_PARALLEL  # from module
    except NameError:
        default_use_parallel = True
    try:
        default_cpu_cores = CPU_CORES
    except NameError:
        default_cpu_cores = 50
    try:
        default_modes = MODES
    except NameError:
        default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                         'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser.set_defaults(use_parallel=default_use_parallel)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-parallel', dest='use_parallel', action='store_true',
                       help='Enable parallel execution (default from script)')
    group.add_argument('--no-parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel execution')

    parser.add_argument('--cpu-cores', type=int, default=default_cpu_cores,
                        help=f'Number of workers for parallel execution (default: {default_cpu_cores})')

    # Mode: allow any existing mode or 'all'
    parser.add_argument('--mode', type=str, default='all',
                        choices=default_modes + ['all'],
                        help="Single mode to run, or 'all' to run all modes (default: 'all')")

    parser.add_argument('--seeds', type=str, default=None,
                        help="Seeds specification: single '5', csv '0,1,2' or range '0-9'. "
                             "Default: original SEEDS (0-9).")

    parsed = parser.parse_args(argv)

    # Parse seeds
    def parse_seeds_arg(seeds_arg):
        if seeds_arg is None:
            try:
                return list(SEEDS)  # module-level default (range)
            except NameError:
                return list(range(0, 10))
        s = seeds_arg.strip()
        if '-' in s:
            parts = s.split('-', 1)
            start = int(parts[0])
            end = int(parts[1])
            if end < start:
                raise ValueError("Invalid seed range: end < start")
            return list(range(start, end + 1))
        if ',' in s:
            items = [item.strip() for item in s.split(',') if item.strip() != ""]
            return [int(x) for x in items]
        # single value
        return [int(s)]

    try:
        seeds_list = parse_seeds_arg(parsed.seeds)
    except Exception as e:
        raise ValueError(f"Failed to parse --seeds argument '{parsed.seeds}': {e}")

    # Determine modes to run
    if parsed.mode == 'all':
        modes_to_run = default_modes
    else:
        modes_to_run = [parsed.mode]

    use_parallel = parsed.use_parallel
    cpu_cores = max(1, parsed.cpu_cores)

    # Build tasks same as original main logic but using the selected modes and seeds_list
    tasks = []

    # Add C10MOP tasks
    try:
        problem_ids = PROBLEM_IDS
        c10_configs = c10_search_space_configs
    except NameError:
        problem_ids = []
        c10_configs = {}
    for pid in problem_ids:
        num_combos = len(c10_configs[pid])
        for combo_idx in range(num_combos):
            for mode in modes_to_run:
                for seed in seeds_list:
                    for run in range(1, 2):  # keep single run as original
                        tasks.append(('c10mop', pid, combo_idx, mode, seed, run))

    # Add CitySegMOP tasks
    try:
        city_ids = CITYSEG_PROBLEM_IDS
        city_configs = cityseg_search_space_configs
    except NameError:
        city_ids = []
        city_configs = {}
    for pid in city_ids:
        num_combos = len(city_configs[pid])
        for combo_idx in range(num_combos):
            for mode in modes_to_run:
                for seed in seeds_list:
                    for run in range(1, 2):
                        tasks.append(('citysegmop', pid, combo_idx, mode, seed, run))

    # Add IN1KMOP tasks
    try:
        in1k_ids = IN1K_PROBLEM_IDS
        in1k_configs = in1k_search_space_configs
    except NameError:
        in1k_ids = []
        in1k_configs = {}
    for pid in in1k_ids:
        num_combos = len(in1k_configs[pid])
        for combo_idx in range(num_combos):
            for mode in modes_to_run:
                for seed in seeds_list:
                    for run in range(1, 2):
                        tasks.append(('in1kmop', pid, combo_idx, mode, seed, run))

    print(f"Starting processing {len(tasks)} tasks (use_parallel={use_parallel}, cpu_cores={cpu_cores}, modes={modes_to_run}, seeds={seeds_list})")
    start_time = time.time()

    completed_tasks = []

    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(run_single_experiment, task) for task in tasks]

            for future in futures:
                try:
                    result = future.result()
                    completed_tasks.append(result)
                    print(
                        f"Completed: {result[0]} PID{result[1]} Combo{result[2]} | Mode: {result[3]} | Seed: {result[4]} | Run: {result[5]} | Time: {result[6]:.2f}s | Best (Validation): {result[7]:.6f}")
                except Exception as e:
                    print(f"Task failed: {str(e)}")
    else:
        for task in tasks:
            try:
                result = run_single_experiment(task)
                completed_tasks.append(result)
                print(
                    f"Completed: {result[0]} PID{result[1]} Combo{result[2]} | Mode: {result[3]} | Seed: {result[4]} | Run: {result[5]} | Time: {result[6]:.2f}s | Best (Validation): {result[7]:.6f}")
            except Exception as e:
                print(f"Task failed: {str(e)}")

    total_time = time.time() - start_time
    print(f"All tasks finished in {total_time:.2f} seconds. Total completed: {len(completed_tasks)}")

    return completed_tasks


if __name__ == '__main__':
    main()