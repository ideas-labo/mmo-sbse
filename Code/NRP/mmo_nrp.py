import collections
import copy
import csv
import os
import random
import time
import uuid
from typing import List
import numpy as np
from abc import ABCMeta
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.operator import Crossover
from jmetal.core.solution import BinarySolution, Solution
from jmetal.operator import BitFlipMutation, BinaryTournamentSelection
from jmetal.core.problem import BinaryProblem
from jmetal.util.ckecking import Check
from jmetal.util.comparator import DominanceComparator, DominanceWithConstraintsComparator, Comparator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
import sys

sys.path.insert(0, '/home/ccj/code/mmo')
from Code.NRP.Utils.Construct_secondary_objective import generate_fa, update_novelty_archive
from Code.Utils.remove_duplicates import crowding_distance_assignment

class NRP_Problem(BinaryProblem, metaclass=ABCMeta):
    def __init__(self, requirements, clients, dependencies, budget_constraint):
        super().__init__()
        self.requirements = requirements.copy()
        self.clients = clients.copy()
        self.dependencies = dependencies.copy()
        self.max_budget = self.get_max_budget(budget_constraint)
        self.number_of_bits = len(self.clients)
        self.evaluation_count = 0
        self.mode = 'ft_fa'
        self.uuid_age_info = {}
        self.novelty_archive = []
        self.t_max = 1000

    @property
    def number_of_constraints(self) -> int:
        return 1

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )
        new_solution.variables[0] = [random.random() < 0.5 for _ in range(self.number_of_bits)]
        return new_solution

    @property
    def number_of_objectives(self) -> int:
        return 2

    @property
    def number_of_variables(self) -> int:
        return 1

    def get_max_budget(self, budget_constraint):
        total_cost = sum(self.requirements)
        return round(total_cost * budget_constraint)

    def get_selected_requirements(self, customer_candidate):
        selected_requirements = set()
        for i, is_selected in enumerate(customer_candidate):
            if is_selected:
                client_requirements = self.clients[i][1]
                selected_requirements.update(client_requirements)
        selected_requirements = {req - 1 for req in selected_requirements if req > 0}
        return selected_requirements

    def get_score(self, customer_candidate):
        total_score = 0
        for i, is_selected in enumerate(customer_candidate):
            if is_selected:
                total_score += self.clients[i][0]
        return total_score

    def get_cost(self, customer_candidate):
        selected_requirements = self.get_selected_requirements(customer_candidate)
        requirement_candidate = [False] * len(self.requirements)
        for req_idx in selected_requirements:
            if 0 <= req_idx < len(self.requirements):
                requirement_candidate[req_idx] = True
        requirement_candidate = self.simplify_dependencies(requirement_candidate)
        return sum(cost for bit, cost in zip(requirement_candidate, self.requirements) if bit)

    def simplify_dependencies(self, requirement_candidate):
        new_candidate = requirement_candidate.copy()
        queue = collections.deque([i for i, val in enumerate(new_candidate) if val])
        while queue:
            req_idx = queue.popleft()
            dependencies_of_req = self.dependencies.get(req_idx + 1, [])
            for dep in dependencies_of_req:
                dep_idx = dep - 1
                if 0 <= dep_idx < len(new_candidate) and not new_candidate[dep_idx]:
                    new_candidate[dep_idx] = True
                    queue.append(dep_idx)
        return new_candidate

    def normalize_population(self, population: List[BinarySolution], t: int = 1, age_info=[]) -> None:
        valid_solutions = [s for s in population if 'original_score' in s.attributes]
        if not valid_solutions:
            return
        ft = [s.attributes['original_score'] for s in valid_solutions]
        fa = [s.attributes['original_cost'] for s in valid_solutions]
        if self.mode in ['ft_fa', 'g1_g2']:
            for sol in valid_solutions:
                if self.mode == 'ft_fa':
                    sol.objectives = [sol.attributes['original_score'], sol.attributes['original_cost']]
                else:
                    ft_min, ft_max = min(ft), max(ft)
                    fa_min, fa_max = min(fa), max(fa)
                    norm_score = (sol.attributes['original_score'] - ft_min) / (
                            ft_max - ft_min) if ft_max != ft_min else 0.5
                    norm_cost = (sol.attributes['original_cost'] - fa_min) / (
                            fa_max - fa_min) if fa_max != fa_min else 0.5
                    sol.objectives = [norm_score + norm_cost, norm_score - norm_cost]
        elif self.mode in ['penalty_fa', 'gaussian_fa', 'reciprocal_fa',
                           'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables[0]) for sol in valid_solutions]
            unique_elements = [sorted({sol.variables[0][i] for sol in valid_solutions})
                               for i in range(len(valid_solutions[0].variables[0]))]
            current_age_info = None
            if self.mode == 'age_maximization_fa' and age_info:
                if len(age_info) == len(population):
                    current_age_info = [age_info[i] for i, s in enumerate(population) if s in valid_solutions]
                else:
                    print(
                        f"Warning: age_info length ({len(age_info)}) does not match population size ({len(population)})")
            novelty_archive = self.novelty_archive if self.mode == 'novelty_maximization_fa' else None
            mode_prefix = self.mode.split('_')[0]
            ft, fa = generate_fa(
                configurations=configurations,
                ft=ft,
                fa_construction=mode_prefix,
                minimize=True,
                file_path='',
                unique_elements_per_column=unique_elements,
                t=t,
                t_max=self.t_max,
                random_seed=self.random_seed,
                age_info=current_age_info,
                novelty_archive=novelty_archive,
                k=len(population) // 2
            )
            for i, sol in enumerate(valid_solutions):
                sol.objectives = [ft[i] + fa[i], ft[i] - fa[i]]

class SinglePointCrossover(Crossover[BinarySolution, BinarySolution]):
    def __init__(self, probability: float):
        super(SinglePointCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "Single point crossover requires exactly two parents")
        offspring = copy.deepcopy(parents)
        if random.random() <= self.probability:
            for var_idx in range(len(parents[0].variables)):
                length = len(parents[0].variables[var_idx])
                if length > 1:
                    crossover_point = random.randint(1, length - 1)
                    offspring[0].variables[var_idx][crossover_point:] = \
                        parents[1].variables[var_idx][crossover_point:]
                    offspring[1].variables[var_idx][crossover_point:] = \
                        parents[0].variables[var_idx][crossover_point:]
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Single Point Crossover"

class BudgetConstraintComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        violation1 = solution1.constraints[0]
        violation2 = solution2.constraints[0]
        feasible1 = violation1 <= 0
        feasible2 = violation2 <= 0
        if feasible1 and not feasible2:
            return -1
        elif feasible2 and not feasible1:
            return 1
        elif not feasible1 and not feasible2:
            return -1 if violation1 < violation2 else (1 if violation1 > violation2 else 0)
        else:
            return 0

class NRP_MOO(NRP_Problem):
    def __init__(self, requirements, clients, dependencies, budget_constraint, population_size):
        super().__init__(requirements, clients, dependencies, budget_constraint)
        self.population_size = population_size

    @property
    def name(self) -> str:
        return "NRP_MOO"

    def create_initial_solutions(self) -> List[BinarySolution]:
        population = []
        for i in range(self.population_size):
            solution = self._create_random_solution()
            population.append(solution)
        return population

    def _create_random_solution(self) -> BinarySolution:
        solution = BinarySolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )
        customer_candidate = [random.random() < 0.5 for _ in range(len(self.clients))]
        current_cost = self.get_cost(customer_candidate)
        if current_cost > self.max_budget:
            customer_candidate = self._ensure_budget_constraint(customer_candidate)
        solution.variables[0] = customer_candidate
        return solution

    def _ensure_budget_constraint(self, customer_candidate: List[bool]) -> List[bool]:
        current_cost = self.get_cost(customer_candidate)
        while current_cost > self.max_budget:
            selected_indices = [i for i, val in enumerate(customer_candidate) if val]
            if not selected_indices:
                return [False] * len(self.clients)
            customer_values = {}
            for client_idx in selected_indices:
                temp_candidate = customer_candidate.copy()
                temp_candidate[client_idx] = False
                original_cost = current_cost
                new_cost = self.get_cost(temp_candidate)
                cost_contribution = original_cost - new_cost
                client_value = self.clients[client_idx][0]
                if cost_contribution > 0:
                    value_density = client_value / cost_contribution
                else:
                    value_density = float('inf')
                customer_values[client_idx] = value_density
            if customer_values:
                to_remove = min(customer_values.keys(), key=lambda x: customer_values[x])
                customer_candidate[to_remove] = False
                current_cost = self.get_cost(customer_candidate)
            else:
                break
        return customer_candidate

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        customer_candidate = solution.variables[0]
        original_score = -self.get_score(customer_candidate)
        original_cost = self.get_cost(customer_candidate)
        solution.attributes.update({
            'original_score': original_score,
            'original_cost': original_cost,
            'max_budget': self.max_budget,
        })
        solution.objectives=[original_score,original_cost]
        constraint_violation = max(0, original_cost - self.max_budget)
        solution.constraints = [constraint_violation]
        self.evaluation_count += 1
        return solution

def parse(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    number_of_levels = int(lines[0])
    requirements = []
    for i in range(number_of_levels):
        req_costs = list(map(int, lines[(i + 1) * 2].rstrip().split()))
        requirements.extend(req_costs)
    reqs_deps_index = (number_of_levels * 2) + 1
    reqs_deps_number = int(lines[reqs_deps_index])
    number_of_clients = int(lines[reqs_deps_index + reqs_deps_number + 1])
    clients = []
    for line in lines[reqs_deps_index + reqs_deps_number + 2:]:
        parts = line.rstrip().split()
        client_value = int(parts[0])
        reqs = list(map(int, parts[2:]))
        clients.append((client_value, reqs))
    dependencies = {}
    for line in lines[reqs_deps_index + 1: reqs_deps_index + reqs_deps_number + 1]:
        dep_from, dep_to = map(int, line.rstrip().split())
        dependencies.setdefault(dep_from, []).append(dep_to)
    return requirements, clients, dependencies

def partial_duplicate_replacement(combined_population, population_size):
    fronts = fast_non_dominated_sort(combined_population)
    processed_fronts = []
    for i in range(len(fronts)):
        current_front = fronts[i]
        unique_sols = []
        duplicate_sols = []
        seen_vars = set()
        for sol in current_front:
            var_tuple = tuple(tuple(var) for var in sol.variables)
            if var_tuple not in seen_vars:
                unique_sols.append(sol)
                seen_vars.add(var_tuple)
            else:
                duplicate_sols.append(sol)
        processed_fronts.append(unique_sols)
        if duplicate_sols and i < len(fronts) - 1:
            fronts[i + 1].extend(duplicate_sols)
    new_population = []
    remaining = population_size
    for front in processed_fronts:
        if len(front) <= remaining:
            new_population.extend(front)
            remaining -= len(front)
        else:
            if len(front) > 1:
                crowding_distance_assignment(front)
                front.sort(key=lambda x: -x.attributes['crowding_distance'])
            new_population.extend(front[:remaining])
            remaining = 0
        if remaining == 0:
            break
    return new_population

def fast_non_dominated_sort(population):
    from jmetal.util.ranking import FastNonDominatedRanking
    from jmetal.util.comparator import DominanceWithConstraintsComparator
    custom_comparator = DominanceWithConstraintsComparator(
        constraint_comparator=BudgetConstraintComparator()
    )
    ranking = FastNonDominatedRanking(comparator=custom_comparator)
    return ranking.compute_ranking(population)

def run_nsga2_for_dataset(dataset_path: str, mode: str, output_dir: str, random_seed: int):
    requirements, clients, dependencies = parse(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_file = os.path.join(output_dir, f"{dataset_name}_{random_seed}_{mode}.csv")
    start_time = time.time()
    MAX_RUNTIME = 24 * 3600
    config = {
        'population_size': 100,
        'max_evaluations': 100000,
        'crossover_rate': 0.9,
        'mutation_rate': 1/len(clients)
    }
    random.seed(random_seed)
    np.random.seed(random_seed)
    problem = NRP_MOO(
        requirements=requirements,
        clients=clients,
        dependencies=dependencies,
        budget_constraint=0.7,
        population_size=config['population_size']
    )
    problem.mode = mode
    problem.random_seed = random_seed
    algorithm = NSGAII(
        problem=problem,
        population_size=config['population_size'],
        offspring_population_size=config['population_size'],
        mutation=BitFlipMutation(config['mutation_rate']),
        crossover=SinglePointCrossover(config['crossover_rate']),
        selection=BinaryTournamentSelection(
            comparator=DominanceWithConstraintsComparator(
                constraint_comparator=BudgetConstraintComparator()
            )
        ),
        dominance_comparator=DominanceWithConstraintsComparator(
            constraint_comparator=BudgetConstraintComparator()
        ),
        termination_criterion=StoppingByEvaluations(max_evaluations=config['max_evaluations']),
    )
    age_info = None
    if mode == 'age_maximization_fa':
        age_info = list(range(1, config['population_size'] + 1))
    best_score = -float('inf')
    best_cost = float('inf')
    best_variables = None
    best_generation = 0
    evaluations_count = 0
    p_values_history = []
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([f"Customer-Encoded NSGA-II Run for {dataset_name} with seed {random_seed} and mode {mode}"])
        writer.writerow([f"Budget: {config['max_evaluations']}"])
        writer.writerow([f"Time limit: {MAX_RUNTIME / 3600:.1f} hours"])
        writer.writerow([f"Encoding: Customer-based ({len(clients)} dimensions)"])
        writer.writerow([])
    population = problem.create_initial_solutions()
    for solution in population:
        solution.uuid = uuid.uuid4()
    evaluated_population = []
    for solution in population:
        problem.evaluate(solution)
        evaluations_count += 1
        evaluated_population.append(solution)
    population = evaluated_population
    current_generation = 1
    if mode == 'novelty_maximization_fa':
        update_novelty_archive(population, problem.novelty_archive)
    problem.normalize_population(population, t=current_generation, age_info=age_info)
    while evaluations_count < config['max_evaluations']:
        current_time = time.time()
        if current_time - start_time >= MAX_RUNTIME:
            print(f"Time limit reached ({MAX_RUNTIME / 3600:.1f} hours). Terminating.")
            break
        mating_population = algorithm.selection(population)
        offspring_population = algorithm.reproduction(mating_population)
        for solution in offspring_population:
            solution.uuid = uuid.uuid4()
        evaluated_offspring = []
        for solution in offspring_population:
            problem.evaluate(solution)
            evaluations_count += 1
            evaluated_offspring.append(solution)
            if evaluations_count >= config['max_evaluations']:
                break
        combined_population = population + evaluated_offspring
        combined_age_info = None
        if mode == 'age_maximization_fa':
            offspring_age = [config['population_size'] + current_generation] * len(evaluated_offspring)
            combined_age_info = age_info + offspring_age
        problem.normalize_population(combined_population, t=current_generation, age_info=combined_age_info)
        if mode == 'novelty_maximization_fa':
            update_novelty_archive(evaluated_offspring, problem.novelty_archive)
        if mode == 'ft_fa':
            new_population = algorithm.replacement(population, evaluated_offspring)
        else:
            new_population = partial_duplicate_replacement(combined_population, config['population_size'])
        if mode == 'age_maximization_fa':
            uuid_to_index = {sol.uuid: idx for idx, sol in enumerate(combined_population)}
            selected_indices = [uuid_to_index[sol.uuid] for sol in new_population]
            age_info = [combined_age_info[idx] for idx in selected_indices]
        population = new_population
        feasible_solutions = [sol for sol in population if sol.constraints[0] <= 0]
        if feasible_solutions:
            current_best = max(feasible_solutions,
                               key=lambda x: (-x.attributes['original_score'], x.attributes['original_cost']))
            current_score = -current_best.attributes['original_score']
            current_cost = current_best.attributes['original_cost']
            if current_score > best_score:
                best_score = current_score
                best_cost = current_cost
                best_variables = current_best.variables[0].copy()
                best_generation = current_generation
        unique_combined_population = []
        unique_variables = set()
        for sol in combined_population:
            var_tuple = tuple(sol.variables[0])
            if var_tuple not in unique_variables:
                unique_combined_population.append(sol)
                unique_variables.add(var_tuple)
        p_value = len(get_non_dominated_solutions(unique_combined_population)) / len(
            unique_combined_population) if unique_combined_population else 0
        p_values_history.append(p_value)
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([f"Generation {current_generation} p-value: {p_value:.4f} best value: {best_score}"])
        print(f"{dataset_name} {mode} seed {random_seed} | Gen {current_generation} | "
              f"Evaluations: {evaluations_count}/{config['max_evaluations']} | "
              f"Best: {best_score if best_score != -float('inf') else 'inf'} | "
              f"Time: {(time.time() - start_time) / 60:.1f} min")
        current_generation += 1
    runtime = time.time() - start_time
    p_values_until_best = p_values_history[:best_generation] if best_generation > 0 else []
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([])
        writer.writerow([f"budget_used:{evaluations_count}"])
        writer.writerow([f"Running time: {runtime:.2f} seconds"])
        writer.writerow([f"Encoding dimensions: {len(clients)} (customers)"])
        writer.writerow([])
        if best_score == -float('inf'):
            writer.writerow(["Best Solution: 'ft': inf, 'fa': inf (No feasible solution found)"])
        else:
            best_p = p_values_history[best_generation - 1] if (
                        best_generation > 0 and best_generation - 1 < len(p_values_history)) else 0.0
            writer.writerow([
                f"Best Solution: 'ft': {best_score:.6f}, 'fa': {best_cost:.6f} "
                f"appeared in Generation {best_generation}, p: {best_p:.4f}"
            ])
        if p_values_until_best:
            p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
            writer.writerow([f"p values until best solution: {p_values_str}"])

def process_dataset_mode(args):
    dataset_path, mode, output_dir, seed = args
    print(f"Starting {os.path.basename(dataset_path)} | Mode: {mode} | Seed: {seed}")
    start_time = time.time()
    try:
        run_nsga2_for_dataset(dataset_path, mode, output_dir, seed)
        elapsed = time.time() - start_time
        return (dataset_path, mode, elapsed, seed)
    except Exception as e:
        print(f"Seed {seed} failed for {os.path.basename(dataset_path)} (mode {mode}): {str(e)}")
        return (dataset_path, mode, -1, seed)

def main(argv=None):
    """
    Main entry point with argparse-style arguments for the NRP script.

    Supported arguments (defaults follow NAS script defaults where possible):
    --use-parallel / --no-parallel   : enable/disable ProcessPoolExecutor (default: enabled)
    --cpu-cores N                     : number of workers for ProcessPoolExecutor (default: 50)
    --mode MODE                       : one of MODES or 'all' (default: 'all')
    --seeds SEEDS                     : seeds specification: single '5', csv '0,1,2' or range '0-9' (default: 0-9)

    argv: list of arguments (like sys.argv[1:]) or None to read from sys.argv.
    Returns the list of completed_tasks.
    """
    import argparse
    import time
    from concurrent.futures import ProcessPoolExecutor

    # Local defaults (match NAS defaults where applicable)
    default_use_parallel = True
    default_cpu_cores = 50
    default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                     'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']
    default_seeds = None  # None means use original SEEDS range(0,10)

    parser = argparse.ArgumentParser(description="Run NRP NSGA-II experiments with argument configuration")
    # Parallel flags
    parser.set_defaults(use_parallel=default_use_parallel)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-parallel', dest='use_parallel', action='store_true',
                       help='Enable parallel execution (default from script)')
    group.add_argument('--no-parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel execution')
    # CPU cores
    parser.add_argument('--cpu-cores', type=int, default=default_cpu_cores,
                        help=f'Number of workers for parallel execution (default: {default_cpu_cores})')
    # Mode: allow any existing mode or 'all'
    parser.add_argument('--mode', type=str, default='all',
                        choices=default_modes + ['all'],
                        help="Single mode to run, or 'all' to run all modes (default: 'all')")
    # Seeds argument
    parser.add_argument('--seeds', type=str, default=default_seeds,
                        help="Seeds specification: single '5', csv '0,1,2' or range '0-9'. Default: 0-9.")

    parsed = parser.parse_args(argv)

    # Parse seeds helper
    def parse_seeds_arg(seeds_arg):
        if seeds_arg is None:
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

    # Preserve original hard-coded dataset names and defaults
    output_dir = "../Results/RQ1-raw-data/NRP"
    os.makedirs(output_dir, exist_ok=True)
    DATASET_NAMES = ['nrp1.txt', 'nrp2.txt', 'nrp3.txt', 'nrp4.txt', 'nrp5.txt',
                     'nrp-e1.txt', 'nrp-e2.txt', 'nrp-e3.txt', 'nrp-e4.txt',
                     'nrp-g1.txt', 'nrp-g2.txt', 'nrp-g3.txt', 'nrp-g4.txt',
                     'nrp-m1.txt', 'nrp-m2.txt', 'nrp-m3.txt', 'nrp-m4.txt']
    MODES_ALL = default_modes  # list kept for internal reference
    # Use the parsed modes_to_run for task generation
    SEEDS_RANGE = seeds_list

    tasks = []
    for dataset in DATASET_NAMES:
        dataset_path = f"./Datasets/{dataset}"
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset {dataset_path} not found, skipping")
            continue
        for mode in modes_to_run:
            for seed in SEEDS_RANGE:
                tasks.append((dataset_path, mode, output_dir, seed))

    print(f"Starting processing {len(tasks)} tasks with customer encoding "
          f"(use_parallel={use_parallel}, cpu_cores={cpu_cores}, modes={modes_to_run}, seeds={SEEDS_RANGE})")
    start_time = time.time()

    completed_tasks = []
    if use_parallel:
        print(f"Using parallel processing with {cpu_cores} processes")
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = []
            for task in tasks:
                futures.append(executor.submit(process_dataset_mode, task))
            for future in futures:
                try:
                    result = future.result()
                    completed_tasks.append(result)
                    print(
                        f"Completed: {os.path.basename(result[0])} | Mode: {result[1]} | Seed: {result[3]} | Time: {result[2]:.2f}s")
                except Exception as e:
                    print(f"Task failed: {str(e)}")
    else:
        print("Running in sequential mode")
        for task in tasks:
            try:
                result = process_dataset_mode(task)
                completed_tasks.append(result)
                print(
                    f"Completed: {os.path.basename(result[0])} | Mode: {result[1]} | Seed: {result[3]} | Time: {result[2]:.2f}s")
            except Exception as e:
                print(f"Task failed: {str(e)}")

    total_time = time.time() - start_time
    print(f"All tasks finished in {total_time:.2f} seconds. Total completed: {len(completed_tasks)}")
    return completed_tasks


if __name__ == '__main__':
    main()