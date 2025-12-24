import csv
import os
import re
import random
import time
import uuid
from copy import copy
from typing import List
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BinaryTournamentSelection
from jmetal.operator.mutation import BitFlipMutation
from jmetal.operator.crossover import Crossover
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.util.ckecking import Check
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
import numpy as np
from jmetal.util.comparator import DominanceWithConstraintsComparator, Comparator
import sys

from Code.Utils.remove_duplicates import crowding_distance_assignment

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../../..')
from Utils.Construct_secondary_objective import generate_fa, update_novelty_archive

class MultiObjectiveKnapsack(BinaryProblem):
    def __init__(self, number_of_items: int, capacity: int, profits: List[float], weights: List[int],
                 mode: str = 'ft_fa', t_max: int = 400):
        super(MultiObjectiveKnapsack, self).__init__()
        self.number_of_bits = number_of_items
        self._number_of_objectives = 2
        self._number_of_variables = 1
        self._number_of_constraints = 1
        self.capacity = capacity
        self.profits = profits
        self.weights = weights
        self.mode = mode
        self.t_max = t_max
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]
        self.obj_labels = ['Total Profit', 'Number of Items Selected']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_weight = 0.0
        total_profit = 0.0
        selected_count = 0

        for i in range(self.number_of_bits):
            if solution.variables[0][i]:
                total_weight += self.weights[i]
                total_profit += self.profits[i]
                selected_count += 1

        solution.objectives[0] = -total_profit
        solution.objectives[1] = selected_count

        constraint_violation = max(0.0, total_weight - self.capacity)
        solution.constraints[0] = constraint_violation

        solution.attributes['original_profit'] = total_profit
        solution.attributes['original_count'] = selected_count
        solution.attributes['total_weight'] = total_weight

        solution.variables[0] = [int(bit) for bit in solution.variables[0]]

        return solution

    def create_solution(self) -> BinarySolution:
        solution = BinarySolution(number_of_variables=self._number_of_variables,
                                  number_of_objectives=self._number_of_objectives)
        solution.variables[0] = [0] * self.number_of_bits
        solution.constraints = [0.0] * self._number_of_constraints

        remaining_capacity = self.capacity
        items = list(range(self.number_of_bits))
        random.shuffle(items)

        for item in items:
            if self.weights[item] <= remaining_capacity:
                solution.variables[0][item] = 1
                remaining_capacity -= self.weights[item]

        return solution

    def normalize_population(self, population: List[BinarySolution],
                             age_info=None, novelty_archive=None, random_seed=None,generation=1) -> None:
        valid_solutions = [s for s in population if not any(np.isinf(obj) for obj in s.objectives)]

        if not valid_solutions:
            return

        ft = [- s.attributes['original_profit'] for s in valid_solutions]
        fa = [s.attributes['original_count'] for s in valid_solutions]

        configurations = [tuple(sol.variables[0]) for sol in valid_solutions]
        num_vars = len(valid_solutions[0].variables) if valid_solutions else 0
        unique_elements_per_column = [sorted({sol.variables[0][i] for sol in valid_solutions})
                           for i in range(len(valid_solutions[0].variables[0]))]

        if self.mode == 'ft_fa':
            for sol in valid_solutions:
                sol.objectives = [-sol.attributes['original_profit'], sol.attributes['original_count']]

        elif self.mode == 'g1_g2':
            profit = [-s.attributes['original_profit'] for s in valid_solutions]
            count = [s.attributes['original_count'] for s in valid_solutions]

            profit_min, profit_max = min(profit), max(profit)
            count_min, count_max = min(count), max(count)

            for sol in valid_solutions:
                norm_profit = ((-sol.attributes['original_profit'] - profit_min) /
                               (profit_max - profit_min)) if profit_max != profit_min else 0.5
                norm_count = ((sol.attributes['original_count'] - count_min) /
                              (count_max - count_min)) if count_max != count_min else 0.5
                sol.objectives = [norm_profit + norm_count, norm_profit - norm_count]

        else:
            mode_prefix = self.mode.split('_')[0]
            k = len(population) // 2

            adjusted_ft, adjusted_fa = generate_fa(
                configurations=configurations,
                ft=ft,
                fa_construction=mode_prefix,
                minimize=True,
                file_path="",
                unique_elements_per_column=unique_elements_per_column,
                t=generation,
                t_max=self.t_max,
                random_seed=random_seed,
                age_info=age_info,
                novelty_archive=novelty_archive,
                k=k
            )

            for i, sol in enumerate(valid_solutions):
                sol.objectives = [adjusted_ft[i] + adjusted_fa[i], adjusted_ft[i] - adjusted_fa[i]]

    @property
    def number_of_objectives(self) -> int:
        return self._number_of_objectives

    @property
    def number_of_variables(self) -> int:
        return self._number_of_variables

    @property
    def number_of_constraints(self) -> int:
        return self._number_of_constraints

    def name(self) -> str:
        return "Multi-Objective Knapsack Problem"


class BudgetConstraintComparator(Comparator):
    def compare(self, solution1, solution2):
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

class SimpleSinglePointCrossover(Crossover[BinarySolution, BinarySolution]):

    def __init__(self, probability: float):
        super(SimpleSinglePointCrossover, self).__init__(probability=probability)
        self.number_of_parents = 2
        self.number_of_children = 2

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "Single point crossover requires exactly two parents")

        offspring = [copy(parents[0]), copy(parents[1])]

        if random.random() <= self.probability:
            for var_idx in range(len(parents[0].variables)):
                length = len(parents[0].variables[var_idx])
                if length > 1:
                    crossover_point = random.randint(1, length - 1)
                    parent1_genes = parents[0].variables[var_idx]
                    parent2_genes = parents[1].variables[var_idx]
                    offspring[0].variables[var_idx] = parent1_genes[:crossover_point] + parent2_genes[crossover_point:]
                    offspring[1].variables[var_idx] = parent2_genes[:crossover_point] + parent1_genes[crossover_point:]
                    offspring[0].attributes['evaluated'] = False
                    offspring[1].attributes['evaluated'] = False

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Simple Single Point Crossover"


class HalfUniformCrossover(Crossover[BinarySolution, BinarySolution]):

    def __init__(self, probability: float):
        super(HalfUniformCrossover, self).__init__(probability=probability)
        self.number_of_parents = 2
        self.number_of_children = 2

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "Half uniform crossover requires exactly two parents")

        offspring = [copy(parents[0]), copy(parents[1])]

        if random.random() <= self.probability:
            for var_idx in range(len(parents[0].variables)):
                length = len(parents[0].variables[var_idx])
                if length > 0:
                    mask = [random.random() < 0.5 for _ in range(length)]

                    parent1_genes = parents[0].variables[var_idx]
                    parent2_genes = parents[1].variables[var_idx]

                    for i in range(length):
                        if mask[i]:
                            offspring[0].variables[var_idx][i] = parent2_genes[i]
                            offspring[1].variables[var_idx][i] = parent1_genes[i]

                    offspring[0].attributes['evaluated'] = False
                    offspring[1].attributes['evaluated'] = False

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Half Uniform Crossover"

def execute_optimization(input_file, output_file, population_size, max_evaluations, mode='ft_fa', random_seed=42):
    start_time = time.time()
    MAX_RUNTIME = 24 * 3600
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([f"NSGA-II Run for {dataset_name} with seed {random_seed} and mode {mode}"])
        writer.writerow(["Budget: 25000"])
        writer.writerow([])
    try:
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("File is empty")

        first_line = lines[0]
        n_match = re.search(r'(\d+)\s+items', first_line)
        if not n_match:
            raise ValueError(f"Could not find number of items in first line: {first_line}")
        n = int(n_match.group(1))

        capacity_line = next((line for line in lines if 'capacity:' in line), None)
        if not capacity_line:
            raise ValueError("Capacity information not found")
        c_match = re.search(r'capacity:\s*\+?(\d+)', capacity_line)
        if not c_match:
            raise ValueError(f"Unable to parse capacity: {capacity_line}")
        capacity = int(c_match.group(1))

        profits = []
        weights = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('item:'):
                if i + 2 >= len(lines):
                    print(f"Warning: item missing weight/profit line")
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
        random.seed(random_seed)
        np.random.seed(random_seed)
        problem = MultiObjectiveKnapsack(
            number_of_items=n,
            capacity=capacity,
            profits=profits,
            weights=weights,
            mode=mode,
            t_max=250
        )

        algorithm = NSGAII(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size,
            mutation=BitFlipMutation(0.01),
            crossover=HalfUniformCrossover(probability=1.0),
            selection=BinaryTournamentSelection(
                comparator=DominanceWithConstraintsComparator(
                    constraint_comparator=BudgetConstraintComparator()
                )
            ),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            dominance_comparator=BudgetConstraintComparator()
        )

        age_info = None
        if mode == 'age_maximization_fa':
            age_info = list(range(1, population_size + 1))

        novelty_archive = [] if mode in ['novelty_maximization_fa', 'diversity_fa'] else None
        random.seed(random_seed)

        population = algorithm.create_initial_solutions()
        for solution in population:
            solution.uuid = uuid.uuid4()

        evaluated_population = []

        for solution in population:
            problem.evaluate(solution)
            evaluated_population.append(solution)
        population = evaluated_population
        evaluations = len(population)
        current_age_info = age_info if mode == 'age_maximization_fa' else None
        current_generation = 1
        if novelty_archive is not None:
            update_novelty_archive(population, novelty_archive)
        problem.normalize_population(
            population=population,
            age_info=current_age_info,
            novelty_archive=novelty_archive,
            random_seed=random_seed,generation=current_generation
        )

        best_profit = float('-inf')
        best_count = float('inf')
        best_weight = float('inf')
        best_solution = None
        best_generation = 1

        p_values_history = []

        for solution in population:
            current_profit = solution.attributes['original_profit']
            current_weight = solution.attributes['total_weight']
            if current_weight <= capacity:
                if current_profit > best_profit or (current_profit == best_profit and solution.attributes['original_count'] < best_count):
                    best_profit = current_profit
                    best_count = solution.attributes['original_count']
                    best_weight = current_weight
                    best_solution = solution
                    best_generation = current_generation

        while evaluations < max_evaluations and (time.time() - start_time) < MAX_RUNTIME:
            if (time.time() - start_time) >= MAX_RUNTIME:
                break

            mating_population = algorithm.selection(population)
            offspring_population = algorithm.reproduction(mating_population)

            for solution in offspring_population:
                solution.uuid = uuid.uuid4()

            evaluated_offspring = []
            for solution in offspring_population:
                if evaluations >= max_evaluations:
                    break
                problem.evaluate(solution)
                evaluated_offspring.append(solution)
                evaluations += 1

            combined_population = population + evaluated_offspring

            unique_combined_population = []
            unique_variables = set()
            for sol in combined_population:
                var_tuple = tuple(tuple(var) for var in sol.variables)
                if var_tuple not in unique_variables:
                    unique_combined_population.append(sol)
                    unique_variables.add(var_tuple)

            p_value = len(get_non_dominated_solutions(unique_combined_population)) / len(unique_combined_population) if unique_combined_population else 0.0
            p_values_history.append(p_value)

            combined_age_info = None
            if mode == 'age_maximization_fa':
                offspring_age = [population_size + current_generation] * len(evaluated_offspring)
                combined_age_info = age_info + offspring_age

            problem.normalize_population(
                population=combined_population,
                age_info=combined_age_info if mode == 'age_maximization_fa' else None,
                novelty_archive=novelty_archive,
                random_seed=random_seed,generation=current_generation
            )

            if novelty_archive is not None and evaluated_offspring:
                update_novelty_archive(evaluated_offspring, novelty_archive)

            if mode == 'ft_fa':
                population = algorithm.replacement(population, evaluated_offspring)
            else:
                population = partial_duplicate_replacement(combined_population, population_size)

            if mode == 'age_maximization_fa':
                uuid_to_index = {sol.uuid: idx for idx, sol in enumerate(combined_population)}
                selected_indices = [uuid_to_index[sol.uuid] for sol in population]
                age_info = [combined_age_info[idx] for idx in selected_indices]

            for solution in population:
                current_profit = solution.attributes['original_profit']
                current_weight = solution.attributes['total_weight']
                if current_weight <= capacity:
                    if current_profit > best_profit or (current_profit == best_profit and solution.attributes['original_count'] < best_count):
                        best_profit = current_profit
                        best_count = solution.attributes['original_count']
                        best_weight = current_weight
                        best_solution = solution
                        best_generation = current_generation
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow([f"Generation {current_generation} p-value: {p_value:.4f} best value: {best_profit:.6f}"])
            print(f"Mode: {mode} | Gen: {current_generation} | Evals: {evaluations}/{max_evaluations} | "
                  f"Best profit: {best_profit:.6f} | Selected count: {best_count} | Weight: {best_weight:.2f}/{capacity}")

            current_generation += 1

        runtime = time.time() - start_time
        budget_used = evaluations
        termination_reason = "max evaluations reached" if evaluations >= max_evaluations else \
            "24-hour timeout" if runtime >= MAX_RUNTIME else "unknown"

        p_values_until_best = p_values_history[:best_generation] if best_generation > 0 else []

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([])
            writer.writerow([f"Termination reason: {termination_reason}"])
            writer.writerow([f"budget_used:{budget_used}"])
            writer.writerow([f"Running time: {runtime:.2f} seconds"])
            writer.writerow([])
            best_p = p_values_history[best_generation - 1] if best_generation > 0 and p_values_history else 0
            writer.writerow([
                f"Best Solution: 'ft': {best_profit:.6f}, 'fa': {best_count:.6f} "
                f"appeared in Generation {best_generation}, p: {best_p:.4f}"
            ])

            if p_values_until_best:
                p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
                writer.writerow([f"p values until best solution: {p_values_str}"])


    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()


def partial_duplicate_replacement(combined_population, population_size):
    fronts = fast_non_dominated_sort_with_budget_comparator(combined_population)

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


def fast_non_dominated_sort_with_budget_comparator(population):
    from jmetal.util.ranking import FastNonDominatedRanking

    dominance_comparator = DominanceWithConstraintsComparator(
        constraint_comparator=BudgetConstraintComparator()
    )

    ranking = FastNonDominatedRanking(comparator=dominance_comparator)
    return ranking.compute_ranking(population)

def process_knapsack_task(args):
    input_file, output_file, population_size, max_evaluations, mode, random_seed = args
    print(f"Starting: {input_file} | Mode: {mode} | Seed: {random_seed}")
    start_time = time.time()

    try:
        execute_optimization(
            input_file=input_file,
            output_file=output_file,
            population_size=population_size,
            max_evaluations=max_evaluations,
            mode=mode,
            random_seed=random_seed
        )
        elapsed = time.time() - start_time
        return (input_file, mode, elapsed, random_seed)
    except Exception as e:
        print(f"Seed {random_seed} failed: {str(e)}")
        return (input_file, mode, -1, random_seed)



import argparse
import os
from concurrent.futures import ProcessPoolExecutor

def _parse_seeds_arg(seeds_arg):
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
    return [int(s)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run knapsack NSGA-II experiments with argument configuration")

    default_cpu_cores = 50
    default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                     'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser.set_defaults(use_parallel=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-parallel', dest='use_parallel', action='store_true',
                       help='Enable parallel execution (default)')
    group.add_argument('--no-parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel execution')

    parser.add_argument('--cpu-cores', type=int, default=default_cpu_cores,
                        help=f'Number of worker processes to use when parallel execution is enabled (default: {default_cpu_cores})')

    parser.add_argument('--mode', type=str, default='all', choices=default_modes + ['all'],
                        help="Single mode to run, or 'all' to run all modes (default: 'all')")

    parser.add_argument('--seeds', type=str, default=None,
                        help="Seeds specification: single '5', csv '0,1,2' or range '0-9'. Default: 0-9.")

    parsed = parser.parse_args(argv)

    try:
        seeds_list = _parse_seeds_arg(parsed.seeds)
    except Exception as e:
        raise ValueError(f"Failed to parse --seeds argument '{parsed.seeds}': {e}")

    if parsed.mode == 'all':
        modes_to_run = default_modes
    else:
        modes_to_run = [parsed.mode]

    use_parallel = parsed.use_parallel
    cpu_cores = max(1, parsed.cpu_cores)

    input_folders = ['input_ALL', 'input_CO' ,'input_MS','input_DS']
    migration_rules = ["migrationRule1","migrationRule2","migrationRule3","migrationRule4","migrationRule5","migrationRule7","migrationRule8","migrationRule10","migrationRule18"]
    population_size = 100
    max_evaluations = 25000
    input_root = "./Datasets"
    output_dir = "../Results/RQ1-raw-data/TPLM"
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for rule in migration_rules:
        for input_folder in input_folders:
            category = input_folder.replace("input_", "")
            if rule == "migrationRule3" and input_folder == "input_DS":
                continue
            for mode in modes_to_run:
                for run_idx, seed in enumerate(seeds_list):
                    input_dir = os.path.join(
                        input_root,
                        rule,
                        "pop_size_250/input/",
                        input_folder,
                        f"run_{run_idx}"
                    )
                    input_file = os.path.join(input_dir, "knapsack_file")

                    dataset_identifier = f"knapsack_{rule}_{category}_run_{run_idx}"

                    output_file = os.path.join(
                        output_dir,
                        f"{rule}_{category}_{seed}_{mode}.csv"
                    )

                    task = (
                        input_file,
                        output_file,
                        population_size,
                        max_evaluations,
                        mode,
                        seed
                    )
                    tasks.append(task)

    print(f"Prepared {len(tasks)} tasks (modes={modes_to_run}, seeds={seeds_list}, use_parallel={use_parallel}, cpu_cores={cpu_cores})")

    if use_parallel:
        print(f"Using multiprocessing, pool size: {cpu_cores}")
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(process_knapsack_task, task) for task in tasks]
            for future in futures:
                try:
                    workflow, mode, elapsed, seed = future.result()
                    print(
                        f"Completed task: {os.path.basename(workflow)} | Mode: {mode} | Seed: {seed} | Time: {elapsed:.2f}s")
                except Exception as e:
                    print(f"Task failed: {str(e)}")
    else:
        print("Running in single-process mode")
        for task in tasks:
            try:
                workflow, mode, elapsed, seed = process_knapsack_task(task)
                print(f"Completed task: {os.path.basename(workflow)} | Mode: {mode} | Seed: {seed} | Time: {elapsed:.2f}s")
            except Exception as e:
                print(f"Task failed: {str(e)}")

    print("All experiments completed.")


if __name__ == "__main__":
    main()