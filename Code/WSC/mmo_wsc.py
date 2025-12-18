import csv
import os
import time
import random
import math
import json
import uuid
from copy import copy
import sys
import numpy as np
from typing import List
from collections import defaultdict
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.comparator import DominanceComparator
sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.append('../../..')
sys.path.append('..')
from Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from Code.Utils.remove_duplicates import partial_duplicate_replacement


WORKFLOW_DIR = "./Datasets/Original_data/"
DATASET_NAMES = [
    "workflow_1","workflow_2","workflow_3","workflow_4","workflow_5","workflow_6","workflow_7","workflow_8","workflow_9","workflow_10"
]

MODES = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa','age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']
USE_PARALLEL = True
CPU_CORES = 50
MAX_RUNTIME = 24 * 3600


class ServiceCompositionProblem(IntegerProblem):
    def __init__(self, workflow_file: str, seed: int = None, mode: str = 'ft_fa', reverse: bool = False):
        super().__init__()
        self.workflow_file = workflow_file
        self.seed = seed
        self.mode = mode
        self.reverse = reverse
        self._load_workflow()

        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.number_of_variables = len(self.abstract_services)

        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [len(candidates) - 1 for candidates in self.abstract_to_concrete.values()]

        self.evaluation_count = 0

    def _load_workflow(self):
        with open(self.workflow_file, 'r', encoding='utf-8') as f:
            meta_line = f.readline().strip()
            if meta_line.startswith("# META:"):
                self.metadata = json.loads(meta_line[7:])
            else:
                raise ValueError("Invalid workflow file format - missing metadata")

            reader = csv.DictReader(f)
            self.abstract_to_concrete = defaultdict(list)
            self.concrete_to_qos = {}

            service_order = {}
            for row in reader:
                abs_svc = row['abstract_service']
                concrete_id = int(row['concrete_service'])
                latency = float(row['latency'])
                throughput = float(row['throughput'])

                if abs_svc not in service_order:
                    service_order[abs_svc] = len(service_order)

                self.abstract_to_concrete[abs_svc].append(concrete_id)
                self.concrete_to_qos[concrete_id] = {
                    'latency': latency,
                    'throughput': throughput
                }

            self.abstract_services = sorted(service_order.keys(), key=lambda x: service_order[x])
            self.connector_pattern = self.metadata['connector_pattern']
            self._build_workflow_structure()

    def _build_workflow_structure(self):
        self.structure = []
        services = self.abstract_services
        connectors = self.connector_pattern

        if not services:
            return

        current_services = [services[0]]
        current_type = "sequential"

        for i in range(len(connectors)):
            next_service = services[i + 1]
            connector = connectors[i]

            if connector == '1':
                if current_type == "parallel":
                    current_services.append(next_service)
                else:
                    if len(current_services) > 1:
                        self.structure.append(("sequential", current_services[:-1]))
                    current_services = [current_services[-1], next_service]
                    current_type = "parallel"
            else:
                if current_type == "sequential":
                    current_services.append(next_service)
                else:
                    self.structure.append(("parallel", current_services))
                    current_services = [next_service]
                    current_type = "sequential"

        if current_services:
            self.structure.append((current_type, current_services))

    def _calculate_workflow_qos(self, selected_services):
        total_latency = 0
        min_throughput = float('inf')
        last_service = None

        for group_type, services in self.structure:
            if group_type == "parallel":
                block_latency = max(selected_services[s]['latency'] for s in services)
                total_latency = max(total_latency, block_latency)
            else:
                start_idx = 1 if services[0] == last_service else 0
                for service in services[start_idx:]:
                    total_latency += selected_services[service]['latency']

            min_throughput = min(min_throughput,
                                 min(selected_services[s]['throughput'] for s in services))
            last_service = services[-1]

        return total_latency, min_throughput

    def create_solution(self) -> IntegerSolution:
        solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )

        for i, abs_svc in enumerate(self.abstract_services):
            candidates = self.abstract_to_concrete[abs_svc]
            solution.variables[i] = random.randint(0, len(candidates) - 1)

        return solution

    def evaluate(self, solution: IntegerSolution) -> None:
        try:
            selected_services = {}
            for i, abs_svc in enumerate(self.abstract_services):
                concrete_idx = solution.variables[i]
                concrete_id = self.abstract_to_concrete[abs_svc][concrete_idx]
                selected_services[abs_svc] = self.concrete_to_qos[concrete_id]

            latency, throughput = self._calculate_workflow_qos(selected_services)

            if self.reverse:
                solution.ft = -throughput
                solution.fa = latency
            else:
                solution.ft = latency
                solution.fa = -throughput

            solution.objectives = [solution.ft, solution.fa]

            solution.attributes['original_latency'] = latency
            solution.attributes['original_throughput'] = throughput
            solution.attributes['reverse'] = self.reverse

        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            solution.ft = float('inf')
            solution.fa = float('inf')
            solution.objectives = [float('inf'), float('inf')]
            solution.attributes['original_latency'] = float('inf')
            solution.attributes['original_throughput'] = float('-inf')
            solution.attributes['reverse'] = self.reverse

    def normalize_population(self, population: List[IntegerSolution],
                             mode: str, t: int, t_max: int,
                             age_info: List[int], novelty_archive: List[tuple]) -> None:
        valid_solutions = [s for s in population if not any(np.isinf(obj) for obj in s.objectives)]
        if not valid_solutions:
            return

        ft_list = [s.ft for s in valid_solutions]
        fa_list = [s.fa for s in valid_solutions]

        ft_min, ft_max = min(ft_list), max(ft_list)
        fa_min, fa_max = min(fa_list), max(fa_list)

        if mode == 'ft_fa':
            for sol in valid_solutions:
                sol.objectives = [sol.ft, sol.fa]

        elif mode == 'g1_g2':
            for sol in valid_solutions:
                norm_ft = (sol.ft - ft_min) / (ft_max - ft_min) if ft_max != ft_min else 0.5
                norm_fa = (sol.fa - fa_min) / (fa_max - fa_min) if fa_max != fa_min else 0.5
                sol.objectives = [norm_ft + norm_fa, norm_ft - norm_fa]

        else:
            configurations = [tuple(sol.variables) for sol in valid_solutions]
            unique_elements_per_column = [
                sorted({sol.variables[i] for sol in valid_solutions})
                for i in range(len(valid_solutions[0].variables))
            ]
            base_mode = mode.split('_')[0]

            ft_transformed, fa_transformed = generate_fa(
                configurations=configurations,
                ft=ft_list,
                fa_construction=base_mode,
                minimize=True,
                file_path="",
                unique_elements_per_column=unique_elements_per_column,
                t=t,
                t_max=t_max,
                random_seed=self.seed,
                age_info=age_info,
                novelty_archive=novelty_archive,
                k=len(valid_solutions) // 2
            )

            for i, sol in enumerate(valid_solutions):
                sol.objectives = [ft_transformed[i] + fa_transformed[i], ft_transformed[i] - fa_transformed[i]]

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.abstract_services)

    def name(self):
        return "ServiceCompositionProblem"


class UniformCrossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float):
        super(UniformCrossover, self).__init__(probability=probability)

    def execute(self, parents: list[IntegerSolution]) -> list[IntegerSolution]:
        if len(parents) != 2:
            raise Exception(f"The number of parents must be two, but got {len(parents)}.")

        offspring = [
            IntegerSolution(
                lower_bound=parents[0].lower_bound,
                upper_bound=parents[0].upper_bound,
                number_of_objectives=2,
                number_of_constraints=0
            ) for _ in range(2)
        ]

        if random.random() <= self.probability:
            for i in range(len(parents[0].variables)):
                if random.random() < 0.5:
                    offspring[0].variables[i] = parents[0].variables[i]
                    offspring[1].variables[i] = parents[1].variables[i]
                else:
                    offspring[0].variables[i] = parents[1].variables[i]
                    offspring[1].variables[i] = parents[0].variables[i]
        else:
            offspring[0].variables = parents[0].variables.copy()
            offspring[1].variables = parents[1].variables.copy()

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Uniform Crossover"


class BoundaryMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, problem: ServiceCompositionProblem):
        super(BoundaryMutation, self).__init__(probability=probability)
        self.problem = problem

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        mutated = copy(solution)

        for i in range(len(mutated.variables)):
            if random.random() <= self.probability:
                num_candidates = len(self.problem.abstract_to_concrete[self.problem.abstract_services[i]])
                boundary_value = random.choice([0, num_candidates - 1])
                mutated.variables[i] = boundary_value

        return mutated

    def get_name(self) -> str:
        return "Boundary Mutation"


def run_experiment(workflow_file: str, mode: str, output_file: str, seed: int, reverse: bool = False):
    start_time = time.time()
    dataset_name = os.path.splitext(os.path.basename(workflow_file))[0]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        primary_obj = "throughput" if reverse else "latency"
        secondary_obj = "latency" if reverse else "throughput"
        writer.writerow([f"NSGA-II Run for {dataset_name} (seed {seed}, mode {mode})"])
        writer.writerow([f"Primary objective: {primary_obj}, Secondary objective: {secondary_obj} (reverse={reverse})"])
        writer.writerow(["Budget: 5000"])
        writer.writerow([])

    try:
        random.seed(seed)
        np.random.seed(seed)

        problem = ServiceCompositionProblem(
            workflow_file=workflow_file,
            seed=seed,
            mode=mode,
            reverse=reverse
        )

        config = {
            'population_size': 100,
            'offspring_size': 100,
            'budget': 5000,
            'crossover_rate': 1.0 / len(problem.abstract_services),
            'mutation_rate': 1.0,
            'output_file': output_file
        }

        algorithm = NSGAII(
            problem=problem,
            population_size=config['population_size'],
            offspring_population_size=config['offspring_size'],
            crossover=UniformCrossover(probability=config['crossover_rate']),
            mutation=BoundaryMutation(probability=config['mutation_rate'], problem=problem),
            selection=BinaryTournamentSelection(comparator=DominanceComparator()),
            termination_criterion=StoppingByEvaluations(max_evaluations=config['budget'])
        )

        if reverse:
            best_primary = -float('inf')
            best_secondary = float('inf')
        else:
            best_primary = float('inf')
            best_secondary = -float('inf')

        best_variables = None
        best_generation = 0
        t_max = 50
        p_values_history = []
        evaluation_count = 0
        current_generation = 1

        age_info = list(range(1, config['population_size'] + 1)) if mode == 'age_maximization_fa' else None
        novelty_archive = [] if mode == 'novelty_maximization_fa' else None
        uuid_age_map = {}

        population = algorithm.create_initial_solutions()
        for i, solution in enumerate(population):
            if mode == 'age_maximization_fa':
                solution.uuid = uuid.uuid4()
                uuid_age_map[solution.uuid] = age_info[i]
            problem.evaluate(solution)
            evaluation_count += 1

            if not any(math.isinf(obj) for obj in solution.objectives):
                if reverse:
                    current_primary = solution.attributes['original_throughput']
                    current_secondary = solution.attributes['original_latency']
                    if (current_primary > best_primary) or (
                            current_primary == best_primary and current_secondary < best_secondary):
                        best_primary = current_primary
                        best_secondary = current_secondary
                        best_variables = solution.variables.copy()
                        best_generation = current_generation
                else:
                    current_primary = solution.attributes['original_latency']
                    current_secondary = solution.attributes['original_throughput']
                    if (current_primary < best_primary) or (
                            current_primary == best_primary and current_secondary > best_secondary):
                        best_primary = current_primary
                        best_secondary = current_secondary
                        best_variables = solution.variables.copy()
                        best_generation = current_generation

        if mode == 'novelty_maximization_fa':
            update_novelty_archive(population, novelty_archive)
        current_age_info = age_info if mode == 'age_maximization_fa' else None
        problem.normalize_population(
            population=population,
            mode=mode,
            t=current_generation,
            t_max=t_max,
            age_info=current_age_info,
            novelty_archive=novelty_archive
        )

        while evaluation_count < config['budget'] and (time.time() - start_time) < MAX_RUNTIME:
            mating_population = algorithm.selection(population)
            offspring_population = algorithm.reproduction(mating_population)

            if mode == 'age_maximization_fa':
                for solution in offspring_population:
                    solution.uuid = uuid.uuid4()
                    uuid_age_map[solution.uuid] = config['population_size'] + current_generation

            evaluated_offspring = []
            for solution in offspring_population:
                if evaluation_count >= config['budget'] or (time.time() - start_time) >= MAX_RUNTIME:
                    break
                problem.evaluate(solution)
                evaluation_count += 1
                if not any(math.isinf(obj) for obj in solution.objectives):
                    evaluated_offspring.append(solution)

            if mode == 'novelty_maximization_fa':
                update_novelty_archive(evaluated_offspring, novelty_archive)

            combined_population = population + evaluated_offspring

            combined_age_info = [uuid_age_map[sol.uuid] for sol in
                                 combined_population] if mode == 'age_maximization_fa' else None

            problem.normalize_population(
                population=combined_population,
                mode=mode,
                t=current_generation,
                t_max=t_max,
                age_info=combined_age_info,
                novelty_archive=novelty_archive
            )
            unique_combined_population = []
            unique_variables = set()
            for sol in combined_population:
                var_tuple = tuple(sol.variables)
                if var_tuple not in unique_variables:
                    unique_combined_population.append(sol)
                    unique_variables.add(var_tuple)

            non_dominated = get_non_dominated_solutions(unique_combined_population)
            p_value = len(non_dominated) / len(unique_combined_population) if unique_combined_population else 0
            p_values_history.append(p_value)

            if mode == 'ft_fa':
                population = algorithm.replacement(population, evaluated_offspring)
            else:
                population = partial_duplicate_replacement(combined_population, config['population_size'])

            if mode == 'age_maximization_fa':
                population_uuids = {sol.uuid for sol in population}
                uuid_age_map = {uid: age for uid, age in uuid_age_map.items() if uid in population_uuids}
                age_info = [uuid_age_map[sol.uuid] for sol in population]

            for solution in population:
                if not any(math.isinf(obj) for obj in solution.objectives):
                    if reverse:
                        current_primary = solution.attributes['original_throughput']
                        current_secondary = solution.attributes['original_latency']
                        if (current_primary > best_primary) or (
                                current_primary == best_primary and current_secondary < best_secondary):
                            best_primary = current_primary
                            best_secondary = current_secondary
                            best_variables = solution.variables.copy()
                            best_generation = current_generation
                    else:
                        current_primary = solution.attributes['original_latency']
                        current_secondary = solution.attributes['original_throughput']
                        if (current_primary < best_primary) or (
                                current_primary == best_primary and current_secondary > best_secondary):
                            best_primary = current_primary
                            best_secondary = current_secondary
                            best_variables = solution.variables.copy()
                            best_generation = current_generation

            print(f"{dataset_name} {mode} (reverse={reverse}) | Gen {current_generation} | "
                  f"Evaluations: {evaluation_count}/{config['budget']} | "
                  f"Best {primary_obj}: {best_primary:.2f}")
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"Generation {current_generation} p-value: {p_value:.4f} best value: {best_primary:.2f}"])

            current_generation += 1

        runtime = time.time() - start_time
        budget_used = evaluation_count
        p_values_until_best = p_values_history[:best_generation] if best_generation > 0 else []

        with open(output_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow([f"budget_used:{budget_used}"])
            writer.writerow([f"Running time: {runtime:.2f} seconds"])
            writer.writerow([])
            best_p = p_values_history[best_generation - 1] if best_generation > 0 and p_values_history else 0
            writer.writerow([
                f"Best Solution: {primary_obj}={best_primary:.2f}, {secondary_obj}={best_secondary:.2f} "
                f"appeared in Generation {best_generation}, p: {best_p:.4f}, reverse={reverse}"
            ])
            if p_values_until_best:
                p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
                writer.writerow([f"p values until best solution: {p_values_str}"])

    except Exception as e:
        print(f"Error processing {workflow_file} (reverse={reverse}): {str(e)}")
        with open(output_file, 'a') as f:
            f.write(f"Error occurred (reverse={reverse}): {str(e)}\n")


def main(argv=None):
    """
    Argument-driven entry for the Service Composition experiments.

    Supported arguments (names aligned with NAS-style defaults):
      --use-parallel / --no-parallel   : enable/disable parallel execution (default: enabled)
      --cpu-cores N                     : number of worker processes (default: CPU_CORES module var or 80)
      --mode MODE                       : single mode or 'all' (default: 'all')
      --seeds SEEDS                     : seeds specification: single '5', csv '0,1,2' or range '0-9' (default: 0..9)

    Only the parameter interface is changed; algorithm implementation remains unmodified.
    """
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    # Attempt to pick defaults from module-level constants when present
    try:
        default_cpu = CPU_CORES
    except NameError:
        default_cpu = 80
    try:
        default_modes = MODES
    except NameError:
        default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                         'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser = argparse.ArgumentParser(description="Run Service Composition NSGA-II experiments with argument configuration")
    parser.set_defaults(use_parallel=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-parallel', dest='use_parallel', action='store_true',
                       help='Enable parallel execution (default)')
    group.add_argument('--no-parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel execution')

    parser.add_argument('--cpu-cores', type=int, default=default_cpu,
                        help=f'Number of worker processes to use when parallel execution is enabled (default: {default_cpu})')

    parser.add_argument('--mode', type=str, default='all', choices=default_modes + ['all'],
                        help="Single mode to run, or 'all' to run all modes (default: 'all')")

    parser.add_argument('--seeds', type=str, default=None,
                        help="Seeds specification: single '5', csv '0,1,2' or range '0-9'. Default: 0-9.")

    parsed = parser.parse_args(argv)

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

    # Build tasks using the original script logic but with chosen modes and seeds
    tasks = []
    for dataset in DATASET_NAMES:
        workflow_file = os.path.join(WORKFLOW_DIR, f"{dataset}.csv")
        if not os.path.exists(workflow_file):
            print(f"Warning: Workflow file {workflow_file} not found, skipping")
            continue
        for seed in seeds_list:
            for mode in modes_to_run:
                for reverse in [False, True]:
                    suffix = "_reverse" if reverse else ""
                    output_file = os.path.join("../Results/RQ1-raw-data/WSC", f"{dataset}_{seed}_{mode}{suffix}.csv")
                    tasks.append((workflow_file, mode, output_file, seed, reverse))

    print(f"Starting {len(tasks)} experiments (use_parallel={use_parallel}, cpu_cores={cpu_cores}, modes={modes_to_run}, seeds={seeds_list})")
    start_time = time.time()

    if use_parallel:
        print(f"Using parallel processing with {cpu_cores} processes")
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(run_experiment, *task) for task in tasks]
            # Wait for completion and surface exceptions
            for future, task in zip(futures, tasks):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed (workflow={task[0]}, mode={task[1]}, seed={task[3]}, reverse={task[4]}): {e}")
    else:
        print("Running in single-process mode")
        for task in tasks:
            try:
                run_experiment(*task)
            except Exception as e:
                print(f"Task failed (workflow={task[0]}, mode={task[1]}, seed={task[3]}, reverse={task[4]}): {e}")

    total_time = time.time() - start_time
    print(f"All experiments completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()