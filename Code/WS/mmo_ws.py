import csv
import threading
import os
import time
import random
import math
import uuid
import sys

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
import numpy as np
from typing import List
from collections import defaultdict
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.comparator import DominanceComparator
from Utils.Construct_secondary_objective import generate_fa, update_novelty_archive
from Utils.Load_workflow_class import PegasusWorkflowLoader
from Utils.MOHEFT_class import MOHEFT, INFRA_CONFIG
from Utils.Nsga2_operator_workfolw import ImprovedZhuMutation, ImprovedZhuCrossover
from Utils.DAGCentralSimulator_class import DAGCentralSimulator
from Utils.DAG_class import Task
from Code.Utils.remove_duplicates import partial_duplicate_replacement

WORKFLOW_DIR = "./Datasets/"

DATASET_NAMES = [
    "CyberShake_30", "CyberShake_50", "CyberShake_100", "CyberShake_1000",
    "Epigenomics_24", "Epigenomics_46", "Epigenomics_100", "Epigenomics_997",
    "Inspiral_30", "Inspiral_50", "Inspiral_100", "Inspiral_1000",
    "Montage_25", "Montage_50", "Montage_100", "Montage_1000",
    "Sipht_30", "Sipht_60", "Sipht_100", "Sipht_1000"
]
MODES = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa','age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']
USE_PARALLEL = True
CPU_CORES = 50
MAX_RUNTIME = 24 * 3600

def get_budget_for_dataset(name):
    budget_map = {
        "CyberShake_30":1900, "CyberShake_50":5000, "CyberShake_100":2300, "CyberShake_1000":1400,
        "Epigenomics_24":1800, "Epigenomics_46":2200, "Epigenomics_100":1500, "Epigenomics_997":1500,
        "Inspiral_30":5000, "Inspiral_50":2100, "Inspiral_100":1200, "Inspiral_1000":1600,
        "Montage_25":4100, "Montage_50":5000, "Montage_100":3000, "Montage_1000":1600,
        "Sipht_30":4500, "Sipht_60":5000, "Sipht_100":5000, "Sipht_1000":1900
    }
    return budget_map.get(name, 200)

class VmsProblem(IntegerProblem):
    def __init__(self, workflow_file: str, max_simultaneous_ins: int = 10, seed: int = None, mode: str = 'ft_fa'):
        super().__init__()
        self.workflow_file = workflow_file
        self.max_simultaneous_ins = max_simultaneous_ins
        self.seed = seed
        self.evaluation_count = 0
        self.mode = mode
        self._load_workflow()
        self._init_infra()
        self._validate_tasks()
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.number_of_variables = 2 * self.num_valid_tasks + max_simultaneous_ins
        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = ([self.num_valid_tasks - 1] * self.num_valid_tasks +
                            [self.max_simultaneous_ins - 1] * self.num_valid_tasks +
                            [7] * self.max_simultaneous_ins)
        self._generate_heft_seeds()

    def _init_infra(self):
        self.vm_configs = INFRA_CONFIG["vm_types"]
        if not any(vm['name'] == 'm3.medium' for vm in self.vm_configs):
            raise ValueError("必须包含m3.medium VM类型")

    def _validate_tasks(self):
        valid_ids = {t.id for t in self.tasks}
        for task in self.tasks:
            if task.id < 0 and task.id not in {self.dag.entry_id, self.dag.exit_id}:
                raise ValueError(f"无效任务ID: {task.id}")
            if task.id >= 0 and not any(p in valid_ids for p in self.dag.requiring.get(task.id, [])):
                raise ValueError(f"任务{task.id}缺少有效前驱")

    def _load_workflow(self):
        loader = PegasusWorkflowLoader(self.workflow_file)
        self.tasks, self.dag = loader.load()
        self.num_valid_tasks = len([t for t in self.tasks if t.id >= 0])
        self.dag.totalCloudletNum = self.num_valid_tasks
        self.dag.defultTotalCloudletNum = self.num_valid_tasks

    def _generate_heft_seeds(self):
        self.heft_seeds = MOHEFT.generate_initial_solutions(self.dag, self.max_simultaneous_ins)

    def create_solution(self) -> IntegerSolution:
        if len(self.heft_seeds) > 0:
            task_order, vm_mapping, vm_types = self.heft_seeds.pop(0)
        else:
            task_order =self.dag.get_topological_order()
            if random.random() < 0.5:
                vm_mapping = [0] * self.num_valid_tasks
            else:
                vm_mapping = [random.randint(0, self.max_simultaneous_ins - 1)
                              for _ in range(self.num_valid_tasks)]
            vm_type = random.randint(0, 7)
            vm_types = [vm_type] * self.max_simultaneous_ins

        solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        solution.variables = task_order + vm_mapping + vm_types
        return solution

    def normalize_population(self, population: List[IntegerSolution],
                             mode: str, t: int, t_max: int,
                             age_info: List[int], novelty_archive: List[tuple]) -> None:
        valid_solutions = [s for s in population if not any(np.isinf(obj) for obj in s.objectives)]

        ft = [s.attributes['original_makespan'] for s in valid_solutions]
        fa = [s.attributes['original_cost'] for s in valid_solutions]

        if not ft:
            return

        ft_min, ft_max = min(ft), max(ft)
        fa_min, fa_max = min(fa), max(fa)

        if mode == 'ft_fa':
            for sol in valid_solutions:
                sol.objectives = [sol.attributes['original_makespan'], sol.attributes['original_cost']]

        elif mode=='g1_g2':
            for sol in valid_solutions:
                norm_sae = (sol.attributes['original_makespan'] - ft_min) / (
                        ft_max - ft_min) if ft_max != ft_min else 0.5
                norm_ci = (sol.attributes['original_cost'] - fa_min) / (fa_max - fa_min) if fa_max != fa_min else 0.5
                sol.objectives = [norm_sae+norm_ci, norm_sae-norm_ci]

        elif mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa',
                      'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_solutions]
            unique_elements = [sorted({sol.variables[i] for sol in valid_solutions})
                               for i in range(len(valid_solutions[0].variables))]

            mode = mode.split('_')[0]
            ft, fa = generate_fa(configurations,ft,mode,True,"",
                unique_elements_per_column=unique_elements,t=t,t_max=t_max,random_seed=self.seed,
                age_info=age_info,novelty_archive=novelty_archive,k=len(population)//2)

            for i, sol in enumerate(valid_solutions):
                sol.objectives = [ft[i] + fa[i], ft[i] - fa[i]]

    def _is_valid_order(self, order: List[int]) -> bool:
        seen = set()
        for task_id in order:
            for pred in self.dag.requiring.get(task_id, []):
                if pred >= 0 and pred not in seen:
                    return False
            seen.add(task_id)
        return True

    def evaluate(self, solution: IntegerSolution) -> None:
        try:
            n = self.num_valid_tasks
            real_task_order = [tid for tid in solution.variables[:n] if tid >= 0]

            if len(real_task_order) != n or not self._is_valid_order(real_task_order):
                solution.objectives = [float('inf'), float('inf')]
                solution.attributes['original_makespan'] = float('inf')
                solution.attributes['original_cost'] = float('inf')
                return

            task2ins = solution.variables[n:2 * n]
            ins2type = solution.variables[2 * n:2 * n + self.max_simultaneous_ins]
            ins2type = [min(max(vm_type, 0), len(self.vm_configs) - 1) for vm_type in ins2type]

            vmlist = self._create_vm_list(task2ins, ins2type)

            entry_task = self.dag.entry_task
            entry_task.vm_id = 0
            entry_task.set_status(Task.READY)

            self.dag.calc_file_transfer_times(task2ins, vmlist)

            simulator = DAGCentralSimulator()
            simulator.set_cloudlet_dag(self.dag)
            simulator.set_vm_list(vmlist)

            submitted_count = 0
            for task_id in real_task_order:
                if task_id >= len(self.dag.tasks):
                    raise ValueError(f"Invalid task ID: {task_id}")
                task = self.dag.tasks[task_id]
                ins_id = task2ins[task_id]
                task.vm_id = ins_id
                transfer_time = self.dag.file_transfer_time.get(task.id, 0.0)
                simulator.task_submit(task, transfer_time)
                submitted_count += 1

            if submitted_count != self.dag.totalCloudletNum:
                raise ValueError(f"任务数不匹配: 提交{submitted_count}, 期望{self.dag.totalCloudletNum}")

            if not simulator.boot():
                solution.objectives = [float('inf'), float('inf')]
                solution.attributes['original_makespan'] = float('inf')
                solution.attributes['original_cost'] = float('inf')
                return

            makespan = simulator._calculate_makespan()
            cost = self._calculate_cost(simulator, vmlist)

            solution.attributes['original_makespan'] = makespan
            solution.attributes['original_cost'] = cost
            solution.objectives = [makespan, cost]

        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            solution.objectives = [float('inf'), float('inf')]
            solution.attributes['original_makespan'] = float('inf')
            solution.attributes['original_cost'] = float('inf')

    def _create_vm_list(self, task2ins: List[int], ins2type: List[int]) -> List[dict]:
        vmlist = []
        for vm_id in range(self.max_simultaneous_ins):
            vm_type_idx = ins2type[vm_id] if vm_id < len(ins2type) else 0
            vm_type_idx = max(0, min(vm_type_idx, len(self.vm_configs) - 1))
            vm_config = self.vm_configs[vm_type_idx]

            if vm_id == 0 and vm_config['name'] != 'm3.medium':
                vm_config = next(vm for vm in self.vm_configs if vm['name'] == 'm3.medium')

            vmlist.append({
                'id': vm_id,
                'mips': vm_config['mips'],
                'bw': vm_config['bw'],
                'price': vm_config['price'],
                'size': vm_config.get('size', 10000),
                'ram': vm_config.get('ram', 4096)
            })
        return vmlist

    def _calculate_cost(self, simulator, vmlist) -> float:
        vm_usage = defaultdict(lambda: {'start': float('inf'), 'end': 0.0})

        for task in (t for t in simulator.finished_tasks if t.id >= 0):
            vm_id = task.vm_id
            vm_usage[vm_id]['start'] = min(vm_usage[vm_id]['start'], task.exec_start_time)
            vm_usage[vm_id]['end'] = max(vm_usage[vm_id]['end'], task.finish_time)

        total_cost = 0.0
        for vm_id, usage in vm_usage.items():
            if usage['end'] > usage['start']:
                duration_hours = math.ceil((usage['end'] - usage['start']) / 3600)
                vm_price = next(vm['price'] for vm in vmlist if vm['id'] == vm_id)
                total_cost += duration_hours * vm_price

        return total_cost

    def number_of_objectives(self) -> int:
        return self.number_of_objectives

    def number_of_constraints(self) -> int:
        return self.number_of_constraints

    def number_of_variables(self) -> int:
        return self.number_of_variables

    def name(self):
        return "VmsProblem (Manual MinMax Normalization)"


def run_experiment(workflow_file: str, mode: str, output_file: str, seed: int):
    start_time = time.time()
    dataset_name = os.path.splitext(os.path.basename(workflow_file))[0]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([f"NSGA-II Run for {dataset_name} with seed {seed} and mode {mode}"])
        writer.writerow(["Budget: 50000"])
        writer.writerow([])

    try:
        random.seed(seed)
        np.random.seed(seed)
        problem = VmsProblem(workflow_file=workflow_file,
                             max_simultaneous_ins=10,
                             seed=seed,
                             mode=mode)

        config = {
            'population_size': 50,
            'budget': 50000,
            'crossover_rate': 1.0,
            'mutation_rate': 1.0 / problem.num_valid_tasks,
            'output_file': output_file
        }

        algorithm = NSGAII(
            problem=problem,
            population_size=config['population_size'],
            offspring_population_size=config['population_size'],
            mutation=ImprovedZhuMutation(probability=config['mutation_rate'], problem=problem),
            crossover=ImprovedZhuCrossover(probability=config['crossover_rate'], problem=problem),
            selection=BinaryTournamentSelection(comparator=DominanceComparator()),
            termination_criterion=StoppingByEvaluations(max_evaluations=config['budget'])
        )

        best_makespan = float('inf')
        best_cost = float('inf')
        best_variables = None
        best_generation = 0
        t_max = 1000
        p_values_history = []
        evaluation_count = 0

        age_info = None
        if mode == 'age_maximization_fa':
            age_info = list(range(1, config['population_size'] + 1))

        novelty_archive = [] if mode == 'novelty_maximization_fa' else None

        population = algorithm.create_initial_solutions()
        for solution in population:
            solution.uuid = uuid.uuid4()

        evaluated_population = []
        for solution in population:
            problem.evaluate(solution)
            evaluation_count += 1
            if not any(math.isinf(obj) for obj in solution.objectives):
                evaluated_population.append(solution)
                current_makespan = solution.attributes['original_makespan']
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_cost = solution.attributes['original_cost']
                    best_variables = solution.variables.copy()
                    best_generation = 1

        population = evaluated_population
        current_generation = 1
        if novelty_archive is not None:
            update_novelty_archive(population, novelty_archive)
        problem.normalize_population(
            population=population,
            mode=mode,
            t=current_generation,
            t_max=t_max,
            age_info=age_info,
            novelty_archive=novelty_archive
        )

        while True:
            if evaluation_count >= config['budget'] or (time.time() - start_time) >= MAX_RUNTIME:
                break

            mating_population = algorithm.selection(population)
            offspring_population = algorithm.reproduction(mating_population)

            for solution in offspring_population:
                solution.uuid = uuid.uuid4()

            evaluated_offspring = []
            for solution in offspring_population:
                if evaluation_count >= config['budget'] or (time.time() - start_time) >= MAX_RUNTIME:
                    break
                problem.evaluate(solution)
                evaluation_count += 1
                if not any(math.isinf(obj) for obj in solution.objectives):
                    evaluated_offspring.append(solution)

            combined_population = population + evaluated_offspring

            combined_age_info = None
            if mode == 'age_maximization_fa':
                offspring_age = [config['population_size'] + current_generation] * len(evaluated_offspring)
                combined_age_info = age_info + offspring_age

            problem.normalize_population(
                population=combined_population,
                mode=mode,
                t=current_generation,
                t_max=t_max,
                age_info=combined_age_info,
                novelty_archive=novelty_archive
            )

            if novelty_archive is not None:
                update_novelty_archive(evaluated_offspring, novelty_archive)

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
                new_population = algorithm.replacement(population, evaluated_offspring)
            else:
                new_population = partial_duplicate_replacement(combined_population, config['population_size'])

            if mode == 'age_maximization_fa':
                uuid_to_index = {sol.uuid: idx for idx, sol in enumerate(combined_population)}
                selected_indices = [uuid_to_index[sol.uuid] for sol in new_population]
                age_info = [combined_age_info[idx] for idx in selected_indices]

            population = new_population

            for solution in population:
                if not any(math.isinf(obj) for obj in solution.objectives):
                    current_makespan = solution.attributes['original_makespan']
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_cost = solution.attributes['original_cost']
                        best_variables = solution.variables.copy()
                        best_generation = current_generation

            print(f"{dataset_name} {mode} | Gen {current_generation} | "
                  f"Evaluations: {evaluation_count}/{config['budget']} | "
                  f"Best: {best_makespan:.2f}")
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(
                    [f"Generation {current_generation} p-value: {p_value:.4f} best value: {best_makespan}"])

            current_generation += 1

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
                f"Best Solution: 'ft': {best_makespan:.6f}, 'fa': {best_cost:.6f} "
                f"appeared in Generation {best_generation}, p: {best_p:.4f}"
            ])

            if p_values_until_best:
                p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
                writer.writerow([f"p values until best solution: {p_values_str}"])

    except Exception as e:
        print(f"Error processing {workflow_file} with mode {mode}: {str(e)}")
        with open(output_file, 'a') as f:
            f.write(f"Error occurred during processing (Seed {seed}): {str(e)}\n")


def process_dataset_mode(args):
    workflow_file, mode, output_file, seed = args
    print(f"Starting {workflow_file} | Mode: {mode} | Seed: {seed}")
    start_time = time.time()

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        run_experiment(workflow_file, mode, output_file, seed)
        elapsed = time.time() - start_time
        return (workflow_file, mode, elapsed, seed)
    except Exception as e:
        print(f"Seed {seed} failed: {str(e)}")
        return (workflow_file, mode, -1, seed)



import argparse
import time
from concurrent.futures import ProcessPoolExecutor

def _parse_seeds_arg(seeds_arg):
    """
    Parse seeds argument string.
    Acceptable formats:
    - None -> default 0-9 (no dependency on an undefined SEEDS name)
    - '5' -> [5]
    - '0,1,2' -> [0,1,2]
    - '0-9' -> [0..9]
    """
    # Default to 0..9 when no explicit seeds are provided
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
    """
    Argument-driven entry for the workflow NSGA-II experiments.

    Supported arguments (names aligned with NAS-style defaults):
      --use-parallel / --no-parallel   : enable/disable ProcessPoolExecutor (default: enabled)
      --cpu-cores N                     : number of worker processes (default: CPU_CORES module var or 50)
      --mode MODE                       : single mode or 'all' (default: 'all')
      --seeds SEEDS                     : seeds specification: single '5', csv '0,1,2' or range '0-9' (default: module SEEDS or 0..9)

    Only the parameter interface is changed; algorithm implementation remains unmodified.
    Returns list of completed_tasks (tuples).
    """
    parser = argparse.ArgumentParser(description="Run workflow NSGA-II experiments with argument configuration")

    # try to use module-level defaults if available
    try:
        default_cpu = CPU_CORES
    except NameError:
        default_cpu = 50
    try:
        default_modes = MODES
    except NameError:
        default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                         'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

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
                        help="Seeds specification: single '5', csv '0,1,2' or range '0-9'. Default: module SEEDS or 0-9.")

    parsed = parser.parse_args(argv)

    try:
        seeds_list = _parse_seeds_arg(parsed.seeds)
    except Exception as e:
        raise ValueError(f"Failed to parse --seeds argument '{parsed.seeds}': {e}")

    # determine modes to run
    if parsed.mode == 'all':
        modes_to_run = default_modes
    else:
        modes_to_run = [parsed.mode]

    use_parallel = parsed.use_parallel
    cpu_cores = max(1, parsed.cpu_cores)

    # Build tasks same as original main logic but using selected modes and seeds_list
    tasks = []
    for dataset in DATASET_NAMES:
        workflow_file = os.path.join(WORKFLOW_DIR, f"{dataset}.xml")
        if not os.path.exists(workflow_file):
            print(f"Warning: Workflow file {workflow_file} not found, skipping")
            continue
        for mode in modes_to_run:
            for seed in seeds_list:
                output_file = os.path.join("../Results/RQ1-raw-data/WS", f"{dataset}_{seed}_{mode}.csv")
                tasks.append((workflow_file, mode, output_file, seed))

    print(f"Starting processing {len(tasks)} tasks (use_parallel={use_parallel}, cpu_cores={cpu_cores}, modes={modes_to_run}, seeds={seeds_list})")
    start_time = time.time()

    completed_tasks = []
    if use_parallel:
        print(f"Using parallel processing with {cpu_cores} processes")
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(process_dataset_mode, task) for task in tasks]
            for future in futures:
                try:
                    result = future.result()
                    completed_tasks.append(result)
                    print(f"Completed: {result[0]} | Mode: {result[1]} | Seed: {result[3]} | Time: {result[2]:.2f}s")
                except Exception as e:
                    print(f"Task failed: {str(e)}")
    else:
        print("Running in sequential mode")
        for task in tasks:
            try:
                result = process_dataset_mode(task)
                completed_tasks.append(result)
                print(f"Completed: {result[0]} | Mode: {result[1]} | Seed: {result[3]} | Time: {result[2]:.2f}s")
            except Exception as e:
                print(f"Task failed: {str(e)}")

    total_time = time.time() - start_time
    print(f"All tasks finished in {total_time:.2f} seconds. Total completed: {len(completed_tasks)}")

    return completed_tasks


if __name__ == '__main__':
    main()