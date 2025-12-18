import csv
import os
import time
import uuid

import numpy as np
import random
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution, Solution
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.util.comparator import DominanceComparator, DominanceWithConstraintsComparator, Comparator
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo/')
sys.path.insert(0, '/mnt/mydrive/ccj/code/mmo/')
import networkx as nx
from typing import List, Dict, Set
from Code.Utils.remove_duplicates import crowding_distance_assignment
from Code.SPSP.Utils.Construct_secondary_objective import generate_fa, update_novelty_archive


class SPSProblem(Problem):
    def __init__(self, instance_file: str, random_seed):
        super().__init__()
        self.instance_file = instance_file
        self.task_efforts: Dict[int, float] = {}
        self.employee_max_dedication = 1.0
        self.parse_instance_file()
        self.ver = self.num_tasks
        self.minded = 1.0 / (self.ver - 1)
        self.random_seed = random_seed
        self._number_of_variables = self.num_employees * self.num_tasks
        self._number_of_objectives = 2
        self._name = "SPSProblem"
        self._number_of_constraints = 1
        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = self._init_upper_bound()
        self.evaluation_count = 0
        self.mode = 'ft_fa'
        self.novelty_archive = []
        self.t_max = 1000
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]

    def _init_upper_bound(self):
        upper_bound = []
        emp_ids = sorted(self.employee_skills.keys())
        max_multiplier = int(self.employee_max_dedication / self.minded)
        upper_val = max_multiplier * self.minded
        for _ in emp_ids:
            upper_bound.extend([upper_val] * self.num_tasks)
        return upper_bound

    @property
    def name(self):
        return self._name

    @property
    def number_of_variables(self):
        return self._number_of_variables

    @property
    def number_of_objectives(self):
        return self._number_of_objectives

    @property
    def number_of_constraints(self):
        return self._number_of_constraints

    def parse_instance_file(self):
        self.employee_skills: Dict[int, List[int]] = {}
        self.employee_salaries: Dict[int, float] = {}
        self.task_skills: Dict[int, List[int]] = {}
        self.task_efforts: Dict[int, float] = {}
        self.dag = nx.DiGraph()
        all_task_ids = set()
        all_employee_ids = set()

        with open(self.instance_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=')
                        if key == 'employee.number':
                            self.num_employees = int(value)
                            continue
                        elif key == 'task.number':
                            self.num_tasks = int(value)
                            continue
                        elif key == 'skill.number':
                            self.skill_number = int(value)
                            continue
                        elif key == 'graph.arc.number':
                            continue
                        if key.startswith('task.'):
                            parts = key.split('.')
                            task_id = int(parts[1])
                            all_task_ids.add(task_id)
                        elif key.startswith('employee.'):
                            parts = key.split('.')
                            emp_id = int(parts[1])
                            all_employee_ids.add(emp_id)

        if not hasattr(self, 'num_employees') or not hasattr(self, 'num_tasks'):
            filename = os.path.basename(self.instance_file)
            import re
            match = re.match(r'inst-(\d+)-(\d+)-skill-.*\.conf', filename)
            if match:
                self.num_tasks = int(match.group(1))
                self.num_employees = int(match.group(2))
            else:
                self.num_employees = len(all_employee_ids) if all_employee_ids else 1
                self.num_tasks = len(all_task_ids) if all_task_ids else 1

        if self.num_employees == 0 or self.num_tasks == 0:
            raise ValueError(f"Invalid instance file {self.instance_file}: number of employees or tasks is zero")

        self._number_of_variables = self.num_employees * self.num_tasks

        with open(self.instance_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=')
                        if key in ['employee.number', 'task.number', 'skill.number', 'graph.arc.number']:
                            continue
                        if key.startswith('employee.') and 'skill.' in key:
                            parts = key.split('.')
                            if len(parts) >= 4 and parts[3].isdigit():
                                emp_id = int(parts[1])
                                skill_id = int(value)
                                if emp_id not in self.employee_skills:
                                    self.employee_skills[emp_id] = []
                                self.employee_skills[emp_id].append(skill_id)
                        elif key.startswith('employee.') and 'salary' in key:
                            emp_id = int(key.split('.')[1])
                            self.employee_salaries[emp_id] = float(value)
                        elif key.startswith('task.') and 'skill.' in key:
                            parts = key.split('.')
                            if len(parts) >= 4 and parts[3].isdigit():
                                task_id = int(parts[1])
                                skill_id = int(value)
                                if task_id not in self.task_skills:
                                    self.task_skills[task_id] = []
                                self.task_skills[task_id].append(skill_id)
                        elif key.startswith('task.') and 'cost' in key:
                            task_id = int(key.split('.')[1])
                            self.task_efforts[task_id] = float(value)
                        elif key.startswith('graph.arc.'):
                            try:
                                from_task, to_task = map(int, value.split())
                                self.dag.add_edge(from_task, to_task)
                            except ValueError:
                                continue

        for task_id in all_task_ids:
            if task_id not in self.task_skills:
                raise ValueError(f"Instance file {self.instance_file} error: Task {task_id} has no skill requirements")
            if task_id not in self.task_efforts:
                raise ValueError(f"Instance file {self.instance_file} error: Task {task_id} has no effort (cost) value")

        for emp_id in all_employee_ids:
            if emp_id not in self.employee_skills:
                raise ValueError(f"Instance file {self.instance_file} error: Employee {emp_id} has no skills")
            if emp_id not in self.employee_salaries:
                raise ValueError(f"Instance file {self.instance_file} error: Employee {emp_id} has no salary value")

        if not hasattr(self, 'skill_number'):
            raise ValueError(f"Instance file {self.instance_file} error: Missing 'skill.number' configuration")

        all_tasks_in_dag = set(self.dag.nodes())
        all_tasks_in_file = set(self.task_skills.keys())
        isolated_tasks = all_tasks_in_file - all_tasks_in_dag

        for task in isolated_tasks:
            self.dag.add_node(task)
            print(f"Warning: Task {task} is isolated and has been added to the DAG")

        if len(list(nx.topological_sort(self.dag))) != self.num_tasks:
            raise ValueError(f"Topological sort does not include all tasks! Instance file: {self.instance_file}")

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        dedication = np.array(solution.variables).reshape((self.num_employees, self.num_tasks))
        multipliers = np.round(dedication / self.minded).astype(int)
        max_multiplier = int(self.employee_max_dedication / self.minded)
        for ei in range(self.num_employees):
            for tj in range(self.num_tasks):
                multipliers[ei, tj] = max(0, min(multipliers[ei, tj], max_multiplier))
        dedication = multipliers * self.minded
        solution.variables = dedication.flatten().tolist()

        emp_ids = sorted(self.employee_skills.keys())
        task_ids = sorted(self.task_skills.keys())

        task_durations = {}
        task_ahr = {}
        for task_id in task_ids:
            total_ahr = sum(dedication[emp_ids.index(emp_id)][task_ids.index(task_id)]
                            for emp_id in emp_ids)
            effort = self.task_efforts.get(task_id, 1.0)
            task_durations[task_id] = effort / total_ahr if total_ahr > 1e-6 else float('inf')
            task_ahr[task_id] = total_ahr

        task_start = {t: 0.0 for t in task_ids}
        task_end = {t: 0.0 for t in task_ids}
        for task_id in nx.topological_sort(self.dag):
            pred_end_times = [task_end[p] for p in self.dag.predecessors(task_id)]
            task_start[task_id] = max(pred_end_times) if pred_end_times else 0.0
            task_end[task_id] = task_start[task_id] + task_durations[task_id]
        project_duration = max(task_end.values())

        project_cost = sum(
            self.employee_salaries[emp_id] *
            dedication[emp_ids.index(emp_id)][task_ids.index(task_id)] *
            task_durations[task_id]
            for emp_id in emp_ids
            for task_id in task_ids
            if dedication[emp_ids.index(emp_id)][task_ids.index(task_id)] > 1e-6
        )

        constraints = []
        unassigned_tasks = sum(1 for ahr in task_ahr.values() if ahr <= 1e-6)
        constraint1_violation = unassigned_tasks
        constraints.append(constraint1_violation)

        total_missing_skills = 0
        for task_id in task_ids:
            required_skills = set(self.task_skills[task_id])
            assigned_skills = set()
            for emp_id in emp_ids:
                if dedication[emp_ids.index(emp_id)][task_ids.index(task_id)] > 1e-6:
                    assigned_skills.update(self.employee_skills[emp_id])
            missing_skills = required_skills - assigned_skills
            total_missing_skills += len(missing_skills)
        constraint2_violation = total_missing_skills
        constraints.append(constraint2_violation)

        total_overload = 0
        for emp_id in emp_ids:
            emp_idx = emp_ids.index(emp_id)
            max_ded = self.employee_max_dedication
            overload = self._calculate_overload(
                dedication[emp_idx],
                max_ded,
                task_durations,
                task_start,
                task_end,
                task_ids
            )
            total_overload += overload
        constraint3_violation = total_overload
        constraints.append(constraint3_violation)

        solution.attributes['original_duration'] = project_duration
        solution.attributes['original_cost'] = project_cost
        solution.objectives = [project_duration, project_cost]
        solution.constraints = constraints
        self.evaluation_count += 1

        return solution

    def normalize_population(self, population: List[Solution], t: int = 1, age_info=None) -> None:
        valid_solutions = [s for s in population if 'original_duration' in s.attributes]
        if not valid_solutions:
            return

        ft = [s.attributes['original_duration'] for s in valid_solutions]
        fa = [s.attributes['original_cost'] for s in valid_solutions]

        if self.mode in ['ft_fa', 'g1_g2']:
            for sol in valid_solutions:
                if self.mode == 'ft_fa':
                    sol.objectives = [sol.attributes['original_duration'],
                                      sol.attributes['original_cost']]
                else:
                    ft_min, ft_max = min(ft), max(ft)
                    fa_min, fa_max = min(fa), max(fa)
                    norm_ft = (sol.attributes['original_duration'] - ft_min) / (
                            ft_max - ft_min) if ft_max != ft_min else 0.5
                    norm_fa = (sol.attributes['original_cost'] - fa_min) / (
                            fa_max - fa_min) if fa_max != fa_min else 0.5
                    sol.objectives = [norm_ft + norm_fa, norm_ft - norm_fa]
        elif self.mode in ['penalty_fa', 'gaussian_fa', 'reciprocal_fa',
                           'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_solutions]
            unique_elements = [sorted({sol.variables[i] for sol in valid_solutions})
                               for i in range(len(valid_solutions[0].variables))]

            current_age_info = None
            if self.mode == 'age_maximization_fa' and age_info:
                if len(age_info) == len(population):
                    current_age_info = [age_info[i] for i, s in enumerate(population) if s in valid_solutions]

            novelty_archive = self.novelty_archive if self.mode == 'novelty_maximization_fa' else None

            mode_prefix = self.mode.split('_')[0]
            ft, fa = generate_fa(
                configurations=configurations,
                ft=ft,
                fa_construction=mode_prefix,
                minimize=True,
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

    def create_solution(self) -> FloatSolution:
        solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=1
        )

        variables = [0.0] * self.number_of_variables

        emp_ids = sorted(self.employee_skills.keys())
        task_ids = sorted(self.task_skills.keys())
        num_tasks = len(task_ids)
        max_total_multiplier = int(self.employee_max_dedication / self.minded)

        for emp_idx, emp_id in enumerate(emp_ids):
            multipliers = [random.randint(0, max_total_multiplier) for _ in range(num_tasks)]
            total_multiplier = sum(multipliers)

            if total_multiplier > max_total_multiplier:
                if total_multiplier == 0:
                    scaled_multipliers = [0] * num_tasks
                else:
                    scaling_factor = max_total_multiplier / total_multiplier
                    scaled_multipliers = [int(round(m * scaling_factor)) for m in multipliers]
                    while sum(scaled_multipliers) > max_total_multiplier:
                        non_zero_indices = [i for i, m in enumerate(scaled_multipliers) if m > 0]
                        if not non_zero_indices:
                            break
                        reduce_idx = random.choice(non_zero_indices)
                        scaled_multipliers[reduce_idx] -= 1
                multipliers = scaled_multipliers

            dedications = [m * self.minded for m in multipliers]

            for task_idx in range(num_tasks):
                variables[emp_idx * num_tasks + task_idx] = dedications[task_idx]

        solution.variables = variables
        return solution

    def _calculate_overload(self, emp_dedication, max_dedication, task_durations, task_start, task_end, task_ids):
        time_points = set()
        for task_id in task_ids:
            time_points.add(task_start[task_id])
            time_points.add(task_end[task_id])
        sorted_time_points = sorted(time_points)

        total_overload = 0.0
        for i in range(len(sorted_time_points) - 1):
            tau_start = sorted_time_points[i]
            tau_end = sorted_time_points[i + 1]
            interval_duration = tau_end - tau_start

            current_dedication = 0.0
            for task_idx, task_id in enumerate(task_ids):
                if task_start[task_id] <= tau_start < task_end[task_id]:
                    current_dedication += emp_dedication[task_idx]

            overload_start = max(0.0, current_dedication - max_dedication)

            next_dedication = 0.0
            for task_idx, task_id in enumerate(task_ids):
                if task_start[task_id] <= tau_end < task_end[task_id]:
                    next_dedication += emp_dedication[task_idx]
            overload_end = max(0.0, next_dedication - max_dedication)

            total_overload += 0.5 * (overload_start + overload_end) * interval_duration

        return total_overload

    def is_feasible(self, solution: FloatSolution) -> bool:
        return all(c <= 1e-6 for c in solution.constraints)


class LiteratureConstraintComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        v1_1, v2_1, v3_1 = solution1.constraints
        v1_2, v2_2, v3_2 = solution2.constraints

        total_violation1 = v1_1 + v2_1 + v3_1
        total_violation2 = v1_2 + v2_2 + v3_2

        feasible1 = total_violation1 <= 1e-6
        feasible2 = total_violation2 <= 1e-6

        if feasible1 and not feasible2:
            return -1
        elif feasible2 and not feasible1:
            return 1
        elif not feasible1 and not feasible2:
            return -1 if total_violation1 < total_violation2 else (1 if total_violation1 > total_violation2 else 0)
        else:
            return 0

def partial_duplicate_replacement(combined_population, population_size):
    fronts = fast_non_dominated_sort(combined_population)

    processed_fronts = []
    for i in range(len(fronts)):
        current_front = fronts[i]

        unique_sols = []
        duplicate_sols = []
        seen_vars = set()

        for sol in current_front:
            var_tuple = tuple(sol.variables)
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
        constraint_comparator=LiteratureConstraintComparator()
    )
    ranking = FastNonDominatedRanking(comparator=custom_comparator)
    return ranking.compute_ranking(population)

def run_nsgaii(instance_file: str, mode: str = 'ft_fa',
               random_seed: int = 42, output_dir: str = "./Results/NSGA2"):
    os.makedirs(output_dir, exist_ok=True)
    max_evaluations = 100000
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    output_file = os.path.join(output_dir, f"{instance_name}_{random_seed}_{mode}.csv")

    problem = SPSProblem(instance_file, random_seed)
    problem.mode = mode
    problem.random_seed = random_seed

    random.seed(random_seed)
    np.random.seed(random_seed)
    population_size = 100

    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=100,

        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=0.9, distribution_index=20),
        selection=BinaryTournamentSelection(
            comparator=DominanceWithConstraintsComparator(
                constraint_comparator=LiteratureConstraintComparator()
            )
        ),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceWithConstraintsComparator(
            constraint_comparator=LiteratureConstraintComparator()
        )
    )

    evaluations_count = 0
    global_best_duration = float('inf')
    global_best_cost = float('inf')
    best_variables = None
    best_generation = 0
    p_values_history = []
    start_time = time.time()
    MAX_RUNTIME = 24 * 3600

    age_info = None
    if mode == 'age_maximization_fa':
        age_info = list(range(1, population_size + 1))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([f"NSGA-II Run for New Dataset: {instance_name} (Seed: {random_seed}, Mode: {mode})"])
        writer.writerow([f"Settings: All Employees Max Dedication=1.0, ver={problem.ver}, minded={problem.minded:.6f}"])
        writer.writerow([f"Budget: {max_evaluations}, Time limit: {MAX_RUNTIME / 3600:.1f} hours"])
        writer.writerow([])

    population = algorithm.create_initial_solutions()
    for solution in population:
        solution.uuid = uuid.uuid4()

    evaluated_population = []
    for solution in population:
        problem.evaluate(solution)
        evaluations_count += 1
        evaluated_population.append(solution)
        if problem.is_feasible(solution):
            current_duration = solution.attributes['original_duration']
            current_cost = solution.attributes['original_cost']
            if current_duration < global_best_duration:
                global_best_duration = current_duration
                global_best_cost = current_cost
                best_variables = solution.variables.copy()
                best_generation = 1
    population = evaluated_population
    if mode == 'novelty_maximization_fa':
        update_novelty_archive(population, problem.novelty_archive)

    problem.normalize_population(population, t=1, age_info=age_info)

    generation = 1
    while evaluations_count < max_evaluations:
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
            if evaluations_count >= max_evaluations:
                break
            problem.evaluate(solution)
            evaluations_count += 1
            evaluated_offspring.append(solution)

        combined_population = population + evaluated_offspring

        combined_age_info = None
        if mode == 'age_maximization_fa':
            offspring_age = [population_size + generation] * len(evaluated_offspring)
            combined_age_info = age_info + offspring_age

        problem.normalize_population(combined_population, t=generation, age_info=combined_age_info)

        if mode == 'novelty_maximization_fa':
            update_novelty_archive(evaluated_offspring, problem.novelty_archive)

        if mode == 'ft_fa':
            population = algorithm.replacement(population, evaluated_offspring)
        else:
            population = partial_duplicate_replacement(combined_population, population_size)

        if mode == 'age_maximization_fa':
            uuid_to_index = {sol.uuid: idx for idx, sol in enumerate(combined_population)}
            selected_indices = [uuid_to_index[sol.uuid] for sol in population]
            age_info = [combined_age_info[idx] for idx in selected_indices]

        feasible_solutions = [sol for sol in population if problem.is_feasible(sol)]
        if feasible_solutions:
            current_best = min(feasible_solutions,
                               key=lambda x: (x.attributes['original_duration']))
            current_duration = current_best.attributes['original_duration']
            current_cost = current_best.attributes['original_cost']
            if current_duration < global_best_duration:
                global_best_duration = current_duration
                global_best_cost = current_cost
                best_variables = current_best.variables.copy()
                best_generation = generation

        unique_solutions = []
        seen = set()
        for sol in combined_population:
            key = tuple(sol.variables)
            if key not in seen:
                seen.add(key)
                unique_solutions.append(sol)
        p_value = len(get_non_dominated_solutions(unique_solutions)) / len(unique_solutions) if unique_solutions else 0
        p_values_history.append(p_value)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([f"Generation {generation} p-value: {p_value:.4f} best value: {global_best_duration}"])

        print(f"New Dataset: {instance_name} {mode} seed {random_seed} | Gen {generation} | "
              f"Evaluations: {evaluations_count}/{max_evaluations} | "
              f"Best: {global_best_duration if global_best_duration != float('inf') else 'inf'} | "
              f"Time: {(time.time() - start_time) / 60:.1f} min")

        generation += 1

    runtime = time.time() - start_time
    p_values_until_best = p_values_history[:best_generation] if best_generation > 0 else []

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([])
        writer.writerow([f"budget_used:{evaluations_count}"])
        writer.writerow([f"Running time: {runtime:.2f} seconds"])
        writer.writerow(
            [f"Fixed Settings: All Employees Max Dedication=1.0, ver={problem.ver}, minded={problem.minded:.6f}"])
        writer.writerow([])

        if global_best_duration == float('inf'):
            writer.writerow(["Best Solution: 'ft': inf, 'fa': inf (No feasible solution found)"])
        else:
            best_p = p_values_history[best_generation - 1] if (
                        best_generation > 0 and best_generation - 1 < len(p_values_history)) else 0.0
            writer.writerow([
                f"Best Solution: 'ft': {global_best_duration:.6f}, 'fa': {global_best_cost:.6f} "
                f"appeared in Generation {best_generation}, p: {best_p:.4f}"
            ])

        if p_values_until_best:
            p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
            writer.writerow([f"p values until best solution: {p_values_str}"])

import argparse
from concurrent.futures import ProcessPoolExecutor

def _parse_seeds_arg(seeds_arg):
    """
    Parse seeds argument string.
    Acceptable formats:
    - None -> default 0-9
    - '5' -> [5]
    - '0,1,2' -> [0,1,2]
    - '0-9' -> [0..9]
    """
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
    Argument-driven entry for running multiple SPS instances.

    Parameters (argparse):
    --use-parallel / --no-parallel  : enable/disable parallel execution (default: enabled)
    --cpu-cores N                   : number of worker processes (default: 50 to match NAS scripts)
    --mode MODE                      : single mode or 'all' (default: 'all')
    --seeds SEEDS                    : seeds specification: single '5', csv '0,1,2' or range '0-9' (default: 0-9)

    Only parameter-passing interface is changed; algorithm implementation remains unmodified.
    """
    parser = argparse.ArgumentParser(description="Run multiple SPS NSGA-II instances with argument configuration")

    # Defaults: align names with NAS-style defaults where reasonable
    default_cpu_cores = 50
    modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
             'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser.set_defaults(use_parallel=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-parallel', dest='use_parallel', action='store_true',
                       help='Enable parallel execution (default)')
    group.add_argument('--no-parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel execution (run sequentially)')

    parser.add_argument('--cpu-cores', type=int, default=default_cpu_cores,
                        help=f'Number of worker processes to use when parallel execution is enabled (default: {default_cpu_cores})')

    parser.add_argument('--mode', type=str, default='all', choices=modes + ['all'],
                        help="Single mode to run, or 'all' to run all modes (default: 'all')")

    parser.add_argument('--seeds', type=str, default=None,
                        help="Seeds specification: single '5', csv '0,1,2' or range '0-9'. Default: 0-9.")

    parsed = parser.parse_args(argv)

    try:
        seeds_list = _parse_seeds_arg(parsed.seeds)
    except Exception as e:
        raise ValueError(f"Failed to parse --seeds argument '{parsed.seeds}': {e}")

    # Determine which modes to run
    if parsed.mode == 'all':
        modes_to_run = modes
    else:
        modes_to_run = [parsed.mode]

    use_parallel = parsed.use_parallel
    cpu_cores = max(50, parsed.cpu_cores)

    instance_base_path = "./Datasets/"
    output_dir = "../Results/RQ1-raw-data/SPSP"

    instance_identifiers = [
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

    seeds_iter = seeds_list

    tasks = []
    for identifier in instance_identifiers:
        instance_file = f"{instance_base_path}inst-{identifier}.conf"
        if os.path.exists(instance_file):
            for mode in modes_to_run:
                for seed in seeds_iter:
                    tasks.append((instance_file, mode, seed))
        else:
            print(f"Warning: New dataset instance file {instance_file} not found, skipping")

    print(f"Prepared {len(tasks)} tasks (modes={modes_to_run}, seeds={seeds_iter}, use_parallel={use_parallel}, cpu_cores={cpu_cores})")

    if use_parallel:
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = []
            for task in tasks:
                instance_file, mode, seed = task
                futures.append(executor.submit(
                    run_nsgaii,
                    instance_file=instance_file,
                    mode=mode,
                    random_seed=seed,
                    output_dir=output_dir
                ))

            for future, task in zip(futures, tasks):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in task (Instance: {task[0]}, Seed: {task[2]}): {str(e)}")
    else:
        for task in tasks:
            instance_file, mode, seed = task
            try:
                run_nsgaii(
                    instance_file=instance_file,
                    mode=mode,
                    random_seed=seed,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"Error in task (Instance: {instance_file}, Seed: {seed}): {str(e)}")


if __name__ == "__main__":
    main()