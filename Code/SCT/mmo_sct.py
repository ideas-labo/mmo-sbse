import csv
import random
import uuid
import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
import sys
import os
import multiprocessing
import time
sys.path.append('../')
sys.path.append('../..')
from Code.SCT.Utils.Construct_secondary_objective import update_novelty_archive
from Code.SCT.Utils.Data_utils import get_data
from Code.Utils.remove_duplicates import partial_duplicate_replacement
from Utils.Nsga2_operator_tuning import CustomProblem, UniformCrossover, BoundaryMutation, selection
from Utils.get_objectives import update_objectives, GotoFailedLabelException
from sklearn.preprocessing import MinMaxScaler


MAXIMIZATION_DATASETS_FT = ['storm_wc', 'storm_rs', 'storm_sol', 'dnn_dsr', 'dnn_coffee', 'dnn_adiac', 'dnn_sa',
                            'trimesh', 'dnn_coffee', 'dnn_dsr','x264']

MAXIMIZATION_DATASETS_FA = []

POPULATION_SIZE_MAP = {
    'dnn_adiac': 20, 'dnn_coffee': 50, 'dnn_dsr': 60, 'dnn_sa': 20,
    'llvm': 20, 'lrzip': 20, 'mariadb': 20, 'mongodb': 20, 'storm_rs': 50, 'storm_sol': 50,
    'storm_wc': 50, 'trimesh': 20, 'vp9': 30, 'x264': 50
}

BUDGET_MAP = {
    'dnn_adiac': 400, 'dnn_coffee': 900, 'dnn_dsr': 800, 'dnn_sa': 400,
    'llvm': 600, 'lrzip': 400, 'mariadb': 400, 'mongodb': 500, 'storm_rs': 700, 'storm_sol': 600,
    'storm_wc': 400, 'trimesh': 1000, 'vp9': 700, 'x264': 2500
}


def nsga2(filename, mode='ft_fa', seed=1, max_generations=400, population_size=20, budget=700, minimize=True, t=1,
          t_max=None, reverse=False, start_time=None):
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    population_size = POPULATION_SIZE_MAP.get(dataset_name, population_size)

    if t_max is None:
        t_max = int(budget / population_size) * 2

    if isinstance(budget, list):
        if len(budget) == 1:
            budget = budget[0]
        else:
            raise ValueError("budget should be an integer or a list with a single integer element.")

    global_pareto_front = []
    random.seed(seed)
    np.random.seed(seed)
    file = get_data(filename, initial_size=population_size, seed=seed, reverse=reverse)
    independent_set = file.independent_set
    dict_search = file.dict_search

    dataset_name = os.path.splitext(os.path.basename(filename))[0]

    is_maximization_dataset_ft = any(dataset in dataset_name for dataset in MAXIMIZATION_DATASETS_FT)
    is_maximization_dataset_fa = any(dataset in dataset_name for dataset in MAXIMIZATION_DATASETS_FA)

    all_ft = [v[0] for v in dict_search.values()]
    all_fa = [v[1] for v in dict_search.values()]

    scaler_ft = MinMaxScaler()
    scaler_fa = MinMaxScaler()
    scaler_ft.fit([[ft] for ft in all_ft])
    scaler_fa.fit([[fa] for fa in all_fa])

    budget1 = budget
    budget_configuration = set()

    unique_elements_per_column = [np.unique([ind.decision[i] for ind in file.training_set]) for i in
                                  range(len(independent_set))]

    problem = CustomProblem( seed,filename, mode, dict_search, independent_set, scaler_ft, scaler_fa, minimize,
                            unique_elements_per_column)
    crossover = UniformCrossover(probability=0.9)

    mutation = BoundaryMutation(probability=0.1, independent_set=independent_set)

    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        selection=selection,
        termination_criterion=StoppingByEvaluations(max_evaluations=budget)
    )

    population = algorithm.create_initial_solutions()
    for sol in population:
        sol.id = str(uuid.uuid4())

    age_info = list(range(1, population_size + 1))

    novelty_archive = []

    evaluated_population = []
    for solution in population:
        try:
            evaluated_solution = problem.evaluate(solution, global_pareto_front, budget1)
            evaluated_population.append(evaluated_solution)

            budget_configuration.add(tuple(evaluated_solution.variables))
            if len(budget_configuration) >= budget:
                print(f"Budget limit reached during initial evaluation: {len(budget_configuration)} >= {budget}")
                raise GotoFailedLabelException
        except GotoFailedLabelException:
            raise

    population = evaluated_population

    if mode == 'novelty_maximization_fa':
        update_novelty_archive(population, novelty_archive)

    population = update_objectives(seed,
                                   population, mode, scaler_ft, scaler_fa, minimize, filename,
                                   unique_elements_per_column, dict_search, t, t_max,
                                   age_info=age_info if mode == 'age_maximization_fa' else None,
                                   novelty_archive=novelty_archive if mode == 'novelty_maximization_fa' else None
                                   )

    current_generation = 1
    generation_info = []
    global_best_ft = float('inf')
    global_best_fa = float('inf')
    global_best_decision = None
    global_best_generation = -1
    global_best_p = 0.0

    suffix = "_reverse" if reverse else ""
    output_filename = f'../Results/RQ1-raw-data/SCT/{dataset_name}-{seed}_{mode}{suffix}.csv'

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w', newline='') as f:
        csv_writer = csv.writer(f, lineterminator='\n')
        csv_writer.writerow([f"NSGA-II Run for {dataset_name} with seed {seed} and mode {mode} (reverse={reverse})"])
        csv_writer.writerow([f"Budget: {budget}"])
        csv_writer.writerow([f"Population size: {population_size}"])
        csv_writer.writerow([f"Calculated t_max: {t_max}"])
        csv_writer.writerow([])

    try:
        while True:
            budget_used = len(budget_configuration)
            if start_time is not None and (time.time() - start_time) > 24 * 3600:
                print("Time limit exceeded (48 hours). Terminating.")
                break

            if len(budget_configuration) >= budget:
                print(f"Budget limit reached: {len(budget_configuration)} >= {budget}")
                break

            mating_population = algorithm.selection(population)
            offspring_population = algorithm.reproduction(mating_population)

            for child in offspring_population:
                child.id = str(uuid.uuid4())

            evaluated_offspring = []
            for child in offspring_population:
                if len(budget_configuration) >= budget:
                    break
                try:
                    evaluated_child = problem.evaluate(child, global_pareto_front, budget1)
                    budget_configuration.add(tuple(evaluated_child.variables))

                    if evaluated_child is None:
                        break
                    evaluated_offspring.append(evaluated_child)
                except GotoFailedLabelException:
                    raise
            if len(budget_configuration) >= budget:
                break

            if mode == 'age_maximization_fa':
                new_age = population_size + current_generation
                offspring_age_info = [new_age] * population_size
                combined_age_info = age_info + offspring_age_info
            else:
                combined_age_info = None

            combined_population = population + evaluated_offspring

            id_to_index = {sol.id: i for i, sol in enumerate(combined_population)}

            if mode == 'novelty_maximization_fa':
                update_novelty_archive(evaluated_offspring, novelty_archive)

            combined_population = update_objectives(seed,
                                                    combined_population,
                                                    mode,
                                                    scaler_ft,
                                                    scaler_fa,
                                                    minimize,
                                                    filename,
                                                    unique_elements_per_column,
                                                    dict_search,
                                                    current_generation,
                                                    t_max,
                                                    age_info=combined_age_info if mode == 'age_maximization_fa' else None,
                                                    novelty_archive=novelty_archive if mode == 'novelty_maximization_fa' else None,
                                                    )

            unique_combined_population = []
            unique_variables = set()
            for sol in combined_population:
                var_tuple = tuple(sol.variables)
                if var_tuple not in unique_variables:
                    unique_combined_population.append(sol)
                    unique_variables.add(var_tuple)

            non_dominated_solutions = get_non_dominated_solutions(unique_combined_population)
            p = len(non_dominated_solutions) / len(unique_combined_population) if len(
                unique_combined_population) > 0 else 0


            if mode=='ft_fa':
                population = algorithm.replacement(population, evaluated_offspring)
            else:
                population = partial_duplicate_replacement(combined_population, population_size)

            selected_indices = [id_to_index[sol.id] for sol in population]

            if mode == 'age_maximization_fa':
                age_info = [combined_age_info[i] for i in selected_indices]

            generation_original_objectives = [individual.original_objectives for individual in population]

            current_best_ft = min([obj[0] for obj in generation_original_objectives])
            if current_best_ft < global_best_ft:
                global_best_ft = current_best_ft
                global_best_fa = [obj[1] for obj in generation_original_objectives if obj[0] == current_best_ft][0]
                global_best_generation = current_generation
                global_best_p = p
            best_ft_gen = global_best_ft
            best_fa_gen = global_best_fa
            if is_maximization_dataset_ft:
                if reverse:
                    best_fa_gen = -best_fa_gen
                else:
                    best_ft_gen = -best_ft_gen
            if is_maximization_dataset_fa:
                if reverse:
                    best_ft_gen = -best_ft_gen
                else:
                    best_fa_gen = -best_fa_gen
            with open(output_filename, 'a', newline='') as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                csv_writer.writerow(
                    [f"Generation {current_generation} p-value: {p:.4f} best value: {best_ft_gen:.6f}"])
            current_generation += 1

    except GotoFailedLabelException:
        print("Reached budget limit. Returning partial Results.")

    running_time = time.time() - start_time if start_time is not None else 0

    best_ft = global_best_ft
    best_fa = global_best_fa
    if is_maximization_dataset_ft:
        if reverse:
            best_fa = -best_fa
        else:
            best_ft = -best_ft
    if is_maximization_dataset_fa:
        if reverse:
            best_ft = -best_ft
        else:
            best_fa = -best_fa

    p_values_until_best = []
    if global_best_generation != -1:
        with open(output_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("Generation ") and "p-value:" in line_stripped:
                    gen_part = line_stripped.split("Generation ")[1].split()[0]
                    try:
                        gen_num = int(gen_part)
                    except ValueError:
                        continue

                    p_value_part = line_stripped.split("p-value:")[1].strip()
                    p_val_str = p_value_part.split()[0]

                    try:
                        p_val = float(p_val_str)
                    except ValueError:
                        print(f"Warning: Failed to parse p-value from line: {line_stripped}")
                        continue

                    if gen_num <= global_best_generation:
                        p_values_until_best.append(p_val)

    with open(output_filename, 'a', newline='') as f:
        csv_writer = csv.writer(f, lineterminator='\n')
        csv_writer.writerow([])
        csv_writer.writerow([f"budget_used:{len(budget_configuration)}"])
        csv_writer.writerow([f"Running time: {running_time:.2f} seconds"])
        csv_writer.writerow([])
        csv_writer.writerow([
            f"Best Solution: 'ft': {best_ft:.6f}, 'fa': {best_fa:.6f} "
            f"appeared in Generation {global_best_generation}, p: {global_best_p:.4f}, reverse: {reverse}"
        ])

        if p_values_until_best:
            p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best])
            csv_writer.writerow([f"p values until best solution: {p_values_str}"])

    return population, len(budget_configuration), {
        'ft': best_ft,
        'fa': best_fa,
        'generation': global_best_generation,
        'p': global_best_p,
        'reverse': reverse,
        'budget_used': len(budget_configuration)
    }


def run_nsga2(seed, name, budget, mode='ft_fa', reverse=False):
    start_time = time.time()

    is_maximization_dataset_ft = any(dataset in name for dataset in MAXIMIZATION_DATASETS_FT)
    is_maximization_dataset_fa = any(dataset in name for dataset in MAXIMIZATION_DATASETS_FA)

    file = get_data(f"./Datasets/{name}.csv", initial_size=30, seed=seed, reverse=reverse)
    dict_search = file.dict_search
    all_ft = [v[0] for v in dict_search.values()]

    if mode == 'reciprocal_fa' and 0 in all_ft:
        print(f"Skipping reciprocal_fa mode for {name} because there are zero ft values.")
        return

    t = 1

    population, budget_used, best_info = nsga2(
        filename=f"./Datasets/{name}.csv",
        mode=mode,
        seed=seed,
        budget=budget,
        minimize=True,
        t=t,
        t_max=None,
        reverse=reverse,
        start_time=start_time
    )

    print(f"Completed run for {name} with seed {seed} and mode {mode} (reverse={reverse})")


def run_tasks(tasks, use_multiprocessing=True, max_processes=200):
    if use_multiprocessing:
        with multiprocessing.Pool(processes=max_processes) as pool:
            results = []
            try:
                async_results = [pool.apply_async(run_nsga2, task) for task in tasks]
                for async_result in async_results:
                    try:
                        result = async_result.get()
                        results.append(result)
                    except Exception as e:
                        print(f"An error occurred while processing a task: {e}")
            except KeyboardInterrupt:
                print("Interrupted by user. Terminating processes...")
                pool.terminate()
                pool.join()
            else:
                pool.close()
                pool.join()
    else:
        for task in tasks:
            try:
                run_nsga2(*task)
            except Exception as e:
                print(f"An error occurred while processing a task: {e}")

    print("All tasks completed.")

import argparse

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
    Argument-driven entry for running the NSGA-II SCT experiments.

    Parameter names are kept consistent with the NAS script:
    --use-parallel / --no-parallel
    --cpu-cores
    --mode
    --seeds

    This function only changes how parameters are passed (argparse). No algorithm logic is modified.
    """
    # Defaults aligned with previous NAS script
    default_cpu_cores = 50
    default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                     'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser = argparse.ArgumentParser(description="Run NSGA2 SCT experiments with argument configuration")
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

    # Determine modes to run
    if parsed.mode == 'all':
        modes_to_run = default_modes
    else:
        modes_to_run = [parsed.mode]

    use_parallel = parsed.use_parallel
    cpu_cores = max(1, parsed.cpu_cores)

    # Keep the same dataset names as in the original script
    names = ['dnn_adiac', 'dnn_coffee', 'dnn_dsr', 'dnn_sa',
             'llvm', 'lrzip', 'mariadb', 'mongodb', 'vp9', 'x264',
             'storm_rs', 'storm_wc', 'trimesh']

    # Build tasks: each task is (seed, name, dataset_budget, selected_mode, reverse)
    tasks = []
    for selected_mode in modes_to_run:
        for name in names:
            dataset_budget = BUDGET_MAP.get(name, 200)
            for seed in seeds_list:
                # include both reverse False and True as in original script
                tasks.append((seed, name, dataset_budget, selected_mode, False))
                tasks.append((seed, name, dataset_budget, selected_mode, True))

    print(f"Starting processing {len(tasks)} tasks")
    print(f"use_parallel={use_parallel}, cpu_cores={cpu_cores}, modes={modes_to_run}, seeds={seeds_list}")

    # Execute tasks using existing run_tasks (unchanged)
    # map use_parallel to the run_tasks parameter name (use_multiprocessing)
    run_tasks(tasks, use_multiprocessing=use_parallel, max_processes=cpu_cores)


if __name__ == "__main__":
    main()


