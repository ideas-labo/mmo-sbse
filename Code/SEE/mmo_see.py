import glob
import os
import csv
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
import sys

from Code.Utils.remove_duplicates import partial_duplicate_replacement

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.append('../')
sys.path.append('../..')
import pandas as pd
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.problem import FloatProblem
from scipy.stats import t
import numpy as np
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import DominanceComparator
import traceback
import multiprocessing

from Utils.Construct_secondary_objective import generate_fa, update_novelty_archive
from Utils.Nsga2_operator_effort import SEEDoubleSimpleRandomMutation, SEESinglePointCrossover

BEST_VARIABLES_PATH = '../Results/RQ1-raw-data/SEE/BestVariables'
VALIDATION_RESULTS_PATH = '../Results/RQ1-raw-data/SEE/ValidationResults'

class DataRow:
    def __init__(self, id):
        self.id = int(id)
        self.values = []
        self.length = 0

    def add_value(self, value):
        self.values.append(float(value))
        self.length = len(self.values)

    def get_value(self, i):
        return self.values[i]

    def get_length(self):
        return self.length


class EffortEstimationFold:
    def __init__(self, name):
        self.name = name.lower().replace(" ", "")
        self.header = []
        self.rows = []
        self.efforts = {}
        self.number_of_rows = 0

    def add_row(self, row):
        self.rows.append(row)
        self.number_of_rows = len(self.rows)

    def add_effort(self, id, effort):
        self.efforts[id] = float(effort)

    def get_effort(self, id):
        if id not in self.efforts:
            raise ValueError(f"There is no such ID!")
        return self.efforts[id]

    def get_number_of_rows(self):
        return self.number_of_rows

    def get_rows(self):
        return self.rows


class Dataset:
    def __init__(self, name):
        self.name = name.lower()
        self.folds = []

    def add_fold(self, fold):
        self.folds.append(fold)

    def get_folds(self):
        return self.folds


class ExcelReader:
    @staticmethod
    def read(file_path):
        name = os.path.basename(file_path)
        name = name[:name.rfind("train") - 1] if "train" in name else name
        name = name[:name.rfind("test") - 1] if "test" in name else name
        name = name[:name.rfind(".")]
        dataset = Dataset(name)

        if file_path.endswith('.xlsx'):
            engine = 'openpyxl'
        elif file_path.endswith('.xls'):
            engine = 'xlrd'
        else:
            raise ValueError("Unsupported file format, use .xls or .xlsx")

        try:
            df = pd.read_excel(file_path, sheet_name=None, engine=engine)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")

        for sheet_name, sheet_df in df.items():
            fold_name = sheet_name.lower().replace(" ", "")
            fold = EffortEstimationFold(fold_name)
            header = sheet_df.columns.tolist()

            id_col = next((i for i, col in enumerate(header) if 'id' in col.lower()), None)
            effort_col = next((i for i, col in enumerate(header) if 'effort' in col.lower()), None)
            if id_col is None or effort_col is None:
                raise ValueError(f"Sheet '{sheet_name}' missing ID or Effort column")

            feature_cols = [col for i, col in enumerate(header) if i not in (id_col, effort_col)]
            fold.header = feature_cols

            for idx, row in sheet_df.iterrows():
                if pd.isna(row[id_col]) or pd.isna(row[effort_col]):
                    continue

                row_id = int(row[id_col])
                effort = float(row[effort_col])

                for col in feature_cols:
                    val = row[col]
                    if pd.isna(val) or str(val).strip() == "":
                        raise ValueError(f"Empty cell in {fold_name} at row {idx + 2} (Excel row)")

                data_row = DataRow(row_id)
                for col in feature_cols:
                    data_row.add_value(float(row[col]))

                fold.add_row(data_row)
                fold.add_effort(row_id, effort)

            dataset.add_fold(fold)
            print(f"Successfully read fold: {fold_name}, rows: {fold.get_number_of_rows()}")
        return dataset


class SEEProblem(FloatProblem):
    WEIGHT_LOWER_LIMIT = -100
    WEIGHT_UPPER_LIMIT = 100
    OPERATOR_VALUES = [0, 1, 2, 3]

    def __init__(self, fold, seed):
        super(SEEProblem, self).__init__()
        self.fold = fold
        self.feature_count = len(fold.header)
        self._number_of_variables = 2 * self.feature_count + 1
        self.seed=seed
        self.lower_bound = []
        self.upper_bound = []
        for i in range(self._number_of_variables):
            if i % 2 == 0:
                self.lower_bound.append(self.WEIGHT_LOWER_LIMIT)
                self.upper_bound.append(self.WEIGHT_UPPER_LIMIT)
            elif i < 2 * self.feature_count:
                self.lower_bound.append(0.0)
                self.upper_bound.append(3.0)
            else:
                self.lower_bound.append(self.WEIGHT_LOWER_LIMIT)
                self.upper_bound.append(self.WEIGHT_UPPER_LIMIT)

        self._number_of_objectives = 2
        self._number_of_constraints = 0

        self.sae_min = float('inf')
        self.sae_max = float('-inf')
        self.ci_min = float('inf')
        self.ci_max = float('-inf')

    def normalize_population(self, population, mode, t, t_max, age_info, novelty_archive):
        valid_solutions = [s for s in population if not any(np.isinf(obj) for obj in s.original_objectives)]
        if not valid_solutions:
            return

        saes = [sol.original_objectives[0] for sol in valid_solutions]
        cis = [sol.original_objectives[1] for sol in valid_solutions]

        sae_min, sae_max = min(saes), max(saes)
        ci_min, ci_max = min(cis), max(cis)

        if mode == 'ft_fa':
            for sol in valid_solutions:
                sol.objectives = [sol.original_objectives[0], sol.original_objectives[1]]

        elif mode == 'g1_g2':
            for sol in valid_solutions:
                norm_sae = (sol.original_objectives[0] - sae_min) / (sae_max - sae_min) if sae_max != sae_min else 0.5
                norm_ci = (sol.original_objectives[1] - ci_min) / (ci_max - ci_min) if ci_max != ci_min else 0.5
                sol.objectives = [norm_sae + norm_ci, norm_sae - norm_ci]

        elif mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa',
                      'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_solutions]
            num_vars = len(valid_solutions[0].variables)
            unique_elements_per_column = [sorted({sol.variables[i] for sol in valid_solutions})
                                          for i in range(num_vars)]

            mode_construction = mode.split('_')[0]
            ft, fa = generate_fa(
                configurations=configurations,
                ft=saes,
                fa_construction=mode_construction,
                minimize=True,
                unique_elements_per_column=unique_elements_per_column,
                t=t,
                t_max=t_max,
                random_seed=self.seed,
                age_info=age_info,
                novelty_archive=novelty_archive,
                k=len(valid_solutions)//2
            )

            for i, sol in enumerate(valid_solutions):
                sol.objectives = [ft[i] + fa[i], ft[i] - fa[i]]

    def evaluate(self, solution):
        predictions = {}
        for row in self.fold.rows:
            row_id = row.id
            prediction = self._predict(row, solution)
            predictions[row_id] = prediction

        if any(pred < 0 for pred in predictions.values()):
            solution.original_objectives = [float('inf'), float('inf')]
            solution.objectives = [float('inf'), float('inf')]
            return

        residuals = []
        for row_id, pred in predictions.items():
            actual = self.fold.get_effort(row_id)
            absolute_error = abs(actual - pred)
            residuals.append(absolute_error)

        sae = math.floor(sum(residuals))
        n = len(residuals)
        if n < 2:
            ci = 0.0
        else:
            mean = sum(residuals) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in residuals) / (n - 1))
            t_crit = t.ppf(0.95, df=n - 1)
            ci = math.floor(t_crit * std / math.sqrt(n))

        solution.original_objectives = [sae, ci]
        solution.objectives = solution.original_objectives

    def _predict(self, row, solution):
        prediction = 0.0
        for i in range(row.get_length()):
            weight_idx = 2 * i
            op_idx = 2 * i + 1
            w = solution.variables[weight_idx]
            f = row.get_value(i)
            op = int(solution.variables[op_idx])
            op = op % 4
            if op == 0:
                wf = w + f
            elif op == 1:
                wf = w - f
            elif op == 3 and f != 0:
                wf = w / f
            else:
                wf = w * f

            prediction += wf
        prediction += solution.variables[-1]
        return math.floor(prediction)

    def name(self):
        return "SEEProblem"

    def number_of_constraints(self):
        return self._number_of_constraints

    def number_of_objectives(self):
        return self._number_of_objectives

class NSGA2Runner:
    def __init__(self, config, train_dataset, test_dataset, selected_folds=None):
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.selected_folds = selected_folds
        self.output_path = '../Results/RQ1-raw-data/SEE'
        self.best_vars_path = os.path.join(BEST_VARIABLES_PATH, train_dataset.name)
        self.validation_path = os.path.join(VALIDATION_RESULTS_PATH, train_dataset.name)
        # Use exist_ok=True to avoid race conditions when creating directories from multiple processes
        for path in [self.output_path, self.best_vars_path, self.validation_path]:
            try:
                os.makedirs(path, exist_ok=True)
            except FileExistsError:
                # Defensive: if another process created the dir between check/create, ignore
                pass
        self.MAX_RUNTIME = 24 * 3600

    def _setup_algorithm(self, problem, run_seed):
        return NSGAII(
            problem=problem,
            population_size=self.config['population_size'],
            offspring_population_size=self.config['population_size'],
            mutation=SEEDoubleSimpleRandomMutation(self.config['mutation_rate']),
            crossover=SEESinglePointCrossover(self.config['crossover_rate']),
            selection=BinaryTournamentSelection(comparator=DominanceComparator()),
        )

    def run(self, mode):
        t_max = 250
        budget = self.config['budget']
        max_generations = self.config['max_generations']
        start_seed = self.config['start_seed']
        end_seed = self.config['end_seed']

        for run_seed in range(start_seed, end_seed + 1):
            print(f"Starting seed: {run_seed}")
            for fold in self.train_dataset.get_folds():
                if self.selected_folds and fold.name not in self.selected_folds:
                    print(f"Skipping fold: {fold.name} (not in selected folds)")
                    continue

                fold_name = fold.name
                print(f"Processing fold: {fold_name}")

                dataset_name = self.train_dataset.name
                output_file = os.path.join(
                    self.output_path,
                    f"{dataset_name}_{fold_name}_{run_seed}_{mode}.csv"
                )

                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"NSGA-II Run for {dataset_name} with seed {run_seed} and mode {mode}"])
                    writer.writerow([f"Budget: {budget}"])
                    writer.writerow([])

                random.seed(run_seed)
                np.random.seed(run_seed)
                problem = SEEProblem(fold, run_seed)
                algorithm = self._setup_algorithm(problem, run_seed)

                population = algorithm.create_initial_solutions()
                pop_size = self.config['population_size']
                total_evaluations = 0
                start_time = time.time()

                age_info = list(range(1, pop_size + 1)) if mode == 'age_maximization_fa' else None
                novelty_archive = [] if mode == 'novelty_maximization_fa' else None

                evaluated_population = []
                for solution in population:
                    if total_evaluations >= budget or (time.time() - start_time) >= self.MAX_RUNTIME:
                        break

                    problem.evaluate(solution)
                    total_evaluations += 1
                    if not any(np.isinf(obj) for obj in solution.original_objectives):
                        evaluated_population.append(solution)

                population = evaluated_population
                if total_evaluations > budget or (time.time() - start_time) >= self.MAX_RUNTIME:
                    continue

                best_sae = float('inf')
                best_ci = float('inf')
                best_variables = None
                best_generation = 0
                best_p = None
                p_values = []
                generation = 1

                if novelty_archive is not None:
                    update_novelty_archive(population, novelty_archive)

                problem.normalize_population(
                    population=population,
                    mode=mode,
                    t=generation,
                    t_max=t_max,
                    age_info=age_info,
                    novelty_archive=novelty_archive
                )

                while True:
                    if total_evaluations > budget or (time.time() - start_time) >= self.MAX_RUNTIME:
                        break

                    mating_population = algorithm.selection(population)
                    offspring_population = algorithm.reproduction(mating_population)

                    evaluated_offspring = []
                    for solution in offspring_population:
                        problem.evaluate(solution)
                        total_evaluations += 1
                        if not any(np.isinf(obj) for obj in solution.original_objectives):
                            evaluated_offspring.append(solution)

                    if total_evaluations > budget or (time.time() - start_time) >= self.MAX_RUNTIME:
                        break

                    combined_age_info = None
                    if age_info is not None:
                        offspring_age = [pop_size + generation] * len(evaluated_offspring)
                        combined_age_info = age_info + offspring_age

                    combined_population = population + evaluated_offspring
                    problem.normalize_population(
                        population=combined_population,
                        mode=mode,
                        t=generation,
                        t_max=t_max,
                        age_info=combined_age_info,
                        novelty_archive=novelty_archive
                    )

                    if novelty_archive is not None:
                        update_novelty_archive(evaluated_offspring, novelty_archive)

                    unique_combined = []
                    seen_vars = set()
                    for sol in combined_population:
                        var_tuple = tuple(sol.variables)
                        if var_tuple not in seen_vars:
                            seen_vars.add(var_tuple)
                            unique_combined.append(sol)

                    non_dominated = get_non_dominated_solutions(unique_combined)
                    p = len(non_dominated) / len(unique_combined) if unique_combined else 0.0
                    p_values.append(p)

                    if mode == 'ft_fa':
                        new_population = algorithm.replacement(population, evaluated_offspring)
                    else:
                        new_population = partial_duplicate_replacement(combined_population, pop_size)

                    if age_info is not None:
                        combined_index = {id(sol): idx for idx, sol in enumerate(combined_population)}
                        age_info = [combined_age_info[combined_index[id(sol)]] for sol in new_population]

                    population = new_population

                    current_valid_sols = [sol for sol in population if
                                          not any(np.isinf(obj) for obj in sol.original_objectives)]
                    if current_valid_sols:
                        current_best_sae = min(sol.original_objectives[0] for sol in current_valid_sols)
                        if current_best_sae < best_sae:
                            best_sae = current_best_sae
                            best_sol = next(sol for sol in current_valid_sols if sol.original_objectives[0] == best_sae)
                            best_ci = best_sol.original_objectives[1]
                            best_variables = [round(v, 4) for v in best_sol.variables]
                            best_generation = generation
                            best_p = p

                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([f"Generation {generation} p-value: {p:.4f} best value : {best_sae:.2f}"])

                    print(
                        f"{dataset_name} {mode} | Seed {run_seed} | Gen {generation} | "
                        f"Evals: {total_evaluations}/{budget} | Best SAE: {best_sae:.2f}"
                    )
                    generation += 1

                runtime = time.time() - start_time
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([])
                    writer.writerow([f"budget_used:{total_evaluations}"])
                    writer.writerow([f"Running time: {runtime:.2f} seconds"])
                    writer.writerow([])

                    best_p = p_values[best_generation - 1] if best_generation > 0 and p_values else 0
                    writer.writerow([
                        f"Best Solution: 'ft': {best_sae:.6f}, 'fa': {best_ci:.6f} "
                        f"appeared in Generation {best_generation}, p: {best_p:.4f}"
                    ])

                    if best_generation > 0 and p_values:
                        p_values_str = ",".join([f"{p:.4f}" for p in p_values[:best_generation]])
                        writer.writerow([f"p values until best solution: {p_values_str}"])

                if best_variables is not None:
                    best_vars_file = os.path.join(
                        self.best_vars_path,
                        f"best_vars_{dataset_name}_{fold_name}_{run_seed}_{mode}.csv"
                    )
                    with open(best_vars_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["variable_index", "value"])
                        for idx, var in enumerate(best_variables):
                            writer.writerow([idx, var])
                    print(f"Best individual variables saved to: {best_vars_file}")
                print(f"Finished fold: {fold_name}")
            print(f"Finished seed: {run_seed}")


def compute_random_guess_mae(target_fold, num_trials=1000, seed=None):
    all_actual_efforts = [target_fold.get_effort(row.id) for row in target_fold.get_rows()]
    num_projects = len(all_actual_efforts)
    if num_projects == 0:
        return 0.0

    random.seed(seed)
    np.random.seed(seed)

    total_random_mae = 0.0
    for _ in range(num_trials):
        trial_errors = []
        for row in target_fold.get_rows():
            actual = target_fold.get_effort(row.id)
            guess = random.choice(all_actual_efforts)
            trial_errors.append(abs(actual - guess))
        total_random_mae += (sum(trial_errors) / len(trial_errors))

    return total_random_mae / num_trials

def validate_saved_variables(test_file, var_file_path, dataset_name, fold_name, run_seed, mode, selected_folds=None):
    validation_path = os.path.join(VALIDATION_RESULTS_PATH, dataset_name)
    os.makedirs(validation_path, exist_ok=True)

    try:
        test_dataset = ExcelReader.read(test_file)
        target_folds = []
        for f in test_dataset.get_folds():
            if selected_folds is None or f.name in selected_folds:
                if fold_name == f.name:
                    target_folds.append(f)
    except Exception as e:
        print(f"Failed to load test dataset: {e}")
        return

    if not target_folds:
        print(f"No matching fold found for validation: {fold_name}")
        return

    try:
        var_df = pd.read_csv(var_file_path)
        best_variables = var_df['value'].tolist()
    except Exception as e:
        print(f"Failed to read variable file: {e}")
        return

    for target_fold in target_folds:
        results = []
        model_absolute_errors = []

        for row in target_fold.get_rows():
            row_id = row.id
            actual_effort = target_fold.get_effort(row_id)

            prediction = 0.0
            for i in range(row.get_length()):
                weight_idx = 2 * i
                op_idx = 2 * i + 1
                if weight_idx >= len(best_variables) or op_idx >= len(best_variables):
                    print(f"Variable index out of range, skipping row {row_id}")
                    prediction = -1
                    break
                w = best_variables[weight_idx]
                f = row.get_value(i)
                op = int(best_variables[op_idx]) % 4

                if op == 0:
                    wf = w + f
                elif op == 1:
                    wf = w - f
                elif op == 3 and f != 0:
                    wf = w / f
                else:
                    wf = w * f
                prediction += wf

            if len(best_variables) > 0 and prediction != -1:
                prediction += best_variables[-1]
            prediction = math.floor(prediction)

            absolute_error = abs(actual_effort - prediction) if prediction != -1 else float('inf')
            results.append({
                'project_id': row_id,
                'actual_effort': actual_effort,
                'predicted_effort': prediction,
                'absolute_error': absolute_error
            })

            if prediction != -1:
                model_absolute_errors.append(absolute_error)

        if model_absolute_errors:
            mae_pi = sum(model_absolute_errors) / len(model_absolute_errors)
        else:
            mae_pi = float('inf')

        mae_p0 = compute_random_guess_mae(target_fold, num_trials=1000, seed=run_seed)

        if mae_p0 == 0:
            sa_value = 100.0 if mae_pi == 0 else -100.0
        else:
            sa_value = (1 - (mae_pi / mae_p0)) * 100

        result_file = os.path.join(
            validation_path,
            f"validation_{dataset_name}_{fold_name}_{run_seed}_{mode}.csv"
        )
        with open(result_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['project_id', 'actual_effort', 'predicted_effort', 'absolute_error'])
            writer.writeheader()
            writer.writerows(results)
            f.write('\n')
            f.write(f"模型MAE,{mae_pi:.4f}\n")
            f.write(f"随机猜测MAE,{mae_p0:.4f}\n")
            f.write(f"标准准确率(SA),{sa_value:.4f}\n")

        print(f"Validation results saved to: {result_file}, SA: {sa_value:.4f}")


def process_project_mode(args):
    dataset_name, train_file, test_file, mode, CONFIG, selected_folds = args
    print(f"Processing project: {dataset_name} Mode: {mode} Fold: {selected_folds or 'all'}")
    try:
        start_time = time.time()
        train_dataset = ExcelReader.read(train_file)
        train_dataset.name = dataset_name
        test_dataset = ExcelReader.read(test_file)
        test_dataset.name = dataset_name

        runner = NSGA2Runner(CONFIG, train_dataset, test_dataset, selected_folds)
        runner.run(mode)
        elapsed = time.time() - start_time
        return (dataset_name, mode, elapsed, "success")

    except Exception as e:
        print(f"Project {dataset_name} Mode {mode} failed: {e}")
        traceback.print_exc()
        return (dataset_name, mode, -1, f"failed: {str(e)}")


import argparse
import time
import os
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
    Argument-driven entry for SEE runner.

    Parameter names are aligned with the NAS script:
      --use-parallel / --no-parallel
      --cpu-cores
      --mode (single mode or 'all')
      --seeds (single, CSV, or range)

    Only the parameter-passing interface is changed; algorithm implementation remains unmodified.
    """
    # Defaults consistent with NAS script where applicable
    default_cpu_cores = 50
    default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                     'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser = argparse.ArgumentParser(description="Run SEE NSGA-II experiments with argument configuration")
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
    num_processes = max(1, parsed.cpu_cores)

    # Keep existing default configuration and datasets
    CONFIG = {
        'root_path': '.',
        'max_generations': 250,
        'population_size': 100,
        'crossover_rate': 0.5,
        'mutation_rate': 0.1,
        'objectives': [],
        'budget': 25000,
        'start_seed': min(seeds_list),
        'end_seed': max(seeds_list),
        'run_mode': 'optimize'
    }

    DATA_ROOT = './Datasets/'
    TARGET_DATASETS = ["china", "desharnais", "finnish", "maxwell", "miyazaki"]
    # keep existing default SELECTED_FOLDS
    SELECTED_FOLDS = ['fold1','fold1','fold3']

    print(f"Data root directory: {DATA_ROOT}")
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory {DATA_ROOT} does not exist.")
        return

    print(f"Start processing target datasets: {', '.join(TARGET_DATASETS)}")
    print(f"Selected folds: {SELECTED_FOLDS or 'all'}")
    print(f"Using modes: {modes_to_run}")
    print(f"Seeds: {seeds_list}")
    print(f"use_parallel={use_parallel}, cpu_cores={num_processes}")

    # Build tasks
    tasks = []
    for dataset_name in TARGET_DATASETS:
        train_file = os.path.join(DATA_ROOT, f'{dataset_name}-train.xls')
        test_file = os.path.join(DATA_ROOT, f'{dataset_name}-test.xls')

        if not os.path.exists(train_file):
            print(f"Warning: Train file not found - {train_file}, skipping dataset")
            continue
        if not os.path.exists(test_file):
            print(f"Warning: Test file not found - {test_file}, skipping dataset")
            continue

        for mode in modes_to_run:
            tasks.append((dataset_name, train_file, test_file, mode, CONFIG, SELECTED_FOLDS))

    if CONFIG['run_mode'] == 'optimize':
        if use_parallel:
            print(f"Using parallel processing with {num_processes} processes")
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(process_project_mode, task) for task in tasks]
                for future in futures:
                    try:
                        result = future.result()
                        project, mode, elapsed, status = result
                        print(f"Completed: {project} | Mode: {mode} | Time: {elapsed:.2f}s | Status: {status}")
                    except Exception as e:
                        print(f"Task failed: {str(e)}")
        else:
            print("Running in single-process mode")
            for task in tasks:
                process_project_mode(task)

    elif CONFIG['run_mode'] == 'validate':
        print("Starting validation mode")
        for dataset_name in TARGET_DATASETS:
            var_files_pattern = os.path.join(BEST_VARIABLES_PATH, dataset_name, f'best_vars_{dataset_name}_*.csv')
            var_files = glob.glob(var_files_pattern)

            if not var_files:
                print(f"Warning: No variable files found for dataset {dataset_name}")
                continue

            for var_file in var_files:
                filename = os.path.basename(var_file)
                parts = filename.replace('best_vars_', '').split('_')
                if len(parts) < 4 or parts[0] != dataset_name:
                    print(f"Skipping invalid filename: {filename}")
                    continue

                fold_name = parts[1]
                if SELECTED_FOLDS and fold_name not in SELECTED_FOLDS:
                    print(f"Skipping fold: {fold_name} (not in selected list)")
                    continue

                run_seed = int(parts[2])
                mode = '_'.join(parts[3:]).replace('.csv', '')

                test_file = os.path.join(DATA_ROOT, f'{dataset_name}-test.xls')
                if not os.path.exists(test_file):
                    print(f"Test file not found: {test_file}, skipping")
                    continue

                print(f"Validating dataset: {dataset_name} Fold: {fold_name} Seed: {run_seed} Mode: {mode}")
                validate_saved_variables(
                    test_file=test_file,
                    var_file_path=var_file,
                    dataset_name=dataset_name,
                    fold_name=fold_name,
                    run_seed=run_seed,
                    mode=mode,
                    selected_folds=SELECTED_FOLDS
                )

    print("All tasks completed!")


if __name__ == "__main__":
    main()
