import sys
from copy import copy
import csv
import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import warnings
import random
import uuid
import time
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.mutation import Mutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.operator import Crossover
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import get_non_dominated_solutions
from Code.Utils.remove_duplicates import partial_duplicate_replacement

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
from Code.SDP.Utils.Construct_secondary_objective import generate_fa, update_novelty_archive

warnings.filterwarnings("ignore")

def evaluate_solution_on_test(solution: BinarySolution, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray, classifier, feature_names: list) -> dict:
    selected_idx = [i for i, v in enumerate(solution.variables) if v == 1]
    feature_count = len(selected_idx)
    selected_feats = [feature_names[i] for i in selected_idx] if selected_idx else []
    train_auc = -solution.attributes['original_auc']
    if feature_count == 0:
        test_auc = 0.5
    else:
        X_train_subset = X_train[:, selected_idx].astype(np.float64)
        X_test_subset = X_test[:, selected_idx].astype(np.float64)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
        cloned_clf = clone(classifier)
        cloned_clf.fit(X_train_scaled, y_train)
        try:
            y_test_proba = cloned_clf.predict_proba(X_test_scaled)[:, 1]
        except AttributeError:
            y_test_score = cloned_clf.decision_function(X_test_scaled)
            y_test_proba = (y_test_score - y_test_score.min()) / (y_test_score.max() - y_test_score.min())
        test_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) >= 2 else 0.5
    return {
        "solution_uuid": str(solution.uuid)[:8],
        "feature_count": feature_count,
        "selected_features": selected_feats,
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4)
    }

class SimulatedBinaryCrossover(Crossover[BinarySolution, BinarySolution]):
    def __init__(self, probability: float = 0.9):
        super().__init__(probability=probability)
        self.probability = probability

    def execute(self, parents: list[BinarySolution]) -> list[BinarySolution]:
        if len(parents) != 2 or not isinstance(parents[0], BinarySolution) or not isinstance(parents[1],
                                                                                             BinarySolution):
            raise ValueError("SimulatedBinaryCrossover requires 2 BinarySolution parents")
        parent1, parent2 = parents
        n_features = len(parent1.variables)
        offspring1 = BinarySolution(number_of_variables=n_features, number_of_objectives=len(parent1.objectives))
        offspring2 = BinarySolution(number_of_variables=n_features, number_of_objectives=len(parent1.objectives))
        offspring1.variables = [0] * n_features
        offspring2.variables = [0] * n_features
        if random.random() <= self.probability and n_features > 1:
            start_idx = random.randint(0, n_features - 2)
            end_idx = random.randint(start_idx + 1, n_features - 1)
            for i in range(n_features):
                if start_idx <= i <= end_idx:
                    offspring1.variables[i] = parent2.variables[i]
                    offspring2.variables[i] = parent1.variables[i]
                else:
                    offspring1.variables[i] = parent1.variables[i]
                    offspring2.variables[i] = parent2.variables[i]
        else:
            offspring1.variables = copy(parent1.variables)
            offspring2.variables = copy(parent2.variables)
        return [offspring1, offspring2]

    def get_number_of_children(self) -> int:
        return 2

    def get_number_of_parents(self) -> int:
        return 2

    def get_name(self) -> str:
        return "SimulatedBinaryCrossover"

class OneBitFlipMutation(Mutation[BinarySolution]):
    def __init__(self, probability: float):
        super().__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                solution.variables[i] = 1 - solution.variables[i]
        return solution

    def get_name(self) -> str:
        return "OneBitFlipMutation"

class AntDefectPredictionProblem(Problem[BinarySolution]):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, classifier, mode: str = 'ft_fa', seed: int = 42):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.n_features = X_train.shape[1]
        self.n_objectives = 2
        self.n_constraints = 0
        self.classifier = classifier
        self.mode = mode
        self.seed = seed
        self.fitness_cache = {}
        self.original_objectives = {}

    @property
    def name(self) -> str:
        return "MOFES: Multi-Objective FEature Selection"

    @property
    def number_of_variables(self) -> int:
        return self.n_features

    @property
    def number_of_objectives(self) -> int:
        return self.n_objectives

    @property
    def number_of_constraints(self) -> int:
        return self.n_constraints

    @property
    def objective_directions(self) -> list:
        return [self.MINIMIZE, self.MINIMIZE]

    @property
    def objective_labels(self) -> list:
        if self.mode == 'ft_fa':
            return ["-AUC (minimize equivalent to maximize AUC)", "Feature count (minimize)"]
        elif self.mode == 'g1_g2':
            return ["g1 = norm-AUC + norm-featurecount (minimize)", "g2 = |norm-AUC - norm-featurecount| (minimize)"]
        else:
            return ["g1 = adjusted ft + adjusted fa (minimize)", "g2 = adjusted ft - adjusted fa (minimize)"]

    def create_solution(self) -> BinarySolution:
        solution = BinarySolution(number_of_variables=self.n_features, number_of_objectives=self.n_objectives)
        solution.variables = [0] * self.n_features
        solution.uuid = uuid.uuid4()
        if self.n_features > 0:
            selected_idx = random.randint(0, self.n_features - 1)
            solution.variables[selected_idx] = 1
        return solution

    def _get_solution_key(self, solution: BinarySolution) -> tuple:
        return tuple(solution.variables)

    def evaluate(self, solution: BinarySolution) -> None:
        solution_key = self._get_solution_key(solution)
        if solution_key in self.fitness_cache:
            original_auc, original_featcount = self.original_objectives[solution_key]
            solution.attributes['original_auc'] = original_auc
            solution.attributes['original_featcount'] = original_featcount
            solution.objectives = self.fitness_cache[solution_key]
            return
        selected_features = [i for i, v in enumerate(solution.variables) if v == 1]
        feature_count = len(selected_features)
        if feature_count == 0:
            original_auc = -0.5
            original_featcount = float(self.n_features)
        else:
            X_train_subset = self.X_train[:, selected_features].astype(np.float64)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
            cv_aucs = []
            cloned_clf = clone(self.classifier)
            for train_idx, test_idx in skf.split(X_train_scaled, self.y_train):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[test_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[test_idx]
                if len(np.unique(y_val)) < 2:
                    cv_aucs.append(0.5)
                    continue
                cloned_clf.fit(X_tr, y_tr)
                try:
                    y_val_proba = cloned_clf.predict_proba(X_val)[:, 1]
                except AttributeError:
                    y_val_score = cloned_clf.decision_function(X_val)
                    y_val_proba = (y_val_score - y_val_score.min()) / (y_val_score.max() - y_val_score.min())
                auc = roc_auc_score(y_val, y_val_proba) if not np.isnan(y_val_proba).any() else 0.5
                cv_aucs.append(auc)
            avg_auc = np.mean(cv_aucs) if cv_aucs else 0.5
            original_auc = -avg_auc
            original_featcount = float(feature_count)
        solution.attributes['original_auc'] = original_auc
        solution.attributes['original_featcount'] = original_featcount
        self.original_objectives[solution_key] = (original_auc, original_featcount)
        solution.objectives = [original_auc, original_featcount]
        self.fitness_cache[solution_key] = copy(solution.objectives)

    def normalize_population(self, population: list[BinarySolution],
                             t: int = 1, t_max: int = 1000,
                             age_info: List[int] = None, novelty_archive: List[Tuple] = None) -> None:
        valid_solutions = [s for s in population if not any(np.isinf(obj) for obj in s.objectives)]
        if not valid_solutions:
            return
        ft = [s.attributes['original_auc'] for s in valid_solutions]
        fa = [s.attributes['original_featcount'] for s in valid_solutions]
        if self.mode == 'ft_fa':
            for sol in valid_solutions:
                sol.objectives = [
                    sol.attributes['original_auc'],
                    sol.attributes['original_featcount']
                ]
        elif self.mode == 'g1_g2':
            ft_min, ft_max = min(ft), max(ft)
            fa_min, fa_max = min(fa), max(fa)
            norm_ft = [(f - ft_min) / (ft_max - ft_min) if ft_max != ft_min else 0.5 for f in ft]
            norm_fa = [(f - fa_min) / (fa_max - fa_min) if fa_max != fa_min else 0.5 for f in fa]
            for idx, sol in enumerate(valid_solutions):
                sol.objectives = [norm_ft[idx] + norm_fa[idx], abs(norm_ft[idx] - norm_fa[idx])]
        elif self.mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa',
                           'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
            configurations = [tuple(sol.variables) for sol in valid_solutions]
            unique_elements = [sorted({sol.variables[i] for sol in valid_solutions})
                               for i in range(len(valid_solutions[0].variables))]
            core_mode = self.mode.split('_')[0]
            k = len(population) // 2
            adjusted_ft, adjusted_fa = generate_fa(
                configurations,
                ft,
                core_mode,
                True,
                "",
                unique_elements,
                t,
                t_max,
                self.seed,
                age_info,
                novelty_archive,
                k
            )
            for idx, sol in enumerate(valid_solutions):
                sol.objectives = [adjusted_ft[idx] + adjusted_fa[idx], adjusted_ft[idx] - adjusted_fa[idx]]

def load_ant_data(file_path: str) -> tuple[np.ndarray, np.ndarray, list, list]:
    df = pd.read_csv(file_path)
    exclude_cols = ["name", "version", "bug"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df = df.drop(col, axis=1)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df = df.dropna(axis=1, how="any")
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df["bug_binary"] = df["bug"].apply(lambda x: 1 if x >= 1 else 0)
    defect_count = df["bug_binary"].sum()
    non_defect_count = len(df) - defect_count
    print("Label conversion")
    print(f"Defective modules: {defect_count}, Non-defective modules: {non_defect_count}, Defect ratio: {defect_count / len(df):.2%}")
    X = df[feature_cols].values.astype(np.float64)
    y = df["bug_binary"].values.astype(np.int64)
    print("\nData loading completed")
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}, Label distribution: Defective {defect_count}, Non-defective {non_defect_count}")
    return X, y, feature_cols, feature_cols

def run_mofes_single_classifier(X, y, feature_names, classifier, classifier_name: str, mode: str = 'ft_fa',
                                dataset_name: str = "unknown", seed: int = 0) -> dict:
    n_features = len(feature_names)
    result_dir = "../Results/RQ1-raw-data/SDP"
    os.makedirs(result_dir, exist_ok=True)
    csv_filename = f"{dataset_name}_{classifier_name}_{seed}_{mode}.csv"
    csv_path = os.path.join(result_dir, csv_filename)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([f"NSGA-II Run for {dataset_name} (Classifier: {classifier_name}, Seed: {seed}, Mode: {mode})"])
        writer.writerow([f"Budget: 3000 (Max Evaluations)"])
        writer.writerow([])
        writer.writerow([f"Data Split: Train {len(X_train)} samples, Test {len(X_test)} samples"])
    print("\n" + "=" * 120)
    print(f"Experiment {dataset_name} | Classifier [{classifier_name}] | Mode [{mode}] | Seed [{seed}]")
    print("=" * 120)
    print(f"Data split: Seed {seed} | Train {len(X_train)} samples | Test {len(X_test)} samples")
    print(f"Termination condition: Evaluations ≤3000 or Time ≤24h")
    best_single_obj_auc = 0.0
    best_single_obj_solution = None
    best_single_obj_generation = 0
    best_single_obj_p = 0.0
    p_values_history = []
    max_evaluations = 3000
    max_time = 86400
    population_size = 100
    t_max = 30
    age_info = None
    novelty_archive = None
    if mode == 'age_maximization_fa':
        age_info = list(range(1, population_size + 1))
    if mode == 'novelty_maximization_fa':
        novelty_archive = []
    problem = AntDefectPredictionProblem(
        X_train=X_train, y_train=y_train, classifier=classifier, mode=mode, seed=seed
    )
    crossover = SimulatedBinaryCrossover(probability=0.9)
    mutation = OneBitFlipMutation(probability=1.0 / n_features)
    selection = BinaryTournamentSelection(comparator=DominanceComparator())
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    print(f"\nEvolutionary optimization: seed {seed} | Budget {max_evaluations} evaluations | Population {population_size} | Max time {max_time}s")
    start_time = time.time()
    population = algorithm.create_initial_solutions()
    evaluated_population = []
    evaluation_count = 0
    for sol in population:
        problem.evaluate(sol)
        evaluation_count += 1
        if not any(math.isinf(obj) for obj in sol.objectives):
            evaluated_population.append(sol)
            current_auc = -sol.attributes['original_auc']
            if current_auc > best_single_obj_auc:
                best_single_obj_auc = current_auc
                best_single_obj_solution = sol
                best_single_obj_generation = 1
    population = evaluated_population
    current_generation = 1
    if novelty_archive is not None:
        update_novelty_archive(population, novelty_archive)
    problem.normalize_population(
        population=population,
        t=current_generation,
        t_max=t_max,
        age_info=age_info,
        novelty_archive=novelty_archive
    )
    while evaluation_count < max_evaluations and (time.time() - start_time) < max_time:
        mating_population = algorithm.selection(population)
        offspring_population = algorithm.reproduction(mating_population)
        for sol in offspring_population:
            sol.uuid = uuid.uuid4()
        evaluated_offspring = []
        for sol in offspring_population:
            if evaluation_count >= max_evaluations or (time.time() - start_time) >= max_time:
                break
            problem.evaluate(sol)
            evaluation_count += 1
            if not any(math.isinf(obj) for obj in sol.objectives):
                evaluated_offspring.append(sol)
        if novelty_archive is not None and evaluated_offspring:
            update_novelty_archive(evaluated_offspring, novelty_archive)
        combined_population = population + evaluated_offspring
        combined_age_info = None
        if mode == 'age_maximization_fa':
            offspring_age = [population_size + current_generation] * len(evaluated_offspring)
            combined_age_info = age_info + offspring_age
        problem.normalize_population(
            population=combined_population,
            t=current_generation,
            t_max=t_max,
            age_info=combined_age_info,
            novelty_archive=novelty_archive
        )
        unique_combined_population = []
        seen_vars = set()
        for sol in combined_population:
            var_tuple = tuple(sol.variables)
            if var_tuple not in seen_vars:
                unique_combined_population.append(sol)
                seen_vars.add(var_tuple)
        non_dominated_sols = get_non_dominated_solutions(unique_combined_population) if unique_combined_population else []
        current_p = len(non_dominated_sols) / len(unique_combined_population) if unique_combined_population else 0.0
        current_p = round(current_p, 4)
        p_values_history.append(current_p)
        if mode == 'ft_fa':
            new_population = algorithm.replacement(population, evaluated_offspring)
        else:
            new_population = partial_duplicate_replacement(combined_population, population_size)
        if mode == 'age_maximization_fa':
            uuid_to_index = {sol.uuid: idx for idx, sol in enumerate(combined_population)}
            selected_indices = [uuid_to_index[sol.uuid] for sol in new_population]
            age_info = [combined_age_info[idx] for idx in selected_indices]
        population = new_population
        current_round_best_auc = 0.0
        current_round_best_sol = None
        for sol in population:
            current_auc = -sol.attributes['original_auc']
            if current_auc > current_round_best_auc:
                current_round_best_auc = current_auc
                current_round_best_sol = sol
        if current_round_best_auc > best_single_obj_auc:
            best_single_obj_auc = current_round_best_auc
            best_single_obj_solution = current_round_best_sol
            best_single_obj_generation = current_generation
            best_single_obj_p = current_p
        elapsed_time = time.time() - start_time
        print(f"{dataset_name} {classifier_name} Seed {seed} | Gen {current_generation} | "
              f"Eval: {evaluation_count}/{max_evaluations} | Time: {elapsed_time:.1f}s | "
              f"p-value: {current_p:.4f} | Best AUC: {best_single_obj_auc:.4f}")
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(
                [f"Generation {current_generation} p-value: {current_p:.4f} best value: {best_single_obj_auc:.4f}"])
        current_generation += 1
    best_single_obj_test_auc = 0.5
    best_single_obj_feat_count = 0
    best_selected_feats = []
    if best_single_obj_solution is not None:
        test_perf = evaluate_solution_on_test(
            best_single_obj_solution, X_train, X_test, y_train, y_test, classifier, feature_names
        )
        best_single_obj_test_auc = test_perf['test_auc']
        best_single_obj_feat_count = test_perf['feature_count']
        best_selected_feats = test_perf['selected_features']
    total_run_time = time.time() - start_time
    p_values_until_best = p_values_history[:best_single_obj_generation] if best_single_obj_generation > 0 else []
    p_values_str = ",".join([f"{p:.4f}" for p in p_values_until_best]) if p_values_until_best else "N/A"
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([])
        writer.writerow([f"budget_used:{evaluation_count}"])
        writer.writerow([f"Running time: {total_run_time:.2f} seconds"])
        writer.writerow([])
        if best_single_obj_solution is not None:
            writer.writerow([
                f"Best Solution: 'auc': {best_single_obj_auc:.6f}, 'test_auc': {best_single_obj_test_auc:.6f}, "
                f"'feature_count': {best_single_obj_feat_count} appeared in Generation {best_single_obj_generation}, p: {best_single_obj_p:.4f}"
            ])
            writer.writerow(
                [f"Best Selected Features: {', '.join(best_selected_feats) if best_selected_feats else 'None'}"])
        else:
            writer.writerow(["Best Solution: No valid optimal solution found"])
        writer.writerow([f"p values until best solution: {p_values_str}"])
    print(f"\nSeed {seed} run finished")
    termination_reason = 'Evaluations reached' if evaluation_count >= max_evaluations else 'Time reached'
    print(f"Total time: {total_run_time:.2f} seconds | Total evaluations: {evaluation_count} | Termination reason: {termination_reason}")
    if best_single_obj_solution:
        print(f"Best solution: Generation {best_single_obj_generation} | p-value {best_single_obj_p:.4f} | "
              f"Train AUC {best_single_obj_auc:.4f} | Test AUC {best_single_obj_test_auc:.4f} | "
              f"Feature count {best_single_obj_feat_count}")
    else:
        print("Warning: No valid optimal solution found")
    print(f"Results file: {csv_path}")
    return {
        "classifier_name": classifier_name,
        "mode": mode,
        "dataset_name": dataset_name,
        "seed": seed,
        "best_auc_train": best_single_obj_auc,
        "best_auc_test": best_single_obj_test_auc,
        "best_feat_count": best_single_obj_feat_count if best_single_obj_solution else 0,
        "best_generation": best_single_obj_generation,
        "best_p_value": best_single_obj_p,
        "best_solution": best_single_obj_solution,
        "best_features": best_selected_feats,
        "p_values_history": p_values_history,
        "total_run_time": total_run_time,
        "total_evaluations": evaluation_count,
        "csv_path": csv_path
    }

def process_single_task(args):
    file_path, clf_name, clf, mode, seed = args
    try:
        X, y, feature_names, _ = load_ant_data(file_path)
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        n_features = len(feature_names)
        if n_features == 0:
            print(f"Warning: Dataset {dataset_name} has no valid features, skipping")
            return None
        result = run_mofes_single_classifier(
            X=X, y=y, feature_names=feature_names,
            classifier=clf, classifier_name=clf_name,
            mode=mode, dataset_name=dataset_name,
            seed=seed
        )
        return result
    except Exception as e:
        print(f"Task failed: file={file_path}, classifier={clf_name}, mode={mode}, seed={seed}, error={str(e)}")
        return None

def run_mofes_with_config(dataset_files, classifiers, modes, seeds, max_workers=50) -> list:
    tasks = []
    for file_path in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        for mode in modes:
            for clf_name, clf in classifiers.items():
                for seed in seeds:
                    cloned_clf = clone(clf)
                    tasks.append((file_path, clf_name, cloned_clf, mode, seed))
    print("\n" + "=" * 80)
    print("Experiment configuration")
    print(f"Number of datasets: {len(dataset_files)} | Number of classifiers: {len(classifiers)}")
    print(f"Number of modes: {len(modes)} | Number of seeds: {len(seeds)}")
    print(f"Total tasks: {len(tasks)} | Max workers: {max_workers}")
    print("Results directory: ./Results/NSGA2 | Single-run termination: 3000 evaluations or 24h")
    print("=" * 80)
    all_results = []
    start_time = time.time()
    for file_path in dataset_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file does not exist: {file_path}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_task, task) for task in tasks]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / len(tasks) * 100
                    print(f"Progress: {i + 1}/{len(tasks)} ({progress:.1f}%) | Elapsed: {elapsed:.2f}s")
            except Exception as e:
                print(f"Task {i + 1} failed: {str(e)}")
    total_time = time.time() - start_time
    print("\n" + "=" * 150)
    print("Global run completed: all tasks executed")
    print(f"Total time: {total_time:.2f}s | Successful tasks: {len(all_results)}/{len(tasks)}")
    print("=" * 150)
    print(f"{len(all_results)} result files generated, sample paths:")
    for idx, result in enumerate(all_results[:10], 1):
        print(f"{idx:2d}. {result['csv_path']}")
    if len(all_results) > 10:
        print(f"... and {len(all_results) - 10} more result files not shown")
    print("=" * 150)
    return all_results

import argparse

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
    default_cpu_cores = 50
    default_modes = ['ft_fa', 'penalty_fa', 'g1_g2', 'gaussian_fa', 'reciprocal_fa',
                     'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']

    parser = argparse.ArgumentParser(description="Run MOFES experiments with argument configuration")
    parser.set_defaults(use_parallel=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-parallel', dest='use_parallel', action='store_true',
                       help='Enable parallel execution (default)')
    group.add_argument('--no-parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel execution (run with a single worker)')

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

    DATASET_NAME = ['ant-1.7', 'camel-1.6', 'ivy-2.0', 'jedit-4.0', 'lucene-2.4',
                    'poi-3.0', 'synapse-1.2', 'velocity-1.6', 'xalan-2.6', 'xerces-1.4']
    DATASET_FILES = [f'./Datasets/{name}.csv' for name in DATASET_NAME]

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    CLASSIFIERS = {
        "J48": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "NB": GaussianNB()
    }

    print(f"Configuration -> datasets: {len(DATASET_FILES)}, classifiers: {len(CLASSIFIERS)}, modes: {modes_to_run}, seeds: {seeds_list}")
    max_workers = cpu_cores if use_parallel else 1
    print(f"use_parallel={use_parallel}, cpu_cores={cpu_cores} -> max_workers set to {max_workers}")

    results = run_mofes_with_config(
        dataset_files=DATASET_FILES,
        classifiers=CLASSIFIERS,
        modes=modes_to_run,
        seeds=seeds_list,
        max_workers=max_workers
    )

    return results


if __name__ == "__main__":
    main()