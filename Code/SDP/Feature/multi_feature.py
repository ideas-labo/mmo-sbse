import concurrent
import csv
import re
from typing import List, Any, Dict
import sys

import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
import numpy as np
import random
import multiprocessing
from itertools import combinations, product
import scipy.stats._qmc as qmc
import os
import uuid

from jmetal.core.solution import BinarySolution
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, '/home/ccj/code/mmo')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.insert(0, '/mnt/mydrive/ccj/code/mmo')
from Code.SDP.Utils.Construct_secondary_objective import update_novelty_archive, generate_fa
from Code.SDP.mmo_sdp import (
    AntDefectPredictionProblem,
    load_ant_data,
    evaluate_solution_on_test
)
from Code.SDP.Feature.utils.multi_feature_compute import plot_figure1, plot_figure2

def generate_binary_samples(dimensions: int, num_samples: int,
                            sampling_method: str, random_seed: int) -> List[List[int]]:
    np.random.seed(random_seed)
    random.seed(random_seed)
    samples = []
    if sampling_method == 'monte_carlo':
        for _ in range(num_samples):
            sample = [random.randint(0, 1) for _ in range(dimensions)]
            samples.append(sample)
    elif sampling_method in ['sobol', 'orthogonal', 'latin_hypercube']:
        if sampling_method == 'sobol':
            sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_seed)
        elif sampling_method == 'orthogonal':
            sampler = qmc.LatinHypercube(d=dimensions, optimization="random-cd", seed=random_seed)
        else:
            sampler = qmc.LatinHypercube(d=dimensions, seed=random_seed)
        if sampling_method == 'sobol' and num_samples & (num_samples - 1) == 0:
            sample = sampler.random_base2(m=int(np.log2(num_samples)))
        else:
            sample = sampler.random(n=num_samples)
        binary_samples = (sample >= 0.5).astype(int)
        samples = binary_samples.tolist()
    elif sampling_method == 'stratified':
        strata = int(np.ceil(num_samples ** (1 / dimensions))) if dimensions > 0 else 1
        for _ in range(num_samples):
            sample = []
            for dim in range(dimensions):
                stratum = random.randint(0, strata - 1)
                sample.append(0 if stratum < strata / 2 else 1)
            samples.append(sample)
    elif sampling_method == 'covering_array':
        dim_pairs = list(combinations(range(dimensions), 2)) if dimensions > 1 else []
        required_pairs = set(product([0, 1], [0, 1]))
        selected = []
        remaining_pairs = required_pairs.copy()
        while len(selected) < num_samples and remaining_pairs:
            if dim_pairs and remaining_pairs:
                target_val1, target_val2 = random.choice(list(remaining_pairs))
                target_dims = random.choice(dim_pairs)
                candidate = [random.randint(0, 1) for _ in range(dimensions)]
                candidate[target_dims[0]] = target_val1
                candidate[target_dims[1]] = target_val2
                if (target_val1, target_val2) in remaining_pairs:
                    remaining_pairs.remove((target_val1, target_val2))
            else:
                candidate = [random.randint(0, 1) for _ in range(dimensions)]
            selected.append(candidate)
        while len(selected) < num_samples:
            selected.append([random.randint(0, 1) for _ in range(dimensions)])
        samples = selected[:num_samples]
    return samples

class SolutionWrapper(BinarySolution):
    def __init__(self, number_of_features: int):
        super().__init__(number_of_variables=number_of_features, number_of_objectives=2)
        self.variables = [0] * number_of_features
        self.objectives = [float('inf'), float('inf')]
        self.constraints = [0.0]
        self.uuid = uuid.uuid4()
        self.attributes = {
            'original_auc': float('-inf'),
            'original_featcount': float('inf'),
            'normalized_ft': 0.0,
            'normalized_fa': 0.0,
        }

class DefectDataManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.dataset_cache = {}
            cls._instance.problem_cache = {}
        return cls._instance
    @classmethod
    def preload_data(cls, dataset_paths: List[str], classifiers: Dict[str, Any],
                     random_seed: int = 42) -> Dict[str, Any]:
        instance = cls()
        for dataset_path in dataset_paths:
            if dataset_path not in instance.dataset_cache:
                try:
                    X, y, feature_names, _ = load_ant_data(dataset_path)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=random_seed, stratify=y
                    )
                    instance.dataset_cache[dataset_path] = (X_train, X_test, y_train, y_test, feature_names)
                    print(f"Successfully preloaded dataset: {os.path.basename(dataset_path)}")
                    print(f"  - features: {len(feature_names)}, train: {len(X_train)}, test: {len(X_test)}")
                except Exception as e:
                    print(f"Failed to load dataset {os.path.basename(dataset_path)}: {str(e)}")
                    continue
            X_train, X_test, y_train, y_test, feature_names = instance.dataset_cache[dataset_path]
            for clf_name, clf in classifiers.items():
                problem_key = (dataset_path, clf_name)
                if problem_key not in instance.problem_cache:
                    cloned_clf = clone(clf)
                    problem = AntDefectPredictionProblem(
                        X_train=X_train,
                        y_train=y_train,
                        classifier=cloned_clf,
                        mode='ft_fa',
                        seed=random_seed
                    )
                    instance.problem_cache[problem_key] = problem
                    print(f"  - Initialized problem instance: {clf_name}")
        print(f"\nPreload complete: {len(instance.dataset_cache)} datasets × {len(classifiers)} classifiers = {len(instance.problem_cache)} problem instances")
        return instance.problem_cache
    @classmethod
    def get_problem(cls, dataset_path: str, clf_name: str) -> AntDefectPredictionProblem:
        instance = cls()
        problem_key = (dataset_path, clf_name)
        if problem_key not in instance.problem_cache:
            raise ValueError(f"Problem instance not preloaded: dataset={os.path.basename(dataset_path)}, clf={clf_name}")
        return instance.problem_cache[problem_key]
    @classmethod
    def get_eval_data(cls, dataset_path: str) -> tuple:
        instance = cls()
        if dataset_path not in instance.dataset_cache:
            raise ValueError(f"Dataset not preloaded: {os.path.basename(dataset_path)}")
        return instance.dataset_cache[dataset_path]
    @classmethod
    def get_dataset_name(cls, dataset_path: str) -> str:
        return os.path.splitext(os.path.basename(dataset_path))[0]
    @classmethod
    def is_loaded(cls, dataset_path: str) -> bool:
        instance = cls()
        return dataset_path in instance.dataset_cache

def deduplicate_samples(sampled_data: List[SolutionWrapper]) -> List[SolutionWrapper]:
    seen = set()
    deduplicated = []
    for sample in sampled_data:
        feature_tuple = tuple(sample.variables)
        if feature_tuple not in seen:
            seen.add(feature_tuple)
            deduplicated.append(sample)
    print(f"Deduplication complete: original {len(sampled_data)} -> deduplicated {len(deduplicated)} (reused NSGA2 logic)")
    return deduplicated

SAMPLING_METHODS = [
    'sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'
]

def generate_samples(problem: AntDefectPredictionProblem, dataset_path: str, num_samples: int,
                     random_seed: int, sampling_method: str = 'monte_carlo', debug=False) -> List[SolutionWrapper]:
    np.random.seed(random_seed)
    random.seed(random_seed)
    sampled_solutions = []
    X_train_problem = problem.X_train
    number_of_features = X_train_problem.shape[1] if X_train_problem.ndim > 1 else 0
    X_train, X_test, y_train, y_test, feature_names = DefectDataManager.get_eval_data(dataset_path)
    if debug:
        print(f"\n[Sampling start] target_samples: {num_samples}, features: {number_of_features}")
        print(f"[Sampling method] {sampling_method}, classifier: {problem.classifier.__class__.__name__}")
        print(f"[Data shapes] train: {X_train.shape}, test: {X_test.shape} (reused NSGA2 data)")
        print(f"[Constraints] at least 1 feature must be selected, main objective: test AUC (reused NSGA2 evaluation)")
        print(f"[UUID type] SolutionWrapper.uuid = {type(SolutionWrapper(number_of_features).uuid)}")
    binary_samples = generate_binary_samples(
        dimensions=number_of_features,
        num_samples=num_samples * 2,
        sampling_method=sampling_method,
        random_seed=random_seed
    )
    valid_samples = []
    discarded_samples = 0
    evaluation_count = 0
    for sample in binary_samples:
        if len(sample) != number_of_features:
            print(f"Skipping invalid sample (length mismatch: {len(sample)} vs {number_of_features})")
            discarded_samples += 1
            continue
        if sum(sample) == 0:
            discarded_samples += 1
            if debug and discarded_samples % 50 == 0:
                print(f"Discarded {discarded_samples} empty solutions (no features selected)")
            continue
        solution = SolutionWrapper(number_of_features)
        try:
            if not hasattr(solution, 'constraints') or len(solution.constraints) == 0:
                solution.constraints = [0.0]
            solution.variables = sample
            problem.evaluate(solution)
            evaluation_count += 1
            test_perf = evaluate_solution_on_test(
                solution=solution,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                classifier=clone(problem.classifier),
                feature_names=feature_names
            )
            test_auc = test_perf['test_auc']
            feature_count = test_perf['feature_count']
            solution.attributes['original_auc'] = -test_auc
            solution.attributes['original_featcount'] = feature_count
            solution.objectives = [solution.attributes['original_auc'], solution.attributes['original_featcount']]
            if any(math.isinf(obj) for obj in solution.objectives):
                discarded_samples += 1
                continue
            valid_samples.append(solution)
            if len(valid_samples) >= num_samples:
                break
        except AttributeError as ae:
            print(f"UUID error while processing sample: {str(ae)}")
            print(f"  - solution.uuid type: {type(solution.uuid) if hasattr(solution, 'uuid') else 'missing'}")
            print(f"  - solution.uuid value: {solution.uuid if hasattr(solution, 'uuid') else 'missing'}")
            print(f"  - NSGA2 expectation: uuid.uuid4() UUID object")
            discarded_samples += 1
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            discarded_samples += 1
    valid_samples = deduplicate_samples(valid_samples)
    final_count = min(num_samples, len(valid_samples))
    if final_count % 10 != 0 and final_count > 0:
        final_count = (final_count // 10) * 10
    final_solutions = valid_samples[:final_count]
    if debug:
        print(f"\n[Sampling completed] valid_samples: {len(valid_samples)}, final retained: {len(final_solutions)}")
        print(f"evaluation_count: {evaluation_count}, discarded_samples: {discarded_samples}")
        if final_solutions:
            avg_test_auc = -np.mean([s.attributes['original_auc'] for s in final_solutions])
            avg_feat = np.mean([s.attributes['original_featcount'] for s in final_solutions])
            print(f"Sample stats: avg test AUC={avg_test_auc:.4f}, avg feature count={avg_feat:.1f}")
    return final_solutions

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

def save_sampled_data_to_csv(sampled_solutions: List[SolutionWrapper],
                             feature_names: List[str], dataset_name: str, clf_name: str,
                             sampling_method: str, num_samples: int, random_seed: int,
                             figure_type: str, mode: str,
                             reverse: bool = False) -> None:
    if mode == 'g1':
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
    header = feature_names + [
        'original_auc', 'original_featcount', 'normalized_ft', 'normalized_fa'
    ]
    rows = []
    for sol in sampled_solutions:
        row = sol.variables + [
            sol.attributes['original_auc'],
            sol.attributes['original_featcount'],
            sol.attributes['normalized_ft'],
            sol.attributes['normalized_fa']
        ]
        rows.append(row)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[Saved] {filename}")

def load_sampled_data_from_csv(dataset_name: str, clf_name: str, mode: str,
                               sampling_method: str, num_samples: int, random_seed: int,
                               figure_type: str, reverse: bool = False) -> List[SolutionWrapper]:
    full_dataset_name = f"{dataset_name}_{clf_name}"
    if reverse:
        filename = f"./Results/Samples_multi/sampled_data_{full_dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}_reverse.csv"
    else:
        filename = f"./Results/Samples_multi/sampled_data_{full_dataset_name}_{mode}_{sampling_method}_{num_samples}_{random_seed}_{figure_type}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Sampled data not found: {filename}")
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        number_of_features = len(header) - 4
        sampled_solutions = []
        for row in reader:
            variables = list(map(int, row[:number_of_features]))
            original_auc = float(row[number_of_features])
            original_featcount = float(row[number_of_features + 1])
            normalized_ft = float(row[number_of_features + 2])
            normalized_fa = float(row[number_of_features + 3])
            sol = SolutionWrapper(number_of_features)
            sol.variables = variables
            sol.objectives = [original_auc, original_featcount]
            sol.attributes['original_auc'] = original_auc
            sol.attributes['original_featcount'] = original_featcount
            sol.attributes['normalized_ft'] = normalized_ft
            sol.attributes['normalized_fa'] = normalized_fa
            sampled_solutions.append(sol)
    print(f"[Loaded] {filename}, samples: {len(sampled_solutions)}")
    return sampled_solutions

def process_g1_mode(problem: AntDefectPredictionProblem, dataset_path: str, clf_name: str,
                       minimize, num_samples, random_seed, sampling_method, sample_type,
                       dataset_name: str, mode, unique_id, reverse=False, first_sample=False):
    if mode != 'g1':
        raise PermissionError(f"Only g1 mode is allowed to save data, current mode: {mode}")
    X_train, X_test, y_train, y_test, feature_names = DefectDataManager.get_eval_data(dataset_path)
    number_of_features = len(feature_names)
    if first_sample:
        sample_and_save(problem, dataset_path, clf_name, sampling_method, num_samples, random_seed, mode, reverse)
        return
    try:
        base_dataset_name = DefectDataManager.get_dataset_name(dataset_path)
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name=base_dataset_name, clf_name=clf_name, mode='g1',
            sampling_method=sampling_method, num_samples=num_samples, random_seed=random_seed,
            figure_type='figure1', reverse=reverse
        )
    except FileNotFoundError:
        print(f"[g1] Sampled data not found: {dataset_name}, method={sampling_method}, samples={num_samples}, seed={random_seed}")
        print("Run this program with first_sample=True to generate sampled CSV first.")
        return
    sampled_dict = {
        tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
        for s in sampled_solutions
    }
    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples, dataset_name,
        mode, unique_id, sample_type, sampled_dict, reverse, None, number_of_features
    )
    try:
        sampled_solutions_fig2 = load_sampled_data_from_csv(
            dataset_name=base_dataset_name, clf_name=clf_name, mode='g1',
            sampling_method=sampling_method, num_samples=num_samples, random_seed=random_seed,
            figure_type='figure2', reverse=reverse
        )
        sampled_dict_fig2 = {
            tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions_fig2
        }
    except FileNotFoundError:
        r0_points = np.column_stack(([s.attributes['normalized_ft'] for s in sampled_solutions],
                                     [s.attributes['normalized_fa'] for s in sampled_solutions]))
        g1, g2 = transform_points_for_figure2(r0_points)
        for i, s in enumerate(sampled_solutions):
            s.attributes['normalized_ft'] = g1[i]
            s.attributes['normalized_fa'] = g2[i]
        sampled_dict_fig2 = {
            tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
            for s in sampled_solutions
        }
    plot_figure2(
        random_seed, 'mean', sampling_method, num_samples, dataset_name,
        mode, unique_id, sample_type, sampled_dict_fig2, reverse, None, number_of_features
    )

def sample_and_save(problem: AntDefectPredictionProblem, dataset_path: str, clf_name: str,
                    sampling_method: str, num_samples: int, random_seed: int, mode: str, reverse: bool = False):
    sampled_solutions = generate_samples(
        problem=problem,
        dataset_path=dataset_path,
        num_samples=num_samples,
        random_seed=random_seed,
        sampling_method=sampling_method,
        debug=True
    )
    if not sampled_solutions:
        print(f"[sample_and_save] Warning: no valid samples generated: {dataset_path}, seed={random_seed}")
        return
    ft_list = [s.attributes['original_auc'] for s in sampled_solutions]
    fa_list = [s.attributes['original_featcount'] for s in sampled_solutions]
    scaler = MinMaxScaler()
    if len(set(ft_list)) == 1:
        ft_list = [v + 1e-10 * i for i, v in enumerate(ft_list)]
    if len(set(fa_list)) == 1:
        fa_list = [v + 1e-10 * i for i, v in enumerate(fa_list)]
    normalized_data = scaler.fit_transform(np.column_stack((ft_list, fa_list)))
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = normalized_data[i, 0]
        sol.attributes['normalized_fa'] = normalized_data[i, 1]
    base_dataset_name = DefectDataManager.get_dataset_name(dataset_path)
    dataset_name_with_clf = f"{base_dataset_name}_{clf_name}"
    X_train, X_test, y_train, y_test, feature_names = DefectDataManager.get_eval_data(dataset_path)
    save_sampled_data_to_csv(
        sampled_solutions=sampled_solutions,
        feature_names=feature_names,
        dataset_name=dataset_name_with_clf,
        clf_name=clf_name,
        sampling_method=sampling_method,
        num_samples=num_samples,
        random_seed=random_seed,
        figure_type='figure1',
        mode='g1',
        reverse=reverse
    )
    r0_points = normalized_data
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(sampled_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]
    save_sampled_data_to_csv(
        sampled_solutions=sampled_solutions,
        feature_names=feature_names,
        dataset_name=dataset_name_with_clf,
        clf_name=clf_name,
        sampling_method=sampling_method,
        num_samples=num_samples,
        random_seed=random_seed,
        figure_type='figure2',
        mode='g1',
        reverse=reverse
    )
    print(f"[sample_and_save] Sampling and saving completed: {dataset_name_with_clf}, method={sampling_method}, samples={num_samples}, seed={random_seed}")


def process_fa_construction_mode(problem: AntDefectPredictionProblem, dataset_path: str, clf_name: str,
                                 minimize, num_samples, random_seed, fa_construction_mode,
                                 sampling_method, sample_type, dataset_name: str, mode, unique_id,
                                 reverse=False, first_sample=False):
    if first_sample:
        sample_and_save(problem, dataset_path, clf_name, sampling_method, num_samples, random_seed, mode, reverse)
        return
    base_dataset_name = DefectDataManager.get_dataset_name(dataset_path)
    try:
        sampled_solutions = load_sampled_data_from_csv(
            dataset_name=base_dataset_name,
            clf_name=clf_name,
            mode='g1',
            sampling_method=sampling_method,
            num_samples=num_samples,
            random_seed=random_seed,
            figure_type='figure1',
            reverse=reverse
        )
        print(f"Loaded data from g1 mode: {dataset_name}")
    except FileNotFoundError:
        print(f"[FA mode] g1 base sampled data not found: {dataset_name}, {sampling_method}, {random_seed}")
        print("Run this program with first_sample=True to generate sampled CSV first.")
        return
    if len(sampled_solutions) % 10 != 0:
        sorted_solutions = sorted(
            sampled_solutions,
            key=lambda x: (x.attributes['original_auc'], x.attributes['original_featcount'])
        )
        target_size = (len(sorted_solutions) // 10) * 10
        sampled_solutions = sorted_solutions[:target_size]
    sorted_solutions = sorted(
        sampled_solutions,
        key=lambda x: (x.attributes['original_auc'], x.attributes['original_featcount'])
    )
    sorted_indices = list(range(len(sorted_solutions)))
    batch_size = 20
    num_batches = (len(sorted_solutions) + batch_size - 1) // batch_size
    batch_indices = get_batch_indices(sorted_indices, batch_size, num_batches)
    t = 1
    t_max = 30
    novelty_archive = []
    all_ft_normalized = []
    all_fa_normalized = []
    age_info = None

    # 获取特征名称
    X_train, X_test, y_train, y_test, feature_names = DefectDataManager.get_eval_data(dataset_path)

    for batch_num in range(num_batches):
        batch_solutions = [sorted_solutions[i] for i in batch_indices[batch_num]]
        batch_ft = [s.attributes['original_auc'] for s in batch_solutions]
        batch_configs = [s.variables for s in batch_solutions]
        if batch_configs:
            unique_elements_per_column = []
            for col in range(len(batch_configs[0])):
                unique_elements = set()
                for config in batch_configs:
                    unique_elements.add(config[col])
                unique_elements_per_column.append(sorted(unique_elements))
        else:
            unique_elements_per_column = []
        if fa_construction_mode == 'age':
            if batch_num == 0:
                age_info = [i + 1 for i in range(len(batch_solutions))]
            else:
                base_age = batch_size + t - 1
                age_info = [base_age] * len(batch_solutions)
        elif fa_construction_mode == 'novelty':
            update_novelty_archive(batch_solutions, novelty_archive)
        batch_ft_normalized, batch_fa_normalized = generate_fa(
            batch_configs,
            batch_ft,
            fa_construction_mode,
            minimize,
            unique_elements_per_column,
            t,
            t_max,
            random_seed,
            age_info if fa_construction_mode == 'age' else None,
            novelty_archive if fa_construction_mode == 'novelty' else None,
            min(10, len(batch_solutions) // 2)
        )
        all_ft_normalized.extend(batch_ft_normalized)
        all_fa_normalized.extend(batch_fa_normalized)
        t += 1

    for i, sol in enumerate(sorted_solutions):
        sol.attributes['normalized_ft'] = all_ft_normalized[i]
        sol.attributes['normalized_fa'] = all_fa_normalized[i]

    sampled_dict = {
        tuple(s.variables): (s.attributes['normalized_ft'], s.attributes['normalized_fa'])
        for s in sorted_solutions
    }


    save_sampled_data_to_csv(
        sampled_solutions=sorted_solutions,
        feature_names=feature_names,
        dataset_name=dataset_name,
        clf_name=clf_name,
        sampling_method=sampling_method,
        num_samples=num_samples,
        random_seed=random_seed,
        figure_type='figure1',
        mode=mode,
        reverse=reverse
    )

    plot_figure1(
        random_seed, 'mean', sampling_method, num_samples, dataset_name,
        mode, unique_id, sample_type, sampled_dict, reverse, None, None
    )

    r0_points = np.column_stack((all_ft_normalized, all_fa_normalized))
    g1, g2 = transform_points_for_figure2(r0_points)
    for i, sol in enumerate(sorted_solutions):
        sol.attributes['normalized_ft'] = g1[i]
        sol.attributes['normalized_fa'] = g2[i]

    sampled_dict_fig2 = {
        tuple(s.variables): (g1[i], g2[i])
        for i, s in enumerate(sorted_solutions)
    }

    save_sampled_data_to_csv(
        sampled_solutions=sorted_solutions,
        feature_names=feature_names,
        dataset_name=dataset_name,
        clf_name=clf_name,
        sampling_method=sampling_method,
        num_samples=num_samples,
        random_seed=random_seed,
        figure_type='figure2',
        mode=mode,
        reverse=reverse
    )

    plot_figure2(
        random_seed, 'mean', sampling_method, num_samples, dataset_name,
        mode, unique_id, sample_type, sampled_dict_fig2, reverse, None, None
    )

def init_worker():
    pass

def transform_points_for_figure2(r0_points):
    g1 = r0_points[:, 1] + r0_points[:, 0]
    g2 = r0_points[:, 1] - r0_points[:, 0]
    return g1, g2

def process_single_task(mode, dataset_path, clf_name, sampling_method, num_samples, sample_type,
                        minimize, random_seed, fa_construction_list, unique_id, reverse,
                        first_sample=False):
    try:
        problem = DefectDataManager.get_problem(dataset_path, clf_name)
        problem.mode = mode
        base_dataset_name = DefectDataManager.get_dataset_name(dataset_path)
        dataset_name = f"{base_dataset_name}_{clf_name}"
        np.random.seed(random_seed)
        random.seed(random_seed)
        if mode == 'g1':
            process_g1_mode(
                problem=problem,
                dataset_path=dataset_path,
                clf_name=clf_name,
                minimize=minimize,
                num_samples=num_samples,
                random_seed=random_seed,
                sampling_method=sampling_method,
                sample_type=sample_type,
                dataset_name=dataset_name,
                mode=mode,
                unique_id=unique_id,
                reverse=reverse,
                first_sample=first_sample
            )
        elif mode in fa_construction_list:
            process_fa_construction_mode(
                problem=problem,
                dataset_path=dataset_path,
                clf_name=clf_name,
                minimize=minimize,
                num_samples=num_samples,
                random_seed=random_seed,
                fa_construction_mode=mode,
                sampling_method=sampling_method,
                sample_type=sample_type,
                dataset_name=dataset_name,
                mode=mode,
                unique_id=unique_id,
                reverse=reverse,
                first_sample=first_sample
            )
        return f"Task completed: {unique_id}"
    except Exception as e:
        return f"Task failed: {unique_id}, Error: {str(e)[:200]}"

def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)
    total_tasks = len(all_tasks)
    print(f"\nStarting batched processing: total_tasks={total_tasks}, batch_size={batch_size}, max_workers={max_workers}")
    for i in range(0, total_tasks, batch_size):
        current_batch = all_tasks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_tasks + batch_size - 1) // batch_size
        print(f"\n=== Processing batch {batch_num}/{total_batches} (tasks in batch: {len(current_batch)}) ===")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker
        ) as executor:
            futures = [executor.submit(process_single_task, **task) for task in current_batch]
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                task_idx_in_batch = i + idx + 1
                try:
                    result = future.result()
                    progress = (task_idx_in_batch / total_tasks) * 100
                    print(f"Progress: {task_idx_in_batch}/{total_tasks} ({progress:.1f}%) | {result}")
                except Exception as e:
                    print(f"Progress: {task_idx_in_batch}/{total_tasks} | Task exception: {str(e)[:150]}")

def main_sdp_multi(
        dataset_paths: List[str],
        fa_construction_list: List[str],
        classifiers: Dict[str, Any],
        minimize: bool = True,
        fixed_sample_sizes: List[int] = None,
        sampling_methods: List[str] = None,
        random_seeds: List[int] = None,
        use_multiprocessing: bool = True,
        max_workers: int = None,
        reverse: bool = False,
        first_sample: bool = False,
        data_base_path: str = "./dataset/"
):
    if fixed_sample_sizes is None:
        fixed_sample_sizes = [1000]
    if sampling_methods is None:
        sampling_methods = ['sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array']
    if random_seeds is None:
        random_seeds = list(range(10))
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 5)
    print("=" * 80)
    print("Starting dataset preload")
    print("=" * 80)
    DefectDataManager.preload_data(dataset_paths, classifiers, random_seed=42)
    sample_configs = []
    for fixed_size in fixed_sample_sizes:
        sample_configs.append(('fixed', fixed_size))
    if first_sample:
        fa_construction_list = ["g1"]
    all_tasks = []
    task_id = 0
    for dataset_path in dataset_paths:
        if not DefectDataManager.is_loaded(dataset_path):
            print(f"Skipping not-loaded dataset: {os.path.basename(dataset_path)}")
            continue
        for clf_name in classifiers.keys():
            for sampling_method in sampling_methods:
                for sample_type, num_samples in sample_configs:
                    for random_seed in random_seeds:
                        for mode in fa_construction_list:
                            task_id += 1
                            base_dataset_name = DefectDataManager.get_dataset_name(dataset_path)
                            unique_id = f"task_{task_id}_{base_dataset_name}_{clf_name}_{mode}_{sampling_method}_fixed_{num_samples}_seed_{random_seed}"
                            task = {
                                'mode': mode,
                                'dataset_path': dataset_path,
                                'clf_name': clf_name,
                                'sampling_method': sampling_method,
                                'num_samples': num_samples,
                                'sample_type': sample_type,
                                'minimize': minimize,
                                'random_seed': random_seed,
                                'fa_construction_list': fa_construction_list,
                                'unique_id': unique_id,
                                'reverse': reverse,
                                'first_sample': first_sample
                            }
                            all_tasks.append(task)
    batch_size = max_workers
    print("\n" + "=" * 80)
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Max workers: {max_workers}, tasks per batch: {batch_size}")
    print("=" * 80)
    if use_multiprocessing and all_tasks:
        print(f"\nStarting multiprocessing batched processing...")
        process_in_batches(
            all_tasks=all_tasks,
            max_workers=max_workers,
            batch_size=batch_size
        )
    elif all_tasks:
        print(f"\nStarting single-process processing...")
        for idx, task in enumerate(all_tasks):
            result = process_single_task(**task)
            progress = (idx + 1) / len(all_tasks) * 100
            print(f"Progress: {idx + 1}/{len(all_tasks)} ({progress:.1f}%) | {result}")
    print("\n" + "=" * 80)
    print("All tasks completed")
    print("=" * 80)

if __name__ == "__main__":
    DATASET_NAMES = [
        'ant-1.7', 'camel-1.6', 'ivy-2.0', 'jedit-4.0', 'lucene-2.4',
        'poi-3.0', 'synapse-1.2', 'velocity-1.6', 'xalan-2.6', 'xerces-1.4'
    ]
    DATASET_PATHS = [f'../Datasets/{name}.csv' for name in DATASET_NAMES]
    CLASSIFIERS = {
        "J48": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "NB": GaussianNB()
    }
    FA_CONSTRUCTION_LIST = [
        'g1','penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity'
    ]
    FIXED_SAMPLE_SIZES = [1000]
    SAMPLING_METHODS = [
        'sobol', 'orthogonal', 'stratified', 'latin_hypercube', 'monte_carlo', 'covering_array'
    ]
    USE_MULTIPROCESSING = True
    MAX_WORKERS = min(multiprocessing.cpu_count(), 50)
    REVERSE = False
    main_sdp_multi(
        dataset_paths=DATASET_PATHS,
        fa_construction_list=FA_CONSTRUCTION_LIST,
        classifiers=CLASSIFIERS,
        fixed_sample_sizes=FIXED_SAMPLE_SIZES,
        sampling_methods=SAMPLING_METHODS,
        use_multiprocessing=True,
        max_workers=MAX_WORKERS,
        reverse=REVERSE,
        first_sample=False
    )