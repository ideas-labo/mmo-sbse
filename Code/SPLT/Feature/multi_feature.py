import csv
import os
import numpy as np
import random
import multiprocessing
from typing import List, Dict, Tuple, Any
import sys
import numpy as np
import random
sys.path.append('../')
sys.path.append('../..')
sys.path.insert(0, '/mnt/sdaDisk/ccj/code/mmo')
sys.path.insert(0, 'home/ccj/code/mmo')
from Code.SPLT.Feature.utils.multi_feature_compute import  plot_figure1, plot_figure2

class JavaIndividual:
    def __init__(self, products: List[List[int]], raw_products: List[List[int]]):
        self.products = products
        self.raw_products = raw_products
        self.originalObjectives = [float('inf'), float('inf')]
        self.normalizedObjectives = [0.0, 0.0]

def get_java_sample_csv_path(
        dataset_name: str,
        mode: str,
        sampling_method: str,
        sample_size: int,
        seed: int,
        figure_type: str
) -> str:
    base_dir = "./Results/Samples_multi/"
    filename = f"sampled_data_{dataset_name}_{mode}_{sampling_method}_{sample_size}_seed_{seed}_{figure_type}.csv"
    return os.path.join(base_dir, filename)

def parse_java_feature_columns(header: List[str]) -> Dict[Tuple[int, int], int]:
    feature_map = {}
    for col_idx, col_name in enumerate(header):
        if col_name.startswith("Feature_"):
            parts = col_name.split("_")
            if len(parts) == 3 and parts[0] == "Feature":
                try:
                    p = int(parts[1])
                    f = int(parts[2])
                    feature_map[(p, f)] = col_idx
                except ValueError:
                    continue
    return feature_map

def load_java_individuals_from_csv(
        csv_path: str,
        fill_to_max: bool = True
) -> List[JavaIndividual]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Java CSV file not found: {csv_path}")

    individuals = []
    raw_products_list = []
    all_feature_counts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        feature_map = parse_java_feature_columns(header)
        if not feature_map:
            raise ValueError(f"CSV {csv_path} does not contain valid feature columns (expected Feature_p_f)")

        all_p = {p for (p, f) in feature_map.keys()}
        all_f = {f for (p, f) in feature_map.keys()}
        max_p_in_header = max(all_p) if all_p else 0
        max_f_in_header = max(all_f) if all_f else 0

        for row_idx, row in enumerate(reader, 1):
            if len(row) < len(header):
                print(f"Warning: Row {row_idx} has insufficient columns (expected {len(header)}, got {len(row)}), skipping")
                continue

            raw_products = []
            for p in range(1, max_p_in_header + 1):
                product_features = []
                for (feat_p, feat_f), col_idx in feature_map.items():
                    if feat_p == p and col_idx < len(row):
                        val_str = row[col_idx].strip()
                        if val_str and val_str != "0":
                            try:
                                product_features.append(int(val_str))
                            except ValueError:
                                pass
                if product_features:
                    raw_products.append(product_features)
                    all_feature_counts.append(len(product_features))
            raw_products_list.append(raw_products)

            feature_cols_count = len(feature_map)
            if feature_cols_count + 3 >= len(row):
                print(f"Warning: Row {row_idx} missing target columns, skipping")
                continue

            original_ft = float(row[feature_cols_count].strip())
            original_fa = float(row[feature_cols_count + 1].strip())
            normalized_ft = float(row[feature_cols_count + 2].strip())
            normalized_fa = float(row[feature_cols_count + 3].strip())

            individuals.append({
                "raw_products": raw_products,
                "originalObjectives": [original_ft, original_fa],
                "normalizedObjectives": [normalized_ft, normalized_fa]
            })

        max_products = max(len(p_list) for p_list in raw_products_list) if raw_products_list else 0
        max_features = max(all_feature_counts) if all_feature_counts else max_f_in_header
        if max_products == 0 or max_features == 0:
            raise ValueError(f"No valid product/feature data read from CSV {csv_path}")

        unified_individuals = []
        for ind_data in individuals:
            raw_products = ind_data["raw_products"]
            unified_products = []
            for raw_prod in raw_products:
                filled_prod = raw_prod + [0] * (max_features - len(raw_prod))
                unified_products.append(filled_prod)

            if fill_to_max and len(unified_products) < max_products:
                need_fill = max_products - len(unified_products)
                for _ in range(need_fill):
                    unified_products.append([0] * max_features)

            unified_ind = JavaIndividual(
                products=unified_products,
                raw_products=raw_products
            )
            unified_ind.originalObjectives = ind_data["originalObjectives"]
            unified_ind.normalizedObjectives = ind_data["normalizedObjectives"]
            unified_individuals.append(unified_ind)

    print(f"Loaded and unified from {csv_path}: {len(unified_individuals)} individuals, unified to {max_products} products x {max_features} features each")
    return unified_individuals

def process_g1_g2_mode(
        dataset_name: str,
        sampling_method: str,
        sample_size: int,
        seed: int,
        unique_id: str,
        reverse: bool
) -> List[JavaIndividual]:
    fig1_csv = get_java_sample_csv_path(dataset_name, "g1_g2", sampling_method, sample_size, seed, "figure1")
    fig2_csv = get_java_sample_csv_path(dataset_name, "g1_g2", sampling_method, sample_size, seed, "figure2")

    fig1_individuals = load_java_individuals_from_csv(fig1_csv)
    fig2_individuals = load_java_individuals_from_csv(fig2_csv)

    if len(fig1_individuals) != len(fig2_individuals):
        raise ValueError(f"g1_g2 mode data mismatch: figure1={len(fig1_individuals)}, figure2={len(fig2_individuals)}")

    fig1_sampled_dict = {
        tuple(tuple(product) for product in ind.products): (ind.normalizedObjectives[0], ind.normalizedObjectives[1])
        for ind in fig1_individuals
    }
    fig2_sampled_dict = {
        tuple(tuple(product) for product in ind.products): (ind.normalizedObjectives[0], ind.normalizedObjectives[1])
        for ind in fig2_individuals
    }

    plot_figure1(
        seed, "mean", sampling_method, sample_size, dataset_name,
        "g1_g2", unique_id, "fixed", fig1_sampled_dict,
        reverse, None, None
    )
    plot_figure2(
        seed, "mean", sampling_method, sample_size, dataset_name,
        "g1_g2", unique_id, "fixed", fig2_sampled_dict,
        reverse, None, None
    )

    return fig1_individuals

def process_fa_construction_mode(
        dataset_name: str,
        mode: str,
        sampling_method: str,
        sample_size: int,
        seed: int,
        unique_id: str,
        reverse: bool
) -> List[JavaIndividual]:
    fig1_csv = get_java_sample_csv_path(dataset_name, mode, sampling_method, sample_size, seed, "figure1")
    fig2_csv = get_java_sample_csv_path(dataset_name, mode, sampling_method, sample_size, seed, "figure2")

    fig1_individuals = load_java_individuals_from_csv(fig1_csv)
    fig2_individuals = load_java_individuals_from_csv(fig2_csv)

    if len(fig1_individuals) != len(fig2_individuals):
        raise ValueError(f"{mode} mode data mismatch: figure1={len(fig1_individuals)}, figure2={len(fig2_individuals)}")

    fig1_sampled_dict = {
        tuple(tuple(product) for product in ind.products): (ind.normalizedObjectives[0], ind.normalizedObjectives[1])
        for ind in fig1_individuals
    }
    fig2_sampled_dict = {
        tuple(tuple(product) for product in ind.products): (ind.normalizedObjectives[0], ind.normalizedObjectives[1])
        for ind in fig2_individuals
    }

    plot_figure1(
        seed, "mean", sampling_method, sample_size, dataset_name,
        mode, unique_id, "fixed", fig1_sampled_dict,
        reverse, None, None
    )
    plot_figure2(
        seed, "mean", sampling_method, sample_size, dataset_name,
        mode, unique_id, "fixed", fig2_sampled_dict,
        reverse, None, None
    )

    return fig1_individuals

def init_worker():
    pass
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_in_batches(all_tasks, max_workers=2, batch_size=2):
    max_workers = min(multiprocessing.cpu_count(), max_workers)

    total_tasks = len(all_tasks)
    for i in range(0, total_tasks, batch_size):
        batch = all_tasks[i:i + batch_size]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_task, task) for task in batch]

            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f"Task error: {str(e)}")

def process_single_task(task: Dict[str, Any]) -> str:
    try:
        mode = task["mode"]
        dataset_name = task["dataset_name"]
        sampling_method = task["sampling_method"]
        sample_size = task["sample_size"]
        seed = task["seed"]
        unique_id = task["unique_id"]
        reverse = task["reverse"]

        np.random.seed(seed)
        random.seed(seed)

        if mode == "g1_g2":
            process_g1_g2_mode(dataset_name, sampling_method, sample_size, seed, unique_id, reverse)
        else:
            process_fa_construction_mode(dataset_name, mode, sampling_method, sample_size, seed, unique_id, reverse)

        return f"✅ Done: {unique_id}"
    except Exception as e:
        return f"❌ Failed: {unique_id}, error: {str(e)}"

def main_splt_multi(
        dataset_names=None,
        fa_construction=None,
        minimize=True,
        fixed_sample_sizes=None,
        percentage_sample_sizes=None,
        sampling_methods=None,
        random_seeds=None,
        use_multiprocessing=True,
        max_workers=None,
        reverse=False,
        workflow_base_path='../Datasets/',
        use_saved_data=False,
        debug=False,
        first_sample=False
):
    if dataset_names is None:
        dataset_names = ["7z", "Amazon", "BerkeleyDBC", "CocheEcologico", "CounterStrikeSimpleFeatureModel",
                         "DSSample", "Dune", "ElectronicDrum", "HiPAcc", "Drupal",
                         "JavaGC", "JHipster", "lrzip", "ModelTransformation",
                         "SmartHomev2.2", "SPLSSimuelESPnP", "VideoPlayer",
                         "VP9", "WebPortal", "x264", 'Polly']

    if fa_construction is None:
        fa_construction = ['g1_g2', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']

    if fixed_sample_sizes is None:
        fixed_sample_sizes = [1000]

    if percentage_sample_sizes is None:
        percentage_sample_sizes = [10, 20, 30, 40, 50]

    if sampling_methods is None:
        sampling_methods = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"]

    if random_seeds is None:
        random_seeds = range(10)

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 5)

    if first_sample:
        print("Warning: SPLT has no sampling implementation. Cannot generate sampled data. Ensure sampled CSVs exist.")
        print(f"Sampled data should be stored under: ./Results/Samples_multi/")
        print(f"Filename pattern: sampled_data_{{dataset}}_{{mode}}_{{sampling_method}}_{{sample_size}}_seed_{{seed}}_figure{{1/2}}.csv")
        return False

    all_modes = fa_construction
    all_tasks = []

    for dataset in dataset_names:
        for mode in all_modes:
            for method in sampling_methods:
                for sample_size in fixed_sample_sizes:
                    for seed in random_seeds:
                        unique_id = f"{dataset}_{mode}_{method}_{sample_size}_seed_{seed}_reverse_{reverse}"
                        all_tasks.append({
                            "mode": mode,
                            "dataset_name": dataset,
                            "sampling_method": method,
                            "sample_size": sample_size,
                            "seed": seed,
                            "unique_id": unique_id,
                            "reverse": reverse
                        })

    print(f"=== Task summary ===")
    print(f"Total tasks: {len(all_tasks)} | Max workers: {max_workers}")
    print(f"Note: SPLT depends on existing sampled data")

    if use_multiprocessing:
        process_in_batches(all_tasks, max_workers=max_workers, batch_size=max_workers)
    else:
        print("=== Single-process mode ===")
        for task in all_tasks:
            print(process_single_task(task))

    return True

if __name__ == "__main__":
    DATASETS = ["7z","Amazon","BerkeleyDBC","CocheEcologico","CounterStrikeSimpleFeatureModel",
                "DSSample","Dune","ElectronicDrum","HiPAcc","Drupal",
                "JavaGC","JHipster","lrzip","ModelTransformation",
                "SmartHomev2.2","SPLSSimuelESPnP","VideoPlayer",
                "VP9","WebPortal","x264",'Polly']
    FA_CONSTRUCTION_MODES = ['g1_g2', 'penalty', 'gaussian', 'reciprocal', 'age', 'novelty', 'diversity']
    FIXED_SAMPLE_SIZES = [1000]
    SAMPLING_METHODS = ["sobol", "halton", "stratified", "latin_hypercube", "monte_carlo", "covering_array"]

    main_splt_multi(
        dataset_names=DATASETS,
        fa_construction=FA_CONSTRUCTION_MODES,
        fixed_sample_sizes=FIXED_SAMPLE_SIZES,
        sampling_methods=SAMPLING_METHODS,
        use_multiprocessing=True,
        max_workers=50,
        reverse=False
    )