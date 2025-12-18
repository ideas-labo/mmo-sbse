import os
import csv

import hnswlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.linear_model import LinearRegression
from jmetal.core.quality_indicator import HyperVolume
from jmetal.util.ranking import FastNonDominatedRanking
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import sys
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
import warnings
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')

def is_pareto_optimal_minimize(point, points):
    is_pareto = True
    for other_point in points:
        condition1 = other_point[0] <= point[0] and other_point[1] < point[1]
        condition2 = other_point[0] < point[0] and other_point[1] <= point[1]
        if condition1 or condition2:
            is_pareto = False
            break
    return is_pareto

import numpy as np
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import random

class ApproxNearestNeighbors:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.index.set_ef(50)
        self.data = []

    def add(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)

        start_idx = len(self.data)
        self.data.extend(vectors.tolist())

        if len(vectors) > 0:
            self.index.add_items(vectors, np.arange(start_idx, start_idx + len(vectors)))

    def search(self, query, k, max_candidates=None):
        if not self.data:
            return [], []

        query = np.array(query, dtype=np.float32).flatten()
        k = min(k, len(self.data))

        indices, distances = self.index.knn_query(query, k=k)

        if k == 1:
            return [tuple(self.data[indices[0][0]])], [distances[0][0]]

        nearest_configs = [tuple(self.data[i]) for i in indices[0]]
        return nearest_configs, distances[0].tolist()

    def __len__(self):
        return len(self.data)

def get_adaptive_k(dimension):
    if dimension <= 200:
        return 20
    elif dimension <= 500:
        return 10
    else:
        return 5

class NeighborFinder:
    def __init__(self):
        self.index = None

    def generate_neighbor_solutions(self, solution, sampled_data, random_seed=None):
        if self.index is None:
            self.index = ApproxNearestNeighbors(len(solution))
            self.index.add(np.array(list(sampled_data)))

        dim = len(solution)
        k = get_adaptive_k(dim)

        neighbors, _ = self.index.search(solution, k)
        return list(neighbors)

neighbor_finder = NeighborFinder()

def generate_neighbor_solutions(solution, sampled_data, random_seed=None):
    return neighbor_finder.generate_neighbor_solutions(solution, sampled_data, random_seed)

def find_local_efficient_points(decision_vars, objective_values, sampled_data,
                               max_values_per_dim=5, random_seed=None,
                               dag=None, num_valid_tasks=None):
    if len(decision_vars) < 2:
        return np.array([])

    var_to_obj = {tuple(v): obj for v, obj in zip(decision_vars, objective_values)}
    sampled_set = set(tuple(v) for v in sampled_data)

    local_efficient = []
    for i in range(len(decision_vars)):
        current_var = tuple(decision_vars[i])
        current_obj = objective_values[i]
        is_local_efficient = True

        neighbors = generate_neighbor_solutions(current_var, sampled_set, random_seed)

        for neighbor_var in neighbors:
            neighbor_obj = var_to_obj[neighbor_var]
            if (neighbor_obj[0] <= current_obj[0] and neighbor_obj[1] < current_obj[1]) or \
               (neighbor_obj[0] < current_obj[0] and neighbor_obj[1] <= current_obj[1]):
                is_local_efficient = False
                break

        local_efficient.append(is_local_efficient)

    return np.where(local_efficient)[0]

def calculate_connected_components(decision_vars, local_efficient_indices,
                                   sampled_data, max_values_per_dim=5,
                                   random_seed=None, dag=None, num_valid_tasks=None):
    if len(local_efficient_indices) < 1:
        return 0

    le_decision_vars = [tuple(decision_vars[i]) for i in local_efficient_indices]
    sampled_set = set(tuple(v) for v in sampled_data)
    var_to_index = {var: idx for idx, var in enumerate(le_decision_vars)}

    G = nx.Graph()
    for var in le_decision_vars:
        G.add_node(var_to_index[var])

        neighbors = generate_neighbor_solutions(var, sampled_set, random_seed)
        for neighbor in neighbors:
            if neighbor in var_to_index:
                G.add_edge(var_to_index[var], var_to_index[neighbor])

    return len(list(nx.connected_components(G)))

def calculate_dominance_change_frequency(sampled_dict, max_values_per_dim=5, random_seed=None):
    if len(sampled_dict) < 2:
        return 0.0

    decision_vars = np.array(list(sampled_dict.keys()))
    objective_values = np.array(list(sampled_dict.values()))
    sampled_set = set(tuple(v) for v in decision_vars)

    dimension = len(decision_vars[0])
    distance_index = ApproxNearestNeighbors(dimension)
    distance_index.add(decision_vars)

    is_pareto_list = np.array([
        is_pareto_optimal_minimize(point, objective_values)
        for point in objective_values
    ])

    dominance_changes = 0
    total_pairs = 0

    for i in range(len(decision_vars)):
        current_var = tuple(decision_vars[i])
        current_is_pareto = is_pareto_list[i]

        neighbors, _ = distance_index.search(current_var, k=get_adaptive_k(dimension))

        for neighbor in neighbors:
            neighbor_indices = [idx for idx, var in enumerate(decision_vars)
                              if tuple(var) == neighbor]
            if not neighbor_indices:
                continue

            neighbor_is_pareto = is_pareto_list[neighbor_indices[0]]
            if current_is_pareto != neighbor_is_pareto:
                dominance_changes += 1
            total_pairs += 1

    return dominance_changes / total_pairs if total_pairs > 0 else 0.0

def find_local_efficient_points_literature(decision_vars, objective_values, sampled_data, max_values_per_dim=5,
                                           random_seed=None, prec_norm=1e-6, prec_angle=1e-6):
    if len(decision_vars) < 2:
        return np.array([])

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    var_to_obj = {tuple(v): obj for v, obj in zip(decision_vars, objective_values)}
    sampled_set = set(tuple(v) for v in sampled_data)

    local_efficient = []
    for i in range(len(decision_vars)):
        current_var = tuple(decision_vars[i])
        current_obj = objective_values[i]

        neighbors = generate_neighbor_solutions(current_var, sampled_set, random_seed)

        dominated = False
        for neighbor_var in neighbors:
            neighbor_obj = var_to_obj[neighbor_var]
            if (neighbor_obj[0] <= current_obj[0] and neighbor_obj[1] < current_obj[1]) or \
                    (neighbor_obj[0] < current_obj[0] and neighbor_obj[1] <= current_obj[1]):
                dominated = True
                break

        if not dominated:
            local_efficient.append(i)

    return np.array(local_efficient)

def find_center_point(r0_points, random_seed=42, center_type='mean'):
    if center_type == 'random':
        np.random.seed(random_seed)
        return r0_points[np.random.choice(len(r0_points))]
    elif center_type == 'mean':
        return np.mean(r0_points, axis=0)
    elif center_type == 'left_bottom_pareto':
        pareto_points = []
        for point in r0_points:
            if is_pareto_optimal_minimize(point, r0_points):
                pareto_points.append(point)
        pareto_points = np.array(pareto_points)
        if len(pareto_points) > 0:
            left_bottom_index = np.argmin(pareto_points[:, 0] + pareto_points[:, 1])
            center_point = pareto_points[left_bottom_index]
            return center_point
        else:
            np.random.seed(random_seed)
            return r0_points[np.random.choice(len(r0_points))]
    else:
        raise ValueError("center_type must be either 'random','mean' or 'left_bottom_pareto'")

def calculate_hypervolume(solutions, ref_point):
    hypervolume = HyperVolume(reference_point=ref_point)
    return hypervolume.compute(solutions)

def calculate_level_count(points):
    ranking = FastNonDominatedRanking()
    solutions = [type('MockSolution', (object,), {'objectives': point, 'attributes': {}}) for point in points]
    ranking.compute_ranking(solutions)
    return ranking.get_number_of_subfronts()

def calculate_objective_variance(points, dict_search):
    variances = np.var(points, axis=0)
    return np.mean(variances)

def calculate_objective_entropy(points, dict_search):
    entropies = []
    for i in range(points.shape[1]):
        unique_values, counts = np.unique(points[:, i], return_counts=True)
        entropies.append(entropy(counts))
    return np.mean(entropies)

def find_global_pareto_points(decision_vars, objective_values):
    if len(decision_vars) < 2:
        return np.array([])

    pareto_indices = []
    all_objectives = np.array(objective_values)

    for i in range(len(decision_vars)):
        current_obj = all_objectives[i]
        is_pareto = True

        for j in range(len(decision_vars)):
            if i == j:
                continue
            other_obj = all_objectives[j]
            if (other_obj[0] <= current_obj[0] and other_obj[1] < current_obj[1]) or \
               (other_obj[0] < current_obj[0] and other_obj[1] <= current_obj[1]):
                is_pareto = False
                break

        if is_pareto:
            pareto_indices.append(i)

    return np.array(pareto_indices)

def calculate_gdx(decision_vars, pareto_set):
    if len(decision_vars) == 0 or len(pareto_set) == 0:
        return 0.0

    total_squared_dist = 0.0
    l = len(decision_vars)

    for var in decision_vars:
        var_np = np.array(var)
        min_dist = min(np.linalg.norm(var_np - np.array(p)) for p in pareto_set)
        total_squared_dist += min_dist ** 2

    gdx = (1.0 / l) * np.sqrt(total_squared_dist)
    return gdx

def calculate_pareto_connected_components(decision_vars, objective_values, sampled_data, random_seed=None):
    global_indices = find_global_pareto_points(decision_vars, objective_values)
    if len(global_indices) < 1:
        return 0

    pareto_decision_vars = [tuple(decision_vars[i]) for i in global_indices]
    sampled_set = set(tuple(v) for v in sampled_data)
    var_to_index = {var: idx for idx, var in enumerate(pareto_decision_vars)}

    G = nx.Graph()
    for var in pareto_decision_vars:
        G.add_node(var_to_index[var])
        neighbors = generate_neighbor_solutions(var, sampled_set, random_seed=random_seed)
        for neighbor in neighbors:
            if neighbor in var_to_index:
                G.add_edge(var_to_index[var], var_to_index[neighbor])

    return len(list(nx.connected_components(G)))

def calculate_dominance_features(decision_vars, objective_values):
    if len(decision_vars) < 2:
        return {
            'mean_dominance': 0, 'median_dominance': 0,
            'max_dominance': 0, 'min_dominance': 0,
            'num_pareto': 0
        }

    all_objectives = np.array(objective_values)
    dominance_counts = []
    num_pareto = 0

    for i in range(len(decision_vars)):
        current_obj = all_objectives[i]
        dominance_count = 0
        is_pareto = True

        for j in range(len(decision_vars)):
            if i == j:
                continue
            other_obj = all_objectives[j]
            if (other_obj[0] <= current_obj[0] and other_obj[1] < current_obj[1]) or \
               (other_obj[0] < current_obj[0] and other_obj[1] <= current_obj[1]):
                dominance_count += 1
                is_pareto = False

        dominance_counts.append(dominance_count)
        if is_pareto:
            num_pareto += 1

    return {
        'mean_dominance': np.mean(dominance_counts),
        'median_dominance': np.median(dominance_counts),
        'max_dominance': np.max(dominance_counts),
        'min_dominance': np.min(dominance_counts),
        'num_pareto': num_pareto
    }

def calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=None):
    global_indices = find_global_pareto_points(decision_vars, objective_values)
    local_indices = find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed)

    global_set = set(global_indices)
    local_set = set(local_indices)
    intersection = global_set.intersection(local_set)

    return {
        'num_global': len(global_set),
        'num_local': len(local_set),
        'num_intersection': len(intersection),
        'local_as_global_ratio': len(intersection) / len(local_set) if len(local_set) > 0 else 0,
        'global_not_local': len(global_set - local_set)
    }

def calculate_avg_continuous_descent_cone(decision_vars, objective_values, sampled_data, random_seed=None):
    if len(decision_vars) == 0:
        return 0.0

    var_to_obj = {tuple(v): obj for v, obj in zip(decision_vars, objective_values)}
    sampled_set = set(tuple(v) for v in sampled_data)
    total_ratio = 0.0
    valid_points = 0

    for current_var in decision_vars:
        current_var_tuple = tuple(current_var)
        current_obj = var_to_obj[current_var_tuple]
        neighbors = generate_neighbor_solutions(current_var, sampled_set, random_seed=random_seed)
        total_neighbors = len(neighbors)
        if total_neighbors == 0:
            continue

        dominating_count = 0
        for neighbor_var in neighbors:
            neighbor_obj = var_to_obj.get(tuple(neighbor_var))
            if neighbor_obj is None:
                continue
            if (neighbor_obj[0] <= current_obj[0] and neighbor_obj[1] < current_obj[1]) or \
               (neighbor_obj[0] < current_obj[0] and neighbor_obj[1] <= current_obj[1]):
                dominating_count += 1

        total_ratio += (dominating_count / total_neighbors)
        valid_points += 1

    return total_ratio / valid_points if valid_points > 0 else 0.0

def create_grid(objective_values, step=0.2, feature_range=None):
    if len(objective_values) == 0:
        return None, None
    min_val = np.min(objective_values, axis=0) * 0.9
    max_val = np.max(objective_values, axis=0) * 1.1
    grid_points_x = np.arange(min_val[0], max_val[0] + step, step)
    grid_points_y = np.arange(min_val[1], max_val[1] + step, step)
    grid_indices = []
    for obj in objective_values:
        i = np.clip(np.searchsorted(grid_points_x, obj[0], side='right') - 1, 0, len(grid_points_x)-2)
        j = np.clip(np.searchsorted(grid_points_y, obj[1], side='right') - 1, 0, len(grid_points_y)-2)
        grid_indices.append((i, j))
    return {'x': grid_points_x, 'y': grid_points_y}, np.array(grid_indices)

def calculate_grid_features(decision_vars, objective_values, is_pareto_list):
    grid_params, grid_indices = create_grid(objective_values)
    if grid_params is None:
        return {'grid_density_var': 0.0, 'pareto_grid_ratio': 0.0, 'adjacent_grid_euclidean_mean': 0.0}
    grid_x, grid_y = grid_params['x'], grid_params['y']
    grid_size_x, grid_size_y = len(grid_x), len(grid_y)

    unique_grids = set(tuple(idx) for idx in grid_indices)
    total_grids = (grid_size_x - 1) * (grid_size_y - 1)
    fill_rate = len(unique_grids) / total_grids if total_grids > 0 else 0

    grid_counts = defaultdict(int)
    for idx in grid_indices:
        grid_counts[tuple(idx)] += 1
    count_values = np.array(list(grid_counts.values()))
    density_mean = np.mean(count_values) if len(count_values) > 0 else 0
    density_var = np.var(count_values) if len(count_values) > 0 else 0

    pareto_grids = set(tuple(grid_indices[i]) for i in range(len(is_pareto_list)) if is_pareto_list[i])
    pareto_ratio = len(pareto_grids) / len(unique_grids) if len(unique_grids) > 0 else 0

    adjacent_pairs = 0
    total_euclidean = 0
    for i in range(len(decision_vars)):
        current_grid = tuple(grid_indices[i])
        current_var = np.array(decision_vars[i])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_grid = (current_grid[0]+dx, current_grid[1]+dy)
            if 0 <= adj_grid[0] < (grid_size_x-1) and 0 <= adj_grid[1] < (grid_size_y-1):
                adj_indices = [j for j in range(len(decision_vars)) if tuple(grid_indices[j]) == adj_grid]
                if adj_indices:
                    adj_var = np.array(decision_vars[adj_indices[0]])
                    total_euclidean += np.linalg.norm(current_var - adj_var)
                    adjacent_pairs += 1
    adj_euclidean_mean = total_euclidean / adjacent_pairs if adjacent_pairs > 0 else 0

    return {
        'grid_fill_rate': fill_rate,
        'grid_density_mean': density_mean,
        'grid_density_var': density_var,
        'pareto_grid_ratio': pareto_ratio,
        'adjacent_grid_euclidean_mean': adj_euclidean_mean
    }

def print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count, fig_num, x, y,
                              random_seed, sampling_method, sample_size, dataset_name, mode, unique_id, sample_type,
                              dominance_change_frequency, reverse):
    adjacent_grid_euclidean_mean = 0.0
    average_euclidean_distance = 0.0
    avg_continuous_descent_cone = 0.0
    connected_components_pareto = 0
    connected_components = 0
    dominance_change_frequency_stat = float(dominance_change_frequency) if isinstance(dominance_change_frequency, (int, float)) else dominance_change_frequency
    grid_density_var = 0.0
    hypervolume = 0.0
    kendalltau_corr = 0.0
    level_count = 0
    local_efficient_points_count = 0
    local_to_global_ratio = 0.0
    max_dominance_count = 0
    median_dominance_count = 0.0
    objective_function_value_entropy = 0.0
    objective_function_value_variance = 0.0
    pareto_grid_ratio = 0.0
    pearson_corr = 0.0
    slope = 0.0
    spearman_corr = 0.0
    pareto_ratio = total_points_count.get('pareto', 0) / (total_points_count.get('pareto', 0) + total_points_count.get('non_pareto', 0)) if (total_points_count.get('pareto', 0) + total_points_count.get('non_pareto', 0)) > 0 else 0.0

    if len(x) >= 2 and len(y) >= 2:
        decision_vars = np.array(list(sampled_dict.keys()))
        objective_values = np.array(list(sampled_dict.values()))
        ref_point = np.array([np.max(x), np.max(y)])

        spearman_corr, _ = spearmanr(x, y)
        pearson_corr, _ = pearsonr(x, y)
        kendalltau_corr, _ = kendalltau(x, y)
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        slope = float(model.coef_[0])

        try:
            level_count = calculate_level_count(np.vstack([x, y]).T)
        except Exception:
            level_count = 0
        objective_function_value_variance = calculate_objective_variance(np.vstack([x, y]).T, sampled_dict)
        objective_function_value_entropy = calculate_objective_entropy(np.vstack([x, y]).T, sampled_dict)

        avg_continuous_descent_cone = calculate_avg_continuous_descent_cone(decision_vars, objective_values, sampled_data, random_seed=random_seed)

        local_idx_lit = find_local_efficient_points_literature(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        local_efficient_points_count = int(len(local_idx_lit))

        local_idx_dom = find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        connected_components = calculate_connected_components(decision_vars, local_idx_dom, sampled_data, random_seed=random_seed)
        connected_components_pareto = calculate_pareto_connected_components(decision_vars, objective_values, sampled_data, random_seed=random_seed)

        dom_feats = calculate_dominance_features(decision_vars, objective_values)
        max_dominance_count = int(dom_feats['max_dominance'])
        median_dominance_count = float(dom_feats['median_dominance'])

        l2g = calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        local_to_global_ratio = float(l2g['local_as_global_ratio'])

        is_pareto_list = np.array([is_pareto_optimal_minimize(p, objective_values) for p in objective_values])
        grid_feats = calculate_grid_features(decision_vars, objective_values, is_pareto_list)
        grid_density_var = float(grid_feats['grid_density_var'])
        pareto_grid_ratio = float(grid_feats['pareto_grid_ratio'])
        adjacent_grid_euclidean_mean = float(grid_feats['adjacent_grid_euclidean_mean'])

        try:
            hypervolume = float(calculate_hypervolume(np.vstack([x, y]).T, ref_point))
        except Exception:
            hypervolume = 0.0

        pareto_idx = find_global_pareto_points(decision_vars, objective_values)
        if len(pareto_idx) > 1:
            pareto_decisions = np.array([np.asarray(decision_vars[i], dtype=float).ravel() for i in pareto_idx])
            dmat = pairwise_distances(pareto_decisions, metric='euclidean')
            triu = np.triu_indices_from(dmat, k=1)
            average_euclidean_distance = float(np.mean(dmat[triu])) if len(triu[0]) > 0 else 0.0
        else:
            average_euclidean_distance = 0.0

    else:
        average_euclidean_distance = 0.0

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Results/Output-draw')
    os.makedirs(output_dir, exist_ok=True)
    csv_file = f'{output_dir}/{mode}_statistics_reverse.csv' if reverse else f'{output_dir}/{mode}_statistics.csv'

    headers = [
        'Figure Number', 'Sampling Method', 'Sample Size', 'Dataset Name', 'Mode', 'Random Seed',
        'Unique ID', 'Sample Type',
        'adjacent grid euclidean mean',
        'average euclidean distance',
        'avg continuous descent cone',
        'connected components (pareto)',
        'connected components',
        'dominance change frequency',
        'grid density variance',
        'hypervolume',
        "kendall's tau correlation coefficient",
        'level count',
        'local efficient points count',
        'local-to-global ratio',
        'max dominance count',
        'median dominance count',
        'objective function value entropy',
        'objective function value variance',
        'pareto grid ratio',
        'pearson correlation coefficient',
        'slope of linear regression fit',
        'spearman correlation coefficient',
        'pareto dominated solution point ratio'
    ]

    row = [
        fig_num, sampling_method, sample_size, dataset_name, mode, random_seed, unique_id, sample_type,
        adjacent_grid_euclidean_mean,
        average_euclidean_distance,
        avg_continuous_descent_cone,
        connected_components_pareto,
        connected_components,
        dominance_change_frequency_stat,
        grid_density_var,
        hypervolume,
        kendalltau_corr,
        level_count,
        local_efficient_points_count,
        local_to_global_ratio,
        max_dominance_count,
        median_dominance_count,
        objective_function_value_entropy,
        objective_function_value_variance,
        pareto_grid_ratio,
        pearson_corr,
        slope,
        spearman_corr,
        pareto_ratio
    ]

    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

def plot_figure1(random_seed=42, center_type='mean', sampling_method='', sample_size='',
                 dataset_name='', mode='', unique_id='', sample_type='', sampled_dict=None,
                 reverse=False):
    sampled_data = np.array(list(sampled_dict.keys()))
    r0_points = np.array(list(sampled_dict.values()))
    center_point = find_center_point(r0_points, random_seed, center_type)
    all_points = np.vstack([r0_points, center_point])

    r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
    total_points_count = {'pareto': 0, 'non_pareto': 0}

    for point in r0_points:
        is_pareto = is_pareto_optimal_minimize(point, all_points)
        if is_pareto:
            total_points_count['pareto'] += 1
        else:
            total_points_count['non_pareto'] += 1

        vector = point - center_point
        point_angle = np.arctan2(vector[1], vector[0])
        if point_angle < 0:
            point_angle += 2 * np.pi

        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            angle_start = angle
            angle_end = angle + np.pi / 4
            if angle_end > 2 * np.pi:
                angle_end -= 2 * np.pi

            if angle_start <= point_angle < angle_end:
                if is_pareto:
                    r_pareto_count[f'r_{i}']['pareto'] += 1
                else:
                    r_pareto_count[f'r_{i}']['non_pareto'] += 1
                break

    x = r0_points[:, 0]
    y = r0_points[:, 1]

    dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, max_values_per_dim=10,
                                                                      random_seed=random_seed)

    pareto_points = []
    for i, point in enumerate(r0_points):
        if is_pareto_optimal_minimize(point, all_points):
            pareto_points.append(point)

    print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count, 1, x, y,
                              random_seed, sampling_method, sample_size, dataset_name, mode, unique_id, sample_type,
                              dominance_change_frequency, reverse)

def plot_figure2(random_seed=42, center_type='mean', sampling_method='', sample_size='', dataset_name='',
                 mode='', unique_id='', sample_type='', sampled_dict=None, reverse=False):
    try:
        sampled_data = np.array(list(sampled_dict.keys()))
        objective_values = np.array(list(sampled_dict.values()))
        g1 = objective_values[:, 0]
        g2 = objective_values[:, 1]
        r0_points = np.column_stack((g1, g2))
        center_point = find_center_point(r0_points, random_seed, center_type)
        all_points = np.vstack([r0_points, center_point])

        r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
        total_points_count = {'pareto': 0, 'non_pareto': 0}

        for point in r0_points:
            is_pareto = is_pareto_optimal_minimize(point, all_points)
            if is_pareto:
                total_points_count['pareto'] += 1
            else:
                total_points_count['non_pareto'] += 1

            vector = point - center_point
            point_angle = np.arctan2(vector[1], vector[0])
            if point_angle < 0:
                point_angle += 2 * np.pi

            for i, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
                angle_start = angle
                angle_end = angle + np.pi / 4
                if angle_end > 2 * np.pi:
                    angle_end -= 2 * np.pi

                if angle_start <= point_angle < angle_end:
                    if is_pareto:
                        r_pareto_count[f'r_{i}']['pareto'] += 1
                    else:
                        r_pareto_count[f'r_{i}']['non_pareto'] += 1
                    break

        x = r0_points[:, 0]
        y = r0_points[:, 1]

        dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, max_values_per_dim=10,
                                                                          random_seed=random_seed)

        pareto_points = []
        for i, point in enumerate(r0_points):
            if is_pareto_optimal_minimize(point, all_points):
                pareto_points.append(point)

        print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count, 2, x, y,
                                  random_seed, sampling_method, sample_size, dataset_name, mode,
                                  unique_id, sample_type, dominance_change_frequency, reverse)
    except Exception as e:
        print(f"Error processing Figure 2: {e}")