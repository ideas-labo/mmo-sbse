import os
import csv
import sys
import random
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
import networkx as nx

from jmetal.core.quality_indicator import HyperVolume
from jmetal.util.ranking import FastNonDominatedRanking

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

class ExactHammingIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = []
        self._position_index = defaultdict(list)

    def add(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.int32)
        start_idx = len(self.data)
        self.data.extend(vectors.tolist())
        for i in range(start_idx, len(self.data)):
            for pos in range(self.dimension):
                val = self.data[i][pos]
                self._position_index[(pos, val)].append(i)

    def search(self, query, k, max_candidates=1000):
        if not self.data:
            return [], []
        k = get_adaptive_k(self.dimension)
        query = np.array(query, dtype=np.int32).flatten()
        query_tuple = tuple(query)
        candidate_counts = defaultdict(int)
        for pos in range(self.dimension):
            val = query[pos]
            for idx in self._position_index.get((pos, val), []):
                candidate_counts[idx] += 1
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in sorted_candidates[:max_candidates]]
        valid_neighbors = []
        query_arr = np.array(query)
        for idx in candidate_indices:
            if tuple(self.data[idx]) == query_tuple:
                continue
            target = np.array(self.data[idx])
            dist = int(np.sum(query_arr != target))
            valid_neighbors.append((idx, dist))
        if len(valid_neighbors) < k and len(candidate_indices) < len(self.data):
            for idx in range(len(self.data)):
                if idx in candidate_indices:
                    continue
                if tuple(self.data[idx]) == query_tuple:
                    continue
                target = np.array(self.data[idx])
                dist = int(np.sum(query_arr != target))
                valid_neighbors.append((idx, dist))
                if len(valid_neighbors) >= k:
                    break
        valid_neighbors.sort(key=lambda x: x[1])
        selected = valid_neighbors[:k]
        nearest_indices = [idx for idx, _ in selected]
        nearest_distances = [dist for _, dist in selected]
        nearest_vectors = [tuple(self.data[i]) for i in nearest_indices]
        return nearest_vectors, nearest_distances

def get_adaptive_k(dimension):
    if dimension <= 50:
        return 20
    elif dimension <= 200:
        return 10
    else:
        return 5

class NeighborFinder:
    def __init__(self):
        self.index = None

    def generate_neighbor_solutions(self, solution, sampled_data, random_seed=None):
        if self.index is None:
            self.index = ExactHammingIndex(len(solution))
            self.index.add(np.array(list(sampled_data)))
        neighbors, _ = self.index.search(solution, k=get_adaptive_k(len(solution)))
        return list(neighbors)

neighbor_finder = NeighborFinder()

def generate_neighbor_solutions(solution, sampled_data, random_seed=None):
    return neighbor_finder.generate_neighbor_solutions(solution, sampled_data, random_seed)

def find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=None):
    if len(decision_vars) < 2:
        return np.array([])
    var_to_obj = {tuple(v): obj for v, obj in zip(decision_vars, objective_values)}
    sampled_set = set(tuple(v) for v in sampled_data)
    local_efficient = []
    for i, v in enumerate(decision_vars):
        cur = tuple(v)
        cur_obj = objective_values[i]
        dominated = False
        neighbors = generate_neighbor_solutions(cur, sampled_set, random_seed)
        for n in neighbors:
            n_obj = var_to_obj.get(n)
            if n_obj is None:
                continue
            if (n_obj[0] <= cur_obj[0] and n_obj[1] < cur_obj[1]) or (n_obj[0] < cur_obj[0] and n_obj[1] <= cur_obj[1]):
                dominated = True
                break
        if not dominated:
            local_efficient.append(i)
    return np.array(local_efficient)

def find_local_efficient_points_literature(decision_vars, objective_values, sampled_data, random_seed=None):
    return find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=random_seed)

def calculate_connected_components(decision_vars, local_efficient_indices, sampled_data, random_seed=None):
    if len(local_efficient_indices) < 1:
        return 0
    le_vars = [tuple(decision_vars[i]) for i in local_efficient_indices]
    sampled_set = set(tuple(v) for v in sampled_data)
    idx_map = {v: i for i, v in enumerate(le_vars)}
    G = nx.Graph()
    for v in le_vars:
        G.add_node(idx_map[v])
        neighbors = generate_neighbor_solutions(v, sampled_set, random_seed=random_seed)
        for n in neighbors:
            if n in idx_map:
                G.add_edge(idx_map[v], idx_map[n])
    return len(list(nx.connected_components(G)))

def find_global_pareto_points(decision_vars, objective_values):
    if len(decision_vars) < 2:
        return np.array([])
    objs = np.array(objective_values)
    pareto_idx = []
    for i in range(len(objs)):
        cur = objs[i]
        dominated = False
        for j in range(len(objs)):
            if i == j:
                continue
            other = objs[j]
            if (other[0] <= cur[0] and other[1] < cur[1]) or (other[0] < cur[0] and other[1] <= cur[1]):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return np.array(pareto_idx)

def calculate_pareto_connected_components(decision_vars, objective_values, sampled_data, random_seed=None):
    pareto_indices = find_global_pareto_points(decision_vars, objective_values)
    if len(pareto_indices) < 1:
        return 0
    pareto_vars = [tuple(decision_vars[i]) for i in pareto_indices]
    sampled_set = set(tuple(v) for v in sampled_data)
    idx_map = {v: i for i, v in enumerate(pareto_vars)}
    G = nx.Graph()
    for v in pareto_vars:
        G.add_node(idx_map[v])
        neighbors = generate_neighbor_solutions(v, sampled_set, random_seed=random_seed)
        for n in neighbors:
            if n in idx_map:
                G.add_edge(idx_map[v], idx_map[n])
    return len(list(nx.connected_components(G)))

def calculate_dominance_change_frequency(sampled_dict, random_seed=None):
    if len(sampled_dict) < 2:
        return 0.0
    decision_vars = np.array(list(sampled_dict.keys()))
    objective_values = np.array(list(sampled_dict.values()))
    sampled_set = set(tuple(v) for v in decision_vars)
    is_pareto_list = np.array([is_pareto_optimal_minimize(p, objective_values) for p in objective_values])
    changes = 0
    pairs = 0
    for i, dv in enumerate(decision_vars):
        dv_t = tuple(dv)
        cur_is = is_pareto_list[i]
        neighbors = generate_neighbor_solutions(dv_t, sampled_set, random_seed=random_seed)
        for n in neighbors:
            idxs = [j for j, vv in enumerate(decision_vars) if tuple(vv) == n]
            if not idxs:
                continue
            n_is = is_pareto_list[idxs[0]]
            if cur_is != n_is:
                changes += 1
            pairs += 1
    return changes / pairs if pairs > 0 else 0.0

def calculate_grid_features(decision_vars, objective_values, is_pareto_list, step=0.2, feature_range=(-2, 2)):
    if len(objective_values) == 0:
        return {'grid_density_var': 0.0, 'pareto_grid_ratio': 0.0, 'adjacent_grid_hamming_mean': 0.0}
    min_val, max_val = feature_range
    grid_points = np.arange(min_val, max_val + step, step)
    grid_indices = []
    for obj in objective_values:
        i = np.clip(np.searchsorted(grid_points, obj[0], side='right') - 1, 0, len(grid_points) - 2)
        j = np.clip(np.searchsorted(grid_points, obj[1], side='right') - 1, 0, len(grid_points) - 2)
        grid_indices.append((int(i), int(j)))
    unique_grids = set(grid_indices)
    grid_counts = {}
    for idx in grid_indices:
        grid_counts[idx] = grid_counts.get(idx, 0) + 1
    count_values = np.array(list(grid_counts.values())) if grid_counts else np.array([0.0])
    density_var = float(np.var(count_values)) if count_values.size > 0 else 0.0
    pareto_grids = set(grid_indices[i] for i in range(len(is_pareto_list)) if is_pareto_list[i])
    pareto_ratio = float(len(pareto_grids) / len(unique_grids)) if len(unique_grids) > 0 else 0.0
    adjacent_pairs = 0
    total_hamming = 0
    grid_size = len(grid_points)
    for i in range(len(decision_vars)):
        cur_grid = grid_indices[i]
        cur_var = decision_vars[i]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (cur_grid[0] + dx, cur_grid[1] + dy)
            if 0 <= adj[0] < (grid_size - 1) and 0 <= adj[1] < (grid_size - 1):
                adj_idxs = [j for j in range(len(decision_vars)) if grid_indices[j] == adj]
                if adj_idxs:
                    adj_var = decision_vars[adj_idxs[0]]
                    hamming = sum(1 for a, b in zip(cur_var, adj_var) if a != b)
                    total_hamming += hamming
                    adjacent_pairs += 1
    adj_mean = float(total_hamming / adjacent_pairs) if adjacent_pairs > 0 else 0.0
    return {'grid_density_var': density_var, 'pareto_grid_ratio': pareto_ratio, 'adjacent_grid_hamming_mean': adj_mean}

def calculate_hypervolume(solutions, ref_point):
    hypervolume = HyperVolume(reference_point=ref_point)
    return hypervolume.compute(solutions)

def calculate_level_count(points):
    ranking = FastNonDominatedRanking()
    solutions = [type('S', (object,), {'objectives': point, 'attributes': {}}) for point in points]
    ranking.compute_ranking(solutions)
    return ranking.get_number_of_subfronts()

def calculate_objective_variance(points, dict_search):
    if points is None or len(points) == 0:
        return 0.0
    variances = np.var(points, axis=0)
    return float(np.mean(variances))

def calculate_objective_entropy(points, dict_search):
    if points is None or len(points) == 0:
        return 0.0
    entropies = []
    for i in range(points.shape[1]):
        _, counts = np.unique(points[:, i], return_counts=True)
        entropies.append(float(entropy(counts)))
    return float(np.mean(entropies))

def calculate_avg_discrete_descent_cone(decision_vars, objective_values, sampled_data, random_seed=None):
    if len(decision_vars) == 0:
        return 0.0
    var_to_obj = {tuple(v): obj for v, obj in zip(decision_vars, objective_values)}
    sampled_set = set(tuple(v) for v in sampled_data)
    total = 0.0
    valid = 0
    for var in decision_vars:
        vt = tuple(var)
        cur_obj = var_to_obj.get(vt)
        if cur_obj is None:
            continue
        neighbors = generate_neighbor_solutions(vt, sampled_set, random_seed=random_seed)
        if not neighbors:
            continue
        dom = 0
        for n in neighbors:
            n_obj = var_to_obj.get(n)
            if n_obj is None:
                continue
            if (n_obj[0] <= cur_obj[0] and n_obj[1] < cur_obj[1]) or (n_obj[0] < cur_obj[0] and n_obj[1] <= cur_obj[1]):
                dom += 1
        total += (dom / len(neighbors))
        valid += 1
    return float(total / valid) if valid > 0 else 0.0

def calculate_dominance_features(decision_vars, objective_values):
    if len(decision_vars) < 2:
        return {'median_dominance': 0.0, 'max_dominance': 0}
    all_objs = np.array(objective_values)
    counts = []
    for i in range(len(all_objs)):
        cur = all_objs[i]
        cnt = 0
        for j in range(len(all_objs)):
            if i == j:
                continue
            other = all_objs[j]
            if (other[0] <= cur[0] and other[1] < cur[1]) or (other[0] < cur[0] and other[1] <= cur[1]):
                cnt += 1
        counts.append(cnt)
    return {'median_dominance': float(np.median(counts)), 'max_dominance': int(np.max(counts))}

def calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=None):
    global_idx = find_global_pareto_points(decision_vars, objective_values)
    local_idx = find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=random_seed)
    gset = set(global_idx)
    lset = set(local_idx)
    inter = gset.intersection(lset)
    return {
        'local_as_global_ratio': float(len(inter) / len(lset)) if len(lset) > 0 else 0.0,
        'num_global': len(gset),
        'num_local': len(lset),
        'num_intersection': len(inter),
        'global_not_local': len(gset - lset)
    }

def print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              fig_num, x, y, random_seed, sampling_method, sample_size,
                              dataset_name, mode, unique_id, sample_type,
                              dominance_change_frequency, reverse, dag=None, num_valid_tasks=None):
    total_points = total_points_count.get('pareto', 0) + total_points_count.get('non_pareto', 0)
    pareto_ratio = total_points_count.get('pareto', 0) / total_points if total_points > 0 else 0.0
    adjacent_grid_hamming_mean = 0.0
    average_hamming_distance = 0.0
    avg_discrete_descent_cone = 0.0
    connected_components_pareto = 0
    connected_components = 0
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
        level_count = calculate_level_count(np.vstack([x, y]).T)
        objective_function_value_variance = calculate_objective_variance(np.vstack([x, y]).T, sampled_dict)
        objective_function_value_entropy = calculate_objective_entropy(np.vstack([x, y]).T, sampled_dict)
        avg_discrete_descent_cone = calculate_avg_discrete_descent_cone(decision_vars, objective_values, sampled_data,
                                                                        random_seed=random_seed)
        local_idx_lit = find_local_efficient_points_literature(decision_vars, objective_values, sampled_data,
                                                               random_seed=random_seed)
        local_efficient_points_count = int(len(local_idx_lit))
        local_idx_dom = find_local_efficient_points(decision_vars, objective_values, sampled_data,
                                                    random_seed=random_seed)
        connected_components = calculate_connected_components(decision_vars, local_idx_dom, sampled_data,
                                                              random_seed=random_seed)
        connected_components_pareto = calculate_pareto_connected_components(decision_vars, objective_values,
                                                                            sampled_data, random_seed=random_seed)
        dom_feats = calculate_dominance_features(decision_vars, objective_values)
        max_dominance_count = int(dom_feats['max_dominance'])
        median_dominance_count = float(dom_feats['median_dominance'])
        l2g = calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        local_to_global_ratio = float(l2g['local_as_global_ratio'])
        is_pareto_list = np.array([is_pareto_optimal_minimize(p, objective_values) for p in objective_values])
        grid_feats = calculate_grid_features(decision_vars, objective_values, is_pareto_list)
        grid_density_var = float(grid_feats['grid_density_var'])
        pareto_grid_ratio = float(grid_feats['pareto_grid_ratio'])
        adjacent_grid_hamming_mean = float(grid_feats['adjacent_grid_hamming_mean'])
        initial_solution = np.vstack([x, y]).T
        try:
            hypervolume = float(calculate_hypervolume(initial_solution, ref_point))
        except Exception:
            hypervolume = 0.0
        pareto_idx = find_global_pareto_points(decision_vars, objective_values)
        pareto_objs = objective_values[pareto_idx] if len(pareto_idx) > 0 else np.array([])
        if len(pareto_objs) > 1:
            dmat = pairwise_distances(pareto_objs, metric='hamming')
            triu = np.triu_indices_from(dmat, k=1)
            average_hamming_distance = float(np.mean(dmat[triu])) if len(triu[0]) > 0 else 0.0
        else:
            average_hamming_distance = 0.0
    dominance_change_frequency_stat = float(dominance_change_frequency) if isinstance(dominance_change_frequency, (int, float)) else dominance_change_frequency
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Results/Output-draw')
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Unable to create output directory: {e}")
        return
    csv_file = f'{output_dir}/{mode}_statistics_reverse.csv' if reverse else f'{output_dir}/{mode}_statistics.csv'
    headers = [
        'Figure Number', 'Sampling Method', 'Sample Size', 'Dataset Name', 'Mode', 'Random Seed',
        'Unique ID', 'Sample Type',
        'adjacent grid hamming mean',
        'average hamming distance',
        'avg discrete descent cone',
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
        adjacent_grid_hamming_mean,
        average_hamming_distance,
        avg_discrete_descent_cone,
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
                 reverse=False, dag=None, num_valid_tasks=None):
    sampled_data = np.array(list(sampled_dict.keys()))
    r0_points = np.array(list(sampled_dict.values()))
    center_point = np.mean(r0_points, axis=0)
    all_points = np.vstack([r0_points, center_point])
    r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
    total_points_count = {'pareto': 0, 'non_pareto': 0}
    for point in r0_points:
        is_p = is_pareto_optimal_minimize(point, all_points)
        if is_p:
            total_points_count['pareto'] += 1
        else:
            total_points_count['non_pareto'] += 1
        vector = point - center_point
        angle = np.arctan2(vector[1], vector[0])
        if angle < 0:
            angle += 2 * np.pi
        for i, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            start = a
            end = a + np.pi / 4
            if end > 2 * np.pi:
                end -= 2 * np.pi
            if start <= angle < end:
                if is_p:
                    r_pareto_count[f'r_{i}']['pareto'] += 1
                else:
                    r_pareto_count[f'r_{i}']['non_pareto'] += 1
                break
    x = r0_points[:, 0]
    y = r0_points[:, 1]
    dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, random_seed=random_seed)
    print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              1, x, y, random_seed, sampling_method, sample_size,
                              dataset_name, mode, unique_id, sample_type,
                              dominance_change_frequency, reverse, dag, num_valid_tasks)

def plot_figure2(random_seed=42, center_type='random', sampling_method='', sample_size='', dataset_name='',
                 mode='', unique_id='', sample_type='', sampled_dict=None, reverse=False,
                 dag=None, num_valid_tasks=None):
    try:
        sampled_data = np.array(list(sampled_dict.keys()))
        objective_values = np.array(list(sampled_dict.values()))
        g1 = objective_values[:, 0]
        g2 = objective_values[:, 1]
        r0_points = np.column_stack((g1, g2))
        center_point = np.mean(r0_points, axis=0)
        all_points = np.vstack([r0_points, center_point])
        r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
        total_points_count = {'pareto': 0, 'non_pareto': 0}
        for point in r0_points:
            is_p = is_pareto_optimal_minimize(point, all_points)
            if is_p:
                total_points_count['pareto'] += 1
            else:
                total_points_count['non_pareto'] += 1
            vector = point - center_point
            angle = np.arctan2(vector[1], vector[0])
            if angle < 0:
                angle += 2 * np.pi
            for i, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
                start = a
                end = a + np.pi / 4
                if end > 2 * np.pi:
                    end -= 2 * np.pi
                if start <= angle < end:
                    if is_p:
                        r_pareto_count[f'r_{i}']['pareto'] += 1
                    else:
                        r_pareto_count[f'r_{i}']['non_pareto'] += 1
                    break
        x = r0_points[:, 0]
        y = r0_points[:, 1]
        dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, random_seed=random_seed)
        print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                                  2, x, y, random_seed, sampling_method, sample_size,
                                  dataset_name, mode, unique_id, sample_type,
                                  dominance_change_frequency, reverse, dag, num_valid_tasks)
    except Exception as e:
        print(f"Error processing Figure 2: {e}")