import os
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
import networkx as nx
import random

try:
    from jmetal.core.quality_indicator import HyperVolume
    from jmetal.util.ranking import FastNonDominatedRanking
except Exception:
    HyperVolume = None
    FastNonDominatedRanking = None

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

def is_pareto_optimal_minimize(point, points):
    p = np.asarray(point, dtype=float).ravel()
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if p.size < 2 or pts.shape[1] < 2:
        raise ValueError("is_pareto_optimal_minimize requires 2 objectives per point")
    for other in pts:
        if (other[0] <= p[0] and other[1] < p[1]) or (other[0] < p[0] and other[1] <= p[1]):
            return False
    return True

class ExactHammingIndex:
    def __init__(self, dimension):
        self.dimension = int(dimension)
        self.data = []
        self._position_index = defaultdict(list)

    def add(self, vectors):
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        start = len(self.data)
        for v in vectors:
            vec = [int(x) for x in v]
            self.data.append(vec)
        for idx in range(start, len(self.data)):
            vec = self.data[idx]
            for pos in range(self.dimension):
                val = vec[pos]
                self._position_index[(pos, val)].append(idx)

    def search(self, query, k, max_candidates=1000):
        if not self.data:
            return [], []
        k = int(min(k, len(self.data)))
        q = np.array(query, dtype=int).flatten()
        candidate_counts = defaultdict(int)
        for pos in range(self.dimension):
            val = int(q[pos])
            for idx in self._position_index.get((pos, val), []):
                candidate_counts[idx] += 1
        if candidate_counts:
            sorted_cands = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
            candidate_indices = [idx for idx, _ in sorted_cands[:max_candidates]]
        else:
            candidate_indices = list(range(len(self.data)))
        distances = []
        for idx in candidate_indices:
            tgt = np.array(self.data[idx], dtype=int)
            distances.append(int(np.sum(q != tgt)))
        order = np.argsort(distances)[:k]
        nearest_idxs = [candidate_indices[i] for i in order]
        nearest_vectors = [tuple(self.data[i]) for i in nearest_idxs]
        nearest_distances = [distances[i] for i in order]
        return nearest_vectors, nearest_distances

    def __len__(self):
        return len(self.data)

def get_adaptive_k(dimension):
    d = int(dimension)
    if d <= 50:
        return 20
    elif d <= 200:
        return 10
    else:
        return 5

class NeighborFinder:
    def __init__(self):
        self.index = None
        self.dim = None

    def generate_neighbor_solutions(self, solution, sampled_data, random_seed=None):
        if isinstance(sampled_data, np.ndarray):
            sampled_list = sampled_data.tolist()
        else:
            sampled_list = list(sampled_data)
        if len(sampled_list) == 0:
            return []
        sol_arr = np.asarray(solution)
        dim = sol_arr.size
        self.dim = dim
        if self.index is None:
            self.index = ExactHammingIndex(dim)
            try:
                arr = np.array(sampled_list, dtype=int)
            except Exception:
                arr = np.array([np.array(v, dtype=int) for v in sampled_list], dtype=int)
            self.index.add(arr)
        k = get_adaptive_k(dim)
        neighbors, _ = self.index.search(solution, k)
        return neighbors

_neighbor_finder = NeighborFinder()

def generate_neighbor_solutions(solution, sampled_data, random_seed=None):
    return _neighbor_finder.generate_neighbor_solutions(solution, sampled_data, random_seed)

def find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=None):
    if len(decision_vars) < 2:
        return np.array([])
    dv = [tuple(v) for v in decision_vars]
    ov = [tuple(o) for o in objective_values]
    sampled_list = list(sampled_data)
    local_flags = []
    for i, var in enumerate(dv):
        cur_obj = ov[i]
        dominated = False
        neighbors = generate_neighbor_solutions(var, sampled_list, random_seed)
        for n in neighbors:
            try:
                j = dv.index(n)
            except ValueError:
                j = None
            if j is not None:
                other_obj = ov[j]
                if (other_obj[0] <= cur_obj[0] and other_obj[1] < cur_obj[1]) or \
                   (other_obj[0] < cur_obj[0] and other_obj[1] <= cur_obj[1]):
                    dominated = True
                    break
        local_flags.append(not dominated)
    return np.where(local_flags)[0]

def find_local_efficient_points_literature(decision_vars, objective_values, sampled_data, random_seed=None):
    return find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed)

def find_global_pareto_points(decision_vars, objective_values):
    objs = np.asarray(objective_values)
    if objs.ndim == 1:
        objs = objs.reshape(1, -1)
    if len(objs) < 2:
        return np.array([], dtype=int)
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
    return np.array(pareto_idx, dtype=int)

def calculate_connected_components(decision_vars, local_efficient_indices, sampled_data, random_seed=None):
    if len(local_efficient_indices) < 1:
        return 0
    if isinstance(decision_vars, np.ndarray):
        decision_vars = [tuple(x) for x in decision_vars.tolist()]
    sampled_list = list(sampled_data)
    le_vars = [decision_vars[i] for i in local_efficient_indices]
    idx_map = {v: i for i, v in enumerate(le_vars)}
    G = nx.Graph()
    for v in le_vars:
        G.add_node(idx_map[v])
        neighbors = generate_neighbor_solutions(v, sampled_list, random_seed)
        for n in neighbors:
            if n in idx_map:
                G.add_edge(idx_map[v], idx_map[n])
    return len(list(nx.connected_components(G)))

def calculate_pareto_connected_components(decision_vars, objective_values, sampled_data, random_seed=None):
    pareto_idx = find_global_pareto_points(decision_vars, objective_values)
    if len(pareto_idx) < 1:
        return 0
    pareto_vars = [tuple(decision_vars[i]) for i in pareto_idx]
    sampled_list = list(sampled_data)
    idx_map = {v: i for i, v in enumerate(pareto_vars)}
    G = nx.Graph()
    for v in pareto_vars:
        G.add_node(idx_map[v])
        neighbors = generate_neighbor_solutions(v, sampled_list, random_seed)
        for n in neighbors:
            if n in idx_map:
                G.add_edge(idx_map[v], idx_map[n])
    return len(list(nx.connected_components(G)))

def calculate_dominance_change_frequency(sampled_dict, random_seed=None):
    if len(sampled_dict) < 2:
        return 0.0
    decision_vars = list(sampled_dict.keys())
    objective_values = np.array(list(sampled_dict.values()))
    _ = generate_neighbor_solutions(decision_vars[0], decision_vars, random_seed)
    is_p = np.array([is_pareto_optimal_minimize(p, objective_values) for p in objective_values])
    var_to_idx = {tuple(v): i for i, v in enumerate(decision_vars)}
    changes = 0
    pairs = 0
    for i, v in enumerate(decision_vars):
        neighbors = generate_neighbor_solutions(v, decision_vars, random_seed)
        for n in neighbors:
            j = var_to_idx.get(n)
            if j is None or j == i:
                continue
            if is_p[i] != is_p[j]:
                changes += 1
            pairs += 1
    return changes / pairs if pairs > 0 else 0.0

def calculate_dominance_features(decision_vars, objective_values):
    objs = np.array(objective_values)
    if objs.shape[0] < 2:
        return {'median_dominance': 0.0, 'max_dominance': 0}
    counts = []
    for i in range(len(objs)):
        cnt = 0
        cur = objs[i]
        for j in range(len(objs)):
            if i == j:
                continue
            other = objs[j]
            if (other[0] <= cur[0] and other[1] < cur[1]) or (other[0] < cur[0] and other[1] <= cur[1]):
                cnt += 1
        counts.append(cnt)
    return {'median_dominance': float(np.median(counts)), 'max_dominance': int(np.max(counts))}

def calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=None):
    gidx = set(find_global_pareto_points(decision_vars, objective_values).tolist())
    lidx = set(find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed).tolist())
    inter = gidx.intersection(lidx)
    return {
        'local_as_global_ratio': float(len(inter) / len(lidx)) if len(lidx) > 0 else 0.0,
        'num_global': len(gidx),
        'num_local': len(lidx),
        'num_intersection': len(inter),
        'global_not_local': len(gidx - lidx)
    }

def calculate_avg_discrete_descent_cone(decision_vars, objective_values, sampled_data, random_seed=None):
    if len(decision_vars) == 0:
        return 0.0
    dv = [tuple(v) for v in decision_vars]
    ov = [tuple(o) for o in objective_values]
    sampled_list = list(sampled_data)
    total = 0.0
    valid = 0
    for i, var in enumerate(dv):
        neighbors = generate_neighbor_solutions(var, sampled_list, random_seed)
        if not neighbors:
            continue
        dom = 0
        cur_obj = ov[i]
        for n in neighbors:
            try:
                j = dv.index(n)
            except ValueError:
                j = None
            if j is not None:
                nobj = ov[j]
                if (nobj[0] <= cur_obj[0] and nobj[1] < cur_obj[1]) or (nobj[0] < cur_obj[0] and nobj[1] <= cur_obj[1]):
                    dom += 1
        total += dom / len(neighbors)
        valid += 1
    return float(total / valid) if valid > 0 else 0.0

def create_grid(objective_values, step=0.2, feature_range=(-2, 2)):
    if len(objective_values) == 0:
        return None, None
    min_val, max_val = feature_range
    grid_points = np.arange(min_val, max_val + step, step)
    grid_indices = []
    for obj in objective_values:
        i = np.clip(np.searchsorted(grid_points, obj[0], side='right') - 1, 0, len(grid_points) - 2)
        j = np.clip(np.searchsorted(grid_points, obj[1], side='right') - 1, 0, len(grid_points) - 2)
        grid_indices.append((int(i), int(j)))
    return {'size': len(grid_points)}, np.array(grid_indices)

def hamming_distance_vec(a, b):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    minlen = min(a_arr.size, b_arr.size)
    mismatches = int(np.sum(a_arr[:minlen] != b_arr[:minlen]))
    mismatches += abs(a_arr.size - b_arr.size)
    return mismatches

def calculate_grid_features(decision_vars, objective_values, is_pareto_list):
    if len(objective_values) == 0:
        return {'grid_density_var': 0.0, 'pareto_grid_ratio': 0.0, 'adjacent_grid_hamming_mean': 0.0}
    if isinstance(decision_vars, np.ndarray):
        decision_vars = [tuple(x) for x in decision_vars.tolist()]
    grid_params, grid_indices = create_grid(objective_values)
    if grid_params is None:
        return {'grid_density_var': 0.0, 'pareto_grid_ratio': 0.0, 'adjacent_grid_hamming_mean': 0.0}
    grid_counts = {}
    for idx in grid_indices:
        grid_counts[tuple(idx)] = grid_counts.get(tuple(idx), 0) + 1
    count_vals = np.array(list(grid_counts.values())) if grid_counts else np.array([0.0])
    density_var = float(np.var(count_vals)) if count_vals.size > 0 else 0.0
    unique = set(tuple(idx) for idx in grid_indices)
    pareto_grids = set(tuple(grid_indices[i]) for i in range(len(is_pareto_list)) if is_pareto_list[i])
    pareto_ratio = float(len(pareto_grids) / len(unique)) if len(unique) > 0 else 0.0
    adj_pairs = 0
    total_hamming = 0
    grid_size = grid_params['size']
    for i in range(len(decision_vars)):
        cur_grid = tuple(grid_indices[i])
        cur_dec = decision_vars[i]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (cur_grid[0] + dx, cur_grid[1] + dy)
            if 0 <= adj[0] < (grid_size - 1) and 0 <= adj[1] < (grid_size - 1):
                adj_idxs = [j for j in range(len(decision_vars)) if tuple(grid_indices[j]) == adj]
                if adj_idxs:
                    adj_dec = decision_vars[adj_idxs[0]]
                    total_hamming += hamming_distance_vec(cur_dec, adj_dec)
                    adj_pairs += 1
    adj_mean = float(total_hamming / adj_pairs) if adj_pairs > 0 else 0.0
    return {'grid_density_var': density_var, 'pareto_grid_ratio': pareto_ratio, 'adjacent_grid_hamming_mean': adj_mean}

def calculate_hypervolume(solutions, ref_point):
    if HyperVolume is None:
        return 0.0
    hv = HyperVolume(reference_point=ref_point)
    return hv.compute(solutions)

def calculate_level_count(points):
    if FastNonDominatedRanking is None:
        return 0
    ranking = FastNonDominatedRanking()
    sols = [type('S', (object,), {'objectives': p, 'attributes': {}}) for p in points]
    ranking.compute_ranking(sols)
    return ranking.get_number_of_subfronts()

def calculate_objective_variance(points, dict_search):
    if points is None or len(points) == 0:
        return 0.0
    vars_ = np.var(points, axis=0)
    return float(np.mean(vars_))

def calculate_objective_entropy(points, dict_search):
    if points is None or len(points) == 0:
        return 0.0
    ent = []
    for i in range(points.shape[1]):
        _, counts = np.unique(points[:, i], return_counts=True)
        ent.append(float(entropy(counts)))
    return float(np.mean(ent))

def print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              fig_num, x, y, random_seed, sampling_method, sample_size,
                              dataset_name, mode, unique_id, sample_type,
                              dominance_change_frequency, reverse, dag=None, num_valid_tasks=None):
    adjacent_grid_hamming_mean = 0.0
    average_hamming_distance = 0.0
    avg_discrete_descent_cone = 0.0
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
        avg_discrete_descent_cone = calculate_avg_discrete_descent_cone(decision_vars, objective_values, sampled_data, random_seed=random_seed)
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
        adjacent_grid_hamming_mean = float(grid_feats['adjacent_grid_hamming_mean'])
        try:
            hypervolume = float(calculate_hypervolume(np.vstack([x, y]).T, ref_point))
        except Exception:
            hypervolume = 0.0
        pareto_idx = find_global_pareto_points(decision_vars, objective_values)
        if len(pareto_idx) > 1:
            pareto_decisions = [tuple(decision_vars[i]) for i in pareto_idx]
            total = 0
            pairs = 0
            for i in range(len(pareto_decisions)):
                for j in range(i + 1, len(pareto_decisions)):
                    total += hamming_distance_vec(pareto_decisions[i], pareto_decisions[j])
                    pairs += 1
            average_hamming_distance = float(total / pairs) if pairs > 0 else 0.0
        else:
            average_hamming_distance = 0.0
    else:
        average_hamming_distance = 0.0
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Results/Output-draw')
    os.makedirs(output_dir, exist_ok=True)
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

def find_center_point(r0_points, random_seed=42, center_type='mean'):
    if center_type == 'random':
        np.random.seed(random_seed)
        return r0_points[np.random.choice(len(r0_points))]
    elif center_type == 'mean':
        return np.mean(r0_points, axis=0)
    elif center_type == 'left_bottom_pareto':
        pareto_points = [p for p in r0_points if is_pareto_optimal_minimize(p, r0_points)]
        pareto_points = np.array(pareto_points)
        if len(pareto_points) > 0:
            left_bottom_index = np.argmin(pareto_points[:, 0] + pareto_points[:, 1])
            return pareto_points[left_bottom_index]
        else:
            np.random.seed(random_seed)
            return r0_points[np.random.choice(len(r0_points))]
    else:
        raise ValueError("center_type must be 'random','mean' or 'left_bottom_pareto'.")

def plot_figure1(random_seed=42, center_type='random', sampling_method='', sample_size='',
                 dataset_name='', mode='', unique_id='', sample_type='', sampled_dict=None,
                 reverse=False, dag=None, num_valid_tasks=None):
    if sampled_dict is None or len(sampled_dict) == 0:
        print("plot_figure1: sampled_dict is empty")
        return
    sampled_data = np.array(list(sampled_dict.keys()))
    r0_points = np.array(list(sampled_dict.values()))
    center_point = find_center_point(r0_points, random_seed, center_type)
    all_points = np.vstack([r0_points, center_point])
    r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
    total_points_count = {'pareto': 0, 'non_pareto': 0}
    for pt in r0_points:
        is_p = is_pareto_optimal_minimize(pt, all_points)
        key = 'pareto' if is_p else 'non_pareto'
        total_points_count[key] += 1
        angle = np.arctan2(pt[1] - center_point[1], pt[0] - center_point[0]) % (2 * np.pi)
        for i in range(8):
            if i * np.pi / 4 <= angle < (i + 1) * np.pi / 4:
                r_pareto_count[f'r_{i}'][key] += 1
                break
    x = r0_points[:, 0]
    y = r0_points[:, 1]
    dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, random_seed=random_seed)
    print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              1, x, y, random_seed, sampling_method, sample_size, dataset_name,
                              mode, unique_id, sample_type, dominance_change_frequency, reverse, dag, num_valid_tasks)

def plot_figure2(random_seed=42, center_type='random', sampling_method='', sample_size='',
                 dataset_name='', mode='', unique_id='', sample_type='', sampled_dict=None,
                 reverse=False, dag=None, num_valid_tasks=None):
    if sampled_dict is None or len(sampled_dict) == 0:
        print("plot_figure2: sampled_dict is empty")
        return
    sampled_data = np.array(list(sampled_dict.keys()))
    objective_values = np.array(list(sampled_dict.values()))
    g1 = objective_values[:, 0]
    g2 = objective_values[:, 1]
    r0_points = np.column_stack((g1, g2))
    center_point = find_center_point(r0_points, random_seed, center_type)
    all_points = np.vstack([r0_points, center_point])
    r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
    total_points_count = {'pareto': 0, 'non_pareto': 0}
    for pt in r0_points:
        is_p = is_pareto_optimal_minimize(pt, all_points)
        key = 'pareto' if is_p else 'non_pareto'
        total_points_count[key] += 1
        angle = np.arctan2(pt[1] - center_point[1], pt[0] - center_point[0]) % (2 * np.pi)
        for i in range(8):
            if i * np.pi / 4 <= angle < (i + 1) * np.pi / 4:
                r_pareto_count[f'r_{i}'][key] += 1
                break
    x = r0_points[:, 0]
    y = r0_points[:, 1]
    dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, random_seed=random_seed)
    print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              2, x, y, random_seed, sampling_method, sample_size, dataset_name,
                              mode, unique_id, sample_type, dominance_change_frequency, reverse, dag, num_valid_tasks)